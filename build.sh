#!/usr/bin/env bash
# Simstack unity build (Linux, clang or gcc).
#
#   ./build.sh [debug|release|ship] [--simd=0] [--asan] [--sim-only] [--cuda] [--tests]
#
# Produces build/simstack (platform exe) and build/simstack_sim.so (hot-reloadable
# sim module). Two compiler invocations. build/lock.tmp exists while the sim
# module is being written so the running exe never loads a half-linked file.
set -euo pipefail
cd "$(dirname "$0")"

MODE=debug
SIMD=1
ASAN=0
SIM_ONLY=0
CUDA=0
TESTS=0
for arg in "$@"; do
    case "$arg" in
        debug|release|ship) MODE=$arg ;;
        --simd=0) SIMD=0 ;;
        --simd=1) SIMD=1 ;;
        --asan) ASAN=1 ;;
        --sim-only) SIM_ONLY=1 ;;
        --cuda) CUDA=1 ;;
        --tests) TESTS=1 ;;
        *) echo "unknown argument: $arg"; exit 1 ;;
    esac
done

CXX=${CXX:-clang++}
if ! command -v "$CXX" >/dev/null 2>&1; then CXX=g++; fi

# i9-12900KF: AVX2 + FMA + BMI2, no AVX-512. -ffp-contract=off keeps the scalar
# f32x8 fallback bit-identical to the AVX2 path (explicit fmadd only).
ARCH_FLAGS="-mavx2 -mfma -mbmi2 -mf16c -ffp-contract=off"
# No C++ runtime is linked: no exceptions, RTTI, new/delete or STL anywhere.
COMMON="-std=c++20 -fno-exceptions -fno-rtti -fno-strict-aliasing -fvisibility=hidden -nostdlib++ \
        -Wall -Wextra -Wno-unused-function -Wno-unused-parameter -Wno-missing-field-initializers \
        -Wno-unused-variable -Wno-missing-braces -Wno-format-truncation -Isrc -Ibuild/gen -DSIM_SIMD=$SIMD $ARCH_FLAGS"

case "$MODE" in
    debug)   CFG="-O0 -g -DSIM_INTERNAL=1 -DSIM_SLOW=1" ;;
    release) CFG="-O2 -g -DSIM_INTERNAL=1 -DSIM_SLOW=0" ;;
    ship)    CFG="-O3 -DSIM_INTERNAL=0 -DSIM_SLOW=0 -DNDEBUG" ;;
esac
if [ "$ASAN" = 1 ]; then CFG="$CFG -fsanitize=address,undefined -fno-omit-frame-pointer"; fi

mkdir -p build/gen
echo "[build] $MODE  cxx=$CXX  simd=$SIMD  asan=$ASAN"

# --- metaprogramming: straight-line kinetics kernels from the mechanism ---------
# kin_codegen is rebuilt only when its inputs change; it rewrites
# build/gen/air_mech_gen.h only when the output differs.
CODEGEN_SRC="src/tools/kin_codegen.cpp src/sim/kinetics/kin_build.cpp src/sim/kinetics/air_mech.cpp src/sim/kinetics/kin_mech.h src/sim/kinetics/kin_solver.h src/base/base_types.h src/base/base_lane.h"
if [ ! -x build/kin_codegen ] || [ -n "$(find $CODEGEN_SRC -newer build/kin_codegen 2>/dev/null)" ]; then
    $CXX -std=c++20 -O1 -fno-exceptions -fno-rtti -Isrc -DSIM_SLOW=1 src/tools/kin_codegen.cpp -o build/kin_codegen -lm
fi
build/kin_codegen build/gen/air_mech_gen.h

# --- sim module -----------------------------------------------------------------
touch build/lock.tmp
trap 'rm -f build/lock.tmp' EXIT
START=$(date +%s%N)
$CXX $COMMON $CFG -shared -fPIC src/sim/simstack.cpp -o build/simstack_sim.so.tmp -lm
mv -f build/simstack_sim.so.tmp build/simstack_sim.so
rm -f build/lock.tmp
echo "[build] simstack_sim.so  $(( ($(date +%s%N) - START) / 1000000 )) ms"

# --- gpu module (optional; needs the CUDA toolkit) -------------------------------
if [ "$CUDA" = 1 ]; then
    # nvcc is often not on PATH (e.g. /opt/cuda/bin): honour NVCC, then CUDA_HOME.
    if [ -z "${NVCC:-}" ]; then
        if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then NVCC="$CUDA_HOME/bin/nvcc"
        elif command -v nvcc >/dev/null 2>&1; then NVCC=nvcc
        elif [ -x /opt/cuda/bin/nvcc ]; then NVCC=/opt/cuda/bin/nvcc
        elif [ -x /usr/local/cuda/bin/nvcc ]; then NVCC=/usr/local/cuda/bin/nvcc
        else echo "[build] nvcc not found: set NVCC or CUDA_HOME"; exit 1
        fi
    fi
    echo "[build] nvcc=$NVCC"
    # -fmad=false: no implicit FMA contraction, so the kinetics kernel is
    # bit-identical to the CPU path (explicit fma() calls are still fused).
    START=$(date +%s%N)
    $NVCC -arch=sm_86 -O3 -fmad=false -lineinfo -Xptxas -warn-spills --shared -Xcompiler -fPIC -Isrc -Ibuild/gen \
          src/gpu/kernels.cu -o build/simstack_gpu.so.tmp
    mv -f build/simstack_gpu.so.tmp build/simstack_gpu.so
    echo "[build] simstack_gpu.so  $(( ($(date +%s%N) - START) / 1000000 )) ms"
fi

# --- tests (kinetics solver; no window, no sim module) --------------------------
if [ "$TESTS" = 1 ]; then
    START=$(date +%s%N)
    $CXX $COMMON $CFG src/tests/test_kinetics.cpp -o build/simstack_tests -lm -ldl -lpthread
    echo "[build] simstack_tests   $(( ($(date +%s%N) - START) / 1000000 )) ms"
fi

# --- platform exe -----------------------------------------------------------------
if [ "$SIM_ONLY" = 0 ]; then
    START=$(date +%s%N)
    $CXX $COMMON $CFG src/platform/linux_simstack.cpp -o build/simstack -lX11 -lGL -ldl -lm -lpthread
    echo "[build] simstack         $(( ($(date +%s%N) - START) / 1000000 )) ms"
fi
