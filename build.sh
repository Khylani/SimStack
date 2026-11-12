#!/usr/bin/env bash
# Simstack unity build (Linux, clang or gcc).
#
#   ./build.sh [debug|release|ship] [--simd=0] [--asan] [--sim-only] [--cuda]
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
for arg in "$@"; do
    case "$arg" in
        debug|release|ship) MODE=$arg ;;
        --simd=0) SIMD=0 ;;
        --simd=1) SIMD=1 ;;
        --asan) ASAN=1 ;;
        --sim-only) SIM_ONLY=1 ;;
        --cuda) CUDA=1 ;;
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
        -Wno-unused-variable -Wno-missing-braces -Wno-format-truncation -Isrc -DSIM_SIMD=$SIMD $ARCH_FLAGS"

case "$MODE" in
    debug)   CFG="-O0 -g -DSIM_INTERNAL=1 -DSIM_SLOW=1" ;;
    release) CFG="-O2 -g -DSIM_INTERNAL=1 -DSIM_SLOW=0" ;;
    ship)    CFG="-O3 -DSIM_INTERNAL=0 -DSIM_SLOW=0 -DNDEBUG" ;;
esac
if [ "$ASAN" = 1 ]; then CFG="$CFG -fsanitize=address,undefined -fno-omit-frame-pointer"; fi

mkdir -p build
echo "[build] $MODE  cxx=$CXX  simd=$SIMD  asan=$ASAN"

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
    NVCC=${NVCC:-nvcc}
    $NVCC -arch=sm_86 -O3 -lineinfo --shared -Xcompiler -fPIC -Isrc src/gpu/kernels.cu -o build/simstack_gpu.so
    echo "[build] simstack_gpu.so"
fi

# --- platform exe -----------------------------------------------------------------
if [ "$SIM_ONLY" = 0 ]; then
    START=$(date +%s%N)
    $CXX $COMMON $CFG src/platform/linux_simstack.cpp -o build/simstack -lX11 -lGL -ldl -lm
    echo "[build] simstack         $(( ($(date +%s%N) - START) / 1000000 )) ms"
fi
