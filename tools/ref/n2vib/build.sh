#!/bin/bash
# Builds the state-to-state N2(v) reference against SimStack's kinetics (bigger tables, table-walk ops).
set -e
cd "$(dirname "$0")/../../.."
ROOT=$(pwd)
mkdir -p build
taskset -c 0-3,16-23 nice -n 19 clang++ -std=c++20 -O2 -fno-exceptions -fno-rtti -fno-strict-aliasing -nostdlib++ \
    -mavx2 -mfma -mbmi2 -mf16c -ffp-contract=off -DSIM_SIMD=1 -DSIM_INTERNAL=1 -DSIM_SLOW=0 \
    -Isrc -Ibuild/gen -Itools/ref/n2vib -DN2VIB_DIR="\"$ROOT/tools/ref/n2vib\"" \
    tools/ref/n2vib/n2vib_ref.cpp -o build/n2vib_ref -lm
echo "built build/n2vib_ref"
