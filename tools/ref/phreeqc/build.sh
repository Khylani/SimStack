#!/bin/sh
# Builds the sim side of the PHREEQC comparison into build/ref_sim_water.
set -e
cd "$(dirname "$0")/../../.."
CXX=${CXX:-clang++}
$CXX -std=c++20 -O2 -fno-exceptions -fno-rtti -fno-strict-aliasing -nostdlib++ -mavx2 -mfma -mbmi2 -mf16c -ffp-contract=off \
     -Wno-format-truncation -Isrc -Ibuild/gen -DSIM_SIMD=1 -DSIM_INTERNAL=1 -DSIM_SLOW=0 -fno-threadsafe-statics \
     tools/ref/phreeqc/sim_water.cpp -o build/ref_sim_water -lm -ldl -lpthread
