# Reference solvers (offline)

SimStack's air chemistry is a reduced model: a 17-species neutral mechanism
with a quasi-steady electron-impact and metastable source. Two external codes
check it. Both run offline, outside the repository; SimStack only consumes
what they produce: the G-value table and the numbers recorded in
`test_reference_source`.

| Tool | What it gives | License |
| --- | --- | --- |
| [LoKI-B++](https://github.com/LoKI-Suite/LoKI-B-cpp) | two-term Boltzmann solution for dry air → energy partition → `src/sim/kinetics/air_source_table.h` | GPL-3.0 |
| [ChemPlasKin](https://github.com/ShaoX96/ChemPlasKin) (Cantera 3.0 + CppBOLOS) | full electron + ion + excited-state kinetics of repeated micro-discharges | LGPL-2.1 |

Everything below lives in `~/Play/ref`. The build tools (scons, cmake,
ruamel.yaml) are in the shared `~/ComfyUI/.venv`. Builds are pinned with
`taskset -c 0-3,16-23 nice -n 19`.

## LoKI-B++ → G-values

```sh
git clone https://github.com/LoKI-Suite/LoKI-B-cpp && cd LoKI-B-cpp
# gcc 16: the project promotes -Wzero-as-null-pointer-constant to an error and bundled Eigen 3.4 trips it
sed -i 's/^\(\s*\)loki_cxxflags(-Werror=zero-as-null-pointer-constant)/\1# &/' CMakeLists.txt
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CXX_COMPILER=g++ \
      -DLOKIB_USE_BUILTIN_NLOHMANN_JSON=ON -DLOKIB_USE_BUILTIN_EIGEN=ON -DCMAKE_CXX_FLAGS=-Wno-error -B build
cmake --build build -j 12
cp <SimStack>/tools/ref/lokib_air_dbd.in input/simstack_air_dbd.in
cd input && ../build/app/loki simstack_air_dbd.in          # ~10 s, 36 fields 50-400 Td
python3 <SimStack>/tools/ref/loki_gvalues.py output/simstack_air_dbd \
        Nitrogen/N2_LXCat.txt Oxygen/O2_LXCat.txt <SimStack>/src/sim/kinetics/air_source_table.h
```

`lokib_air_dbd.in` is the shipped `lokib_setup_air.in` with these changes:
- 1 atm, 330 K, dry air 79/21, 50–400 Td.
- No electron–electron collisions, which are negligible at filament ionization degrees.
- A 60 eV grid.
- Statistical weights for the ground and reversible O₂ states. The shipped file does not run
  without them.

The channel mapping is in the docstring of `loki_gvalues.py`. At 180 Td, per 100 eV:
- **O₂ dissociations:** 3.88, 53 % of them through O(¹D).
- **N₂(A):** 4.34, counting all triplets.
- **N₂ → N + N(²D):** 0.48.
- **N₂ vibration:** 20 % of the power.

## ChemPlasKin

```sh
git clone --branch 3.0 https://github.com/Cantera/cantera && cd cantera && git submodule update --init --recursive
git clone https://github.com/ShaoX96/ChemPlasKin ../ChemPlasKin
(cd ext && git clone https://github.com/ShaoX96/CppBOLOS bolos && git clone https://github.com/beltoforion/muparser)
cp ../ChemPlasKin/ext/SConscript ext/
cp ../ChemPlasKin/include/kinetics/*.h include/cantera/kinetics/ && cp ../ChemPlasKin/include/base/Solution.h include/cantera/base/
cp ../ChemPlasKin/src/kinetics/*.cpp src/kinetics/ && cp ../ChemPlasKin/src/base/Solution.cpp src/base/
# Boost 1.84 headers unpacked to ~/Play/ref/deps; no system packages
cat > cantera.conf <<'EOF'
python_package = 'none'
f90_interface = 'n'
boost_inc_dir = '/home/aurora/Play/ref/deps/boost_1_84_0'
system_fmt = 'n'
system_yamlcpp = 'y'      # the bundled yaml-cpp does not compile with gcc 16
system_sundials = 'n'
system_eigen = 'n'
system_blas_lapack = 'n'
hdf_support = 'n'
CXX = 'g++'
CC = 'gcc'
cxx_flags = '-std=c++17 -Wno-error'
googletest = 'none'
EOF
scons build -j 12
```

Local patches to `ChemPlasKin/src` (none change the physics):
- `utilities.h`: add `#include <optional>`, which gcc 16 no longer pulls in transitively.
- `master.cpp`:
  - `integrator->setMaxSteps(2000000)`. Long afterglows between kHz pulses exceed the
    default of 20000.
  - At each pulse start, re-seed electrons to `electronDens` if they have decayed below it.
    After long afterglows n_e undershoots, and the next avalanche would start from a negative
    seed and return NaN.
  - Print the total deposited energy over all pulses to stderr. The logged power is
    diagnostic only and keeps being evaluated after the pulse ends.

Case: `chemplaskin_case/`, with `air_plasma_n2o5.yaml` made by `add_n2o5.py`.
- Dry air, 180 Td, gas temperature held fixed.
- `pulseEnergy` = filament energy E_f, `nPulses` = 2.16 J/cm³ / E_f, over 1.32 s (the skid
  module).
- Pulses get up to 200 ns, at 1e-10 s steps, so each one reaches its cap.
- The runs deposited 2.11–2.15 J/cm³, checked with the stderr total.

## Results (dry air, 180 Td, 2.16 J/cm³ over 1.32 s)

| Case | ChemPlasKin | SimStack |
| --- | --- | --- |
| One pulse, fresh gas: odd O / fixed N per 100 eV | 16.0 / 0.84 | 14.4 / ~0.5 |
| 330 K, E_f 10: fixed-N cost, N₂O, O₃ | 39.9 MJ/mol, 356, 30,130 ppm | 31.2, 394, 41,466 |
| 330 K, E_f 30 | 44.1, 210, 32,025 | 42.9, 289, 43,025 |
| 330 K, E_f 100 | 37.2, 97, 30,717 | 53.2, 209, 41,670 |
| 413 K (skid gap), E_f 30 | O₃ 0, NO 522 + NO₂ 467 ppm, 73 MJ/mol | O₃ 3470, 43 MJ/mol (collapses as the reference does at E_f ≤ 10: 74 MJ/mol) |

**What the comparison established:**
- **Mechanism corrections:**
  - N + NO₂ needs Kossyi's four-way branching. JPL's single N₂O + O channel tripled N₂O and
    cut fixed N.
  - N₂(A) + N₂O gives N₂ + N + NO.
- **Agreement:** with those, fixed N agrees within ±45 % (3 % at E_f 30) and N₂O within
  1.1–2.2×.
- **Remaining ozone bias:** about +35 % in our model. On a single fresh pulse we are 10 %
  low on odd oxygen, so the excess is ozone loss at high O₃ that the reduced model lumps
  away: singlet oxygen chains and the ions.
- **Absolute yield:** both give 90–125 g O₃/kWh per electron joule at this SEI, inside
  air-fed ozonizer practice. The Boltzmann source needs no extra efficiency factor.
- **The original mechanism needed fixing too:** without N₂O₅ it falls into NOx mode at
  330 K. `add_n2o5.py` fixes that.
- **Not in either code:** N₂(v) chemistry (N₂(v) + O → NO + N). ChemPlasKin carries N₂
  vibrational energy only as a heat reservoir.
