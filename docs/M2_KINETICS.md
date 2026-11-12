# M2 (part 1): time-dependent DBD chemistry solver

The first milestone-2 piece from the architecture plan is the dynamic (time-dependent)
plasma-chemistry solver that the plan calls `dbd_fast`. It is a batched stiff ODE solver for
mass-action chemistry. Every reactor zone is a lane: four per AVX2 register on the CPU, one
per thread on CUDA. The same source code runs on both, and the results are bit-identical.

Not in this part (still M2): the ChemPlasKin/Cantera reference solver (`dbd_ref`), the
reduction tables generated from it, and the LoKI-B++ cross-checks. Until those exist, the
discharge is represented by G-values (radicals per 100 eV), which are tunable surrogates.

## What runs

```sh
./build.sh release --tests --cuda
./build/simstack_tests --gpu            # 43 checks: solver, mechanism, SIMD/CUDA bit-identity
./build/simstack_tests --gpu --bench    # kernel cycles, tolerance study, CPU/GPU throughput

./build/simstack --scene dbd --headless 300 --csv runs/reactor.csv          # 16-zone plug-flow reactor, 5 s
./build/simstack --scene dbd --averaged --headless 300                      # same, continuous source (~200x faster)
./build/simstack --scene sweep --zones 1024 --averaged --headless 121 --csv runs/sweep.csv   # power x humidity
./build/simstack --scene sweep --zones 1024 --averaged --headless 121 --gpu                  # same on CUDA
./build/simstack --scene dbd            # windowed: species profiles along the reactor (log ppm)
```

Reactor defaults: humid air (x_H2O = 0.02) at 310 K and 1 atm, 0.1 W/cm³ at 10 kHz (one
micro-discharge burst per half-cycle), 2 s residence over 16 zones, which is 200 J/L. Outlet
at steady state (t ≥ 3 s):

| O3 | NO | NO2 | N2O5 | HNO2 | HNO3 | H2O2 | N2O |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2268 ppm | 0.08 | 1.9 | 3.4 | 0.11 | 35 | 15 | 154 |

That is about 77 g O3/kWh, inside the 50–100 g/kWh range of air-fed ozonizers. The sweep
reproduces the known trends: humidity roughly halves O3 at 1000 J/L and drives HNO3/H2O2 up,
dry air makes no HNO3, and ozone is sublinear in energy.

Pulsed and averaged discharge agree to five significant figures at 10 kHz. The per-pulse
radical doses (~1e13 cm⁻³) are small enough that the chemistry is near-linear in the source.
Averaged mode is therefore the default candidate for interactive use; pulsed mode matters at
low frequency or high energy per pulse, where radical–radical recombination becomes nonlinear.

## Layout

| File | Role |
| --- | --- |
| `src/base/base_lane.h` | Lane vocabulary (`lane_fmadd`, `lane_select`, masks, …) for `V = f64`; nvcc-safe |
| `src/base/base_simd.h` | `f64x4` / `b64x4` (AVX2 or scalar) implementing the same vocabulary |
| `src/sim/kinetics/kin_mech.h` | `KinMech`: flat reaction tables + symbolic sparse LU (POD, copied to CUDA constant memory) |
| `src/sim/kinetics/kin_solver.h` | Rosenbrock integrator templated on lane type: `kin_fun`, `kin_neg_jac`, `kin_factor`, `kin_solve`, `kin_integrate`, `kin_advance` (pulse events) |
| `src/sim/kinetics/kin_build.cpp` | Equation parser (`"O + O2 + M -> O3 + M"`), element-balance check, symbolic LU, rate laws |
| `src/sim/kinetics/air_mech.cpp` | 17-species / 40-reaction humid-air afterglow mechanism (JPL/IUPAC), G-value source, feed gas |
| `src/sim/dbd_lanes.h` | `dbd_advance_group<V, Cap>`: load lanes from SoA, advance, store (shared CPU/CUDA) |
| `src/sim/dbd.cpp` | `DbdZones` SoA block, rate/source prepare, work-queue CPU advance, GPU advance, plug flow |
| `src/sim/dbd_scene.cpp` | Reactor and sweep scenes, CSV, render batches, summary, state hash |
| `src/platform/linux_work_queue.cpp` | Handmade-style work queue, workers pinned one per P-core |
| `src/platform/linux_gpu.cpp` | dlopen of `simstack_gpu.so` into a `GpuAPI` table |
| `src/gpu/kernels.cu` | `kin_advance_kernel`: one thread per zone, `-fmad=false` |
| `src/tests/test_kinetics.cpp` | `build/simstack_tests` |

## Solver design

- **Methods.** KPP's Rosenbrock family (Sandu et al. 1997): Ros2, Ros3, and Rodas3, which is
  the default (stiffly accurate). Each lane has its own adaptive step-size control (weighted
  RMS norm, KPP's factor limits). Lanes that reach their end time or fail are masked out
  while the others continue.
- **Linear algebra.** No-pivot sparse LU in IKJ order over a compact non-zero array. The fill
  pattern and the flat op lists come from a symbolic pass at mechanism build: 213 non-zeros
  of 289, 104 eliminations, 707 updates. Pivots are stored as reciprocals, so there is one
  division per row per step.
- **Events.** A discharge pulse is an instantaneous injection at `(n+1)·period`. Integration
  stops exactly on it, applies it, and restarts at `h_start = 1 ns`. Averaged mode puts the
  same energy into a constant source term.
- **No libm in lane code.** The step-size factor's cube root uses `x^(21/64)` from six
  `sqrt`s plus two Newton steps. `exp`/`pow` appear only in host-side rate constants, which
  are shared by every path.
- **Determinism.** Every path performs the same IEEE operations in the same order.
  `-ffp-contract=off` on clang/gcc and `-fmad=false` on nvcc mean that fused ops happen only
  where the source says `lane_fmadd`/`lane_fnmadd`.
- **Memory.** All per-zone arrays are one contiguous f64 block in the permanent arena, padded
  to 8 lanes. The GPU mirror is one allocation at the same offsets, and a download copies
  only the solver-written prefix.
- **Mechanism as data.** Reactions are equations in `air_mech.cpp`. A typo that breaks N/O/H
  balance fails at build time; the tests prove this.

## Verification (all in `simstack_tests`)

| Check | Result |
| --- | --- |
| Analytic Jacobian vs central differences | worst 1.3e-12 of row scale |
| Sparse LU solve vs dense partial-pivot GE (h from 1e-12 to 1 s) | worst 8e-15 relative |
| Robertson problem, t = 40, rtol 1e-6 | Ros2 9e-7, Ros3 1e-8, Rodas3 1e-6 relative error; mass conserved to 1e-12 |
| Fixed-step order on A + B → C (closed form) | 1.86 / 2.99 / 2.94 (expected 2 / 3 / 3) |
| f64x4 vs f64, pulsed and averaged | bit-identical |
| CUDA vs AVX2, pulsed and averaged | bit-identical |
| N/O/H atom conservation, 8 zones × 50 ms pulsed | drift 6e-13 |
| Dry-air O3 yield at 0.1 mJ/cm³ | 0.97 of the odd oxygen made (O + N atoms) |
| Reactor state hash, 30 frames | `1d27d1d629fbc024` for clang/gcc, -O0/-O2, ASan, `--simd=0`, 1/8 threads |
| Hot reload ×2 during a 5 s pulsed run | same final hash as without reloads (`e77be511ffd6c8e6`) |

## Performance (i9-12900KF, RTX 3060 12 GB)

Tolerance study (`--bench`): 8 humid-air zones, 20 ms, compared with a reference at
rtol 1e-9:

| Mode | rtol / atol | Steps per zone | Worst product error |
| --- | --- | --- | --- |
| pulsed | 1e-4 / 1e2 | 23 317 | 1.1e-6 |
| pulsed | **1e-3 / 1e4 (default)** | 10 549 | 1.4e-5 |
| pulsed | 1e-2 / 1e4 | 5 195 | 1.8e-4 |
| averaged | 1e-3 / 1e4 | 78 | 3.7e-5 |

Throughput:

| Case | Rate |
| --- | --- |
| AVX2 1 thread, pulsed | 3.3 zone-ms per wall-ms (~0.7 µs per zone-step) |
| AVX2 1 thread, averaged | 436 zone-ms per wall-ms |
| Reactor 16 zones pulsed, 8 threads | 0.66 sim-s per wall-s (16 zones = 4 lane groups, so ≤ 4 threads busy) |
| Reactor 16 zones averaged | 130–170 sim-s per wall-s |
| Sweep 1024 zones averaged | 5.6 sim-s/s on 8 CPU threads vs 1.35 on CUDA |

Kernel cost per f64x4 group, in TSC cycles: fun 382, Jacobian 1150, factor 1718, solve 453.
The factor's ~2.4 cycles per sparse update is close to the load/store limit of a table-driven
formulation.

### What we learned about the GPU path

It works and is bit-identical, but on the RTX 3060 it is the slower engine for this kernel.
It overtakes one CPU core at about 1 000 zones and reaches about 2× one core at 4 096+ zones;
eight P-cores win at every size.

| Zones (pulsed, 2 ms) | CUDA | AVX2 1 thread |
| --- | --- | --- |
| 64 | 560 ms | 39 ms |
| 1 024 | 707 ms | 622 ms |
| 4 096 | 1 190 ms | 2 490 ms |
| 16 384 | 2 670 ms | — |

Reasons, measured rather than guessed:

1. **FP64 on GA106 runs at 1/64 of FP32**, and f64 division and sqrt are software sequences.
2. **Per-lane working set.** The Jacobian/LU (213 entries) plus stage vectors is about 4 KB
   per thread in local memory. Cutting it from 11 KB with `KinCapGpu` did not move the
   per-step latency; a single warp per SM still takes about 0.6 ms per Rosenbrock step.
3. **Constant cache is not the bottleneck.** Staging the 30 KB of tables in shared memory
   made it slower, because occupancy dropped.

Making the GPU worthwhile needs a different decomposition: one warp per zone, with the sparse
LU and stage vectors in shared memory and the parallelism inside the system. On FP64-limited
consumer GPUs, the alternative is f32 with a careful scaling of the state. Both are
follow-ups. For the interactive path the plan already has the GPU doing fog droplets (M5),
which is f32 and suits it.

## Known limits of the chemistry

- Neutral afterglow only: no ions, no O(¹D) / N₂(A) / N₂(v) kinetics, no surface chemistry.
  Their net effect is folded into the G-values. N2O comes mostly from N + NO2 here; real DBDs
  also make it via N₂(A) + O₂.
- Fixed gas temperature per zone, so no thermal O3 destruction and no NOx-mode
  ("discharge poisoning") switch at high specific energy.
- Third-body efficiencies are those of air for every reaction. Fall-off uses the JPL form
  with Fc = 0.6.
- Plug flow is a chain of stirred zones (16 CSTRs), so some axial dispersion is built in.

## Next steps

1. Parameter registry (`dbd.power_W_cm3`, `dbd.freq_Hz`, rtol, method, …) with hot-reload
   survival, as in the plan's §5.2. The zones already take per-zone operating points.
2. Gas → liquid transfer from the reactor outlet: the first NFT coupling (M3).
3. `dbd_ref` (ChemPlasKin) and `tools/reduce_dbd` tables. The G-values become table lookups,
   and `dbd_fast` is validated against `dbd_ref`.
4. CPU speed: generate straight-line code for a fixed mechanism (KPP-style) to get the
   Jacobian/LU out of table-driven loops, and/or split groups finer for the 16-zone reactor.
5. Warp-per-zone CUDA kernel if large sweeps need the GPU.
