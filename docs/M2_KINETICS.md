# M2 (part 1): time-dependent DBD chemistry solver and real-time UI

The first milestone-2 piece from the architecture plan is the dynamic (time-dependent)
plasma-chemistry solver that the plan calls `dbd_fast`, together with a real-time UI to
drive it. It is a batched stiff ODE solver for mass-action chemistry in which every reactor
zone is a lane. The same source code runs on AVX2 (4 f64 or 8 f32 lanes per register) and
on CUDA (one zone per thread), and the results are bit-identical across all of them.

Not in this part (still M2): the ChemPlasKin/Cantera reference solver (`dbd_ref`), the
reduction tables generated from it, and the LoKI-B++ cross-checks. Until those exist, the
discharge is represented by G-values per 100 eV, which are tunable surrogates:
- 1.5 O₂ dissociations, half of them through the O + O(¹D) channel
- 1.5 N₂(A), including cascade from the higher triplets
- 0.4 N₂ dissociations to N + N(²D)
- 0.3 H₂O dissociations at 2 % water

The metastables' fates are quasi-steady and use literature rate constants (Herron 1999,
Kossyi 1992, JPL 19-5). N(²D) + O₂ → NO + O is the main NO source at room temperature.

## What runs

```sh
./build.sh release --tests --cuda
./build/simstack_tests --gpu            # 96 checks: solver, mechanism, codegen, f32, SIMD/CUDA bit-identity
./build/simstack_tests --gpu --bench    # kernel cycles, tolerance study, CPU/GPU throughput

./build/simstack --scene dbd            # real-time UI: 64-zone plug-flow reactor
./build/simstack --scene sweep --gpu    # real-time UI: 32 x 32 power x humidity sweep on CUDA
./build/simstack --scene dbd --headless 300 --csv runs/reactor.csv        # 5 s, outlet time series
./build/simstack --scene sweep --zones 4096 --averaged --headless 125 --csv runs/sweep.csv
```

Reactor defaults: humid air (x_H2O = 0.02) at 310 K and 1 atm, 0.1 W/cm³ at 10 kHz (one
micro-discharge burst per half-cycle), 2 s residence over 64 zones, which is 200 J/L. Outlet
at steady state (t ≥ 3 s):

| O3 | NO | NO2 | N2O5 | HNO2 | HNO3 | H2O2 | N2O |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2135 ppm | 0.16 | 9.0 | 29.3 | 0.25 | 121.8 | 8.2 | 77 |

That is about 72 g O3/kWh, inside the 50–100 g/kWh range of air-fed ozonizers. Two changes
raised the nitrogen oxides: the excited-state channels, and Kossyi's branching of N + NO₂,
which moves N₂O into fixed N. The ground-state-only mechanism gave 2286 ppm O3, 3.3 N2O5,
35.6 HNO3 and 152 N2O. The sweep
reproduces the known trends: humidity roughly halves O3 at 1000 J/L and drives HNO3/H2O2 up,
dry air makes no HNO3, and ozone is sublinear in energy. The outlet converges with zone
count toward the plug-flow limit: O3 is 2268 / 2280 / 2286 / 2289 ppm at 16 / 32 / 64 /
128 zones.

## Real-time UI

Dear ImGui + ImPlot (see `THIRD_PARTY.md`) run inside the sim module; the platform only
renders the triangles and textures the sim emits as render commands.

- **Layout.** Docked: Controls on the left, the 3D view on top right, Plots and Performance
  as tabs below. Everything can be re-docked by dragging, and the layout survives hot
  reloads.
- **Controls.**
  - Scene, run/pause/step/reset, speed (simulated seconds per wall second, or Max).
  - Zone count; backend (CPU or CUDA), lane precision (f32 or f64), method, rtol/atol.
  - Discharge mode; live operating point: power, humidity, frequency, temperature,
    pressure, residence, and the chemistry coupling step.
  - G-values, an outlet table, and CSV export.
- **Plots.** Outlet (or sweep corner) history on a log axis, the axial profile along the
  reactor, and the sweep heatmap with a colour scale.
- **3D view.** The reactor is drawn as a tube of rings coloured by the chosen species' log
  mole fraction; the sweep as a coloured surface; plus a colour bar and inlet/outlet
  labels.
- **Performance.** Frame / chemistry / speed history, the chemistry budget and achieved
  frame rate, arena usage bars (permanent, frame, UI heap, GPU), and the per-block
  profiler.

Frame pacing: the windowed chemistry runs as one **background job per frame** (background
queue, one thread on CPU 1) that overlaps the platform's render and buffer swap. It is
joined at the top of the next update, before the UI reads any state. Its wall budget is 90%
of the target frame time minus the main thread's UI work. The target follows the display
rate automatically: this machine's monitor is 3840×2160 at 30 Hz, so the solver gets about
28 ms per frame without dropping frames. Headless runs stay synchronous and deterministic.

Memory: no OS allocation after start-up. The platform reserves a 256 MB UI block. Font
files live in its persistent part; ImGui and ImPlot allocate from a power-of-two size-class
free-list heap in the rest, so usage is bounded by the peak. On a hot reload,
`sim_before_unload` saves the layout while the old code is still mapped, and then the heap
is reset wholesale. The GPU device arena is reset with the scene.

## Layout

| File | Role |
| --- | --- |
| `src/base/base_lane.h` | Lane vocabulary (`lane_fmadd`, `lane_select`, masks, f64-storage I/O) for `V = f64` and `f32`; nvcc-safe |
| `src/base/base_simd.h` | `f64x4`/`b64x4` and `f32x8`/`b32x8` (AVX2 or scalar) implementing the same vocabulary |
| `src/sim/kinetics/kin_mech.h` | `KinMech`: flat reaction tables + symbolic sparse LU (POD, copied to CUDA constant memory) |
| `src/sim/kinetics/kin_solver.h` | Rosenbrock integrator templated on lane type and kernel policy (`Ops`), delta formulation, local-time pulse driver |
| `src/sim/kinetics/kin_build.cpp` | Equation parser (`"O + O2 + M -> O3 + M"`), element-balance check, symbolic LU, rate laws |
| `src/sim/kinetics/air_mech.cpp` | 17-species / 44-reaction humid-air afterglow mechanism (JPL/IUPAC, Kossyi branching for N + NO₂), electron-impact source with quasi-steady N(²D) / N₂(A) / O(¹D) chemistry, Boltzmann G-value table and filament pulses, feed gas |
| `src/sim/kinetics/air_source_table.h` | Generated by `tools/ref/loki_gvalues.py` from LoKI-B++: G-values vs E/N |
| `src/tools/kin_codegen.cpp` | Metaprogram: emits straight-line kernels for the mechanism into `build/gen/air_mech_gen.h` |
| `src/sim/dbd_lanes.h` | `dbd_advance_group<V, Ops, Delta>`: loads lanes from the SoA block, advances, stores (shared CPU/CUDA) |
| `src/sim/dbd.cpp` | `DbdZones` SoA block, rate/source/pulse scheduling, threaded CPU advance, GPU advance, plug flow |
| `src/sim/dbd_scene.cpp` | Reactor and sweep scenes, operating point, history ring, CSV, 3D view, hash |
| `src/ui/ui_simstack.cpp`, `src/ui/ui_panels.cpp` | UI heap, context lifetime, input, render-command emission; the panels |
| `src/platform/linux_work_queue.cpp` | Handmade-style work queues: P-core pool (CPUs 2..14) and a background queue (CPU 1) |
| `src/platform/linux_gpu.cpp` | dlopen of `simstack_gpu.so` into a `GpuAPI` table |
| `src/platform/render_gl.cpp` | Coloured point batches in the scene viewport, UI pass (textures, scissor, 32-bit indices) |
| `src/gpu/kernels.cu` | `kin_advance_kernel<V, Ops, Delta>`, `-fmad=false` |
| `src/tests/test_kinetics.cpp` | `build/simstack_tests` |

## Solver design

- **Methods.** KPP's Rosenbrock family (Sandu et al. 1997): Ros2, Ros3, and Rodas3, which is
  the default (stiffly accurate). Each lane has its own adaptive step-size control (weighted
  RMS norm, KPP's factor limits). Lanes that reach their end time or fail are masked out
  while the others continue.
- **Linear algebra.** No-pivot sparse LU in IKJ order over a compact non-zero array (213
  non-zeros of 289, 104 eliminations, 707 updates), with reciprocal pivots.
- **Generated kernels.** `kin_codegen` builds the mechanism with the sim's own code and
  writes `fun`, `-J`, LU and solve as straight-line code with constant indices. They perform
  the same operations in the same order as the table walk, so results are bit-identical
  (tested), but the index loads and loop control disappear. A signature check falls back to
  the table walk if the mechanism and the generated header disagree.
- **Local time and events.** The solver integrates [0, dt] in local time. The host keeps the
  absolute pulse clock in f64 and passes each zone's first pulse and pulse count for the
  step, so rounding can never drop or duplicate a pulse. A pulse is an instantaneous
  injection; integration stops on it, applies it, and restarts at 1 ns.
- **f32 lanes (delta formulation).** f32 lanes integrate the *increment* over the step-start
  f64 state and evaluate rates at base + increment. Changes to N2/O2 (~1e19 cm⁻³) that are
  far below their f32 ulp are exact relative to the increment, and the f64 storage never
  round-trips through f32.
- **No libm in lane code.** The step-size factor's cube root uses `x^(21/64)` from six
  `sqrt`s plus two Newton steps.
- **Determinism.** `-ffp-contract=off` (clang/gcc) and `-fmad=false` (nvcc) mean that fused
  ops happen only where the source says `lane_fmadd`/`lane_fnmadd`. Lane groups are
  independent, so results don't depend on the thread count.
- **Memory.** All per-zone arrays are one contiguous f64 block in the permanent arena. The
  GPU mirror is one device allocation at the same offsets ("host owns the layout, device
  gets offsets").
- **Coupling interval.** One chemistry advance can span several 1 ms master steps
  (`--chem-dt`, UI slider). Independent sweep zones use 20 ms; the reactor uses 1 ms
  because its plug-flow exchange is operator-split.

## Verification (all in `simstack_tests`, plus the build matrix)

| Check | Result |
| --- | --- |
| Analytic Jacobian vs central differences | worst 1.3e-12 of row scale |
| Sparse LU solve vs dense partial-pivot GE (h from 1e-12 to 1 s) | worst 8e-15 relative |
| Robertson problem, t = 40, rtol 1e-6 | Ros2 9e-7, Ros3 1e-8, Rodas3 1e-6 relative error |
| Fixed-step order on A + B → C (closed form) | 1.86 / 2.99 / 2.94 (expected 2 / 3 / 3) |
| f64x4 == f64, f32x8 == f32, generated == table, CUDA == AVX2 (f64 and f32) | bit-identical |
| N/O/H atom conservation | f64: 2e-13 over 50 ms; f32: ~4e-8 per simulated second |
| f32 vs f64 products | 1e-6 (pulsed, 100 ms), 2e-7 (averaged, 1 s) |
| Dry-air O3 yield at 0.1 mJ/cm³ | 0.97 of the odd oxygen made |
| Hot reload ×2 mid-run (headless and windowed with the UI) | state preserved, same hash as without reloads |
| gcc ASan + UBSan | clean: tests, headless scenes, windowed reactor and sweep |

Reference hashes for `--scene dbd --headless 30` hold for any `--threads`, for CPU and
`--gpu`, and for clang and gcc:

| Lanes | Hash |
| --- | --- |
| f32 (default) | `32256818f6f35bc9` |
| `--precision f64` | `9beb634425faf2c2` |

## Performance (i9-12900KF, RTX 3060 12 GB)

Kernel cost per f64x4 lane group, in TSC cycles:

| Kernel | Table walk | Generated |
| --- | --- | --- |
| fun | 345 | 46 |
| −J | 862 | 174 |
| LU factor | 1218 | 350 |
| solve | 437 | 296 |

Tolerance study (`--bench`): 8 humid-air zones, 20 ms pulsed, compared with a reference at
rtol 1e-9:

| rtol / atol (cm⁻³) | Steps per zone | Worst product error |
| --- | --- | --- |
| 1e-4 / 1e2 | 23 321 | 1.1e-6 |
| 1e-3 / 1e4 | 10 553 | 1.4e-5 |
| **1e-3 / 1e8 (default)** | 6 902 | 1.7e-5 |
| 1e-3 / 1e9 | 5 254 | 3.6e-5 |

atol 1e8 cm⁻³ is about 4 ppt. It stops tracking radicals between pulses below that level,
and the products don't notice.

Throughput history of the 64-zone pulsed reactor, in simulated seconds per wall second:

| Stage | Rate |
| --- | --- |
| table kernels, f64 | ~0.4 |
| + generated kernels | 1.7 |
| + atol 1e8 | **3.0** |
| averaged discharge | 103 (1 ms coupling), 260 (5 ms) |

Sweeps over 4096 independent zones, 8 P-cores vs CUDA, f32 lanes:

| Case | 8 P-cores | CUDA |
| --- | --- | --- |
| averaged, 1 ms coupling | 1.47 | 1.13 |
| averaged, 20 ms coupling (default) | **21.3** | 18.1 |
| pulsed, 20 ms coupling | 0.06 | **0.09** |

### Hybrid: CPU and GPU together (`--hybrid`, UI "Hybrid")

CPU and GPU lanes perform identical IEEE operations, so a batch can be **split between the
two in any ratio without changing a single bit** (tested at splits 0/8/24/40/64 of 64 zones,
f32 and f64). The GPU takes lanes [0, n) asynchronously: the zone block is page-locked
once, and lanes [0, n) of every row move in one `cudaMemcpy2DAsync`. Meanwhile the CPU work
queue integrates [n, N).

The split is chosen by measurement:
1. One CPU-only step gives the CPU cost per lane.
2. Two GPU probes (8 lanes and half) give the GPU's fixed latency and per-lane cost.
3. The split that equalizes the two follows analytically, or is 0 if the GPU's latency
   alone exceeds the whole CPU step.
4. A local search refines it: step time is kept as an EMA per split; neighbours are probed
   at several scales; the probe interval backs off to 1024 steps while probes lose, and
   splits known to be worse are skipped.

The split is timing-dependent, but the results are not.

| Case (4096 zones unless noted) | CPU (8 P-cores) | CUDA | Hybrid |
| --- | --- | --- | --- |
| sweep, pulsed | 0.06 | 0.09 | **0.14** (GPU takes ~63%) |
| sweep, averaged (20 ms chemistry step) | ~17–20 | 18 | ~17 (the per-step work is too small for the GPU's latency) |
| reactor, 64 zones, pulsed | ~2.8–3.4 | 0.006 | ~2.8–3.1 (picks CPU-only) |

Machine noise is about ±15% (other jobs were running).

### Warp per zone (`gpu_kin_set_mode(1)`)

A second CUDA decomposition: one warp cooperates on each zone.
- Rates, Jacobian terms and each LU elimination's updates run across the 32 threads.
- The triangular solves are column-oriented.
- Scratch lives in shared memory, about 4 KB per zone.
- Every element is produced by the same operation sequence as the CPU table walk.
  - To make the backward solve column-orientable, **all paths now accumulate each row's
    upper entries in descending column order**, which changed the reference hashes.
- It is bit-identical to the CPU in f32 and f64, pulsed and averaged (tested).

It is also **3–36× slower than one thread per zone** for this 17-species mechanism, and the
reason was measured:
- The LU has 104 strictly sequential eliminations averaging 7 updates each.
- There are about 300 warp syncs per Rosenbrock step.
- The table-driven kernels can't use the generated straight-line code.

It stays as a verified option for larger mechanisms (e.g. reduced `dbd_ref` sets with 40–80
species), where one thread's state no longer fits in registers. Thread per zone remains the
default.

| f32 lanes, time for the run | thread / zone | warp / zone | 1 AVX2 core |
| --- | --- | --- | --- |
| 64 zones × 2 ms pulsed | 16 ms | 97 ms | 9 ms |
| 4096 zones × 2 ms pulsed | 31 ms | 1112 ms | 553 ms |
| 16384 zones × 2 ms pulsed | 119 ms | 4321 ms | — |

### A GPU race found and fixed

A sweep at 169+ zones gave a different hash on CUDA than on the CPU, and so did two CUDA runs
of the same case. The cause was in the synchronous path:
- Uploads used a plain `cudaMemcpy` on the legacy default stream.
- From pageable memory that call returns once the data is **staged**, before the DMA lands.
- The kernel ran on the module's **non-blocking** stream, which doesn't order against the
  default stream, so it could read partly-old inputs.

Now every copy and kernel goes on the module's stream, and synchronous copies wait for it.
The regression test runs 64 zones × 10 × 20 ms with an upload every step, in both precisions.
Sweeps at 169, 256 and 4096 zones now give identical hashes on CPU, CUDA and hybrid.

### Why CUDA is not faster here, and what fp32 changed

The RTX 3060 (GA106) runs FP64 at 1/64 of its FP32 rate, about 0.2 TFLOPS, and has no
hardware f64 divide or square root. Eight P-cores with AVX2 FMA reach about 0.6 TFLOPS of
FP64. A one-thread-per-zone Rosenbrock step is also a chain of thousands of dependent
operations over a working set of several KB that lives in local memory, which spills out of
L1. The f64 kernel therefore loses to the CPU at every size. These were measured, not
assumed: shrinking the frame from 11 KB to 4 KB or staging tables in shared memory did not
move the per-step latency.

fp32 lanes (delta formulation) made the GPU path 8× faster than CUDA f64. At 4096 pulsed
zones it is about 22× one AVX2 core, and about 1.5× all eight cores. The remaining limit is
occupancy: the generated f32 kernel uses 255 registers per thread, so only 8 warps fit per
SM, and the per-thread dependency chains can't be hidden. Making the GPU clearly win would
need one warp per zone, with the LU parallel inside the system and the working set in shared
memory. That is listed as a follow-up. On this machine the CPU is the default backend, and
CUDA is the better choice for large pulsed sweeps.

## Reference solvers and validation

Two offline codes check the reduced chemistry; build steps, local patches and the full
comparison are in `tools/ref/README.md`.

- **LoKI-B++** (two-term Boltzmann, IST-Lisbon N₂ / O₂ cross sections) gives the source's
  G-values against E/N: `air_source_table.h`.
  - At 180 Td, per 100 eV: 3.88 O₂ dissociations (53 % via O(¹D)), 4.34 N₂(A), 0.48 N₂ → N +
    N(²D). The vibrational share is 20 % of the power.
  - Turn on **Boltzmann G-values** in the source panel and set the micro-discharge E/N.
- **Filament pulses.** A real DBD deposits its energy in micro-discharges of tens of mJ/cm³,
  not evenly.
  - With **filament mJ/cm³** set, each gas parcel is hit at rate power / E_f.
  - 0 keeps the averaged pulse per half-cycle, which is the default and keeps the hashes
    above.
- **ChemPlasKin** (Cantera + CppBOLOS) runs full electron, ion and excited-state kinetics
  (710 reactions, with N₂O₅ added). The run is dry air at 180 Td, 2.16 J/cm³ over 1.32 s.
  Against it, the reduced model:
  - **Fixed nitrogen:** within ±45 % (3 % at E_f 30).
  - **N₂O:** 1.1–2.2×.
  - **O₃:** +35 %. On one fresh pulse odd oxygen is 10 % low; the excess is lumped ozone loss
    at high O₃.
  - `test_reference_source` records these and fails if the mechanism drifts away.
- **What the comparison fixed:**
  - N + NO₂ now uses Kossyi's four-way branching (N₂O + O 43 %, 2 NO 33 %, N₂ + 2 O 13 %,
    N₂ + O₂ 10 %). The single JPL channel tripled N₂O.
  - N₂(A) + N₂O → N₂ + N + NO.

## Known limits of the chemistry

- No ions, no N₂(v) vibrational kinetics, no surface chemistry. Above about 400 K, NO from
  N₂(v) + O → NO + N and the thermal ozone/NOx switch it drives are missing.
- **Metastables are quasi-steady.** N(²D), N₂(A) and O(¹D) live 2–50 ns at 1 atm, so each
  pulse distributes them over their partners in proportion to k·[partner].
  - A run with them as explicit species gave the same outlet to 2e-4 and 1.8× the solver
    steps.
  - Before this, the mechanism made NO only from ground-state N, so NOx cost 450–6650 MJ/mol.
- **N(⁴S) fate:** N + NO₂ is now branched (Kossyi). N₂O still runs 1.1–2.2× the full-kinetics
  reference.
- **Ozone:** about 35 % above full kinetics at high O₃. Singlet oxygen and ions are lumped.
- Gas temperature is fixed per zone (no heating feedback). Thermal O₃ decomposition
  (O₃ + M) is included.
- Third-body efficiencies are those of air; fall-off uses the JPL form with Fc = 0.6.
- Plug flow is a chain of stirred zones (64 CSTRs by default).
- f32 lanes drift up to ~2e-7 of an element per simulated second (H in dry-ish air, where
  O(¹D) + H₂O and N₂(A) + H₂O turn OH over fast relative to the H inventory; N and O stay
  ~1e-9). Use `--precision f64` for multi-hour runs.

## Next steps

1. Parameter registry with save/load (§5.2 of the plan); the UI already edits the
   operating point live.
2. Gas → liquid transfer from the reactor outlet: the first NFT coupling (M3).
3. `dbd_ref` (ChemPlasKin) and `tools/reduce_dbd` tables replacing the G-values.
4. N₂(v) chemistry (N₂(v) + O → NO + N). It isn't in ChemPlasKin either, so it needs a
   state-to-state vibrational reference (LoKI-C / LoKI-GM) to calibrate against.
4. Warp-per-zone CUDA kernel for large sweeps.
5. Windows: `build.bat` does not build the UI object, the codegen step or the tests yet.
