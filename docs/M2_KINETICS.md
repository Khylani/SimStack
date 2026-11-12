# M2 (part 1): time-dependent DBD chemistry solver and real-time UI

The first milestone-2 piece from the architecture plan is the dynamic (time-dependent)
plasma-chemistry solver that the plan calls `dbd_fast`, together with a real-time UI to
drive it. It is a batched stiff ODE solver for mass-action chemistry in which every reactor
zone is a lane. The same source code runs on AVX2 (4 f64 or 8 f32 lanes per register) and
on CUDA (one zone per thread), and the results are bit-identical across all of them.

Not in this part (still M2): the ChemPlasKin/Cantera reference solver (`dbd_ref`), the
reduction tables generated from it, and the LoKI-B++ cross-checks. Until those exist, the
discharge is represented by G-values (radicals per 100 eV), which are tunable surrogates.

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
| 2286 ppm | 0.08 | 1.9 | 3.3 | 0.11 | 35.6 | 15 | 152 |

That is about 77 g O3/kWh, inside the 50–100 g/kWh range of air-fed ozonizers. The sweep
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
| `src/sim/kinetics/air_mech.cpp` | 17-species / 40-reaction humid-air afterglow mechanism (JPL/IUPAC), G-value source, feed gas |
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
| f32 (default) | `3741c9877f88ad5d` |
| `--precision f64` | `a2be2f374ec0cfb4` |

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

## Known limits of the chemistry

- Neutral afterglow only: no ions, no O(¹D) / N₂(A) / N₂(v) kinetics, no surface chemistry.
  Their net effect is folded into the G-values.
- Fixed gas temperature per zone, so no thermal O3 destruction and no NOx-mode
  ("discharge poisoning") switch.
- Third-body efficiencies are those of air; fall-off uses the JPL form with Fc = 0.6.
- Plug flow is a chain of stirred zones (64 CSTRs by default).
- f32 lanes drift ~4e-8 of each element per simulated second. Use `--precision f64` for
  multi-hour runs.

## Next steps

1. Parameter registry with save/load (§5.2 of the plan); the UI already edits the
   operating point live.
2. Gas → liquid transfer from the reactor outlet: the first NFT coupling (M3).
3. `dbd_ref` (ChemPlasKin) and `tools/reduce_dbd` tables replacing the G-values.
4. Warp-per-zone CUDA kernel for large sweeps.
5. Windows: `build.bat` does not build the UI object, the codegen step or the tests yet.
