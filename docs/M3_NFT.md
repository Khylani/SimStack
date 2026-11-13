# M3: NFT system — channels, hydraulics, reservoir + chiller, plants

A nutrient-film-technique growing system simulated end to end. It covers the PVC hydraulic
network that feeds the channels, the thin film flowing down each channel, heat gain through
the channel wrap, the reservoir with its chiller, dosing and top-off, and plants growing,
transpiring, taking up nutrients and being harvested. It scales by **channel count, channel
length and plants per channel**, and is built for throughput: lanes are channels in f32x8
registers, and each 8-channel group is an L1/L2-resident block owned by one thread.

```sh
./build/simstack --scene nft                                   # real-time UI
./build/simstack --scene nft --layer 1                         # start on the "temperature rise vs inlet" layer
./build/simstack --scene nft --headless 4200 --speed 86400 --csv runs/nft.csv           # 70 days in ~15 s
./build/simstack --scene nft --channels 512 --channel-length 12 --plants 80 --headless 1260 --speed 86400
./build/simstack_tests                                         # includes the hydraulics and NFT checks
```

Defaults:
- 16 channels × 4 m × 26 plants (15 cm spacing), 100 × 50 mm profile at 2% grade.
- **Cilantro** in bank A and **arugula** in bank B, **both wrapped in bubble mylar**.
- Indoor herb room at 22 ± 3 °C with a 16 h photoperiod and 150 W/m² radiant load on the
  channel tops.
- Chiller set to 20 °C, EC 1.6 mS/cm with uptake-matched dosing.
- Auto-sizing on: 312 L reservoir, 1200 GPH pump, ¾″ Sch 40 supply and manifold, 2″
  return, about 490 W chiller.

## Model

**Hydraulics** (`src/sim/nft_hydraulics.cpp`, f64, solved on change or when the tank moves
1 K):
- **Pipes:** PVC Schedule 40/80 inside diameters per ASTM D1785 (½″ to 4″), roughness
  1.5 µm.
- **Friction:** Darcy–Weisbach. Colebrook–White is solved by Newton to machine precision;
  laminar flow uses 64/Re, blended across the transition.
- **Water properties:** viscosity (Vogel) and density (Tanaka) at the current tank
  temperature.
- **Pump:** submersible presets (400 to 3200 GPH) with H(Q) and η(Q) curves. Shaft power
  ρgQH/η heats the solution (submersible) or only its hydraulic power does (external).
- **Network:** pump → supply riser (lift + fitting K) → end-fed, dead-ended manifold with
  one tap per channel. Each tap is a feed tube discharging to atmosphere. The manifold is
  marched tap by tap from the dead end, nested in Illinois regula falsi on the pump curve:
  flow conserved to 1e-12, curve balanced to 1e-9, in about 5 iterations. Optional
  balancing valves equalize channel flows.
- **Mixing eductor** on the pump discharge: motive flow Cd·A·√(2gH), circulation × (1 +
  entrainment), tank turnover reported.
- **Return drain:** full-pipe Manning capacity, with a warning above 80%.

**Channels** (per lane):
- Film depth and velocity come from the channel's own flow and its bank's grade. The
  regime is chosen by film Reynolds number: Nusselt laminar film below 400, Manning above
  1600, blended in log space between.
- Upwind advection of temperature, DO and 7 ions (NO3-N, NH4-N, P, K, Ca, Mg, S) at Courant
  ≤ 0.9.
- **Wall heat:** U × outer perimeter × (ambient − film). Radiant gain is absorptivity ×
  width × load, while lights are on.
  - Bubble mylar wrap: U 2.0 W/m²K, absorptivity 0.06.
  - Bare white PVC: U 6.0 W/m²K, absorptivity 0.35.
- O2 re-aeration toward saturation at the film temperature.

**Reservoir** (f64):
- Return mixing; tank walls; pump heat.
- **Chiller** with hysteresis band, capacity and COP (electrical energy accounted).
- Aeration; transpiration loss; top-off with fresh water.
- **EC dosing:** 100× stock, rate-limited. The stock is in the crops' uptake ratios by
  default, because a recipe-ratio stock lets Ca and S accumulate and K run out in a
  recirculating system (the model reproduces that drift).
- Optional scheduled solution change.

**Plants** (one per site):
- Logistic growth × root-zone temperature (crop trapezoid) × DO factor × nutrient
  sufficiency. Sufficiency is the Liebig minimum of Michaelis–Menten uptake capacity over
  demand.
- Uptake = tissue composition × actual growth, so nutrient mass balances exactly.
- Transpiration follows growth; root respiration has Q10 = 2.
- Harvest at the crop's cycle length, then replant. Planting is staggered.
- Bolting risk from air degree-days above the crop's threshold.
- **Crops:** cilantro, arugula, lettuce, basil, kale, spinach, strawberry. The values are
  illustrative literature ranges and tunable per cultivar.

## Exactness

The hard part of fp32 is small changes on large values. Three mechanisms make the books
close:

1. Rail cells store **deltas from the window's inlet**, which are about 1000× smaller than
   the absolute values, so every rounding is about 1000× smaller. This took the 30-day ion
   mass-balance error from 1.2e-4 to 3e-8.
2. The tank is debited **exactly the f32 inlet** the rails received, so the f32/f64
   boundary neither creates nor destroys mass.
3. Uptake is booked as **exactly what was removed**: `x_before − x_after` is exact in f32
   by Sterbenz's lemma.

| Check (`simstack_tests`) | Result |
| --- | --- |
| f32x8 lanes vs scalar f32 lanes | bit-identical |
| Ion mass balance, 30 days with dosing, top-off and a solution change | 3e-8 |
| Water balance | 1e-15 |
| Steady-state energy balance (pump heat = channel walls + tank walls) | residual 2e-4 |
| Chiller holds its band (mylar) | 19.6..20.3 °C at set 20 ± 0.3 |
| Mylar vs bare PVC at 28 ± 4 °C, 300 W/m² | chiller 2.8 vs 8.4 kWh/day |
| Growth vs the analytic logistic curve (all limits 1) | 0.01% |
| Water viscosity / density vs IAPWS | 0.02% |
| Colebrook residual / Moody value | 0 / f = 0.01851 at Re 1e5, ε/D 1e-4 |
| Network flow conservation / pump-curve balance / balanced taps | 1e-12 / 1e-9 / 7e-11 spread |

## Memory layout and performance

```
NftGroup (alignas 64, one per 8 channels, owned by one thread per 60 s window)
  lane params    f32[8] each: courant, k_wall, k_rad, k_o2, 1/V, flow, depth, velocity
  crop params    f32[8] each (one crop per lane)
  rows[cells]    NftRow  { f32 f[9 fields][8 lanes] }   288 B per cell
  sites[plants]  NftSiteRow { W, age, drop[9], uptake[7], removed[9], ... }[8 lanes]
```

- An 80-cell group is about 23 KB of rows (L1). A 240-cell group is 69 KB (L2).
- One window is about 230 substeps over every row of a group, with no synchronisation;
  groups sync with the reservoir once per window.
- **Row kernel:** about 50 vector µops per cell-row at about 4 instructions per cycle. That
  is about 18 G lane-field updates per second per core, so it's issue-bound.
- What was measured and dropped:
  - Carrying the upstream row in registers spills (9 + 9 YMM > 16), so it's slower.
  - Small systems run the window inline, because dispatch costs more than about 0.3 ms of
    work.
  - Workers spin about 50 µs before sleeping, so back-to-back windows skip the OS wake.

| System | Speed |
| --- | --- |
| 16 × 4 m × 26 plants | ~0.13 ms per window, 70 days in ~15 s |
| 512 × 12 m × 80 plants (40,960 plants) | ~3 ms per window, ~29,000× real time |

## Real-time UI

- **Controls:**
  - System: channels, length, plants per channel, rebuild, auto-size.
  - Banks: crop, wrap, grade, stagger.
  - Hydraulics: pump, PVC schedule and sizes, lift, fittings, manifold, feed bore and
    length, profile, balancing, eductor, return. The operating point is shown live: pump
    flow, head and power, channel spread, eductor circulation and turnover, film depth,
    velocity, Re and regime, and return-drain use.
  - Climate.
  - Reservoir and chiller, with energy readouts.
  - Nutrients: EC setpoint, dosing mode, solution changes, ion table.
  - Harvest summary and CSV export.
- **Plots:** temperatures with chiller duty, nutrients with EC, plant mass and harvest,
  limitation factors, along-channel profiles of any field, and manifold flow with tap head.
- **3D layers:**
  - Film: temperature; **rise vs inlet** and **vs ambient** (diverging blue–white–red).
  - Solution: DO, NO3-N, K, Ca, local EC.
  - Flow: film velocity, depth, channel flow.
  - Plants: mass and fT / fN / fDO.
  - The grade is drawn exaggerated 10×, and plants rise with dry mass. Each layer has a
    colour bar with units.

## Limits

- The film is laminar or turbulent sheet flow on a smooth floor. Root mats in mature
  channels deepen and slow it; that isn't modelled yet (a root-mat roughness factor is the
  next step).
- The tank is one well-mixed volume (the eductor justifies that); there is no
  stratification and no chiller-coil heat-transfer model.
- No pH model yet: NO3/NH4 uptake drives pH in practice.
- Ambient is a sinusoid; bolting uses air degree-days only, not photoperiod.
