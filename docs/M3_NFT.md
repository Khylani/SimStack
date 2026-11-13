# M3: NFT system — channels, hydraulics, reservoir + chiller, plants

A nutrient-film-technique growing system simulated end to end. It covers the PVC hydraulic
network that feeds the channels, the thin film flowing down each channel and through the root
mats, heat gain through the channel wrap and evaporative cooling, the reservoir with its
chiller, dosing, pH control and top-off, and plants growing, transpiring, taking up nutrients
(which moves the pH) and being harvested. It scales by **channel count, channel
length and plants per channel**, and is built for throughput: lanes are channels in f32x8
registers, and each 8-channel group is an L1/L2-resident block owned by one thread.

```sh
./build/simstack --scene nft                                   # real-time UI
./build/simstack --scene nft --layer 1                         # start on the "temperature rise vs inlet" layer
./build/simstack --scene nft --headless 4200 --speed 86400 --csv runs/nft.csv           # 70 days in ~18 s
./build/simstack --scene nft --channels 512 --channel-length 12 --plants 80 --headless 1260 --speed 86400
./build/simstack_tests                                         # includes the hydraulics and NFT checks
```

Defaults:
- 16 channels × 4 m × 26 plants (15 cm spacing), 100 × 50 mm profile at 2% grade.
- **Cilantro** in bank A and **arugula** in bank B, **both wrapped in bubble mylar**.
- Indoor herb room at 22 ± 3 °C with a 16 h photoperiod and 150 W/m² radiant load on the
  channel tops.
- Chiller set to 20 °C, EC 1.6 mS/cm with uptake-matched dosing, pH 5.8 ± 0.2 with 1 M HNO3 /
  KOH. Room at 60% RH and 420 ppm CO2; tap water with 1.5 meq/L alkalinity.
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

**Channels** (per lane, per cell):
- The root-free film depth comes from the channel's own flow and its bank's grade. The
  regime is chosen by film Reynolds number: Nusselt laminar film below 400, Manning above
  1600, blended in log space between. Root mats then deepen and slow the film cell by cell
  (see *Root mats* below).
- Upwind advection of temperature, DO and 7 ions (NO3-N, NH4-N, P, K, Ca, Mg, S) in flux
  form, `x_j += (Q dt / V_j)(x_{j-1} − x_j)`, so it conserves mass with a different water
  volume in every cell. Courant ≤ 0.9 on the root-free film, and roots only lower it.
- **Wall heat:** U × outer perimeter × (ambient − film). Radiant gain is absorptivity ×
  width × load, while lights are on.
  - Bubble mylar wrap: U 2.0 W/m²K, absorptivity 0.06.
  - Bare white PVC: U 6.0 W/m²K, absorptivity 0.35.
- **Evaporative cooling** of the film's open area (planting holes and gaps, 8% by default):
  latent heat λ·E per cell (see *Evaporation*).
- O2 re-aeration toward saturation at the film temperature. kLa = kL·A/V, with kL fixed by
  the root-free film, so a film deepened by roots holds less O2 per unit of root.

**Reservoir** (f64):
- Return mixing; tank walls; pump heat; evaporation from the uncovered tank surface (10%
  by default) with its latent heat.
- **Chiller** with hysteresis band, capacity and COP (electrical energy accounted).
- Aeration; water drawn into or released by the root mats (with its solutes and heat);
  plant water use and evaporation; top-off with tap water (which carries alkalinity).
- **EC dosing:** 100× stock, rate-limited. The stock is in the crops' uptake ratios by
  default, because a recipe-ratio stock lets Ca and S accumulate and K run out in a
  recirculating system (the model reproduces that drift).
- **pH control:** 1 M HNO3 (adds NO3-N) or 1 M KOH (adds K), dosed to the set point once
  the tank leaves the band, rate-limited. The fresh batch is adjusted at make-up.
- Optional scheduled solution change.

**Plants** (one per site):
- Logistic growth × root-zone temperature (crop trapezoid) × DO factor × nutrient
  sufficiency. Sufficiency is the Liebig minimum of Michaelis–Menten uptake capacity over
  demand.
- Uptake = tissue composition × actual growth, so nutrient mass balances exactly.
- **Water use:** canopy transpiration (Baille et al. 1994) plus the water held in new tissue
  (see *Water* below). Root respiration has Q10 = 2.
- Growth × a pH envelope (1 over pH 5.0–7.0, falling to 0 at 3.5 and 9.5). This is an
  empirical stand-in for P, Fe and Mn availability, not a micronutrient speciation model.
- Harvest at the crop's cycle length, then replant. Planting is staggered.
- Bolting risk from air degree-days above the crop's threshold.
- **Crops:** cilantro, arugula, lettuce, basil, kale, spinach, strawberry. The values are
  illustrative literature ranges and tunable per cultivar.

## Root mats, water and pH

All three are physics with closed books, not adjustment factors. The water code is in
`src/sim/nft_water.cpp` (f64).

### Root mats

- **Volume:** each plant's roots (root share × dry mass × fresh/dry 15, at 1 g/cm³) spread
  over the cells of its planting span. That gives a root "thickness" t_r per unit floor
  area.
- **Permeability K** of a bed of 0.4 mm fibres at solid fraction φ: the smaller of Jackson &
  James (1986, dilute random fibres) and Carman–Kozeny for fibres (Kozeny constant 5, dense
  beds).
- **Flow through the mat:** Brinkman's equation for a film of depth h on a grade S, with no
  slip on the floor and a free surface:

  `q = (g S K / ν) · (h − √K · tanh(h / √K))`

  This is exactly the Nusselt film g S h³ / 3ν as K → ∞, and Darcy flow in a dense mat.
- **Depth:** the deeper of two limits.
  - Sparse roots displace water and the film rises by their volume: h₀ + t_r.
  - In a dense mat the flow has to pass through the roots: solve q_Brinkman(h, K(φ = t_r/h))
    = q for h (φ capped at 0.35; above that the mat stands above the water).
  - Either way the water depth h(1 − φ) ≥ h₀, so roots only lower the Courant number and
    the substep stays set by the root-free film.
- **Solver:** Newton in log h with the analytic derivative, warm-started from the last
  solution. It converges to 1e-12 in 1–2 iterations.
- **Cadence:** each group re-solves its cells every 30 windows (staggered by group), and
  after any hydraulic change.
- **Water balance:** water drawn into a thickening mat leaves the tank with its solutes and
  heat; a harvest releases it again.
- **Warning:** the UI flags cells whose film rises above 80% of the channel wall.
- **Default system, 30 days:** mature mats raise the film from 1.6 mm to about 9 mm in
  places, and the channels hold 23 → 24 L.

### Water

- **Transpiration** (Baille et al. 1994, greenhouse crops):

  `λE = A (1 − e^(−k LAI)) Rn + B · LAI · VPD`

  - A = 0.30, B = 16 W m⁻² kPa⁻¹ per unit LAI, k = 0.7.
  - Leaf area comes from shoot dry mass × the crop's specific leaf area, over the ground area
    per plant (channel pitch × plant spacing).
  - With the lights off the radiation term is 0 and the B term is scaled by 0.25 (stomata
    mostly closed).
  - Plants also hold 14 g of water per g of dry mass grown.
  - Default system: about 50 mL per plant per day.
- **Evaporation** from the film's open area and the tank's uncovered surface:
  - Mass transfer by the Lewis (Chilton–Colburn) analogy, `E = h_c / (c_p Le^(2/3)) ·
    0.622 / p · (e_s(T_w) − RH·e_s(T_a))`, with Buck's (1996) vapour pressure.
  - Latent heat λ(T)·E is taken from the water. In the kernel, e_s is linearized around the
    inlet temperature each window, so the per-cell heat update stays one fused
    multiply-add.
  - Water leaves without its ions, so the tank concentrates until top-off.

### pH

- **Alkalinity is conservative.** By charge balance, TA (Dickson's proton reference:
  H2CO3*, H2PO4⁻, NH4⁺, H2O) equals the strong-ion sum NH4 + K + 2Ca + 2Mg − NO3 − P − 2SO4,
  plus a tracked remainder R. R holds ions the model doesn't track: Na, Cl, the tap water's
  bicarbonate.
- So nothing else moves pH:
  - **Plants** change it only through the ions they take up. Excess anion uptake
    (nitrate-dominated) raises pH; ammonium uptake lowers it.
  - **Acid and base** change it through the NO3 / K they add.
  - **Top-off** water adds its alkalinity to R. The dosing stock is neutral salts.
- **pH from TA:** solved by speciation, with the carbonate system open to the room's CO2
  plus phosphate, ammonium and water.
  - Constants: K0 (Weiss 1974), K1 and K2 (Plummer & Busenberg 1982), Kw (Harned & Owen),
    NH4⁺ (Emerson et al. 1975), phosphate by van't Hoff from 25 °C.
  - Activities by Davies, with A(T) from the dielectric constant of water.
  - Bracketed Newton converges to 1e-10 in about 6 steps.
- **Local pH in the 3D view:** solved exactly per cell up to 32k cells. Above that it is
  linearized through the tank's buffer capacity.
- **Behaviour reproduced:**
  - With 10 mg/L NH4-N, ammonium takes about 22% of N uptake and nearly balances the plants'
    charge uptake. This is why growers add 5–15% NH4-N to steady the pH.
  - All-nitrate feeding drives the pH up (5.80 → 6.40 in a day without control); an
    ammonium-rich feed drives it down (5.80 → 5.43).
  - With control on, the acid demand comes mostly from tap-water alkalinity and the batch
    make-up.

## Exactness

The hard part of fp32 is small changes on large values. Three mechanisms make the books
close:

1. Rail cells store **deltas from the window's inlet**, which are about 1000× smaller than
   the absolute values, so every rounding is about 1000× smaller. This took the 30-day ion
   mass-balance error from 1.2e-4 to 3e-8.
   - NH4 is the exception. It is depleted strongly along the channel, so its deltas are not
     small, and it now sets the worst figure: 2.7e-7.
   - The other ions stay at 1e-8. The new water terms, the root-mat volumes and the uptake
     sum were each ruled out as the source.
2. The tank is debited **exactly the f32 inlet** the rails received, so the f32/f64
   boundary neither creates nor destroys mass.
3. Uptake is booked as **exactly what was removed**: `x_before − x_after` is exact in f32
   by Sterbenz's lemma.
4. Each cell's book volume is defined as **exactly Q·dt / c** (f64 of the f32 Courant
   number), so the per-cell upwind fluxes telescope exactly despite variable volumes. Root-mat
   volume changes move water, solutes and heat between tank and channels in f64.

| Check (`simstack_tests`) | Result |
| --- | --- |
| f32x8 lanes vs scalar f32 lanes | bit-identical |
| Ion mass balance, 30 days with dosing, top-off, pH control and a solution change | 2.7e-7 (NH4); others ≤ 4e-8 |
| Water balance (plants, film + tank evaporation, top-off, stock, acid/base, dump, root holdup) | 9e-15 |
| Root-mat holdup books vs channel volume | 7e-16 |
| Steady-state energy balance (pump heat = channel walls + tank walls + latent heat) | residual 2.4e-4 |
| Brinkman film: Nusselt limit / dense limit / series vs closed form | 4e-12 / 0 / 1e-12 |
| Root-mat depth: flow residual; analytic vs numerical Newton derivative | 1e-12; 1e-6 |
| Vapour pressure vs IAPWS (20, 25 °C) | 0.04% |
| pK1, pK2, pKw, pKa(NH4), pK2(P) at 25 °C vs reference | within 0.005 |
| pH: pure water / rain at 420 ppm CO2 / 1 meq/L bicarbonate under air | 7.00 / 5.60 / 8.18 |
| pH controller over 30 days | 5.80 at set 5.8 ± 0.2 |
| pH drift without control, 1 day: all nitrate / 40 mg/L NH4-N | 5.80 → 6.40 / 5.80 → 5.43 |
| Chiller holds its band (mylar) | 19.6..20.3 °C at set 20 ± 0.3 |
| Mylar vs bare PVC at 28 ± 4 °C, 300 W/m² | chiller 2.8 vs 8.4 kWh/day |
| Growth vs the analytic logistic curve (all limits 1) | 0.01% |
| Water viscosity / density vs IAPWS | 0.02% |
| Colebrook residual / Moody value | 0 / f = 0.01851 at Re 1e5, ε/D 1e-4 |
| Network flow conservation / pump-curve balance / balanced taps | 1e-12 / 1e-9 / 7e-11 spread |

## Memory layout and performance

```
NftGroup (alignas 64, one per 8 channels, owned by one thread per 60 s window)
  lane params    f32[8] each: flow, root-free depth and velocity, evaporated kg
  crop params    f32[8] each (one crop per lane)
  rows[cells]    NftRow      { f32 f[9 fields][8 lanes] }               288 B per cell
  hot[cells]     NftCellHot  { c, kT, bT, ko }[8 lanes]                 128 B per cell  (read every substep)
  cold[cells]    NftCellCold { V (f64), 1/V, kw, kr, ge, ev, depth, phi, velocity }[8]  (per window / root solve)
  sites[plants]  NftSiteRow  { W, age, drop[9], uptake[7], removed[9], ... }[8 lanes]
```

- An 80-cell group is about 23 KB of rows (L1). A 240-cell group is 69 KB (L2).
- One window is about 230 substeps over every row of a group, with no synchronisation;
  groups sync with the reservoir once per window.
- **Row kernel:** about 55 vector µops per cell-row at about 4 instructions per cycle, about
  4.0 ns per row-substep, so it's issue-bound. The per-cell coefficients are four extra
  aligned loads per row; the heat update (walls, radiant, latent) is one fnmadd + add.
- **Flush-to-zero in the NFT work:** after every rebase, deltas near the inlet decay
  geometrically toward the inflow's 0. In root cells (Courant 0.1–0.3) they spend about 100
  substeps in the subnormal range, which made the whole transport 1.5× slower. The group's
  work runs with FTZ/DAZ and restores MXCSR afterwards, so the f64 kinetics keep IEEE
  behaviour; values below 1e-38 mg/L carry no physics. Results are unchanged across thread
  counts, and f32x8 vs f32 is still bit-identical.
- What was measured and dropped:
  - Carrying the upstream row in registers spills (9 + 9 YMM > 16), so it's slower.
  - Small systems run the window inline, because dispatch costs more than about 0.3 ms of
    work.
  - Workers spin about 50 µs before sleeping, so back-to-back windows skip the OS wake.

| System | Speed |
| --- | --- |
| 8 × 4 m × 26 plants, 1 thread | 77 µs per window (transport 68, root solve 4 amortized), 780,000× real time |
| 16 × 4 m × 26 plants | ~0.16 ms per window, 10 days in 2.3 s |
| 512 × 12 m × 80 plants (40,960 plants) | ~2.3 ms per window, ~26,000× real time |

Roots, evaporation, transpiration and pH cost about 22% over M3 on one thread (8 channels:
111 vs 91 ms per simulated day).

## Real-time UI

- **Controls:**
  - System: channels, length, plants per channel, rebuild, auto-size.
  - Banks: crop, wrap, grade, stagger.
  - Hydraulics: pump, PVC schedule and sizes, lift, fittings, manifold, feed bore and
    length, profile, balancing, eductor, return. The operating point is shown live: pump
    flow, head and power, channel spread, eductor circulation and turnover, film depth,
    velocity, Re and regime, and return-drain use.
  - Climate: RH, CO2, air-film coefficient, channel and tank open fractions; VPD, dew point
    and evaporation readouts.
  - Water & roots: Baille A / B, night stomata, channel pitch, root diameter, fresh/dry,
    packed-mat fraction. Readouts: plant water and evaporation, channel holdup, deepest film,
    and an overflow warning.
  - pH: control, set point, band, dose rate, tap alkalinity. Readouts: pH, alkalinity, ionic
    strength, acid and base used, growth factor.
  - Reservoir and chiller, with energy readouts.
  - Nutrients: EC setpoint, dosing mode, solution changes, ion table.
  - Harvest summary and CSV export.
- **Plots:** temperatures with chiller duty, nutrients with EC, plant mass and harvest, pH
  with acid/base doses, water use / evaporation / holdup / deepest film, limitation
  factors, along-channel profiles of any field, and manifold flow with tap head.
- **3D layers:**
  - Film: temperature; **rise vs inlet** and **vs ambient** (diverging blue–white–red).
  - Solution: DO, NO3-N, K, Ca, local EC.
  - Flow: film velocity and depth per cell (through the root mats), channel flow.
  - Water and pH: local pH, root-mat volume fraction, evaporative cooling (W/m).
  - Plants: mass and fT / fN / fDO.
  - The grade is drawn exaggerated 10×, and plants rise with dry mass. Each layer has a
    colour bar with units.

## Limits

- Root mats occupy their own plant's span, uniformly; real mats also grow downstream and
  can dam the flow.
- Brinkman flow assumes the mat fills the film's depth. A mat thinner than the film, with
  free water flowing over it, would be a two-layer problem.
- Water lost from the film is booked at the tank. Evaporation and transpiration don't
  concentrate the film along the channel itself; that effect is about 1e-4 per pass.
- Root respiration CO2 doesn't lower the film's pH locally; the films are taken as
  equilibrated with room CO2.
- The tank is one well-mixed volume (the eductor justifies that); there is no
  stratification and no chiller-coil heat-transfer model.
- pH acts on growth through an empirical envelope, not through Fe/Mn/P speciation and
  precipitation.
- Ambient is a sinusoid; bolting uses air degree-days only, not photoperiod.
