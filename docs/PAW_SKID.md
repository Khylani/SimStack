# PAW skid preset: three DBD modules, 50 gal batches, 300 gal NFT

`--preset paw-skid` loads the plasma-activated-water skid from the design review ("DBD
Reactor Design Review — 50 gal PAW" and its "PAW Plasma Skid" page) and couples it to the
NFT system as the reservoir's top-off source.

```sh
./build/simstack --scene nft --preset paw-skid                          # UI (also: preset picker + Load)
./build/simstack --scene nft --preset paw-skid --headless 1800 --speed 86400   # 30 days
./build/simstack --scene dbd --preset paw-skid                          # one module's annulus in the M2 kinetics
```

## What the preset is

| Part | Value (from the design) |
| --- | --- |
| Modules | 3 coaxial DBDs, submerged in the drum |
| Electrodes | 15 mm 316 rod (HV) in alumina 22.2 × 15.9 mm (εr 9.6), inside 1″ Sch 10 316 pipe (ground) |
| Gap / active length | 2.8 mm air annulus (Sch 5: 3.9 mm), 200 mm, 142 cm² per module |
| Drive | half-bridge (120 V bus, 5 A) into a 25 kV flyback (300 W), 12.5 kV pk, 6.5 kHz, burst control |
| Gas | 6 L/min dry air (dew point −20 °C), 2 L/min per module → venturi on the recirculation loop |
| Off-gas | stage-2 bucket (5 gal of tomorrow's source water), knock-out, MnO₂ destruct, outdoors |
| Batch | 189 L (50 gal) municipal ag water, pH 10, KH 1.5 meq/L, treated to pH 5.5, held until ozone decays |
| Delivery | tops off a 300 gal (1136 L) NFT reservoir; tap water only if the PAW stock runs out |

The NFT channel layout is the default 16 × 4 m × 26 plants: the design does not specify it.
Load the preset and set the real channel count, length and plants in **System**.

## Model (`src/sim/paw.cpp`, f64, one step per 60 s window)

**Electrical.**
- Each module is the alumina (C_d) in series with the gas gap (C_g), both coaxial
  capacitors.
- The gap burns at 3 kV/mm + 1.3 kV, the design's uniform-field estimate.
- Ignition is U_b (C_d + C_g) / C_d.
- Power is the Manley relation, 4 f C_d U_b (U₀ − U_ign).
- The result is multiplied by a measured/ideal factor (default 0.7; the design expects
  60–80 %), then capped by the flyback (300 W) and half-bridge (chain efficiency × bus V ×
  5 A).
- Burst duty sets average power.
- The UI flags power density over the submerged limit of about 1 W/cm², and supply limits.

**Gas.**
- SEI = P / air flow sets the regime: ozone mode below 0.3 kJ/L, NOx mode above 1 kJ/L,
  smooth in between.
- NOx-mode nitrogen fixation costs 22 MJ/mol (the literature range is 15–30).
- Ozone is 30 g/kWh in ozone mode and 1 g/kWh left over in the NOx effluent.
- 60 % of the fixed N leaves as N₂O₅, which makes nitrate only. The rest is NO₂, which
  makes nitrite and nitrate 1:1.

**Drum and bucket** (well mixed, f64):
- Nitrate, nitrite, dissolved O₃, DIC, untracked charge R (the source water's
  counter-ions) and temperature.
- **Capture:** venturi NOx capture is 75 % at pH ≤ 6 and 85 % at pH ≥ 8. NO₂ absorbs faster
  into alkaline water: 2 NO₂ + 2 OH⁻ → NO₂⁻ + NO₃⁻. Ozone transfer is 60 %.
- **Stage 2:** escaping NO₂ (70 %) and O₃ (30 %) go into the stage-2 bucket. Its water
  starts the next batch.
- **Liquid chemistry:**
  - Nitrite + O₃ → nitrate, with k = 3.7×10⁵ M⁻¹ s⁻¹ (Hoigné & Bader 1985), integrated
    implicitly in 1 s steps.
  - O₃ decays at k₀ + k_OH [OH⁻]. This is empirical: seconds at pH 10, about 20 min near
    neutral.
- **CO₂:** the venturi exchanges it toward air equilibrium, with kLa 2×10⁻³ s⁻¹.
- **pH:** a closed-system speciation solve from TA = R − NO₃⁻ − NO₂⁻ and DIC. It includes
  carbonate, water and HNO₂ (pKa 3.25), with the same constants and Davies activities as the
  NFT pH model.
- **Source water:** built from pH 10 and KH, and reproduces its own pH to 1e-6.
- **Heat:** 90 % of the plasma power goes into the drum water through the submerged pipes,
  plus the pump; the drum walls exchange with the room.

**Batch cycle.**
1. **Treat.** The plasma runs until pH reaches the target. It stops inside the window that
   reaches it, so there is no overshoot.
2. **Hold.** Wait at least the hold time (1.5 h), and until dissolved O₃ is below
   0.05 mg/L.
3. **Ready.** The NFT top-off draws finished PAW. Its nitrate and nitrite enter the NO₃
   field (nitrite is nitrified within hours in an aerated loop), its counter-ions enter R,
   and its temperature mixes in.
4. **Refill.** When the drum is empty, it refills from the bucket plus fresh source water,
   and the bucket refills.

**Module in the M2 kinetics** (`--scene dbd --preset paw-skid`):
- One module's annulus: 44.0 cm³ at 2 L/min (1.32 s residence), its delivered power per
  volume (1.6 W/cm³), 6.5 kHz, dry air, gas about 30 K above the water.

## Checks (`simstack_tests`)

| Check | Result |
| --- | --- |
| Manley power vs the design's table (7 points, 5–20 kHz, 12–16 kV, 152/203 mm) | within the table's rounding (≤ 5 %) |
| C_d, C_g per 152 mm; ignition Sch 10 / Sch 5 | 241 / 38 pF; 11.2 / 14.5 kV pk (design: 241 / 37 pF, 11.3 / 14.6 kV) |
| Batch to pH 5.5 at 300 W, 75 % capture, vs the design's calculator | 5.48 / 10.95 h vs 5.61 / 11.22 h (15 / 30 MJ/mol) |
| Fixed N = drum + bucket + vented | 1.5e-13 |
| Alkalinity equation at the solved pH | 1.5e-21 mol/L |
| Source water pH from KH + DIC | 10.000000 |
| Preset + NFT, 30 days: NFT ion / water books, PAW N handover | 2.7e-7 / 1.1e-14 / exact |
| Nitrite share of batch N | 9 % (design: 5–20 %) |

## What the preset shows

At the default 0.7 Manley factor:
- **Power:** the three modules deliver 216 W (0.52 W/cm², SEI 2.2 kJ/L, NOx mode).
- **Batch:** a 50 gal batch of the pH 10, KH 1.5 water reaches pH 5.5 in 8.8 h, using
  1.9 kWh of plasma energy. That's inside the design's 3–12 h.
- **Product:** 1.43 mM nitrate and 0.13 mM nitrite (9 % of N), with 5 % of the fixed N
  vented.
- **NFT reservoir:** the PAW arrives with almost no alkalinity. The plants' own uptake
  slightly acidifies the reservoir, so its pH controller now doses base (209 mmol in 30
  days) instead of the acid that tap-water top-off needs. This is the downward drift the
  design warns about.
- **Batch pace:** with the default 416-plant layout the reservoir only needs about 23 L a
  day, so one batch lasts about 8 days. A daily 50 gal top-off needs roughly 3,800 plants of
  this size.

## Calibrate first

The design marks three numbers as estimates. Each is a slider in **PAW skid**:
- **Manley factor:** set it from a Lissajous measurement, a 100 nF capacitor in the ground
  strap plus a 1000:1 probe; loop area × f is the power.
- **NOx energy cost:** set it from nitrate strips against plasma kWh per batch.
- **Venturi capture:** set it from nitrate in the drum against nitrate in the stage-2
  bucket.

## Limits

- **Breakdown estimate:** the burning voltage uses the uniform-field estimate. Coaxial
  field enhancement and humidity shift it by about ±10 %, which the Lissajous calibration
  absorbs.
- **Chemistry gap:** the NOx/O₃ split comes from SEI regimes and an energy cost, not yet from
  the M2 kinetics. The current air mechanism predicts the wrong regime at these conditions:
  2 % O₃ and about 100 ppm NOx, against a few thousand ppm NOx. Its surrogate electron
  source makes only ground-state N, O and OH. Real air DBDs make NO mainly through
  N(²D) + O₂ → NO + O and through N₂(A) and O(¹D) channels. Adding those excited states is
  the next refinement. The module preset in the dbd scene is set up to validate it.
- **Not modelled:** H₂O₂ (low in a gas-fed design, per the review), HNO₂ decomposition below
  pH 4, sodium from caustic softening, and Fe-chelate oxidation by residual ozone.
