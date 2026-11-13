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

**Gas source.** There are two choices for what leaves the modules:
- **Design estimate** (default): SEI regimes plus an energy cost.
- **M2 air kinetics:** one module's annulus as plug flow, at its power density, residence
  time and frequency, with dry feed and gas `gas_dT_K` (30 K) above the drum water.
  - It runs at init, on preset load and on **Run chemistry**. The UI flags it as stale when
    the drive or flow changes.
  - In the water, NO₂ gives nitrite and nitrate 1:1; N₂O₅, HNO₃ and NO₃ give nitrate;
    HNO₂ gives nitrite.
  - NO pairs with NO₂ as N₂O₃ (NO + NO₂ + 2 OH⁻ → 2 NO₂⁻ + H₂O) and goes to nitrite.
  - Unpaired NO and all N₂O are insoluble and go to the vent. N₂O is reported because it's a
    greenhouse gas.

**Module heat balance.** The rod and the alumina have no cooling path of their own, so all
the discharge heat (90 % of the power) has to cross the 2.8 mm air gap to the water-cooled
pipe.
- Heat leaves by conduction (coaxial, k(T) of air) and by gray-body radiation (alumina ε
  0.7, oxidized 316 ε 0.4).
- The pipe sits behind a 500 W/m²K water film.
- The gas sees the mean of the two walls.
- The kinetics source uses this gas temperature, unless **manual gap temperature** is set.

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
| Nitrite share of batch N | 15 % (design: 5–20 %) |
| Module heat balance: conduction + radiation vs discharge heat | 1.4e-12 |
| Kinetics-sourced batch: N books (with NO / N₂O vented) | 7e-14 |

## What the preset shows

At the default 0.7 Manley factor:
- **Power:** the three modules deliver 216 W (0.52 W/cm², SEI 2.2 kJ/L, NOx mode).
- **Batch:** a 50 gal batch of the pH 10, KH 1.5 water reaches pH 5.5 in 8.8 h, using
  1.9 kWh of plasma energy. That's inside the design's 3–12 h.
- **Product:** 1.33 mM nitrate and 0.23 mM nitrite (15 % of N), with 5 % of the fixed N
  vented.
- **NFT reservoir:** the PAW arrives with almost no alkalinity. The plants' own uptake
  slightly acidifies the reservoir, so its pH controller now doses base (209 mmol in 30
  days) instead of the acid that tap-water top-off needs. This is the downward drift the
  design warns about.
- **Batch pace:** with the default 416-plant layout the reservoir only needs about 23 L a
  day, so one batch lasts about 8 days. A daily 50 gal top-off needs roughly 3,800 plants of
  this size.

## What the thermal and chemistry models say

**The alumina runs hot.** At 72 W per module, the heat balance gives:
- **Pipe outside:** 26 °C. It feels barely warm, as the design's start-up check expects.
- **Alumina:** 257 °C.
- **Gap gas:** 415 K.

The Viton radial glands sit on the alumina in the caps, right at the ends of the discharge,
so they see close to that temperature. Viton is rated about 200 °C continuous, and the UI
flags it. Two ways to bring the alumina down:
- Run at half burst duty: the alumina falls to about 163 °C, at half the production rate.
- Give the rod a cooling path, for example a hollow rod with water through it, or a
  conductive path out of the bore to a sink above the cap.

**The gas chemistry at that temperature (superseded below).** The first M2 run, with the
averaged source, ran one module at 1.64 W/cm³, 413 K, 1.32 s, 2.2 kJ/L:
- **Ozone collapses to 38 ppm** (the "ozone zero" of hot DBDs). At a cooled 321 K it would be
  2.1 %, ozonizer behaviour.
- **NO survives instead of being oxidized:** 220 ppm NO with 170 ppm NO₂, 170 ppm HNO₃,
  50 ppm HNO₂, and 720 ppm N₂O, which is vented.
- **Soluble nitrogen costs 92 MJ/mol**, counting NO paired with NO₂ as N₂O₃. The design
  assumes 22, and the cooled case gave 62.
- **A batch takes about 42 h** instead of 8.8 h.

This is a lower bound on the hot-gap NOx. The mechanism has no vibrationally excited N₂, and
N₂(v) + O → NO + N is the main reason hot air discharges reach 15–30 MJ/mol in the
literature. Its yield depends on the vibrational distribution inside microsecond filaments,
which needs a state-resolved solver: the planned ChemPlasKin / LoKI-B reference, not a tuned
lumped model. Until then the design estimate stays the default gas source. The kinetics show
the cool-gap and no-N₂(v) limits.

**With the validated source** (LoKI-B++ G-values at 180 Td, 30 mJ/cm³ filaments, Kossyi
N + NO₂; see `tools/ref/README.md`):
- **SimStack kinetics:** at 413 K the skid sits right at the ozone-to-NOx switch. The
  kinetics source gives about 1500 ppm O₃ and soluble nitrogen at **28 MJ/mol**, for a
  **12.9 h** batch.
- **ChemPlasKin** (full electron, ion and excited-state kinetics) at the same point: ozone
  fully gone, 520 ppm NO with 470 ppm NO₂, **73 MJ/mol**.
- **The two models place the switch differently.** The reduced model collapses the same
  way at E_f ≤ 10 mJ/cm³ (74 MJ/mol) but sits half in ozone mode at 30.
- **Best current estimate: 30–75 MJ/mol, ozone near zero.** That's 12–30 h for a pH 10, KH
  1.5 batch at 216 W, against the design's 22 MJ/mol and 8.8 h.
- **Vibrational N₂ doesn't change this.** A state-to-state reference (`tools/ref/n2vib`)
  shows that the water and CO₂ in the feed hold T_v at about 600–780 K, matching CARS in
  1 atm air DBDs, which keeps the vibrational NO path below 0.3 %.
- The default gas source stays the design estimate until a measurement anchors it.

**What settles it in the first runs:**
- Ozone and NO₂ at the drum vent (ozone means a cool gap; NO₂ with no ozone means NOx mode).
- Nitrate per kWh of plasma energy.
- An infrared spot reading of the alumina at the top cap, through the creepage section
  (it's bare ceramic).

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
- **Chemistry:** the M2 kinetics include the N(²D), N₂(A) and O(¹D) channels and the gap
  temperature, but not N₂(v). In the hot gap they underestimate NOx (see above).
- **Thermal model:** steady state and one-dimensional across the gap. It ignores axial
  conduction along the alumina and rod toward the caps (which would cool the glands
  somewhat), filament hot spots, and the alumina's own dielectric loss.
- **Not modelled:** H₂O₂ (low in a gas-fed design, per the review), HNO₂ decomposition below
  pH 4, sodium from caustic softening, and Fe-chelate oxidation by residual ozone.
