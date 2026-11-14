# Source water: Moloaʻa, Kauaʻi

The skid's batches and the NFT's top-off use the farm's own well water. This page collects
the public data on that water and where it comes from. The raw records are in
`data/water/moloaa_area_usgs.csv` (USGS / Water Quality Portal, six sites).

## Who supplies water on Koʻolau Road

There is no county (Department of Water) main through Moloaʻa. Water comes from deep wells
in the basalt aquifer, 380–700 ft:

| Source | Status | Notes |
| --- | --- | --- |
| State **Moloaʻa Well No. 1** (2-1020-02, W79, 586 ft, 1962) | Ran as a regulated public water system from 1 Oct 2015 (DOH No. 435; EPA HI0000436, now inactive); the permit holder stopped operating it in Oct 2024 | Supplied the Moloaʻa Hui lands, the Moloaʻa Irrigation Cooperative (MIC) and a three-home DOW system |
| **Moloaʻa Irrigation Cooperative** (EPA HI0000437) | Active, groundwater, ~100 people; its own well online since 26 Sep 2024 | Annual water-quality report goes to customers |
| **Koʻolau Ranch** (EPA HI0000438) | Active, groundwater, ~60 people; wells "KR1 Well 1 / 2" | Probably state wells 2-1221-08/09 ("Koolau 1 / 2", 485 / 440 ft) west of Moloaʻa |
| State **Moloaʻa 2** (2-1020-03, W78, 700 ft, 1963) | USGS-sampled 1972–1991 | 100 m from Well No. 1; **the only full analysis in Moloaʻa** |

Sources: Kauaʻi DOW water quality reports (kauaiwater.org/water-quality), EPA SDWIS
(data.epa.gov/efservice), BLNR submittal on PWS No. 435 (dlnr.hawaii.gov D-7TF, 2023),
Environment Hawaiʻi "Moloaʻa Well Update" (Nov 2024), USGS Water Quality Portal.

## Chemistry

Moloaʻa 2 (W78) is in the same aquifer as Well No. 1 and close to the Koʻolau wells. It is
the best public estimate of the farm's water until the supplier's report or our own test
replaces it.

| | Moloaʻa 2 (W78) | Anahola W-A (DOW) | DOW Anahola 2025 report | DOW Kīlauea 2025 report |
| --- | --- | --- | --- | --- |
| pH | **7.6** | 7.3 | 7.8–8.3 | 7.9–8.3 |
| Alkalinity, mg/L as CaCO₃ | **72** (1.44 meq/L) | 71 | — | — |
| Bicarbonate, mg/L | 88 | 87 | — | — |
| Free CO₂, mg/L | 3.5 | 7 | — | — |
| Hardness, mg/L as CaCO₃ | 73 | 70 | 40–50 | 65–75 |
| Calcium, mg/L | 11 | 10 | — | — |
| Magnesium, mg/L | 11 | 11 | — | — |
| Sodium, mg/L | 14 | 26 | 19 | 11 |
| Potassium, mg/L | 0.7 | 1.6 | — | — |
| Chloride, mg/L | 20 (15–44, n = 57) | 21.5 | — | — |
| Sulfate, mg/L | 3.4 | 20 | — | — |
| Silica (SiO₂), mg/L | 36 | 28 | — | — |
| Nitrate, mg/L as N | 0 | 0.2 | 0.3 | — |
| Fluoride, mg/L | 0.2 | 0.2 | — | — |
| Specific conductance, µS/cm | 210 (190–360, n = 58) | 240 | — | — |
| TDS, mg/L | 139 | 167 | — | — |
| Temperature, °C | 23 (21.5–25) | 24 | — | — |

Single full analyses are from the 1970s–80s; chloride and conductance were sampled
repeatedly to 1991 and are stable. Surface water (Anahola Stream and ditches) is far
softer: alkalinity 8–25 mg/L, pH ~7.

## What it means for the sim

- **Alkalinity matches the design.** 1.44 meq/L against the design's KH 1.5, and the
  acid a batch needs to reach pH 5.5 is set by alkalinity, so the batch's nitrogen demand
  barely changes.
- **pH does not.** No local source is near the design's pH 10; the aquifer gives 7.3–7.6
  and the county's chlorinated systems 7.8–8.3. At pH 7.6 the same alkalinity carries
  about twice the dissolved CO₂, ozone in the drum decays far more slowly (minutes rather
  than seconds, so the hold before dosing gets longer), and NO₂ dissolves less readily.
  If pH 10 came from a measurement (for example after caustic treatment or a concrete
  tank), keep it; otherwise W78 is the better default.
- **Sodium and chloride build up.** Plants take up little Na. With 14 mg/L Na and 20 mg/L
  Cl in every litre of top-off, a recirculating reservoir concentrates them until a
  solution change. The NFT model now books them (tank level) and adds them to EC.
- **Silica 36 mg/L** is beneficial for many crops and is booked as a passive solute.

**Source water** in the PAW and NFT panels picks between the design figure (pH 10, KH 1.5,
other ions unknown), Moloaʻa 2 (W78), and Custom.

## To replace these numbers

1. Ask the supplier (MIC or Koʻolau Ranch) for this year's water quality report and any
   raw-water analysis from DOH new-source monitoring.
2. Test a tap sample: pH right at the tap, and alkalinity (a titration kit or lab). Those
   two set the batch chemistry; a lab panel (Ca, Mg, Na, K, Cl, SO₄, SiO₂, NO₃) sets the
   reservoir build-up.
