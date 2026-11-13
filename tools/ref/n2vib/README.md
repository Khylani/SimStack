# N2(v) state-to-state reference

**Question:** does vibrationally excited nitrogen (N₂(v) + O → NO + N) add to the PAW skid's
NOx? Neither ChemPlasKin nor the real-time model has this chemistry, and LoKI-GM ships an
N₂-only ladder (and needs MATLAB R2024b).

**Answer: no.** Below 0.3 % in every case, under all three Zeldovich models in the
literature, which differ by 100×.

The reason is that the model's vibrational temperature, 730–780 K in dry air, reproduces the
CARS measurement in a 1 atm air DBD (~800 K, Baeva et al. 1999). At that temperature the
levels the reaction needs (v ≥ 13) are essentially empty.

```sh
tools/ref/n2vib/build.sh            # build/n2vib_ref (SimStack kinetics with bigger tables)
tools/ref/n2vib/check.sh            # detailed balance + T_v vs CARS + vibrational NO share, 13 runs, ~10 s
build/n2vib_ref T_K x_H2O E_f_mJ_cm3 P_W_cm3 residence_s EN_Td [novib|step13|step11|noair|check]
```

## Model

SimStack's own mechanism and discharge source run in the same Rosenbrock solver. On top of
them, the ladder N₂(X, v = 1–40) is written as mass-action reactions: 57 species, 1965
reactions. "N2" is v = 0.

| Process | Expression | Source |
| --- | --- | --- |
| Levels | Dunham: ωe 2358.57, ωexe 14.324, ωeye −0.00226 cm⁻¹ | Huber–Herzberg / Lofthus & Krupenie, JPCRD 6, 113 (1977) |
| e-V | per filament, G(0→Δv), Δv = 1–10, from LoKI-B++ (`ev_gvalues.csv`); from v > 0: G/(1 + 0.15 v) | LoKI-B++ IST-Lisbon; Bourdon & Vervisch, JTHT 14, 489 (2000) |
| V-V N₂–N₂ | SSH fitted to Billing, as LoKI `nitrogenMolecularVV.m`; k₁₀⁰¹ = 1.0e-14 cm³/s at 330 K; detailed balance | Alves et al., PSST 21, 045008 (2012); Guerra et al., PSST 28, 073001 (2019) |
| V-T by N₂ | Millikan–White (the LoKI fit diverges below ~330 K) | Millikan & White, JCP 39, 3209 (1963) |
| V-T by O₂ | pτ = exp(162(T^−⅓ − 0.03) − 18.42) | Colgan & Levitt, Trans. Faraday Soc. 63, 2898 (1967) |
| V-T by O | 3.3e-15 exp(−1235(1/T − 1/300)), measured 300–740 K; k_v = v k₁₀ | McNeal, Whitson & Cook, JGR 79, 1527 (1974) |
| V-T/V-V by H₂O | pτ = exp(21 T^−⅓ − 1.35) µs·atm: 8.3e-15 at 330 K, 1.3e-14 at 415 K | Center & Newton, JCP 68, 3327 (1978). Whitson & McNeal 1977 measured 1.2e-14 at 293 K and ISO 9613-1 / Griffin 1981 give 5–7e-15 (within ~2×) |
| Dry atmospheric air as a whole (N₂–O₂ V-V′, CO₂ near-resonance) | ISO 9613-1: f_r = 9 (T/293)^−½ Hz·atm⁻¹, 1/τ = 2π f_r = 57 s⁻¹ | ISO 9613-1:1993; Bass et al. |
| Zeldovich | **FM:** Fridman–Macheret α = 1, A set so a Boltzmann ladder at 3000 K gives 3.0e-10 exp(−38250/T) (Baulch et al., JPCRD 23, 847 (1994)). **step13:** 1e-13 for v ≥ 13 (Guerra & Loureiro 1995, LoKI-C). **step11:** 1e-11 for v ≥ 13 (Capitelli / Gordiets). | spread ×100 |

The literature survey (sources checked, uncertainties, which processes dominate) is
summarised in the table. The main uncertainties:
- **Zeldovich step:** ×100.
- **O-atom V-T above v = 1:** ×3–10.
- **k₁₀⁰¹ near room temperature:** ×2–3. A 2015 stimulated-Raman measurement disagrees with
  the T^1.5 law.

## Results

Skid module: dry air, 180 Td, 30 mJ/cm³ filaments, 2.16 J/cm³ over 1.32 s.

| Gas, water | T_v(0,1) | Vibrational share of fixed N (FM / step13 / step11) |
| --- | --- | --- |
| 330 K, dry | 729 K | −0.26 % (all three) |
| 330 K, 1000 ppm (−20 °C frost point) | 604 K | −0.01 % |
| 413 K, dry | 781 K | −0.41 % |
| 413 K, 1000 ppm | 627 K | −0.03 % |
| 330 K, dry, **without** the dry-air relaxation channel | 2090 K | +53 % (FM), +57 % (step11) |

The small negative shares are bookkeeping, not chemistry. The electron-impact source acts
on N₂(v = 0), and 1–2 % of N₂ sits in v ≥ 1.

The last row shows that the conclusion rests on the dry-air relaxation channel, and the
measurement is what settles it:
- **With it:** T_v ≈ 730–780 K, matching the ~800 K CARS value in a 1 atm air DBD (Baeva et
  al., PCPP 19, 445 (1999)).
- **Without it:** T_v ≈ 2090 K, 2.6× too high.

Humid air relaxing faster agrees with Teramoto & Ono, JAP 116, 073302 (2014): CARS in a 1 atm
air corona.

No published measurement shows a vibrational NOx contribution in a near-room-temperature
1 atm DBD. Qiao et al., PSST 34, 065008 (2025) found under 1 % of vibrational energy going
into NO even at T_v ≈ 7000 K.

## Consequence for the real-time model

No N₂(v) ladder is needed. The ~20 % of the electron power that goes into N₂ vibration
relaxes to heat within about 5–15 ms (water plus dry-air channels). The module heat balance
already counts it, as 90 % of the discharge power ending up as heat.

This would change if the feed were CO₂-free, very dry, and the gas hot. In that case run
this reference again.
