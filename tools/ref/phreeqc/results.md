# Sim water chemistry vs PHREEQC 3.8.6 (phreeqc.dat)

## Source water speciation (closed, at the skid source temperature)

| | design: sim | design: PHREEQC | W78: sim | W78: PHREEQC |
| --- | --- | --- | --- | --- |
| DIC, mmol/L | 1.143 | 1.140 | 1.522 | 1.516 |
| CO₂*, mmol/L | 0.00019 | 0.00018 | 0.086 | 0.084 |
| log pCO₂ | -5.34 | -5.35 | -2.68 | -2.69 |
| ionic strength, mmol/L | 1.35 | 1.92 | 1.48 | 2.82 |
| calcite SI | — | -1000.00 | — | -0.90 |
| amorphous silica SI | — | — | — | -0.45 |
| charge balance error, % | — | 0.00 | — | -0.09 |

## Batch: closed-system acid N against pH (85 % nitric, 15 % nitrous)

| acid N, mmol/L | design: sim | design: PHREEQC | W78: sim | W78: PHREEQC |
| --- | --- | --- | --- | --- |
| 0.0 | 10.000 | 10.000 | 7.600 | 7.600 |
| 0.3 | 9.345 | 9.361 | 6.849 | 6.848 |
| 0.6 | 7.095 | 7.105 | 6.465 | 6.462 |
| 0.9 | 6.512 | 6.517 | 6.115 | 6.112 |
| 1.2 | 6.040 | 6.045 | 5.651 | 5.648 |
| 1.3 | 5.849 | 5.854 | 5.396 | 5.392 |
| 1.4 | 5.599 | 5.604 | 4.950 | 4.947 |
| 1.5 | 5.189 | 5.194 | 4.295 | 4.292 |
| 1.8 | 3.777 | 3.769 | 3.587 | 3.586 |
| 2.1 | 3.415 | 3.408 | 3.316 | 3.317 |
| 2.4 | 3.214 | 3.209 | 3.148 | 3.149 |
| 2.7 | 3.076 | 3.071 | 3.025 | 3.027 |
| 3.0 | 2.970 | 2.965 | 2.929 | 2.931 |

Acid N to reach pH 5.5: design sim 1.430 vs PHREEQC 1.425 mmol/L; W78 sim 1.264 vs PHREEQC 1.258 mmol/L. Worst pH difference along the curves: design 0.057, W78 0.003.

## NFT working solution, open to 420 ppm CO₂

| case | pH: sim | pH: PHREEQC | HNO₃ to pH 5.8 (mmol/L): sim | PHREEQC | ionic strength: sim | PHREEQC | free Ca²⁺ / total | hydroxyapatite SI | gypsum SI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nft_w78_20 | 7.533 | 7.528 | 1.306 | 1.303 | 26.1 | 24.3 | 0.89 | 9.18 | -1.67 |
| nft_w78_30 | 7.528 | 7.519 | 1.292 | 1.287 | 26.1 | 24.0 | 0.88 | 9.89 | -1.70 |
| nft_design_20 | 7.673 | 7.671 | 1.473 | 1.466 | 23.8 | 21.6 | 0.88 | 9.76 | -1.68 |

