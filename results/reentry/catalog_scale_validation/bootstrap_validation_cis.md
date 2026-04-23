# STTS-Reentry — Validation-Set Bootstrap (three-pass decay-epoch sensitivity)

Percentile bootstrap over per-object aggregate statistics, B = 1000 replicates, seed 20260422. Three passes:

- **Full**: all rows as committed.
- **Excluded**: drop the 12 known catalog-error NORADs.
- **Corrected**: substitute GCAT DDate for the 7 CONFIRMED year-off-by-one objects; recompute lead times on the corrected reference epoch. Drop the 5 LIKELY-error objects whose true date falls outside the local TLE window.

## 78-object reference (committed aggregate)

**median max P**

| pass | n | point | 95% CI |
|---|---:|---:|---:|
| full | 78 | 0.904700 | [0.904100, 0.905400] |
| excluded | 77 | 0.904700 | [0.904197, 0.905400] |
| corrected | 78 | 0.904700 | [0.904100, 0.905400] |

_Delta corrected − full = +0.0000; full CI half-width = 0.0006; load-bearing on labels = **no**_

**mean max P**

| pass | n | point | 95% CI |
|---|---:|---:|---:|
| full | 78 | 0.887626 | [0.861805, 0.904914] |
| excluded | 77 | 0.896073 | [0.878655, 0.905059] |
| corrected | 78 | 0.887626 | [0.861805, 0.904914] |

_Delta corrected − full = +0.0000; full CI half-width = 0.0216; load-bearing on labels = **no**_

**rate max P < 0.5**

| pass | n | point | 95% CI |
|---|---:|---:|---:|
| full | 78 | 0.025641 | [0.000000, 0.064103] |
| excluded | 77 | 0.012987 | [0.000000, 0.038961] |
| corrected | 78 | 0.025641 | [0.000000, 0.064103] |

_Delta corrected − full = +0.0000; full CI half-width = 0.0321; load-bearing on labels = **no**_

**rate max P < 0.25 (bimodal-outlier)**

| pass | n | point | 95% CI |
|---|---:|---:|---:|
| full | 78 | 0.025641 | [0.000000, 0.064103] |
| excluded | 77 | 0.012987 | [0.000000, 0.038961] |
| corrected | 78 | 0.025641 | [0.000000, 0.064103] |

_Delta corrected − full = +0.0000; full CI half-width = 0.0321; load-bearing on labels = **no**_

**median lead @ P≥0.10 (days)**

| pass | n | point | 95% CI |
|---|---:|---:|---:|
| full | 78 | 446.52 | [393.42, 478.16] |
| excluded | 77 | 442.88 | [392.88, 468.48] |
| corrected | 78 | 446.52 | [393.42, 478.16] |

_Delta corrected − full = +0.0000; full CI half-width = 42.3715; load-bearing on labels = **no**_

**median lead @ P≥0.25 (days)**

| pass | n | point | 95% CI |
|---|---:|---:|---:|
| full | 76 | 308.62 | [233.59, 422.00] |
| excluded | 76 | 308.62 | [233.59, 422.00] |
| corrected | 76 | 308.62 | [233.59, 422.00] |

_Delta corrected − full = +0.0000; full CI half-width = 94.2056; load-bearing on labels = **no**_

**median lead @ P≥0.50 (days)**

| pass | n | point | 95% CI |
|---|---:|---:|---:|
| full | 76 | 209.50 | [98.54, 265.50] |
| excluded | 76 | 209.50 | [98.54, 265.50] |
| corrected | 76 | 209.50 | [98.54, 265.50] |

_Delta corrected − full = +0.0000; full CI half-width = 83.4801; load-bearing on labels = **no**_

**median lead @ P≥0.75 (days)**

| pass | n | point | 95% CI |
|---|---:|---:|---:|
| full | 76 | 91.00 | [74.00, 131.57] |
| excluded | 76 | 91.00 | [74.00, 131.57] |
| corrected | 76 | 91.00 | [74.00, 131.57] |

_Delta corrected − full = +0.0000; full CI half-width = 28.7850; load-bearing on labels = **no**_

## Catalog-scale stratified sample (n=3,424)

**median max P**

| pass | n | point | 95% CI |
|---|---:|---:|---:|
| full | 3424 | 0.903900 | [0.903898, 0.904000] |
| excluded | 3412 | 0.903900 | [0.903900, 0.904000] |
| corrected | 3419 | 0.903900 | [0.903900, 0.904000] |

_Delta corrected − full = +0.0000; full CI half-width = 0.0001; load-bearing on labels = **no**_

**mean max P**

| pass | n | point | 95% CI |
|---|---:|---:|---:|
| full | 3424 | 0.846124 | [0.840742, 0.851703] |
| excluded | 3412 | 0.848438 | [0.842957, 0.853769] |
| corrected | 3419 | 0.847187 | [0.841997, 0.852382] |

_Delta corrected − full = +0.0011; full CI half-width = 0.0055; load-bearing on labels = **no**_

**rate max P < 0.5**

| pass | n | point | 95% CI |
|---|---:|---:|---:|
| full | 3424 | 0.064544 | [0.056075, 0.072729] |
| excluded | 3412 | 0.061254 | [0.053334, 0.069461] |
| corrected | 3419 | 0.063176 | [0.055572, 0.071366] |

_Delta corrected − full = -0.0014; full CI half-width = 0.0083; load-bearing on labels = **no**_

**rate max P < 0.25 (bimodal-outlier)**

| pass | n | point | 95% CI |
|---|---:|---:|---:|
| full | 3424 | 0.034463 | [0.028621, 0.040603] |
| excluded | 3412 | 0.031067 | [0.024912, 0.036936] |
| corrected | 3419 | 0.033051 | [0.027493, 0.039193] |

_Delta corrected − full = -0.0014; full CI half-width = 0.0060; load-bearing on labels = **no**_

**median lead @ P≥0.10 (days)**

| pass | n | point | 95% CI |
|---|---:|---:|---:|
| full | 3377 | 567.68 | [527.21, 611.59] |
| excluded | 3368 | 565.78 | [525.04, 611.25] |
| corrected | 3375 | 568.20 | [525.07, 611.59] |

_Delta corrected − full = +0.5175; full CI half-width = 42.1912; load-bearing on labels = **no**_

**median lead @ P≥0.25 (days)**

| pass | n | point | 95% CI |
|---|---:|---:|---:|
| full | 3306 | 205.47 | [193.03, 215.44] |
| excluded | 3306 | 205.47 | [193.03, 215.44] |
| corrected | 3306 | 205.47 | [193.03, 215.44] |

_Delta corrected − full = +0.0000; full CI half-width = 11.2089; load-bearing on labels = **no**_

**median lead @ P≥0.50 (days)**

| pass | n | point | 95% CI |
|---|---:|---:|---:|
| full | 3203 | 76.50 | [74.19, 79.40] |
| excluded | 3203 | 76.50 | [74.19, 79.40] |
| corrected | 3203 | 76.50 | [74.19, 79.40] |

_Delta corrected − full = +0.0000; full CI half-width = 2.6056; load-bearing on labels = **no**_

**median lead @ P≥0.75 (days)**

| pass | n | point | 95% CI |
|---|---:|---:|---:|
| full | 3074 | 40.26 | [38.58, 41.96] |
| excluded | 3074 | 40.26 | [38.58, 41.96] |
| corrected | 3074 | 40.26 | [38.58, 41.96] |

_Delta corrected − full = +0.0000; full CI half-width = 1.6909; load-bearing on labels = **no**_

