# Test 3 — Reentry-Calibrated Bridge

Tests whether the within-cell heteroscedasticity correction term, calibrated from the catalog-scale empirical Φ distribution per cell, closes the gap between Kemeny-Snell prediction and the catalog-scale Var_emp = 33,396,043.

Decision rule (from the brief): r_corrected ≈ 1 ⇒ within-cell heteroscedasticity is the verified failure mode of the canonical specification on the catalog-scale result.

## Setup

| parameter | value |
|---|---|
| α (frozen)       | 4.27517 |
| W (D_KL window) | 5 |
| ε                 | 1e-09 |
| n restricted catalog objects | 1916 |
| surviving transient cells    | [0, 3, 5, 8, 9, 12, 13, 16, 17, 19] |

## Decomposition

| quantity | value |
|---|---:|
| Var_pred,weighted (recomputed under empirical π) | 147,991.94 |
| Var_starts(v) (across-start)                    | 84,311.53 |
| Var_pred,total                                  | 232,303.47 |
| Var_correction = Σ_c E_π[visits] · σ²_c,emp     | 1,800,755.50 |
| Var_pred,corrected = weighted + correction      | 1,948,747.44 |
| Var_pred,total + correction                     | 2,033,058.97 |
| catalog Var_emp                                 | 33,396,042.88 |

## Ratios

| ratio | value |
|---|---:|
| r_uncorrected = Var_pred,weighted / Var_emp        | 0.004431 |
| r_corrected   = (weighted + correction) / Var_emp  | 0.058353 |
| r_total_corrected = (total + correction) / Var_emp | 0.060877 |

## Verdict

**Diagnosis NOT supported on catalog-scale data.** Even with the full empirical within-cell variance correction, r_corrected is far from 1. Within-cell heteroscedasticity alone does not explain the catalog-scale result. The actual failure mode is unknown and requires further investigation. Candidate sources: within-cell autocorrelation of Φ (catalog Φ has empirical kurtosis ≈ 298, far heavier-tailed than Gaussian); non-Markovian trajectory structure (D_P10 measured at 0.033 nats, non-zero); structural mismatch between the Markov reward model and the per-trajectory σ_raw construction.

## Per-cell contribution to Var_correction

| cell | n windows | κ_c | σ²_c,emp | E_π[visits] | contribution | share |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 33,734 | 3.9035 | 2.5307 | 9.9168 | 25.10 | 0.00 % |
| 3 | 281,721 | 2.9218 | 2.0866 | 38.9628 | 81.30 | 0.00 % |
| 5 | 48,858 | 4.1646 | 0.7245 | 4.6914 | 3.40 | 0.00 % |
| 8 | 1,250 | 5.5006 | 1.3370 | 0.3650 | 0.49 | 0.00 % |
| 9 | 308,226 | 3.9591 | 0.4595 | 9.9837 | 4.59 | 0.00 % |
| 12 | 2,907 | 4.7527 | 1.3862 | 0.9874 | 1.37 | 0.00 % |
| 13 | 41 | 4231.3808 | 287798249.1570 | 0.0037 | 1,051,454.98 | 58.39 % |
| 16 | 178 | 551.3335 | 19550255.3807 | 0.0383 | 748,978.44 | 41.59 % |
| 17 | 566,927 | 3.6400 | 7.0340 | 24.0694 | 169.31 | 0.01 % |
| 19 | 5,801 | 4.1072 | 5.1110 | 7.1484 | 36.54 | 0.00 % |

## Caveats (scientific-rigor)

- The correction term assumes per-visit Φ values are conditionally independent given the cell sequence. The empirical Φ within a cell may be autocorrelated across consecutive visits (a trajectory crossing cell c twice may have similar Φ both times because of slow underlying physics). The linear correction over-estimates Var_correction in that case.
- σ²_c,empirical pools across all (trajectory, window) Φ values in cell c. It does not condition on the trajectory.
- The catalog Φ distribution has heavy tails (excess kurtosis ≈ 298 from the catalog-scale run); the Gaussian-moment assumption underlying the correction (Test 2 untruncated mode) is approximate. Test 3's r_corrected reflects whether the linear correction is sufficient for this data, not whether the underlying distribution is well-behaved.
