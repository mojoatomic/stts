# Within-Cell Autocorrelation Characterization (catalog-scale Φ)

Pre-registered characterization on the catalog-scale per-window Φ data (commit `fdd1ba3`). Determines whether AR(1) within-cell autocorrelation, applied to the variance correction term, closes the gap to catalog-scale Var_emp = 33,396,043.

Decision rule (pre-registered):

- **FULL**: r_AR1 ∈ [0.5, 1.5] AND 95 % CI in same band → AR(1) closed-form prediction is feasible (Fix A extended).
- **PARTIAL**: r_AR1 ∈ [0.2, 0.5] ∪ [1.5, 4.0] AND CI in same band → partial recovery; need additional terms or pivot.
- **OUTSIDE**: outside [0.2, 4.0] → AR(1) does not capture; pivot to bootstrap/simulation.
- **INCONCLUSIVE**: CI spans two bands.

## Headline result

| quantity | value |
|---|---:|
| Kemeny baseline (catalog π)            | 147,991.94 |
| Var_AR1_term = Σ_c E[v]·σ²·(1+ρ)/(1−ρ) | 2,359,916.78 |
| Var_pred,full_AR1 = sum                | 2,507,908.72 |
| catalog Var_emp                        | 33,396,042.88 |
| **r_AR1**                              | **0.075096** |
| **95 % bootstrap CI on r_AR1**         | **[0.004677, 0.182566]** |
| linear correction term (Test 3)        | 1,800,755.50 |
| AR(1)/linear ratio                     | 1.31× |
| sparse-cell share of Var_AR1           | 99.85 % |
| sparse-cell dominated?                 | True |

## Decision

Point r_AR1 = **0.075096** → band **OUTSIDE**.

95 % CI: [0.004677, 0.182566] → low band OUTSIDE, high band OUTSIDE.

**Decision: OUTSIDE.**

AR(1) correction does not capture the variance source. Path forward: pivot to bootstrap/simulation; abandon closed-form variance prediction.

## Sparse-cell artifact triggered

More than 99 % of Var_AR1_term comes from cells with effective sample size < 1000. The decision rule is sparse-cell-dominated and does not directly apply.

Bulk-only re-run (cells with effective n ≥ 1000):

- Var_AR1_term (bulk only): 3,488.45
- r_AR1 (bulk only):        0.004536

## Per-cell ρ_c^(1) and contribution to Var_AR1

| cell | n qual. windows | σ²_c | ρ^(1) | 95 % CI | (1+ρ)/(1−ρ) | E[visits] | contribution | share |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 31,755 | 2.5307 | 0.9312 | [0.9280, 0.9338] | 28.08 | 9.9168 | 704.63 | 0.03 % |
| 3 | 277,493 | 2.0866 | 0.8971 | [0.8933, 0.9008] | 18.43 | 38.9628 | 1,498.26 | 0.06 % |
| 5 | 47,410 | 0.7245 | 0.4483 | [0.4351, 0.4624] | 2.63 | 4.6914 | 8.92 | 0.00 % |
| 8 | 1,149 | 1.3370 | 0.8526 | [0.8341, 0.8697] | 12.57 | 0.3650 | 6.13 | 0.00 % |
| 9 | 303,267 | 0.4595 | 0.5412 | [0.5344, 0.5486] | 3.36 | 9.9837 | 15.41 | 0.00 % |
| 12 | 2,674 | 1.3862 | 0.8252 | [0.8027, 0.8459] | 10.44 | 0.9874 | 14.29 | 0.00 % |
| 13 | 39 | 287798249.1570 | 0.2097 | [0.0476, 0.8279] | 1.53 | 0.0037 | 1,609,542.36 | 68.20 % |
| 16 | 162 | 19550255.3807 | -0.0014 | [-0.0185, 0.9964] | 1.00 | 0.0383 | 746,885.97 | 31.65 % |
| 17 | 559,949 | 7.0340 | 0.2942 | [0.2265, 0.8070] | 1.83 | 24.0694 | 310.45 | 0.01 % |
| 19 | 5,265 | 5.1110 | 0.9244 | [0.9052, 0.9414] | 25.46 | 7.1484 | 930.35 | 0.04 % |

## ρ^(k) for k = 1, 2, 3

| cell | ρ^(1) | ρ^(2) | ρ^(3) | n_pairs (lag 1) |
|---:|---:|---:|---:|---:|
| 0 | 0.9312 | 0.6814 | -0.5857 | 28,791 |
| 3 | 0.8971 | 0.5750 | -0.1362 | 262,013 |
| 5 | 0.4483 | -0.4234 | -0.6885 | 43,138 |
| 8 | 0.8526 | 0.6008 | 0.3619 | 979 |
| 9 | 0.5412 | -0.4159 | -0.7165 | 286,821 |
| 12 | 0.8252 | 0.2529 | -0.4023 | 2,329 |
| 13 | 0.2097 | 0.7565 | 0.1911 | 35 |
| 16 | -0.0014 | 0.1670 | 0.1780 | 147 |
| 17 | 0.2942 | 0.1037 | -0.0835 | 534,647 |
| 19 | 0.9244 | 0.7088 | 0.1797 | 4,760 |

## Stationarity (early/middle/late thirds, lag 1)

| cell | ρ_early | ρ_middle | ρ_late | non-stationary? | reason |
|---:|---:|---:|---:|---|---|
| 0 | 0.9323 | 0.9322 | 0.9227 | True | |ρ_late − ρ_early| = 0.0096 exceeds CI half-width (early 0.0026, late 0.0067) |
| 3 | 0.9080 | 0.8964 | 0.8735 | True | |ρ_late − ρ_early| = 0.0345 exceeds CI half-width (early 0.0045, late 0.0051) |
| 5 | 0.4862 | 0.4409 | 0.4387 | True | |ρ_late − ρ_early| = 0.0475 exceeds CI half-width (early 0.0367, late 0.0156) |
| 8 | 0.8599 | 0.8176 | 0.8624 | False | stationary |
| 9 | 0.5222 | 0.5204 | 0.5665 | True | |ρ_late − ρ_early| = 0.0442 exceeds CI half-width (early 0.0100, late 0.0073) |
| 12 | 0.8208 | 0.8255 | 0.7976 | False | stationary |
| 13 | 0.5877 | 0.1577 | 0.1483 | True | monotonic decreasing trend, |Δ|=0.4393 > 0.1 |
| 16 | -0.0159 | 0.6830 | 0.9541 | True | monotonic increasing trend, |Δ|=0.9700 > 0.1 |
| 17 | 0.2254 | 0.7882 | 0.8254 | True | monotonic increasing trend, |Δ|=0.6001 > 0.1 |
| 19 | 0.9032 | 0.9379 | 0.9174 | False | stationary |

## Sensitivity — minimum run length grid

Pre-registered grid: 3 (primary), 5, 10. Stability across min-run-length suggests the AR(1) estimate is robust to short-run noise.

| cell | ρ^(1) at L≥3 | ρ^(1) at L≥5 | ρ^(1) at L≥10 |
|---:|---:|---:|---:|
| 0 | 0.9312 | 0.9236 | 0.9181 |
| 3 | 0.8971 | 0.8914 | 0.8945 |
| 5 | 0.4483 | 0.4222 | 0.4286 |
| 8 | 0.8526 | 0.8420 | 0.8307 |
| 9 | 0.5412 | 0.5191 | 0.5027 |
| 12 | 0.8252 | 0.7996 | 0.7317 |
| 13 | 0.2097 | 0.2097 | 0.1962 |
| 16 | -0.0014 | -0.0021 | -0.0000 |
| 17 | 0.2942 | 0.2813 | 0.2601 |
| 19 | 0.9244 | 0.8710 | 0.8482 |

## Sensitivity — heavy-tail trim (top 1 % of Φ per cell)

| cell | excess kurt | ρ_full | ρ_trimmed | Δρ |
|---:|---:|---:|---:|---:|
| 0 | 1.19 | 0.9312 | 0.9279 | -0.0033 |
| 3 | 7.71 | 0.8971 | 0.8688 | -0.0282 |
| 5 | 60.58 | 0.4483 | 0.3942 | -0.0541 |
| 8 | 9.10 | 0.8526 | 0.8448 | -0.0078 |
| 9 | 12.48 | 0.5412 | 0.4401 | -0.1011 |
| 12 | 4.53 | 0.8252 | 0.8110 | -0.0142 |
| 13 | 30.76 | 0.2097 | 0.2097 | 0.0000 |
| 16 | 96.43 | -0.0014 | 0.2932 | 0.2946 |
| 17 | 448511.84 | 0.2942 | 0.7432 | 0.4490 |
| 19 | 2.12 | 0.9244 | 0.9225 | -0.0019 |

## Caveats (scientific-rigor)

- AR(1) variance-of-sum formula `(1+ρ)/(1−ρ)` is asymptotic in n → ∞ visits per cell. For cells with small E[visits], the finite-sum correction `(1 + ρ − 2ρ(1−ρⁿ)/(n(1−ρ)))` differs; not used here.
- The bootstrap propagates trajectory-level uncertainty through σ²_c, ρ_c^(1), and the empirical π simultaneously. CI reflects all three sources of variance.
- Per-cell ρ for cells with very small effective sample size (< 1,000 qualifying-run windows) is unstable; sparse-cell rule documents whether such cells dominate Var_AR1_term.
- The (1+ρ)/(1−ρ) factor diverges as ρ → 1; clipped to 0.999 here to avoid numerical overflow. Cells with ρ very close to 1 may have inflated contributions that the AR(1) model struggles to represent quantitatively.
- σ²_c is the all-windows variance, matching the linear correction term in synthetic Test 3 (commit e4a9036). The all-windows pooling does not condition on within-run membership; under stationarity this is equivalent to the marginal AR(1) variance.

## Companion artifacts

- `run_identification.json`  — per-cell qualifying-run counts
- `autocorrelation.json`     — ρ_c^(k) for k=1,2,3 with bootstrap CIs
- `stationarity.json`        — early/middle/late thirds analysis
- `variance_prediction.json` — Var_AR1_term, r_AR1, decision, sparse-cell rule
- `sensitivity.json`         — min run length grid, heavy-tail trim
