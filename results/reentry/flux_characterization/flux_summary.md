# Flux Accumulator (𝒜) — Reentry Characterization Summary

Single-page neutral summary of the six-part characterization. Read-only against frozen STTS-Reentry artifacts. No modifications to the canonical pipeline. All measurements are characterization-level statements; no "validation" or "confirmation" claims.

## Headline measurements

| quantity | value |
|---|---:|
| α (single global)        | 4.27517 |
| σ_failure (median)       | 5978.7707 |
| Var_empirical            | 10347138.1547 |
| 95 % CI on Var_empirical | [7424468.8532, 13194660.6323] |
| Var_pred,weighted        | 79982.1181 |
| ratio r = Var_pred / Var_emp | 0.0077 |
| median R² (linear σ_raw) | 0.9884 |
| pre-registered R² gate   | 0.92 |
| multimodality flag       | False |
| η (negative-flux fraction) | 0 (structurally) |

## What the numbers mean (characterization level)

σ_failure is the median of σ_raw integrated to the corpus decay epoch over 78 reentry trajectories under the canonical Pythagorean integrand Φ = √(H² + 2 α² · D_KL). Var_empirical is the sample variance of those 78 σ_raw(T_i) values; its bootstrap 95 % CI gives the precision on that summary statistic. Var_pred,weighted is the closed-form Kemeny-Snell prediction averaged over the empirical starting-cell distribution from the same 78 objects. The ratio r reports how much of empirical variance the Markov reward theory captures under the operator's per-cell mean cost κ_c. A value near 1 means the analytical predictor tracks empirical variance; departures point to within-cell heteroscedasticity, position-dependent fiber metric, or cross-coupling between drift and deformation directions.

## Note 1 — D_KL implementation

The brief assumed a runtime KL trust monitor exists in the pipeline. It does not. D_KL(t) was defined and computed for this characterization as the KL divergence between the empirical destination distribution of the last W = 5 transitions and the trained transition row at the current cell c(t), with additive log smoothing ε = 1e-09 applied to both numerator and denominator inside the log to keep the divergence finite at empirical zeros against trained-row nonzero entries. W = 5 was chosen to match the existing signal-separator buffer (config: ENTROPY_WINDOW). For windows where fewer than W transitions are available (early in a trajectory), D_KL(t) is set to NaN and the window is excluded from σ_raw integration. Total early-window exclusions across the 78 objects: 390 of 100,316 windows (the remaining 99,926 windows enter the integrals).

## Note 2 — H non-negativity

The brief described H(t) as "signed". The current pipeline implementation defines H = T + V where T = ½(ΔM₂)² ≥ 0 and V = |M₂ − μ_nominal| ≥ 0, so H ≥ 0 by construction. This is a documentation discrepancy in the canonical operator specification, not a code defect: under H ≥ 0, the Pythagorean form Φ = √(H² + 2 α² D_KL) trivially gives Φ ≥ 0, η is structurally zero, and the brief's mention of "signed information preserved via raw H trace" is moot. No code change in this run; the canonical specification should be amended to match the implementation.

## Note 3 — Three-pass sensitivity on NORAD 44929

NORAD 44929 has a corpus DECAY_DATE that the REV 2/3 audit established is wrong by ≈ 364 days (corpus 2023-03-09 vs corrected 2024-03-07). σ_raw integrates to T_i, so a wrong T_i for one of 78 samples changes its σ_raw(T_i) by ~1 year of accumulated Φ. Three configurations:

| pass | n | σ_failure (median) | Var_empirical | 95 % CI |
|---|---:|---:|---:|---:|
| full | 78 | 5978.7707 | 10347138.1547 | [7424468.8532, 13194660.6323] |
| corrected | 78 | 5978.7707 | 10977143.7083 | [7803391.0320, 13870361.5576] |
| excluded | 77 | 5949.3015 | 10354832.2895 | [7245516.2645, 13490309.7481] |

Per-object σ_raw(T_i) for 44929: full = 9062.8999; corrected = 13601.2675.

Decision rule: corrected configuration becomes the headline iff |Var_corrected − Var_full| > full-pass CI half-width on Var.
  Δ_Var = 630005.5537
  full-pass CI half-width = 2885095.8895
  load-bearing on Var: **False**

**Headline configuration: full.** Contamination of T_i for NORAD 44929 is not load-bearing on Var_empirical at the 95 % CI level. The full-pass headline number stands.

## Note 4 — Half the test set starts in absorbing cells

39 of 78 reentry objects have their **first** evaluable window assigned to a failure-class cell (cell index in {1, 2, 4, 6, 7, 10, 11, 14, 15, 18}). This is observed in the per-object records emitted by Part 4. The Kemeny-Snell variance formula assumes a transient starting state; objects starting in an absorbing cell contribute weight zero to `Var_pred,weighted`. The weighted aggregate in Part 6 is therefore an average over only the 39 transient-starting objects.

For the objects starting in absorbing cells, the analytical theory's per-object prediction is `Var_i[σ_failure] = 0` (already absorbed), while their empirical `σ_raw(T_i)` continues to accumulate Φ over the full corpus-defined time horizon `T_i`. This is a **structural mismatch in what the two quantities measure**:

- **Analytical:** variance of the cumulative cost paid until first basin entry (a stopping-time integral).
- **Empirical:** variance of the cumulative cost paid over a fixed time horizon `T_i`, regardless of whether/when the trajectory enters the basin.

These are different quantities. The reported ratio `r = 0.0077` should be read with that mismatch in mind: the closed-form predictor is solving a different problem than the empirical measurement, and the order-of-magnitude gap is consistent with that. A like-for-like analytical predictor would integrate Φ over a fixed-horizon expectation rather than a hitting-time expectation. Implementing that predictor is out of scope for this characterization; reported here as the most likely explanation for the magnitude of `r`.

## Note 5 — κ_16 is anomalous on sparse support

Cell 16 has 31 windows total across the 78 objects and `κ_16 = 60.56` — an order of magnitude above the next-largest κ. This is a small-sample mean dominated by a few high-Φ windows. Per Part 6's per-cell breakdown, see the weight column `w_c` for cell 16 to assess whether this materially affects `Var_pred,weighted`. With a small `w_c` it does not.

## Caveats

- **Within-cell heteroscedasticity.** The Kemeny-Snell formula treats κ_c as a deterministic per-state cost. Real per-visit Φ has within-cell variance that contributes to Var_empirical but is absent from Var_predicted. A correctly-implemented analytical prediction will systematically under-estimate empirical variance to the extent that within-cell flux is heteroscedastic across visits to the same state. This is one of the "additional variance sources may be present" cases the brief warned about and is the most likely explanation for any r < 1 observed here.
- **Block-diagonal bundle metric.** The Pythagorean construction assumes no cross-coupling between drift (H) and deformation (α·√(2 D_KL)) directions. Empirical evidence of cross-coupling would appear as systematic per-cell residual patterns; see the per-cell breakdown in `comparison.md`.
- **Position-independent α.** A single global α is used. Per-cell residual variation in `Var_predicted[c]` is the right diagnostic for whether α(c) is needed; not implemented in this run.

## Companion artifacts

- `sanity_check.txt`         — Part 1 toy-chain Kemeny-Snell verification
- `alpha.txt`                — Part 2 α calibration
- `predicted_variance.json`  — Part 3 closed-form CIs per starting cell
- `empirical_variance.json`  — Part 4 σ_raw(T_i), Var_emp + bootstrap CI, three-pass on 44929
- `diagnostics.json`         — Part 5 η, R², histogram, multimodality
- `comparison.md`            — Part 6 analytical vs empirical, per-cell breakdown
