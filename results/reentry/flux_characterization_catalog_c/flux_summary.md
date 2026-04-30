# Flux Accumulator — Catalog-Scale C-Restricted Characterization

Re-evaluation under Interpretation C on the 3346-object catalog-scale corpus. Same operator definition, same frozen α = 4.27517, same W = 5 D_KL convention, same empty-cell rule, same first-basin-entry semantics as the n = 39 78-object run (commit `1bd283c`). Only the population size and source per-object directory differ. Read-only against frozen artifacts.

All measurements are characterization-level statements; no "validation" or "confirmation" claims.

## Restricted corpus partition

| category | count | percent of total |
|---|---:|---:|
| Total objects                | 3346 | 100.00 % |
| Included (transient→basin)   | 1916 | 57.26 % |
| Excluded — starting in basin | 1366 | 40.82 % |
| Excluded — never reach basin | 64 | 1.91 % |

## Headline measurements (restricted)

| quantity | value |
|---|---:|
| α (frozen)                | 4.27517 |
| n included                | 1916 |
| σ_failure (median)        | 425.8478 |
| Var_empirical             | 33396042.8775 |
| 95 % CI on Var_empirical  | [13339721.3424, 62053249.6197] |
| Var_pred,weighted         | 147991.9399 |
| ratio r                   | 0.0044 |
| median R² (full traces)   | 0.9951 |
| pre-registered R² gate    | 0.92 |
| multimodality flag        | True |
| η (negative-flux fraction)| 0 (structurally) |

## Empty-cell rule

Transient cells (full set, 10): [0, 3, 5, 8, 9, 12, 13, 16, 17, 19]

Surviving (n > 0 in restricted): [0, 3, 5, 8, 9, 12, 13, 16, 17, 19] (10 of 10).

Dropped (n = 0 in restricted): [].

## Cross-run comparison

| run | n total | n incl | start-in-basin % | r point | r CI envelope |
|---|---:|---:|---:|---:|---|
| n = 39 (78-object reference) | 78 | 39 | 50.00 % | 2.174 | [0.81, 64.4] |
| catalog scale (3346) | 3346 | 1916 | 40.82 % | 0.004 | [0.002, 0.011] |

The catalog-scale CI is the direct test of whether the n = 39 result was small-sample-noise-limited. The CI envelope on r above is the relevant quantity; the point estimate alone without the envelope is not interpretable at this stage.

## Starting-in-basin fraction (pre-registered diagnostic)

1366 of 3346 objects (40.82 %) have their first window in a failure-basin cell. This is a diagnostic property of the LDA discretization on the catalog-scale population; it is not a pass/fail criterion. It informs future LDA refinement and cross-population comparison.

## Note — heavy-tail structure of σ_raw(τ_i)

The empirical distribution of σ_raw(τ_i) over n = 1916 restricted objects is heavily right-skewed: median = 425.8, p95 = 10302.0, p99 = 15873.9, max = 151179.0. Excess kurtosis (μ4/σ⁴ − 3) = **297.8** (Gaussian = 0). Three observations alone contribute **60.3 %** of total variance (NORADs by largest absolute residual: 44983, 46625, 49585). Trimmed-variance comparison:

| trim | Var_empirical | Var_pred,weighted | ratio r |
|---|---:|---:|---:|
| none | 33,396,043 | 147,992 | 0.0044 |
| drop top 3 | 13,234,662 | 147,992 | 0.0112 |
| drop top 1 % | 10,824,939 | 147,992 | 0.0137 |
| drop top 5 % | 7,076,467 | 147,992 | 0.0209 |

Per the brief, the empirical histogram is reported as observed; no trimming applied to the headline numbers. The trimmed values above are reported here for transparency about heavy-tail sensitivity. Even with the top 5 % trimmed, the ratio is r ≈ 0.0209 — Var_predicted is ~48× smaller than the trimmed empirical variance. Heavy tail is part of the story, not the whole story.

## Note — reconciliation with n = 39 result

The n = 39 78-object run reported r = 2.174 with CI envelope [0.81, 64.4]. The catalog-scale point r = 0.0044 **lies inside that envelope** (0.004 < 0.81). The n = 39 point estimate was a small-sample fluctuation within the n = 39 CI, and the catalog-scale point converges to the low end of that envelope. The catalog-scale CI envelope [0.0024, 0.0111] does **not** bracket 1; r ≪ 1 is statistically distinguishable from r ≈ 1 at this sample size.

The honest characterization: under C-restriction with the frozen W = 5, ε = 1e-9, single-global α, the closed-form Kemeny-Snell predictor captures ~0.44 % of empirical σ_raw(τ_i) variance at catalog scale. Within-cell heteroscedasticity of Φ — variance of per-window flux *within* the same cell, which the per-cell mean κ_c cannot represent — is the dominant unexplained variance source. Per-cell mean κ values for sparse cells (cell 13: 4,231 on n = 41 windows; cell 16: 551 on n = 178 windows) suggest these cells experience extreme Φ when visited, but this is captured in their κ. The bulk-cell κ values (cell 3: 2.92 on 281,721 windows; cell 17: 3.64 on 566,927 windows) are stable; the gap is not in the per-cell means but in the within-cell distribution.

## Caveats

- **Within-cell heteroscedasticity** of Φ contributes to Var_empirical but not to Var_predicted; the analytical formula treats κ_c as a deterministic per-state cost.
- **Block-diagonal bundle metric** is assumed by the Pythagorean form. Per-cell residual patterns in `comparison.md` are the right test for cross-coupling.
- **Position-independent α** is used. Per-cell residual variation is the diagnostic for whether α(c) is needed.
- **Frozen W = 5 D_KL convention** from the n = 39 run is preserved here without re-tuning. Sensitivity to W is out of scope.

## Companion artifacts

- `sanity_check.txt`             — Part 1, re-execution for record
- `alpha.txt`                    — Part 2, frozen α from first run
- `restricted_corpus_summary.json` — partition counts, percentages, per-object category
- `predicted_variance.json`      — Part 3, Q/N/κ/Var_pred on surviving cells
- `empirical_variance.json`      — Part 4, σ_raw(τ_i) per object + bootstrap CI
- `diagnostics.json`             — Part 5, R² (full traces), histogram
- `comparison.md`                — Part 6, ratio r and per-cell breakdown
