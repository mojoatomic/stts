# Flux Accumulator — C-Restricted Reentry Characterization

Re-evaluation under Interpretation C (Hybrid Restricted-Corpus Semantics). Empirical σ_failure is integrated to first basin-entry time τ_i over the restricted sub-corpus of trajectories that begin in a transient cell and later enter the failure basin. Predicted variance is computed via Kemeny-Snell on the transient sub-chain spanned by the restricted corpus, with the empty-cell rule applied. α is frozen from the first run (commit 310a9b7); no re-fit. Read-only against frozen artifacts.

All measurements are characterization-level statements; no "validation" or "confirmation" claims.

## Restricted corpus partition

| category | count |
|---|---:|
| Total objects                | 78 |
| Included (transient→basin)   | 39 |
| Excluded — starting in basin | 39 |
| Excluded — never reach basin | 0 |

## Headline measurements (restricted)

| quantity | value |
|---|---:|
| α (frozen)                | 4.27517 |
| n included                | 39 |
| σ_failure (median)        | 118.3420 |
| Var_empirical             | 188153.0925 |
| 95 % CI on Var_empirical  | [6351.2102, 503014.2515] |
| Var_pred,weighted         | 409071.8091 |
| ratio r                   | 2.1741 |
| median R² (full traces)   | 0.9884 |
| pre-registered R² gate    | 0.92 |
| multimodality flag        | False |
| η (negative-flux fraction)| 0 (structurally) |

## Empty-cell rule

Transient cells (full set, 10): [0, 3, 5, 8, 9, 12, 13, 16, 17, 19]

Surviving (n > 0 in restricted set): [0, 3, 5, 8, 9, 12, 17, 19] (8 of 10).

Dropped (n = 0 in restricted): [13, 16].

Q is built as `P_trained[surviving, surviving]` without renormalization. Mass leaving this surviving transient subset (to dropped transient cells or to the failure basin) is treated as exit for the absorbing-chain calculation, the standard construction.

## What the numbers mean (characterization level)

σ_failure is the median of σ_raw integrated to τ_i (first basin entry) over the restricted sub-corpus. Var_empirical is the sample variance of those σ_raw(τ_i) values; its bootstrap 95 % CI gives the precision on that summary statistic. Var_pred,weighted is the closed-form Kemeny-Snell prediction averaged over the restricted starting-cell distribution. The ratio r reports how much of empirical variance the Markov reward theory captures under the operator's per-cell mean cost κ_c on this restricted population.

Under C-restriction, the empirical and analytical quantities are now both stopping-time integrals, removing the structural mismatch that produced r ≈ 0.008 in the first run. Any remaining gap is attributable to within-cell heteroscedasticity of Φ, position-dependent α(c), or cross-coupling between drift and deformation directions.

## Note — cell-16 starts dropped from weighted aggregate

4 of 39 restricted objects have their first window in a transient cell that the empty-cell rule subsequently dropped. This happens when an object touches a transient cell only at the first ≤ W = 5 windows (D_KL = NaN by the early-window convention) and exits before any finite-D_KL window in that cell is observed. κ for that cell is undefined on the restricted corpus, so the cell is dropped from Q; objects starting there are then unmapped in the weighted Var_pred aggregate. Effective n in the weighted Var_pred,weighted: 35 of 39.

This is a consequence of the locked W = 5 D_KL convention interacting with cell 16's low operational frequency (cell 16 had n = 31 windows total in the first run's full corpus; under restriction, none survive the W = 5 NaN mask). It is not load-bearing on the per-cell variance prediction for the surviving 8 cells but is reported here so the gap between n_included and the effective weighted n is explicit.

## Note — statistical power on r

The point ratio `r = 2.1741` is computed from Var_pred,weighted = 409,072 and Var_empirical = 188,153. The 95 % bootstrap CI on Var_empirical is [6,351, 503,014], which maps to an r envelope of [0.813, 64.408]. The CI width is driven by the small restricted-corpus size (n = 39) combined with a right-skewed distribution of σ_raw(τ_i) (see histogram in `diagnostics.json`: 34 of 39 below the first bin edge, with one extreme observation at the high end). At this sample size and this empirical distribution, the r envelope brackets 1 from above; we cannot distinguish r ≈ 1 from r ≈ 2 at 95 % CI.

## Caveats

- **Within-cell heteroscedasticity** of Φ contributes to Var_empirical but not to Var_predicted; the analytical formula treats κ_c as a deterministic per-state cost.
- **Block-diagonal bundle metric** is assumed by the Pythagorean form. Per-cell residual patterns in `comparison.md` are the right test for cross-coupling.
- **Position-independent α** is used. Per-cell residual variation is the diagnostic for whether α(c) is needed.
- **Restricted-corpus size dictates CI width.** With n = 39 restricted objects, the bootstrap CI on Var_empirical is correspondingly wider than the full-78 number from the first run.
- **First basin-entry semantics.** τ_i is defined as the first window index at which the trajectory's M₂ cell falls in the failure basin {1, 2, 4, 6, 7, 10, 11, 14, 15, 18}. This is frozen as part of the C specification; no alternative basin definitions are explored here.

## Companion artifacts

- `sanity_check.txt`             — Part 1, re-execution for record
- `alpha.txt`                    — Part 2, frozen α from first run
- `restricted_corpus_summary.json` — partition counts and per-object category
- `predicted_variance.json`      — Part 3, Q/N/κ/Var_pred on surviving cells
- `empirical_variance.json`      — Part 4, σ_raw(τ_i) per object + bootstrap CI
- `diagnostics.json`             — Part 5, R² (full traces), histogram
- `comparison.md`                — Part 6, ratio r and per-cell breakdown
