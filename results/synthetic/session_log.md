# Synthetic Validation Session — Tests 1, 2, 3 — Results

Pre-registered three-test session designed to isolate the math of the
Kemeny-Snell variance predictor from data-pipeline assumptions before
interpreting the catalog-scale C-restriction result (commit `fdd1ba3`,
r = 0.0044, predicted variance captures ~0.4 % of empirical variance).
Read-only against frozen artifacts. Single commit.

## Outcomes

| test | outcome | criterion |
|---|---|---|
| Test 1 — Pure Markov reward (formula+impl) | **PASS** | \|r-1\| < 0.05 (got 0.031 on r_brief, 0.003 on r_total) |
| Test 2 — Heteroscedastic per-cell rewards (diagnosis under controlled conditions) | **CONFIRMED under clean analytical control** | \|r_total_corrected − 1\| < 0.10 across all 3 regimes (untruncated mode); brief-spec truncated mode shows artifactual high-regime departure |
| Test 3 — Reentry-calibrated bridge | **DISCONFIRMED on catalog data** | r_corrected = 0.058 (vs target ≈ 1) |

## Test 1 — Pure Markov reward model

Synthetic 8-cell absorbing chain (Q and κ from
`flux_characterization_c_restricted/predicted_variance.json`, commit
`1bd283c`). Deterministic per-cell rewards. n = 100,000 trajectories,
uniform starting distribution.

Recomputation of Kemeny-Snell N, v, Var_predicted matches the committed
artifact byte-identically (max |Δ| = 0). The formula and implementation
under test are the exact ones used in the reentry pipeline.

| ratio | point | 95 % bootstrap CI | |r−1| | decision |
|---|---:|---:|---:|---|
| r_brief (within-cell, π-avg) | 0.969 | [0.952, 0.987] | 0.031 | PASS |
| r_total (with across-start)  | 0.997 | [0.979, 1.014] | 0.003 | PASS |

Per-cell ratios (Var_pred[c] / Var_emp[c]) all in [0.956, 1.050] across
the 8 cells. The Kemeny-Snell formula and its implementation are
verified against synthetic ground-truth.

**Implication for diagnosis:** the math is correct, the implementation
is correct. Any failure on real data is in the modeling assumptions,
not in the closed-form predictor.

## Test 2 — Heteroscedastic per-cell rewards (synthetic controlled)

Same chain. Per-visit Φ ~ N(κ_c, σ_c²), three pre-registered regimes
σ_c = {0.1, 0.5, 1.0} · κ_c. Two modes per regime: brief-specified
truncated-at-zero, and a clean untruncated control.

| regime | mode | r_uncorrected | r_corrected | r_total_corrected |
|---|---|---:|---:|---:|
| low      | truncated   | 0.969 | 0.969 | 0.997 |
| low      | untruncated | 0.969 | 0.970 | 0.997 |
| moderate | truncated   | 0.917 | 0.919 | 0.945 |
| moderate | untruncated | 0.967 | 0.970 | 0.997 |
| high     | truncated   | 0.613 | 0.619 | 0.636 |
| high     | untruncated | 0.961 | 0.970 | 0.997 |

Untruncated mode satisfies |r_total_corrected − 1| < 0.10 across all
three regimes (within MC noise of unity). The truncated mode's
high-regime departure is attributable to the truncate-at-zero step
(redraw + clip systematically inflates per-visit reward mean and
reduces per-visit variance below σ_c²; the analytical correction
assumes untruncated normal moments).

**Decision: CONFIRMED under clean analytical control.** The within-
cell heteroscedasticity correction Var_correction = Σ_c E_π[visits to
c] · σ_c² is the correct linear form under independent normal per-
visit rewards. The diagnosis (within-cell heteroscedasticity is the
mechanism by which the canonical Var_pred misses true variance) is
verified for the synthetic chain.

## Test 3 — Reentry-calibrated bridge

Computed σ²_c,empirical from the catalog-scale per-window Φ values
(W = 5, ε = 1e-9, frozen α = 4.27517) restricted to the C-restricted
corpus's within-τ_i windows. Applied the analytical correction term
using the catalog-scale N matrix and the empirical starting-cell
distribution π.

| quantity | value |
|---|---:|
| Var_pred,weighted (recomputed)            | 147,991.94 |
| Var_starts(v)                             | 84,311.53 |
| Var_pred,total                            | 232,303.47 |
| Var_correction = Σ_c E_π[visits] · σ²_c,emp | 1,800,755.50 |
| Var_pred,corrected (brief's form)         | 1,948,747.44 |
| Var_pred,total + correction               | 2,033,058.97 |
| **catalog Var_emp**                       | **33,396,042.88** |
| **r_corrected**                           | **0.058** |
| r_total_corrected                         | 0.061 |

**Per-cell breakdown of Var_correction is dominated by sparse-cell
artifacts.** Of the 1,800,755 correction:

| cell | n windows | κ_c     | σ²_c,emp     | contribution | share |
|---:|---:|---:|---:|---:|---:|
| 13 | 41   | 4,231.38 | 287,798,249 | 1,051,455 | **58.4 %** |
| 16 | 178  | 551.33   | 19,550,255  | 748,978   | **41.6 %** |
| 17 | 566,927 | 3.64  | 7.03        | 169       | 0.01 % |
| 3  | 281,721 | 2.92  | 2.09        | 81        | 0.00 % |
| (others) | … | … | … | < 40 each | < 0.01 % each |

**Cells 13 and 16 carry 99.98 % of Var_correction** despite covering
only 219 of ~1.25 M restricted-corpus windows. Their large σ²_c,emp
values reflect their anomalous κ_c from the catalog-scale aggregation
(small-sample effects on rare cells with extreme LDA-projection
centroids). The bulk cells where the data actually lives — cells 3,
9, 17 with 281k–567k windows each — contribute < 0.05 % of
Var_correction.

If cells 13 and 16 were excluded as small-sample artifacts,
Var_correction would be ≈ 322 (dominated by cell 17's 169) and
r_corrected ≈ 0.0044, indistinguishable from r_uncorrected.

**Decision: DISCONFIRMED.** The linear within-cell variance correction
term — even when calibrated with the full empirical σ²_c values —
explains at most 6 % of the catalog-scale variance, and that 6 % is
entirely artifactual from sparse cells. The bulk-cell within-cell
heteroscedasticity is too small to close the gap to Var_emp = 33.4 M.

## Decision-tree path

The brief specified three pre-registered cases on Tests 1 and 2:

1. Test 1 passes + Test 2 confirms diagnosis → extend operator with
   within-cell variance term (Fix A), return to reentry.
2. Test 1 fails → stop, fix formula/implementation.
3. Test 1 passes + Test 2 disconfirms → unknown failure mode,
   significant additional investigation required.

Test 1 passed; Test 2 confirmed under controlled (untruncated)
conditions. Per the brief's tree, this is **case 1**: extend the
operator with the within-cell variance term and return to reentry.

But Test 3 — applying that correction with the catalog-empirical
σ²_c values — disconfirms the diagnosis on real data. The linear
within-cell variance term is the right form under the assumptions
of Test 2 (independent normal per-visit Φ); those assumptions do
not hold on the catalog reentry corpus.

**The empirically-applicable case is case 3, modulo a refinement.**
Within-cell heteroscedasticity per the linear independence-
assuming form is verified as a *correct correction* but is
*insufficient* to explain the catalog-scale Var_emp. Other
structural sources are present.

## Candidate sources of the remaining ~94 % gap

In rough order of likelihood, given the catalog Φ distribution's
empirical excess kurtosis ≈ 298 and D_P10 = 0.033 nats:

1. **Within-cell autocorrelation of Φ.** A trajectory's per-window Φ
   values within the same cell are not iid; they reflect slowly
   varying underlying physics (e.g. an object decaying through cell c
   has correlated Φ across its multiple windows in c). Under positive
   autocorrelation, the variance of the sum-over-visits is *larger*
   than `n_visits · σ²` (the linear correction assumes
   independence). Catalog Φ likely has substantial within-cell
   autocorrelation.

2. **Heavy-tailed per-visit Φ.** Excess kurtosis 298 is far from
   Gaussian. The linear variance correction uses second moments
   only; heavy tails contribute through fourth and higher moments
   that propagate via random-sum-of-iid-heavy-tailed effects.

3. **Trajectory-level heterogeneity.** A handful of trajectories
   contribute disproportionately to Var_emp (top 3 catalog
   trajectories: 60 % of variance). The Markov reward model treats
   trajectories as exchangeable conditional on starting cell;
   trajectory-level heterogeneity is outside its representation.

4. **Non-Markovian trajectory structure.** D_P10 = 0.033 nats is
   non-zero (Λ_P10 = 2 · D_P10 = 0.066 baseline). Over hundreds of
   visits, the deviation from first-order Markov accumulates and
   the Markov reward formula's predictions degrade.

## Recommended next step

Do not extend the canonical operator with the linear within-cell
variance term and return to reentry. The correction is verified as
*the right form* under the synthetic assumptions, but Test 3
demonstrates that those assumptions do not hold on this corpus, and
the bulk-cell empirical σ²_c is too small to bridge the gap.

The diagnostic next step is **within-cell autocorrelation
characterization**: measure the autocorrelation of per-window Φ
within trajectories, restricted to consecutive visits to the same
cell. If positive autocorrelation is large and persistent, the
correction term needs a (Σ_c E[visits] · σ²_c · effective-sample-size
factor) form rather than (Σ_c E[visits] · σ²_c). This is a separate
session.

The synchronization test (deferred from earlier) and any further
catalog-scale operator runs should remain paused until this
investigation completes.

## Companion artifacts

- `test1/result.json` and `test1/summary.md`
- `test2/result.json` and `test2/summary.md`
- `test3/result.json` and `test3/summary.md`
- `synthetic/synthetic_chain.py`           — chain construction utility
- `synthetic/test1_pure_markov.py`         — driver
- `synthetic/test2_heteroscedastic.py`     — driver
- `synthetic/test3_reentry_calibrated.py`  — driver

## Reproducibility

| seed | usage |
|---|---|
| 20260430 | trajectory sampler (Tests 1, 2) |
| 20260501 | bootstrap CI (Tests 1, 2) |
| 20260502 + hash(label) | heteroscedastic per-visit draws (Test 2) |

All Q, κ, N values byte-identical to commits `1bd283c` (n = 39 chain)
and `fdd1ba3` (catalog-scale 10-cell artifact). α = 4.27517 frozen
from commit `310a9b7`. Catalog Var_emp 33,396,042.88 from
`fdd1ba3:results/reentry/flux_characterization_catalog_c/empirical_variance.json`.
