# Test 1 — Pure Markov Reward Model

Validates the Kemeny-Snell closed-form variance predictor and its reentry-pipeline implementation against ground-truth Monte Carlo on a synthetic absorbing chain. Deterministic per-cell rewards (no within-cell variance). Pre-registered decision rule:

- Pass: |r − 1| < 0.05
- Marginal: 0.05 ≤ |r − 1| < 0.20
- Fail: |r − 1| ≥ 0.20

## Setup

| parameter | value |
|---|---|
| Transient cells | [0, 3, 5, 8, 9, 12, 17, 19] |
| Q source        | `results/reentry/flux_characterization_c_restricted/predicted_variance.json` (commit 1bd283c) |
| κ               | [5.9249, 7.792, 4.4805, 5.3342, 4.6406, 4.836, 4.7779, 5.8799] |
| Starting distribution | uniform over 8 transient cells |
| n trajectories  | 100,000 |
| Sampler seed    | 20260430 |
| Bootstrap B     | 10,000, seed 20260501 |

## Kemeny-Snell recomputation cross-check

N, v, Var_predicted recomputed from the loaded Q + κ: max |Δ| vs committed = **0.000e+00** (byte-identical to commit 1bd283c). The formula and implementation under test are the exact ones used in the reentry pipeline.

## Two ratios reported (metric-clarity note)

- **r_brief**  = Var_pred,weighted / Var_emp
    Var_pred,weighted = Σ_c π(c) · Var_per_cell[c]. This is the
    average within-starting-cell variance, weighted by the
    starting distribution. **This is what the canonical
    operator computes.**

- **r_total** = (Var_pred,weighted + Var_starts(v)) / Var_emp
    Adds the across-start variance term Var_starts(v), which is
    the variance of E[σ | starting cell] across starting cells.
    This is the LAW-OF-TOTAL-VARIANCE expected total marginal
    variance of σ_failure across all trajectories.

Var_emp is the marginal variance of σ_failure across all sampled
trajectories. Under non-degenerate starting distribution, Var_emp
differs from Var_pred,weighted by exactly Var_starts(v); r_total
should be ≈ 1 if and only if the formula and implementation are
correct, regardless of the starting distribution. r_brief = 1
only when Var_starts(v) is small relative to Var_pred,weighted —
which depends on chain structure.

## Headline result

| quantity | value |
|---|---:|
| Var_pred,weighted (within-cell, π-avg) | 444230.6413 |
| Var_starts(v) (across-start)            | 12526.0003 |
| Var_pred,total = sum                    | 456756.6415 |
| Var_emp (n = 100,000)                  | 458248.3949 |
| r_brief = Var_pred,weighted / Var_emp   | 0.96941 |
| 95 % CI on r_brief                      | [0.95241, 0.98660] |
| r_total = Var_pred,total / Var_emp      | 0.99674 |
| 95 % CI on r_total                      | [0.97927, 1.01442] |
| |r_brief − 1|                           | 0.03059 |
| |r_total − 1|                           | 0.00326 |

## Pre-registered decision

- **Decision on r_brief**: **PASS** (|r-1| = 0.03059)
- **Decision on r_total**: **PASS** (|r-1| = 0.00326)

**r_total passes the formula-correctness gate. The Kemeny-Snell variance predictor and its reentry-pipeline implementation are verified against synthetic ground-truth.** If r_brief differs from 1 while r_total ≈ 1, this is the expected across-start contribution under non-degenerate starting distribution; the formula is not 'wrong' — it is computing what it claims to compute (within-start variance, π-averaged), which is a different quantity from total marginal variance.

Proceed to Test 2.

## Convergence check

| n        | Var_emp     | r_brief  | r_total  |
|---:|---:|---:|---:|
|   1,000 | 450053.1075 | 0.98706 | 1.01489 |
|  10,000 | 474697.4095 | 0.93582 | 0.96221 |
| 100,000 | 458248.3949 | 0.96941 | 0.99674 |

## Per-cell variance comparison

| cell | n | mean_emp | mean_pred | var_emp | var_pred | ratio (pred/emp) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12447 | 743.39 | 734.58 | 483808.17 | 462364.05 | 0.95568 |
| 3 | 12639 | 706.82 | 703.54 | 468060.70 | 457276.46 | 0.97696 |
| 5 | 12371 | 382.93 | 382.19 | 363610.19 | 359001.84 | 0.98733 |
| 8 | 12405 | 638.43 | 642.58 | 450932.63 | 461587.10 | 1.02363 |
| 9 | 12462 | 539.31 | 548.70 | 411588.48 | 432219.73 | 1.05013 |
| 12 | 12488 | 685.28 | 684.55 | 469372.24 | 463936.10 | 0.98842 |
| 17 | 12543 | 658.26 | 657.12 | 459266.96 | 453964.88 | 0.98846 |
| 19 | 12645 | 737.84 | 741.11 | 455783.44 | 463494.98 | 1.01692 |

## Diagnostics

- Trajectory lengths: min 1, median 70, max 1223, mean 103.0
- All trajectories absorbed: True

