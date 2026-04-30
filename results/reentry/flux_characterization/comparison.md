# Flux Accumulator — Part 6: Analytical vs Empirical Variance Comparison

## Aggregates

| quantity | value |
|---|---:|
| Var_pred,weighted        | 79982.118097 |
| Var_empirical            | 10347138.154654 |
| 95 % CI on Var_empirical | [7424468.853215, 13194660.632254] |
| ratio r                  | 0.007730 |
| n objects total          | 78 |
| n started in absorbing cell (dropped) | 39 |

## Per-cell breakdown

| transient cell | Var_predicted[c] | n_c (empirical starts) | weight w_c | contribution w_c · Var_pred[c] |
|---:|---:|---:|---:|---:|
| 0 | 160081.183108 | 1 | 0.0128 | 2052.322860 |
| 3 | 155432.374496 | 0 | 0.0000 | 0.000000 |
| 5 | 123461.304141 | 2 | 0.0256 | 3165.674465 |
| 8 | 162991.960231 | 12 | 0.1538 | 25075.686189 |
| 9 | 147129.392085 | 1 | 0.0128 | 1886.274258 |
| 12 | 162235.731683 | 12 | 0.1538 | 24959.343336 |
| 13 | 162991.960231 | 0 | 0.0000 | 0.000000 |
| 16 | 165610.038010 | 4 | 0.0513 | 8492.822462 |
| 17 | 153998.677049 | 1 | 0.0128 | 1974.342013 |
| 19 | 160883.482667 | 6 | 0.0769 | 12375.652513 |

## Neutral interpretation

r < 0.85: predicted variance is smaller than empirical; additional variance sources (within-cell flux heteroscedasticity, position-dependent fiber metric α(c), or cross-coupling between drift and deformation) may be present.

**Fiber metric note.** α is treated as position-independent in this characterization. If the per-cell contribution column shows systematic variation in `Var_predicted[c]` not balanced by the empirical weights, this is a signal that position-dependent α(c) may be needed in future iterations. The per-cell residual pattern is the right test for cross-coupling in the bundle metric; it is reported here without remediation in this run.
