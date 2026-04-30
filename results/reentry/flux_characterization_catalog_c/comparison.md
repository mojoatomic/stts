# Flux Accumulator — Part 6 (Catalog-scale C, restricted)

## Aggregates

| quantity | value |
|---|---:|
| Var_pred,weighted        | 147991.939900 |
| Var_empirical            | 33396042.877547 |
| 95 % CI on Var_empirical | [13339721.342390, 62053249.619738] |
| ratio r                  | 0.004431 |
| n included objects       | 1916 |
| dropped (start outside surviving) | 0 |

## Per-cell breakdown (catalog-scale restricted)

| starting cell | Var_predicted[c] | n_c (restricted starts) | weight w_c | contribution w_c · Var_pred[c] |
|---:|---:|---:|---:|---:|
| 0 | 152756.766703 | 90 | 0.0470 | 7175.422236 |
| 3 | 146639.919927 | 263 | 0.1373 | 20128.548508 |
| 5 | 115698.461153 | 391 | 0.2041 | 23610.698492 |
| 8 | 217523.410731 | 121 | 0.0632 | 13737.125626 |
| 9 | 138321.935628 | 468 | 0.2443 | 33786.360059 |
| 12 | 174393.561407 | 90 | 0.0470 | 8191.764367 |
| 13 | 217523.410731 | 7 | 0.0037 | 794.709747 |
| 16 | 401776.098632 | 26 | 0.0136 | 5452.076495 |
| 17 | 145125.916774 | 405 | 0.2114 | 30676.407251 |
| 19 | 154632.595651 | 55 | 0.0287 | 4438.827119 |

## Neutral interpretation

r ≪ 1.0: predicted variance is smaller than empirical; additional variance sources (within-cell heteroscedasticity, curvature effects, or residual model mismatch) are present.

**Fiber metric note.** α is treated as position-independent. If the per-cell contribution column shows systematic variation in `Var_predicted[c]` not balanced by the empirical weights, this is a signal that position-dependent α(c) may be needed in future iterations.
