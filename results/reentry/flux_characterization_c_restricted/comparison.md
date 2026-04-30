# Flux Accumulator — Part 6 (Interpretation C, restricted)

## Aggregates

| quantity | value |
|---|---:|
| Var_pred,weighted        | 409071.809077 |
| Var_empirical            | 188153.092532 |
| 95 % CI on Var_empirical | [6351.210217, 503014.251482] |
| ratio r                  | 2.174143 |
| n included objects       | 39 |
| dropped (start outside surviving) | 4 |

## Per-cell breakdown (restricted)

| starting cell | Var_predicted[c] | n_c (restricted starts) | weight w_c | contribution w_c · Var_pred[c] |
|---:|---:|---:|---:|---:|
| 0 | 462364.047524 | 1 | 0.0256 | 11855.488398 |
| 3 | 457276.463725 | 0 | 0.0000 | 0.000000 |
| 5 | 359001.836799 | 2 | 0.0513 | 18410.350605 |
| 8 | 461587.097616 | 12 | 0.3077 | 142026.799266 |
| 9 | 432219.728849 | 1 | 0.0256 | 11082.557150 |
| 12 | 463936.099477 | 12 | 0.3077 | 142749.569070 |
| 17 | 453964.879765 | 1 | 0.0256 | 11640.125122 |
| 19 | 463494.976527 | 6 | 0.1538 | 71306.919466 |

## Neutral interpretation

r ≫ 1.0: formula overestimates; empirical trajectories are more constrained than the absorbing-chain prediction.

**Fiber metric note.** α is treated as position-independent. If the per-cell contribution column shows systematic variation in `Var_predicted[c]` not balanced by the empirical weights, this is a signal that position-dependent α(c) may be needed in future iterations.
