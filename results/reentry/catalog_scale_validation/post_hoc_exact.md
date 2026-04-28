# STTS-Reentry — Post-hoc Exact Hitting-Time Per Object

Exact `max_t P_end[M2_cell[t], Δt=10]` and `max_t P_hit[M2_cell[t], Δt=10]` per object, computed from per-object CSVs (M2_cell per window) and the frozen transition matrix. No pipeline code changes; committed MC outputs are preserved byte-identically.

## MC-vs-exact cross-check

Objects compared: 3424 (MC max and exact max, per-trajectory).

| statistic | value |
|---|---:|
| mean |Δ| | 0.00537 |
| median |Δ| | 0.00532 |
| p95 |Δ| | 0.00952 |
| p99 |Δ| | 0.01320 |
| max |Δ| | 0.02110 |

Expected MC stderr at p ≈ 0.9, N = 10000 samples: `√(0.9·0.1/10000) ≈ 0.003`. Max-over-trajectory of ~N windows is on the order of a few times stderr; consistent agreement if p99 |Δ| < ~0.01.

## Per-source summary

### reference_78

n = 78

| metric | exact P_end (endpoint) | exact P_hit (hitting CDF) | MC P_forward |
|---|---:|---:|---:|
| median | 0.8994 | 1.0000 | 0.9047 |
| mean   | 0.8822 | 1.0000 | 0.8876 |

| bucket | exact P_end count | exact P_hit count |
|---|---:|---:|
| max < 0.5 | 2 | 0 |
| max < 0.25 | 2 | 0 |
| max = 1.0 (hit basin) | — | 78 |

### catalog_scale

n = 3346

| metric | exact P_end (endpoint) | exact P_hit (hitting CDF) | MC P_forward |
|---|---:|---:|---:|
| median | 0.8994 | 1.0000 | 0.9039 |
| mean   | 0.8400 | 0.9845 | 0.8452 |

| bucket | exact P_end count | exact P_hit count |
|---|---:|---:|
| max < 0.5 | 219 | 64 |
| max < 0.25 | 116 | 51 |
| max = 1.0 (hit basin) | — | 3282 |

