# STTS-Reentry — REV 3 Summary: Exact Hitting-Time + Two-Bootstrap Envelope

Results from plan REV 3 execution. Read-only against frozen model artifacts; no pipeline code changes; no modifications to `artifacts/reentry/markov_table.npz` (md5 `bc7902e3073305bbba96d148f0c4a1bb`).

**Stopping rule applied**: report what the numbers show and stop. Interpretation of whether the framework holds on this domain is downstream.

## 1. Exact hitting-time CDF over the trained Markov table

Primitives: `reentry/hitting_time.py`. Two quantities derived from the frozen 20×20 transition matrix P and the 10-cell failure basin F:

- `P_end[x, Δt] = P(X_Δt ∈ F | X_0 = x) = [P^Δt · 1_F]_x` — endpoint probability (what MC currently approximates).
- `P_hit[x, Δt] = P(τ_F ≤ Δt | X_0 = x) = 1 − [Q^Δt · 1]_x` for `x ∉ F`, where `Q = P[non-F, non-F]` is the 10×10 sub-stochastic submatrix — hitting-time CDF.

Inequality `P_hit ≥ P_end` verified at all (x, Δt). Strict inequality measures basin leakage; some failure cells are nearly absorbing (cell 10: 0.000 leak), others are highly leaky (cell 11: 0.77 leak at Δt=10, cell 2: 0.60 leak at Δt=10).

**Selected point values at Δt = 10** (per-cell, from frozen P):

| cell | class | P_end | P_hit | leakage |
|---:|---:|---:|---:|---:|
| 0 | nominal | 0.020 | 0.035 | 0.015 |
| 3 | nominal | 0.028 | 0.049 | 0.021 |
| 5 | nominal | 0.158 | 0.446 | 0.289 |
| 9 | nominal | 0.094 | 0.215 | 0.121 |
| 11 | **FAILURE** (leaky) | 0.229 | 1.000 | 0.771 |
| 17 | nominal | 0.045 | 0.087 | 0.042 |
| 19 | nominal | 0.019 | 0.034 | 0.014 |

Full table in `bootstrap_training_cis.md`.

## 2. Training-set bootstrap (B = 1000, frozen projection)

`bootstrap_training_matrix.py` — resamples 529 training satellites with replacement, rebuilds transition counts under the frozen scaler/LDA/KMeans/basin/failure_cells, and recomputes each quantity per replicate. Full-training reconstruction against the committed Markov table: `max|Δ| = 0.00e+00` (byte-identical).

| quantity | point | 95% CI | CI width |
|---|---:|---:|---:|
| **`λ_2`** (second eigenvalue) | 0.98287 | [0.93435, 0.99933] | 0.0650 |
| **`D_P10`** (Markov-order KL aggregate) | 0.03258 | [0.03094, 0.03877] | 0.0078 |
| **`μ_nominal`** (LDA projection mean) | −0.1527 | [−0.1979, −0.1101] | 0.0878 |

**Mixing-time implication.** `τ = 1/(1−λ_2)`:
- Point: 58.4 Markov steps
- 95 % CI: **[14.8, 1493] steps** — three orders of magnitude. The mixing time is severely under-constrained by the training data.

**Stationary distribution π** (selected cells):

| cell | π point | 95 % CI |
|---:|---:|---:|
| 3  (nominal, bulk) | 0.358 | [0.000, 0.403] |
| 17 (nominal, bulk) | 0.226 | [0.000, 0.249] |
| 9  (nominal)       | 0.104 | [0.000, 0.124] |
| 7  (failure, sparse) | 0.007 | [0.001, 0.362] |
| 18 (failure, sparse) | 0.010 | [0.001, 0.611] |

Stationary probability is under-constrained for cells with sparse training support. For the bulk cells (3, 17, 9), the lower bound of the CI sits at 0.000 because some bootstrap resamples produce nearly-reducible chains in which those cells are effectively disconnected. Aggregate basin mass (point): 10.15 % failure vs 89.85 % nominal; the individual-cell CI widths do not aggregate up the basin-level picture.

**Hitting-time CDF CIs** (selected cells, at Δt = 10):

| cell | class | P_hit point | 95 % CI | width |
|---:|---:|---:|---:|---:|
| 0  | nominal | 0.0354 | [0.0279, 0.0456] | 0.018 |
| 3  | nominal | 0.0491 | [0.0425, 0.0565] | 0.014 |
| 5  | nominal | 0.4462 | [0.4202, 0.4703] | 0.050 |
| 9  | nominal | 0.2152 | [0.1942, 0.2344] | 0.040 |
| 17 | nominal | 0.0870 | [0.0767, 0.0973] | 0.021 |
| 19 | nominal | 0.0337 | [0.0253, 0.0454] | 0.020 |
| failure cells (all 10) | F | 1.0000 | [1.0000, 1.0000] | 0 |

Hitting-time CDF values for the bulk nominal cells (which is where operational trajectories spend most of their time) are well-constrained: CI widths of 0.014–0.050 on probability values between 0.03 and 0.45.

## 3. Validation-set bootstrap with decay-epoch sensitivity (B = 1000, three passes)

`bootstrap_validation_set.py` — resamples per-object rows from `per_object.csv` (catalog-scale, n = 3,424) and `aggregate_summary.csv` (78-object reference). Three passes per statistic:

- **Full**: rows as committed.
- **Excluded**: drop the 12 NORADs in `catalog_errors_provenance.csv` (year-off-by-one and related catalog errors).
- **Corrected**: substitute GCAT DDate for 7 CONFIRMED-year-off-by-one objects; recompute lead times against the corrected reference epoch; drop the 5 LIKELY cases whose true date falls outside the 2018–2025 TLE window.

**78-object reference (headline numbers with CIs):**

| statistic | full | excluded | corrected |
|---|---|---|---|
| median max P (MC) | 0.9047 [0.9041, 0.9054] | 0.9047 [0.9042, 0.9054] | 0.9047 [0.9041, 0.9054] |
| rate max P < 0.25 | 0.0256 [0.000, 0.064] | 0.0130 [0.000, 0.039] | 0.0256 [0.000, 0.064] |
| median lead @ P≥0.50 (d) | 209.5 [98.5, 265.5] | 209.5 [98.5, 265.5] | 209.5 [98.5, 265.5] |
| median lead @ P≥0.75 (d) | 91.0 [74.0, 131.6] | 91.0 [74.0, 131.6] | 91.0 [74.0, 131.6] |

**Load-bearing check**: for every 78-object statistic, `|corrected − full| ≤ full CI half-width`. The 210-day median is not load-bearing on the single 78-object contaminated label (44929); it is labeled as a bimodal outlier with no recorded lead time at any tested threshold, so corrections cannot move the median.

**Catalog-scale (n = 3,424) headline:**

| statistic | full | excluded | corrected |
|---|---|---|---|
| median max P (MC) | 0.9039 [0.9039, 0.9040] | 0.9039 [0.9039, 0.9040] | 0.9039 [0.9039, 0.9040] |
| rate max P < 0.25 | 0.0345 [0.0286, 0.0406] | 0.0311 [0.0249, 0.0369] | 0.0331 [0.0275, 0.0392] |
| median lead @ P≥0.50 (d) | 76.50 [74.19, 79.40] | 76.50 [74.19, 79.40] | 76.50 [74.19, 79.40] |

For every catalog-scale statistic, `|corrected − full|` is within full-pass CI half-width. Decay-epoch contamination is not load-bearing at the 95 % CI level.

## 4. Exact per-object hitting-time (post-hoc)

`post_hoc_exact.py` — computes exact `max_t P_end[M2_cell(t), Δt=10]` and `max_t P_hit[M2_cell(t), Δt=10]` per object from the per-object CSVs. No pipeline code was modified; committed MC outputs are byte-identical by construction.

**MC-vs-exact diagnostic** (verification 1 from the plan):

| level | n | mean \|Δ\| | p95 \|Δ\| | p99 \|Δ\| | max \|Δ\| |
|---|---:|---:|---:|---:|---:|
| per-window | 5,000,724 | 0.00237 | 0.00657 | 0.00947 | 0.02586 |
| per-trajectory max | 3,424 | 0.00537 | 0.00952 | 0.01320 | 0.02110 |

Per-window signed mean Δ = +0.000000 (unbiased). Expected MC stderr at p ≈ 0.9 = 0.00300. Per-window agreement is within MC stderr; per-trajectory max is slightly wider due to extreme-value statistics over long trajectories. Verification passes at the per-window level.

**Per-object exact summary** (what the framework actually predicts under hitting-time semantics):

**78-object reference:**

| metric | exact P_end (endpoint) | exact P_hit (hitting CDF) | MC P_forward |
|---|---:|---:|---:|
| median | 0.8994 | **1.0000** | 0.9047 |
| mean   | 0.8822 | **1.0000** | 0.8876 |
| n with max < 0.5 | 2 | **0** | — |
| n with max = 1.0 | — | **78** | — |

**Every one of the 78 reference trajectories visits a failure basin cell at some window.** Under the endpoint metric, 2 objects (44929, 46774) show max P_end < 0.5 because they only visit cell 11 (the leaky failure cell with P_end = 0.229). Under the hitting-time CDF, those same 2 objects reach max P_hit = 1.0 because they do enter the basin — the bimodal-outlier signature at Δt=10 is an artifact of basin-leakage at the endpoint metric, not a property of whether the trajectory reached the basin.

**Catalog-scale (3,346 non-reference objects):**

| metric | exact P_end | exact P_hit | MC P_forward |
|---|---:|---:|---:|
| median | 0.8994 | 1.0000 | 0.9039 |
| mean   | 0.8400 | 0.9845 | 0.8452 |
| n with max < 0.5 | 219 | **64** | — |
| n with max < 0.25 | 116 | **51** | — |
| n with max = 1.0 | — | **3,282** | — |

At catalog scale, 3,282 / 3,346 = 98.1 % of trajectories reach the failure basin at some window under the hitting-time metric. 64 trajectories (1.9 %) never do — these are the genuine bimodal outliers under hitting-time semantics, a tighter set than the 116 identified under endpoint.

## 5. Verification (plan REV 3, §5 Verification)

1. **P_forward exact vs MC agreement** — per-window mean |Δ| = 0.00237 ≈ MC stderr at p=0.9 (0.003), signed mean Δ ≈ 0 (unbiased). **Passes.**
2. **Byte-identical P_forward_mc** — no pipeline code was changed; MC-based `aggregate_summary.csv` values are preserved verbatim. **Passes trivially.**
3. **Hitting-time inequality** — `assert_hit_ge_end(P_hit, P_end)` in `hitting_time.py` verified at all (x, Δt) ∈ 20 × 30. **Passes.**
4. **Bootstrap sanity** — training-bootstrap full-sample P reconstruction: `max |Δ|` vs frozen = `0.00e+00`; point estimate λ_2 = 0.98287 matches the frozen value identically. Validation-bootstrap envelope asserts passed for every reported statistic. **Passes.**
5. **Decay-epoch sensitivity delta** — for every statistic on both the 78-object reference and the 3,424-object catalog sample, `|corrected − full| ≤ full CI half-width`. The 210-day figure is not load-bearing on contaminated labels. **Passes.**

## 6. Companion files

- `bootstrap_training_cis.{json,md}` — training-set bootstrap, per-replicate stats, full P_hit / P_end CIs.
- `bootstrap_validation_cis.{json,md}` — validation-set bootstrap, three-pass table per statistic.
- `per_object_exact.csv` — per-object exact `max_t P_end` and `max_t P_hit` at Δt = 10, plus horizon sweep.
- `mc_vs_exact.csv` — per-object MC-max vs exact-max diagnostic.
- `post_hoc_exact.md` — MC-vs-exact cross-check summary.

## 7. What the numbers show

This section records measurements. Downstream interpretation is outside scope.

1. The trained Markov transition matrix's second eigenvalue λ_2 is 0.983 [0.934, 0.999]. The mixing time implied by the CI spans 15 to 1,493 Markov steps.
2. The first-vs-second-order KL aggregate `D_P10^reentry` is 0.033 [0.031, 0.039]. Under the task-specified k = 2 multiplier, `Λ_P10 = 0.065` with CI [0.062, 0.078].
3. Per-cell hitting-time CDF values at Δt = 10 for bulk nominal cells (0, 3, 17, 19) are 0.03–0.09 with CI widths < 0.025. For cell 5 (boundary between nominal and failure cluster) it is 0.446 [0.420, 0.470].
4. Under hitting-time semantics, every one of the 78 reference trajectories reaches the failure basin at some window; 98.1 % of the 3,346 catalog-scale trajectories do; 1.9 % do not.
5. The 210-day median lead time at P ≥ 0.5 on the 78-object reference has 95 % CI [98.5, 265.5] days. It is not load-bearing on the single contaminated-label object in that set.
6. The catalog-scale median lead time at P ≥ 0.5 is 76.5 [74.2, 79.4] days — three times tighter CI than the 78-object reference due to larger n.
7. The catalog-scale false-alarm rate (max MC P < 0.5 on clean natural-decay objects) is 1.05 % with 95 % CI [0.64 %, 1.51 %].
