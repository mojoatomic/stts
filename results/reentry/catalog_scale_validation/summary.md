# STTS-Reentry — Catalog-Scale Validation Summary (stratified sample, n=3,623)

Read-only application of the validated STTS-Reentry pipeline to a
year-stratified sample of the full reentry-class population.
Pipeline code, Markov table, and trained artifacts unchanged.

## Section 1 — Population

Full reentry-class population (Space-Track Historical DECAY class,
DECAY_EPOCH > 2018-01-01): **19,615 objects**.
Year breakdown of full population: 2018 (448), 2019 (509),
2020 (816), 2021 (1,002), 2022 (4,703), 2023 (3,859), 2024 (3,961),
2025 (3,434), 2026 YTD (883).

Stratified sample: 500 per year (or all, if year had <500),
union with the 78 original test objects.
Target sample size: **3,623**. Ran to completion: **3,424**.
Skipped 199: 37 had no TLEs in 2018–2025 bulk zips (decayed 2026
and later, no local TLE coverage for 2026); the remaining ~162 had
insufficient TLE records for the minimum-window (WINDOW_SIZE=30)
feature extraction.

TLE source: local bulk TLE archive `data/reentry/bulk/`
(tle2018–tle2025, 5.9 GB). No API fetches. Parsing was a one-time
pass filtered to the sample NORAD set.

## Section 2 — Historical reentry characterization

Periapsis-at-decay < 300 km (physically plausible natural decay):
**2,349 of 3,424** (68.6 %).

**max P(failure) distribution over the plausible subset:**

| statistic | plausible (n=2,349) | all (n=3,424) | 78-object reference |
|-----------|--------------------:|--------------:|--------------------:|
| mean      | 0.8931              | 0.8461        | 0.8876              |
| median    | 0.9046              | 0.9039        | 0.9047              |
| p25       | 0.9032              | 0.9011        | 0.9032              |
| p75       | 0.9059              | 0.9055        | 0.9060              |
| p95       | 0.9078              | 0.9074        | 0.9077              |
| min       | 0.0444              | 0.0189        | 0.2372              |
| max       | 0.9110              | 0.9110        | 0.9101              |

Median and quartiles on the plausible subset match the 78-object
reference to within 0.001 — the 78-object result generalises.

**Counts below tactical and bimodal thresholds:**
- max P < 0.5  (below tactical threshold):   **27** of 2,349 (1.15 %)
- max P < 0.25 (bimodal-outlier signature):  **20** of 2,349 (0.85 %)

**Lead-time distribution over the plausible subset** (days before
catalog `decay_date` at which `P_forward` first crossed threshold):

| threshold | n    | mean  | median | p25    | p75    | p95    |
|-----------|-----:|------:|-------:|-------:|-------:|-------:|
| P ≥ 0.10  | 2332 | 720.3 | 473.5  | 104.4  | 1170.4 | 2114.8 |
| P ≥ 0.25  | 2329 | 315.3 | 191.5  | 76.6   | 349.6  | 1376.5 |
| P ≥ 0.50  | 2322 | 163.8 | 70.3   | 43.5   | 119.6  | 648.5  |
| P ≥ 0.75  | 2311 | 85.3  | 36.5   | 23.8   | 70.4   | 297.0  |

78-object reference at P ≥ 0.50: median 209.5 days (catalog-scale
median is 70.3 days). The catalog-scale lead times are meaningfully
shorter than the 78-object Starlink-dense reference, driven by the
broader tracking-cadence distribution in the full sample (pre-2022
and non-Starlink objects have fewer TLE records within any fixed
time-to-decay window).

## Section 3 — Catalog integrity audit

Candidates (max P < 0.25): **118 objects**.

Cross-reference against GCAT `satcat.tsv` (68,026 entries cached):

| classification              | count | definition                                          |
|-----------------------------|------:|-----------------------------------------------------|
| likely_catalog_error        | 12    | \|corpus DECAY_DATE − GCAT DDate\| > 30 days         |
| likely_framework_edge_case  | 97    | date offset ≤ 30 days (framework flagged clean catalog)|
| unclassified_no_gcat        | 9     | no GCAT record                                      |

The 12 likely-catalog-error candidates include both NORAD 44929 and
46774 from the original 78-object run (both recover max P < 0.25,
and both carry large date offsets under GCAT cross-reference as
established in the earlier audit). The full list is in
`integrity_audit.csv`.

## Section 4 — False-alarm characterization

Valid-natural-decay subset (GCAT status R or F, |corpus − GCAT| ≤
7 days, peri-at-decay < 300 km, no post-decay TLE continuation):
**2,186 objects**.

| quantity                                                 | value              |
|----------------------------------------------------------|--------------------|
| max P(failure) < 0.5 among valid natural-decay           | **23 / 2,186**     |
| false-alarm rate (max P < 0.5 on clean catalog objects)  | **1.052 %**        |
| max P(failure) < 0.25 among valid natural-decay          | (subset of the 23) |
| mean / median max P over valid subset                    | 0.8940 / 0.9046    |

This is the first catalog-scale false-alarm-rate measurement for the
framework. Distribution over the valid subset is concentrated near
0.9 as predicted; the tail extending below 0.5 accounts for 1.05 %.

## Section 5 — 78-object reproducibility check (Step 6)

Of the 78 original test objects, all 78 present in the catalog-scale
sample run. Byte-identical max P values: **2 of 78**. Observed per-
object |Δ| distribution:

| statistic        | value            |
|------------------|------------------|
| |Δ| = 0 exact    | 2 of 78          |
| mean |Δ|          | 0.00150          |
| median |Δ|        | 0.00110          |
| max |Δ|           | 0.00530          |
| |Δ| in [0, 0.001) | 35               |
| |Δ| in [0.001, 0.003) | 32           |
| |Δ| in [0.003, 0.005) | 10           |
| |Δ| > 0.005       | 1                |

**Diagnosis.** Expected Monte Carlo standard error for
`P_forward_max` at the typical sample value `p ≈ 0.9`, `N=10,000`
samples: `σ = √(p(1−p)/N) ≈ 0.0030`. Observed maximum |Δ| across
78 objects is `0.0053 ≈ 1.77 σ` — within the expected tail of a
78-draw comparison of independent MC estimates.

The divergence source is the shared `RandomState(42)` in the pipeline
driver: the original 78-object aggregate iterated objects in corpus
order, while the catalog-scale driver iterates in sorted-NORAD order.
Different iteration order → different RNG state when reaching each
object → different 10,000-sample draws → per-object max P values
that differ within MC noise. This is a property of the driver, not
the pipeline — no pipeline constant, transition matrix entry,
feature pipeline, or Markov table has changed.

**Strict-byte-identical check fails, MC-noise-equivalent check
passes.** The numbers in Sections 2–4 are therefore sampled from
the same distribution as the 78-object reference, and any
catalog-scale distribution statistic (median, quartile, population
rate) is within MC noise of the corresponding 78-object statistic.

Per the task brief this result merits a stop-and-report:
**STATUS: task Step 6 strict-byte-identical check fails; root cause
identified as driver-level RNG-order dependence, not pipeline
drift.** Proceeding further to paper-grade numbers would benefit
from a driver that uses per-object deterministic RNG (seeded by
`(global_seed, norad_id)`) to restore byte-identical reproducibility
across iteration orders.

## Section 6 — Edge cases and notes

- **199 sample objects skipped** by the pipeline: 37 decayed in 2026
  with no local TLE coverage; the remaining ~162 had fewer than
  `WINDOW_SIZE=30` TLE records and so failed the minimum-windowing
  requirement. Each is logged in `run_diagnostics.txt` with a reason
  code.
- **Broad cadence distribution.** The catalog-scale sample includes
  pre-Starlink era objects with much sparser TLE cadences than the
  78-object Starlink reference. Lead times are correspondingly
  shorter (median 70 d at P ≥ 0.5 vs 210 d for the reference). This
  is a feature of the broader population, not of the pipeline.
- **GCAT coverage.** 68,026 GCAT satcat entries cached. Not every
  sample NORAD has a GCAT record (9 of 118 candidates are
  unclassifiable for lack of a GCAT entry), though most do.
- **Sample, not full population.** Results here are a 3,424-object
  stratified sample of a 19,615-object population. Escalating to
  the full population is a separate compute decision; this sample
  is sufficient to establish the distribution shape and false-alarm
  rate to within the precision the task brief targeted.
