# STTS-Reentry — Signal Separator Operating Characteristic

Measured operating characteristic of the signal separator on the
78-object test run. Read-only against existing artifacts. No pipeline
modifications.

Companion file: `results/reentry/signal_separator_characteristic.csv`
(per-object transition counts).

---

## Section 1 — Implementation

**Location.** `reentry/energy_state.py:230-260` (canonical) and
`reentry/energy_state_aggregate.py:108-121` (the 78-object aggregate
path used for this run). Both implement the same rule; the aggregate
file was the one that produced `results/reentry/per_object/*.csv`.

**Criterion.** At each window `t ≥ ENTROPY_WINDOW`, the separator:

1. Takes the M₂ cell assignments of the last `ENTROPY_WINDOW`
   windows (`M2_cells[t − ENTROPY_WINDOW + 1 : t + 1]`).
2. Builds a normalized empirical distribution over the 20 Markov
   cells from those samples and computes its Shannon entropy
   `obs_H` in nats via `scipy.stats.entropy`.
3. Reads the expected entropy `exp_H` for the current cell from
   the trained Markov table (`expected_entropy[M2_cells[t]]`).
4. Flags the window as `artifact` (`artifact_flag = 1`) when:
   - `exp_H ≥ 1e-10` and `obs_H > ENTROPY_THRESHOLD × exp_H`, **or**
   - `exp_H < 1e-10` (deterministic cell) and `obs_H > 1e-10`.

Windows with `t < ENTROPY_WINDOW` are not classified (the warm-up
region; they retain the zero-initialized default).

**Thresholds used in the 78-object run.** Fixed module-level
constants declared in `reentry/config.py:332, 338`:

- `ENTROPY_WINDOW = 5`
- `ENTROPY_THRESHOLD = 2.0`

Both are fixed constants. The `exp_H` reference value is
per-cell (adaptive to which cell the current window occupies),
loaded from `artifacts/reentry/markov_table.npz`
(`expected_entropy`, shape (20,)). Values range from `0.000` (cell
13, deterministic) to `1.435` (cell 16); mean `1.060`.

---

## Section 2 — Transition classifications

Per-object counts saved to
`results/reentry/signal_separator_characteristic.csv` (78 rows,
columns: `norad_id, total_transitions, signal_transitions,
artifact_transitions, artifact_rate`).

- `total_transitions` is the number of windows that were classified
  (`n_windows − ENTROPY_WINDOW`).
- `signal_transitions = total_transitions − artifact_transitions`.
- `artifact_rate = artifact_transitions / total_transitions`.

Per-object window counts range from 46 (shortest trajectory) to
3,338 (longest); median 1,148.

Cross-check against `results/reentry/aggregate_summary.csv`
(`artifact_count` column): 0 mismatches across 78 objects.

---

## Section 3 — Population-level aggregates

| quantity                                   | value                  |
|--------------------------------------------|------------------------|
| Test objects                               | 78                     |
| Total classifiable transitions             | 99,926                 |
| Total classified as signal                 | 99,856                 |
| Total classified as artifact               | 70                     |
| Population artifact rate                   | 7.01 × 10⁻⁴ (0.0701 %) |
| Per-object artifact rate — min             | 0.000000               |
| Per-object artifact rate — median          | 0.000000               |
| Per-object artifact rate — max             | 0.005315 (7/1,317)     |

**Objects with `artifact_rate > 0`: 32 of 78.**
All 32 objects, sorted by absolute count (ties broken by NORAD):

| NORAD | artifact / total | rate (%) |
|-------|------------------|---------:|
| 45737 | 7/1,317          | 0.532    |
| 44249 | 4/2,767          | 0.145    |
| 45091 | 4/982            | 0.407    |
| 47678 | 4/759            | 0.527    |
| 44284 | 3/1,195          | 0.251    |
| 44293 | 3/1,201          | 0.250    |
| 45070 | 3/928            | 0.323    |
| 46774 | 3/1,301          | 0.231    |
| 47669 | 3/2,309          | 0.130    |
| 44242 | 2/1,115          | 0.179    |
| 44285 | 2/1,134          | 0.176    |
| 44722 | 2/3,192          | 0.063    |
| 44963 | 2/3,007          | 0.067    |
| 45069 | 2/965            | 0.207    |
| 45533 | 2/2,520          | 0.079    |
| 46373 | 2/973            | 0.205    |
| 46739 | 2/802            | 0.249    |
| 46795 | 2/1,274          | 0.157    |
| 47349 | 2/1,860          | 0.107    |
| 47765 | 2/1,198          | 0.167    |
| 47988 | 2/1,577          | 0.127    |
| 48017 | 2/1,345          | 0.149    |
| 44286 | 1/1,578          | 0.063    |
| 44288 | 1/1,152          | 0.087    |
| 44294 | 1/1,193          | 0.084    |
| 44935 | 1/1,882          | 0.053    |
| 45063 | 1/1,096          | 0.091    |
| 45086 | 1/1,974          | 0.051    |
| 45218 | 1/1,463          | 0.068    |
| 46761 | 1/877            | 0.114    |
| 47123 | 1/864            | 0.116    |
| 48012 | 1/2,081          | 0.048    |

No object has an `artifact_rate` above 0.53 %. The remaining 46
objects (59.0 %) have `artifact_rate = 0` — zero flags across their
entire classifiable window count.

---

## Section 4 — Threshold behavior

**Setup.** With `ENTROPY_WINDOW = 5`, the empirical distribution
from step 2 of the implementation is over at most 5 samples. The
maximum possible Shannon entropy of a 5-sample discrete distribution
(5 distinct destination cells, each occurring once) is
`ln(5) ≈ 1.6094` nats.

**Reachability of the threshold per cell.** The separator flags
when `obs_H > 2.0 × exp_H[cur_cell]`. The threshold is therefore
reachable only when `2.0 × exp_H[cur_cell] < ln(5)`, i.e.
`exp_H[cur_cell] < 0.8047`. Per-cell reachability:

| cell | exp_H  | 2·exp_H | reachable? | flags observed |
|-----:|-------:|--------:|------------|---------------:|
|  0   | 0.7690 | 1.5380  | yes        | 7              |
|  1   | 1.3724 | 2.7447  | no         | 0              |
|  2   | 1.3559 | 2.7119  | no         | 0              |
|  3   | 0.5352 | 1.0705  | yes        | 55             |
|  4   | 1.2849 | 2.5698  | no         | 0              |
|  5   | 1.0999 | 2.1998  | no         | 0              |
|  6   | 1.2707 | 2.5414  | no         | 0              |
|  7   | 0.8570 | 1.7140  | no         | 0              |
|  8   | 1.2204 | 2.4407  | no         | 0              |
|  9   | 1.0388 | 2.0777  | no         | 0              |
| 10   | 1.2606 | 2.5212  | no         | 0              |
| 11   | 1.0637 | 2.1273  | no         | 0              |
| 12   | 1.2161 | 2.4322  | no         | 0              |
| 13   | 0.0000 | 0.0000  | deterministic (fires if `obs_H > 0`) | 0 |
| 14   | 1.3488 | 2.6977  | no         | 0              |
| 15   | 1.3966 | 2.7931  | no         | 0              |
| 16   | 1.4352 | 2.8703  | no         | 0              |
| 17   | 0.8400 | 1.6801  | no         | 0              |
| 18   | 1.1537 | 2.3075  | no         | 0              |
| 19   | 0.6840 | 1.3681  | yes        | 8              |

- **Cells where the flag can fire under this configuration: 4 of 20**
  (cells 0, 3, 19, and the deterministic cell 13).
- **Cells where the flag cannot fire: 16 of 20**. For these the
  per-cell threshold `2 × exp_H` strictly exceeds the maximum
  possible `obs_H = ln(5) ≈ 1.6094` that a 5-sample empirical
  distribution can produce, so the condition `obs_H > 2 × exp_H`
  cannot be satisfied regardless of data.
- All 70 artifact flags observed in the 78-object run fell in cells
  3 (55), 19 (8), and 0 (7). Cell 13 never fired in the run.
- The 10 Markov failure-basin cells (1, 2, 4, 6, 7, 10, 11, 14, 15,
  18) all fall in the unreachable group under this threshold;
  every one has `exp_H > 0.85`.

**Hypothetical.** In a cell with `exp_H = 0.5` (near cell 3), the
separator would flag a rolling 5-window whose empirical destination
entropy exceeds `1.0` nats — achieved, for example, when 3 or more
of the 5 windows land in distinct destination cells with reasonably
balanced counts. In a cell with `exp_H = 1.0` (near cell 9), the
separator would require `obs_H > 2.0` nats, which is not achievable
from a 5-sample window (max `ln(5) ≈ 1.61`).
