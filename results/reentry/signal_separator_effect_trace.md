# STTS-Reentry — Signal Separator Effect Trace

Trace of the `artifact_flag` from classification through every
downstream consumer, to determine whether the flag influences
forward Monte-Carlo sampling or is recorded and ignored.
Read-only against code paths. No modifications.

---

## Section 1 — Consumers of the artifact flag

Classification sites:
- `reentry/energy_state_aggregate.py:108-121` — write (the 78-object
  run used this path).
- `reentry/energy_state.py:230-260` — write (canonical single-object).

Every subsequent read of `artifact_flags` (enumerated by
`grep -n artifact_flag` across the whole repo):

| file | line | what the consumer does | inside MC path? |
|------|-----:|------------------------|-----------------|
| `reentry/energy_state_aggregate.py` | 138 | writes the literal string `"artifact_flag"` as a CSV header. | no |
| `reentry/energy_state_aggregate.py` | 143 | writes `artifact_flags[t]` as the last CSV column of the per-object output. | no |
| `reentry/energy_state_aggregate.py` | 147 | computes `n_flags = int(artifact_flags.sum())` as a summary statistic. | no |
| `reentry/energy_state_aggregate.py` | 174 | stores `n_flags` in the per-object return dict as `"artifact_count"`. | no |
| `reentry/energy_state_aggregate.py` | 175 | stores `100 * n_flags / n_windows` as `"artifact_pct"`. | no |
| `reentry/energy_state_aggregate.py` | 273 | writes those two fields to `aggregate_summary.csv`. | no |
| `reentry/energy_state.py` | 103 | static metric semantics string documented in output JSON. | no |
| `reentry/energy_state.py` | 260 | `log.info(...)` progress log. | no |
| `reentry/energy_state.py` | 319 | CSV header string. | no |
| `reentry/energy_state.py` | 325 | writes `artifact_flags[t]` to output CSV. | no |
| `reentry/energy_state.py` | 366 | stores `int(artifact_flags.sum())` as `n_artifact_flags` in summary JSON. | no |
| `reentry/energy_state.py` | 413 | `flag_x = x[artifact_flags == 1]` — selects x-coordinates of flagged windows for the `vlines` plot (`axes[4]`). | no |
| `reentry/run_all.py` | 125 | prints `n_artifact_flags` in the end-of-run console summary. | no |

**No consumer of `artifact_flags` sits inside the forward Monte-Carlo
sampling loop in either file.** All consumers are output sinks
(CSV, JSON, log, plot) or simple counts of the same output sinks.

---

## Section 2 — Forward sampler's transition source

Location of the MC forward sampler (the path executed for the
78-object run):

- `reentry/energy_state_aggregate.py:123-132`.

Relevant excerpt:

```python
# P(failure, t+Δt)
P_forward = np.zeros(n_windows)
for t in range(n_windows):
    states = np.full(MC_N_SAMPLES, M2_cells[t], dtype=int)
    for step in range(MC_HORIZON):
        u = rng.random(MC_N_SAMPLES)
        cdfs = cum_trans[states]
        states = (cdfs < u[:, np.newaxis]).sum(axis=1)
        np.clip(states, 0, MARKOV_K - 1, out=states)
    P_forward[t] = np.mean(np.isin(states, failure_arr))
```

Key observations from the source:

1. The per-timestep loop `for t in range(n_windows)` iterates over
   every window index without consulting `artifact_flags[t]`.
2. The data structure consulted to draw a next-cell from the current
   cell is `cum_trans`, defined earlier in the file as
   `cum_trans = np.cumsum(transition_matrix, axis=1)`, where
   `transition_matrix` is loaded directly from the frozen artifact
   `artifacts/reentry/markov_table.npz` (md5
   `bc7902e3073305bbba96d148f0c4a1bb`). The matrix is read into
   `cum_trans` once at the start of the function and is not mutated
   during the MC loop.
3. `transition_matrix` was built by `reentry/train.py:211-233`
   from the full set of consecutive-stride training transitions,
   with no reference to artifact flags (the separator operates only
   on test-time sequences; artifact classification does not exist
   at training time). No filtering is applied to transitions before
   the table is serialised.

The canonical single-object path `reentry/energy_state.py:270-293`
uses exactly the same construction and has the same independence
from `artifact_flags`.

---

## Section 3 — Effective behaviour

The question posed (three possibilities):

- **(a)** Artifact-classified transitions are excluded from the
  Markov table used by the forward sampler. Flag has real
  downstream effect.
- **(b)** Artifact flag is logged to output but does not alter the
  Markov table or the forward sampling. Flag is informational only.
- **(c)** Some third behaviour — flag modifies a weight, triggers a
  fallback, or has domain-specific downstream consequences.

The trace supports **case (b)**:

1. The Markov table used by the forward sampler is the frozen
   `artifacts/reentry/markov_table.npz`, built at training time from
   all consecutive-stride training transitions with no artifact
   filtering.
2. The forward sampler reads only `cum_trans` (derived from that
   table) and `M2_cells[t]`; it does not read `artifact_flags[t]`
   anywhere in its loop body.
3. Every read of `artifact_flags` downstream of the classifier
   writes to an output sink (CSV / JSON / log / plot) or produces a
   summary count of such writes.

Stated in plain terms: when a window `t` is classified as artifact
during the 78-object run, the computation of `P_forward[t]` is
byte-identical to the computation that would have occurred if the
same window had been classified as signal. The two paths produce
the same 10,000-sample forward draw from the same starting cell
using the same transition matrix; they differ only in a single
integer in the per-object CSV's `artifact_flag` column and in the
plotted `axes[4]` vline overlay.

---

## Section 4 — Differential measurement

Per the task specification: if the flag has no downstream effect
(case b), the comparison between flagged-object and non-flagged-
object `max P` populations is not meaningful as a measure of flag
effect. Any observed difference between the 32 objects with
`artifact_count > 0` and the 46 objects with `artifact_count = 0`
would reflect underlying trajectory properties (cell occupancy,
window count, decay profile) rather than the flag itself, because
the flag does not enter the computation of `P_forward` for any
object. Step 4 is therefore skipped.
