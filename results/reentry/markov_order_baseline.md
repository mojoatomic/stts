# STTS-Reentry — Markov-Order Baseline (𝒟_P10^reentry)

Concrete computation of the first-order vs second-order KL-divergence
aggregate on the reentry training data. Establishes the reference
value against which future-domain Markov applicability gate
comparisons will be calibrated. Read-only; no pipeline modifications.

Companion file: `results/reentry/markov_order_pair_kl.csv` (per-pair
counts, frequencies, and KL divergences — 400 rows, one per
(prev, current) pair).

---

## Section 1 — Methodology

**Cell discretization.** `MARKOV_K = 20` cells obtained from KMeans
on the 1-D LDA projection of training-window features
(`artifacts/reentry/kmeans.pkl`, fitted during `train.py`). A cell is
denoted `c ∈ {0, …, 19}`.

**Source of training cell sequences.** The Markov table artifact
(`artifacts/reentry/markov_table.npz`, md5
`bc7902e3073305bbba96d148f0c4a1bb`) stores the flat per-window
`train_proj` vector but not the per-trajectory ordering. The ordering
was reconstructed deterministically by replaying the pipeline:

1. `reentry.corpus.load_corpus()` → training satellite set
   (529 satellites).
2. `reentry.features.build_feature_matrix(..., stride=WINDOW_STRIDE_TRAIN)`
   with `WINDOW_STRIDE_TRAIN = 5`, producing per-window tuples with
   identifiers `"norad_id:window_start"`.
3. Exclude ambiguous windows (`label == -1`), leaving **144,878**
   training windows — byte-identical to the stored `train_proj`
   (max `|recon − stored| = 0.000e+00`).
4. `scaler.transform → × W → lda.transform → kmeans.predict` on each
   window, yielding the per-window cell assignment. Reconstructed
   `cell_counts` matches the stored array exactly.
5. Group windows by NORAD id; within each satellite sort by
   `window_start`; emit an ordered cell sequence.

**Transition constraint.** First-order pairs `(c_t, c_{t+1})` and
second-order triples `(c_{t−1}, c_t, c_{t+1})` are counted only when
successive windows are separated by exactly `WINDOW_STRIDE_TRAIN = 5`
(the same constraint train.py applies when building the Markov
table). In the training set, 0 pairs and 0 triples were skipped for
stride mismatch; the non-ambiguous training set is dense.

**Counts obtained.**
- First-order pairs: **144,349** (= 144,878 windows − 529
  satellites; one "tail" window per satellite has no successor).
  Normalized to the stored row-stochastic `transition_matrix`,
  reconstruction matches exactly (`max |ΔT| = 0.000e+00`).
- Second-order triples: **143,820** (= 144,878 − 2 × 529; two
  boundary windows per satellite have no predecessor or successor
  forming a triple).

**Distributions used.**
- `P(c_{t+1} | c_t) = N1[c_t, c_{t+1}] / Σ_{c'} N1[c_t, c']`.
- `P(c_{t+1} | c_{t−1}, c_t) = N2[c_{t−1}, c_t, c_{t+1}] / Σ_{c'} N2[c_{t−1}, c_t, c']`.
- Pair frequency `π(c_{t−1}, c_t) = pair_count(c_{t−1}, c_t) / Σ pair_count`,
  where `pair_count(a, b) = Σ_{c'} N2[a, b, c']`.

**Sufficient-support threshold.** Pairs with `pair_count < 10` are
excluded from the aggregate to avoid divergences dominated by
sparse-count noise, per the task specification. Excluded pairs are
preserved in the companion CSV with `excluded_reason = "N<10"`.

**Numerical conventions.** KL is computed in nats (`np.log`).
Terms where `P(c_{t+1} | c_{t−1}, c_t) = 0` are dropped
(convention `0 log 0 = 0`). A pair where
`P(c_{t+1} | c_t) = 0` while `P(c_{t+1} | c_{t−1}, c_t) > 0` would
produce infinite divergence — such pairs are counted separately and
excluded from the aggregate.

---

## Section 2 — Aggregate 𝒟_P10^reentry

| quantity                                          | value                 |
|---------------------------------------------------|-----------------------|
| `MARKOV_K`                                        | 20                    |
| Pair contexts enumerated                          | 400 (20 × 20)         |
| Triples used                                      | 143,820               |
| Pairs included (`pair_count ≥ 10`, finite KL)     | **104**               |
| Pairs excluded (`pair_count < 10`)                | 296                   |
| Pairs with infinite divergence                    | 0                     |
| π-mass on included pairs                          | 0.998373              |
| π-mass on excluded low-support pairs              | 0.001627              |
| π-mass on infinite-divergence pairs               | 0.000000              |
| **𝒟_P10^reentry** (aggregate, nats)                | **0.03263131**        |

The aggregate is the π-weighted sum over the 104 included pairs, with
π evaluated from the raw pair-frequency distribution. The 296
excluded low-support pairs collectively carry 0.163 % of the
π-mass, so the aggregate reflects transition statistics on
essentially the full training distribution.

---

## Section 3 — Top and bottom contributors

**Top 5 contributors** (`π × KL`, largest — pairs where the
empirical second-order distribution departs most from first-order
under the π weighting):

| prev | current | pair count | π          | KL (nats) | π·KL         |
|-----:|--------:|-----------:|-----------:|----------:|-------------:|
|  0   | 3       | 983        | 0.006835   | 0.3139    | 2.1455 × 10⁻³ |
|  3   | 17      | 5,970      | 0.041510   | 0.0290    | 1.2030 × 10⁻³ |
| 19   | 3       | 95         | 0.000661   | 1.7891    | 1.1818 × 10⁻³ |
| 12   | 0       | 51         | 0.000355   | 2.7814    | 9.8632 × 10⁻⁴ |
|  5   | 17      | 263        | 0.001829   | 0.5150    | 9.4185 × 10⁻⁴ |

**Bottom 5 contributors** (`π × KL`, smallest — pairs whose
second-order distribution most cleanly matches first-order):

| prev | current | pair count | π          | KL (nats) | π·KL         |
|-----:|--------:|-----------:|-----------:|----------:|-------------:|
|  4   | 7       | 12         | 0.000083   | 0.0083    | 6.9125 × 10⁻⁷ |
|  3   | 12      | 14         | 0.000097   | 0.1651    | 1.6067 × 10⁻⁵ |
| 16   | 16      | 15         | 0.000104   | 0.1553    | 1.6201 × 10⁻⁵ |
| 10   |  4      | 55         | 0.000382   | 0.0502    | 1.9212 × 10⁻⁵ |
| 17   | 12      | 11         | 0.000076   | 0.2980    | 2.2794 × 10⁻⁵ |

---

## Section 4 — Initial engineering tolerance Λ_P10

With the task-specified multiplier `k = 2`:

**Λ_P10 = 2 · 𝒟_P10^reentry = 0.06526261 nats**

This is the initial threshold value for future-domain Markov
applicability (Gate 2) comparisons. A domain whose own
`𝒟_P10 > Λ_P10` fails Gate 2, and STTS exits for that domain under
the current configuration.
