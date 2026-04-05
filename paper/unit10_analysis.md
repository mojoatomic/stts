# Unit 10 Analysis — DS03 Test Engine with Inverted V2

## Observation

Unit 10 in the DS03 test set exhibits an **inverted V2** (Spearman rho = -0.628, p = 3.19e-05):
its k-NN distance to the failure basin *increases* as RUL decreases, opposite to the expected
monotonic approach behavior seen in all other engines.

## Comparison with Other DS03 Test Engines

| Unit | Cycles | V2 rho | Flight Classes | Initial h_s |
|------|--------|--------|----------------|-------------|
| 10 ** | 66 | -0.628 | {3: 66} | 1 |
| 11 | 59 | +0.799 | {3: 59} | 1 |
| 12 | 93 | +0.663 | {1: 93} | 1 |
| 13 | 77 | +0.723 | {3: 77} | 1 |
| 14 | 76 | +0.792 | {1: 76} | 1 |
| 15 | 67 | +0.803 | {2: 67} | 1 |

## Flight Class Representation

Dev set flight class distribution (by cycle count):
{
  "1": 249,
  "2": 280,
  "3": 134
}

Unit 10's flight class distribution:
{
  "3": 66
}

Unit 10's dominant flight class is 3, which has 134/663 cycles (20.2%) in the dev set.

## Cruise-Phase Data Fraction

| Unit | Raw Cycles | After Aggregation | Retention |
|------|-----------|-------------------|-----------|
| 10 ** | 66 | 66 | 100.0% |
| 11 | 59 | 59 | 100.0% |
| 12 | 93 | 93 | 100.0% |
| 13 | 77 | 77 | 100.0% |
| 14 | 76 | 76 | 100.0% |
| 15 | 67 | 67 | 100.0% |

## Hypothesis

The inverted V2 indicates that Unit 10's trajectory in the LDA manifold moves *away* from the
failure basin as it approaches failure, rather than toward it. Possible explanations:

1. **Flight profile mismatch**: If Unit 10 operates in a flight regime that is underrepresented
   in the dev set, the regime normalization may not fully remove operating-point effects. The
   LDA, trained primarily on other flight profiles, may project Unit 10's trajectory into a
   region of the manifold where the basin distance metric is not meaningful.

2. **Anomalous degradation trajectory**: DS03's failure mode (LPT + HPT efficiency and flow)
   may manifest differently in Unit 10 due to its specific initial health state or operating
   history. The degradation signature may follow a non-standard geometric path in feature space.

3. **Regime normalization artifact**: With only 6 regime clusters, some operating points may be
   poorly represented. If Unit 10's cruise conditions consistently map to a cluster boundary,
   the normalization could introduce systematic bias.

## Conclusion

Unit 10's inverted V2 is a **real limitation**, not a data error. It represents a case where the
single-LDA manifold projection does not capture the degradation geometry for this specific engine.
The pooled V2 (0.954) masks this per-engine variation. The paper should report per-engine V2
breakdowns and acknowledge that 1/6 test engines shows inverted behavior. This is honest and
consistent with the framework's design — STTS does not claim universal monotonicity, only that
the geometric structure is *generally* preserved. V1 separation (104.2x) remains valid even for
Unit 10's final position.
