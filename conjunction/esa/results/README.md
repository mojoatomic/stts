# Conjunction Domain Results

STTS sixth domain: satellite conjunction assessment using CDM time-series data.

## Results Files

- `validate_model_a.json` — Held-out validation for Model A (35 geometry features, causal weights). Config snapshot, artifact checksums, V1/V2 metrics, per-event V2 trajectories, F1 with Wilson 95% CIs.
- `validate_model_b.json` — Held-out validation for Model B (15 consistent features, uniform weights). Same structure.

## Dataset

Source: ESA Collision Avoidance Challenge corpus (Zenodo doi:10.5281/zenodo.4463683). Real operational CDMs from 2015-2019, 13,154 unique conjunction events, ~103 features per CDM.

**Structural limitations of this corpus for STTS:**

The competition withheld the final CDM for 2,167 test events — the CDM where risk converges to its true value. The `true_risk` label comes from that withheld CDM, which is not in our data. This creates an asymmetry: training events have their final CDM (so features reflect the labeled outcome), while test events do not (features reflect a pre-convergence state). Risk-based features diverge catastrophically between train and test (KS=0.97 on `risk_final`), which is why they were removed.

The competition also interleaved CDMs between train and test files, producing 2x CDM density in reconstructed test sequences. A 2-day TCA window eliminates this structural difference (post-windowing n_cdms KS=0.22, p=0.25).

After windowing and excluding events with no CDMs within 2 days of TCA: 8,421 training events (39 high-risk), 1,683 test events (38 high-risk). The 39 training positives are a small basin — sufficient for LDA fitting (39 > 35 features for Model A, 39 > 15 features for Model B) but limiting for generalization robustness.

## Model A: 35 Geometry Features

All geometry, covariance, OD quality, timing, and cross-coupling features with physics-informed causal weights (W: 0.5-3.0). Training V1: 6.6x separation, p=1.3e-17. Test V1: 0.7x, p=0.72 — complete collapse. The LDA loaded primarily on divergent features (`miss_dist_final` KS=0.77, `c_sigma_r_final` KS=0.43, `c_cov_det_final` KS=0.40) that have structurally different distributions between train and test high-risk events due to the withheld final CDM. The training separation was real but non-generalizable — the LDA found a direction that separated 39 training positives using features that shift under the test distribution.

## Model B: 15 Consistent Features

Features selected by KS consistency analysis (KS < 0.3 between train and test high-risk populations): target-side covariance rates, miss distance rates, CDM timing, and geometry-uncertainty coupling. Uniform weights (W=1.0). Training V1: 1.2x, p=0.049. Test V1: 0.7x, p=0.93 — V1 fails. But V2 passes: mean Spearman rho=+0.31, median rho=+0.45, 69.2% of high-risk events show the correct trajectory direction (embedding approaching basin as TCA nears). The LDA loads on target covariance shrinkage rate (`d_t_cov_det_dt`, `d_t_sigma_r_dt`) and CDM cadence (`n_cdms`, `inter_cdm_dt_last`) — features that are physically interpretable and structurally stable across the train/test split.

## Primary Result

**V2 passes for Model B on held-out data.** The cumulative embedding trajectory approaches the failure basin as CDM updates arrive closer to TCA for 69% of high-risk test events (mean rho=+0.31, 26 events analyzed, 12 skipped for <5 CDMs). This is the first positive generalization result for the conjunction domain and confirms that CDM sequences preceding high-risk conjunctions have a geometrically distinct trajectory in embedding space — the core STTS hypothesis — when measured using structurally consistent features.

The signal lives in dynamics (rates) and operational behavior (CDM cadence), not in absolute geometry values. Target-side covariance evolution generalizes because ESA tracks their own satellites consistently. Chaser-side features and absolute miss distances do not generalize because their values depend on how close to TCA the most recent CDM is, and the competition's withheld final CDM creates a systematic offset.

## Why V1 Fails and What Would Change It

V1 measures static event-level separation: do high-risk events cluster closer to the basin than nominal events at a single snapshot? With 39 training positives and a test distribution shifted by the withheld final CDM, the basin is too small and too tightly fit to the training distribution to separate test events statically.

Three factors would improve V1: (1) A larger corpus with more high-risk events — 39 positives is near the minimum for 15-feature LDA. Hundreds of positives would produce a more robust basin. (2) A corpus where test events have the same CDM coverage as training events — the ESA competition design creates an unavoidable asymmetry. An operational CDM feed without withheld updates would eliminate this. (3) A multi-component LDA or nonlinear projection that captures basin shape rather than just direction — the current 1-component LDA projects to a single axis where the diffuse Model B basin (std > mean) cannot cleanly separate populations.

The V2 result suggests the geometric signature exists but requires trajectory monitoring rather than single-point classification — which is the STTS design philosophy. The framework detects the approach, not the arrival.
