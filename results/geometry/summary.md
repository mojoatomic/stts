# Curvature Experiment — Protein Unfolding

## Hypothesis under test

Current STTS measures centroid drift in Euclidean feature space. If this is an
"echo" of a deeper geometric signal, then the true signal should be the local
curvature of the data manifold at each point in the trajectory — an
"accelerometer reading" that requires no training on failure data.

**Prediction:** At matched false-positive rate on a healthy baseline, a local
Ollivier-Ricci curvature probe should detect degradation earlier than
Euclidean distance-to-nearest-healthy — and should do so without ever seeing
a failure example during training.

## Setup

- **Domain:** mdCATH 5sicI00 protein (already on disk from protein domain work).
- **Healthy manifold:** 320K replicates 0-4 only. No failure training.
- **Test:** held-out 450K replicates 0-4.
- **Sensor space:** raw 8-channel structural descriptors (RMSD, Rg, SASA, Q,
  helix/sheet/coil fractions, end-to-end distance).
- **Window:** 20 ns windows, per-channel mean → 8-D points.
- **Graph:** symmetric kNN on healthy windows, k selected from sweep
  [5, 10, 15, 20, 30] by **stability on 320K-only** data (lower combined
  within- and between-replicate std of per-node scalar Ricci).
  **Chosen k = 15.** Locked before any test-data evaluation.
- **Curvature:** Ollivier-Ricci with alpha = 0.5, implemented directly with
  `networkx` + `POT` (Wasserstein). Per-node scalar = mean over incident edges.
- **Test-point probe:** drop the test window into the healthy graph as node 0;
  compute OR on edges incident to node 0 only. Return mean as the local
  curvature at that test point.
- **Distance:** Euclidean distance from test window to nearest 320K window.
- **Detection:** threshold both at matched 5% FPR on healthy 320K baseline
  (leave-one-out for curvature). Alarm fires on first sustained crossing
  (5 consecutive above-threshold windows).

## Result

**Distance beats curvature.**

| | Curvature | Distance |
|---|---|---|
| Mean lead time (frames / ns) | 175.6 | **184.6** |
| Fires first on | 0/5 replicates | **3/5 replicates** |
| Tied | 2/5 | 2/5 |
| Mean Spearman rho vs RUL | -0.79 | **-0.81** |

On the primary metric (matched-FPR lead time), distance wins by ~9 frames on
average. Curvature never fires first on any held-out replicate. Spearman
correlation with RUL is also slightly stronger for distance.

## Why: curvature saturates

The curvature-vs-distance scatter plot
(`figures/curvature_vs_distance.png`) shows the fundamental issue: across all
450K test windows, curvature is a **deterministic saturating function of
distance**. Once Euclidean distance exceeds ~1, Ollivier-Ricci saturates at
≈ 0.52 (the α = 0.5 fixed point) and no longer conveys new information.

This is an artifact of the estimator, not of the data. When a test point is
far from the manifold, its local neighborhood becomes a star from the test
point to distant healthy nodes of roughly uniform distance. In that limit,
W_1 / edge_weight → 0.5 and κ → 0.5, regardless of how far the test point
actually is. The estimator collapses to a bounded distance proxy.

Within the manifold (distance < ~0.5), the curvature does carry information,
and the healthy baseline distribution (mean κ = 0.14, σ = 0.09) is
physically reasonable: a folded-state basin with positive curvature
(convergent geodesics). But this in-manifold regime is exactly where all
the healthy windows live, and where degradation detection is NOT needed.

## What this means for the Euclidean-vs-Riemannian question

The "flat regime" hypothesis is consistent with this result. For protein
unfolding in the observed range of feature space, Euclidean distance
captures everything Ollivier-Ricci on a kNN graph captures, and more
cleanly (analog magnitude post-transition; no saturation).

This does NOT rule out that a different curvature estimator would find
structure Euclidean distance misses. Candidates:
- Scalar curvature via heat-kernel / Laplace-Beltrami methods.
- Sectional curvature from geodesic triangles.
- Intrinsic dimensionality variation along the trajectory.

It DOES demonstrate that the most standard published technique
(Ollivier-Ricci on kNN graph) does not operationalize the
"accelerometer" intuition on this data. The burden of proof now falls on
finding a different estimator that does.

It also independently confirms, without relying on the STTS training
pipeline at all, that a healthy-manifold-only detector with no failure
training achieves strong early detection on held-out trajectories — mean
Spearman |rho| ≈ 0.81 between Euclidean distance and RUL, mean lead of
~185 frames before failure at 5% FPR. That's a useful finding on its own:
the STTS framework's need for failure-side training is not fundamental
for detection; it is useful for characterization but not detection.

## Artifacts

```
artifacts/geometry/
├── graph_k{5,10,15,20,30}.pkl       # kNN graphs at each k
├── node_curv_k{5,10,15,20,30}.npy    # per-node scalar Ricci at each k
├── edge_curv_k{5,10,15,20,30}.pkl    # per-edge Ricci values
├── healthy_points.npy                # 411 × 8 healthy manifold features
├── healthy_origin.json               # (replicate, end_frame) per point
├── baseline_curvature.npy            # leave-one-out curvature for each 320K window
└── baseline_distance.npy             # nearest-neighbor distance for each 320K window

results/geometry/
├── k_sweep.json                      # stability-at-k table
├── probe.json                        # per-window curvature + distance for each 450K replicate
├── lead_time.json                    # matched-FPR lead time comparison
├── summary.md                        # this file
└── figures/
    ├── healthy_baseline.png
    ├── signal_trajectories.png
    ├── lead_time_comparison.png
    └── curvature_vs_distance.png
```

## Reproducibility

```bash
python -m geometry.k_sweep     # build graphs, pick k (deterministic)
python -m geometry.probe       # curvature + distance on all test replicates
python -m geometry.lead_time   # matched-FPR analysis
python -m geometry.figures     # plots
```

All random seeds fixed. No failure data used in any step. Config snapshot
embedded in every results JSON.
