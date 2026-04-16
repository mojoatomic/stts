# Geometric analysis — protein curvature probe + field/geodesic decomposition

Two experiments on this branch, both related to testing whether STTS's
Euclidean formulation is capturing a deeper geometric signal.

## Scope change: PRONOSTIA bearing excluded

As of 2026-04-16, bearing (PRONOSTIA) is removed from the mechanism
discovery experiment and all downstream cross-domain analysis. Six
accelerated-life-test bearings from a lab rig under controlled
conditions is not a sample that can support cross-domain geometric
claims. `load_bearing_domain()` is retained in `run_experiment.py` as
inactive code; re-introduce only when a real-operating-conditions
bearing dataset is available.

Current cross-domain count: **8 domains** (protein, cmapss_fd001,
battery, orbital, ncmapss_ds02, ncmapss_ds03, reentry, solar).

---

## Experiment 1: Ollivier-Ricci curvature vs Euclidean distance on protein

### Hypothesis

Current STTS measures centroid drift in Euclidean feature space. If
this is an "echo" of a deeper geometric signal, local curvature at each
query point should fire earlier than Euclidean distance-to-healthy,
without needing any failure training.

### Setup

- Domain: mdCATH 5sicI00 protein (already on disk).
- Healthy manifold: 320K replicates 0-4 (411 windows). No failure data.
- Test: held-out 450K replicates 0-4 (481 windows each).
- Sensor space: raw 8-channel structural descriptors.
- Windows: 20 ns windows, per-channel mean → 8-D points.
- Graph: symmetric kNN. k = 15 selected by stability on 320K-only data
  BEFORE any test-data evaluation. Locked.
- Curvature: Ollivier-Ricci, α = 0.5. Implemented directly with
  `networkx` + `POT` (Wasserstein).
- Test-point probe: drop test window into the healthy graph as node 0;
  compute OR on edges incident to node 0 only; mean = local curvature.
- Distance: Euclidean distance from test window to nearest 320K window.
- Detection: threshold both at matched 5% FPR on healthy 320K baseline
  (leave-one-out for curvature). Alarm on first sustained crossing
  (5 consecutive above-threshold windows).

### Result

**Distance beats curvature.**

| | Curvature | Distance |
|---|---|---|
| Mean lead time | 175.6 frames | **184.6** |
| Fires first on | 0/5 replicates | **3/5** |
| Tied | 2/5 | 2/5 |
| Mean Spearman ρ vs RUL | -0.79 | **-0.81** |

### Why

`figures/curvature_vs_distance.png` shows the core issue directly:
across all 450K test windows, Ollivier-Ricci curvature is a
**deterministic saturating function of Euclidean distance**. Once
distance exceeds ~1, OR saturates at ≈ 0.52 (the α/2 fixed point) and
conveys no additional information. The probe collapses to a bounded
nonlinear distance proxy when the test point leaves the manifold.

### Takeaways

1. Supports the "flat regime" reading for protein: Euclidean distance
   captures everything OR on a kNN graph does, and more cleanly.
2. Does not rule out that a different curvature estimator (heat-kernel,
   sectional, intrinsic-dimensionality) could find structure distance
   misses. Rules out this specific standard published technique.
3. Bonus: a healthy-manifold-only detector with zero failure training
   achieves mean Spearman |ρ| = 0.81 vs RUL, ~185 frames lead at 5% FPR.
   STTS's reliance on failure-side training is useful for
   characterization but not required for detection.

---

## Experiment 2: Field/geodesic decomposition across cross-domain data

### Hypothesis

If damage is tensorial (as continuum damage mechanics suggests), the
7 mechanisms in the discovery experiment may decompose into:

- **Geodesic displacement** — how the system state moves through the
  (damaged) manifold. Proxied by M2 (centroid drift).
- **Curvature source** — how the damage tensor shapes the manifold.
  Proxied by M1 (participation ratio), M5 (correlation), M6 (subspace
  rotation).

Prediction: domains stratify by dominance. Some show pure geodesic
displacement (M2 wins), some show curvature-source dominance
(M1/M5/M6 wins), some show both.

### Setup

- Source: `experiments/mechanism_discovery/mechanism_discovery.json`
  (8 domains, bearing excluded).
- G = |mean ρ of M2 vs RUL| per domain
- K = max |mean ρ| over M1, M5, M6 per domain
- Reliability filter: domain's M2 signal must be sign-coherent
  (per-trajectory ρ values don't cross zero) AND n ≥ 5. Incoherence
  means the domain-mean is averaging over qualitatively different
  trajectory behaviors, not reliably estimating a single phenomenon.

### Result (reliable domains only)

| domain | n | G | K | G/K | classification |
|---|---|---|---|---|---|
| cmapss_fd001 | 100 | 0.979 | 0.674 | 1.45 | geodesic-dominant |
| battery | 10 | 0.848 | 0.311 | 2.73 | geodesic-dominant |
| ncmapss_ds03 | 15 | 0.827 | 0.935 | 0.88 | mixed, leans curvature |
| ncmapss_ds02 | 9 | 0.866 | 0.942 | 0.92 | mixed, leans curvature |

**Unreliable (cannot classify at domain level):** orbital (200 real
asteroids, n fine but M2 crosses zero), reentry (50 Starlink reentries,
same issue), solar (n=2), protein (n=3).

### Takeaways

1. **Zero clean curvature-source-dominant domains in reliable data.**
   My first pass incorrectly cited "solar CONFIRMED" based on n=2. With
   an honest reliability filter, that claim doesn't stand.
2. **Mixed N-CMAPSS domains** show both M2 and M6 strong simultaneously.
   Consistent with tensorial damage having multiple active components,
   but doesn't uniquely support the field/geodesic decomposition over
   "multiple mechanisms always contribute at different strengths."
3. **The decomposition hypothesis is not supported at the domain-mean
   level on reliable data.** It is not falsified either — we simply
   don't have a case that would cleanly falsify it.
4. **Orbital and reentry require per-trajectory analysis.** Their "M2
   crosses zero" is real object-level heterogeneity (different asteroids
   / different reentries have different physics), not bad data. A
   per-trajectory G/K scatter across all 200 orbital objects would be
   the proper test of whether tensorial damage decomposition manifests
   at the trajectory level within a large real dataset. That analysis
   is not done in this branch.

---

## Artifacts

```
artifacts/geometry/
├── graph_k{5,10,15,20,30}.pkl     # kNN graphs at each k
├── node_curv_k{5,10,15,20,30}.npy  # per-node scalar Ricci
├── edge_curv_k{5,10,15,20,30}.pkl  # per-edge Ricci
├── healthy_points.npy              # 411 × 8 healthy manifold features
├── healthy_origin.json             # (replicate, end_frame) per point
├── baseline_curvature.npy          # leave-one-out curvature on 320K
└── baseline_distance.npy           # nearest-neighbor distance on 320K

results/geometry/
├── k_sweep.json                    # stability-at-k on 320K only
├── probe.json                      # per-window curvature + distance per 450K replicate
├── lead_time.json                  # matched-FPR analysis
├── field_geodesic_test.json        # 8-domain decomposition scores
├── summary.md                      # this file
└── figures/
    ├── healthy_baseline.png
    ├── signal_trajectories.png
    ├── lead_time_comparison.png
    ├── curvature_vs_distance.png
    └── field_geodesic_decomposition.png
```

## Reproducibility

```bash
# Mechanism discovery (bearing excluded)
python experiments/mechanism_discovery/run_experiment.py
python experiments/mechanism_discovery/generate_figures.py

# Geometric analysis
python -m geometry.k_sweep
python -m geometry.probe
python -m geometry.lead_time
python -m geometry.figures
python -m geometry.field_geodesic_test
```

All random seeds fixed. No failure data used in the protein curvature
probe. Config snapshots embedded in every results JSON.
