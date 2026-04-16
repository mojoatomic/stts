"""
For each held-out 450K replicate, walk the trajectory window-by-window.

At each window position, compute two scalars:

  curvature(w) : local Ollivier-Ricci at w, using the 320K healthy
                 manifold as reference. ZERO training on failure data.
                 The "accelerometer reading."

  distance(w)  : Euclidean distance from w to nearest 320K window.
                 The current STTS-style signal (without the feature
                 engineering — just raw sensor-space).

Saves per-window arrays of both signals for every replicate. Downstream
lead-time analysis (lead_time.py) compares them at matched FPR.

Usage:
    python -m geometry.probe
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from geometry.config import (
    ALPHA_OR,
    ARTIFACTS_DIR,
    HEALTHY_REPLICATES,
    HEALTHY_TEMP,
    K_FINAL,
    RESULTS_DIR,
    TEST_REPLICATES,
    TEST_TEMP,
    config_snapshot,
)
from geometry.data import (
    load_descriptor,
    load_healthy_manifold,
    load_test_trajectory,
    window_means,
)
from geometry.ricci import local_curvature_at_test_point


def compute_healthy_baseline(healthy_points: np.ndarray, k: int) -> dict:
    """Compute curvature and distance for every healthy 320K window, treating
    each in turn as a 'query' against the others. Used to calibrate the
    matched-FPR threshold.

    For each window w in 320K: remove it from the manifold, compute
    curvature and distance at w against the remaining healthy windows.
    Leave-one-out is expensive but principled — it gives the true
    distribution of signals at healthy points.

    For efficiency with 411 windows: instead of leave-one-out, we just
    compute curvature and distance using kNN excluding the window itself.
    Same result, much faster.
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(healthy_points)
    distances, indices = nbrs.kneighbors(healthy_points)

    # Distance excluding self (column 0 is self at distance 0)
    self_free_dist = distances[:, 1]  # nearest non-self

    # For curvature, use leave-one-out: the local probe of a healthy window
    # against the OTHER healthy windows.
    n = len(healthy_points)
    curvatures = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        others = healthy_points[mask]
        kappa = local_curvature_at_test_point(
            healthy_points[i], others, k, ALPHA_OR
        )
        curvatures[i] = kappa
        if (i + 1) % 50 == 0:
            print(f"    healthy baseline: {i + 1}/{n}")

    return {
        "curvature": curvatures,
        "distance": self_free_dist,
    }


def probe_trajectory(
    rep: int,
    healthy_points: np.ndarray,
    k: int,
) -> dict:
    """Probe one held-out 450K trajectory.

    Returns a dict with per-window curvature and distance arrays plus RUL.
    """
    print(f"  450K rep {rep}: loading and windowing...")
    traj = load_test_trajectory(rep)
    features = traj["features"]
    n_windows = len(features)

    print(f"    {n_windows} windows, failure_frame={traj['failure_frame']}")

    # Distances to nearest healthy neighbor
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(healthy_points)
    distances, _ = nbrs.kneighbors(features)
    distances = distances[:, 0]

    # Local curvature at each test window
    curvatures = np.zeros(n_windows)
    for i in range(n_windows):
        curvatures[i] = local_curvature_at_test_point(
            features[i], healthy_points, k, ALPHA_OR
        )
        if (i + 1) % 50 == 0:
            print(f"    curvature probe: {i + 1}/{n_windows}")

    return {
        "replicate": rep,
        "failure_frame": traj["failure_frame"],
        "end_frames": traj["end_frames"].tolist(),
        "rul": traj["rul"].tolist(),
        "curvature": curvatures.tolist(),
        "distance": distances.tolist(),
    }


def main():
    print(f"Curvature probe (k={K_FINAL}, alpha={ALPHA_OR})")
    print()

    print("Loading healthy manifold (320K)...")
    healthy_points, origin = load_healthy_manifold()
    print(f"  {len(healthy_points)} healthy windows")
    print()

    print("Computing healthy baseline (leave-one-out)...")
    baseline = compute_healthy_baseline(healthy_points, K_FINAL)
    print(f"  curvature: mean={baseline['curvature'].mean():.4f}, "
          f"std={baseline['curvature'].std():.4f}")
    print(f"  distance:  mean={baseline['distance'].mean():.4f}, "
          f"std={baseline['distance'].std():.4f}")
    print()

    # Save baseline
    np.save(ARTIFACTS_DIR / "baseline_curvature.npy", baseline["curvature"])
    np.save(ARTIFACTS_DIR / "baseline_distance.npy", baseline["distance"])

    # Probe each 450K replicate
    print("Probing held-out 450K replicates...")
    probes = []
    for rep in TEST_REPLICATES:
        probe = probe_trajectory(rep, healthy_points, K_FINAL)
        probes.append(probe)

    output = {
        "config": config_snapshot(),
        "k_final": K_FINAL,
        "baseline": {
            "n_windows": len(baseline["curvature"]),
            "curvature_mean": float(baseline["curvature"].mean()),
            "curvature_std": float(baseline["curvature"].std()),
            "distance_mean": float(baseline["distance"].mean()),
            "distance_std": float(baseline["distance"].std()),
        },
        "probes": probes,
    }

    out_path = RESULTS_DIR / "probe.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
