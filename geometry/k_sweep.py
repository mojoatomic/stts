"""
Sweep k for the kNN graph, pick the k that gives the most stable
Ricci signal on 320K-only data.

Stability criterion: for each k, measure the mean absolute deviation of
Ricci values across the 5 healthy replicates (each replicate's windows
should give similar curvature if the manifold is being estimated
consistently). Lower MAD = more stable = better k.

IMPORTANT: this runs entirely on 320K (healthy) data. No test data touched.
That's the whole point — pick k before the held-out test, not after.

Usage:
    python -m geometry.k_sweep
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from geometry.config import (
    ALPHA_OR,
    ARTIFACTS_DIR,
    HEALTHY_REPLICATES,
    HEALTHY_TEMP,
    K_SWEEP,
    RESULTS_DIR,
    config_snapshot,
)
from geometry.data import load_descriptor, load_healthy_manifold, window_means
from geometry.ricci import (
    build_knn_graph,
    compute_all_edge_curvatures,
    scalar_curvature_per_node,
)


def stability_at_k(k: int, points: np.ndarray, origin: list) -> dict:
    """Build kNN graph at given k, compute per-node Ricci, measure stability
    across healthy replicates.

    origin[i] = (replicate, end_frame) for points[i].

    Stability metric: std of per-node Ricci within each replicate,
    averaged across replicates. Low std = stable (consistent curvature
    within a replicate). Also report between-replicate variation of
    the replicate-mean curvature — this should be LOW too if the
    manifold is being estimated consistently.
    """
    print(f"  k={k}: building graph on {len(points)} nodes...")
    G = build_knn_graph(points, k)
    print(f"    graph has {G.number_of_edges()} edges")

    print(f"    computing Ollivier-Ricci on all edges (alpha={ALPHA_OR})...")
    edge_curv = compute_all_edge_curvatures(G, ALPHA_OR)
    node_curv = scalar_curvature_per_node(G, edge_curv)

    reps = sorted(set(r for (r, _) in origin))
    rep_means = {}
    rep_stds = {}
    for rep in reps:
        mask = np.array([r == rep for (r, _) in origin])
        rep_vals = node_curv[mask]
        rep_means[rep] = float(np.mean(rep_vals))
        rep_stds[rep] = float(np.std(rep_vals))

    between_rep_std = float(np.std(list(rep_means.values())))
    within_rep_std = float(np.mean(list(rep_stds.values())))

    return {
        "k": k,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "global_mean_kappa": float(np.mean(node_curv)),
        "global_std_kappa": float(np.std(node_curv)),
        "between_replicate_std": between_rep_std,
        "within_replicate_std": within_rep_std,
        "replicate_means": rep_means,
        "replicate_stds": rep_stds,
        # Combined stability score: want both within and between variation small,
        # but the core claim is that a single replicate's windows should have
        # similar curvature (within-stability), so weight that higher.
        "stability_score": within_rep_std + 0.5 * between_rep_std,
    }, G, node_curv, edge_curv


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading healthy manifold (320K, all replicates)...")
    points, origin = load_healthy_manifold()
    print(f"  {len(points)} windows across {len(HEALTHY_REPLICATES)} replicates")

    sweep_results = []
    for k in K_SWEEP:
        summary, G, node_curv, edge_curv = stability_at_k(k, points, origin)
        sweep_results.append(summary)

        # Save per-k artifacts
        with open(ARTIFACTS_DIR / f"graph_k{k}.pkl", "wb") as f:
            pickle.dump(G, f)
        np.save(ARTIFACTS_DIR / f"node_curv_k{k}.npy", node_curv)
        with open(ARTIFACTS_DIR / f"edge_curv_k{k}.pkl", "wb") as f:
            pickle.dump(edge_curv, f)

        print(f"    mean kappa={summary['global_mean_kappa']:+.4f}, "
              f"within-rep std={summary['within_replicate_std']:.4f}, "
              f"between-rep std={summary['between_replicate_std']:.4f}, "
              f"score={summary['stability_score']:.4f}")

    # Pick the best k
    best = min(sweep_results, key=lambda r: r["stability_score"])
    print()
    print(f"Best k by stability: k={best['k']}")
    print(f"  within-replicate std:  {best['within_replicate_std']:.4f}")
    print(f"  between-replicate std: {best['between_replicate_std']:.4f}")

    # Save healthy manifold points and origin
    np.save(ARTIFACTS_DIR / "healthy_points.npy", points)
    with open(ARTIFACTS_DIR / "healthy_origin.json", "w") as f:
        json.dump([[r, e] for (r, e) in origin], f)

    # Save sweep summary
    output = {
        "config": config_snapshot(),
        "sweep": sweep_results,
        "chosen_k": best["k"],
        "rationale": (
            f"k={best['k']} gave lowest combined within+between replicate "
            "std on 320K-only data. Selected BEFORE any test data evaluation."
        ),
    }
    with open(RESULTS_DIR / "k_sweep.json", "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Saved artifacts to {ARTIFACTS_DIR}/")
    print(f"Saved sweep summary to {RESULTS_DIR / 'k_sweep.json'}")


if __name__ == "__main__":
    main()
