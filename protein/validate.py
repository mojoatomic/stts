"""
Validate the STTS protein model on held-out 450K replicates.

Loads frozen artifacts from train.py — NEVER calls .fit().
Evaluates V1, V2, V3 on held-out test replicates (450K reps 3-4).
Generates figures: manifold projection, distance vs RUL, V3 loadings.

Usage:
    python protein/validate.py

Requires: model trained (python protein/train.py)
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from protein.config import (
    DESCRIPTORS_DIR,
    DEV_REPLICATES_320K,
    HEALTHY_TEMP,
    N_FEATURES,
    Q_FAILURE_THRESHOLD,
    RESULTS_DIR,
    RUL_CLIP,
    TARGET_DOMAINS,
    TEST_REPLICATES_450K,
    UNFOLDING_TEMP,
    V1_P_VALUE_THRESHOLD,
    V2_SPEARMAN_THRESHOLD,
    WINDOW_SIZE,
    WINDOW_STRIDE_EVAL,
    config_snapshot,
)
from protein.descriptors import compute_rul
from protein.features import build_feature_matrix, get_feature_class_indices
from protein.train import load_model


def load_test_set(domains: list[str] | None = None) -> tuple[dict, dict]:
    """Load descriptor arrays and compute RUL for the test set.

    Test set: 450K replicates 3-4 (held out from training).
    Also loads 320K replicates as healthy reference for V1.

    Returns:
        descriptors_dict: {traj_key: (n_frames, 8) array}
        rul_dict: {traj_key: (n_frames,) array}
    """
    if domains is None:
        domains = TARGET_DOMAINS

    descriptors_dict = {}
    rul_dict = {}

    for domain_id in domains:
        desc_dir = DESCRIPTORS_DIR / domain_id

        # 320K healthy replicates for V1 comparison
        for rep in DEV_REPLICATES_320K:
            key = f"{domain_id}_{HEALTHY_TEMP}K_rep{rep}"
            npy_path = desc_dir / f"{HEALTHY_TEMP}K_rep{rep}.npy"
            if not npy_path.exists():
                continue
            desc = np.load(npy_path)
            descriptors_dict[key] = desc
            rul_dict[key] = np.full(len(desc), np.inf)

        # 450K held-out replicates
        for rep in TEST_REPLICATES_450K:
            key = f"{domain_id}_{UNFOLDING_TEMP}K_rep{rep}"
            npy_path = desc_dir / f"{UNFOLDING_TEMP}K_rep{rep}.npy"
            if not npy_path.exists():
                print(f"  WARNING: {npy_path} not found, skipping")
                continue
            desc = np.load(npy_path)
            q_trajectory = desc[:, 3]
            rul, failure_frame = compute_rul(q_trajectory, Q_FAILURE_THRESHOLD)

            if failure_frame is None:
                print(f"  {key}: did not unfold — including as negative control")
                rul = np.full(len(desc), np.inf)

            rul = np.clip(rul, 0, RUL_CLIP)
            descriptors_dict[key] = desc
            rul_dict[key] = rul

    return descriptors_dict, rul_dict


def validate():
    """Evaluate the frozen model on held-out test set."""
    print("Loading frozen model...")
    model = load_model()
    scaler = model["scaler"]
    lda = model["lda"]
    basin = model["basin"]
    W = model["W"]
    epsilon = model["epsilon"]
    dist_to_basin = model["dist_to_basin"]

    print("Loading test set descriptors...")
    descriptors_dict, rul_dict = load_test_set()

    n_healthy = sum(1 for k in rul_dict if np.all(np.isinf(rul_dict[k])))
    n_test = sum(1 for k in rul_dict if not np.all(np.isinf(rul_dict[k])))
    print(f"\nTest set: {n_healthy} healthy trajectories, {n_test} unfolding trajectories")

    # -- Extract features (using frozen model) -----------------------
    print("Extracting windowed features...")
    X_all, y_all, ruls_all, ids_all = build_feature_matrix(
        descriptors_dict, rul_dict,
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE_EVAL,
    )

    # Exclude ambiguous
    valid_mask = y_all != -1
    X_test = X_all[valid_mask]
    y_test = y_all[valid_mask]
    ruls_test = ruls_all[valid_mask]

    print(f"  Test windows: {len(X_test)} "
          f"({(y_test == 1).sum()} precursor, {(y_test == 0).sum()} nominal)")

    # -- Apply frozen model (.transform only, NEVER .fit) ------------
    X_scaled = scaler.transform(X_test)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_w = X_scaled * W
    X_proj = lda.transform(X_w).ravel()

    test_dists = np.array([dist_to_basin(p) for p in X_proj])

    # -- V1: Statistical separation ----------------------------------
    nom_d = test_dists[y_test == 0]
    pre_d = test_dists[y_test == 1]

    if len(pre_d) > 0 and len(nom_d) > 0:
        v1_sep = float(np.median(nom_d) / (np.median(pre_d) + 1e-10))
        _, v1_p = mannwhitneyu(nom_d, pre_d, alternative="greater")
        v1_p = float(v1_p)
        v1_pass = v1_p < V1_P_VALUE_THRESHOLD
    else:
        v1_sep, v1_p, v1_pass = 0.0, 1.0, False

    print(f"\n  V1: {v1_sep:.1f}x separation (p={v1_p:.2e})")
    print(f"      {'PASS' if v1_pass else 'FAIL'} (threshold p<{V1_P_VALUE_THRESHOLD})")

    # -- V2: Monotonic approach (test 450K windows only) -------------
    mask_unfolding = ~np.isinf(ruls_test) & (ruls_test < RUL_CLIP)
    if mask_unfolding.sum() > 10:
        v2_rho, v2_p = spearmanr(ruls_test[mask_unfolding], test_dists[mask_unfolding])
        v2_rho, v2_p = float(v2_rho), float(v2_p)
    else:
        v2_rho, v2_p = 0.0, 1.0
    v2_pass = v2_rho > V2_SPEARMAN_THRESHOLD and v2_p < 0.05

    print(f"  V2: rho={v2_rho:.3f} (p={v2_p:.2e})")
    print(f"      {'PASS' if v2_pass else 'FAIL'} (threshold rho>{V2_SPEARMAN_THRESHOLD})")

    # -- V3: Feature class ablation ----------------------------------
    feature_class_indices = get_feature_class_indices()

    def fit_and_query(features, weights):
        """Apply frozen scaler + LDA with given weights, return distances."""
        s = scaler.transform(features)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        sw = s * weights
        proj = lda.transform(sw).ravel()
        return np.array([dist_to_basin(p) for p in proj])

    from pipeline.evaluation import verify_v3_ablation
    v3_results = verify_v3_ablation(
        X_test, W, feature_class_indices, fit_and_query
    )

    print("\n  V3: Feature class ablation impact:")
    for cls_name, impact in sorted(v3_results.items(), key=lambda x: -x[1]):
        print(f"    {cls_name:>15s}: {impact:+.4f}")

    # -- Detection metrics -------------------------------------------
    preds = (test_dists < epsilon).astype(int)
    test_tp = int(((preds == 1) & (y_test == 1)).sum())
    test_fp = int(((preds == 1) & (y_test == 0)).sum())
    test_fn = int(((preds == 0) & (y_test == 1)).sum())
    test_tn = int(((preds == 0) & (y_test == 0)).sum())
    test_prec = test_tp / max(1, test_tp + test_fp)
    test_rec = test_tp / max(1, test_tp + test_fn)
    test_f1 = 2 * test_prec * test_rec / max(1e-10, test_prec + test_rec)

    print(f"\n  Test: TP={test_tp} FP={test_fp} FN={test_fn} TN={test_tn}")
    print(f"  Test: Prec={test_prec:.3f} Rec={test_rec:.3f} F1={test_f1:.3f}")

    # -- Save results ------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "config": config_snapshot(),
        "artifact_checksums": model["meta"]["artifact_checksums"],
        "test_set": {
            "n_healthy_trajectories": n_healthy,
            "n_unfolding_trajectories": n_test,
            "n_windows": len(X_test),
            "n_precursor": int((y_test == 1).sum()),
            "n_nominal": int((y_test == 0).sum()),
        },
        "v1": {
            "separation": v1_sep,
            "p_value": v1_p,
            "passed": v1_pass,
            "median_precursor": float(np.median(pre_d)) if len(pre_d) > 0 else None,
            "median_nominal": float(np.median(nom_d)) if len(nom_d) > 0 else None,
        },
        "v2": {
            "spearman_rho": v2_rho,
            "p_value": v2_p,
            "passed": v2_pass,
        },
        "v3": v3_results,
        "detection": {
            "epsilon": epsilon,
            "test_f1": test_f1,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "tp": test_tp,
            "fp": test_fp,
            "fn": test_fn,
            "tn": test_tn,
        },
    }

    results_path = RESULTS_DIR / "validate.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # -- Generate figures --------------------------------------------
    _generate_figures(
        X_proj, y_test, test_dists, ruls_test,
        nom_d, pre_d, v3_results,
    )

    return results


def _generate_figures(
    projections, labels, distances, ruls,
    nom_dists, pre_dists, v3_results,
):
    """Generate validation figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: LDA manifold projection ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    nom_mask = labels == 0
    pre_mask = labels == 1

    ax.hist(projections[nom_mask], bins=50, alpha=0.6, label="320K (folded)",
            color="steelblue", density=True)
    ax.hist(projections[pre_mask], bins=50, alpha=0.6, label="450K (unfolding)",
            color="firebrick", density=True)
    ax.set_xlabel("LDA projection")
    ax.set_ylabel("Density")
    ax.set_title("STTS Manifold: Protein Folded vs Unfolding")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "protein_manifold.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved protein_manifold.png")

    # --- Figure 2: k-NN distance vs RUL ---
    mask_finite = ~np.isinf(ruls) & (ruls < 500)
    if mask_finite.sum() > 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.scatter(ruls[mask_finite], distances[mask_finite],
                   alpha=0.3, s=8, color="firebrick")
        ax.set_xlabel("RUL (frames / ns)")
        ax.set_ylabel("k-NN distance to failure basin")
        ax.set_title("STTS Distance vs RUL — Protein Unfolding")

        # Add Spearman annotation
        if mask_finite.sum() > 10:
            rho, p = spearmanr(ruls[mask_finite], distances[mask_finite])
            ax.annotate(f"Spearman ρ = {rho:.3f}\np = {p:.2e}",
                        xy=(0.95, 0.95), xycoords="axes fraction",
                        ha="right", va="top", fontsize=10,
                        bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "protein_distance_vs_rul.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved protein_distance_vs_rul.png")

    # --- Figure 3: V3 feature class importance ---
    if v3_results:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        classes = list(v3_results.keys())
        impacts = [v3_results[c] for c in classes]
        colors = ["firebrick" if v > 0 else "steelblue" for v in impacts]
        ax.barh(classes, impacts, color=colors)
        ax.set_xlabel("Relative distance change when ablated")
        ax.set_title("V3: Feature Class Importance — Protein")
        ax.axvline(x=0, color="black", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "protein_v3_loadings.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved protein_v3_loadings.png")


if __name__ == "__main__":
    validate()
