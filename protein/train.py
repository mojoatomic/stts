"""
Train the STTS protein unfolding model.

This is the ONLY script that calls .fit() on any preprocessor or model.
All downstream scripts (validate.py) call load_model() which loads
serialized artifacts — never refits.

Dev set composition:
  - 320K all 5 replicates (healthy baseline, label=0)
  - 450K replicates 0-2 (unfolding trajectories with RUL, label=1 near failure)

Usage:
    python protein/train.py

Requires: descriptors already extracted (python protein/descriptors.py)
"""

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from protein.config import (
    ARTIFACTS_DIR,
    BASIN_SIGMA_CUTOFF,
    DESCRIPTORS_DIR,
    DEV_REPLICATES_320K,
    DEV_REPLICATES_450K,
    HEALTHY_TEMP,
    K_NEIGHBORS,
    LDA_COMPONENTS,
    N_REPLICATES,
    Q_FAILURE_THRESHOLD,
    RUL_CLIP,
    TARGET_DOMAINS,
    TEMPERATURES,
    UNFOLDING_TEMP,
    V1_P_VALUE_THRESHOLD,
    V2_SPEARMAN_THRESHOLD,
    WINDOW_SIZE,
    WINDOW_STRIDE_TRAIN,
    build_weight_vector,
    config_snapshot,
)
from protein.descriptors import compute_rul
from protein.features import build_feature_matrix


# -- Paths ---------------------------------------------------------

SCALER_FILE = ARTIFACTS_DIR / "scaler.pkl"
LDA_FILE = ARTIFACTS_DIR / "lda.pkl"
BASIN_FILE = ARTIFACTS_DIR / "basin.npy"
MODEL_META_FILE = ARTIFACTS_DIR / "model_meta.json"


def md5_file(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_dev_set(domains: list[str] | None = None) -> tuple[dict, dict]:
    """Load descriptor arrays and compute RUL for the dev set.

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

        # 320K healthy replicates — all label=0, RUL=inf
        for rep in DEV_REPLICATES_320K:
            key = f"{domain_id}_{HEALTHY_TEMP}K_rep{rep}"
            npy_path = desc_dir / f"{HEALTHY_TEMP}K_rep{rep}.npy"
            if not npy_path.exists():
                print(f"  WARNING: {npy_path} not found, skipping")
                continue
            desc = np.load(npy_path)
            descriptors_dict[key] = desc
            rul_dict[key] = np.full(len(desc), np.inf)

        # 450K unfolding replicates (dev split only)
        for rep in DEV_REPLICATES_450K:
            key = f"{domain_id}_{UNFOLDING_TEMP}K_rep{rep}"
            npy_path = desc_dir / f"{UNFOLDING_TEMP}K_rep{rep}.npy"
            if not npy_path.exists():
                print(f"  WARNING: {npy_path} not found, skipping")
                continue
            desc = np.load(npy_path)
            q_trajectory = desc[:, 3]  # native contact fraction
            rul, failure_frame = compute_rul(q_trajectory, Q_FAILURE_THRESHOLD)

            if failure_frame is None:
                print(f"  {key}: Q never dropped below {Q_FAILURE_THRESHOLD} — "
                      f"excluding from run-to-failure set")
                continue

            # Clip RUL
            rul = np.clip(rul, 0, RUL_CLIP)
            descriptors_dict[key] = desc
            rul_dict[key] = rul
            print(f"  {key}: failure at frame {failure_frame}, "
                  f"Q_final={q_trajectory[-1]:.3f}")

    return descriptors_dict, rul_dict


def train():
    """Train model on dev set. Serialize all artifacts."""
    print("Loading dev set descriptors...")
    descriptors_dict, rul_dict = load_dev_set()

    n_healthy = sum(1 for k in rul_dict if np.all(np.isinf(rul_dict[k])))
    n_unfolding = sum(1 for k in rul_dict if not np.all(np.isinf(rul_dict[k])))
    print(f"\nDev set: {n_healthy} healthy trajectories, "
          f"{n_unfolding} unfolding trajectories")

    if n_unfolding == 0:
        raise ValueError(
            "No unfolding trajectories found. Check that 450K replicates "
            "unfold (Q < 0.3) for the selected domains."
        )

    # -- Extract features --------------------------------------------
    print("\nExtracting windowed features...")
    X_all, y_all, ruls_all, ids_all = build_feature_matrix(
        descriptors_dict, rul_dict,
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE_TRAIN,
    )

    print(f"  Total windows: {len(X_all)}")
    print(f"    Precursor (y=1):  {(y_all == 1).sum()}")
    print(f"    Nominal (y=0):    {(y_all == 0).sum()}")
    print(f"    Ambiguous (y=-1): {(y_all == -1).sum()} (excluded from training)")

    # Exclude ambiguous windows
    train_mask = y_all != -1
    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    ruls_train = ruls_all[train_mask]

    assert (y_train == 1).sum() >= 5, \
        f"Insufficient precursor windows: {(y_train == 1).sum()}"
    assert (y_train == 0).sum() >= 5, \
        f"Insufficient nominal windows: {(y_train == 0).sum()}"

    print(f"\n  Training windows: {len(X_train)} "
          f"({(y_train == 1).sum()} precursor, {(y_train == 0).sum()} nominal)")

    # -- Stage W: Causal weighting -----------------------------------
    W = build_weight_vector()

    # -- Fit scaler — THIS IS THE ONLY PLACE scaler.fit() IS CALLED
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    X_w = X_scaled * W

    # -- Stage M: Fit LDA — THIS IS THE ONLY PLACE lda.fit() IS CALLED
    lda = LinearDiscriminantAnalysis(
        n_components=LDA_COMPONENTS, solver="svd"
    )
    lda.fit(X_w, y_train)
    X_proj = lda.transform(X_w).ravel()

    # -- Failure basin (tightened) -----------------------------------
    precursor_proj = X_proj[y_train == 1]
    basin_threshold = np.mean(precursor_proj) - BASIN_SIGMA_CUTOFF * np.std(precursor_proj)
    basin = precursor_proj[precursor_proj >= basin_threshold]
    print(f"\n  Basin: {len(basin)}/{len(precursor_proj)} precursor windows "
          f"(threshold={basin_threshold:.4f}, mean={np.mean(basin):.4f})")

    # -- Training distances ------------------------------------------
    k = K_NEIGHBORS

    def _dist_to_basin(p):
        dists = np.abs(basin - p)
        return np.mean(np.sort(dists)[:min(k, len(basin))])

    train_dists = np.array([_dist_to_basin(p) for p in X_proj])

    # -- V1: Statistical separation ----------------------------------
    nom_d = train_dists[y_train == 0]
    pre_d = train_dists[y_train == 1]
    v1_sep = float(np.median(nom_d) / (np.median(pre_d) + 1e-10))
    _, v1_p = mannwhitneyu(nom_d, pre_d, alternative="greater")
    v1_p = float(v1_p)

    # -- V2: Monotonic approach (on 450K windows only) ---------------
    mask_unfolding = ~np.isinf(ruls_train) & (ruls_train < RUL_CLIP)
    if mask_unfolding.sum() > 10:
        v2_rho, v2_p = spearmanr(ruls_train[mask_unfolding], train_dists[mask_unfolding])
        v2_rho, v2_p = float(v2_rho), float(v2_p)
    else:
        v2_rho, v2_p = 0.0, 1.0

    # -- Calibrate epsilon -------------------------------------------
    EPSILON_PERCENTILE = 5
    best_eps = float(np.percentile(nom_d, EPSILON_PERCENTILE))

    # Training detection metrics
    preds = (train_dists < best_eps).astype(int)
    train_tp = int(((preds == 1) & (y_train == 1)).sum())
    train_fp = int(((preds == 1) & (y_train == 0)).sum())
    train_fn = int(((preds == 0) & (y_train == 1)).sum())
    train_tn = int(((preds == 0) & (y_train == 0)).sum())
    train_prec = train_tp / max(1, train_tp + train_fp)
    train_rec = train_tp / max(1, train_tp + train_fn)
    train_f1 = 2 * train_prec * train_rec / max(1e-10, train_prec + train_rec)

    print(f"\n  V1: {v1_sep:.1f}x separation (p={v1_p:.2e})")
    print(f"      {'PASS' if v1_p < V1_P_VALUE_THRESHOLD else 'FAIL'} "
          f"(threshold p<{V1_P_VALUE_THRESHOLD})")
    print(f"  V2: rho={v2_rho:.3f} (p={v2_p:.2e})")
    print(f"      {'PASS' if v2_rho > V2_SPEARMAN_THRESHOLD else 'FAIL'} "
          f"(threshold rho>{V2_SPEARMAN_THRESHOLD})")
    print(f"  epsilon={best_eps:.4f} (p{EPSILON_PERCENTILE} of nominal distances)")
    print(f"  Train: TP={train_tp} FP={train_fp} FN={train_fn} TN={train_tn}")
    print(f"  Train: Prec={train_prec:.3f} Rec={train_rec:.3f} F1={train_f1:.3f}")

    # -- Serialize artifacts -----------------------------------------
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

    with open(LDA_FILE, "wb") as f:
        pickle.dump(lda, f)

    np.save(BASIN_FILE, basin)

    meta = {
        "config": config_snapshot(),
        "training": {
            "n_healthy_trajectories": n_healthy,
            "n_unfolding_trajectories": n_unfolding,
            "n_windows_total": len(X_all),
            "n_windows_train": len(X_train),
            "n_precursor": int((y_train == 1).sum()),
            "n_nominal": int((y_train == 0).sum()),
            "n_ambiguous": int((y_all == -1).sum()),
        },
        "metrics": {
            "v1_separation": v1_sep,
            "v1_p": v1_p,
            "v1_pass": v1_p < V1_P_VALUE_THRESHOLD,
            "v2_rho": v2_rho,
            "v2_p": v2_p,
            "v2_pass": v2_rho > V2_SPEARMAN_THRESHOLD,
            "train_f1": train_f1,
            "train_precision": train_prec,
            "train_recall": train_rec,
            "epsilon": best_eps,
        },
        "artifact_checksums": {
            "scaler": md5_file(SCALER_FILE),
            "lda": md5_file(LDA_FILE),
            "basin": md5_file(BASIN_FILE),
        },
    }

    with open(MODEL_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Artifacts saved to {ARTIFACTS_DIR}/")
    return meta


def load_model() -> dict:
    """Load the frozen trained model. NEVER calls .fit().

    Returns dict with: scaler, lda, basin, W, epsilon,
    dist_to_basin, meta.
    Verifies artifact checksums.
    """
    for path in [SCALER_FILE, LDA_FILE, BASIN_FILE, MODEL_META_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python protein/train.py"
            )

    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)

    with open(LDA_FILE, "rb") as f:
        lda = pickle.load(f)

    basin = np.load(BASIN_FILE)

    with open(MODEL_META_FILE) as f:
        meta = json.load(f)

    # Verify artifact integrity
    checksums = meta["artifact_checksums"]
    assert md5_file(SCALER_FILE) == checksums["scaler"], \
        "Scaler checksum mismatch — retrain required"
    assert md5_file(LDA_FILE) == checksums["lda"], \
        "LDA checksum mismatch — retrain required"
    assert md5_file(BASIN_FILE) == checksums["basin"], \
        "Basin checksum mismatch — retrain required"

    W = build_weight_vector()
    epsilon = meta["metrics"]["epsilon"]
    k = K_NEIGHBORS

    def dist_to_basin(p):
        dists = np.abs(basin - p)
        return np.mean(np.sort(dists)[:min(k, len(basin))])

    return {
        "scaler": scaler,
        "lda": lda,
        "basin": basin,
        "W": W,
        "epsilon": epsilon,
        "dist_to_basin": dist_to_basin,
        "meta": meta,
    }


if __name__ == "__main__":
    train()
