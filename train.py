"""
Train the canonical STTS orbital model.

This is the ONLY script that calls .fit() on any preprocessor or model.
All downstream scripts (validate.py, case_study.py) call load_model()
which loads serialized artifacts — never refits.

Usage:
    python3 train.py

Requires: corpus already built (python3 corpus.py)
"""

import os
import json
import pickle
import hashlib
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

import config
from corpus import load_corpus
from horizons_stts_pipeline import (
    canonical_weights,
    build_dataset,
)


def md5_file(path: str) -> str:
    """Compute MD5 checksum of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def train():
    """Train model on corpus train split. Serialize all artifacts."""
    corpus = load_corpus()
    trajectories = corpus["trajectories"]
    train_idx = corpus["train_idx"]

    # Build training trajectories as (elements, ca_jd) tuples
    train_trajs = [
        (trajectories[i]["elements"], trajectories[i]["ca_jd"])
        for i in train_idx
    ]

    print(f"Training on {len(train_trajs)} trajectories")

    # Extract features (uses library WINDOW_DAYS, STRIDE_DAYS, WARNING_DAYS)
    X_train, r_train, y_train = build_dataset(train_trajs)
    print(f"  Windows: {len(X_train)} "
          f"({y_train.sum():.0f} precursor, {(y_train==0).sum():.0f} nominal)")

    assert y_train.sum() >= 5, "Insufficient precursor windows"
    assert (y_train == 0).sum() >= 5, "Insufficient nominal windows"

    # W weights — from canonical_weights() (the single operational definition)
    W = canonical_weights()

    # Fit scaler — THIS IS THE ONLY PLACE scaler.fit() IS CALLED
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_w = X_scaled * W

    # Fit LDA — THIS IS THE ONLY PLACE lda.fit() IS CALLED
    lda = LinearDiscriminantAnalysis(
        n_components=config.LDA_COMPONENTS, solver="svd"
    )
    lda.fit(X_w, y_train)
    X_proj = lda.transform(X_w).ravel()

    # Failure basin
    basin = X_proj[y_train == 1]

    # Training distances
    k = config.K_NEIGHBORS
    def _dist_to_basin(p):
        dists = np.abs(basin - p)
        return np.mean(np.sort(dists)[:min(k, len(basin))])

    train_dists = np.array([_dist_to_basin(p) for p in X_proj])

    # V1: separation
    nom_d = train_dists[y_train == 0]
    pre_d = train_dists[y_train == 1]
    v1_sep = float(np.median(nom_d) / (np.median(pre_d) + 1e-10))
    _, v1_p = mannwhitneyu(nom_d, pre_d, alternative="greater")
    v1_p = float(v1_p)

    # V2: monotonic approach
    mask = r_train < 365
    if mask.sum() > 5:
        v2_rho, v2_p = spearmanr(r_train[mask], train_dists[mask])
        v2_rho, v2_p = float(v2_rho), float(v2_p)
    else:
        v2_rho, v2_p = 0.0, 1.0

    # Calibrate epsilon (maximize F1 on training data only)
    thresholds = np.percentile(train_dists, np.linspace(5, 95, 40))
    best_f1, best_eps = 0.0, float(thresholds[0])
    for eps in thresholds:
        preds = (train_dists < eps).astype(int)
        tp = int(((preds == 1) & (y_train == 1)).sum())
        fp = int(((preds == 1) & (y_train == 0)).sum())
        fn = int(((preds == 0) & (y_train == 1)).sum())
        pr = tp / max(1, tp + fp)
        re = tp / max(1, tp + fn)
        f1 = 2 * pr * re / max(1e-10, pr + re)
        if f1 > best_f1:
            best_f1, best_eps = f1, float(eps)

    print(f"  V1: {v1_sep:.1f}x (p={v1_p:.2e})")
    print(f"  V2: ρ={v2_rho:.3f} (p={v2_p:.2e})")
    print(f"  ε={best_eps:.4f} (train F1={best_f1:.3f})")

    # ── Serialize artifacts ───────────────────────────────
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)

    with open(config.SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

    with open(config.LDA_FILE, "wb") as f:
        pickle.dump(lda, f)

    np.save(config.BASIN_FILE, basin)

    # Model metadata — includes full config snapshot
    meta = {
        "config": config.config_snapshot(),
        "training": {
            "n_trajectories": len(train_trajs),
            "n_windows": len(X_train),
            "n_precursor": int(y_train.sum()),
            "n_nominal": int((y_train == 0).sum()),
        },
        "metrics": {
            "v1_separation": v1_sep,
            "v1_p": v1_p,
            "v2_rho": v2_rho,
            "v2_p": v2_p,
            "train_f1": best_f1,
            "epsilon": best_eps,
        },
        "artifact_checksums": {
            "scaler": md5_file(config.SCALER_FILE),
            "lda": md5_file(config.LDA_FILE),
            "basin": md5_file(config.BASIN_FILE),
        },
    }

    with open(config.MODEL_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Artifacts saved to {config.ARTIFACTS_DIR}/")
    print(f"    scaler: {meta['artifact_checksums']['scaler']}")
    print(f"    lda:    {meta['artifact_checksums']['lda']}")
    print(f"    basin:  {meta['artifact_checksums']['basin']}")

    return meta


def load_model() -> dict:
    """
    Load the frozen trained model from serialized artifacts.

    Returns a dict with: scaler, lda, basin, W, epsilon, dist_to_basin,
    meta. Verifies artifact checksums against model_meta.json.

    This function NEVER calls .fit(). It loads and returns.
    """
    for path in [config.SCALER_FILE, config.LDA_FILE,
                 config.BASIN_FILE, config.MODEL_META_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run: python3 train.py"
            )

    with open(config.SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)

    with open(config.LDA_FILE, "rb") as f:
        lda = pickle.load(f)

    basin = np.load(config.BASIN_FILE)

    with open(config.MODEL_META_FILE, "r") as f:
        meta = json.load(f)

    # Verify artifact integrity
    checksums = meta["artifact_checksums"]
    assert md5_file(config.SCALER_FILE) == checksums["scaler"], \
        "Scaler artifact checksum mismatch — retrain required"
    assert md5_file(config.LDA_FILE) == checksums["lda"], \
        "LDA artifact checksum mismatch — retrain required"
    assert md5_file(config.BASIN_FILE) == checksums["basin"], \
        "Basin artifact checksum mismatch — retrain required"

    # Verify config consistency
    stored_config = meta["config"]
    current_config = config.config_snapshot()
    if stored_config != current_config:
        import warnings
        warnings.warn(
            "Config has changed since model was trained. "
            "Consider retraining: python3 train.py"
        )

    W = canonical_weights()
    epsilon = meta["metrics"]["epsilon"]
    k = config.K_NEIGHBORS

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
