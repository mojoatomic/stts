"""
Train STTS conjunction models and run V1 basin separation test.

This is the ONLY script that calls .fit() on any preprocessor or model.
All downstream scripts (validate.py) call load_model() which loads
serialized artifacts — never refits.

Two models trained:
  Model A: All 35 geometry features with original causal weights.
  Model B: 15 consistent features only (KS < 0.3 between train/test HR),
           uniform weights (W=1.0) since KS consistency already selected
           for generalizability.

Pipeline: F (corpus.py) → W (causal weights) → M (LDA projection)
  then: construct failure basin, compute k-NN distances, verify V1.

Usage:
    python conjunction/train.py

Requires: features already extracted (python conjunction/corpus.py)
"""

import csv
import hashlib
import json
import os
import pickle
import sys

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "conjunction")
TRAIN_FEATURES = os.path.join(DATA_DIR, "train_features.csv")

# ---------------------------------------------------------------------------
# All 35 geometry features — must match corpus.py FEATURE_NAMES
# ---------------------------------------------------------------------------
ALL_FEATURES = [
    # F_geom (10)
    "miss_dist_final",
    "d_miss_dist_dt_mean",
    "d_miss_dist_dt_last",
    "mahal_dist_final",
    "d_mahal_dt_mean",
    "rel_pos_r_final",
    "rel_pos_t_final",
    "rel_pos_n_final",
    "miss_dist_late_early_ratio",
    "mahal_late_early_ratio",
    # F_cov (12)
    "t_sigma_r_final",
    "c_sigma_r_final",
    "d_t_sigma_r_dt",
    "d_c_sigma_r_dt",
    "t_cov_det_final",
    "c_cov_det_final",
    "d_t_cov_det_dt",
    "d_c_cov_det_dt",
    "cov_det_ratio_tc",
    "sigma_r_ratio_tc",
    "t_cov_late_early_ratio",
    "c_cov_late_early_ratio",
    # F_od (8)
    "t_obs_used_final",
    "c_obs_used_final",
    "d_c_obs_used_dt",
    "c_weighted_rms_final",
    "c_actual_od_span_final",
    "t_time_lastob_end_final",
    "c_time_lastob_end_final",
    "od_quality_ratio",
    # F_timing (4)
    "inter_cdm_dt_mean",
    "inter_cdm_dt_std",
    "inter_cdm_dt_last",
    "n_cdms",
    # F_cross (1)
    "corr_miss_c_cov_det",
]

# ---------------------------------------------------------------------------
# 15 consistent features (KS < 0.3 between train HR and test HR)
# These features have structurally identical distributions across train/test
# and are safe for generalization.
# ---------------------------------------------------------------------------
CONSISTENT_FEATURES = [
    # F_geom (2): rates generalize, absolute values don't
    "d_miss_dist_dt_mean",       # KS=0.267 — rate of miss distance change
    "d_miss_dist_dt_last",       # KS=0.190 — most recent miss distance rate
    # F_cov (5): target-side generalizes, chaser-side doesn't
    "t_sigma_r_final",           # KS=0.126 — target radial uncertainty
    "d_t_sigma_r_dt",            # KS=0.138 — target uncertainty shrinkage rate
    "t_cov_det_final",           # KS=0.138 — target covariance volume
    "d_t_cov_det_dt",            # KS=0.219 — target covariance shrinkage rate
    "t_cov_late_early_ratio",    # KS=0.224 — target covariance tightening
    # F_od (3): target-side OD is consistent
    "t_obs_used_final",          # KS=0.255 — target observations
    "d_c_obs_used_dt",           # KS=0.162 — chaser tracking improvement rate
    "t_time_lastob_end_final",   # KS=0.000 — target data staleness (always 0)
    # F_timing (4): CDM cadence generalizes completely
    "inter_cdm_dt_mean",         # KS=0.228 — mean CDM cadence
    "inter_cdm_dt_std",          # KS=0.174 — cadence regularity
    "inter_cdm_dt_last",         # KS=0.246 — most recent update interval
    "n_cdms",                    # KS=0.221 — CDM count in 2-day window
    # F_cross (1)
    "corr_miss_c_cov_det",       # KS=0.191 — geometry-uncertainty coupling
]

# ---------------------------------------------------------------------------
# Causal weights for Model A (all 35 features)
# ---------------------------------------------------------------------------
WEIGHT_SPEC_A = {
    # F_geom (10)
    "miss_dist_final":             1.0,
    "d_miss_dist_dt_mean":         2.5,
    "d_miss_dist_dt_last":         2.5,
    "mahal_dist_final":            1.5,
    "d_mahal_dt_mean":             3.0,
    "rel_pos_r_final":             0.8,
    "rel_pos_t_final":             0.8,
    "rel_pos_n_final":             0.8,
    "miss_dist_late_early_ratio":  2.0,
    "mahal_late_early_ratio":      2.5,
    # F_cov (12)
    "t_sigma_r_final":         1.0,
    "c_sigma_r_final":         1.0,
    "d_t_sigma_r_dt":          2.0,
    "d_c_sigma_r_dt":          2.0,
    "t_cov_det_final":         1.0,
    "c_cov_det_final":         1.0,
    "d_t_cov_det_dt":          2.5,
    "d_c_cov_det_dt":          2.5,
    "cov_det_ratio_tc":        1.5,
    "sigma_r_ratio_tc":        1.5,
    "t_cov_late_early_ratio":  2.0,
    "c_cov_late_early_ratio":  2.0,
    # F_od (8)
    "t_obs_used_final":            1.0,
    "c_obs_used_final":            1.5,
    "d_c_obs_used_dt":             1.5,
    "c_weighted_rms_final":        1.5,
    "c_actual_od_span_final":      1.0,
    "t_time_lastob_end_final":     1.0,
    "c_time_lastob_end_final":     1.5,
    "od_quality_ratio":            1.2,
    # F_timing (4)
    "inter_cdm_dt_mean":   1.5,
    "inter_cdm_dt_std":    1.0,
    "inter_cdm_dt_last":   1.5,
    "n_cdms":              0.5,
    # F_cross (1)
    "corr_miss_c_cov_det":  1.5,
}

# Model hyperparameters
LDA_COMPONENTS = 1
LDA_SOLVER = "svd"
K_NEIGHBORS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_training_data(feature_names):
    """Load train_features.csv, selecting only the given feature columns."""
    X_rows = []
    labels = []
    event_ids = []

    with open(TRAIN_FEATURES, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            features = [float(row[fname]) for fname in feature_names]
            X_rows.append(features)
            labels.append(int(row["label"]))
            event_ids.append(row["event_id"])

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(labels, dtype=np.int32)
    return X, y, event_ids


def train_model(model_name, feature_names, weight_spec, artifacts_dir):
    """Train one model variant. Returns metadata dict.

    THIS IS THE ONLY FUNCTION THAT CALLS .fit().
    """
    n_feat = len(feature_names)

    print(f"\n{'=' * 60}")
    print(f"  TRAINING: {model_name}")
    print(f"  Features: {n_feat}")
    print(f"  Artifacts: {artifacts_dir}")
    print(f"{'=' * 60}")

    X, y, event_ids = load_training_data(feature_names)
    n_events = len(y)
    n_pos = int(y.sum())
    n_neg = n_events - n_pos

    print(f"\n  {n_events} events: {n_pos} high-risk, {n_neg} nominal")

    assert n_pos >= 5, f"Insufficient high-risk events: {n_pos}"

    # ── Stage W: Causal weighting ──────────────────────────────
    W = np.array([weight_spec.get(f, 1.0) for f in feature_names])
    print(f"  W vector: range [{W.min():.1f}, {W.max():.1f}]")

    # ── Fit scaler — .fit() called here ────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    X_w = X_scaled * W

    # ── Fit LDA — .fit() called here ──────────────────────────
    lda = LinearDiscriminantAnalysis(
        n_components=LDA_COMPONENTS, solver=LDA_SOLVER
    )
    lda.fit(X_w, y)
    X_proj = lda.transform(X_w).ravel()

    # LDA diagnostics: condition number of within-class scatter
    Sw = np.zeros((n_feat, n_feat))
    for c in np.unique(y):
        Xc = X_w[y == c]
        diff = Xc - Xc.mean(axis=0)
        Sw += diff.T @ diff
    eigvals = np.linalg.eigvalsh(Sw)
    eigvals_pos = eigvals[eigvals > 1e-15]
    cond_number = float(eigvals_pos[-1] / eigvals_pos[0]) if len(eigvals_pos) > 0 else float("inf")

    print(f"  LDA condition number: {cond_number:.2e}")

    # Top 5 features by absolute LDA loading coefficient
    coef = lda.coef_.ravel()
    abs_coef = np.abs(coef)
    top_idx = np.argsort(abs_coef)[::-1][:5]
    print(f"\n  Top 5 features by LDA loading:")
    for rank, idx in enumerate(top_idx):
        print(f"    {rank+1}. {feature_names[idx]:30s}  coef={coef[idx]:+.4f}  "
              f"(W={W[idx]:.1f}, effective={coef[idx]*W[idx]:+.4f})")

    # ── Failure basin ──────────────────────────────────────────
    basin = X_proj[y == 1].copy()
    nominal_proj = X_proj[y == 0]

    print(f"\n  Basin: {len(basin)} embeddings")
    print(f"  Basin mean={np.mean(basin):.4f}, std={np.std(basin):.4f}")
    print(f"  Nominal mean={np.mean(nominal_proj):.4f}, std={np.std(nominal_proj):.4f}")

    # ── k-NN distances ─────────────────────────────────────────
    k = K_NEIGHBORS

    def _dist_to_basin(p):
        dists = np.abs(basin - p)
        return np.mean(np.sort(dists)[:min(k, len(basin))])

    all_dists = np.array([_dist_to_basin(p) for p in X_proj])
    nom_dists = all_dists[y == 0]
    hr_dists = all_dists[y == 1]

    # ── V1 ─────────────────────────────────────────────────────
    median_nominal = float(np.median(nom_dists))
    median_hr = float(np.median(hr_dists))
    v1_sep = median_nominal / max(median_hr, 1e-10)
    _, v1_p = mannwhitneyu(nom_dists, hr_dists, alternative="greater")
    v1_p = float(v1_p)

    print(f"\n  V1: {v1_sep:.1f}x separation, p={v1_p:.2e} "
          f"{'PASS' if v1_p < 0.001 else 'FAIL'}")

    # ── Serialize ──────────────────────────────────────────────
    os.makedirs(artifacts_dir, exist_ok=True)

    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    lda_path = os.path.join(artifacts_dir, "lda.pkl")
    basin_path = os.path.join(artifacts_dir, "basin.npy")
    weights_path = os.path.join(artifacts_dir, "weights.npy")
    meta_path = os.path.join(artifacts_dir, "model_meta.json")

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(lda_path, "wb") as f:
        pickle.dump(lda, f)
    np.save(basin_path, basin)
    np.save(weights_path, W)

    config = {
        "model_name": model_name,
        "feature_names": feature_names,
        "n_features": n_feat,
        "weights": {f: float(weight_spec.get(f, 1.0)) for f in feature_names},
        "lda_components": LDA_COMPONENTS,
        "lda_solver": LDA_SOLVER,
        "k_neighbors": K_NEIGHBORS,
        "high_risk_threshold": -5.0,
    }

    meta = {
        "config": config,
        "training": {
            "n_events": n_events,
            "n_high_risk": n_pos,
            "n_nominal": n_neg,
        },
        "diagnostics": {
            "lda_scatter_condition_number": cond_number,
            "basin_size": len(basin),
            "basin_mean": float(np.mean(basin)),
            "basin_std": float(np.std(basin)),
            "nominal_mean": float(np.mean(nominal_proj)),
            "nominal_std": float(np.std(nominal_proj)),
            "top_lda_loadings": [
                {"feature": feature_names[idx], "coefficient": float(coef[idx])}
                for idx in top_idx
            ],
        },
        "v1": {
            "separation_ratio": v1_sep,
            "p_value": v1_p,
            "median_nominal_distance": median_nominal,
            "median_high_risk_distance": median_hr,
            "passed": v1_p < 0.001,
        },
        "artifact_checksums": {
            "scaler": md5_file(scaler_path),
            "lda": md5_file(lda_path),
            "basin": md5_file(basin_path),
            "weights": md5_file(weights_path),
        },
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Artifacts saved to {artifacts_dir}/")

    return meta


def load_model(model_name="model_b"):
    """Load a frozen trained model. NEVER calls .fit().

    Args:
        model_name: "model_a" or "model_b"

    Returns dict with: scaler, lda, basin, W, meta, dist_to_basin, feature_names.
    Verifies artifact checksums.
    """
    artifacts_dir = os.path.join(DATA_DIR, "artifacts", model_name)

    paths = {
        "scaler": os.path.join(artifacts_dir, "scaler.pkl"),
        "lda": os.path.join(artifacts_dir, "lda.pkl"),
        "basin": os.path.join(artifacts_dir, "basin.npy"),
        "weights": os.path.join(artifacts_dir, "weights.npy"),
        "meta": os.path.join(artifacts_dir, "model_meta.json"),
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} not found at {path}. Run: python conjunction/train.py"
            )

    with open(paths["scaler"], "rb") as f:
        scaler = pickle.load(f)
    with open(paths["lda"], "rb") as f:
        lda = pickle.load(f)
    basin = np.load(paths["basin"])
    W = np.load(paths["weights"])
    with open(paths["meta"]) as f:
        meta = json.load(f)

    # Verify artifact integrity
    checksums = meta["artifact_checksums"]
    assert md5_file(paths["scaler"]) == checksums["scaler"], "Scaler checksum mismatch"
    assert md5_file(paths["lda"]) == checksums["lda"], "LDA checksum mismatch"
    assert md5_file(paths["basin"]) == checksums["basin"], "Basin checksum mismatch"
    assert md5_file(paths["weights"]) == checksums["weights"], "Weights checksum mismatch"

    k = meta["config"]["k_neighbors"]
    feature_names = meta["config"]["feature_names"]

    def dist_to_basin(p):
        dists = np.abs(basin - p)
        return np.mean(np.sort(dists)[:min(k, len(basin))])

    return {
        "scaler": scaler,
        "lda": lda,
        "basin": basin,
        "W": W,
        "meta": meta,
        "dist_to_basin": dist_to_basin,
        "feature_names": feature_names,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading training data...")
    # Quick count check
    X_all, y_all, _ = load_training_data(ALL_FEATURES)
    print(f"  {len(y_all)} events, {int(y_all.sum())} high-risk, "
          f"{len(y_all) - int(y_all.sum())} nominal")

    # Model A: all 35 features, original causal weights
    meta_a = train_model(
        model_name="Model A (35 geometry features)",
        feature_names=ALL_FEATURES,
        weight_spec=WEIGHT_SPEC_A,
        artifacts_dir=os.path.join(DATA_DIR, "artifacts", "model_a"),
    )

    # Model B: 15 consistent features, uniform weights
    # W=1.0 for all: KS consistency analysis already selected features
    meta_b = train_model(
        model_name="Model B (15 consistent features)",
        feature_names=CONSISTENT_FEATURES,
        weight_spec={f: 1.0 for f in CONSISTENT_FEATURES},
        artifacts_dir=os.path.join(DATA_DIR, "artifacts", "model_b"),
    )

    # ── Comparison table ───────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<35s} {'Model A':>12s} {'Model B':>12s}")
    print(f"  {'-'*35} {'-'*12} {'-'*12}")
    print(f"  {'Features':<35s} {meta_a['config']['n_features']:>12d} {meta_b['config']['n_features']:>12d}")
    print(f"  {'V1 separation ratio':<35s} {meta_a['v1']['separation_ratio']:>12.1f}x {meta_b['v1']['separation_ratio']:>12.1f}x")
    print(f"  {'V1 p-value':<35s} {meta_a['v1']['p_value']:>12.2e} {meta_b['v1']['p_value']:>12.2e}")
    print(f"  {'V1 passed':<35s} {'PASS' if meta_a['v1']['passed'] else 'FAIL':>12s} {'PASS' if meta_b['v1']['passed'] else 'FAIL':>12s}")
    print(f"  {'LDA condition number':<35s} {meta_a['diagnostics']['lda_scatter_condition_number']:>12.2e} {meta_b['diagnostics']['lda_scatter_condition_number']:>12.2e}")
    print(f"  {'Basin size':<35s} {meta_a['diagnostics']['basin_size']:>12d} {meta_b['diagnostics']['basin_size']:>12d}")
    print(f"  {'Basin mean':<35s} {meta_a['diagnostics']['basin_mean']:>12.4f} {meta_b['diagnostics']['basin_mean']:>12.4f}")
    print(f"  {'Basin std':<35s} {meta_a['diagnostics']['basin_std']:>12.4f} {meta_b['diagnostics']['basin_std']:>12.4f}")


if __name__ == "__main__":
    main()
