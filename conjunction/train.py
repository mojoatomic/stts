"""
Train the STTS conjunction model and run V1 basin separation test.

This is the ONLY script that calls .fit() on any preprocessor or model.
All downstream scripts (validate.py) call load_model() which loads
serialized artifacts — never refits.

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
ARTIFACTS_DIR = os.path.join(DATA_DIR, "artifacts")
TRAIN_FEATURES = os.path.join(DATA_DIR, "train_features.csv")

SCALER_FILE = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
LDA_FILE = os.path.join(ARTIFACTS_DIR, "lda.pkl")
BASIN_FILE = os.path.join(ARTIFACTS_DIR, "basin.npy")
WEIGHTS_FILE = os.path.join(ARTIFACTS_DIR, "weights.npy")
MODEL_META_FILE = os.path.join(ARTIFACTS_DIR, "model_meta.json")

# ---------------------------------------------------------------------------
# Feature names — must match corpus.py FEATURE_NAMES exactly
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    # F_risk (8)
    "risk_final",
    "risk_mean",
    "d_risk_dt_mean",
    "d_risk_dt_last",
    "d2_risk_dt2_mean",
    "risk_late_early_ratio",
    "risk_range",
    "risk_monotonicity",
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
    # F_cross (4)
    "corr_risk_miss",
    "corr_risk_mahal",
    "corr_risk_c_sigma_r",
    "corr_miss_c_cov_det",
]

N_FEATURES = len(FEATURE_NAMES)

# ---------------------------------------------------------------------------
# Causal weight vector — from research report
#
# Weight hierarchy reflects causal proximity to high-risk outcome:
#   HIGHEST (3.0-3.5): Risk rate features — most direct signal
#   HIGH    (2.0-2.5): Geometry rate, covariance rate, cross-correlations
#   MEDIUM  (1.0-1.5): Final values, OD quality, timing
#   LOW     (0.5-0.8): Static context, CDM count, RTN components
# ---------------------------------------------------------------------------
WEIGHT_SPEC = {
    # F_risk (8): risk trajectory is the most direct state variable
    "risk_final":              1.0,   # direct state
    "risk_mean":               0.8,   # central tendency
    "d_risk_dt_mean":          3.0,   # risk trend — primary causal signal
    "d_risk_dt_last":          3.5,   # current risk acceleration — most upstream
    "d2_risk_dt2_mean":        2.5,   # risk acceleration
    "risk_late_early_ratio":   3.0,   # late amplification (cf. reentry MEAN_MOTION_DOT ratio)
    "risk_range":              1.5,   # volatility span
    "risk_monotonicity":       2.5,   # sustained trend indicator

    # F_geom (10): miss distance and geometry evolution
    "miss_dist_final":             1.0,   # current geometry
    "d_miss_dist_dt_mean":         2.5,   # geometry trend
    "d_miss_dist_dt_last":         2.5,   # current approach rate
    "mahal_dist_final":            1.5,   # scaled geometry (uncertainty-weighted)
    "d_mahal_dt_mean":             3.0,   # uncertainty-weighted trend
    "rel_pos_r_final":             0.8,   # RTN component
    "rel_pos_t_final":             0.8,   # RTN component (dominant)
    "rel_pos_n_final":             0.8,   # RTN component
    "miss_dist_late_early_ratio":  2.0,   # convergence pattern
    "mahal_late_early_ratio":      2.5,   # scaled convergence

    # F_cov (12): covariance structure evolution
    "t_sigma_r_final":         1.0,   # current target uncertainty
    "c_sigma_r_final":         1.0,   # current chaser uncertainty
    "d_t_sigma_r_dt":          2.0,   # target uncertainty trend
    "d_c_sigma_r_dt":          2.0,   # chaser uncertainty trend
    "t_cov_det_final":         1.0,   # target covariance volume
    "c_cov_det_final":         1.0,   # chaser covariance volume
    "d_t_cov_det_dt":          2.5,   # target covariance shrinkage rate
    "d_c_cov_det_dt":          2.5,   # chaser covariance shrinkage rate
    "cov_det_ratio_tc":        1.5,   # uncertainty asymmetry
    "sigma_r_ratio_tc":        1.5,   # radial uncertainty balance
    "t_cov_late_early_ratio":  2.0,   # target covariance tightening
    "c_cov_late_early_ratio":  2.0,   # chaser covariance tightening

    # F_od (8): orbit determination quality
    "t_obs_used_final":            1.0,   # target OD quality
    "c_obs_used_final":            1.5,   # chaser OD quality (limiting factor)
    "d_c_obs_used_dt":             1.5,   # tracking improvement rate
    "c_weighted_rms_final":        1.5,   # chaser OD fit quality
    "c_actual_od_span_final":      1.0,   # OD freshness
    "t_time_lastob_end_final":     1.0,   # target data staleness
    "c_time_lastob_end_final":     1.5,   # chaser data staleness
    "od_quality_ratio":            1.2,   # OD asymmetry

    # F_timing (4): temporal structure
    "inter_cdm_dt_mean":   1.5,   # update cadence (ESA increases for high-risk)
    "inter_cdm_dt_std":    1.0,   # cadence regularity
    "inter_cdm_dt_last":   1.5,   # current update rate
    "n_cdms":              0.5,   # sequence richness (low: static context)

    # F_cross (4): cross-parameter coupling
    "corr_risk_miss":       2.0,   # risk-geometry coupling
    "corr_risk_mahal":      2.0,   # risk-scaled geometry coupling
    "corr_risk_c_sigma_r":  1.5,   # risk-uncertainty coupling
    "corr_miss_c_cov_det":  1.5,   # geometry-uncertainty coupling
}

# Model hyperparameters
LDA_COMPONENTS = 1
LDA_SOLVER = "svd"
K_NEIGHBORS = 5


def build_weight_vector():
    """Construct the 46-element weight vector from WEIGHT_SPEC."""
    W = np.ones(N_FEATURES)
    for i, fname in enumerate(FEATURE_NAMES):
        W[i] = WEIGHT_SPEC[fname]
    return W


def config_snapshot():
    """Capture all hyperparameters for reproducibility audit."""
    return {
        "feature_names": FEATURE_NAMES,
        "n_features": N_FEATURES,
        "weights": {k: v for k, v in WEIGHT_SPEC.items()},
        "lda_components": LDA_COMPONENTS,
        "lda_solver": LDA_SOLVER,
        "k_neighbors": K_NEIGHBORS,
        "high_risk_threshold": -5.0,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_training_data():
    """Load train_features.csv into numpy arrays."""
    X_rows = []
    labels = []
    event_ids = []

    with open(TRAIN_FEATURES, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            features = [float(row[fname]) for fname in FEATURE_NAMES]
            X_rows.append(features)
            labels.append(int(row["label"]))
            event_ids.append(row["event_id"])

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(labels, dtype=np.int32)
    return X, y, event_ids


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train():
    """Train the conjunction STTS model. Serialize all artifacts."""

    print("Loading training features...")
    X, y, event_ids = load_training_data()
    n_events = len(y)
    n_pos = int(y.sum())
    n_neg = n_events - n_pos

    print(f"  {n_events} events: {n_pos} high-risk (label=1), {n_neg} nominal (label=0)")

    assert n_pos >= 5, f"Insufficient high-risk events: {n_pos}"
    assert n_neg >= 10, f"Insufficient nominal events: {n_neg}"

    # ── Stage W: Causal weighting ──────────────────────────────
    W = build_weight_vector()
    print(f"\n  W vector: {N_FEATURES} weights, range [{W.min():.1f}, {W.max():.1f}]")

    # ── Fit scaler — THIS IS THE ONLY PLACE scaler.fit() IS CALLED ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle NaN/Inf from constant features (single-CDM events have zero
    # variance in rate features, producing NaN after standardization)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Apply causal weights
    X_w = X_scaled * W

    # ── Stage M: Fit LDA — THIS IS THE ONLY PLACE lda.fit() IS CALLED ──
    lda = LinearDiscriminantAnalysis(
        n_components=LDA_COMPONENTS, solver=LDA_SOLVER
    )
    lda.fit(X_w, y)
    X_proj = lda.transform(X_w).ravel()

    # LDA diagnostics
    # Condition number of within-class scatter matrix
    classes = np.unique(y)
    Sw = np.zeros((N_FEATURES, N_FEATURES))
    for c in classes:
        Xc = X_w[y == c]
        diff = Xc - Xc.mean(axis=0)
        Sw += diff.T @ diff
    eigvals = np.linalg.eigvalsh(Sw)
    eigvals_pos = eigvals[eigvals > 1e-15]
    if len(eigvals_pos) > 0:
        cond_number = eigvals_pos[-1] / eigvals_pos[0]
    else:
        cond_number = float("inf")

    print(f"\n  LDA within-class scatter condition number: {cond_number:.2e}")

    # ── Failure basin ──────────────────────────────────────────
    # All training embeddings where label=1 enter the basin.
    # No sigma-tightening for conjunction: we have only 62 high-risk
    # events — too few to discard any.
    basin = X_proj[y == 1].copy()
    nominal_proj = X_proj[y == 0]

    print(f"\n  Basin: {len(basin)} high-risk embeddings")
    print(f"  Nominal: {len(nominal_proj)} nominal embeddings")
    print(f"  Basin mean: {np.mean(basin):.4f}, std: {np.std(basin):.4f}")
    print(f"  Nominal mean: {np.mean(nominal_proj):.4f}, std: {np.std(nominal_proj):.4f}")

    # ── k-NN distances to basin ────────────────────────────────
    k = K_NEIGHBORS

    def _dist_to_basin(p):
        dists = np.abs(basin - p)
        return np.mean(np.sort(dists)[:min(k, len(basin))])

    all_dists = np.array([_dist_to_basin(p) for p in X_proj])

    nom_dists = all_dists[y == 0]
    hr_dists = all_dists[y == 1]

    # ── V1: Mann-Whitney U basin separation test ───────────────
    median_nominal = float(np.median(nom_dists))
    median_hr = float(np.median(hr_dists))
    v1_sep = median_nominal / max(median_hr, 1e-10)

    # alternative="greater": test that nominal distances are stochastically
    # greater than high-risk distances (nominal is farther from basin)
    _, v1_p = mannwhitneyu(nom_dists, hr_dists, alternative="greater")
    v1_p = float(v1_p)

    print(f"\n{'=' * 60}")
    print(f"  V1 BASIN SEPARATION TEST")
    print(f"{'=' * 60}")
    print(f"  Median distance to basin:")
    print(f"    Nominal:   {median_nominal:.6f}")
    print(f"    High-risk: {median_hr:.6f}")
    print(f"  Separation ratio: {v1_sep:.1f}x")
    print(f"  Mann-Whitney U p-value: {v1_p:.2e}")
    print(f"  Result: {'PASS' if v1_p < 0.001 else 'FAIL'} (threshold p<0.001)")

    # ── Serialize artifacts ────────────────────────────────────
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

    with open(LDA_FILE, "wb") as f:
        pickle.dump(lda, f)

    np.save(BASIN_FILE, basin)
    np.save(WEIGHTS_FILE, W)

    meta = {
        "config": config_snapshot(),
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
        },
        "v1": {
            "separation_ratio": v1_sep,
            "p_value": v1_p,
            "median_nominal_distance": median_nominal,
            "median_high_risk_distance": median_hr,
            "passed": v1_p < 0.001,
        },
        "artifact_checksums": {
            "scaler": md5_file(SCALER_FILE),
            "lda": md5_file(LDA_FILE),
            "basin": md5_file(BASIN_FILE),
            "weights": md5_file(WEIGHTS_FILE),
        },
    }

    with open(MODEL_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Artifacts saved to {ARTIFACTS_DIR}/")
    print(f"    scaler.pkl  ({md5_file(SCALER_FILE)[:12]})")
    print(f"    lda.pkl     ({md5_file(LDA_FILE)[:12]})")
    print(f"    basin.npy   ({md5_file(BASIN_FILE)[:12]})")
    print(f"    weights.npy ({md5_file(WEIGHTS_FILE)[:12]})")
    print(f"    model_meta.json")

    return meta


def load_model():
    """Load the frozen trained model. NEVER calls .fit().

    Returns dict with: scaler, lda, basin, W, meta, dist_to_basin.
    Verifies artifact checksums.
    """
    for path, name in [
        (SCALER_FILE, "scaler.pkl"),
        (LDA_FILE, "lda.pkl"),
        (BASIN_FILE, "basin.npy"),
        (WEIGHTS_FILE, "weights.npy"),
        (MODEL_META_FILE, "model_meta.json"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} not found at {path}. Run: python conjunction/train.py"
            )

    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)

    with open(LDA_FILE, "rb") as f:
        lda = pickle.load(f)

    basin = np.load(BASIN_FILE)
    W = np.load(WEIGHTS_FILE)

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
    assert md5_file(WEIGHTS_FILE) == checksums["weights"], \
        "Weights checksum mismatch — retrain required"

    k = meta["config"]["k_neighbors"]

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
    }


if __name__ == "__main__":
    train()
