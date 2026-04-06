#!/usr/bin/env python3
"""
STTS-Solar V1 Separation Test
==============================
Minimal V1 (statistical separation) test using Pattern A active regions
(clean quiet→flare→quiet arc): AR 3341 and AR 3721.

Failure basin:  FL (M/X-class flaring) trajectory segments
Nominal pool:   NF (quiet/B/C-class) trajectory segments

Pipeline: F → W → M → k-NN → Mann-Whitney U

Usage:
    python solar/v1_separation_test.py [--data-dir DATA_DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# ── Import stitching from sibling module ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stitch_trajectories import stitch_ar, SHARP_PARAMS

# ── Configuration ─────────────────────────────────────────────────────
PATTERN_A_ARS = ["3341", "3721"]
WINDOW_SIZE = 60          # 60 timesteps = 12 hours at 12-min cadence
WINDOW_STRIDE = 10        # 10 timesteps = 2 hours
K_NEIGHBORS = 5           # k for k-NN mean distance
LDA_COMPONENTS = 1
FFT_NUM_FREQS = 5         # low-frequency FFT bins to keep

# ── Stage W: Solar causal weight groups ───────────────────────────────
# Based on magnetic free energy buildup physics:
#   - Current helicity and twist parameters LEAD flare onset (upstream)
#   - Lorentz force and shear measure stress accumulation (mid-stream)
#   - Field gradients and geometry are more descriptive (downstream)
CAUSAL_GROUPS = {
    "high": {  # 2.5 — causally upstream: current, helicity, PIL flux
        "R_VALUE",    # log unsigned flux near polarity inversion line
        "TOTUSJH",    # total unsigned current helicity
        "ABSNJZH",    # absolute net current helicity
        "MEANJZH",    # mean current helicity
        "MEANALP",    # mean twist parameter (alpha)
        "USFLUX",     # total unsigned magnetic flux
    },
    "medium_high": {  # 1.8 — stress/energy buildup
        "MEANPOT",    # mean excess magnetic energy density
        "TOTPOT",     # total magnetic energy density
        "SAVNCPP",    # sum abs net currents per polarity
        "MEANSHR",    # mean shear angle
        "SHRGT45",    # fraction of high-shear pixels
        "TOTUSJZ",    # total unsigned vertical current
        "MEANJZD",    # mean vertical current density
    },
    "medium": {  # 1.0 — Lorentz force (stress indicator)
        "TOTBSQ",     # total Lorentz force magnitude
        "TOTFX",      # x-component Lorentz force
        "TOTFY",      # y-component Lorentz force
        "TOTFZ",      # z-component Lorentz force
    },
    "low": {  # 0.5 — field geometry/gradients (descriptive, noisy)
        "MEANGAM",    # mean inclination angle
        "MEANGBT",    # mean gradient total field
        "MEANGBZ",    # mean gradient vertical field
        "MEANGBH",    # mean gradient horizontal field
        "EPSZ",       # normalized Lorentz z
        "EPSX",       # normalized Lorentz x
        "EPSY",       # normalized Lorentz y
    },
}
WEIGHT_VALUES = {"high": 2.5, "medium_high": 1.8, "medium": 1.0, "low": 0.5}

# Feature-class multipliers (same logic as C-MAPSS domain)
FEATURE_CLASS_MULTIPLIERS = {
    "time_domain": 1.0,
    "rate": 1.5,
    "frequency": 1.3,
    "covariance": 1.2,
}


# ── Stage F: Feature Extraction ──────────────────────────────────────

def extract_window_features(window: np.ndarray) -> np.ndarray:
    """Extract features from a single window (n_steps x n_sensors).

    Four feature classes matching the STTS pipeline:
    1. Time-domain: mean, std, min, max per sensor
    2. Rate: mean/std of 1st derivative, mean of 2nd derivative
    3. Frequency: top FFT bins per sensor
    4. Covariance: upper-triangle correlations + top eigenvalues
    """
    n_steps, n_sensors = window.shape
    features = []
    sensor_map = []  # tracks which sensor each feature belongs to
    class_map = []   # tracks feature class

    # 1. Time-domain (4 x n_sensors)
    for i in range(n_sensors):
        col = window[:, i]
        features.extend([np.nanmean(col), np.nanstd(col),
                         np.nanmin(col), np.nanmax(col)])
        sensor_map.extend([i] * 4)
        class_map.extend(["time_domain"] * 4)

    # 2. Rate features (3 x n_sensors)
    d1 = np.diff(window, axis=0)
    d2 = np.diff(d1, axis=0)
    for i in range(n_sensors):
        features.extend([np.nanmean(d1[:, i]), np.nanstd(d1[:, i]),
                         np.nanmean(d2[:, i])])
        sensor_map.extend([i] * 3)
        class_map.extend(["rate"] * 3)

    # 3. Frequency-domain (FFT_NUM_FREQS x n_sensors)
    for i in range(n_sensors):
        col = window[:, i]
        # Replace NaN with column mean for FFT
        col_clean = np.where(np.isnan(col), np.nanmean(col), col)
        fft_vals = np.abs(np.fft.rfft(col_clean))
        bins = fft_vals[1:1 + FFT_NUM_FREQS]
        # Pad if window too short
        if len(bins) < FFT_NUM_FREQS:
            bins = np.pad(bins, (0, FFT_NUM_FREQS - len(bins)))
        features.extend(bins.tolist())
        sensor_map.extend([i] * FFT_NUM_FREQS)
        class_map.extend(["frequency"] * FFT_NUM_FREQS)

    # 4. Covariance structure
    # Replace NaN for correlation computation
    window_clean = np.where(np.isnan(window), np.nanmean(window, axis=0), window)
    # Upper triangle of correlation matrix
    corr = np.corrcoef(window_clean.T)
    corr = np.nan_to_num(corr, nan=0.0)
    triu_idx = np.triu_indices(n_sensors, k=1)
    corr_feats = corr[triu_idx].tolist()
    features.extend(corr_feats)
    sensor_map.extend([-1] * len(corr_feats))  # cross-sensor
    class_map.extend(["covariance"] * len(corr_feats))

    # Top eigenvalues of covariance
    cov = np.cov(window_clean.T)
    cov = np.nan_to_num(cov, nan=0.0)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    n_eig = min(5, len(eigvals))
    features.extend(eigvals[:n_eig].tolist())
    sensor_map.extend([-1] * n_eig)
    class_map.extend(["covariance"] * n_eig)

    return np.array(features), sensor_map, class_map


def extract_all_features(trajectory: np.ndarray) -> tuple[np.ndarray, list, list]:
    """Extract windowed features from a full trajectory."""
    n_steps, n_sensors = trajectory.shape
    all_features = []

    sensor_map = None
    class_map = None

    for start in range(0, n_steps - WINDOW_SIZE + 1, WINDOW_STRIDE):
        window = trajectory[start:start + WINDOW_SIZE]
        feats, sm, cm = extract_window_features(window)
        all_features.append(feats)
        if sensor_map is None:
            sensor_map = sm
            class_map = cm

    if not all_features:
        return np.empty((0, 0)), [], []

    return np.vstack(all_features), sensor_map, class_map


# ── Stage W: Causal Weighting ────────────────────────────────────────

def build_weight_vector(sensor_map: list, class_map: list) -> np.ndarray:
    """Build causal weight vector for solar SHARP parameters."""
    n_features = len(sensor_map)
    weights = np.ones(n_features)

    # Build sensor -> weight lookup
    sensor_weight = {}
    for group, params in CAUSAL_GROUPS.items():
        w = WEIGHT_VALUES[group]
        for param in params:
            if param in SHARP_PARAMS:
                sensor_weight[SHARP_PARAMS.index(param)] = w

    # Apply sensor-level weights
    for i in range(n_features):
        sensor_idx = sensor_map[i]
        if sensor_idx >= 0 and sensor_idx in sensor_weight:
            weights[i] = sensor_weight[sensor_idx]
        elif sensor_idx == -1:  # covariance features
            weights[i] = 1.2  # moderate weight for cross-sensor structure

    # Apply feature-class multipliers
    for i in range(n_features):
        multiplier = FEATURE_CLASS_MULTIPLIERS.get(class_map[i], 1.0)
        weights[i] *= multiplier

    return weights


# ── Stage M: Manifold Projection + k-NN ──────────────────────────────

def dist_to_basin(point: float, basin: np.ndarray, k: int = K_NEIGHBORS) -> float:
    """Mean distance to k nearest basin points in 1D LDA space."""
    dists = np.abs(basin - point)
    k_actual = min(k, len(basin))
    return float(np.mean(np.sort(dists)[:k_actual]))


# ── V1 Computation ───────────────────────────────────────────────────

def compute_v1(nom_dists: np.ndarray, pre_dists: np.ndarray) -> dict:
    """Compute V1 separation metric with Mann-Whitney U test."""
    v1_sep = float(np.median(nom_dists) / (np.median(pre_dists) + 1e-10))
    stat, p_value = mannwhitneyu(nom_dists, pre_dists, alternative="greater")
    return {
        "v1_separation": v1_sep,
        "p_value": p_value,
        "u_statistic": float(stat),
        "median_nominal": float(np.median(nom_dists)),
        "median_precursor": float(np.median(pre_dists)),
        "n_nominal": len(nom_dists),
        "n_precursor": len(pre_dists),
    }


# ── Main ─────────────────────────────────────────────────────────────

def build_segments(result: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract FL and NF trajectory segments from a stitched AR result.

    Returns (fl_trajectory, nf_trajectory) as numpy arrays of shape
    (n_timesteps, 24).
    """
    sorted_ts = result["trajectory_timestamps"]
    data = result["trajectory_data"]
    source = result["trajectory_source"]

    fl_rows = []
    nf_rows = []

    for ts in sorted_ts:
        values = [data[ts][p] for p in SHARP_PARAMS]
        row = np.array([v if v is not None else np.nan for v in values])
        if source[ts] == "FL":
            fl_rows.append(row)
        else:
            nf_rows.append(row)

    fl_arr = np.vstack(fl_rows) if fl_rows else np.empty((0, len(SHARP_PARAMS)))
    nf_arr = np.vstack(nf_rows) if nf_rows else np.empty((0, len(SHARP_PARAMS)))
    return fl_arr, nf_arr


def main():
    parser = argparse.ArgumentParser(description="STTS-Solar V1 Separation Test")
    parser.add_argument("--data-dir", default="data/swan-sf")
    parser.add_argument("--partition", type=int, default=3)
    args = parser.parse_args()

    sep = "=" * 72

    print(f"\n{sep}")
    print("  STTS-Solar V1 Separation Test")
    print(f"  Pattern A ARs: {', '.join(PATTERN_A_ARS)}")
    print(sep)

    # ── Step 1: Stitch trajectories and extract FL/NF segments ────────
    all_fl_features = []
    all_nf_features = []
    sensor_map = None
    class_map = None

    for ar in PATTERN_A_ARS:
        print(f"\n  Stitching AR {ar}...", end=" ", flush=True)
        result = stitch_ar(args.data_dir, args.partition, ar)
        if "error" in result:
            print(f"ERROR: {result['error']}")
            continue

        fl_traj, nf_traj = build_segments(result)
        print(f"FL={fl_traj.shape[0]} steps, NF={nf_traj.shape[0]} steps")

        # ── Step 2: Extract windowed features ─────────────────────────
        if fl_traj.shape[0] >= WINDOW_SIZE:
            fl_feats, sm, cm = extract_all_features(fl_traj)
            all_fl_features.append(fl_feats)
            if sensor_map is None:
                sensor_map = sm
                class_map = cm
            print(f"    FL windows: {fl_feats.shape[0]} (feature dim: {fl_feats.shape[1]})")
        else:
            print(f"    FL segment too short ({fl_traj.shape[0]} < {WINDOW_SIZE}), skipped")

        if nf_traj.shape[0] >= WINDOW_SIZE:
            nf_feats, _, _ = extract_all_features(nf_traj)
            all_nf_features.append(nf_feats)
            print(f"    NF windows: {nf_feats.shape[0]} (feature dim: {nf_feats.shape[1]})")
        else:
            print(f"    NF segment too short ({nf_traj.shape[0]} < {WINDOW_SIZE}), skipped")

    if not all_fl_features or not all_nf_features:
        print("\nERROR: Insufficient data for V1 test.")
        sys.exit(1)

    X_fl = np.vstack(all_fl_features)  # precursor (failure basin)
    X_nf = np.vstack(all_nf_features)  # nominal

    # Combined training set
    X_train = np.vstack([X_fl, X_nf])
    y_train = np.concatenate([np.ones(len(X_fl)), np.zeros(len(X_nf))])

    print(f"\n  Combined: {len(X_fl)} FL windows + {len(X_nf)} NF windows "
          f"= {len(X_train)} total")
    print(f"  Feature dimensionality: {X_train.shape[1]}")

    # ── Step 3: Handle NaN/Inf in features ────────────────────────────
    nan_mask = np.isnan(X_train) | np.isinf(X_train)
    nan_count = nan_mask.sum()
    if nan_count > 0:
        col_means = np.nanmean(X_train, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0, posinf=0.0, neginf=0.0)
        for j in range(X_train.shape[1]):
            mask = nan_mask[:, j]
            X_train[mask, j] = col_means[j]
        print(f"  Imputed {nan_count} NaN/Inf values with column means")

    # ── Step 4: F → W → M pipeline ───────────────────────────────────
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Causal weighting
    W = build_weight_vector(sensor_map, class_map)
    X_w = X_scaled * W

    # LDA projection to 1D
    lda = LinearDiscriminantAnalysis(n_components=LDA_COMPONENTS, solver="svd")
    lda.fit(X_w, y_train)
    X_proj = lda.transform(X_w).ravel()

    # ── Step 5: Build failure basin and compute k-NN distances ────────
    basin = X_proj[y_train == 1]  # FL projections = failure basin
    train_dists = np.array([dist_to_basin(p, basin) for p in X_proj])

    nom_dists = train_dists[y_train == 0]
    pre_dists = train_dists[y_train == 1]

    # ── Step 6: V1 ───────────────────────────────────────────────────
    v1 = compute_v1(nom_dists, pre_dists)

    print(f"\n{sep}")
    print("  V1 RESULTS")
    print(sep)
    print(f"\n  V1 separation:   {v1['v1_separation']:.1f}x")
    print(f"  p-value:         {v1['p_value']:.2e}")
    print(f"  n_failure (FL):  {v1['n_precursor']}")
    print(f"  n_nominal (NF):  {v1['n_nominal']}")
    print(f"\n  Median distance (nominal):    {v1['median_nominal']:.6f}")
    print(f"  Median distance (precursor):  {v1['median_precursor']:.6f}")
    print(f"  U statistic:                  {v1['u_statistic']:.0f}")

    passed = v1["v1_separation"] > 2.0 and v1["p_value"] < 0.05
    print(f"\n  V1 PASS (sep > 2.0 AND p < 0.05): {'YES' if passed else 'NO'}")
    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
