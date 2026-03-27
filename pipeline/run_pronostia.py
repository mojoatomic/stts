"""STTS validation pipeline -- PRONOSTIA bearing degradation.

Run: python -m pipeline.run_pronostia

Validates the STTS framework on vibrational degradation:
IEEE PHM 2012 Prognostics Challenge (PRONOSTIA) accelerated bearing
degradation data.  2-channel accelerometer at 25.6 kHz, 6 training
bearings run to failure across 3 operating conditions, 11 test bearings.

Pipeline order (same as C-MAPSS and Battery):
  1. Load accelerometer snapshots (2560 samples / 0.1 s per file)
  2. Extract per-snapshot vibration features F  (25 features)
  3. Build sliding-window trajectory features  (20 snapshots -> 75 dims)
  4. StandardScaler on raw features
  5. W = uniform (no domain-specific causal chain)
  6. M = 1-component regularised LDA on RUL-bucketed classes
  7. Build failure basin B_f, run monitoring query
  8. Evaluate: verification conditions V1, V2 on training and test sets
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from pipeline.failure_basin import (
    build_failure_basin, build_index, distance_to_basin, distance_to_corpus,
)
from pipeline.evaluation import (
    verify_v1, verify_v2, calibrate_epsilon,
)

# -- Configuration -------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = (
    PROJECT_ROOT / "data" / "pronostia"
    / "ieee-phm-2012-data-challenge-dataset-master"
)
LEARNING_DIR = DATA_ROOT / "Learning_set"
FULL_TEST_DIR = DATA_ROOT / "Full_Test_Set"
RESULTS_DIR = PROJECT_ROOT / "results" / "pronostia"

# Training bearings (run to failure)
TRAIN_BEARINGS = [
    "Bearing1_1", "Bearing1_2",  # Condition 1: 1800 rpm, 4 kN
    "Bearing2_1", "Bearing2_2",  # Condition 2: 1650 rpm, 4.2 kN
    "Bearing3_1", "Bearing3_2",  # Condition 3: 1500 rpm, 5 kN
]

# Test bearings (full run-to-failure from Full_Test_Set)
TEST_BEARINGS = [
    "Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7",
    "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7",
    "Bearing3_3",
]

# Accelerometer parameters
SAMPLE_RATE = 25_600        # Hz
SAMPLES_PER_SNAPSHOT = 2560  # 0.1 seconds at 25.6 kHz

# FFT band boundaries (8 bands spanning 0 - 12.8 kHz Nyquist)
N_FFT_BANDS = 8

# Sliding window
WINDOW_SIZE = 20           # consecutive snapshots per window
WINDOW_STRIDE = 1

# Basin / LDA
RUL_CLIP = 275             # cap RUL — covers ~half of shortest bearing's life
BASIN_RUL_THRESHOLD = 30   # last 30 snapshots = failure basin
BASIN_K = 3
WARNING_RUL = 30           # bearings with RUL <= 30 should be flagged
N_LDA_CLASSES = 6          # RUL bucket count for LDA

N_SNAPSHOT_FEATURES = 25   # 8 time-domain + 16 spectral + 1 cross-channel


# -- Data loading ---------------------------------------------------------

def _detect_delimiter(filepath: Path) -> str:
    """Auto-detect CSV delimiter (comma or semicolon) from first line."""
    with open(filepath, "r") as fh:
        first_line = fh.readline()
    if ";" in first_line:
        return ";"
    return ","


def load_bearing(bearing_dir: Path) -> np.ndarray:
    """Load all accelerometer snapshots for one bearing.

    Each acc_NNNNN.csv has 2560 rows and 6 columns:
        hour, minute, second, microsecond, horiz_accel, vert_accel
    Some files use comma delimiters, others use semicolons.

    Returns:
        snapshots: (n_snapshots, 2560, 2) -- last dim is [horiz, vert]
    """
    csv_files = sorted(bearing_dir.glob("acc_*.csv"))
    # Detect delimiter from the first file
    delim = _detect_delimiter(csv_files[0])
    snapshots = []
    for f in csv_files:
        data = np.loadtxt(str(f), delimiter=delim, usecols=(4, 5))
        snapshots.append(data)
    return np.array(snapshots)  # (n_files, 2560, 2)


# -- Per-snapshot feature extraction --------------------------------------

def extract_snapshot_features(snapshot: np.ndarray) -> np.ndarray:
    """Extract 25 vibration features from one snapshot (2560, 2).

    Per channel (x2):
      Time-domain (4): RMS, peak, crest factor, kurtosis
      Spectral   (8): FFT energy in 8 frequency bands (log-scaled)

    Cross-channel (1): Pearson correlation between horizontal and vertical

    Total: 2*(4+8) + 1 = 25 features
    """
    horiz = snapshot[:, 0]
    vert = snapshot[:, 1]

    feats = []
    for signal in [horiz, vert]:
        # Time-domain
        rms = np.sqrt(np.mean(signal ** 2))
        peak = np.max(np.abs(signal))
        crest = peak / max(rms, 1e-12)
        kurt = float(stats.kurtosis(signal, fisher=True))

        # Spectral: FFT energy in N_FFT_BANDS bands (log-scaled)
        fft_vals = np.fft.rfft(signal)
        fft_power = np.abs(fft_vals) ** 2
        # Split into equal-width frequency bands
        n_bins = len(fft_power)
        band_edges = np.linspace(0, n_bins, N_FFT_BANDS + 1, dtype=int)
        band_energy = []
        for i in range(N_FFT_BANDS):
            e = np.sum(fft_power[band_edges[i]:band_edges[i + 1]])
            band_energy.append(np.log10(e + 1.0))

        feats.extend([rms, peak, crest, kurt])
        feats.extend(band_energy)

    # Cross-channel correlation
    cc = np.corrcoef(horiz, vert)[0, 1]
    if np.isnan(cc):
        cc = 0.0
    feats.append(cc)

    return np.array(feats, dtype=np.float64)


def extract_all_snapshot_features(snapshots: np.ndarray) -> np.ndarray:
    """Extract features for all snapshots.

    Args:
        snapshots: (n_snapshots, 2560, 2)

    Returns:
        features: (n_snapshots, 25)
    """
    return np.array([extract_snapshot_features(s) for s in snapshots])


# -- Trajectory features (windowed) --------------------------------------

def build_trajectory_features(
    snapshot_features: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Build windowed trajectory features from per-snapshot features.

    For each window of consecutive snapshots, extract:
      - mean of each snapshot feature   (25 features)
      - std of each snapshot feature     (25 features)
      - rate of change (mean 1st diff)   (25 features)

    Returns:
        features: (n_windows, 75)
        window_ruls: (n_windows,) RUL at end of each window (in snapshot steps)
    """
    n_snapshots = len(snapshot_features)
    if n_snapshots < window_size:
        return np.empty((0, 0)), np.empty(0)

    # RUL: distance from end (last snapshot = RUL 0), clipped at RUL_CLIP
    rul = np.arange(n_snapshots - 1, -1, -1, dtype=float)
    rul = np.clip(rul, 0, RUL_CLIP)

    all_features = []
    all_ruls = []

    for start in range(0, n_snapshots - window_size + 1, stride):
        end = start + window_size
        window = snapshot_features[start:end]  # (window_size, 25)

        # Time-domain statistics
        w_mean = np.mean(window, axis=0)
        w_std = np.std(window, axis=0)

        # Rate of change: mean of first differences
        d1 = np.diff(window, axis=0)
        rate_mean = np.mean(d1, axis=0)

        feat = np.concatenate([w_mean, w_std, rate_mean])
        all_features.append(feat)
        all_ruls.append(rul[end - 1])

    return np.array(all_features), np.array(all_ruls)


# -- LDA helpers ----------------------------------------------------------

def rul_bucket_labels(rul: np.ndarray, n_classes: int = N_LDA_CLASSES) -> np.ndarray:
    """Bucket RUL values into classes for LDA."""
    boundaries = np.linspace(0, RUL_CLIP, n_classes + 1)
    labels = np.digitize(rul, boundaries) - 1
    return np.clip(labels, 0, n_classes - 1)


# -- Main -----------------------------------------------------------------

def main():
    # Suppress LDA internal overflow warnings (unused discriminant components
    # may contain inf scalings; only component 1 is used and is always finite)
    warnings.filterwarnings("ignore", category=RuntimeWarning,
                            module="sklearn.discriminant_analysis")

    print("=== STTS PRONOSTIA Bearing Validation Pipeline ===")
    print(f"    Training bearings: {len(TRAIN_BEARINGS)}")
    print(f"    Test bearings:     {len(TEST_BEARINGS)}")
    print(f"    Snapshot features: {N_SNAPSHOT_FEATURES}")
    print(f"    Window:            {WINDOW_SIZE} snapshots, stride {WINDOW_STRIDE}")
    print(f"    Trajectory dim:    {N_SNAPSHOT_FEATURES * 3}")
    print()

    # --- 1. Load all bearing data ---
    print("1. Loading bearing data...")
    train_data = {}
    for bname in TRAIN_BEARINGS:
        bdir = LEARNING_DIR / bname
        snapshots = load_bearing(bdir)
        feats = extract_all_snapshot_features(snapshots)
        train_data[bname] = feats
        print(f"   {bname}: {len(snapshots)} snapshots -> {feats.shape}")

    test_data = {}
    for bname in TEST_BEARINGS:
        bdir = FULL_TEST_DIR / bname
        snapshots = load_bearing(bdir)
        feats = extract_all_snapshot_features(snapshots)
        test_data[bname] = feats
        print(f"   {bname}: {len(snapshots)} snapshots -> {feats.shape}")

    # --- 2. Build trajectory features ---
    print("\n2. Building trajectory features (window={}, stride={})...".format(
        WINDOW_SIZE, WINDOW_STRIDE))

    train_traj = {}
    train_feats_list = []
    train_ruls_list = []
    for bname in TRAIN_BEARINGS:
        feats, ruls = build_trajectory_features(train_data[bname])
        train_traj[bname] = (feats, ruls)
        if len(feats) > 0:
            train_feats_list.append(feats)
            train_ruls_list.append(ruls)
        print(f"   {bname}: {len(feats)} windows")

    train_features = np.vstack(train_feats_list)
    train_rul = np.concatenate(train_ruls_list)
    print(f"   Total training windows: {train_features.shape[0]}, "
          f"dim: {train_features.shape[1]}")

    test_traj = {}
    for bname in TEST_BEARINGS:
        feats, ruls = build_trajectory_features(test_data[bname])
        test_traj[bname] = (feats, ruls)
        print(f"   {bname}: {len(feats)} windows")

    # --- 3. StandardScaler + LDA ---
    print("\n3. StandardScaler + regularised LDA (1 component)...")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)

    # Drop features with near-zero variance after scaling
    feature_std = np.std(train_scaled, axis=0)
    good_features = feature_std > 1e-8
    train_scaled = train_scaled[:, good_features]
    n_dropped = (~good_features).sum()
    if n_dropped > 0:
        print(f"   Dropped {n_dropped} near-constant features")

    labels = rul_bucket_labels(train_rul)
    unique_labels = np.unique(labels)
    print(f"   LDA classes: {len(unique_labels)} (from {N_LDA_CLASSES} buckets)")

    lda = LinearDiscriminantAnalysis(
        n_components=1, solver="eigen", shrinkage="auto",
    )
    train_lda = lda.fit_transform(train_scaled, labels)
    print(f"   LDA projection: {train_scaled.shape[1]}d -> {train_lda.shape[1]}d")

    # --- 4. Build failure basin from training data ---
    print(f"\n4. Building failure basin (RUL <= {BASIN_RUL_THRESHOLD})...")
    basin = build_failure_basin(train_lda, train_rul, BASIN_RUL_THRESHOLD)
    basin_index = build_index(basin)
    print(f"   Basin size: {len(basin)} embeddings")

    # --- 5. Training verification ---
    print("\n5. Training verification conditions")

    train_distances = distance_to_basin(train_lda, basin_index, BASIN_K)
    precursor_mask = train_rul <= BASIN_RUL_THRESHOLD
    nominal_mask = train_rul > BASIN_RUL_THRESHOLD + 20

    v1_train = verify_v1(train_distances[precursor_mask],
                         train_distances[nominal_mask])
    v2_train = verify_v2(train_distances, train_rul)

    sep_ratio = v1_train["separation_ratio"]
    print(f"   V1 (train): {'PASS' if v1_train['passed'] else 'FAIL'} -- "
          f"sep={sep_ratio:.1f}x, p={v1_train['mannwhitney_p']:.2e}")
    print(f"   V2 (train): {'PASS' if v2_train['passed'] else 'FAIL'} -- "
          f"rho={v2_train['spearman_rho']:.3f}, p={v2_train['p_value']:.2e}")

    # Calibrate epsilon
    epsilon = calibrate_epsilon(train_distances, train_rul, BASIN_RUL_THRESHOLD)
    print(f"   epsilon (calibrated): {epsilon:.4f}")

    # --- 6. Test set evaluation (V2 per bearing) ---
    print("\n6. Test set evaluation (V2 per bearing)")

    test_results = []
    for bname in TEST_BEARINGS:
        feats, ruls = test_traj[bname]
        if len(feats) == 0:
            print(f"   {bname}: SKIP (no windows)")
            continue

        test_scaled = scaler.transform(feats)[:, good_features]
        test_lda = lda.transform(test_scaled)
        test_distances = distance_to_basin(test_lda, basin_index, BASIN_K)

        v2_test = verify_v2(test_distances, ruls)
        test_results.append({
            "bearing": bname,
            "v2_rho": v2_test["spearman_rho"],
            "v2_p": v2_test["p_value"],
            "n_windows": len(feats),
            "final_dist": float(test_distances[-1]),
        })

        print(f"   {bname}: V2 rho={v2_test['spearman_rho']:.3f}, "
              f"p={v2_test['p_value']:.2e}, n={len(feats)}")

    # --- 7. Summary ---
    rhos = [r["v2_rho"] for r in test_results]
    positive_rho = sum(1 for r in rhos if r > 0)
    n_gt_half = sum(1 for r in rhos if r > 0.5)
    n_negative = sum(1 for r in rhos if r < 0)
    mean_rho = np.mean(rhos)

    print(f"\n7. Summary")
    print(f"   V1 training separation: {sep_ratio:.1f}x")
    print(f"   V2 training rho:        {v2_train['spearman_rho']:.3f}")
    print(f"   Test bearings with V2 rho > 0.5:  {n_gt_half}/{len(test_results)}")
    print(f"   Test bearings with V2 rho > 0:    {positive_rho}/{len(test_results)}")
    print(f"   Test bearings with V2 rho < 0:    {n_negative}/{len(test_results)}")
    print(f"   Mean test V2 rho:                 {mean_rho:.3f}")

    # --- 8. Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # V2 per test bearing (matches results/pronostia_results.csv format)
    results_df = pd.DataFrame(test_results)
    results_df[["bearing", "v2_rho"]].to_csv(
        RESULTS_DIR / "test_v2_results.csv", index=False,
    )

    # Full results
    results_df.to_csv(RESULTS_DIR / "test_results.csv", index=False)

    # Training summary
    summary = {
        "v1_sep_ratio": sep_ratio,
        "v1_p": v1_train["mannwhitney_p"],
        "v2_train_rho": v2_train["spearman_rho"],
        "v2_train_p": v2_train["p_value"],
        "epsilon": epsilon,
        "n_train_windows": len(train_features),
        "basin_size": len(basin),
        "n_test_bearings": len(test_results),
        "mean_test_v2_rho": mean_rho,
        "positive_rho_count": positive_rho,
    }
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "training_summary.csv", index=False)

    print(f"\n   Results saved to {RESULTS_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
