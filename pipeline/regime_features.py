"""Per-regime feature extraction for multi-operating-condition datasets.

The FD002/FD004 precision collapse is caused by operating-regime variation
leaking into the embedding space. The fix: extract features WITHIN each
operating regime separately, so regime-dependent baseline variation is
removed before the degradation signal is computed.

Pipeline for multi-condition datasets:
  1. Cluster operating conditions into regimes (k-means on settings)
  2. For each trajectory window, identify its regime
  3. Normalize the window's sensor readings using regime-specific statistics
  4. Extract features from the regime-normalized window
  5. Apply W and M as usual
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pipeline.config import ACTIVE_SENSORS, SETTING_COLS, WINDOW_SIZE, WINDOW_STRIDE
from pipeline.feature_extraction import extract_features


def identify_regimes(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_regimes: int = 6,
) -> tuple[np.ndarray, np.ndarray, KMeans]:
    """Cluster operating settings into discrete regimes.

    Returns:
        train_regimes: (n_train_rows,) regime labels
        test_regimes: (n_test_rows,) regime labels
        km: fitted KMeans model
    """
    settings_cols = [c for c in SETTING_COLS if c in train_df.columns]
    train_settings = train_df[settings_cols].round(3).values
    test_settings = test_df[settings_cols].round(3).values

    km = KMeans(n_clusters=n_regimes, n_init=10, random_state=42)
    train_regimes = km.fit_predict(train_settings)
    test_regimes = km.predict(test_settings)

    return train_regimes, test_regimes, km


def compute_regime_statistics(
    train_df: pd.DataFrame,
    train_regimes: np.ndarray,
    sensor_cols: list[str],
    n_regimes: int = 6,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Compute per-regime mean and std for each sensor.

    Returns:
        {regime_id: (mean_vector, std_vector)}
    """
    stats = {}
    for regime in range(n_regimes):
        mask = train_regimes == regime
        if mask.sum() == 0:
            continue
        data = train_df.loc[mask, sensor_cols].values
        stats[regime] = (data.mean(axis=0), data.std(axis=0) + 1e-10)
    return stats


def extract_regime_normalized_features(
    engine_sensors: np.ndarray,
    engine_regimes: np.ndarray,
    regime_stats: dict[int, tuple[np.ndarray, np.ndarray]],
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> tuple[np.ndarray, list[int]]:
    """Extract features from regime-normalized windows.

    For each window, identify its dominant regime (mode of regime labels
    within the window), normalize the window using that regime's statistics,
    then extract features from the normalized window.

    Args:
        engine_sensors: (n_cycles, n_sensors) raw sensor values
        engine_regimes: (n_cycles,) regime labels per cycle
        regime_stats: {regime_id: (mean, std)} from training data
        window_size: sliding window length
        stride: window stride

    Returns:
        features: (n_windows, feature_dim) array
        window_regimes: dominant regime per window
    """
    n_cycles = engine_sensors.shape[0]
    if n_cycles < window_size:
        return np.array([]).reshape(0, 0), []

    all_features = []
    window_regimes = []

    for start in range(0, n_cycles - window_size + 1, stride):
        end = start + window_size
        window = engine_sensors[start:end]
        w_regimes = engine_regimes[start:end]

        # Dominant regime in this window
        regime = int(np.bincount(w_regimes).argmax())
        window_regimes.append(regime)

        if regime in regime_stats:
            mean, std = regime_stats[regime]
            normalized = (window - mean) / std
        else:
            # Fallback: global z-score (shouldn't happen with well-clustered data)
            normalized = (window - window.mean(axis=0)) / (window.std(axis=0) + 1e-10)

        feats = extract_features(normalized)
        all_features.append(feats)

    if len(all_features) == 0:
        return np.array([]).reshape(0, 0), []

    return np.array(all_features), window_regimes


def build_regime_feature_matrix(
    engines_df: dict[int, pd.DataFrame],
    regime_labels: np.ndarray,
    full_df: pd.DataFrame,
    regime_stats: dict[int, tuple[np.ndarray, np.ndarray]],
    sensor_cols: list[str],
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> tuple[np.ndarray, dict]:
    """Build feature matrix with per-regime normalization for all engines.

    Args:
        engines_df: {uid: DataFrame} per engine
        regime_labels: regime label for every row in the original df
        full_df: the original full dataframe (to map regime labels)
        regime_stats: per-regime statistics
        sensor_cols: active sensor columns
        window_size: sliding window
        stride: stride

    Returns:
        features: (n_total_windows, feature_dim)
        meta: dict with unit_id, cycle_end, rul arrays
    """
    all_features = []
    meta_unit = []
    meta_cycle = []
    meta_rul = []

    for uid, df in engines_df.items():
        sensors = df[sensor_cols].values
        # Get regime labels for this engine's rows
        engine_regime = regime_labels[df.index.values]
        rul = df["rul"].values if "rul" in df.columns else None

        feats, _ = extract_regime_normalized_features(
            sensors, engine_regime, regime_stats, window_size, stride,
        )

        if feats.size == 0:
            continue

        n_windows = feats.shape[0]
        for i in range(n_windows):
            cycle_idx = i * stride + window_size - 1
            all_features.append(feats[i])
            meta_unit.append(uid)
            meta_cycle.append(cycle_idx)
            if rul is not None:
                meta_rul.append(rul[cycle_idx])

    features = np.array(all_features)
    meta = {
        "unit_id": np.array(meta_unit),
        "cycle_end": np.array(meta_cycle),
    }
    if meta_rul:
        meta["rul"] = np.array(meta_rul)

    return features, meta
