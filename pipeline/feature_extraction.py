"""Stage 1: Feature extraction F(T) — per paper Section 4.2.

Extracts four feature classes from each trajectory window:
  F_td   — time-domain summaries (mean, std, min, max)
  F_rate — rate of change (first/second derivative statistics)
  F_freq — frequency-domain (FFT magnitudes at low-frequency bins)
  F_cov  — cross-sensor covariance structure (correlation + eigenvalues)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pipeline.config import ACTIVE_SENSORS, FFT_NUM_FREQS, WINDOW_SIZE, WINDOW_STRIDE


def extract_time_domain(window: np.ndarray) -> np.ndarray:
    """Mean, std, min, max of each sensor over the window.

    Args:
        window: (window_size, n_sensors)
    Returns:
        (4 * n_sensors,) feature vector
    """
    return np.concatenate([
        np.mean(window, axis=0),
        np.std(window, axis=0),
        np.min(window, axis=0),
        np.max(window, axis=0),
    ])


def extract_rate(window: np.ndarray) -> np.ndarray:
    """First and second derivative statistics.

    Args:
        window: (window_size, n_sensors)
    Returns:
        (3 * n_sensors,) feature vector
    """
    d1 = np.diff(window, axis=0)  # (window_size-1, n_sensors)
    d2 = np.diff(d1, axis=0)      # (window_size-2, n_sensors)
    return np.concatenate([
        np.mean(d1, axis=0),
        np.std(d1, axis=0),
        np.mean(d2, axis=0),
    ])


def extract_frequency(window: np.ndarray, n_freqs: int = FFT_NUM_FREQS) -> np.ndarray:
    """FFT magnitudes at low-frequency bins for each sensor.

    For C-MAPSS cycle-by-cycle data, "frequency" captures periodicity
    in degradation patterns, not physical Hz.

    Args:
        window: (window_size, n_sensors)
        n_freqs: number of frequency bins to keep (excluding DC)
    Returns:
        (n_freqs * n_sensors,) feature vector
    """
    n_sensors = window.shape[1]
    freqs = np.zeros((n_freqs, n_sensors))
    for i in range(n_sensors):
        fft_vals = np.abs(np.fft.rfft(window[:, i]))
        # Skip DC component (index 0), take next n_freqs bins
        available = min(n_freqs, len(fft_vals) - 1)
        freqs[:available, i] = fft_vals[1:1 + available]
    return freqs.flatten()


def extract_covariance(window: np.ndarray, n_eigenvalues: int = 5) -> np.ndarray:
    """Cross-sensor covariance structure.

    Extracts:
      - Upper triangle of correlation matrix (captures pairwise relationships)
      - Top-k eigenvalues of covariance matrix (captures overall structure)

    Args:
        window: (window_size, n_sensors)
        n_eigenvalues: number of top eigenvalues to include
    Returns:
        (n_sensors*(n_sensors-1)/2 + n_eigenvalues,) feature vector
    """
    n_sensors = window.shape[1]

    # Correlation matrix upper triangle
    corr = np.corrcoef(window, rowvar=False)  # (n_sensors, n_sensors)
    # Handle NaN from constant columns
    corr = np.nan_to_num(corr, nan=0.0)
    upper_tri = corr[np.triu_indices(n_sensors, k=1)]

    # Top eigenvalues of covariance matrix
    cov = np.cov(window, rowvar=False)
    cov = np.nan_to_num(cov, nan=0.0)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    top_eigs = eigenvalues[:min(n_eigenvalues, len(eigenvalues))]
    # Pad if fewer eigenvalues than requested
    if len(top_eigs) < n_eigenvalues:
        top_eigs = np.pad(top_eigs, (0, n_eigenvalues - len(top_eigs)))

    return np.concatenate([upper_tri, top_eigs])


def extract_features(window: np.ndarray) -> np.ndarray:
    """Extract all four feature classes from a single trajectory window.

    Args:
        window: (window_size, n_sensors)
    Returns:
        1-D feature vector
    """
    return np.concatenate([
        extract_time_domain(window),
        extract_rate(window),
        extract_frequency(window),
        extract_covariance(window),
    ])


def get_feature_class_indices(n_sensors: int) -> dict[str, np.ndarray]:
    """Return index ranges for each feature class within the feature vector.

    Used by causal weighting to apply per-class multipliers, and by
    V3 ablation to measure feature class sensitivity.
    """
    idx = 0
    indices = {}

    # Time domain: 4 * n_sensors
    n_td = 4 * n_sensors
    indices["time_domain"] = np.arange(idx, idx + n_td)
    idx += n_td

    # Rate: 3 * n_sensors
    n_rate = 3 * n_sensors
    indices["rate"] = np.arange(idx, idx + n_rate)
    idx += n_rate

    # Frequency: FFT_NUM_FREQS * n_sensors
    n_freq = FFT_NUM_FREQS * n_sensors
    indices["frequency"] = np.arange(idx, idx + n_freq)
    idx += n_freq

    # Covariance: upper_tri + eigenvalues
    n_upper = n_sensors * (n_sensors - 1) // 2
    n_eig = 5  # matches default n_eigenvalues
    indices["covariance"] = np.arange(idx, idx + n_upper + n_eig)

    return indices


def get_feature_sensor_mapping(n_sensors: int, sensor_names: list[str]) -> list[Optional[str]]:
    """Map each feature index to its source sensor name (or None for covariance).

    Used by causal weighting to apply per-sensor weights.
    """
    mapping = []

    # Time domain: [mean_s1, mean_s2, ..., std_s1, ..., min_s1, ..., max_s1, ...]
    for _ in range(4):  # mean, std, min, max
        mapping.extend(sensor_names)

    # Rate: [mean_d1_s1, ..., std_d1_s1, ..., mean_d2_s1, ...]
    for _ in range(3):  # mean_d1, std_d1, mean_d2
        mapping.extend(sensor_names)

    # Frequency: [freq1_s1, freq2_s1, ..., freq1_s2, ...]
    for sensor in sensor_names:
        mapping.extend([sensor] * FFT_NUM_FREQS)

    # Covariance: no single-sensor attribution
    n_upper = n_sensors * (n_sensors - 1) // 2
    n_eig = 5
    mapping.extend([None] * (n_upper + n_eig))

    return mapping


def build_feature_matrix(
    engines: dict[int, np.ndarray],
    rul_dict: dict[int, np.ndarray] | None = None,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> tuple[np.ndarray, dict]:
    """Extract windowed features for all engines.

    Args:
        engines: {unit_id: (n_cycles, n_sensors) array}
        rul_dict: {unit_id: (n_cycles,) RUL array}, optional
        window_size: sliding window length
        stride: window stride

    Returns:
        features: (n_total_windows, feature_dim) array
        meta: dict with 'unit_id', 'cycle_end', 'rul' arrays
    """
    all_features = []
    meta_unit = []
    meta_cycle = []
    meta_rul = []

    for uid, data in engines.items():
        n_cycles = data.shape[0]
        if n_cycles < window_size:
            continue

        rul = rul_dict[uid] if rul_dict is not None else None

        for start in range(0, n_cycles - window_size + 1, stride):
            end = start + window_size
            window = data[start:end]
            feats = extract_features(window)
            all_features.append(feats)
            meta_unit.append(uid)
            meta_cycle.append(end - 1)  # last cycle index in the window
            if rul is not None:
                meta_rul.append(rul[end - 1])

    features = np.array(all_features)
    meta = {
        "unit_id": np.array(meta_unit),
        "cycle_end": np.array(meta_cycle),
    }
    if meta_rul:
        meta["rul"] = np.array(meta_rul)

    return features, meta
