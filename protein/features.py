"""
Stage F: Feature extraction for protein unfolding trajectories.

Extracts four feature classes from each descriptor window:
  F_td    — time-domain summaries (mean, std, min, max per channel)
  F_rate  — rate of change (d/dt mean, d/dt std, d^2/dt^2 mean per channel)
  F_ratio — late/early window ratio per channel (approach discriminator)
  F_cross — cross-channel correlation structure (upper triangle)

The window slides over structural descriptor time series (1 frame = 1 ns).
All 8 descriptor channels have uniform 1 ns cadence, so rate features
are simple finite differences (no time normalization needed, unlike TLE).
"""

from __future__ import annotations

import numpy as np

from protein.config import (
    N_CHANNELS,
    N_CROSS_FEATURES,
    N_FEATURES,
    N_RATE_FEATURES,
    N_RATIO_FEATURES,
    N_TD_FEATURES,
    PRECURSOR_FRAMES,
    NOMINAL_BUFFER_FRAMES,
    WINDOW_SIZE,
)


def extract_time_domain(window: np.ndarray) -> np.ndarray:
    """Mean, std, min, max of each channel over the window.

    Args:
        window: (window_size, n_channels)
    Returns:
        (4 * n_channels,) feature vector
    """
    return np.concatenate([
        np.mean(window, axis=0),
        np.std(window, axis=0),
        np.min(window, axis=0),
        np.max(window, axis=0),
    ])


def extract_rate(window: np.ndarray) -> np.ndarray:
    """Rate features: first and second derivatives.

    Uniform 1 ns frame spacing — no time normalization needed.

    Args:
        window: (window_size, n_channels)
    Returns:
        (3 * n_channels,) feature vector: [mean_d1, std_d1, mean_d2]
    """
    d1 = np.diff(window, axis=0)  # (window_size-1, n_channels)

    if len(d1) > 1:
        d2 = np.diff(d1, axis=0)  # (window_size-2, n_channels)
    else:
        d2 = np.zeros((1, window.shape[1]))

    return np.concatenate([
        np.mean(d1, axis=0),
        np.std(d1, axis=0),
        np.mean(d2, axis=0),
    ])


def extract_ratio(window: np.ndarray) -> np.ndarray:
    """Late/early window ratio per channel.

    Ratio = mean(second half) / mean(first half).
    Values >> 1 or << 1 indicate monotonic change within the window.
    Safe division with offset to handle zero denominators.

    Args:
        window: (window_size, n_channels)
    Returns:
        (n_channels,) feature vector
    """
    mid = window.shape[0] // 2
    early = np.mean(window[:mid], axis=0)
    late = np.mean(window[mid:], axis=0)

    denom = np.abs(early) + 1e-10
    return late / denom


def extract_cross(window: np.ndarray) -> np.ndarray:
    """Cross-channel correlation upper triangle.

    Captures coupled unfolding dynamics (e.g., Q-RMSD correlation
    tightens during cooperative unfolding events).

    Args:
        window: (window_size, n_channels)
    Returns:
        (n_channels * (n_channels - 1) / 2,) feature vector
    """
    corr = np.corrcoef(window, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    upper = corr[np.triu_indices(window.shape[1], k=1)]
    return upper


def extract_features(window: np.ndarray) -> np.ndarray:
    """Extract all four feature classes from a single descriptor window.

    Args:
        window: (window_size, n_channels) structural descriptor values
    Returns:
        (N_FEATURES,) feature vector
    """
    feat = np.concatenate([
        extract_time_domain(window),
        extract_rate(window),
        extract_ratio(window),
        extract_cross(window),
    ])
    assert len(feat) == N_FEATURES, \
        f"Feature vector length {len(feat)} != expected {N_FEATURES}"
    return feat


def get_feature_class_indices() -> dict[str, np.ndarray]:
    """Return index ranges for each feature class (for V3 ablation).

    Returns:
        dict mapping class name to numpy index array
    """
    return {
        "time_domain": np.arange(0, N_TD_FEATURES),
        "rate": np.arange(N_TD_FEATURES, N_TD_FEATURES + N_RATE_FEATURES),
        "ratio": np.arange(
            N_TD_FEATURES + N_RATE_FEATURES,
            N_TD_FEATURES + N_RATE_FEATURES + N_RATIO_FEATURES,
        ),
        "cross": np.arange(
            N_TD_FEATURES + N_RATE_FEATURES + N_RATIO_FEATURES,
            N_FEATURES,
        ),
    }


def build_feature_matrix(
    descriptors_dict: dict[str, np.ndarray],
    rul_dict: dict[str, np.ndarray],
    window_size: int = WINDOW_SIZE,
    stride: int = 1,
    label_mode: str = "binary",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract windowed features for a set of trajectories.

    Labeling (binary mode):
      - Trajectories with classification="healthy" (320K): all windows label 0
      - Trajectories with classification="unfolding" (450K):
          windows within PRECURSOR_FRAMES of failure = 1 (precursor)
          windows > NOMINAL_BUFFER_FRAMES before failure = -1 (ambiguous)
          windows in between = -1 (transition zone, excluded from training)

    Args:
        descriptors_dict: {trajectory_key: (n_frames, 8) descriptor array}
        rul_dict: {trajectory_key: (n_frames,) RUL array}
            Use np.inf for healthy trajectories (no failure).
        window_size: frames per window
        stride: window advance step
        label_mode: "binary" for STTS training, "rul_only" for mechanism experiment

    Returns:
        X: (n_windows, N_FEATURES) feature matrix
        y: (n_windows,) labels — 1=precursor, 0=nominal, -1=ambiguous
        rul_at_window: (n_windows,) RUL at end of each window
        window_ids: list of "traj_key:window_start" for traceability
    """
    all_features = []
    all_labels = []
    all_ruls = []
    all_ids = []

    for traj_key, descriptors in descriptors_dict.items():
        rul = rul_dict[traj_key]
        n_frames = len(descriptors)

        if n_frames < window_size:
            continue

        is_healthy = np.all(np.isinf(rul))

        for start in range(0, n_frames - window_size + 1, stride):
            end = start + window_size
            window = descriptors[start:end]
            feat = extract_features(window)
            window_rul = rul[end - 1]

            if is_healthy:
                label = 0  # nominal
            elif label_mode == "rul_only":
                label = 0  # not used, just need RUL
            else:
                if window_rul <= PRECURSOR_FRAMES:
                    label = 1  # precursor
                elif window_rul >= NOMINAL_BUFFER_FRAMES:
                    label = -1  # ambiguous: far from failure
                else:
                    label = -1  # transition zone

            all_features.append(feat)
            all_labels.append(label)
            all_ruls.append(window_rul)
            all_ids.append(f"{traj_key}:{start}")

    X = np.array(all_features) if all_features else np.zeros((0, N_FEATURES))
    y = np.array(all_labels)
    ruls = np.array(all_ruls)

    return X, y, ruls, all_ids
