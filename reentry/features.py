"""
Stage F: Feature extraction for reentry trajectories.

Extracts four feature classes from each TLE window:
  F_td    — time-domain summaries (mean, std, min, max per channel)
  F_rate  — rate of change (d/dt mean, d/dt std, d²/dt² mean per channel)
  F_ratio — late/early window ratio per channel (approach discriminator)
  F_cross — cross-channel correlation structure (upper triangle)

The window slides over successive TLE records (not fixed time intervals).
TLE update cadence varies: ~2/day during active decay, ~1/week at
operational altitude. Rate features are normalized by inter-epoch time
delta to account for non-uniform spacing.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from reentry.config import (
    N_CHANNELS,
    N_CROSS_FEATURES,
    N_FEATURES,
    N_RATE_FEATURES,
    N_RATIO_FEATURES,
    N_TD_FEATURES,
    PRECURSOR_DAYS,
    NOMINAL_BUFFER_DAYS,
    STATE_CHANNELS,
    WINDOW_SIZE,
)


def _parse_epoch_days(epoch_str: str) -> float:
    """Convert ISO epoch string to fractional days since J2000.

    Used for computing inter-record time deltas for rate normalization.
    """
    # Space-Track epochs: "2024-12-30T06:24:46.664928"
    try:
        dt = datetime.fromisoformat(epoch_str.replace("Z", "+00:00"))
    except ValueError:
        # Fallback for truncated formats
        dt = datetime.strptime(epoch_str[:19], "%Y-%m-%dT%H:%M:%S")
    j2000 = datetime(2000, 1, 1, 12, 0, 0)
    return (dt - j2000).total_seconds() / 86400.0


def tle_records_to_array(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Convert parsed TLE records to arrays.

    Args:
        records: list of dicts with 'epoch' and STATE_CHANNELS keys

    Returns:
        values: (n_records, n_channels) array of state channel values
        epochs: (n_records,) array of epoch in fractional days since J2000
    """
    n = len(records)
    values = np.zeros((n, N_CHANNELS))
    epochs = np.zeros(n)

    for i, rec in enumerate(records):
        epochs[i] = _parse_epoch_days(rec["epoch"])
        for j, ch in enumerate(STATE_CHANNELS):
            values[i, j] = rec[ch]

    return values, epochs


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


def extract_rate(window: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Rate features normalized by inter-epoch time delta.

    Computes d(channel)/dt and d²(channel)/dt² using finite differences
    normalized by actual time between TLE records. This accounts for
    non-uniform TLE update cadence.

    Args:
        window: (window_size, n_channels) state channel values
        dt: (window_size,) epoch in days (for time normalization)
    Returns:
        (3 * n_channels,) feature vector: [mean_d1, std_d1, mean_d2]
    """
    # Time deltas between successive records (days)
    time_delta = np.diff(dt)
    # Clamp to avoid division by zero (duplicate epochs)
    time_delta = np.maximum(time_delta, 1e-6)

    # First derivative: d(channel)/dt
    d_values = np.diff(window, axis=0)  # (window_size-1, n_channels)
    d1 = d_values / time_delta[:, np.newaxis]

    # Second derivative
    if len(d1) > 1:
        time_delta2 = np.maximum(time_delta[:-1], 1e-6)
        d2 = np.diff(d1, axis=0) / time_delta2[:, np.newaxis]
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
    This is the primary precursor discriminator (analogous to the
    late/early |q-1AU| ratio in the orbital paper, weight 4.0).

    For channels that can be zero or negative (BSTAR), uses
    additive offset to avoid division by zero.

    Args:
        window: (window_size, n_channels)
    Returns:
        (n_channels,) feature vector
    """
    mid = window.shape[0] // 2
    early = np.mean(window[:mid], axis=0)
    late = np.mean(window[mid:], axis=0)

    # Safe ratio: offset denominator to avoid div-by-zero
    # Use absolute value of early mean + small epsilon
    denom = np.abs(early) + 1e-10
    ratio = late / denom

    return ratio


def extract_cross(window: np.ndarray) -> np.ndarray:
    """Cross-channel correlation upper triangle.

    Captures coupled decay dynamics (e.g., PERIAPSIS-MEAN_MOTION
    correlation tightens as Kepler's law dominates near reentry).

    Args:
        window: (window_size, n_channels)
    Returns:
        (n_channels * (n_channels - 1) / 2,) feature vector
    """
    corr = np.corrcoef(window, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    upper = corr[np.triu_indices(window.shape[1], k=1)]
    return upper


def extract_features(window: np.ndarray, epochs: np.ndarray) -> np.ndarray:
    """Extract all four feature classes from a single TLE window.

    Args:
        window: (window_size, n_channels) state channel values
        epochs: (window_size,) epoch in days since J2000
    Returns:
        (N_FEATURES,) feature vector
    """
    feat = np.concatenate([
        extract_time_domain(window),
        extract_rate(window, epochs),
        extract_ratio(window),
        extract_cross(window),
    ])
    assert len(feat) == N_FEATURES, \
        f"Feature vector length {len(feat)} != expected {N_FEATURES}"
    return feat


def build_feature_matrix(
    satellites: dict,
    norad_ids: list[str],
    window_size: int = WINDOW_SIZE,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract windowed features for a set of satellites.

    Two-population labeling:
      - Operational satellites: all windows labeled 0 (nominal)
      - Reentry satellites: windows within PRECURSOR_DAYS of decay = 1 (precursor),
        windows > NOMINAL_BUFFER_DAYS before decay = 0 (nominal),
        windows in between = -1 (ambiguous, excluded from training)

    Args:
        satellites: {norad_id: satellite_dict} from corpus
        norad_ids: list of NORAD IDs to process
        window_size: TLE records per window
        stride: window advance step

    Returns:
        X: (n_windows, N_FEATURES) feature matrix
        y: (n_windows,) labels — 1=precursor, 0=nominal, -1=ambiguous
        days_to_reentry: (n_windows,) days from window end to decay epoch
            (np.inf for operational satellites)
        window_ids: list of "NORAD_ID:window_idx" for traceability
    """
    all_features = []
    all_labels = []
    all_days = []
    all_ids = []

    for nid in norad_ids:
        sat = satellites[nid]
        records = sat["tle_records"]
        decay_str = sat.get("decay_epoch")
        classification = sat.get("classification", "")
        is_reentry = classification == "reentry" and decay_str

        # Parse decay epoch for reentry satellites
        decay_days = None
        if is_reentry:
            decay_clean = decay_str.replace(" ", "T") if "T" not in decay_str else decay_str
            decay_days = _parse_epoch_days(decay_clean)

        values, epochs = tle_records_to_array(records)
        n_records = len(records)

        # Adaptive window for short-lived objects (storm satellites)
        effective_window = min(window_size, n_records)
        if effective_window < 3:
            continue

        for start in range(0, n_records - effective_window + 1, stride):
            end = start + effective_window
            window = values[start:end]
            win_epochs = epochs[start:end]

            feat = extract_features(window, win_epochs)

            if is_reentry:
                days_to_decay = decay_days - win_epochs[-1]
                if days_to_decay < 0:
                    break  # past reentry

                if days_to_decay <= PRECURSOR_DAYS:
                    label = 1  # precursor
                elif days_to_decay >= NOMINAL_BUFFER_DAYS:
                    label = -1  # ambiguous: reentry satellite but far from decay
                                # These look operational — excluding prevents
                                # contaminating the nominal class with pre-deorbit
                                # trajectories that will eventually decay
                else:
                    label = -1  # transition zone
            else:
                # Operational satellite — always nominal
                days_to_decay = float("inf")
                label = 0

            all_features.append(feat)
            all_labels.append(label)
            all_days.append(days_to_decay)
            all_ids.append(f"{nid}:{start}")

    X = np.array(all_features) if all_features else np.zeros((0, N_FEATURES))
    y = np.array(all_labels)
    days = np.array(all_days)

    return X, y, days, all_ids
