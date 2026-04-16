"""
Data loading and windowing for the curvature experiment.

Uses the existing protein descriptor files on disk — this module does not
download or reprocess anything. All pre-computed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from geometry.config import (
    DESCRIPTORS_DIR,
    HEALTHY_REPLICATES,
    HEALTHY_TEMP,
    N_CHANNELS,
    Q_FAILURE_THRESHOLD,
    TEST_REPLICATES,
    TEST_TEMP,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    WINDOW_STRIDE_EVAL,
)


def load_descriptor(temp: int, rep: int) -> np.ndarray:
    """Load the 8-channel descriptor array for one trajectory.

    Returns (n_frames, 8) array.
    """
    path = DESCRIPTORS_DIR / f"{temp}K_rep{rep}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Descriptor file not found: {path}")
    desc = np.load(path)
    assert desc.shape[1] == N_CHANNELS, \
        f"Expected {N_CHANNELS} channels, got {desc.shape[1]}"
    return desc


def window_means(
    descriptor: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over the descriptor, compute per-channel mean in each window.

    Returns:
        features: (n_windows, n_channels) array of window-mean vectors
        end_frames: (n_windows,) frame index of the last frame in each window
    """
    n_frames = len(descriptor)
    if n_frames < window_size:
        return np.zeros((0, descriptor.shape[1])), np.zeros(0, dtype=int)

    starts = np.arange(0, n_frames - window_size + 1, stride)
    features = np.array([
        descriptor[s:s + window_size].mean(axis=0) for s in starts
    ])
    end_frames = starts + window_size - 1
    return features, end_frames


def load_healthy_manifold() -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Build the healthy manifold point cloud from all 320K replicates.

    Returns:
        points: (n_total_windows, n_channels) array of window-mean features
        origin: list of (replicate, end_frame) per window — for provenance
    """
    all_points = []
    origin = []
    for rep in HEALTHY_REPLICATES:
        desc = load_descriptor(HEALTHY_TEMP, rep)
        feats, ends = window_means(desc, WINDOW_SIZE, WINDOW_STRIDE)
        all_points.append(feats)
        origin.extend([(rep, int(e)) for e in ends])
    points = np.concatenate(all_points, axis=0)
    return points, origin


def compute_rul(q_trajectory: np.ndarray) -> tuple[np.ndarray, int | None]:
    """RUL from native contact fraction Q. Failure = first permanent crossing below threshold.

    Returns (rul_array, failure_frame). If no permanent crossing, failure_frame is None
    and rul is all np.inf.
    """
    n = len(q_trajectory)
    failure_frame = None
    for i in range(n):
        if q_trajectory[i] < Q_FAILURE_THRESHOLD:
            if np.all(q_trajectory[i:] < Q_FAILURE_THRESHOLD):
                failure_frame = i
                break

    if failure_frame is None:
        return np.full(n, np.inf), None

    rul = np.zeros(n)
    for i in range(n):
        rul[i] = max(0, failure_frame - i)
    return rul, failure_frame


def load_test_trajectory(rep: int) -> dict:
    """Load one held-out 450K replicate with windowed features and RUL.

    Returns dict with keys:
        descriptor: (n_frames, 8)
        features: (n_windows, 8) dense-evaluation windows (stride 1)
        end_frames: (n_windows,) frame indices
        rul: (n_windows,) frames remaining until failure at each window
        failure_frame: int or None
    """
    desc = load_descriptor(TEST_TEMP, rep)
    features, ends = window_means(desc, WINDOW_SIZE, WINDOW_STRIDE_EVAL)
    q_trajectory = desc[:, 3]  # Q is channel 3
    full_rul, failure_frame = compute_rul(q_trajectory)

    # RUL at the END of each window
    window_rul = np.array([full_rul[e] for e in ends])

    return {
        "descriptor": desc,
        "features": features,
        "end_frames": ends,
        "rul": window_rul,
        "failure_frame": failure_frame,
    }
