"""
Curvature experiment configuration.

Tests whether local Ollivier-Ricci curvature at a new sensor-space window fires
earlier than Euclidean STTS distance on a held-out trajectory, with NO failure
training — only a healthy manifold built from 320K protein replicates.

Research question: is STTS detecting centroid drift (flat-geometry echo) or
local curvature deviation from a geodesic (true geometric signal)?

Domain: protein (mdCATH 5sicI00) — already has descriptors on disk.
  - 320K replicates 0-4: healthy manifold (folded state)
  - 450K replicates 0-4: trajectories that unfold — held out, no training
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# -- Data paths ------------------------------------------------------
DESCRIPTORS_DIR = PROJECT_ROOT / "data" / "protein" / "descriptors" / "5sicI00"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "geometry"
RESULTS_DIR = PROJECT_ROOT / "results" / "geometry"
FIG_DIR = RESULTS_DIR / "figures"


# -- Experiment structure -------------------------------------------

# Healthy manifold: all 5 replicates at 320K.
# Intermediate temps (348, 379, 413) left unused here — reserved for
# follow-up "curvature vs temperature" experiment.
HEALTHY_TEMP = 320
HEALTHY_REPLICATES = [0, 1, 2, 3, 4]

# Test: held-out 450K replicates. No failure training.
TEST_TEMP = 450
TEST_REPLICATES = [0, 1, 2, 3, 4]


# -- Windowing (same as protein/config.py for fair comparison) ------

WINDOW_SIZE = 20          # frames (ns) per window
WINDOW_STRIDE = 5         # stride for healthy manifold construction
WINDOW_STRIDE_EVAL = 1    # dense evaluation on test trajectories


# -- Feature extraction ---------------------------------------------

# Raw-channel window features. Keep it simple for the curvature probe:
# mean of each channel over the window. The geometry we're probing is
# the manifold of sensor-space states, not the feature space above it.
# Using means preserves the physical dimensionality (8 descriptor channels).
#
# Rationale: the curvature hypothesis says local geometry IS the signal.
# Feeding in 92-dim engineered features conflates the geometry of the
# feature map with the geometry of the underlying sensor manifold.
# Start with the sensor space itself (8D) and add richer embeddings only
# if the 8D probe is silent.
N_CHANNELS = 8


# -- kNN graph ------------------------------------------------------

# Sweep range for k in the kNN graph. Selected by STABILITY of Ricci
# values on 320K-only data — NOT by performance on the held-out test.
K_SWEEP = [5, 10, 15, 20, 30]

# After sweep, the chosen k is locked here (initially None; filled after
# stability analysis completes). Downstream scripts read K_FINAL.
K_FINAL = 15  # locked from k_sweep on 320K-only data (2026-04-16)


# -- Ollivier-Ricci --------------------------------------------------

# Probability measure on each node's neighborhood. Standard choice:
#   alpha = 0.5: half mass on node itself, half uniform on neighbors
# alpha = 0 puts all mass on neighbors (standard OR); alpha near 1 makes
# the measure concentrate on the node and curvature saturates near 1.
ALPHA_OR = 0.5


# -- Detection metric -----------------------------------------------

# Matched false-positive rate on healthy 320K windows. At this FPR,
# threshold both curvature and distance; compare first-alarm lead time
# on held-out 450K replicates.
FPR_TARGET = 0.05

# Require this many consecutive above-threshold windows for a "sustained"
# alarm. Single-window false positives are common; sustained ones are not.
SUSTAINED_WINDOW = 5


# -- Q failure threshold (from protein config, re-stated here) ------

# Q < 0.3 defines "unfolded" — we use the same threshold as the protein
# validation for the RUL definition. RUL = frames remaining until first
# permanent crossing.
Q_FAILURE_THRESHOLD = 0.3


# -- Reproducibility ------------------------------------------------

RANDOM_SEED = 42


def config_snapshot() -> dict:
    """Full config for embedding in results JSON."""
    return {
        "healthy": {
            "temp": HEALTHY_TEMP,
            "replicates": HEALTHY_REPLICATES,
        },
        "test": {
            "temp": TEST_TEMP,
            "replicates": TEST_REPLICATES,
        },
        "windowing": {
            "window_size": WINDOW_SIZE,
            "stride_train": WINDOW_STRIDE,
            "stride_eval": WINDOW_STRIDE_EVAL,
        },
        "features": {
            "space": "raw_channel_means",
            "n_channels": N_CHANNELS,
        },
        "knn": {
            "k_sweep": K_SWEEP,
            "k_final": K_FINAL,
        },
        "ollivier_ricci": {
            "alpha": ALPHA_OR,
        },
        "detection": {
            "fpr_target": FPR_TARGET,
            "sustained_window": SUSTAINED_WINDOW,
        },
        "rul": {
            "q_failure_threshold": Q_FAILURE_THRESHOLD,
        },
        "random_seed": RANDOM_SEED,
    }
