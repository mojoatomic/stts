"""Central configuration for the STTS N-CMAPSS validation pipeline.

N-CMAPSS (Chao et al. 2021) differs from original C-MAPSS:
  - HDF5 format (.h5) with dev/test splits per file
  - 14 sensors (vs. 21), different set with named symbols
  - 4 flight condition variables (alt, Mach, TRA, T2)
  - Within-flight time-series (seconds) — must aggregate per cycle
  - 9 sub-datasets (DS01..DS08d) with 1-7 failure modes
  - Failure modes affect flow (F) and/or efficiency (E) of 5 subcomponents

Aggregation strategy:
  1. Cruise-phase filter: within each flight cycle, retain only timesteps
     where TRA (throttle-resolver angle) is stable AND altitude > 10,000 ft.
     This removes takeoff/landing transients that are flight-phase dynamics,
     not degradation signal. TRA stability is the primary criterion; altitude
     is a secondary guard.
  2. Per-cycle mean: cruise-phase sensor readings are reduced to per-cycle
     means (14 channels). The downstream windowed feature extractor computes
     time-domain (mean, std, min, max), rate, frequency, and covariance
     statistics over the cycle-level trajectory. Mean-only aggregation avoids
     double-aggregation (e.g., mean-of-means) that would wash out signal.
  3. Regime normalization: flight operating conditions (W: alt, Mach, TRA, T2)
     are used to cluster operating regimes, then sensors are Z-score normalized
     per regime using dev statistics only. This removes operating-point effects
     so the feature extractor measures degradation, not flight profile variation.
     This mirrors the original C-MAPSS normalize_by_regime for FD002/FD004.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "N-CMAPSSData"
RESULTS_DIR = PROJECT_ROOT / "results" / "ncmapss"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"

# HDF5 dataset keys — WHITELIST ONLY.
# N-CMAPSS files also contain X_v (virtual/model-internal sensors) and T
# (theta/health parameters = ground truth degradation state). These are
# explicitly excluded: using them as features would be data leakage.
# Only load physical measurements (W, X_s), labels (Y), and metadata (A).
H5_LOAD_KEYS_DEV = ["W_dev", "X_s_dev", "Y_dev", "A_dev"]
H5_LOAD_KEYS_TEST = ["W_test", "X_s_test", "Y_test", "A_test"]

# Flight condition (scenario descriptor) columns — w
FLIGHT_COND_NAMES = ["alt", "Mach", "TRA", "T2"]

# 14 sensor columns — x_s
# ORDER MATCHES THE HDF5 FILES (verified across DS01, DS02, DS03), NOT the
# challenge spec table order (which lists Wf first). The HDF5 column order is:
# temperatures (T24-T50), pressures (P15-P50), speeds (Nf, Nc), fuel (Wf).
SENSOR_NAMES = [
    "T24",   # 1  Total temperature at LPC outlet (°R)
    "T30",   # 2  Total temperature at HPC outlet (°R)
    "T48",   # 3  Total temperature at HPT outlet (°R)
    "T50",   # 4  Total temperature at LPT outlet (°R)
    "P15",   # 5  Total pressure in bypass-duct (psia)
    "P2",    # 6  Total pressure at fan inlet (psia)
    "P21",   # 7  Total pressure at fan outlet (psia)
    "P24",   # 8  Total pressure at LPC outlet (psia)
    "Ps30",  # 9  Static pressure at HPC outlet (psia)
    "P40",   # 10 Total pressure at burner outlet (psia)
    "P50",   # 11 Total pressure at LPT outlet (psia)
    "Nf",    # 12 Physical fan speed (rpm)
    "Nc",    # 13 Physical core speed (rpm)
    "Wf",    # 14 Fuel flow (pps)
]

# Auxiliary columns
AUX_NAMES = ["unit", "cycle", "Fc", "h_s"]

# All 14 sensors are active (no near-constant sensors to drop — real flight
# conditions create natural variation in all channels)
ACTIVE_SENSORS = SENSOR_NAMES.copy()

# ---------------------------------------------------------------------------
# Causal weighting — adapted for N-CMAPSS 14-sensor layout
#
# Same turbofan causal chain as original C-MAPSS:
#   mechanical degradation → speed change → efficiency loss
#   → pressure change → temperature rise
#
# Physical justification for each assignment documented inline.
# ---------------------------------------------------------------------------
SENSOR_CAUSAL_GROUPS = {
    "high": [
        # Causally upstream — rotor speed is the earliest mechanical indicator
        # of degradation in rotating subcomponents (fan, LPC, HPC, HPT, LPT).
        # Blade erosion, tip clearance changes, and bearing wear manifest here
        # before propagating downstream.
        "Nf",        # physical fan speed — direct measure of fan rotor health
        "Nc",        # physical core speed — direct measure of HPC/HPT rotor health
    ],
    "medium_high": [
        # Mid-chain: pressure measurements respond to efficiency changes in
        # upstream compressor/turbine stages. Fuel flow is controller-mediated
        # (FADEC adjusts Wf to maintain thrust demand), so it reflects
        # degradation indirectly — the controller compensates for efficiency
        # loss by increasing fuel flow.
        "Wf",        # fuel flow — controller-mediated response to degradation,
                     # not a direct physical measurement of degradation itself
        "P21",       # fan outlet pressure — responds to fan efficiency loss
        "P24",       # LPC outlet pressure — responds to LPC degradation
        "Ps30",      # HPC outlet static pressure — responds to HPC degradation
        "P40",       # burner outlet pressure — downstream of combustion,
                     # reflects combined compressor efficiency changes
    ],
    "medium": [
        # Downstream effect: temperature sensors reflect thermodynamic
        # consequences of upstream efficiency loss. Temperature rises when
        # compressor efficiency drops (more work input for same pressure ratio)
        # or turbine efficiency drops (less energy extracted).
        "T24",       # LPC outlet temp — rises with LPC efficiency loss
        "T30",       # HPC outlet temp — rises with HPC efficiency loss
        "T48",       # HPT outlet temp — rises with HPT efficiency loss
        "T50",       # LPT outlet temp — rises with LPT efficiency loss
        "P50",       # LPT outlet pressure — downstream of all turbine stages,
                     # carries cumulative degradation signal from HPT and LPT
    ],
    "low": [
        # Boundary/ambient-driven: fan inlet conditions are set by flight state
        # (altitude, Mach), not by engine health. Bypass-duct pressure is
        # partially ambient-driven but does carry some fan degradation signal.
        "P2",        # fan inlet pressure — determined by altitude and Mach,
                     # not engine degradation; included for completeness only
        "P15",       # bypass-duct pressure — partially ambient-driven, weak
                     # degradation signal compared to core-path sensors
    ],
}
CAUSAL_WEIGHT_VALUES = {
    "high": 2.0,
    "medium_high": 1.5,
    "medium": 1.0,
    "low": 0.7,
}

# Feature class multipliers — same as original pipeline
FEATURE_CLASS_MULTIPLIERS = {
    "time_domain": 1.0,
    "rate": 1.5,
    "frequency": 1.3,
    "covariance": 1.2,
}

# ---------------------------------------------------------------------------
# Cruise-phase detection
#
# Within each flight cycle, we retain only the cruise segment to remove
# takeoff/landing transients. Primary criterion: TRA stability (throttle
# resolver angle has low variance when the engine is at steady cruise thrust).
# Secondary criterion: altitude > 10,000 ft (guards against ground-level
# segments where TRA may also be briefly stable, e.g., taxi).
# ---------------------------------------------------------------------------
CRUISE_TRA_WINDOW = 50          # timesteps for rolling TRA std
CRUISE_TRA_STD_MAX = 5.0        # % — max rolling std(TRA) to be "stable"
CRUISE_ALT_MIN = 10000          # ft — minimum altitude for cruise
CRUISE_MIN_TIMESTEPS = 100      # fallback: if cruise detection yields fewer
                                # than this, use altitude-only filter; if still
                                # fewer, use the full flight

# ---------------------------------------------------------------------------
# Regime normalization
#
# Flight operating conditions (W: alt, Mach, TRA, T2) vary realistically in
# N-CMAPSS. Sensor readings at different operating points are not directly
# comparable — a pressure reading at 35,000 ft / Mach 0.8 differs from the
# same engine's reading at 25,000 ft / Mach 0.6 for purely aerodynamic
# reasons, not degradation. Regime-based normalization removes this effect.
#
# We cluster cycle-level W means into regimes (KMeans on dev data), then
# Z-score normalize sensors per regime using dev statistics only. Test data
# is assigned to the nearest dev cluster and normalized with dev statistics.
# This mirrors the original C-MAPSS normalize_by_regime for FD002/FD004.
# ---------------------------------------------------------------------------
N_REGIMES = 6                   # number of operating regime clusters

# Per-cycle aggregation: mean only (applied after cruise-phase filtering).
CYCLE_AGG_FUNCTION = "mean"

# Feature extraction
WINDOW_SIZE = 30        # sliding window in flight cycles
WINDOW_STRIDE = 1
FFT_NUM_FREQS = 5

# RUL clipping — set to None to compute from dev data at runtime.
# Original C-MAPSS used 125 (engines run 200-350 cycles). N-CMAPSS may differ
# substantially. When None, clip is set to the 95th percentile of dev RUL.
RUL_CLIP = None

# Manifold projection
EMBEDDING_DIM = 64
PROJECTION_METHOD = "none"
USE_CAUSAL_WEIGHTS = False

# Failure basin
BASIN_K = 5

# BASIN_RUL_THRESHOLD and WARNING_RUL — set to None to compute from dev data.
# When None, derived from the RUL clip value:
#   BASIN_RUL_THRESHOLD = 0.20 * rul_clip (bottom 20% of life)
#   WARNING_RUL = 0.40 * rul_clip (bottom 40% of life)
# These ratios match the original C-MAPSS proportions (25/125 and 50/125).
BASIN_RUL_THRESHOLD = None
WARNING_RUL = None

THRESHOLD_SIGMA = 5.0

# N-CMAPSS sub-datasets and their failure mode counts
SUBDATASETS = {
    "DS01": {"n_units": 10, "n_failure_modes": 1},
    "DS03": {"n_units": 15, "n_failure_modes": 2},
    "DS04": {"n_units": 10, "n_failure_modes": 3},
    "DS05": {"n_units": 10, "n_failure_modes": 4},
    "DS06": {"n_units": 10, "n_failure_modes": 5},
    "DS07": {"n_units": 10, "n_failure_modes": 6},
    "DS08a": {"n_units": 15, "n_failure_modes": 7},
    "DS08c": {"n_units": 10, "n_failure_modes": 7},
    "DS08d": {"n_units": 10, "n_failure_modes": 7},
}

N_LDA_CLASSES = 6


def compute_rul_thresholds(dev_ruls: np.ndarray) -> dict:
    """Compute RUL_CLIP, BASIN_RUL_THRESHOLD, and WARNING_RUL from dev data.

    Called at runtime when config values are None. Uses the 95th percentile
    of raw dev RUL as the clip value, then derives basin and warning thresholds
    proportionally (matching the 25/125 and 50/125 ratios from original C-MAPSS).

    Returns dict with 'rul_clip', 'basin_rul_threshold', 'warning_rul'.
    """
    rul_clip = int(np.percentile(dev_ruls, 95))
    basin_rul_threshold = int(round(0.20 * rul_clip))
    warning_rul = int(round(0.40 * rul_clip))
    return {
        "rul_clip": rul_clip,
        "basin_rul_threshold": basin_rul_threshold,
        "warning_rul": warning_rul,
    }
