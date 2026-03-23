"""
Single source of truth for all STTS-Reentry pipeline hyperparameters.

Every value has a physical justification. The same F→W→M pipeline
(feature extraction, causal weighting, LDA projection) used across
all STTS domains. This config defines the reentry-specific parameters.

Domain: uncontrolled reentry prediction from TLE orbital decay trajectories.
Data source: Space-Track gp_history (TLE records) + decay class (confirmed reentries).
Corpus: 1,490 confirmed Starlink reentries, May 2019 – March 2026.

Physical basis: atmospheric drag causes orbital energy loss visible as
monotonic periapsis decline (550→147 km), mean motion increase (15.06→16.45
rev/day), and explosive mean-motion-dot growth (170x increase in terminal
phase). These signatures are geometrically distinct from stable orbits.
"""

import numpy as np
from pathlib import Path


# ── Paths ───────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "reentry"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "reentry"
RESULTS_DIR = PROJECT_ROOT / "results" / "reentry"


# ── Space-Track API ─────────────────────────────────────────

SPACETRACK_BASE_URL = "https://www.space-track.org"
SPACETRACK_LOGIN_URL = f"{SPACETRACK_BASE_URL}/ajaxauth/login"
SPACETRACK_QUERY_URL = f"{SPACETRACK_BASE_URL}/basicspacedata/query"

# Rate limit: 300 requests/hour per Space-Track policy.
# gp_history queries for Starlink return large responses (thousands of TLE
# records), which can trigger 429s even within the nominal rate limit.
# 3.0s base delay + exponential backoff on 429 (60s, 120s, 240s).
REQUEST_DELAY_S = 3.0
REQUEST_BATCH_SIZE = 50  # objects per batch query (comma-separated NORAD IDs)
BACKOFF_BASE_S = 60      # initial wait on 429 Too Many Requests
BACKOFF_MAX_RETRIES = 4  # max retries per request before giving up


# ── Corpus Selection ────────────────────────────────────────

# Only confirmed reentries (MSG_TYPE = "Historical" in Space-Track decay class).
# "Prediction" records are forward-looking estimates, not ground truth.
DECAY_MSG_TYPE = "Historical"

# Object name filter for Space-Track query.
# "STARLINK~~" uses Space-Track wildcard syntax (contains match).
OBJECT_NAME_FILTER = "STARLINK~~"

# Date range for corpus. First Starlink launch was 2019-05-24.
# Exclude objects still decaying (reentry after analysis cutoff).
CORPUS_DATE_MIN = "2019-05-01"
CORPUS_DATE_MAX = "2026-03-01"

# Minimum TLE records required per object. Objects with fewer records
# have insufficient trajectory information for windowed feature extraction.
# At ~2 TLEs/day for decaying objects, 60 records ≈ 30 days of coverage.
MIN_TLE_RECORDS = 60

# Storm objects (Feb 2022 geomagnetic event) only existed for ~14 days.
# They need a lower threshold. The 6 confirmed storm objects with TLE
# records have 3–25 TLEs spanning 1–4 days. Window size is adapted to
# min(available_records, WINDOW_SIZE) for these objects.
STORM_MIN_TLE_RECORDS = 3

RANDOM_SEED = 42


# ── Geomagnetic Storm Event ────────────────────────────────
# February 3, 2022: geomagnetic storm caused loss of up to 40 Starlink
# satellites from the Starlink Group 4-7 launch (2022-02-03).
# These objects decayed within ~2 weeks of launch due to elevated
# atmospheric density, NOT deliberate deorbit maneuvers.
#
# Scientific significance: TERRA_INCOGNITA candidates. Their decay
# trajectories are geometrically unlike deliberate deorbit (rapid onset,
# no orbit-raising phase, abnormal BSTAR evolution). The framework should
# flag these as OOD — a paper result in itself.
#
# Treatment: flagged separately in corpus. Never mixed with deliberate
# deorbit without explicit investigation. Used as OOD test case.

GEOMAGNETIC_STORM_DATE = "2022-02-03"
GEOMAGNETIC_STORM_WINDOW_DAYS = 30  # objects launched within this window
GEOMAGNETIC_STORM_LAUNCH_INTLDES = "2022-010"  # international designator for Group 4-7


# ── Train/Test Split ────────────────────────────────────────

# Split by NORAD_CAT_ID (satellite identity), not by TLE record.
# All TLE records for a given satellite go to the same split.
# This prevents temporal leakage from the same object appearing in both.
# Same designation-level split strategy as the orbital (NEA) paper.
TRAIN_FRACTION = 0.7
TEST_FRACTION = 0.3

# Cap nominal satellites used in training/testing. With ~15K operational
# satellites vs 272 reentry, uncapped nominal windows would overwhelm
# the feature space. We sample this many operational satellites per split.
# The remainder are available for OOD analysis but not used in training.
NOMINAL_SAMPLE_SIZE = 500  # ~2x the reentry count, balanced enough for LDA

# Geomagnetic storm objects are excluded from train/test and held out
# as a separate TERRA_INCOGNITA evaluation set.


# ── Feature Extraction (Stage F) ───────────────────────────

# Sliding window over successive TLE records (not fixed time intervals).
# TLE update cadence varies: ~2/day during active decay, ~1/week at
# operational altitude. Window is defined by record count, not days.
WINDOW_SIZE = 30  # TLE records per window

# Stride: advance by this many TLE records per evaluation step.
# Stride=1 gives maximum temporal resolution for monitoring queries.
# Stride>1 for training efficiency (every window is highly correlated
# with its neighbor given 1-record shift).
WINDOW_STRIDE_TRAIN = 5   # training: every 5th window (reduce redundancy)
WINDOW_STRIDE_EVAL = 1    # evaluation: every window (maximum resolution)

# State vector channels extracted from each TLE record.
# These are the raw orbital elements that carry the decay signal.
STATE_CHANNELS = [
    "PERIAPSIS",       # km — primary decay indicator (monotonic decline)
    "MEAN_MOTION",     # rev/day — increases as orbit shrinks (Kepler's 3rd)
    "MEAN_MOTION_DOT", # rev/day² — rate of orbital decay (explosive near reentry)
    "BSTAR",           # 1/Earth-radii — drag coefficient (reflects atmospheric density)
    "ECCENTRICITY",    # dimensionless — orbit shape evolution during decay
    "APOAPSIS",        # km — highest orbit point (tracks orbit circularization)
]
N_CHANNELS = len(STATE_CHANNELS)

# Feature classes extracted per window (same F(T) architecture as all STTS domains):
#   F_td:   time-domain summaries (mean, std, min, max per channel)
#   F_rate: rate features (d/dt mean, d/dt std, d²/dt² mean per channel)
#   F_ratio: late/early window ratio per channel (approach discriminator)
#   F_cross: cross-channel correlation structure

# Time-domain: 4 stats × 6 channels = 24 features
N_TD_FEATURES = 4 * N_CHANNELS

# Rate: 3 stats × 6 channels = 18 features
N_RATE_FEATURES = 3 * N_CHANNELS

# Late/early ratio: 1 per channel = 6 features
# Ratio = mean(last half) / mean(first half). Values >> 1 or << 1
# indicate monotonic change within the window — the primary precursor signal.
N_RATIO_FEATURES = N_CHANNELS

# Cross-channel correlation: upper triangle of 6×6 = 15 features
# Captures coupled decay dynamics (e.g., PERIAPSIS-MM correlation tightens
# approaching reentry as Kepler's law dominates over maneuver effects).
N_CROSS_FEATURES = N_CHANNELS * (N_CHANNELS - 1) // 2

N_FEATURES = N_TD_FEATURES + N_RATE_FEATURES + N_RATIO_FEATURES + N_CROSS_FEATURES
# = 24 + 18 + 6 + 15 = 63


# ── Causal Weighting (Stage W) ─────────────────────────────

# Physical causal chain for orbital decay:
#   Atmospheric drag → BSTAR increase → energy loss →
#   MEAN_MOTION increase → PERIAPSIS decrease → reentry
#
# Upstream indicators (BSTAR, MEAN_MOTION_DOT) change first but are noisy.
# Downstream indicators (PERIAPSIS, MEAN_MOTION) are cleaner but lag.
# The weighting balances early detection vs. signal quality.

CHANNEL_CAUSAL_WEIGHTS = {
    # Primary signals: direct decay indicators
    "PERIAPSIS":        2.0,  # Cleanest signal, monotonic, physically unambiguous
    "MEAN_MOTION_DOT":  3.0,  # Strongest precursor — 170x growth before reentry
                              # Highest weight because it leads the other signals

    # Secondary signals: orbital mechanics consequences
    "MEAN_MOTION":      1.5,  # Kepler's 3rd law consequence of energy loss
    "APOAPSIS":         1.0,  # Tracks orbit circularization, less informative alone

    # Upstream/noisy signals: atmospheric interaction
    "BSTAR":            1.5,  # Drag coefficient — leading indicator but noisy
                              # Complex: deliberate drag-raise maneuvers spike BSTAR
                              # before sustained natural decay
    "ECCENTRICITY":     0.8,  # Orbit shape — informative for circularization phase
                              # but less discriminating than altitude/rate features
}

# Feature class multipliers (same structure as C-MAPSS/orbital configs)
FEATURE_CLASS_WEIGHTS = {
    "time_domain": 1.0,  # Baseline: mean/std/min/max of each channel
    "rate":        2.0,  # Rate features carry the approach dynamics signal
    "ratio":       2.5,  # Late/early ratio is the most discriminating precursor
                         # (validated in orbital paper: late/early |q-1AU| ratio
                         # had weight 4.0 — the strongest single feature)
    "cross":       1.2,  # Cross-channel correlations: structural information
}


# ── Weight Vector Construction ──────────────────────────────

WEIGHT_SPEC = [
    # Time-domain features: 4 stats × 6 channels = indices [0:24]
    # Applied per-channel with CHANNEL_CAUSAL_WEIGHTS × FEATURE_CLASS_WEIGHTS["time_domain"]
    ((0,  4),   2.0, "PERIAPSIS td (mean/std/min/max): clean altitude signal"),
    ((4,  8),   1.5, "MEAN_MOTION td: Kepler consequence of energy loss"),
    ((8,  12),  3.0, "MEAN_MOTION_DOT td: primary rate-of-decay indicator"),
    ((12, 16),  1.5, "BSTAR td: drag coefficient, leading but noisy"),
    ((16, 20),  0.8, "ECCENTRICITY td: orbit shape, moderate signal"),
    ((20, 24),  1.0, "APOAPSIS td: orbit circularization tracking"),

    # Rate features: 3 stats × 6 channels = indices [24:42]
    # Rate multiplier (2.0) × channel weight
    ((24, 27),  4.0, "d(PERIAPSIS)/dt: rate of altitude loss — critical"),
    ((27, 30),  3.0, "d(MEAN_MOTION)/dt: acceleration of orbital period change"),
    ((30, 33),  6.0, "d(MEAN_MOTION_DOT)/dt: jerk of decay — earliest signal"),
    ((33, 36),  3.0, "d(BSTAR)/dt: drag evolution rate"),
    ((36, 39),  1.6, "d(ECCENTRICITY)/dt: circularization rate"),
    ((39, 42),  2.0, "d(APOAPSIS)/dt: apoapsis lowering rate"),

    # Late/early ratio features: 1 per channel = indices [42:48]
    # Ratio multiplier (2.5) × channel weight
    ((42, 43),  5.0, "PERIAPSIS late/early: altitude decline within window"),
    ((43, 44),  3.75, "MEAN_MOTION late/early: period change within window"),
    ((44, 45),  7.5, "MEAN_MOTION_DOT late/early: MOST DISCRIMINATING — "
                     "explosive growth signature within single window"),
    ((45, 46),  3.75, "BSTAR late/early: drag evolution within window"),
    ((46, 47),  2.0, "ECCENTRICITY late/early: shape change within window"),
    ((47, 48),  2.5, "APOAPSIS late/early: apoapsis change within window"),

    # Cross-channel correlation: 15 features = indices [48:63]
    ((48, 63),  1.2, "Cross-channel correlations: coupled decay dynamics"),
]


def build_weight_vector() -> np.ndarray:
    """Build W vector from WEIGHT_SPEC."""
    W = np.ones(N_FEATURES)
    for (start, end), mult, _justification in WEIGHT_SPEC:
        W[start:end] = mult
    return W


# ── Labeling ────────────────────────────────────────────────

# Two-population design (v2):
#   Failure basin: windows from confirmed-reentry satellites (272 objects)
#   Nominal class: windows from confirmed-operational satellites (~15K objects)
#
# This is stronger than the v1 single-population design (precursor vs.
# early windows on the same satellite) because the nominal class represents
# genuinely stable orbits at ~550km, not pre-deorbit trajectories.
#
# For reentry satellites, we further label by proximity to reentry:

# Windows within PRECURSOR_DAYS of DECAY_EPOCH are "precursor" (label=1)
PRECURSOR_DAYS = 30

# Windows more than NOMINAL_BUFFER_DAYS before DECAY_EPOCH on reentry
# satellites are in the ambiguous zone for that satellite — they look
# operational because the satellite hasn't started deorbiting yet.
# These are excluded from training to avoid contamination.
NOMINAL_BUFFER_DAYS = 90

# Operational satellite windows are always label=0 (nominal).
# They don't have a DECAY_EPOCH, so no ambiguous zone exists.

# Cutoff: satellites with last TLE epoch at or after this date are
# classified as operational. The bulk TLE archive ends in Dec 2025.
OPERATIONAL_CUTOFF_DATE = "2025-12-01"


# ── Manifold Projection (Stage M) ──────────────────────────

LDA_COMPONENTS = 1  # 1D projection (binary classification: precursor vs. nominal)
K_NEIGHBORS = 5     # k-NN query depth for distance-to-basin computation

# Consecutive window fire requirement. A genuine approach signal is persistent
# across multiple TLE update cycles (~daily cadence near reentry). A single
# window below epsilon is noise; three consecutive windows represents 3+ days
# of persistent signal heading toward the failure basin.
CONSECUTIVE_FIRE_THRESHOLD = 3

# Basin definition: only precursor windows projecting above this threshold
# (in standard deviations below the precursor mean) enter the failure basin.
# Early-life reentry windows where the satellite still looks operational are
# not good exemplars of the failure signature. Removing them tightens the
# basin to the characteristic reentry geometry and eliminates overlap with
# the operational cluster near zero in LDA space.
BASIN_SIGMA_CUTOFF = 1.0  # basin = precursor projections > (mean - 1*std)

# OOD detection threshold: objects whose k-NN distance to the full
# training corpus exceeds this percentile are flagged as TERRA_INCOGNITA.
# The geomagnetic storm objects should trigger this.
OOD_PERCENTILE = 95


# ── Validation Conditions ───────────────────────────────────

# V1: Statistical separation between precursor and nominal embeddings.
# Mann-Whitney U test, p < 0.001 required. Same criterion as all STTS papers.
V1_P_VALUE_THRESHOLD = 0.001

# V2: Temporal consistency — STTS distance must decrease monotonically
# (on average) as the object approaches reentry. Measured by Spearman
# correlation between STTS distance and days-to-reentry.
#
# FINDING (first run, 257 satellites): V2 fails at ρ=0.394.
# This is physically expected for deliberate Starlink deorbits:
#   orbit raise → stable cruise (years) → deorbit burn → rapid decay (~weeks)
# The STTS distance signal has a sharp phase transition at the deorbit
# burn, not a gradual monotonic approach. The distance is LOW during the
# brief terminal phase and HIGH during the long stable phase, with an
# abrupt transition — producing moderate but not strong positive ρ.
# This is a domain-specific finding, not a framework failure.
# The threshold is kept at 0.5 for scientific honesty — we report the
# actual ρ and explain the physical mechanism.
V2_SPEARMAN_THRESHOLD = 0.5  # ρ > 0.5 (moderate positive correlation)

# V3: Feature class ablation — removing any single feature class should
# degrade V1 separation. Validates that all feature classes contribute.

# Detection lead time: how many days before Pc > 10⁻⁴ equivalent
# (or before DECAY_EPOCH) does the STTS query first fire?
# This is the operational metric — not a pass/fail criterion.


# ── Pre-Experiment Checklist ────────────────────────────────
#
# Before running the pipeline, verify:
#
# [ ] Corpus size: ≥100 unique objects in failure basin (we have ~1,490)
# [ ] Corpus size: ≥100 unique objects in nominal class (same objects, earlier windows)
# [ ] P1 sufficiency: failure basin has enough geometric diversity
# [ ] Train/test split: by NORAD_CAT_ID, no satellite in both splits
# [ ] Geomagnetic storm objects: identified and held out separately
# [ ] Temporal leakage: no future information in training features
# [ ] Feature NaN handling: TLE records with missing fields excluded
# [ ] BSTAR sign convention: Space-Track uses positive for drag,
#     negative values indicate maneuver corrections — handle both
# [ ] Window size vs. TLE cadence: verify 30 records ≈ 15-30 days
#     for decaying objects (cadence increases near reentry)
# [ ] Rate features: computed from TLE record differences, not
#     fixed time intervals — normalize by inter-epoch time delta
#
# Scientific rigor requirements:
# [ ] All results include 95% confidence intervals
# [ ] Train/test split is fixed by RANDOM_SEED for reproducibility
# [ ] No hyperparameter tuning on test set
# [ ] V1, V2, V3 conditions evaluated on held-out test split only
# [ ] TERRA_INCOGNITA evaluation on geomagnetic storm hold-out only


def config_snapshot() -> dict:
    """Full config for embedding in results JSON. Enables reproducibility audit."""
    return {
        "corpus": {
            "object_filter": OBJECT_NAME_FILTER,
            "decay_msg_type": DECAY_MSG_TYPE,
            "date_min": CORPUS_DATE_MIN,
            "date_max": CORPUS_DATE_MAX,
            "min_tle_records": MIN_TLE_RECORDS,
            "random_seed": RANDOM_SEED,
            "train_fraction": TRAIN_FRACTION,
        },
        "features": {
            "window_size": WINDOW_SIZE,
            "stride_train": WINDOW_STRIDE_TRAIN,
            "stride_eval": WINDOW_STRIDE_EVAL,
            "state_channels": STATE_CHANNELS,
            "n_features": N_FEATURES,
        },
        "causal_weights": {
            "channel_weights": CHANNEL_CAUSAL_WEIGHTS,
            "feature_class_weights": FEATURE_CLASS_WEIGHTS,
        },
        "labeling": {
            "precursor_days": PRECURSOR_DAYS,
            "nominal_buffer_days": NOMINAL_BUFFER_DAYS,
        },
        "model": {
            "lda_components": LDA_COMPONENTS,
            "k_neighbors": K_NEIGHBORS,
            "ood_percentile": OOD_PERCENTILE,
        },
        "validation": {
            "v1_p_threshold": V1_P_VALUE_THRESHOLD,
            "v2_spearman_threshold": V2_SPEARMAN_THRESHOLD,
        },
        "weights": {
            f"W[{s}:{e}]": {"multiplier": m, "justification": j}
            for (s, e), m, j in WEIGHT_SPEC
        },
        "geomagnetic_storm": {
            "date": GEOMAGNETIC_STORM_DATE,
            "window_days": GEOMAGNETIC_STORM_WINDOW_DAYS,
            "launch_intldes": GEOMAGNETIC_STORM_LAUNCH_INTLDES,
        },
    }
