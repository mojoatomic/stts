"""
Single source of truth for all orbital STTS pipeline hyperparameters.

Every value has a physical justification. All experimental scripts
(corpus.py, train.py, validate.py, case_study.py) import from here.
The W weight vector is defined in canonical_weights() in
horizons_stts_pipeline.py — that is the operational definition.
This file documents the spec and verifies consistency at import time.
"""

import numpy as np


# ── Corpus Selection ─────────────────────────────────────────

CORPUS_DIST_MAX_AU = 0.02       # ~8 lunar distances; operationally significant threshold
CORPUS_DATE_MIN    = "2000-01-01"  # Start of reliable CNEOS database coverage
CORPUS_DATE_MAX    = "2024-01-01"  # Cutoff: excludes objects discovered during analysis
CORPUS_V_INF_MAX   = 15.0       # km/s; exclude hyperbolic encounters
CORPUS_RANDOM_SEED = 42         # Arbitrary; fixed for reproducibility
CORPUS_N_TRAIN     = 200        # Training corpus size; chosen for P1 sufficiency
CORPUS_FETCH_LIMIT = 1000       # Max objects to fetch from Horizons


# ── Feature Extraction ───────────────────────────────────────

WINDOW_DAYS  = 30               # Minimum arc for one complete feature window
STRIDE_DAYS  = 7                # Advance one week per evaluation step
WARNING_DAYS = 90               # RUL ≤ 90 days = precursor zone
N_FEATURES   = 30               # Dimensionality of feature vector


# ── Model ────────────────────────────────────────────────────

LDA_COMPONENTS = 1              # 1D projection; higher dims do not improve separation
K_NEIGHBORS    = 5              # k-NN query; insensitive to ±2 in empirical sweep
LOOKBACK_DAYS  = 365            # Training trajectory length before close approach


# ── Apophis Case Study ───────────────────────────────────────

APOPHIS_DESIGNATION    = "99942"
APOPHIS_DISCOVERY_DATE = "2004-Jun-19"
APOPHIS_FLYBY_DATE     = "2029-Apr-13"
APOPHIS_FLYBY_JD       = 2462136.5    # April 13, 2029 TDB
APOPHIS_FLYBY_DIST_AU  = 0.000253     # ~38,000 km from Earth center


# ── Paths ────────────────────────────────────────────────────

ARTIFACTS_DIR            = "artifacts"
RESULTS_DIR              = "results/orbital"
CORPUS_FILE              = f"{ARTIFACTS_DIR}/corpus.pkl"
TRAIN_DESIGNATIONS_FILE  = f"{ARTIFACTS_DIR}/corpus_train.json"
TEST_DESIGNATIONS_FILE   = f"{ARTIFACTS_DIR}/corpus_test.json"
SCALER_FILE              = f"{ARTIFACTS_DIR}/scaler.pkl"
LDA_FILE                 = f"{ARTIFACTS_DIR}/lda.pkl"
BASIN_FILE               = f"{ARTIFACTS_DIR}/basin.npy"
MODEL_META_FILE          = f"{ARTIFACTS_DIR}/model_meta.json"


# ── Weight Specification (documentation + config snapshot) ───
# The operational W vector is built by canonical_weights() in
# horizons_stts_pipeline.py. This spec documents the physical
# justification for each weight and is used in config_snapshot().

WEIGHT_SPEC = [
    ((0,  4),  0.5, "q summaries (mean/std/min/max): useful but dominated by rate features"),
    ((4,  8),  0.3, "a, e summaries: slow-varying, low signal-to-noise"),
    ((8,  10), 3.0, "dq/dt (mean, std): primary perihelion approach signal"),
    ((12, 16), 2.0, "da/dt, de/dt: secondary perturbation indicators"),
    ((16, 18), 3.0, "|q - 1 AU|: direct distance from Earth's orbit"),
    ((18, 20), 3.0, "d|q-1AU|/dt: rate of perihelion approach to Earth orbit"),
    ((20, 22), 2.0, "d²q/dt²: acceleration of perihelion evolution"),
    ((22, 24), 2.0, "e-q, a-q correlation: coupled orbital evolution under perturbation"),
    ((24, 25), 3.0, "q linear trend: directional approach signal over window"),
    ((25, 26), 4.0, "late/early |q-1AU| ratio: most discriminating precursor feature"),
    ((26, 27), 0.2, "mean inclination: orbital geometry context, not approach signal"),
    ((27, 28), 1.0, "mean motion: neutral weight, included for completeness"),
    ((28, 30), 2.0, "time to perihelion: approach phase context"),
]


def build_weight_vector() -> np.ndarray:
    """Build W vector from WEIGHT_SPEC. Must match canonical_weights()."""
    W = np.ones(N_FEATURES)
    for (start, end), mult, _justification in WEIGHT_SPEC:
        W[start:end] *= mult
    return W


def config_snapshot() -> dict:
    """Full config for embedding in results JSON. Enables reproducibility audit."""
    return {
        "corpus": {
            "dist_max_au": CORPUS_DIST_MAX_AU,
            "date_min": CORPUS_DATE_MIN,
            "date_max": CORPUS_DATE_MAX,
            "v_inf_max": CORPUS_V_INF_MAX,
            "random_seed": CORPUS_RANDOM_SEED,
            "n_train": CORPUS_N_TRAIN,
            "fetch_limit": CORPUS_FETCH_LIMIT,
        },
        "features": {
            "window_days": WINDOW_DAYS,
            "stride_days": STRIDE_DAYS,
            "warning_days": WARNING_DAYS,
            "n_features": N_FEATURES,
        },
        "model": {
            "lda_components": LDA_COMPONENTS,
            "k_neighbors": K_NEIGHBORS,
            "lookback_days": LOOKBACK_DAYS,
        },
        "weights": {
            f"W[{s}:{e}]": {"multiplier": m, "justification": j}
            for (s, e), m, j in WEIGHT_SPEC
        },
        "apophis": {
            "designation": APOPHIS_DESIGNATION,
            "flyby_jd": APOPHIS_FLYBY_JD,
            "flyby_dist_au": APOPHIS_FLYBY_DIST_AU,
        },
    }


# ── Import-time consistency checks ──────────────────────────

from horizons_stts_pipeline import (
    canonical_weights,
    WINDOW_DAYS  as _LIB_WD,
    STRIDE_DAYS  as _LIB_SD,
    WARNING_DAYS as _LIB_WAD,
)

assert WINDOW_DAYS == _LIB_WD, \
    f"WINDOW_DAYS mismatch: config={WINDOW_DAYS}, lib={_LIB_WD}"
assert STRIDE_DAYS == _LIB_SD, \
    f"STRIDE_DAYS mismatch: config={STRIDE_DAYS}, lib={_LIB_SD}"
assert WARNING_DAYS == _LIB_WAD, \
    f"WARNING_DAYS mismatch: config={WARNING_DAYS}, lib={_LIB_WAD}"

_lib_W = canonical_weights()
_cfg_W = build_weight_vector()
assert np.array_equal(_lib_W, _cfg_W), \
    "Weight vector mismatch between config WEIGHT_SPEC and canonical_weights()!"
