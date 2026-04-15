"""
Single source of truth for all STTS-Protein pipeline hyperparameters.

Every value has a physical justification. The same F->W->M pipeline
(feature extraction, causal weighting, LDA projection) used across
all STTS domains. This config defines the protein-specific parameters.

Domain: protein conformational dynamics / thermal unfolding.
Data source: mdCATH dataset (Mirarchi et al., Nature Scientific Data 2024).
    5,398 protein domains, 5 replicates each, 5 temperatures (320-450K).
    HDF5 format, per-domain files on HuggingFace.

Physical basis: at elevated temperature (450K), proteins unfold —
losing secondary structure, expanding radius of gyration, losing
native contacts. These signatures are geometrically distinct from
the folded basin at 320K. STTS detects the approach to the unfolded
state as centroid drift in structural descriptor space.
"""

import numpy as np
from pathlib import Path


# -- Paths ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "protein"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "protein"
RESULTS_DIR = PROJECT_ROOT / "results" / "protein"
DESCRIPTORS_DIR = DATA_DIR / "descriptors"


# -- Dataset -------------------------------------------------------

# mdCATH HuggingFace repository for individual domain HDF5 files.
HUGGINGFACE_REPO = "compsciencelab/mdCATH"

# Temperatures available in mdCATH (Kelvin).
TEMPERATURES = [320, 348, 379, 413, 450]

# Healthy baseline temperature. Proteins remain folded at 320K
# throughout the 500 ns simulation for the domains we select.
HEALTHY_TEMP = 320

# Unfolding temperature. 450K is the highest temperature in mdCATH.
# Selected domains show clear unfolding (Q < 0.3) at this temperature.
UNFOLDING_TEMP = 450

# Number of replicates per temperature in mdCATH.
N_REPLICATES = 5

# Frame resolution: 1 ns per frame in mdCATH trajectories.
# 500 frames = 500 ns total simulation time per replicate.
FRAME_RESOLUTION_NS = 1

# Target domains. Start with 5sicI00 (subtilisin inhibitor, 106 residues,
# 2-layer alpha-beta sandwich) which unfolds cleanly at 450K per the paper.
# Additional domains added after pipeline validation on this one.
TARGET_DOMAINS = [
    "5sicI00",
]

RANDOM_SEED = 42


# -- Domain Selection Criteria -------------------------------------

# These criteria determine which mdCATH domains qualify for the experiment.
# Applied during download/screening, not during training.

# RMSD ratio: at 450K, RMSD must exceed 2x the 320K mean in >= 4/5 reps.
# This ensures clear unfolding signal, not just thermal fluctuation.
RMSD_RATIO_THRESHOLD = 2.0
RMSD_RATIO_MIN_REPS = 4

# Stability criterion: at 320K, RMSD must stay < 3A from initial structure
# in >= 4/5 replicates. Ensures a clean healthy baseline.
STABLE_RMSD_MAX_ANGSTROM = 3.0
STABLE_MIN_REPS = 4

# Domain size range (residues). Smaller domains have fewer contacts
# and less interesting unfolding pathways. Larger domains are expensive.
MIN_RESIDUES = 50
MAX_RESIDUES = 200


# -- Structural Descriptor Channels --------------------------------

# Eight "sensor" channels extracted per frame from MD coordinates.
# These are the raw observables that carry the unfolding signal,
# analogous to orbital elements (NEA) or TLE channels (reentry).
DESCRIPTOR_CHANNELS = [
    "rmsd",              # nm -- backbone RMSD vs initial structure (pre-computed in mdCATH)
    "rg",                # nm -- radius of gyration (pre-computed as gyrationRadius)
    "sasa",              # nm^2 -- solvent-accessible surface area (computed via mdtraj)
    "native_contacts",   # dimensionless -- fraction Q of native contacts preserved
    "helix_fraction",    # dimensionless -- fraction of residues in alpha-helix (from DSSP)
    "sheet_fraction",    # dimensionless -- fraction of residues in beta-sheet (from DSSP)
    "coil_fraction",     # dimensionless -- fraction of residues in coil (from DSSP)
    "end_to_end",        # nm -- distance between N and C terminal Ca atoms
]
N_CHANNELS = len(DESCRIPTOR_CHANNELS)

# Native contact definition (for Q computation):
# Contact = Ca-Ca distance < CONTACT_CUTOFF for residues |i-j| > CONTACT_SEQ_SEP.
# 8A (0.8 nm) cutoff and sequence separation > 3 are standard in protein folding literature.
# mdCATH coordinates are in nm, so cutoff is in nm.
CONTACT_CUTOFF_NM = 0.8   # 8 Angstrom = 0.8 nm
CONTACT_SEQ_SEPARATION = 3

# Reference structure for RMSD and native contacts:
# Frame 0 of 320K replicate 1 (the initial folded structure).
REFERENCE_TEMP = 320
REFERENCE_REPLICATE = 0  # 0-indexed
REFERENCE_FRAME = 0


# -- RUL Assignment ------------------------------------------------

# "Failure" = unfolding. Defined as Q (native contact fraction) dropping
# below Q_FAILURE_THRESHOLD and staying below for the remainder.
# Q < 0.3 means less than 30% of native contacts remain -- clearly unfolded.
Q_FAILURE_THRESHOLD = 0.3

# If Q never drops below threshold in 500 ns, the trajectory did NOT unfold.
# Exclude from run-to-failure set (but can serve as negative control).

# RUL = time remaining (in frames = ns) until Q first drops below threshold
# and stays below. Measured from each frame.


# -- Train/Test Split ----------------------------------------------

# Dev set: 320K all replicates (healthy) + 450K replicates 0-2 (degradation with RUL)
# Test set: 450K replicates 3-4 (held out)
# Split is by replicate index, not by random sampling -- deterministic.
DEV_REPLICATES_450K = [0, 1, 2]   # 3 of 5 replicates for training
TEST_REPLICATES_450K = [3, 4]     # 2 of 5 replicates held out
DEV_REPLICATES_320K = [0, 1, 2, 3, 4]  # all 5 replicates as healthy baseline


# -- Feature Extraction (Stage F) ---------------------------------

# Sliding window over descriptor time series (frames at 1 ns resolution).
# 20 frames = 20 ns of simulation time per window.
WINDOW_SIZE = 20  # frames (ns)

# Step size: advance by 5 frames per window position.
# Gives 4x overlap -- adequate temporal resolution without excessive redundancy.
WINDOW_STRIDE_TRAIN = 5   # training: every 5th position
WINDOW_STRIDE_EVAL = 1    # evaluation: every position (max resolution)

# Feature classes (same 4-class architecture as all STTS domains):
#   F_td:    time-domain summaries (mean, std, min, max per channel)
#   F_rate:  rate features (d/dt mean, d/dt std, d^2/dt^2 mean per channel)
#   F_ratio: late/early window ratio per channel
#   F_cross: cross-channel correlation (upper triangle)

N_TD_FEATURES = 4 * N_CHANNELS       # 4 stats x 8 channels = 32
N_RATE_FEATURES = 3 * N_CHANNELS     # 3 stats x 8 channels = 24
N_RATIO_FEATURES = N_CHANNELS        # 1 per channel = 8
N_CROSS_FEATURES = N_CHANNELS * (N_CHANNELS - 1) // 2  # upper triangle 8x8 = 28

N_FEATURES = N_TD_FEATURES + N_RATE_FEATURES + N_RATIO_FEATURES + N_CROSS_FEATURES
# = 32 + 24 + 8 + 28 = 92


# -- Causal Weighting (Stage W) -----------------------------------

# Physical causal chain for protein unfolding:
#   Temperature increase -> backbone flexibility -> loss of secondary structure ->
#   loss of native contacts -> expansion (Rg increase) -> full unfolding (high RMSD)
#
# Native contacts (Q) is the most direct unfolding indicator -- it measures
# structural integrity at the residue level. RMSD and Rg are consequences.
# Secondary structure fractions are upstream indicators that change early.

CHANNEL_CAUSAL_WEIGHTS = {
    "rmsd":              2.0,  # Direct measure of structural deviation from native
    "rg":                1.5,  # Expansion indicator, consequence of contact loss
    "sasa":              1.0,  # Surface exposure, correlates with Rg
    "native_contacts":   3.0,  # PRIMARY indicator -- direct measure of fold integrity
                               # Highest weight: Q is the operational failure definition
    "helix_fraction":    2.0,  # Secondary structure loss is an early unfolding signal
    "sheet_fraction":    2.0,  # Same as helix -- structural change precursor
    "coil_fraction":     1.0,  # Complement of helix+sheet, less informative alone
    "end_to_end":        1.5,  # Chain extension, correlates with Rg but captures
                               # different geometry (end-to-end vs radius)
}

FEATURE_CLASS_WEIGHTS = {
    "time_domain": 1.0,  # Baseline: mean/std/min/max of each channel
    "rate":        2.0,  # Rate features carry the approach dynamics signal
    "ratio":       2.5,  # Late/early ratio is the most discriminating precursor
    "cross":       1.2,  # Cross-channel correlations: structural coupling information
}


# -- Weight Vector Construction ------------------------------------

WEIGHT_SPEC = [
    # Time-domain features: 4 stats x 8 channels = indices [0:32]
    ((0,  4),   2.0, "RMSD td: structural deviation from native fold"),
    ((4,  8),   1.5, "Rg td: compactness / expansion"),
    ((8,  12),  1.0, "SASA td: surface exposure"),
    ((12, 16),  3.0, "Q td: native contact fraction -- PRIMARY SIGNAL"),
    ((16, 20),  2.0, "helix_frac td: alpha-helix content"),
    ((20, 24),  2.0, "sheet_frac td: beta-sheet content"),
    ((24, 28),  1.0, "coil_frac td: coil content"),
    ((28, 32),  1.5, "end-to-end td: chain extension"),

    # Rate features: 3 stats x 8 channels = indices [32:56]
    ((32, 35),  4.0, "d(RMSD)/dt: rate of structural change"),
    ((35, 38),  3.0, "d(Rg)/dt: expansion rate"),
    ((38, 41),  2.0, "d(SASA)/dt: surface exposure rate"),
    ((41, 44),  6.0, "d(Q)/dt: rate of contact loss -- STRONGEST RATE SIGNAL"),
    ((44, 47),  4.0, "d(helix)/dt: helix loss rate"),
    ((47, 50),  4.0, "d(sheet)/dt: sheet loss rate"),
    ((50, 53),  2.0, "d(coil)/dt: coil gain rate"),
    ((53, 56),  3.0, "d(end-to-end)/dt: extension rate"),

    # Late/early ratio: 1 per channel = indices [56:64]
    ((56, 57),  5.0, "RMSD late/early: deviation growth within window"),
    ((57, 58),  3.75, "Rg late/early: expansion within window"),
    ((58, 59),  2.5, "SASA late/early: exposure change within window"),
    ((59, 60),  7.5, "Q late/early: MOST DISCRIMINATING -- contact loss within window"),
    ((60, 61),  5.0, "helix late/early: helix loss within window"),
    ((61, 62),  5.0, "sheet late/early: sheet loss within window"),
    ((62, 63),  2.5, "coil late/early: coil gain within window"),
    ((63, 64),  3.75, "end-to-end late/early: extension within window"),

    # Cross-channel correlation: 28 features = indices [64:92]
    ((64, 92),  1.2, "Cross-channel correlations: coupled unfolding dynamics"),
]


def build_weight_vector() -> np.ndarray:
    """Build W vector from WEIGHT_SPEC."""
    W = np.ones(N_FEATURES)
    for (start, end), mult, _justification in WEIGHT_SPEC:
        W[start:end] = mult
    return W


# -- Manifold Projection (Stage M) --------------------------------

LDA_COMPONENTS = 1   # 1D projection (binary: healthy vs unfolding)
K_NEIGHBORS = 5      # k-NN query depth for distance-to-basin

# Basin definition: only precursor windows projecting above threshold
# enter the failure basin. Removes early 450K windows that still look folded.
BASIN_SIGMA_CUTOFF = 1.0

# RUL clip: cap RUL at 500 ns (full simulation length).
RUL_CLIP = 500


# -- Labeling ------------------------------------------------------

# Binary labeling for LDA:
#   1 = precursor (450K windows with RUL <= PRECURSOR_FRAMES)
#   0 = nominal (320K windows -- always folded)
#  -1 = ambiguous (450K windows far from failure -- exclude from training)
PRECURSOR_FRAMES = 100   # windows within 100 ns of Q < 0.3 are "precursor"
NOMINAL_BUFFER_FRAMES = 200  # 450K windows > 200 ns before failure are ambiguous


# -- Validation Conditions -----------------------------------------

V1_P_VALUE_THRESHOLD = 0.001   # Mann-Whitney U test significance
V2_SPEARMAN_THRESHOLD = 0.3   # Spearman rho for monotonic approach
                               # Lower than reentry (0.5) because protein
                               # unfolding may not be perfectly monotonic --
                               # partial refolding events are physically possible


# -- Mechanism Experiment ------------------------------------------

MECHANISM_RUL_CLIP = 500  # same as RUL_CLIP
MECHANISM_META_WINDOW_K = 5  # meta-window for M7


def config_snapshot() -> dict:
    """Full config for embedding in results JSON. Enables reproducibility audit."""
    return {
        "dataset": {
            "source": "mdCATH (Mirarchi et al., 2024)",
            "huggingface_repo": HUGGINGFACE_REPO,
            "target_domains": TARGET_DOMAINS,
            "healthy_temp": HEALTHY_TEMP,
            "unfolding_temp": UNFOLDING_TEMP,
            "n_replicates": N_REPLICATES,
            "frame_resolution_ns": FRAME_RESOLUTION_NS,
            "random_seed": RANDOM_SEED,
        },
        "selection": {
            "rmsd_ratio_threshold": RMSD_RATIO_THRESHOLD,
            "rmsd_ratio_min_reps": RMSD_RATIO_MIN_REPS,
            "stable_rmsd_max_angstrom": STABLE_RMSD_MAX_ANGSTROM,
            "stable_min_reps": STABLE_MIN_REPS,
            "min_residues": MIN_RESIDUES,
            "max_residues": MAX_RESIDUES,
        },
        "descriptors": {
            "channels": DESCRIPTOR_CHANNELS,
            "n_channels": N_CHANNELS,
            "contact_cutoff_nm": CONTACT_CUTOFF_NM,
            "contact_seq_separation": CONTACT_SEQ_SEPARATION,
            "reference_temp": REFERENCE_TEMP,
            "reference_replicate": REFERENCE_REPLICATE,
            "reference_frame": REFERENCE_FRAME,
        },
        "rul": {
            "q_failure_threshold": Q_FAILURE_THRESHOLD,
        },
        "split": {
            "dev_replicates_450K": DEV_REPLICATES_450K,
            "test_replicates_450K": TEST_REPLICATES_450K,
            "dev_replicates_320K": DEV_REPLICATES_320K,
        },
        "features": {
            "window_size": WINDOW_SIZE,
            "stride_train": WINDOW_STRIDE_TRAIN,
            "stride_eval": WINDOW_STRIDE_EVAL,
            "n_features": N_FEATURES,
        },
        "causal_weights": {
            "channel_weights": CHANNEL_CAUSAL_WEIGHTS,
            "feature_class_weights": FEATURE_CLASS_WEIGHTS,
        },
        "labeling": {
            "precursor_frames": PRECURSOR_FRAMES,
            "nominal_buffer_frames": NOMINAL_BUFFER_FRAMES,
        },
        "model": {
            "lda_components": LDA_COMPONENTS,
            "k_neighbors": K_NEIGHBORS,
            "basin_sigma_cutoff": BASIN_SIGMA_CUTOFF,
            "rul_clip": RUL_CLIP,
        },
        "validation": {
            "v1_p_threshold": V1_P_VALUE_THRESHOLD,
            "v2_spearman_threshold": V2_SPEARMAN_THRESHOLD,
        },
        "weights": {
            f"W[{s}:{e}]": {"multiplier": m, "justification": j}
            for (s, e), m, j in WEIGHT_SPEC
        },
    }
