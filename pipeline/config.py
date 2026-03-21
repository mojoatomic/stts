"""Central configuration for the STTS C-MAPSS validation pipeline."""

from __future__ import annotations

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "CMAPSSData"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset selection — start with the simplest subset
DATASET = "FD001"

# Column names for C-MAPSS (26 columns, no header in raw files)
COLUMN_NAMES = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

# Sensors to drop — near-constant in FD001, carry no degradation signal
# sensor_1 (fan inlet temp), sensor_5 (fan inlet pressure),
# sensor_6 (bypass duct pressure), sensor_10 (engine pressure ratio),
# sensor_16 (burner fuel-air ratio), sensor_18 (demanded fan speed),
# sensor_19 (demanded corrected fan speed)
DROP_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                "sensor_16", "sensor_18", "sensor_19"]

# Active sensors after dropping
ACTIVE_SENSORS = [s for s in SENSOR_COLS if s not in DROP_SENSORS]

# Feature extraction
WINDOW_SIZE = 30        # sliding window in cycles
WINDOW_STRIDE = 1       # stride
FFT_NUM_FREQS = 5       # number of FFT magnitude bins to keep per sensor
RUL_CLIP = 125          # cap RUL at this value (standard in literature)

# Causal weighting — per paper Section 4.3 / 5.1
# Turbofan causal chain: mechanical degradation → vibration/speed change
# → efficiency loss → temperature rise → failure
# Upstream (speed/mechanical) sensors get higher weight
SENSOR_CAUSAL_GROUPS = {
    "high": [  # causally upstream — speed/mechanical sensors
        "sensor_8",   # physical fan speed
        "sensor_9",   # physical core speed
        "sensor_13",  # corrected fan speed
        "sensor_14",  # corrected core speed
    ],
    "medium_high": [  # efficiency/pressure — mid-chain
        "sensor_7",   # HPC outlet pressure
        "sensor_11",  # HPC outlet static pressure
        "sensor_12",  # fuel flow ratio to Ps30
        "sensor_15",  # bypass ratio
    ],
    "medium": [  # temperature sensors — downstream
        "sensor_2",   # LPC outlet temperature
        "sensor_3",   # HPC outlet temperature
        "sensor_4",   # LPT outlet temperature
        "sensor_17",  # bleed enthalpy
    ],
    "low": [  # coolant bleed — least informative for early detection
        "sensor_20",  # HPT coolant bleed
        "sensor_21",  # LPT coolant bleed
    ],
}
CAUSAL_WEIGHT_VALUES = {
    "high": 2.0,
    "medium_high": 1.5,
    "medium": 1.0,
    "low": 0.7,
}

# Feature class multipliers — rate/freq features capture approach dynamics
FEATURE_CLASS_MULTIPLIERS = {
    "time_domain": 1.0,
    "rate": 1.5,
    "frequency": 1.3,
    "covariance": 1.2,
}

# Manifold projection
EMBEDDING_DIM = 64
PROJECTION_METHOD = "pca"  # "pca" or "umap"

# Failure basin
BASIN_RUL_THRESHOLD = 30  # trajectories with RUL <= this form B_f
BASIN_K = 5               # k for k-NN distance to basin

