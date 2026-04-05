"""Load and preprocess N-CMAPSS turbofan engine degradation data from HDF5.

N-CMAPSS stores within-flight time-series (sampled per second) in HDF5 files.
Each .h5 file contains one sub-dataset (e.g., DS01) with dev and test splits.

This loader:
  1. Reads ONLY whitelisted HDF5 keys (W, X_s, Y, A) — excludes X_v
     (virtual sensors) and T (health params) to prevent data leakage
  2. Filters to cruise phase using TRA stability + altitude threshold
  3. Aggregates cruise-phase sensor readings to per-cycle means (14 channels)
  4. Normalizes sensors per operating regime using flight conditions (W),
     removing flight-profile variation so features reflect degradation only
  5. Processes per-unit to keep peak memory bounded during aggregation
"""

from __future__ import annotations

import glob as globmod
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pipeline.ncmapss.config import (
    DATA_DIR, FLIGHT_COND_NAMES, SENSOR_NAMES, AUX_NAMES,
    ACTIVE_SENSORS, CYCLE_AGG_FUNCTION,
    H5_LOAD_KEYS_DEV, H5_LOAD_KEYS_TEST,
    CRUISE_TRA_WINDOW, CRUISE_TRA_STD_MAX, CRUISE_ALT_MIN,
    CRUISE_MIN_TIMESTEPS, N_REGIMES,
)


# ---------------------------------------------------------------------------
# HDF5 loading — whitelist only
# ---------------------------------------------------------------------------

def load_h5_arrays(filepath: Path, keys: list[str]) -> dict[str, np.ndarray]:
    """Read only the specified datasets from an N-CMAPSS HDF5 file.

    Loads by explicit whitelist to prevent accidental use of X_v (virtual
    sensors) or T (health parameters), which would be data leakage.
    """
    data = {}
    with h5py.File(filepath, "r") as f:
        for key in keys:
            if key in f:
                data[key] = f[key][:]
            else:
                raise KeyError(
                    f"Expected key '{key}' not found in {filepath.name}. "
                    f"Available keys: {list(f.keys())}"
                )
    return data


def find_h5_file(dataset_name: str) -> Path:
    """Find the HDF5 file for a given sub-dataset name.

    Handles naming variants: N-CMAPSS_DS03.h5, N-CMAPSS_DS03-012.h5,
    DS03.h5, etc.
    """
    # Exact matches first
    exact = [
        DATA_DIR / f"N-CMAPSS_{dataset_name}.h5",
        DATA_DIR / f"{dataset_name}.h5",
    ]
    for p in exact:
        if p.exists():
            return p

    # Glob for suffix variants (e.g., N-CMAPSS_DS03-012.h5)
    pattern = str(DATA_DIR / f"N-CMAPSS_{dataset_name}*.h5")
    matches = sorted(globmod.glob(pattern))
    if matches:
        return Path(matches[0])

    pattern = str(DATA_DIR / f"{dataset_name}*.h5")
    matches = sorted(globmod.glob(pattern))
    if matches:
        return Path(matches[0])

    raise FileNotFoundError(
        f"Cannot find HDF5 file for {dataset_name} in {DATA_DIR}. "
        f"Tried exact names and glob patterns."
    )


# ---------------------------------------------------------------------------
# Cruise-phase filtering
# ---------------------------------------------------------------------------

def _cruise_mask(
    alt: np.ndarray, tra: np.ndarray,
) -> np.ndarray:
    """Identify cruise-phase timesteps within a single flight cycle.

    Primary criterion: TRA (throttle-resolver angle) stability — cruise is
    when the engine is at steady thrust, so TRA has low rolling variance.
    Secondary criterion: altitude > 10,000 ft — guards against stable-TRA
    segments at low altitude (taxi, ground idle).

    Fallback cascade:
      1. TRA stable AND alt > threshold → cruise
      2. If too few timesteps: alt > threshold only
      3. If still too few: all timesteps (full flight)

    Args:
        alt: altitude array for one (unit, cycle), shape (n_timesteps,)
        tra: TRA array for one (unit, cycle), shape (n_timesteps,)

    Returns:
        boolean mask, shape (n_timesteps,)
    """
    n = len(alt)

    # TRA stability: rolling std over window
    if n >= CRUISE_TRA_WINDOW:
        # Cumulative sum trick for rolling std
        cumsum = np.cumsum(tra)
        cumsum2 = np.cumsum(tra ** 2)
        w = CRUISE_TRA_WINDOW
        # Rolling mean and variance
        roll_mean = (cumsum[w:] - cumsum[:-w]) / w
        roll_var = (cumsum2[w:] - cumsum2[:-w]) / w - roll_mean ** 2
        roll_var = np.maximum(roll_var, 0)  # numerical safety
        roll_std = np.sqrt(roll_var)

        # Pad: first (w) timesteps get the first valid value
        tra_stable = np.ones(n, dtype=bool)
        tra_stable[:w] = roll_std[0] < CRUISE_TRA_STD_MAX
        tra_stable[w:] = roll_std < CRUISE_TRA_STD_MAX
    else:
        # Too short for windowed analysis — check overall std
        tra_stable = np.full(n, np.std(tra) < CRUISE_TRA_STD_MAX)

    alt_high = alt > CRUISE_ALT_MIN

    # Primary: TRA stable AND altitude high
    cruise = tra_stable & alt_high
    if cruise.sum() >= CRUISE_MIN_TIMESTEPS:
        return cruise

    # Fallback 1: altitude only
    if alt_high.sum() >= CRUISE_MIN_TIMESTEPS:
        return alt_high

    # Fallback 2: full flight
    return np.ones(n, dtype=bool)


# ---------------------------------------------------------------------------
# Per-unit, per-cycle aggregation (memory-efficient)
# ---------------------------------------------------------------------------

def _aggregate_split(
    W: np.ndarray, X_s: np.ndarray, Y: np.ndarray, A: np.ndarray,
    rul_clip: int | None = None,
) -> pd.DataFrame:
    """Aggregate one split (dev or test) to per-cycle rows.

    Processes per-unit to bound peak memory — the raw timestep arrays
    are sliced, not copied into a giant DataFrame.

    Returns DataFrame with columns:
      unit_id, cycle, Fc, rul, {sensor_names}, {flight_cond_names}
    """
    # A columns: unit(0), cycle(1), Fc(2), h_s(3)
    units = A[:, 0].astype(int)
    cycles = A[:, 1].astype(int)
    unique_units = np.unique(units)

    rows = []
    for uid in unique_units:
        unit_mask = units == uid
        unit_cycles = cycles[unit_mask]
        unit_W = W[unit_mask]
        unit_X = X_s[unit_mask]
        unit_Y = Y[unit_mask]
        unit_A = A[unit_mask]

        unique_cycles = np.unique(unit_cycles)
        for cyc in unique_cycles:
            cyc_mask = unit_cycles == cyc
            cyc_alt = unit_W[cyc_mask, 0]    # alt
            cyc_tra = unit_W[cyc_mask, 2]    # TRA
            cyc_X = unit_X[cyc_mask]
            cyc_W = unit_W[cyc_mask]

            # Cruise-phase filter
            cruise = _cruise_mask(cyc_alt, cyc_tra)
            cyc_X_cruise = cyc_X[cruise]
            cyc_W_cruise = cyc_W[cruise]

            # Per-cycle means (14 sensor channels + 4 W channels)
            sensor_means = cyc_X_cruise.mean(axis=0)
            w_means = cyc_W_cruise.mean(axis=0)

            # RUL and Fc are constant within a cycle — take first
            # Y is (n, 1), so index both row and column
            rul_val = float(unit_Y[cyc_mask].flat[0])
            fc_val = float(unit_A[cyc_mask, 2][0])

            row = {
                "unit_id": uid,
                "cycle": int(cyc),
                "Fc": fc_val,
                "rul": rul_val,
            }
            for i, name in enumerate(SENSOR_NAMES):
                row[name] = sensor_means[i]
            for i, name in enumerate(FLIGHT_COND_NAMES):
                row[name] = w_means[i]

            rows.append(row)

    df = pd.DataFrame(rows)
    if rul_clip is not None:
        df["rul"] = df["rul"].clip(upper=rul_clip)

    return df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Regime normalization
# ---------------------------------------------------------------------------

def normalize_by_regime(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_regimes: int = N_REGIMES,
) -> tuple[pd.DataFrame, pd.DataFrame, KMeans]:
    """Per-operating-regime normalization using flight conditions (W).

    Clusters on cycle-level W means (alt, Mach, TRA, T2) using dev data,
    then Z-score normalizes sensors within each regime using dev statistics.
    Test data is assigned to the nearest dev cluster and normalized with
    dev statistics — no test data influences the normalization.

    This mirrors the original C-MAPSS normalize_by_regime (pipeline/data_loader.py)
    which clusters on setting_1, setting_2, setting_3 for FD002/FD004.
    """
    dev_df = dev_df.copy()
    test_df = test_df.copy()

    # Cluster on flight conditions (dev only)
    w_cols = FLIGHT_COND_NAMES
    w_dev = dev_df[w_cols].values
    km = KMeans(n_clusters=n_regimes, n_init=10, random_state=42)
    dev_df["regime"] = km.fit_predict(w_dev)

    w_test = test_df[w_cols].values
    test_df["regime"] = km.predict(w_test)

    # Per-regime sensor normalization (dev statistics only)
    for regime in range(n_regimes):
        mask_dev = dev_df["regime"] == regime
        mask_test = test_df["regime"] == regime
        if mask_dev.sum() == 0:
            continue
        scaler = StandardScaler()
        dev_df.loc[mask_dev, ACTIVE_SENSORS] = scaler.fit_transform(
            dev_df.loc[mask_dev, ACTIVE_SENSORS]
        )
        if mask_test.sum() > 0:
            test_df.loc[mask_test, ACTIVE_SENSORS] = scaler.transform(
                test_df.loc[mask_test, ACTIVE_SENSORS]
            )

    return dev_df, test_df, km


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
    dataset_name: str, rul_clip: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load an N-CMAPSS sub-dataset from HDF5, cruise-filter, and aggregate.

    Args:
        dataset_name: e.g., "DS01", "DS03", etc.
        rul_clip: if provided, clip RUL to this value

    Returns:
        dev_df: development data, aggregated per cycle, with RUL
        test_df: test data, aggregated per cycle, with RUL
    """
    filepath = find_h5_file(dataset_name)
    print(f"   Loading {filepath.name}...")

    # Validate sensor column order matches config
    import h5py as _h5
    with _h5.File(filepath, "r") as _f:
        if "X_s_var" in _f:
            file_order = [x.decode() if isinstance(x, bytes) else x
                          for x in _f["X_s_var"][:]]
            if file_order != SENSOR_NAMES:
                raise ValueError(
                    f"Sensor column order in {filepath.name} does not match config.\n"
                    f"  File:   {file_order}\n"
                    f"  Config: {SENSOR_NAMES}"
                )

    # Load whitelisted keys only (excludes X_v, T)
    dev_raw = load_h5_arrays(filepath, H5_LOAD_KEYS_DEV)
    test_raw = load_h5_arrays(filepath, H5_LOAD_KEYS_TEST)

    n_dev_raw = len(dev_raw["A_dev"])
    n_test_raw = len(test_raw["A_test"])
    print(f"   Raw dev: {n_dev_raw:,} time-steps, test: {n_test_raw:,} time-steps")

    # Per-unit, per-cycle aggregation with cruise-phase filtering
    dev_df = _aggregate_split(
        dev_raw["W_dev"], dev_raw["X_s_dev"],
        dev_raw["Y_dev"], dev_raw["A_dev"],
        rul_clip=rul_clip,
    )
    # Free raw dev arrays
    del dev_raw

    test_df = _aggregate_split(
        test_raw["W_test"], test_raw["X_s_test"],
        test_raw["Y_test"], test_raw["A_test"],
        rul_clip=rul_clip,
    )
    del test_raw

    print(f"   Aggregated dev: {len(dev_df):,} cycles "
          f"({len(ACTIVE_SENSORS)} sensor channels, cruise-filtered), "
          f"test: {len(test_df):,} cycles")

    return dev_df, test_df


def get_engine_data(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Group by engine unit, return dict of per-engine DataFrames."""
    return {
        int(uid): group.sort_values("cycle")
        for uid, group in df.groupby("unit_id")
    }
