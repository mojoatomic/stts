"""Load and preprocess NASA C-MAPSS turbofan engine degradation data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pipeline.config import (
    COLUMN_NAMES, DATA_DIR, DROP_SENSORS, ACTIVE_SENSORS,
    SETTING_COLS, RUL_CLIP,
)


def load_dataset(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load train, test, and RUL truth files for a C-MAPSS sub-dataset.

    Returns:
        train_df: training data with RUL column added
        test_df: test data (partial trajectories)
        rul_truth: 1-D array of true RUL for each test engine
    """
    train_path = DATA_DIR / f"train_{dataset_name}.txt"
    test_path = DATA_DIR / f"test_{dataset_name}.txt"
    rul_path = DATA_DIR / f"RUL_{dataset_name}.txt"

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    rul_truth = pd.read_csv(rul_path, sep=r"\s+", header=None).values.flatten()

    # Compute RUL for training data: each engine runs to failure
    train_df["rul"] = train_df.groupby("unit_id")["cycle"].transform(
        lambda c: c.max() - c
    )
    train_df["rul"] = train_df["rul"].clip(upper=RUL_CLIP)

    return train_df, test_df, rul_truth


def drop_flat_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Remove sensors with near-constant readings (no degradation signal)."""
    return df.drop(columns=[s for s in DROP_SENSORS if s in df.columns])


def normalize_sensors(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Z-score normalize active sensor columns using training statistics.

    For FD001/FD003 (single operating condition) this is straightforward.
    For FD002/FD004 (multiple conditions), call normalize_by_regime instead.
    """
    scaler = StandardScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()

    cols = [s for s in ACTIVE_SENSORS if s in train_df.columns]
    train_df[cols] = scaler.fit_transform(train_df[cols])
    test_df[cols] = scaler.transform(test_df[cols])

    return train_df, test_df, scaler


def normalize_by_regime(
    train_df: pd.DataFrame, test_df: pd.DataFrame, n_regimes: int = 6
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-operating-regime normalization for FD002/FD004.

    Clusters on operational settings, then normalizes sensors within each
    regime using training data statistics.
    """
    from sklearn.cluster import KMeans

    train_df = train_df.copy()
    test_df = test_df.copy()

    # Cluster operating settings (they cluster cleanly when rounded)
    settings_train = train_df[SETTING_COLS].round(3).values
    km = KMeans(n_clusters=n_regimes, n_init=10, random_state=42)
    train_df["regime"] = km.fit_predict(settings_train)

    settings_test = test_df[SETTING_COLS].round(3).values
    test_df["regime"] = km.predict(settings_test)

    cols = [s for s in ACTIVE_SENSORS if s in train_df.columns]
    for regime in range(n_regimes):
        mask_train = train_df["regime"] == regime
        mask_test = test_df["regime"] == regime
        if mask_train.sum() == 0:
            continue
        scaler = StandardScaler()
        train_df.loc[mask_train, cols] = scaler.fit_transform(
            train_df.loc[mask_train, cols]
        )
        if mask_test.sum() > 0:
            test_df.loc[mask_test, cols] = scaler.transform(
                test_df.loc[mask_test, cols]
            )

    return train_df, test_df


def get_engine_data(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Group dataframe by engine unit, return dict of per-engine DataFrames."""
    return {uid: group.sort_values("cycle") for uid, group in df.groupby("unit_id")}
