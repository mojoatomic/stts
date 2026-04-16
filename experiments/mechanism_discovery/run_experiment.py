"""Geometric Mechanism Discovery Experiment.

Determines WHY STTS works by measuring seven candidate geometric changes
at every point in every run-to-failure trajectory. The data arbitrates
which mechanism(s) the LDA is actually detecting.

Run: python experiments/mechanism_discovery/run_experiment.py

Reads from existing data files only. Does not re-run the STTS pipeline.
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXP_DIR = PROJECT_ROOT / "experiments" / "mechanism_discovery"
FIG_DIR = EXP_DIR / "figures"

# ---------------------------------------------------------------------------
# Mechanism computations
# ---------------------------------------------------------------------------

def compute_participation_ratio(window: np.ndarray) -> float:
    """M1: PR = (sum(eig))^2 / sum(eig^2). Window is (W, S)."""
    if window.shape[0] < 3:
        return np.nan
    cov = np.cov(window, rowvar=False)
    cov = np.nan_to_num(cov, nan=0.0)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    if len(eigenvalues) == 0:
        return np.nan
    return float(np.sum(eigenvalues)**2 / np.sum(eigenvalues**2))


def compute_centroid_distance(window_mean: np.ndarray, ref_mean: np.ndarray) -> float:
    """M2: Euclidean distance from healthy centroid."""
    return float(np.linalg.norm(window_mean - ref_mean))


def compute_velocity(prev_mean: np.ndarray, curr_mean: np.ndarray) -> float:
    """M3: Rate of change of the mean feature vector."""
    return float(np.linalg.norm(curr_mean - prev_mean))


def compute_mean_variance(window: np.ndarray) -> float:
    """M4: Mean per-feature variance within window."""
    return float(np.mean(np.var(window, axis=0)))


def compute_mean_abs_correlation(window: np.ndarray) -> float:
    """M5: Mean |off-diagonal correlation|."""
    if window.shape[0] < 3 or window.shape[1] < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.corrcoef(window, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    upper = corr[np.triu_indices_from(corr, k=1)]
    if len(upper) == 0:
        return np.nan
    return float(np.mean(np.abs(upper)))


def compute_subspace_angle(window: np.ndarray, ref_cov: np.ndarray, k: int = 3) -> float:
    """M6: Maximum principal angle between current and reference subspace."""
    if window.shape[0] < 3:
        return np.nan
    cov = np.cov(window, rowvar=False)
    cov = np.nan_to_num(cov, nan=0.0)
    ref_cov = np.nan_to_num(ref_cov, nan=0.0)

    _, ref_vecs = np.linalg.eigh(ref_cov)
    _, cur_vecs = np.linalg.eigh(cov)

    k = min(k, window.shape[1], ref_vecs.shape[1])
    if k == 0:
        return np.nan

    inner = ref_vecs[:, -k:].T @ cur_vecs[:, -k:]
    svs = np.linalg.svd(inner, compute_uv=False)
    return float(np.arccos(np.clip(svs[-1], -1, 1)))


def compute_lda_fluctuation(lda_scores: list[float], idx: int, K: int = 5) -> float:
    """M7: Variance of LDA scores in a meta-window."""
    if idx < K:
        return np.nan
    window = lda_scores[idx - K:idx]
    return float(np.var(window))


def compute_autocorr_lag1(lda_scores: list[float], idx: int, K: int = 5) -> float:
    """M7 supplement: Lag-1 autocorrelation of LDA scores in meta-window."""
    if idx < K:
        return np.nan
    window = np.array(lda_scores[idx - K:idx])
    if len(window) < 3 or np.std(window) < 1e-15:
        return np.nan
    return float(np.corrcoef(window[:-1], window[1:])[0, 1])


# ---------------------------------------------------------------------------
# Scaling test: power law vs linear vs exponential
# ---------------------------------------------------------------------------

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def scaling_test(rul: np.ndarray, pr: np.ndarray, rul_clip: float) -> dict:
    """Fit power law, linear, and exponential to PR vs RUL."""
    mask = (rul > 0) & (rul < rul_clip) & np.isfinite(pr) & (pr > 0)
    if mask.sum() < 5:
        return {"alpha": np.nan, "r2_power": np.nan, "r2_linear": np.nan,
                "r2_exp": np.nan, "r2_power_orig": np.nan,
                "r2_linear_orig": np.nan, "r2_exp_orig": np.nan,
                "best_model": "insufficient_data"}

    r = rul[mask].astype(float)
    p = pr[mask].astype(float)

    # Model 1: Power law — log(PR) = α·log(RUL) + log(c)
    log_r = np.log(r)
    log_p = np.log(p)
    try:
        alpha, log_c = np.polyfit(log_r, log_p, 1)
        r2_power_log = r_squared(log_p, alpha * log_r + log_c)
        pred_power = np.exp(log_c) * r ** alpha
        r2_power_orig = r_squared(p, pred_power)
    except (np.linalg.LinAlgError, ValueError):
        alpha, r2_power_log, r2_power_orig = np.nan, np.nan, np.nan
        pred_power = np.full_like(p, np.nan)

    # Model 2: Linear — PR = a·RUL + b
    try:
        a, b = np.polyfit(r, p, 1)
        pred_linear = a * r + b
        r2_linear_orig = r_squared(p, pred_linear)
    except (np.linalg.LinAlgError, ValueError):
        r2_linear_orig = np.nan

    # Model 3: Exponential — log(PR) = b·RUL + log(a)
    try:
        b_exp, log_a_exp = np.polyfit(r, log_p, 1)
        r2_exp_log = r_squared(log_p, b_exp * r + log_a_exp)
        pred_exp = np.exp(log_a_exp) * np.exp(b_exp * r)
        r2_exp_orig = r_squared(p, pred_exp)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        r2_exp_log, r2_exp_orig = np.nan, np.nan

    # Best model by original-space R²
    r2s = {"power": r2_power_orig, "linear": r2_linear_orig, "exponential": r2_exp_orig}
    valid = {k: v for k, v in r2s.items() if np.isfinite(v)}
    best = max(valid, key=valid.get) if valid else "none"

    return {
        "alpha": float(alpha) if np.isfinite(alpha) else None,
        "r2_power": float(r2_power_log) if np.isfinite(r2_power_log) else None,
        "r2_linear": float(r2_linear_orig) if np.isfinite(r2_linear_orig) else None,
        "r2_exp": float(r2_exp_log) if np.isfinite(r2_exp_log) else None,
        "r2_power_orig": float(r2_power_orig) if np.isfinite(r2_power_orig) else None,
        "r2_linear_orig": float(r2_linear_orig) if np.isfinite(r2_linear_orig) else None,
        "r2_exp_orig": float(r2_exp_orig) if np.isfinite(r2_exp_orig) else None,
        "best_model": best,
    }


# ---------------------------------------------------------------------------
# Per-engine mechanism computation
# ---------------------------------------------------------------------------

def compute_engine_mechanisms(
    sensor_windows: list[np.ndarray],
    ruls: np.ndarray,
    lda_scores: np.ndarray,
    rul_clip: float,
    meta_window_k: int = 5,
) -> dict:
    """Compute all 7 mechanisms for one engine's sliding windows.

    Args:
        sensor_windows: list of (W, S) arrays, one per window position
        ruls: (n_windows,) RUL at each position
        lda_scores: (n_windows,) LDA score at each position
        rul_clip: RUL clip value for scaling test
        meta_window_k: meta-window size for M7
    """
    n = len(sensor_windows)

    # Healthy reference: FIRST window (highest RUL) — locked in pre-experiment
    ref_window = sensor_windows[0]
    ref_mean = np.mean(ref_window, axis=0)
    ref_cov = np.cov(ref_window, rowvar=False)
    ref_cov = np.nan_to_num(ref_cov, nan=0.0)

    m1, m2, m3, m4, m5, m6, m7_fluct, m7_ac = [], [], [], [], [], [], [], []

    prev_mean = ref_mean
    lda_list = lda_scores.tolist()

    for i, win in enumerate(sensor_windows):
        win_mean = np.mean(win, axis=0)

        m1.append(compute_participation_ratio(win))
        m2.append(compute_centroid_distance(win_mean, ref_mean))
        m3.append(compute_velocity(prev_mean, win_mean) if i > 0 else 0.0)
        m4.append(compute_mean_variance(win))
        m5.append(compute_mean_abs_correlation(win))
        m6.append(compute_subspace_angle(win, ref_cov))
        m7_fluct.append(compute_lda_fluctuation(lda_list, i, meta_window_k))
        m7_ac.append(compute_autocorr_lag1(lda_list, i, meta_window_k))

        prev_mean = win_mean

    m1 = np.array(m1)
    ruls_arr = np.array(ruls)

    # Scaling test
    sc = scaling_test(ruls_arr, m1, rul_clip)

    return {
        "rul": ruls_arr.tolist(),
        "lda_score": lda_scores.tolist(),
        "m1_participation_ratio": m1.tolist(),
        "m2_centroid_distance": m2,
        "m3_trajectory_velocity": m3,
        "m4_mean_variance": m4,
        "m5_mean_abs_correlation": m5,
        "m6_subspace_angle": m6,
        "m7_lda_fluctuation": m7_fluct,
        "m7_autocorr_lag1": m7_ac,
        "scaling": sc,
    }


# ---------------------------------------------------------------------------
# Correlation tables
# ---------------------------------------------------------------------------

MECHANISM_KEYS = [
    ("m1_participation_ratio", "M1: PR"),
    ("m2_centroid_distance", "M2: Drift"),
    ("m3_trajectory_velocity", "M3: Vel"),
    ("m4_mean_variance", "M4: Var"),
    ("m5_mean_abs_correlation", "M5: Corr"),
    ("m6_subspace_angle", "M6: Rot"),
    ("m7_lda_fluctuation", "M7: Fluct"),
]


def compute_correlation_tables(domain_results: dict) -> dict:
    """Compute ρ-vs-RUL and r-vs-LDA tables for a domain."""
    rho_table = {}
    r_table = {}

    for mech_key, mech_name in MECHANISM_KEYS:
        rhos = []
        rs = []
        for eid, eng in domain_results.items():
            vals = np.array(eng[mech_key])
            rul = np.array(eng["rul"])
            lda = np.array(eng["lda_score"])

            valid = np.isfinite(vals)
            if valid.sum() < 5:
                continue

            v, ru, ld = vals[valid], rul[valid], lda[valid]

            rho, _ = stats.spearmanr(v, ru)
            rhos.append(rho)

            r_val, _ = stats.pearsonr(v, ld)
            rs.append(r_val)

        rho_table[mech_key] = {
            "name": mech_name,
            "mean": float(np.mean(rhos)) if rhos else None,
            "std": float(np.std(rhos)) if rhos else None,
            "range": [float(np.min(rhos)), float(np.max(rhos))] if rhos else None,
            "values": [round(r, 4) for r in rhos],
        }
        r_table[mech_key] = {
            "name": mech_name,
            "mean": float(np.mean(rs)) if rs else None,
            "std": float(np.std(rs)) if rs else None,
            "range": [float(np.min(rs)), float(np.max(rs))] if rs else None,
            "values": [round(r, 4) for r in rs],
        }

    return {"rho_vs_rul": rho_table, "r_vs_lda": r_table}


def compute_collinearity(domain_results: dict) -> np.ndarray:
    """7x7 inter-mechanism Pearson correlation from pooled dev windows."""
    all_mechs = {k: [] for k, _ in MECHANISM_KEYS}

    for eid, eng in domain_results.items():
        for mech_key, _ in MECHANISM_KEYS:
            all_mechs[mech_key].extend(eng[mech_key])

    # Stack into matrix, filter NaN rows
    arrays = [np.array(all_mechs[k]) for k, _ in MECHANISM_KEYS]
    mat = np.column_stack(arrays)
    valid_rows = np.all(np.isfinite(mat), axis=1)
    mat = mat[valid_rows]

    if len(mat) < 10:
        return np.full((7, 7), np.nan)

    return np.corrcoef(mat.T)


# ---------------------------------------------------------------------------
# Domain loaders — extract raw sensor windows + LDA scores
# ---------------------------------------------------------------------------

def load_ncmapss_domain(dataset_name: str, split: str = "both") -> dict:
    """Load N-CMAPSS raw sensor windows and LDA projections."""
    from pipeline.ncmapss.config import (
        ACTIVE_SENSORS, WINDOW_SIZE, WINDOW_STRIDE, ARTIFACTS_DIR,
    )
    from pipeline.ncmapss.data_loader import (
        load_dataset, normalize_by_regime, get_engine_data,
    )
    from pipeline.ncmapss.run_ncmapss import resolve_thresholds
    from pipeline.feature_extraction import build_feature_matrix

    # Load artifacts
    with open(ARTIFACTS_DIR / "scaler_DS03.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(ARTIFACTS_DIR / "lda_DS03.pkl", "rb") as f:
        lda = pickle.load(f)
    gf = np.load(ARTIFACTS_DIR / "good_features_DS03.npy")

    dev_df, test_df = load_dataset(dataset_name, rul_clip=None)
    thresholds = resolve_thresholds(dev_df["rul"].values)
    rul_clip = thresholds["rul_clip"]
    dev_df["rul"] = dev_df["rul"].clip(upper=rul_clip)
    test_df["rul"] = test_df["rul"].clip(upper=rul_clip)

    dev_df, test_df, _ = normalize_by_regime(dev_df, test_df)

    results = {}
    cols = [c for c in ACTIVE_SENSORS if c in dev_df.columns]

    frames = []
    if split in ("both", "dev"):
        frames.append(("dev", get_engine_data(dev_df)))
    if split in ("both", "test"):
        frames.append(("test", get_engine_data(test_df)))

    for split_name, engines in frames:
        for uid, df in engines.items():
            sensor_data = df[cols].values
            rul_data = df["rul"].values
            n_cycles = len(sensor_data)

            if n_cycles < WINDOW_SIZE:
                continue

            # Build raw sensor windows
            sensor_windows = []
            window_ruls = []
            for start in range(0, n_cycles - WINDOW_SIZE + 1, WINDOW_STRIDE):
                end = start + WINDOW_SIZE
                sensor_windows.append(sensor_data[start:end])
                window_ruls.append(rul_data[end - 1])

            # Get LDA scores via saved artifacts
            engine_dict = {uid: sensor_data}
            rul_dict = {uid: rul_data}
            features, meta = build_feature_matrix(
                engine_dict, rul_dict, WINDOW_SIZE, WINDOW_STRIDE,
            )
            scaled = scaler.transform(features)[:, gf]
            lda_scores = lda.transform(scaled).flatten()

            n_windows = min(len(sensor_windows), len(lda_scores))
            sensor_windows = sensor_windows[:n_windows]
            window_ruls = window_ruls[:n_windows]
            lda_scores_trimmed = lda_scores[:n_windows]

            eng_results = compute_engine_mechanisms(
                sensor_windows,
                np.array(window_ruls),
                lda_scores_trimmed,
                rul_clip,
            )
            eng_results["split"] = split_name
            results[f"{split_name}_{uid}"] = eng_results

    return results, rul_clip


def load_cmapss_domain() -> dict:
    """Load C-MAPSS FD001 raw sensor windows."""
    from pipeline.config import (
        COLUMN_NAMES, DATA_DIR, DROP_SENSORS, ACTIVE_SENSORS,
        WINDOW_SIZE, WINDOW_STRIDE, RUL_CLIP,
    )
    from pipeline.data_loader import load_dataset, drop_flat_sensors, normalize_sensors
    from pipeline.feature_extraction import build_feature_matrix

    train_df, test_df, rul_truth = load_dataset("FD001")
    train_df = drop_flat_sensors(train_df)
    train_df, _, _ = normalize_sensors(train_df, test_df.pipe(drop_flat_sensors))

    active = [s for s in ACTIVE_SENSORS if s in train_df.columns]

    # Fit scaler + LDA on training data for LDA scores
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    engines = {}
    for uid, grp in train_df.groupby("unit_id"):
        grp = grp.sort_values("cycle")
        engines[int(uid)] = grp

    sensor_arrays = {uid: df[active].values for uid, df in engines.items()}
    rul_arrays = {uid: df["rul"].values for uid, df in engines.items()}

    features, meta = build_feature_matrix(
        sensor_arrays, rul_arrays, WINDOW_SIZE, WINDOW_STRIDE,
    )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    feat_std = np.std(scaled, axis=0)
    gf = feat_std > 1e-8
    scaled = scaled[:, gf]

    boundaries = np.linspace(0, RUL_CLIP, 7)
    labels = np.clip(np.digitize(meta["rul"], boundaries) - 1, 0, 5)
    lda = LinearDiscriminantAnalysis(n_components=1, solver="eigen", shrinkage="auto")
    lda.fit(scaled, labels)

    # Now compute per-engine
    results = {}
    for uid, df in engines.items():
        sensor_data = df[active].values
        rul_data = df["rul"].values
        n_cycles = len(sensor_data)

        if n_cycles < WINDOW_SIZE:
            continue

        sensor_windows = []
        window_ruls = []
        for start in range(0, n_cycles - WINDOW_SIZE + 1, WINDOW_STRIDE):
            end = start + WINDOW_SIZE
            sensor_windows.append(sensor_data[start:end])
            window_ruls.append(rul_data[end - 1])

        eng_features, eng_meta = build_feature_matrix(
            {uid: sensor_data}, {uid: rul_data}, WINDOW_SIZE, WINDOW_STRIDE,
        )
        eng_scaled = scaler.transform(eng_features)[:, gf]
        lda_scores = lda.transform(eng_scaled).flatten()

        n_windows = min(len(sensor_windows), len(lda_scores))
        sensor_windows = sensor_windows[:n_windows]
        window_ruls = window_ruls[:n_windows]

        eng_results = compute_engine_mechanisms(
            sensor_windows,
            np.array(window_ruls),
            lda_scores[:n_windows],
            RUL_CLIP,
        )
        eng_results["split"] = "train"
        results[f"train_{uid}"] = eng_results

    return results, RUL_CLIP


def load_battery_domain() -> dict:
    """Load NASA battery raw sensor windows."""
    from pipeline.run_battery import (
        load_battery, CLEAN_BATTERIES, WINDOW_SIZE, WINDOW_STRIDE,
        RUL_CLIP, N_LDA_CLASSES,
    )
    from pipeline.feature_extraction import build_feature_matrix
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    all_features_list = []
    all_ruls_list = []
    battery_data = {}

    for bname in CLEAN_BATTERIES:
        try:
            bdata = load_battery(bname)
        except Exception:
            continue

        features = bdata["features"]  # (n_cycles, n_features) — already per-cycle
        n_cycles = len(features)
        rul = np.arange(n_cycles - 1, -1, -1, dtype=float)
        rul = np.clip(rul, 0, RUL_CLIP)

        battery_data[bname] = {
            "features": features,  # these ARE the sensor-level data for battery
            "rul": rul,
        }

    # Fit LDA across all batteries
    sensor_arrays = {b: d["features"] for b, d in battery_data.items()}
    rul_arrays = {b: d["rul"] for b, d in battery_data.items()}

    features, meta = build_feature_matrix(
        sensor_arrays, rul_arrays, WINDOW_SIZE, WINDOW_STRIDE,
    )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    feat_std = np.std(scaled, axis=0)
    gf = feat_std > 1e-8
    scaled = scaled[:, gf]

    boundaries = np.linspace(0, RUL_CLIP, N_LDA_CLASSES + 1)
    labels = np.clip(np.digitize(meta["rul"], boundaries) - 1, 0, N_LDA_CLASSES - 1)
    lda = LinearDiscriminantAnalysis(n_components=1, solver="eigen", shrinkage="auto")
    lda.fit(scaled, labels)

    results = {}
    for bname, bdata in battery_data.items():
        sensor_data = bdata["features"]
        rul_data = bdata["rul"]
        n_cycles = len(sensor_data)

        if n_cycles < WINDOW_SIZE:
            continue

        sensor_windows = []
        window_ruls = []
        for start in range(0, n_cycles - WINDOW_SIZE + 1, WINDOW_STRIDE):
            end = start + WINDOW_SIZE
            sensor_windows.append(sensor_data[start:end])
            window_ruls.append(rul_data[end - 1])

        eng_features, eng_meta = build_feature_matrix(
            {bname: sensor_data}, {bname: rul_data}, WINDOW_SIZE, WINDOW_STRIDE,
        )
        eng_scaled = scaler.transform(eng_features)[:, gf]
        lda_scores = lda.transform(eng_scaled).flatten()

        n_windows = min(len(sensor_windows), len(lda_scores))

        eng_results = compute_engine_mechanisms(
            sensor_windows[:n_windows],
            np.array(window_ruls[:n_windows]),
            lda_scores[:n_windows],
            RUL_CLIP,
        )
        eng_results["split"] = "train"
        results[f"train_{bname}"] = eng_results

    return results, RUL_CLIP


def load_bearing_domain() -> dict:
    """Load PRONOSTIA bearing degradation data."""
    from pipeline.run_pronostia import (
        load_bearing, extract_all_snapshot_features,
        TRAIN_BEARINGS, LEARNING_DIR,
        WINDOW_SIZE, WINDOW_STRIDE, RUL_CLIP, N_LDA_CLASSES,
    )
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    bearing_data = {}
    for bname in TRAIN_BEARINGS:
        bdir = LEARNING_DIR / bname
        if not bdir.exists():
            continue
        snapshots = load_bearing(bdir)
        features = extract_all_snapshot_features(snapshots)
        n = len(features)
        rul = np.clip(np.arange(n - 1, -1, -1, dtype=float), 0, RUL_CLIP)
        bearing_data[bname] = {"features": features, "rul": rul}

    if not bearing_data:
        raise FileNotFoundError("No bearing data found")

    # Fit LDA across all training bearings
    # Use the windowed trajectory features for LDA (same as pipeline)
    from pipeline.run_pronostia import build_trajectory_features
    all_traj_feats = []
    all_traj_ruls = []
    for bname, bd in bearing_data.items():
        traj, ruls = build_trajectory_features(bd["features"], WINDOW_SIZE, WINDOW_STRIDE)
        if len(traj) > 0:
            all_traj_feats.append(traj)
            all_traj_ruls.append(ruls)

    pooled_feats = np.vstack(all_traj_feats)
    pooled_ruls = np.concatenate(all_traj_ruls)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(pooled_feats)
    feat_std = np.std(scaled, axis=0)
    gf = feat_std > 1e-8
    scaled = scaled[:, gf]

    boundaries = np.linspace(0, RUL_CLIP, N_LDA_CLASSES + 1)
    labels = np.clip(np.digitize(pooled_ruls, boundaries) - 1, 0, N_LDA_CLASSES - 1)
    lda = LinearDiscriminantAnalysis(n_components=1, solver="eigen", shrinkage="auto")
    lda.fit(scaled, labels)

    # Per-bearing mechanism computation
    results = {}
    for bname, bd in bearing_data.items():
        sensor_data = bd["features"]  # (n_snapshots, 25) — per-snapshot features
        rul_data = bd["rul"]
        n = len(sensor_data)

        if n < WINDOW_SIZE:
            continue

        sensor_windows = []
        window_ruls = []
        for start in range(0, n - WINDOW_SIZE + 1, WINDOW_STRIDE):
            end = start + WINDOW_SIZE
            sensor_windows.append(sensor_data[start:end])
            window_ruls.append(rul_data[end - 1])

        # LDA scores via windowed trajectory features
        traj, traj_ruls = build_trajectory_features(sensor_data, WINDOW_SIZE, WINDOW_STRIDE)
        traj_scaled = scaler.transform(traj)[:, gf]
        lda_scores = lda.transform(traj_scaled).flatten()

        n_windows = min(len(sensor_windows), len(lda_scores))

        eng_results = compute_engine_mechanisms(
            sensor_windows[:n_windows],
            np.array(window_ruls[:n_windows]),
            lda_scores[:n_windows],
            RUL_CLIP,
        )
        eng_results["split"] = "train"
        results[f"train_{bname}"] = eng_results

    return results, RUL_CLIP


def load_reentry_domain() -> dict:
    """Load Starlink reentry TLE decay trajectories."""
    from reentry.config import (
        STATE_CHANNELS, WINDOW_SIZE, WINDOW_STRIDE_EVAL,
        ARTIFACTS_DIR as REENTRY_ARTIFACTS,
    )
    from reentry.features import tle_records_to_array

    with open(REENTRY_ARTIFACTS / "corpus.pkl", "rb") as f:
        corpus = pickle.load(f)
    with open(REENTRY_ARTIFACTS / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(REENTRY_ARTIFACTS / "lda.pkl", "rb") as f:
        lda = pickle.load(f)

    # Load feature mask if saved, otherwise use all
    try:
        import json as _json
        with open(REENTRY_ARTIFACTS / "model_meta.json") as f:
            meta = _json.load(f)
    except Exception:
        meta = {}

    # Reentry satellites from train split only (these have full trajectories)
    train_reentry_ids = corpus.get("train_reentry_ids", [])
    satellites = corpus["satellites"]

    # We need to build features the same way the pipeline does
    from reentry.features import extract_features as extract_reentry_features

    results = {}
    n_processed = 0
    for norad_id in train_reentry_ids:
        if norad_id not in satellites:
            continue
        sat = satellites[norad_id]
        records = sat["tle_records"]
        if len(records) < WINDOW_SIZE:
            continue

        values, epochs = tle_records_to_array(records)
        n_records = len(values)

        # Compute RUL: days from each record to decay epoch
        from datetime import datetime
        try:
            decay_dt = datetime.fromisoformat(sat["decay_epoch"])
        except (ValueError, TypeError):
            decay_dt = datetime.strptime(str(sat["decay_epoch"])[:10], "%Y-%m-%d")
        j2000 = datetime(2000, 1, 1, 12, 0, 0)
        decay_day = (decay_dt - j2000).total_seconds() / 86400.0
        rul_days = decay_day - epochs
        rul_days = np.clip(rul_days, 0, 365)

        # Build raw sensor windows and extract features for LDA
        sensor_windows = []
        window_ruls = []
        feature_list = []

        stride = WINDOW_STRIDE_EVAL
        for start in range(0, n_records - WINDOW_SIZE + 1, stride):
            end = start + WINDOW_SIZE
            win_vals = values[start:end]
            win_epochs = epochs[start:end]
            sensor_windows.append(win_vals)
            window_ruls.append(rul_days[end - 1])

            feats = extract_reentry_features(win_vals, win_epochs)
            feature_list.append(feats)

        if len(feature_list) < 5:
            continue

        features = np.array(feature_list)
        # Apply saved scaler + LDA
        try:
            scaled = scaler.transform(features)
            lda_scores = lda.transform(scaled).flatten()
        except Exception:
            continue

        n_windows = min(len(sensor_windows), len(lda_scores))

        eng_results = compute_engine_mechanisms(
            sensor_windows[:n_windows],
            np.array(window_ruls[:n_windows]),
            lda_scores[:n_windows],
            365,  # rul_clip for reentry
        )
        eng_results["split"] = "train"
        results[f"train_{norad_id}"] = eng_results
        n_processed += 1

        if n_processed >= 50:  # cap at 50 satellites for tractability
            break

    return results, 365


def load_orbital_domain() -> dict:
    """Load NEA orbital close-approach trajectories."""
    import json as _json

    corpus_path = PROJECT_ROOT / "artifacts" / "corpus.pkl"
    if not corpus_path.exists():
        raise FileNotFoundError("artifacts/corpus.pkl not found — run corpus.py first")

    with open(corpus_path, "rb") as f:
        corpus = pickle.load(f)
    with open(PROJECT_ROOT / "artifacts" / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(PROJECT_ROOT / "artifacts" / "lda.pkl", "rb") as f:
        lda = pickle.load(f)

    train_idx = corpus["train_idx"]
    trajectories = corpus["trajectories"]

    from horizons_stts_pipeline import extract_features

    # Orbital element channel names (from OrbitalElements attrs)
    ORBITAL_CHANNELS = ["q", "a", "e", "i", "om", "w", "ma", "n"]
    WINDOW_DAYS = 30
    STRIDE_DAYS = 7
    RUL_CLIP = 365

    results = {}
    for idx in train_idx:
        traj = trajectories[idx]
        elements = traj["elements"]
        ca_jd = traj["ca_jd"]
        desig = traj["designation"]

        if len(elements) < WINDOW_DAYS:
            continue

        # Build sensor array from OrbitalElements objects
        n_days = len(elements)
        sensor_array = np.zeros((n_days, len(ORBITAL_CHANNELS)))
        jd_array = np.zeros(n_days)
        for i, elem in enumerate(elements):
            jd_array[i] = elem.jd
            for j, ch in enumerate(ORBITAL_CHANNELS):
                sensor_array[i, j] = getattr(elem, ch, 0.0)

        # RUL = days to close approach
        rul_days = np.clip(ca_jd - jd_array, 0, RUL_CLIP)

        # Build raw sensor windows + extract features for LDA
        sensor_windows = []
        window_ruls = []
        feature_list = []

        for start in range(0, n_days - WINDOW_DAYS + 1, STRIDE_DAYS):
            end = start + WINDOW_DAYS
            win_elements = elements[start:end]
            sensor_windows.append(sensor_array[start:end])
            window_ruls.append(rul_days[end - 1])

            feats = extract_features(win_elements, ca_jd)
            feature_list.append(feats)

        if len(feature_list) < 5:
            continue

        features = np.array(feature_list)
        try:
            scaled = scaler.transform(features)
            lda_scores = lda.transform(scaled).flatten()
        except Exception:
            continue

        n_windows = min(len(sensor_windows), len(lda_scores))

        eng_results = compute_engine_mechanisms(
            sensor_windows[:n_windows],
            np.array(window_ruls[:n_windows]),
            lda_scores[:n_windows],
            RUL_CLIP,
        )
        eng_results["split"] = "train"
        results[f"train_{desig}"] = eng_results

    return results, RUL_CLIP


def load_solar_domain() -> dict:
    """Load SWAN-SF solar active region trajectories."""
    # Import stitching from stts-solar branch
    solar_dir = PROJECT_ROOT / "solar"
    tmp_dir = EXP_DIR / "_solar_tmp"
    tmp_dir.mkdir(exist_ok=True)

    if not (solar_dir / "stitch_trajectories.py").exists():
        # Extract scripts from git branch
        import subprocess
        for script in ["stitch_trajectories.py", "v1_separation_test.py"]:
            result = subprocess.run(
                ["git", "show", f"stts-solar:solar/{script}"],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            )
            if result.returncode != 0:
                raise FileNotFoundError(f"Cannot extract solar/{script} from stts-solar branch")
            (tmp_dir / script).write_text(result.stdout)
        use_dir = tmp_dir
    else:
        use_dir = solar_dir

    import importlib.util
    spec_s = importlib.util.spec_from_file_location(
        "stitch_trajectories", str(use_dir / "stitch_trajectories.py"),
    )
    stitch_mod = importlib.util.module_from_spec(spec_s)
    spec_s.loader.exec_module(stitch_mod)

    spec_v = importlib.util.spec_from_file_location(
        "v1_separation_test", str(use_dir / "v1_separation_test.py"),
    )
    v1_mod = importlib.util.module_from_spec(spec_v)
    spec_v.loader.exec_module(v1_mod)

    stitch_ar = stitch_mod.stitch_ar
    SHARP_PARAMS = stitch_mod.SHARP_PARAMS
    extract_window_features = v1_mod.extract_window_features
    WINDOW_SIZE = v1_mod.WINDOW_SIZE
    WINDOW_STRIDE = v1_mod.WINDOW_STRIDE

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    data_dir = str(PROJECT_ROOT / "data" / "swan-sf")
    partition = 3
    target_ars = ["3341", "3721"]

    all_windows = []
    all_labels = []
    ar_trajectories = {}

    for ar_num in target_ars:
        try:
            traj = stitch_ar(data_dir, partition, ar_num)
        except Exception as e:
            print(f"      Solar AR {ar_num}: {e}")
            continue

        if traj is None:
            continue

        timestamps = traj["trajectory_timestamps"]
        traj_data = traj["trajectory_data"]
        traj_source = traj["trajectory_source"]
        n_steps = len(timestamps)

        if n_steps < 10:
            continue

        sensor_array = np.zeros((n_steps, len(SHARP_PARAMS)))
        labels = []

        for i, ts in enumerate(timestamps):
            row = traj_data.get(ts, {})
            if isinstance(row, dict):
                for j, param in enumerate(SHARP_PARAMS):
                    sensor_array[i, j] = row.get(param, 0.0)
            src = traj_source.get(ts, "NF")
            labels.append(1 if src == "FL" else 0)

        labels = np.array(labels)
        sensor_array = np.nan_to_num(sensor_array)
        ar_trajectories[ar_num] = {
            "sensors": sensor_array,
            "labels": labels,
        }

    if not ar_trajectories:
        raise FileNotFoundError("No solar AR trajectories loaded")

    # Fit LDA on pooled data
    all_features = []
    all_y = []
    per_ar_windows = {}

    for ar_num, ar_data in ar_trajectories.items():
        sensors = ar_data["sensors"]
        labels = ar_data["labels"]
        n = len(sensors)

        if n < WINDOW_SIZE:
            continue

        windows = []
        win_labels = []
        for start in range(0, n - WINDOW_SIZE + 1, WINDOW_STRIDE):
            end = start + WINDOW_SIZE
            win = sensors[start:end]
            feat_result = extract_window_features(win)
            # extract_window_features returns (features, sensor_map, class_map)
            feats = feat_result[0] if isinstance(feat_result, tuple) else feat_result
            windows.append(win)
            all_features.append(feats)
            # Label = 1 if any FL in window, 0 if all NF
            lab = 1 if labels[start:end].sum() > 0 else 0
            all_y.append(lab)
            win_labels.append(lab)

        per_ar_windows[ar_num] = {
            "windows": windows,
            "labels": np.array(win_labels),
        }

    features = np.array(all_features)
    y = np.array(all_y)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    gf = np.std(scaled, axis=0) > 1e-8
    scaled = scaled[:, gf]

    lda = LinearDiscriminantAnalysis(n_components=1, solver="eigen", shrinkage="auto")
    lda.fit(scaled, y)

    # Compute mechanisms per AR
    # For solar, "RUL" isn't natural. Use distance-to-next-FL as proxy:
    # count windows until next FL label
    results = {}
    feat_idx = 0
    for ar_num, ar_info in per_ar_windows.items():
        windows = ar_info["windows"]
        labels = ar_info["labels"]
        n_win = len(windows)

        # Compute "time to next FL" as a RUL-like metric
        rul_proxy = np.zeros(n_win)
        next_fl = n_win  # sentinel
        for i in range(n_win - 1, -1, -1):
            if labels[i] == 1:
                next_fl = 0
            else:
                next_fl += 1
            rul_proxy[i] = next_fl

        # LDA scores
        ar_features = features[feat_idx:feat_idx + n_win]
        ar_scaled = scaler.transform(ar_features)[:, gf]
        lda_scores = lda.transform(ar_scaled).flatten()
        feat_idx += n_win

        eng_results = compute_engine_mechanisms(
            windows,
            rul_proxy,
            lda_scores,
            float(n_win),  # rul_clip
        )
        eng_results["split"] = "train"
        results[f"train_AR{ar_num}"] = eng_results

    # Clean up temp files
    for tmp in [PROJECT_ROOT / "_tmp_stitch.py", PROJECT_ROOT / "_tmp_v1.py"]:
        if tmp.exists():
            tmp.unlink()

    return results, float(max(len(w["windows"]) for w in per_ar_windows.values()))


def load_protein_domain() -> dict:
    """Load mdCATH protein unfolding trajectories."""
    from protein.config import (
        DESCRIPTORS_DIR, TARGET_DOMAINS, UNFOLDING_TEMP,
        DEV_REPLICATES_450K, Q_FAILURE_THRESHOLD, RUL_CLIP,
        WINDOW_SIZE, WINDOW_STRIDE_EVAL, ARTIFACTS_DIR as PROTEIN_ARTIFACTS,
        build_weight_vector as build_protein_weights,
    )
    from protein.descriptors import compute_rul
    from protein.features import extract_features as extract_protein_features

    # Load frozen model artifacts
    with open(PROTEIN_ARTIFACTS / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(PROTEIN_ARTIFACTS / "lda.pkl", "rb") as f:
        lda = pickle.load(f)
    W_protein = build_protein_weights()

    results = {}
    for domain_id in TARGET_DOMAINS:
        desc_dir = DESCRIPTORS_DIR / domain_id

        for rep in DEV_REPLICATES_450K:
            key = f"{UNFOLDING_TEMP}K_rep{rep}"
            npy_path = desc_dir / f"{key}.npy"
            if not npy_path.exists():
                continue

            desc = np.load(npy_path)  # (n_frames, 8)
            q_trajectory = desc[:, 3]
            rul, failure_frame = compute_rul(q_trajectory, Q_FAILURE_THRESHOLD)

            if failure_frame is None:
                continue  # did not unfold

            rul = np.clip(rul, 0, RUL_CLIP)
            n_frames = len(desc)

            if n_frames < WINDOW_SIZE:
                continue

            # Build raw sensor windows and extract features for LDA
            sensor_windows = []
            window_ruls = []
            feature_list = []

            stride = WINDOW_STRIDE_EVAL
            for start in range(0, n_frames - WINDOW_SIZE + 1, stride):
                end = start + WINDOW_SIZE
                win = desc[start:end]
                sensor_windows.append(win)
                window_ruls.append(rul[end - 1])
                feats = extract_protein_features(win)
                feature_list.append(feats)

            if len(feature_list) < 5:
                continue

            features = np.array(feature_list)
            try:
                scaled = scaler.transform(features)
                scaled_w = scaled * W_protein
                lda_scores = lda.transform(scaled_w).flatten()
            except Exception:
                continue

            n_windows = min(len(sensor_windows), len(lda_scores))

            eng_results = compute_engine_mechanisms(
                sensor_windows[:n_windows],
                np.array(window_ruls[:n_windows]),
                lda_scores[:n_windows],
                RUL_CLIP,
            )
            eng_results["split"] = "train"
            results[f"train_{domain_id}_{key}"] = eng_results

    return results, RUL_CLIP


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Geometric Mechanism Discovery Experiment")
    print("=" * 70)
    print()

    all_results = {}
    all_tables = {}
    all_collinearity = {}
    all_scaling = {}

    # --- N-CMAPSS DS03 ---
    print("Loading N-CMAPSS DS03...")
    ds03_results, ds03_clip = load_ncmapss_domain("DS03")
    all_results["ncmapss_ds03"] = ds03_results
    tables_ds03 = compute_correlation_tables(ds03_results)
    all_tables["ncmapss_ds03"] = tables_ds03
    collin_ds03 = compute_collinearity(ds03_results)
    all_collinearity["ncmapss_ds03"] = collin_ds03.tolist()
    print(f"   {len(ds03_results)} engines processed")

    # --- N-CMAPSS DS02 ---
    print("Loading N-CMAPSS DS02...")
    ds02_results, ds02_clip = load_ncmapss_domain("DS02")
    all_results["ncmapss_ds02"] = ds02_results
    tables_ds02 = compute_correlation_tables(ds02_results)
    all_tables["ncmapss_ds02"] = tables_ds02
    collin_ds02 = compute_collinearity(ds02_results)
    all_collinearity["ncmapss_ds02"] = collin_ds02.tolist()
    print(f"   {len(ds02_results)} engines processed")

    # --- C-MAPSS FD001 ---
    print("Loading C-MAPSS FD001...")
    try:
        cmapss_results, cmapss_clip = load_cmapss_domain()
        all_results["cmapss_fd001"] = cmapss_results
        tables_cmapss = compute_correlation_tables(cmapss_results)
        all_tables["cmapss_fd001"] = tables_cmapss
        collin_cmapss = compute_collinearity(cmapss_results)
        all_collinearity["cmapss_fd001"] = collin_cmapss.tolist()
        print(f"   {len(cmapss_results)} engines processed")
    except Exception as e:
        print(f"   SKIPPED: {e}")

    # --- Battery ---
    print("Loading NASA Battery...")
    try:
        battery_results, battery_clip = load_battery_domain()
        all_results["battery"] = battery_results
        tables_battery = compute_correlation_tables(battery_results)
        all_tables["battery"] = tables_battery
        collin_battery = compute_collinearity(battery_results)
        all_collinearity["battery"] = collin_battery.tolist()
        print(f"   {len(battery_results)} engines processed")
    except Exception as e:
        print(f"   SKIPPED: {e}")

    # --- Bearing (PRONOSTIA) removed 2026-04-16 ---
    # Six accelerated-rig bearings is not a sample that can support any
    # geometric cross-domain claim. If a real-operating-condition bearing
    # dataset becomes available, re-introduce; until then, bearing is
    # excluded from this experiment. The load_bearing_domain() helper
    # above is retained so the re-introduction is a one-line change.

    # --- Reentry ---
    print("Loading Starlink Reentry...")
    try:
        reentry_results, reentry_clip = load_reentry_domain()
        all_results["reentry"] = reentry_results
        tables_reentry = compute_correlation_tables(reentry_results)
        all_tables["reentry"] = tables_reentry
        collin_reentry = compute_collinearity(reentry_results)
        all_collinearity["reentry"] = collin_reentry.tolist()
        print(f"   {len(reentry_results)} satellites processed")
    except Exception as e:
        print(f"   SKIPPED: {e}")
        import traceback; traceback.print_exc()

    # --- Orbital ---
    print("Loading NEA Orbital...")
    try:
        orbital_results, orbital_clip = load_orbital_domain()
        all_results["orbital"] = orbital_results
        tables_orbital = compute_correlation_tables(orbital_results)
        all_tables["orbital"] = tables_orbital
        collin_orbital = compute_collinearity(orbital_results)
        all_collinearity["orbital"] = collin_orbital.tolist()
        print(f"   {len(orbital_results)} objects processed")
    except Exception as e:
        print(f"   SKIPPED: {e}")
        import traceback; traceback.print_exc()

    # --- Solar ---
    print("Loading SWAN-SF Solar...")
    try:
        solar_results, solar_clip = load_solar_domain()
        all_results["solar"] = solar_results
        tables_solar = compute_correlation_tables(solar_results)
        all_tables["solar"] = tables_solar
        collin_solar = compute_collinearity(solar_results)
        all_collinearity["solar"] = collin_solar.tolist()
        print(f"   {len(solar_results)} active regions processed")
    except Exception as e:
        print(f"   SKIPPED: {e}")
        import traceback; traceback.print_exc()

    # --- Protein ---
    print("Loading mdCATH Protein...")
    try:
        protein_results, protein_clip = load_protein_domain()
        all_results["protein"] = protein_results
        tables_protein = compute_correlation_tables(protein_results)
        all_tables["protein"] = tables_protein
        collin_protein = compute_collinearity(protein_results)
        all_collinearity["protein"] = collin_protein.tolist()
        print(f"   {len(protein_results)} trajectories processed")
    except Exception as e:
        print(f"   SKIPPED: {e}")
        import traceback; traceback.print_exc()

    # --- Print summary tables ---
    print()
    print("=" * 70)
    print("ARBITRATION RESULTS")
    print("=" * 70)

    for domain_name, tables in all_tables.items():
        print(f"\n--- {domain_name} ---")
        print(f"{'Mechanism':<15s} {'ρ vs RUL (mean)':>16s} {'ρ range':>20s} "
              f"{'r vs LDA (mean)':>16s} {'r range':>20s}")
        print("-" * 90)
        for mech_key, mech_name in MECHANISM_KEYS:
            rho = tables["rho_vs_rul"].get(mech_key, {})
            r = tables["r_vs_lda"].get(mech_key, {})
            rho_mean = f"{rho['mean']:+.3f}" if rho.get("mean") is not None else "N/A"
            rho_range = (f"[{rho['range'][0]:+.3f}, {rho['range'][1]:+.3f}]"
                         if rho.get("range") else "N/A")
            r_mean = f"{r['mean']:+.3f}" if r.get("mean") is not None else "N/A"
            r_range = (f"[{r['range'][0]:+.3f}, {r['range'][1]:+.3f}]"
                       if r.get("range") else "N/A")
            print(f"{mech_name:<15s} {rho_mean:>16s} {rho_range:>20s} "
                  f"{r_mean:>16s} {r_range:>20s}")

    # --- Scaling test summary ---
    print()
    print("=" * 70)
    print("SCALING TEST")
    print("=" * 70)

    for domain_name, domain_data in all_results.items():
        alphas = []
        power_wins = 0
        total = 0
        for eid, eng in domain_data.items():
            sc = eng.get("scaling", {})
            if sc.get("alpha") is not None:
                alphas.append(sc["alpha"])
                total += 1
                if sc.get("best_model") == "power":
                    power_wins += 1

        if alphas:
            print(f"\n--- {domain_name} ---")
            print(f"   Mean α: {np.mean(alphas):.3f}  Std: {np.std(alphas):.3f}  "
                  f"Range: [{np.min(alphas):.3f}, {np.max(alphas):.3f}]")
            print(f"   Power law wins: {power_wins}/{total} "
                  f"({100*power_wins/total:.0f}%)")

    # --- Collinearity ---
    print()
    print("=" * 70)
    print("COLLINEARITY (flagged pairs |r| > 0.85)")
    print("=" * 70)

    mech_names = [name for _, name in MECHANISM_KEYS]
    for domain_name, cmat in all_collinearity.items():
        cmat = np.array(cmat)
        print(f"\n--- {domain_name} ---")
        flagged = []
        for i in range(7):
            for j in range(i + 1, 7):
                if abs(cmat[i, j]) > 0.85:
                    flagged.append(
                        f"   {mech_names[i]} <-> {mech_names[j]}: r={cmat[i,j]:.3f}"
                    )
        if flagged:
            for f in flagged:
                print(f)
        else:
            print("   No collinear pairs detected")

    # --- Save JSON ---
    output = {
        "domains": {},
        "correlation_tables": all_tables,
        "collinearity": all_collinearity,
    }

    # Trim trajectories for JSON size — keep summaries
    for domain_name, domain_data in all_results.items():
        domain_summary = {}
        for eid, eng in domain_data.items():
            domain_summary[eid] = {
                "n_windows": len(eng["rul"]),
                "rul_range": [min(eng["rul"]), max(eng["rul"])],
                "split": eng.get("split", "unknown"),
                "scaling": eng.get("scaling", {}),
                # Mechanism-RUL correlations
                "correlations": {},
            }
            for mech_key, mech_name in MECHANISM_KEYS:
                vals = np.array(eng[mech_key])
                rul = np.array(eng["rul"])
                lda = np.array(eng["lda_score"])
                valid = np.isfinite(vals)
                if valid.sum() >= 5:
                    rho, rho_p = stats.spearmanr(vals[valid], rul[valid])
                    r, r_p = stats.pearsonr(vals[valid], lda[valid])
                    domain_summary[eid]["correlations"][mech_key] = {
                        "rho_vs_rul": round(rho, 4),
                        "r_vs_lda": round(r, 4),
                    }

        output["domains"][domain_name] = domain_summary

    # Cross-domain scaling summary
    cross_domain_scaling = {}
    for domain_name, domain_data in all_results.items():
        alphas = [eng["scaling"]["alpha"] for eng in domain_data.values()
                  if eng.get("scaling", {}).get("alpha") is not None]
        power_wins = sum(1 for eng in domain_data.values()
                         if eng.get("scaling", {}).get("best_model") == "power")
        total = sum(1 for eng in domain_data.values()
                    if eng.get("scaling", {}).get("best_model") not in (None, "insufficient_data"))
        if alphas:
            cross_domain_scaling[domain_name] = {
                "mean_alpha": round(np.mean(alphas), 4),
                "std_alpha": round(np.std(alphas), 4),
                "range_alpha": [round(np.min(alphas), 4), round(np.max(alphas), 4)],
                "power_law_wins_pct": round(100 * power_wins / max(total, 1), 1),
                "n_engines": len(alphas),
            }
    output["cross_domain_scaling"] = cross_domain_scaling

    json_path = EXP_DIR / "mechanism_discovery.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # --- Save summary markdown ---
    md_lines = ["# Geometric Mechanism Discovery — Summary\n"]
    for domain_name, tables in all_tables.items():
        md_lines.append(f"\n## {domain_name}\n")
        md_lines.append("| Mechanism | ρ vs RUL (mean) | ρ vs RUL (range) "
                        "| r vs LDA (mean) | r vs LDA (range) |")
        md_lines.append("|-----------|-----------------|------------------"
                        "|-----------------|------------------|")
        for mech_key, mech_name in MECHANISM_KEYS:
            rho = tables["rho_vs_rul"].get(mech_key, {})
            r = tables["r_vs_lda"].get(mech_key, {})
            rho_m = f"{rho['mean']:+.3f}" if rho.get("mean") is not None else "N/A"
            rho_r = (f"[{rho['range'][0]:+.3f}, {rho['range'][1]:+.3f}]"
                     if rho.get("range") else "N/A")
            r_m = f"{r['mean']:+.3f}" if r.get("mean") is not None else "N/A"
            r_r = (f"[{r['range'][0]:+.3f}, {r['range'][1]:+.3f}]"
                   if r.get("range") else "N/A")
            md_lines.append(f"| {mech_name} | {rho_m} | {rho_r} | {r_m} | {r_r} |")

    md_lines.append("\n## Scaling Test — Cross-Domain Summary\n")
    md_lines.append("| Domain | Mean α | Std α | Range α | % power law wins |")
    md_lines.append("|--------|--------|-------|---------|-----------------|")
    for domain_name, sc in cross_domain_scaling.items():
        md_lines.append(
            f"| {domain_name} | {sc['mean_alpha']:.3f} | {sc['std_alpha']:.3f} "
            f"| [{sc['range_alpha'][0]:.3f}, {sc['range_alpha'][1]:.3f}] "
            f"| {sc['power_law_wins_pct']:.0f}% |"
        )

    md_path = EXP_DIR / "mechanism_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Summary saved to {md_path}")

    # --- Save collinearity markdown ---
    col_lines = ["# Mechanism Collinearity Matrix\n"]
    for domain_name, cmat in all_collinearity.items():
        cmat = np.array(cmat)
        col_lines.append(f"\n## {domain_name}\n")
        header = "|      | " + " | ".join(f"{n:>7s}" for n in mech_names) + " |"
        col_lines.append(header)
        col_lines.append("|------|" + "|".join(["--------"] * 7) + "|")
        for i, name in enumerate(mech_names):
            row = f"| {name:<4s} |"
            for j in range(7):
                val = cmat[i, j]
                flag = " *" if i != j and abs(val) > 0.85 else "  "
                row += f" {val:+.3f}{flag}|"
            col_lines.append(row)
        col_lines.append("\n\\* Flagged: |r| > 0.85")

    col_path = EXP_DIR / "mechanism_collinearity.md"
    with open(col_path, "w") as f:
        f.write("\n".join(col_lines))
    print(f"Collinearity saved to {col_path}")

    # Stash full trajectory data for figure generation
    traj_path = EXP_DIR / "trajectories.npz"
    traj_data = {}
    for domain_name, domain_data in all_results.items():
        for eid, eng in domain_data.items():
            prefix = f"{domain_name}__{eid}"
            for key in ["rul", "lda_score", "m1_participation_ratio",
                        "m2_centroid_distance", "m3_trajectory_velocity",
                        "m4_mean_variance", "m5_mean_abs_correlation",
                        "m6_subspace_angle", "m7_lda_fluctuation",
                        "m7_autocorr_lag1"]:
                arr = np.array(eng[key], dtype=float)
                traj_data[f"{prefix}__{key}"] = arr
            # Scaling info
            sc = eng.get("scaling", {})
            traj_data[f"{prefix}__scaling_alpha"] = np.array(
                [sc.get("alpha", np.nan)]
            )

    np.savez_compressed(traj_path, **traj_data)
    print(f"Trajectories saved to {traj_path}")

    print("\nDone. Run generate_figures.py next.")


if __name__ == "__main__":
    main()
