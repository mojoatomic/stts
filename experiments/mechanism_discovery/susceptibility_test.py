"""Susceptibility Divergence Test.

Tests whether degrading systems exhibit divergent susceptibility — increasing
sensitivity to operating-condition perturbations as failure approaches.

N-CMAPSS only: requires per-timestep W (operating conditions) and X_s (sensors)
within each flight cycle. C-MAPSS/Battery/Bearing lack within-cycle resolution.

Run: python experiments/mechanism_discovery/susceptibility_test.py
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXP_DIR = PROJECT_ROOT / "experiments" / "mechanism_discovery"
FIG_DIR = EXP_DIR / "figures"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 8, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
})

from pipeline.ncmapss.config import (
    ACTIVE_SENSORS, SENSOR_NAMES, FLIGHT_COND_NAMES,
    H5_LOAD_KEYS_DEV, H5_LOAD_KEYS_TEST,
    WINDOW_SIZE, WINDOW_STRIDE, ARTIFACTS_DIR,
    CRUISE_TRA_WINDOW, CRUISE_TRA_STD_MAX, CRUISE_ALT_MIN,
    CRUISE_MIN_TIMESTEPS,
)
from pipeline.ncmapss.data_loader import (
    load_h5_arrays, find_h5_file, _cruise_mask,
    load_dataset, normalize_by_regime, get_engine_data,
)
from pipeline.ncmapss.run_ncmapss import resolve_thresholds
from pipeline.feature_extraction import build_feature_matrix

ENGINE_COLORS = plt.cm.tab10.colors


# ---------------------------------------------------------------------------
# Per-cycle sensitivity computation
# ---------------------------------------------------------------------------

def compute_cycle_sensitivity(
    W_cycle: np.ndarray, X_s_cycle: np.ndarray,
) -> np.ndarray:
    """Compute sensitivity matrix for one flight cycle.

    For each (sensor, operating condition) pair, compute |slope| of
    the within-cycle linear regression. Uses raw per-timestep data.

    Args:
        W_cycle: (n_timesteps, 4) operating conditions
        X_s_cycle: (n_timesteps, 14) sensor values

    Returns:
        (14, 4) sensitivity matrix — |slope| for each pair
    """
    n_sensors = X_s_cycle.shape[1]
    n_conditions = W_cycle.shape[1]
    sensitivity = np.zeros((n_sensors, n_conditions))

    for s in range(n_sensors):
        for c in range(n_conditions):
            x = W_cycle[:, c]
            y = X_s_cycle[:, s]
            # Skip if no variance in the operating condition
            if np.std(x) < 1e-10:
                sensitivity[s, c] = 0.0
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                slope, _, _, _, _ = stats.linregress(x, y)
            sensitivity[s, c] = abs(slope)

    return sensitivity


def compute_chi_variants(sensitivity: np.ndarray) -> dict:
    """Reduce sensitivity matrix to scalar susceptibility measures."""
    return {
        "chi_mean": float(np.mean(sensitivity)),
        "chi_frob": float(np.linalg.norm(sensitivity, "fro")),
        "chi_max": float(np.max(sensitivity)),
    }


# ---------------------------------------------------------------------------
# Per-engine susceptibility trajectory
# ---------------------------------------------------------------------------

def compute_engine_susceptibility(
    dataset_name: str,
    unit_id: int,
    split: str,
) -> dict | None:
    """Compute per-cycle susceptibility for one engine from raw HDF5 data.

    Uses UN-normalized sensor data — the whole point is measuring
    sensitivity to operating conditions, so the variation must be present.
    """
    filepath = find_h5_file(dataset_name)
    suffix = "dev" if split == "dev" else "test"

    with h5py.File(filepath, "r") as f:
        A = f[f"A_{suffix}"][:]
        W = f[f"W_{suffix}"][:]
        X_s = f[f"X_s_{suffix}"][:]
        Y = f[f"Y_{suffix}"][:]

    units = A[:, 0].astype(int)
    cycles = A[:, 1].astype(int)
    unit_mask = units == unit_id

    if unit_mask.sum() == 0:
        return None

    unit_W = W[unit_mask]
    unit_X = X_s[unit_mask]
    unit_Y = Y[unit_mask]
    unit_cycles = cycles[unit_mask]
    unit_alt = unit_W[:, 0]
    unit_tra = unit_W[:, 2]

    unique_cycles = np.unique(unit_cycles)
    chi_mean_list = []
    chi_frob_list = []
    chi_max_list = []
    rul_list = []
    sensitivity_matrices = []

    for cyc in unique_cycles:
        cyc_mask = unit_cycles == cyc
        cyc_W = unit_W[cyc_mask]
        cyc_X = unit_X[cyc_mask]
        cyc_alt = unit_alt[cyc_mask]
        cyc_tra = unit_tra[cyc_mask]

        # Cruise filter (same as main pipeline)
        cruise = _cruise_mask(cyc_alt, cyc_tra)
        cyc_W_cruise = cyc_W[cruise]
        cyc_X_cruise = cyc_X[cruise]

        if len(cyc_W_cruise) < 20:
            # Too few cruise timesteps for reliable regression
            continue

        sens = compute_cycle_sensitivity(cyc_W_cruise, cyc_X_cruise)
        chi = compute_chi_variants(sens)

        chi_mean_list.append(chi["chi_mean"])
        chi_frob_list.append(chi["chi_frob"])
        chi_max_list.append(chi["chi_max"])
        rul_list.append(float(unit_Y[cyc_mask].flat[0]))
        sensitivity_matrices.append(sens)

    if len(rul_list) < 5:
        return None

    return {
        "rul": rul_list,
        "chi_mean": chi_mean_list,
        "chi_frob": chi_frob_list,
        "chi_max": chi_max_list,
        "sensitivity_early": sensitivity_matrices[0].tolist(),
        "sensitivity_late": sensitivity_matrices[-1].tolist(),
    }


# ---------------------------------------------------------------------------
# Scaling test (reused from mechanism experiment)
# ---------------------------------------------------------------------------

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def scaling_test_susceptibility(rul, chi, rul_clip):
    """Fit power law χ ~ RUL^(-γ), linear, exponential."""
    mask = (np.array(rul) > 0) & (np.array(rul) < rul_clip) & (np.array(chi) > 0)
    r = np.array(rul, dtype=float)[mask]
    c = np.array(chi, dtype=float)[mask]

    if len(r) < 5:
        return {"gamma": None, "r2_power_orig": None, "r2_linear_orig": None,
                "r2_exp_orig": None, "best_model": "insufficient_data"}

    log_r, log_c = np.log(r), np.log(c)

    try:
        neg_gamma, log_k = np.polyfit(log_r, log_c, 1)
        gamma = -neg_gamma  # χ ~ RUL^(-γ), so slope in log-log is -γ
        pred_power = np.exp(log_k) * r ** neg_gamma
        r2_pow = r_squared(c, pred_power)
    except Exception:
        gamma, r2_pow = None, None

    try:
        a, b = np.polyfit(r, c, 1)
        r2_lin = r_squared(c, a * r + b)
    except Exception:
        r2_lin = None

    try:
        b_exp, log_a = np.polyfit(r, log_c, 1)
        pred_exp = np.exp(log_a) * np.exp(b_exp * r)
        r2_exp = r_squared(c, pred_exp)
    except Exception:
        r2_exp = None

    r2s = {"power": r2_pow, "linear": r2_lin, "exponential": r2_exp}
    valid = {k: v for k, v in r2s.items() if v is not None}
    best = max(valid, key=valid.get) if valid else "none"

    return {
        "gamma": round(gamma, 4) if gamma is not None else None,
        "r2_power_orig": round(r2_pow, 4) if r2_pow is not None else None,
        "r2_linear_orig": round(r2_lin, 4) if r2_lin is not None else None,
        "r2_exp_orig": round(r2_exp, 4) if r2_exp is not None else None,
        "best_model": best,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_dataset(dataset_name: str) -> dict:
    """Run susceptibility analysis on one N-CMAPSS dataset."""
    print(f"\n--- {dataset_name} ---")

    # Get unit IDs from HDF5
    filepath = find_h5_file(dataset_name)
    with h5py.File(filepath, "r") as f:
        dev_units = np.unique(f["A_dev"][:, 0].astype(int)).tolist()
        test_units = np.unique(f["A_test"][:, 0].astype(int)).tolist()

    # Get RUL thresholds and LDA scores
    dev_df, test_df = load_dataset(dataset_name, rul_clip=None)
    thresholds = resolve_thresholds(dev_df["rul"].values)
    rul_clip = thresholds["rul_clip"]
    dev_df["rul"] = dev_df["rul"].clip(upper=rul_clip)
    test_df["rul"] = test_df["rul"].clip(upper=rul_clip)
    dev_df_norm, test_df_norm, _ = normalize_by_regime(dev_df, test_df)

    # LDA scores from saved DS03 artifacts
    with open(ARTIFACTS_DIR / "scaler_DS03.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(ARTIFACTS_DIR / "lda_DS03.pkl", "rb") as f:
        lda = pickle.load(f)
    gf = np.load(ARTIFACTS_DIR / "good_features_DS03.npy")

    cols = [c for c in ACTIVE_SENSORS if c in dev_df_norm.columns]

    # Compute LDA scores per engine
    lda_by_engine = {}
    for split_name, df_norm, unit_list in [
        ("dev", dev_df_norm, dev_units), ("test", test_df_norm, test_units)
    ]:
        engines = get_engine_data(df_norm)
        for uid in unit_list:
            if uid not in engines:
                continue
            eng_df = engines[uid]
            sensor_data = eng_df[cols].values
            rul_data = eng_df["rul"].values
            if len(sensor_data) < WINDOW_SIZE:
                continue
            eng_dict = {uid: sensor_data}
            rul_dict = {uid: rul_data}
            features, meta = build_feature_matrix(
                eng_dict, rul_dict, WINDOW_SIZE, WINDOW_STRIDE,
            )
            scaled = scaler.transform(features)[:, gf]
            lda_scores = lda.transform(scaled).flatten()
            lda_by_engine[f"{split_name}_{uid}"] = {
                "lda": lda_scores,
                "rul": meta["rul"],
            }

    # Compute centroid drift (M2) for partial correlation
    # Centroid = first window mean per engine from regime-normalized data
    m2_by_engine = {}
    for split_name, df_norm, unit_list in [
        ("dev", dev_df_norm, dev_units), ("test", test_df_norm, test_units)
    ]:
        engines = get_engine_data(df_norm)
        for uid in unit_list:
            if uid not in engines:
                continue
            eng_df = engines[uid]
            sensor_data = eng_df[cols].values
            if len(sensor_data) < WINDOW_SIZE:
                continue
            ref_mean = np.mean(sensor_data[:WINDOW_SIZE], axis=0)
            dists = []
            for start in range(0, len(sensor_data) - WINDOW_SIZE + 1, WINDOW_STRIDE):
                end = start + WINDOW_SIZE
                win_mean = np.mean(sensor_data[start:end], axis=0)
                dists.append(float(np.linalg.norm(win_mean - ref_mean)))
            m2_by_engine[f"{split_name}_{uid}"] = np.array(dists)

    # Process each engine
    results = {}
    for split_name, unit_list in [("dev", dev_units), ("test", test_units)]:
        for uid in unit_list:
            key = f"{split_name}_{uid}"
            print(f"   Processing {key}...")

            susc = compute_engine_susceptibility(dataset_name, uid, split_name)
            if susc is None:
                print(f"      Skipped (insufficient data)")
                continue

            rul_arr = np.array(susc["rul"])
            rul_clipped = np.clip(rul_arr, 0, rul_clip)

            # Match susceptibility cycles to LDA windows
            # Susceptibility is per-cycle; LDA is per-window (window_size cycles)
            # Align by taking the last (n_windows) susceptibility values
            lda_info = lda_by_engine.get(key)
            m2_info = m2_by_engine.get(key)

            eng_result = {
                "split": split_name,
                "rul": rul_clipped.tolist(),
                "chi_mean": susc["chi_mean"],
                "chi_frob": susc["chi_frob"],
                "chi_max": susc["chi_max"],
                "sensitivity_early": susc["sensitivity_early"],
                "sensitivity_late": susc["sensitivity_late"],
            }

            # Correlations with RUL
            for chi_key in ["chi_mean", "chi_frob", "chi_max"]:
                chi_arr = np.array(susc[chi_key])
                if len(chi_arr) >= 5:
                    rho, p = stats.spearmanr(chi_arr, rul_clipped)
                    eng_result[f"rho_{chi_key}_rul"] = round(rho, 4)
                    eng_result[f"rho_p_{chi_key}_rul"] = float(p)

            # Correlation with LDA (align lengths)
            if lda_info is not None:
                n_lda = len(lda_info["lda"])
                n_chi = len(susc["chi_mean"])
                # Window positions correspond to cycles [WINDOW_SIZE-1 ... n_cycles-1]
                # Susceptibility is per-cycle [0 ... n_cycles-1]
                # Align: take the last n_lda chi values (which correspond to later cycles)
                if n_chi >= n_lda:
                    chi_aligned = np.array(susc["chi_mean"])[n_chi - n_lda:]
                    lda_aligned = lda_info["lda"]
                    rul_aligned = lda_info["rul"]

                    if len(chi_aligned) >= 5:
                        r_val, _ = stats.pearsonr(chi_aligned, lda_aligned)
                        eng_result["r_chi_mean_lda"] = round(r_val, 4)

                    # Partial correlation controlling for M2
                    if m2_info is not None and len(m2_info) == n_lda:
                        m2_aligned = m2_info
                        if len(chi_aligned) >= 5:
                            lr = LinearRegression()
                            m2_col = m2_aligned.reshape(-1, 1)

                            lr.fit(m2_col, chi_aligned)
                            chi_resid = chi_aligned - lr.predict(m2_col)

                            rul_f = rul_aligned.astype(float)
                            lr.fit(m2_col, rul_f)
                            rul_resid = rul_f - lr.predict(m2_col)

                            partial_rho, partial_p = stats.spearmanr(chi_resid, rul_resid)
                            eng_result["partial_rho_chi_rul_given_m2"] = round(partial_rho, 4)
                            eng_result["partial_p"] = float(partial_p)

            # Scaling test (use best chi variant — chi_mean as default)
            sc = scaling_test_susceptibility(rul_clipped, susc["chi_mean"], rul_clip)
            eng_result["scaling"] = sc

            results[key] = eng_result

    return results, rul_clip


def build_summary(results: dict, domain: str) -> dict:
    """Build summary statistics for one domain."""
    summary = {}
    for chi_key in ["chi_mean", "chi_frob", "chi_max"]:
        rho_key = f"rho_{chi_key}_rul"
        rhos = [eng[rho_key] for eng in results.values() if rho_key in eng]
        r_key = "r_chi_mean_lda"
        rs = [eng[r_key] for eng in results.values() if r_key in eng]
        partial_key = "partial_rho_chi_rul_given_m2"
        partials = [eng[partial_key] for eng in results.values() if partial_key in eng]

        summary[chi_key] = {
            "mean_rho_vs_rul": round(np.mean(rhos), 4) if rhos else None,
            "std_rho": round(np.std(rhos), 4) if rhos else None,
            "range_rho": [round(min(rhos), 4), round(max(rhos), 4)] if rhos else None,
            "n_negative_rho": sum(1 for r in rhos if r < 0),
            "n_positive_rho": sum(1 for r in rhos if r > 0),
            "n_total": len(rhos),
        }

    summary["lda_correlation"] = {
        "mean_r_vs_lda": round(np.mean(rs), 4) if rs else None,
        "range_r": [round(min(rs), 4), round(max(rs), 4)] if rs else None,
    }
    summary["partial_correlation"] = {
        "mean_partial_rho": round(np.mean(partials), 4) if partials else None,
        "range_partial": [round(min(partials), 4), round(max(partials), 4)] if partials else None,
    }

    # Scaling
    gammas = [eng["scaling"]["gamma"] for eng in results.values()
              if eng.get("scaling", {}).get("gamma") is not None]
    power_wins = sum(1 for eng in results.values()
                     if eng.get("scaling", {}).get("best_model") == "power")
    total_sc = sum(1 for eng in results.values()
                   if eng.get("scaling", {}).get("best_model") not in (None, "insufficient_data"))
    summary["scaling"] = {
        "mean_gamma": round(np.mean(gammas), 4) if gammas else None,
        "std_gamma": round(np.std(gammas), 4) if gammas else None,
        "pct_power_wins": round(100 * power_wins / max(total_sc, 1), 1),
    }

    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_susceptibility_vs_rul(results, domain, chi_key="chi_mean"):
    dev_engines = {k: v for k, v in results.items() if v["split"] == "dev"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (eid, eng) in enumerate(sorted(dev_engines.items())):
        color = ENGINE_COLORS[i % len(ENGINE_COLORS)]
        ax.plot(eng["rul"], eng[chi_key], color=color, alpha=0.7,
                linewidth=1.2, label=eid.replace("dev_", "U"))
    ax.set_xlabel("RUL (cycles)")
    ax.set_ylabel("Susceptibility (χ mean)")
    ax.set_title(f"Susceptibility vs RUL — {domain}")
    ax.invert_xaxis()
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(FIG_DIR / f"susceptibility_vs_rul_{domain.lower()}.png")
    plt.close(fig)
    print(f"   Saved susceptibility_vs_rul_{domain.lower()}.png")


def plot_susceptibility_unit10(results_ds03):
    test_engines = {k: v for k, v in results_ds03.items() if v["split"] == "test"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for eid, eng in sorted(test_engines.items()):
        is_u10 = "10" in eid
        style = {"color": "#b2182b", "linewidth": 2.5, "linestyle": "--",
                 "zorder": 10, "label": "Unit 10"} if is_u10 else {
            "color": "gray", "linewidth": 1.0, "alpha": 0.5,
            "label": eid.replace("test_", "U")}
        ax.plot(eng["rul"], eng["chi_mean"], **style)
    ax.set_xlabel("RUL (cycles)")
    ax.set_ylabel("Susceptibility (χ mean)")
    ax.set_title("Unit 10 Susceptibility vs Other Test Engines")
    ax.invert_xaxis()
    ax.legend(fontsize=9)
    fig.savefig(FIG_DIR / "susceptibility_unit10.png")
    plt.close(fig)
    print("   Saved susceptibility_unit10.png")


def plot_sensitivity_matrix(results, domain):
    """Heatmap of sensitivity matrix early vs late life."""
    dev_engines = {k: v for k, v in results.items() if v["split"] == "dev"}
    # Pick longest
    rep = max(dev_engines, key=lambda k: len(dev_engines[k]["rul"]))
    eng = dev_engines[rep]

    early = np.array(eng["sensitivity_early"])
    late = np.array(eng["sensitivity_late"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    vmax = max(early.max(), late.max())

    im1 = ax1.imshow(early, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(FLIGHT_COND_NAMES, rotation=45, ha="right")
    ax1.set_yticks(range(14))
    ax1.set_yticklabels(SENSOR_NAMES, fontsize=8)
    ax1.set_title(f"Early Life (RUL={int(eng['rul'][0])})")
    ax1.set_xlabel("Operating Condition")
    ax1.set_ylabel("Sensor")

    im2 = ax2.imshow(late, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(FLIGHT_COND_NAMES, rotation=45, ha="right")
    ax2.set_yticks(range(14))
    ax2.set_yticklabels(SENSOR_NAMES, fontsize=8)
    ax2.set_title(f"End of Life (RUL={int(eng['rul'][-1])})")
    ax2.set_xlabel("Operating Condition")

    fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, label="|Sensitivity| (slope)")
    uid = rep.replace("dev_", "")
    fig.suptitle(f"Sensitivity Matrix — {domain} Unit {uid}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"susceptibility_sensitivity_matrix_{domain.lower()}.png")
    plt.close(fig)
    print(f"   Saved susceptibility_sensitivity_matrix_{domain.lower()}.png")


def plot_scaling(results, domain):
    """Log-log plot of χ vs RUL with competing fits."""
    dev_engines = {k: v for k, v in results.items() if v["split"] == "dev"}
    rep = max(dev_engines, key=lambda k: len(dev_engines[k]["rul"]))
    eng = dev_engines[rep]

    rul = np.array(eng["rul"], dtype=float)
    chi = np.array(eng["chi_mean"], dtype=float)
    mask = (rul > 0) & (chi > 0)
    r, c = rul[mask], chi[mask]

    log_r, log_c = np.log(r), np.log(c)
    sort_idx = np.argsort(r)

    try:
        neg_gamma, log_k = np.polyfit(log_r, log_c, 1)
        pred_pow = np.exp(log_k) * r ** neg_gamma
        r2_pow = r_squared(c, pred_pow)
    except Exception:
        pred_pow, r2_pow, neg_gamma = np.full_like(c, np.nan), 0, 0

    try:
        a, b = np.polyfit(r, c, 1)
        pred_lin = a * r + b
        r2_lin = r_squared(c, pred_lin)
    except Exception:
        pred_lin, r2_lin = np.full_like(c, np.nan), 0

    try:
        b_exp, log_a = np.polyfit(r, log_c, 1)
        pred_exp = np.exp(log_a) * np.exp(b_exp * r)
        r2_exp = r_squared(c, pred_exp)
    except Exception:
        pred_exp, r2_exp = np.full_like(c, np.nan), 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(r, c, c="#999999", s=10, alpha=0.5)
    ax1.plot(r[sort_idx], pred_pow[sort_idx], "-", color="#2166ac", linewidth=2,
             label=f"Power: γ={-neg_gamma:.3f}, R²={r2_pow:.3f}")
    ax1.plot(r[sort_idx], pred_lin[sort_idx], "--", color="#b2182b", linewidth=2,
             label=f"Linear: R²={r2_lin:.3f}")
    ax1.plot(r[sort_idx], pred_exp[sort_idx], ":", color="#1b7837", linewidth=2,
             label=f"Exp: R²={r2_exp:.3f}")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("RUL (log)")
    ax1.set_ylabel("χ (log)")
    ax1.set_title("Log-Log")
    ax1.legend(fontsize=9)

    ax2.scatter(r, c, c="#999999", s=10, alpha=0.5)
    ax2.plot(r[sort_idx], pred_pow[sort_idx], "-", color="#2166ac", linewidth=2,
             label=f"Power: R²={r2_pow:.3f}")
    ax2.plot(r[sort_idx], pred_lin[sort_idx], "--", color="#b2182b", linewidth=2,
             label=f"Linear: R²={r2_lin:.3f}")
    ax2.plot(r[sort_idx], pred_exp[sort_idx], ":", color="#1b7837", linewidth=2,
             label=f"Exp: R²={r2_exp:.3f}")
    ax2.set_xlabel("RUL")
    ax2.set_ylabel("χ")
    ax2.set_title("Original Space")
    ax2.legend(fontsize=9)

    uid = rep.replace("dev_", "")
    fig.suptitle(f"Susceptibility Scaling — {domain} Unit {uid}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"susceptibility_scaling_{domain.lower()}.png")
    plt.close(fig)
    print(f"   Saved susceptibility_scaling_{domain.lower()}.png")


# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Susceptibility Divergence Test")
    print("=" * 70)

    all_results = {}
    all_summaries = {}

    for ds in ["DS03", "DS02"]:
        results, rul_clip = process_dataset(ds)
        all_results[ds] = results
        all_summaries[ds] = build_summary(results, ds)

    # --- Print results ---
    print("\n" + "=" * 70)
    print("SUSCEPTIBILITY RESULTS")
    print("=" * 70)

    for ds, summary in all_summaries.items():
        print(f"\n--- {ds} ---")
        for chi_key in ["chi_mean", "chi_frob", "chi_max"]:
            s = summary[chi_key]
            print(f"   {chi_key}: ρ vs RUL mean={s['mean_rho_vs_rul']}, "
                  f"range={s['range_rho']}, "
                  f"neg/pos={s['n_negative_rho']}/{s['n_positive_rho']}")
        lda = summary["lda_correlation"]
        print(f"   r vs LDA mean={lda['mean_r_vs_lda']}, range={lda['range_r']}")
        partial = summary["partial_correlation"]
        print(f"   Partial ρ (given M2) mean={partial['mean_partial_rho']}, "
              f"range={partial['range_partial']}")
        sc = summary["scaling"]
        print(f"   Scaling: γ mean={sc['mean_gamma']}, "
              f"power wins={sc['pct_power_wins']}%")

    # --- Save JSON ---
    output = {
        "results": {},
        "summaries": all_summaries,
    }
    for ds, results in all_results.items():
        output["results"][ds] = {}
        for eid, eng in results.items():
            # Trim large arrays for JSON
            eng_out = {k: v for k, v in eng.items()
                       if k not in ("sensitivity_early", "sensitivity_late")}
            eng_out["sensitivity_early"] = eng["sensitivity_early"]
            eng_out["sensitivity_late"] = eng["sensitivity_late"]
            output["results"][ds][eid] = eng_out

    json_path = EXP_DIR / "susceptibility_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # --- Summary markdown ---
    md_lines = ["# Susceptibility Divergence Test — Summary\n"]
    md_lines.append("| Domain | χ variant | ρ vs RUL (mean) | ρ range | "
                    "r vs LDA (mean) | Partial ρ (given M2) | γ (mean) | % power wins |")
    md_lines.append("|--------|-----------|-----------------|---------|"
                    "-----------------|----------------------|----------|--------------|")
    for ds, summary in all_summaries.items():
        for chi_key in ["chi_mean", "chi_frob", "chi_max"]:
            s = summary[chi_key]
            lda_r = summary["lda_correlation"]["mean_r_vs_lda"]
            partial = summary["partial_correlation"]["mean_partial_rho"]
            sc = summary["scaling"]
            rho_m = f"{s['mean_rho_vs_rul']:+.3f}" if s["mean_rho_vs_rul"] else "N/A"
            rho_r = (f"[{s['range_rho'][0]:+.3f}, {s['range_rho'][1]:+.3f}]"
                     if s["range_rho"] else "N/A")
            lda_m = f"{lda_r:+.3f}" if lda_r else "N/A"
            part_m = f"{partial:+.3f}" if partial else "N/A"
            gamma_m = f"{sc['mean_gamma']:.3f}" if sc["mean_gamma"] else "N/A"
            pw = f"{sc['pct_power_wins']:.0f}%"
            md_lines.append(
                f"| {ds} | {chi_key.split('_')[1]} | {rho_m} | {rho_r} "
                f"| {lda_m} | {part_m} | {gamma_m} | {pw} |"
            )

    md_path = EXP_DIR / "susceptibility_summary.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Summary saved to {md_path}")

    # --- Figures ---
    print("\nGenerating figures...")
    plot_susceptibility_vs_rul(all_results["DS03"], "DS03")
    plot_susceptibility_vs_rul(all_results["DS02"], "DS02")
    plot_susceptibility_unit10(all_results["DS03"])
    plot_sensitivity_matrix(all_results["DS03"], "DS03")
    plot_scaling(all_results["DS03"], "DS03")

    print("\nDone.")


if __name__ == "__main__":
    main()
