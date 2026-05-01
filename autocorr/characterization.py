#!/usr/bin/env python3
"""
Within-cell autocorrelation characterization on catalog-scale Φ data.

Pre-registered design:
  - Per-window Φ from catalog-scale per-object CSVs (commit fdd1ba3),
    truncated to [0, τ_i] per the C-restricted aggregation in
    flux_characterization_catalog_c.py. α frozen at 4.27517;
    D_KL window W = 5; ε = 1e-9.
  - Within-cell run: contiguous sequence of windows with the same
    M2 cell label, for each transient cell c.
  - Qualifying run length ≥ 3 (primary). Sensitivity at ≥ 5 and ≥ 10.
  - Pooled Pearson ρ_c^(k) for k = 1, 2, 3 across all
    (trajectory, run-position) pairs. Per-trajectory sums precomputed
    so trajectory-level bootstrap (10,000 resamples) is a weighted
    matrix multiply.
  - Stationarity per cell: split each trajectory into early/middle/
    late thirds by window count; recompute pooled ρ_c^(1) per third.
  - AR(1)-corrected variance prediction:
        Var_pred,AR1 = Σ_c E_π[visits to c] · σ²_c · (1 + ρ_c^(1)) / (1 − ρ_c^(1))
    σ²_c is the all-windows marginal variance of Φ in cell c (same
    quantity used in synthetic Test 3's linear correction).
  - Full corrected = 147,991.94 + Var_pred,AR1; r_AR1 = full /
    catalog_Var_emp (33,396,043).
  - Decision bands (pre-registered, NOT adjusted on intermediate
    results):
        r_AR1 ∈ [0.5, 1.5]      → AR(1) recovers gap (Fix A extended).
        r_AR1 ∈ [0.2, 0.5] ∪ [1.5, 4.0] → partial; need additional terms.
        r_AR1 outside [0.2, 4.0] → pivot to bootstrap/simulation.
  - Sparse-cell artifact: if > 50 % of Var_pred,AR1 contribution comes
    from cells with effective sample size (total windows in qualifying
    runs) < 1000, decision rule does NOT apply; flag, then re-run with
    bulk-only cells.

Read-only against frozen catalog-scale artifacts. No new pipeline run.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reentry.flux_accumulator import (
    DKL_EPS,
    DKL_WINDOW,
    c_restricted_corpus,
    per_window_dkl,
    phi as compute_phi,
)
from reentry.train import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/autocorr")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PER_OBJECT_DIR = Path("results/reentry/catalog_scale_validation/per_object")
CATALOG_PRED_VAR = Path(
    "results/reentry/flux_characterization_catalog_c/predicted_variance.json"
)
CATALOG_EMP_VAR = Path(
    "results/reentry/flux_characterization_catalog_c/empirical_variance.json"
)
FROZEN_ALPHA = 4.27517
CATALOG_VAR_EMP = 33_396_042.877546873  # commit fdd1ba3
KEMENY_BASELINE = 147_991.9398997442    # commit fdd1ba3, recomputed under empirical π

LAGS = [1, 2, 3]
MIN_RUN_LENGTH_PRIMARY = 3
MIN_RUN_LENGTH_SENSITIVITIES = [3, 5, 10]
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 20260502
HEAVY_TAIL_TRIM_PCT = 0.99   # keep bottom 99 %, trim top 1 %
SPARSE_CELL_N_THRESHOLD = 1000
SPARSE_CELL_SHARE_THRESHOLD = 0.50

# Decision bands on r_AR1
R_BAND_FULL_LO, R_BAND_FULL_HI = 0.5, 1.5
R_BAND_PARTIAL_LO, R_BAND_PARTIAL_HI = 0.2, 4.0


# ── Data loading ────────────────────────────────────────────────

def load_catalog_trajectory(path: Path) -> dict:
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    cells = np.array([int(r["M2_cell"]) for r in rows], dtype=np.int64)
    H = np.array([float(r["H_t"]) for r in rows], dtype=np.float64)
    return {"nid": path.stem, "cells": cells, "H": H, "n_windows": len(rows)}


# ── Run identification ──────────────────────────────────────────

def find_runs(cells: np.ndarray, target_cell: int, min_length: int) -> list[tuple[int, int]]:
    """Return [(start, end_exclusive), ...] for contiguous runs of
    `target_cell` of length ≥ `min_length`.
    """
    runs = []
    n = len(cells)
    i = 0
    while i < n:
        if cells[i] == target_cell:
            start = i
            while i < n and cells[i] == target_cell:
                i += 1
            length = i - start
            if length >= min_length:
                runs.append((start, i))
        else:
            i += 1
    return runs


# ── Per-trajectory pair-sum tensor ──────────────────────────────

def per_traj_sums(trajectories: list[dict], cell: int, lag: int,
                   min_run_length: int) -> tuple[np.ndarray, int]:
    """Return (n_traj, 6) array of [n_pairs, sum_X, sum_Y, sum_XX,
    sum_YY, sum_XY] per trajectory, and total effective sample size
    (sum of run lengths across all qualifying runs in cell c).

    Pairs come from runs of length ≥ min_run_length; for lag k a run
    of length L contributes (L − k) pairs only if L > k.
    """
    n_traj = len(trajectories)
    sums = np.zeros((n_traj, 6), dtype=np.float64)
    total_window_count = 0
    for j, traj in enumerate(trajectories):
        cells = traj["cells_truncated"]
        phi_t = traj["phi_truncated"]
        runs = find_runs(cells, cell, min_run_length)
        for (s, e) in runs:
            run_phi = phi_t[s:e]
            total_window_count += len(run_phi)
            if len(run_phi) <= lag:
                continue
            X = run_phi[:-lag]
            Y = run_phi[lag:]
            sums[j, 0] += len(X)
            sums[j, 1] += float(X.sum())
            sums[j, 2] += float(Y.sum())
            sums[j, 3] += float((X * X).sum())
            sums[j, 4] += float((Y * Y).sum())
            sums[j, 5] += float((X * Y).sum())
    return sums, total_window_count


def pearson_from_sums(weighted_sums: np.ndarray) -> float:
    """Pearson ρ from the 6-vector (n, sum_X, sum_Y, sum_XX, sum_YY, sum_XY)."""
    n, sx, sy, sxx, syy, sxy = weighted_sums
    if n < 2:
        return float("nan")
    mean_x = sx / n
    mean_y = sy / n
    var_x = sxx / n - mean_x * mean_x
    var_y = syy / n - mean_y * mean_y
    cov_xy = sxy / n - mean_x * mean_y
    if var_x <= 0 or var_y <= 0:
        return float("nan")
    return float(cov_xy / np.sqrt(var_x * var_y))


# ── Bootstrap ───────────────────────────────────────────────────

def bootstrap_pearson_ci(per_traj_sum_arr: np.ndarray, B: int, seed: int,
                          alpha: float = 0.05) -> tuple[float, float, float, np.ndarray]:
    """Trajectory-level bootstrap percentile CI on Pearson ρ.

    Returns (point, ci_lo, ci_hi, replicate_array).
    Replicate array is the bootstrap distribution (length B).
    """
    n_traj = per_traj_sum_arr.shape[0]
    point = pearson_from_sums(per_traj_sum_arr.sum(axis=0))
    rng = np.random.default_rng(seed)
    M = np.zeros((B, n_traj), dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n_traj, size=n_traj)
        M[b] = np.bincount(idx, minlength=n_traj)
    weighted = M @ per_traj_sum_arr  # (B, 6)
    repl = np.array([pearson_from_sums(weighted[b]) for b in range(B)])
    finite = np.isfinite(repl)
    if not finite.any():
        return point, float("nan"), float("nan"), repl
    lo = float(np.quantile(repl[finite], alpha / 2))
    hi = float(np.quantile(repl[finite], 1 - alpha / 2))
    return point, lo, hi, repl


# ── Build trajectory data with truncated Φ ──────────────────────

def build_trajectory_data() -> list[dict]:
    log.info("Loading catalog-scale per-object trajectories...")
    csv_paths = sorted(PER_OBJECT_DIR.glob("*.csv"))
    t0 = time.time()
    per_object = []
    for p in csv_paths:
        traj = load_catalog_trajectory(p)
        if traj is None or traj["n_windows"] == 0:
            continue
        per_object.append(traj)
    log.info(f"  loaded {len(per_object)} non-empty trajectories in "
             f"{time.time()-t0:.1f}s")

    model = load_model()
    P = model["markov"]["transition_matrix"]
    failure_cells = sorted(model["markov"]["failure_cells"].tolist())

    log.info(f"Computing D_KL trajectories (W={DKL_WINDOW}, ε={DKL_EPS})...")
    t0 = time.time()
    for t in per_object:
        t["dkl"] = per_window_dkl(t["cells"], P, window=DKL_WINDOW, eps=DKL_EPS)
    log.info(f"  done in {time.time()-t0:.1f}s")

    restriction = c_restricted_corpus(per_object, failure_cells)
    log.info(f"  C-restricted: {restriction['n_included']} included")

    nid_to_traj = {t["nid"]: t for t in per_object}
    trajectories = []
    for r in restriction["restricted"]:
        end = int(r["tau_i"]) + 1
        traj = nid_to_traj[r["nid"]]
        H = traj["H"][:end]
        dkl = traj["dkl"][:end]
        cells = traj["cells"][:end]
        # Drop windows with NaN D_KL (early-window convention; within-run
        # contiguity is preserved if we drop those windows, because the
        # first DKL_WINDOW windows are always at trajectory start and are
        # contiguous).
        mask = np.isfinite(H) & np.isfinite(dkl)
        Hf = H[mask]; dklf = dkl[mask]; cf = cells[mask]
        phi_t = compute_phi(Hf, dklf, FROZEN_ALPHA)
        trajectories.append({
            "nid": r["nid"],
            "c_first": int(r["c_first"]),
            "tau_i": int(r["tau_i"]),
            "cells_truncated": cf.astype(np.int64),
            "phi_truncated": phi_t.astype(np.float64),
            "n_windows_truncated": int(len(phi_t)),
        })
    log.info(f"  trajectory data ready: {len(trajectories)} restricted "
             f"trajectories with truncated Φ")
    return trajectories


# ── E_π[visits to c] from catalog artifacts ─────────────────────

def load_catalog_E_visits(restricted_trajectories: list[dict]) -> tuple[np.ndarray, list[int], np.ndarray, np.ndarray]:
    pv = json.loads(CATALOG_PRED_VAR.read_text())
    surviving = pv["surviving_cells"]
    N_committed = np.array(pv["N"], dtype=np.float64)  # (n_surv, n_surv)
    cell_idx = {c: i for i, c in enumerate(surviving)}
    pi_counts = np.zeros(len(surviving), dtype=np.float64)
    for traj in restricted_trajectories:
        s = traj["c_first"]
        if s in cell_idx:
            pi_counts[cell_idx[s]] += 1.0
    pi = pi_counts / pi_counts.sum() if pi_counts.sum() > 0 else None
    E_visits = pi @ N_committed  # (n_surv,)
    return surviving, pi, E_visits, N_committed


# ── Marginal σ²_c from all-windows Φ ───────────────────────────

def per_cell_sigma2(trajectories: list[dict], surviving: list[int]) -> tuple[np.ndarray, np.ndarray, dict]:
    """All-windows marginal σ²_c (matches Test 3 convention).

    Also returns the per-cell pooled Φ array for later kurtosis /
    trim sensitivity.
    """
    cell_phi = {c: [] for c in surviving}
    for traj in trajectories:
        cf = traj["cells_truncated"]
        pt = traj["phi_truncated"]
        for c in surviving:
            mask = cf == c
            if mask.any():
                cell_phi[c].extend(pt[mask].tolist())
    sigma2 = np.zeros(len(surviving), dtype=np.float64)
    n_windows = np.zeros(len(surviving), dtype=np.int64)
    for i, c in enumerate(surviving):
        vals = np.array(cell_phi[c], dtype=np.float64)
        n_windows[i] = len(vals)
        if len(vals) >= 2:
            sigma2[i] = float(np.var(vals, ddof=1))
    return sigma2, n_windows, cell_phi


# ── Stationarity per cell — early/middle/late thirds ───────────

def per_traj_sums_in_third(trajectories: list[dict], cell: int, lag: int,
                            min_run_length: int, third_index: int) -> np.ndarray:
    """Same as per_traj_sums but restrict to runs that fall entirely
    within the early (0), middle (1), or late (2) third of the
    trajectory's window range.
    """
    n_traj = len(trajectories)
    sums = np.zeros((n_traj, 6), dtype=np.float64)
    for j, traj in enumerate(trajectories):
        cells = traj["cells_truncated"]
        phi_t = traj["phi_truncated"]
        n = len(cells)
        if n < 6:
            continue  # too short to split into thirds
        boundaries = [0, n // 3, (2 * n) // 3, n]
        lo, hi = boundaries[third_index], boundaries[third_index + 1]
        if hi - lo < min_run_length:
            continue
        sub_cells = cells[lo:hi]
        sub_phi = phi_t[lo:hi]
        runs = find_runs(sub_cells, cell, min_run_length)
        for (s, e) in runs:
            run_phi = sub_phi[s:e]
            if len(run_phi) <= lag:
                continue
            X = run_phi[:-lag]; Y = run_phi[lag:]
            sums[j, 0] += len(X)
            sums[j, 1] += X.sum()
            sums[j, 2] += Y.sum()
            sums[j, 3] += (X * X).sum()
            sums[j, 4] += (Y * Y).sum()
            sums[j, 5] += (X * Y).sum()
    return sums


# ── AR(1) variance correction ──────────────────────────────────

def ar1_corrected_variance(E_visits: np.ndarray, sigma2: np.ndarray,
                             rho: np.ndarray) -> tuple[float, np.ndarray]:
    """Var_AR1_term = Σ_c E[visits] · σ²_c · (1 + ρ) / (1 − ρ).
    Returns (total, per_cell_contribution).
    """
    # Clip ρ away from ±1 to avoid divergence
    rho_safe = np.clip(rho, -0.999, 0.999)
    factor = (1.0 + rho_safe) / (1.0 - rho_safe)
    contrib = E_visits * sigma2 * factor
    return float(contrib.sum()), contrib


# ── Heavy-tail trim sensitivity ────────────────────────────────

def trimmed_sums(trajectories: list[dict], cell: int, lag: int,
                  min_run_length: int, trim_threshold: float) -> np.ndarray:
    """Per-traj pair sums after trimming Φ values exceeding the per-cell
    trim_threshold (e.g. 99th percentile of pooled Φ in cell c).

    Pairs that include any trimmed value are dropped.
    """
    n_traj = len(trajectories)
    sums = np.zeros((n_traj, 6), dtype=np.float64)
    for j, traj in enumerate(trajectories):
        cells = traj["cells_truncated"]
        phi_t = traj["phi_truncated"]
        runs = find_runs(cells, cell, min_run_length)
        for (s, e) in runs:
            run_phi = phi_t[s:e]
            if len(run_phi) <= lag:
                continue
            X = run_phi[:-lag]
            Y = run_phi[lag:]
            keep = (X <= trim_threshold) & (Y <= trim_threshold)
            X = X[keep]; Y = Y[keep]
            if len(X) == 0:
                continue
            sums[j, 0] += len(X)
            sums[j, 1] += X.sum()
            sums[j, 2] += Y.sum()
            sums[j, 3] += (X * X).sum()
            sums[j, 4] += (Y * Y).sum()
            sums[j, 5] += (X * Y).sum()
    return sums


# ── Main ────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("Within-cell autocorrelation characterization (catalog-scale Φ)")
    log.info("=" * 70)

    trajectories = build_trajectory_data()
    n_traj = len(trajectories)

    surviving, pi, E_visits, N_committed = load_catalog_E_visits(trajectories)
    log.info(f"  surviving transient cells: {surviving}")
    log.info(f"  E_π[visits to c]: {E_visits.tolist()}")

    sigma2, sigma2_n_windows, cell_phi_pool = per_cell_sigma2(trajectories, surviving)
    log.info("  σ²_c per surviving cell:")
    for i, c in enumerate(surviving):
        log.info(f"    cell {c:>2d}  n={sigma2_n_windows[i]:>7,d}  σ²={sigma2[i]:.4f}")

    # ── Run identification ─────────────────────────────────────
    log.info("Run identification (length ≥ 3)...")
    run_id = {"min_run_length": MIN_RUN_LENGTH_PRIMARY, "per_cell": {}}
    for c in surviving:
        n_runs_total = 0
        n_windows_total = 0
        n_traj_with_run = 0
        for traj in trajectories:
            runs = find_runs(traj["cells_truncated"], c, MIN_RUN_LENGTH_PRIMARY)
            if runs:
                n_traj_with_run += 1
                n_runs_total += len(runs)
                n_windows_total += sum((e - s) for (s, e) in runs)
        run_id["per_cell"][c] = {
            "n_qualifying_runs": n_runs_total,
            "n_windows_in_qualifying_runs": n_windows_total,
            "n_trajectories_with_qualifying_run": n_traj_with_run,
        }
        log.info(f"    cell {c:>2d}: runs={n_runs_total:>6,d}  "
                 f"windows={n_windows_total:>7,d}  n_traj={n_traj_with_run:>4,d}")
    (OUT_DIR / "run_identification.json").write_text(json.dumps(run_id, indent=2))

    # ── Pooled ρ_c^(k) for k=1,2,3 with bootstrap CI ───────────
    log.info("Pooled Pearson ρ_c^(k) for k = 1, 2, 3 with bootstrap CI...")
    rho_results = {}
    rho_point_lag1 = np.zeros(len(surviving))  # used downstream
    for i, c in enumerate(surviving):
        rho_results[c] = {}
        for k in LAGS:
            sums = per_traj_sums(trajectories, c, k, MIN_RUN_LENGTH_PRIMARY)[0]
            n_pairs_total = float(sums[:, 0].sum())
            point, lo, hi, repl = bootstrap_pearson_ci(
                sums, BOOTSTRAP_B, BOOTSTRAP_SEED + 1000 * c + k,
            )
            rho_results[c][k] = {
                "point": point,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_pairs": int(n_pairs_total),
                "effective_n_windows": run_id["per_cell"][c]["n_windows_in_qualifying_runs"],
            }
            if k == 1:
                rho_point_lag1[i] = point if np.isfinite(point) else 0.0
        log.info(f"    cell {c:>2d}: "
                 f"ρ^(1)={rho_results[c][1]['point']:.4f} CI[{rho_results[c][1]['ci_lo']:.4f},{rho_results[c][1]['ci_hi']:.4f}]  "
                 f"ρ^(2)={rho_results[c][2]['point']:.4f}  "
                 f"ρ^(3)={rho_results[c][3]['point']:.4f}")
    (OUT_DIR / "autocorrelation.json").write_text(json.dumps({
        "min_run_length": MIN_RUN_LENGTH_PRIMARY,
        "BOOTSTRAP_B": BOOTSTRAP_B,
        "per_cell": {str(c): rho_results[c] for c in surviving},
    }, indent=2))

    # ── Stationarity sensitivity ───────────────────────────────
    log.info("Stationarity sensitivity (early/middle/late thirds)...")
    stat_results = {}
    for c in surviving:
        stat_results[c] = {}
        thirds_rho = {}
        for third_index, label in [(0, "early"), (1, "middle"), (2, "late")]:
            sums = per_traj_sums_in_third(trajectories, c, 1,
                                            MIN_RUN_LENGTH_PRIMARY, third_index)
            point, lo, hi, _ = bootstrap_pearson_ci(
                sums, BOOTSTRAP_B,
                BOOTSTRAP_SEED + 100 * c + third_index,
            )
            n_pairs = int(sums[:, 0].sum())
            stat_results[c][label] = {
                "rho_lag1": point, "ci_lo": lo, "ci_hi": hi, "n_pairs": n_pairs,
            }
            thirds_rho[label] = point
        # Pre-registered criterion
        rho_e = thirds_rho["early"]; rho_m = thirds_rho["middle"]; rho_l = thirds_rho["late"]
        non_stationary = False
        reason = "stationary"
        if all(np.isfinite([rho_e, rho_m, rho_l])):
            ci_e_half = (stat_results[c]["early"]["ci_hi"] - stat_results[c]["early"]["ci_lo"]) / 2
            ci_l_half = (stat_results[c]["late"]["ci_hi"] - stat_results[c]["late"]["ci_lo"]) / 2
            delta = abs(rho_l - rho_e)
            if delta > max(ci_e_half, ci_l_half):
                non_stationary = True
                reason = (f"|ρ_late − ρ_early| = {delta:.4f} exceeds CI "
                          f"half-width (early {ci_e_half:.4f}, late {ci_l_half:.4f})")
            mono_inc = rho_e < rho_m < rho_l
            mono_dec = rho_e > rho_m > rho_l
            if (mono_inc or mono_dec) and delta > 0.1:
                non_stationary = True
                reason = (f"monotonic {'increasing' if mono_inc else 'decreasing'} "
                          f"trend, |Δ|={delta:.4f} > 0.1")
        else:
            reason = "insufficient pairs in one or more thirds"
        stat_results[c]["non_stationary_flag"] = non_stationary
        stat_results[c]["reason"] = reason
        log.info(f"    cell {c:>2d}: ρ_early={rho_e:.4f}  ρ_middle={rho_m:.4f}  "
                 f"ρ_late={rho_l:.4f}  non_stat={non_stationary} ({reason})")
    (OUT_DIR / "stationarity.json").write_text(json.dumps(
        {str(c): stat_results[c] for c in surviving}, indent=2,
    ))

    # ── AR(1)-corrected variance prediction ────────────────────
    log.info("AR(1)-corrected variance prediction...")
    Var_AR1_term, contrib = ar1_corrected_variance(E_visits, sigma2, rho_point_lag1)
    Var_pred_full_AR1 = KEMENY_BASELINE + Var_AR1_term
    r_AR1 = Var_pred_full_AR1 / CATALOG_VAR_EMP
    log.info(f"  Var_AR1_term      = {Var_AR1_term:,.2f}")
    log.info(f"  Var_pred,full_AR1 = {Var_pred_full_AR1:,.2f}")
    log.info(f"  catalog Var_emp   = {CATALOG_VAR_EMP:,.2f}")
    log.info(f"  r_AR1             = {r_AR1:.6f}")

    # Bootstrap CI on r_AR1: resample trajectories with replacement,
    # recompute σ²_c, ρ_c^(1), and E_visits (E_visits depends on
    # starting-cell distribution, which itself depends on the resample).
    log.info(f"  Bootstrap CI on r_AR1 (B = {BOOTSTRAP_B:,}, "
             f"seed = {BOOTSTRAP_SEED})...")
    log.info("    (This is the load-bearing step — propagating "
             "trajectory-level uncertainty through σ²_c, ρ_c^(1), and π.)")
    rng = np.random.default_rng(BOOTSTRAP_SEED + 7)
    n_surv = len(surviving)
    cell_to_idx = {c: i for i, c in enumerate(surviving)}
    # Pre-organize per-trajectory data needed for bootstrap:
    #  - count of starts (8/etc) → π (via multiplicities × starts)
    #  - per-cell per-trajectory: sum_phi, sumsq_phi, n_phi (for σ²_c)
    #  - per-cell per-trajectory pair sums for lag 1 (for ρ_c^(1))
    n_phi_jc = np.zeros((n_traj, n_surv), dtype=np.float64)
    sum_phi_jc = np.zeros((n_traj, n_surv), dtype=np.float64)
    sumsq_phi_jc = np.zeros((n_traj, n_surv), dtype=np.float64)
    starts = np.zeros((n_traj, n_surv), dtype=np.float64)
    pair_sums_jc = np.zeros((n_traj, n_surv, 6), dtype=np.float64)  # for lag 1
    for j, traj in enumerate(trajectories):
        s = traj["c_first"]
        if s in cell_to_idx:
            starts[j, cell_to_idx[s]] = 1.0
        cf = traj["cells_truncated"]
        pt = traj["phi_truncated"]
        for i, c in enumerate(surviving):
            mask = cf == c
            if mask.any():
                vals = pt[mask]
                n_phi_jc[j, i] = float(len(vals))
                sum_phi_jc[j, i] = float(vals.sum())
                sumsq_phi_jc[j, i] = float((vals * vals).sum())
        for i, c in enumerate(surviving):
            runs = find_runs(cf, c, MIN_RUN_LENGTH_PRIMARY)
            for (s_run, e_run) in runs:
                run_phi = pt[s_run:e_run]
                if len(run_phi) <= 1:
                    continue
                X = run_phi[:-1]; Y = run_phi[1:]
                pair_sums_jc[j, i, 0] += len(X)
                pair_sums_jc[j, i, 1] += X.sum()
                pair_sums_jc[j, i, 2] += Y.sum()
                pair_sums_jc[j, i, 3] += (X * X).sum()
                pair_sums_jc[j, i, 4] += (Y * Y).sum()
                pair_sums_jc[j, i, 5] += (X * Y).sum()

    r_repl = np.zeros(BOOTSTRAP_B)
    for b in range(BOOTSTRAP_B):
        idx = rng.integers(0, n_traj, size=n_traj)
        m = np.bincount(idx, minlength=n_traj).astype(np.float64)
        # Resampled σ²_c
        n_b = m @ n_phi_jc       # (n_surv,)
        sx_b = m @ sum_phi_jc
        ss_b = m @ sumsq_phi_jc
        sigma2_b = np.zeros(n_surv)
        for i in range(n_surv):
            if n_b[i] >= 2:
                mean_x = sx_b[i] / n_b[i]
                # Use n−1 (unbiased) divisor; here we want sample variance
                sigma2_b[i] = (ss_b[i] - n_b[i] * mean_x * mean_x) / (n_b[i] - 1)
                sigma2_b[i] = max(0.0, float(sigma2_b[i]))
        # Resampled ρ_c^(1)
        ps_b = np.einsum("j,jck->ck", m, pair_sums_jc)  # (n_surv, 6)
        rho_b = np.zeros(n_surv)
        for i in range(n_surv):
            rho_b[i] = pearson_from_sums(ps_b[i])
            if not np.isfinite(rho_b[i]):
                rho_b[i] = 0.0
        # Resampled π (empirical)
        pi_counts_b = m @ starts
        if pi_counts_b.sum() > 0:
            pi_b = pi_counts_b / pi_counts_b.sum()
        else:
            pi_b = np.full(n_surv, 1.0 / n_surv)
        E_visits_b = pi_b @ N_committed
        # Var_AR1 for this resample
        rho_safe = np.clip(rho_b, -0.999, 0.999)
        factor = (1.0 + rho_safe) / (1.0 - rho_safe)
        Var_AR1_b = float((E_visits_b * sigma2_b * factor).sum())
        Var_full_b = KEMENY_BASELINE + Var_AR1_b
        r_repl[b] = Var_full_b / CATALOG_VAR_EMP

    r_AR1_lo = float(np.quantile(r_repl, 0.025))
    r_AR1_hi = float(np.quantile(r_repl, 0.975))
    log.info(f"  r_AR1 95 % CI: [{r_AR1_lo:.6f}, {r_AR1_hi:.6f}]")

    # Decision rule (point estimate + CI band)
    def band(x: float) -> str:
        if R_BAND_FULL_LO <= x <= R_BAND_FULL_HI: return "FULL"
        if R_BAND_PARTIAL_LO <= x <= R_BAND_PARTIAL_HI: return "PARTIAL"
        return "OUTSIDE"
    band_point = band(r_AR1)
    band_lo = band(r_AR1_lo); band_hi = band(r_AR1_hi)
    if band_lo != band_hi:
        decision = "INCONCLUSIVE_AT_CURRENT_DATA"
    else:
        decision = band_point
    log.info(f"  decision: {decision}")

    # Per-cell contribution to Var_AR1 + share
    per_cell_contrib = []
    for i, c in enumerate(surviving):
        per_cell_contrib.append({
            "cell": int(c),
            "kappa_c": None,  # not needed here
            "sigma2_c": float(sigma2[i]),
            "rho_lag1": float(rho_point_lag1[i]),
            "E_visits": float(E_visits[i]),
            "factor_AR1": float((1 + rho_point_lag1[i]) / (1 - rho_point_lag1[i])
                                  if abs(rho_point_lag1[i]) < 0.999 else float("inf")),
            "contribution": float(contrib[i]),
            "share_of_AR1_term": float(contrib[i] / Var_AR1_term)
                if Var_AR1_term > 0 else 0.0,
            "n_qualifying_run_windows": int(run_id["per_cell"][c]["n_windows_in_qualifying_runs"]),
        })

    # Sparse-cell rule
    sparse_share = sum(
        x["share_of_AR1_term"] for x in per_cell_contrib
        if x["n_qualifying_run_windows"] < SPARSE_CELL_N_THRESHOLD
    )
    sparse_dominated = sparse_share > SPARSE_CELL_SHARE_THRESHOLD
    log.info(f"  sparse-cell share of Var_AR1: {sparse_share*100:.2f} %  "
             f"(threshold: {SPARSE_CELL_SHARE_THRESHOLD*100:.0f} %)")
    if sparse_dominated:
        log.info("  → SPARSE-CELL DOMINATED. Decision rule does not apply. "
                 "Re-run with bulk-only cells.")
        # Recompute with bulk cells only
        bulk_mask = np.array([
            x["n_qualifying_run_windows"] >= SPARSE_CELL_N_THRESHOLD
            for x in per_cell_contrib
        ])
        E_visits_bulk = E_visits * bulk_mask
        sigma2_bulk = sigma2 * bulk_mask
        rho_bulk = rho_point_lag1 * bulk_mask
        Var_AR1_bulk = float((E_visits_bulk * sigma2_bulk
                              * (1 + np.clip(rho_bulk, -0.999, 0.999))
                              / (1 - np.clip(rho_bulk, -0.999, 0.999))).sum())
        r_AR1_bulk = (KEMENY_BASELINE + Var_AR1_bulk) / CATALOG_VAR_EMP
        log.info(f"  bulk-only Var_AR1_term: {Var_AR1_bulk:,.2f}")
        log.info(f"  bulk-only r_AR1:         {r_AR1_bulk:.6f}")
    else:
        Var_AR1_bulk = None
        r_AR1_bulk = None

    var_pred_out = {
        "Kemeny_baseline": KEMENY_BASELINE,
        "catalog_Var_emp": CATALOG_VAR_EMP,
        "Var_AR1_term": Var_AR1_term,
        "Var_pred_full_AR1": Var_pred_full_AR1,
        "r_AR1_point": r_AR1,
        "r_AR1_CI95": [r_AR1_lo, r_AR1_hi],
        "decision_band_point": band_point,
        "decision_band_ci_lo": band_lo,
        "decision_band_ci_hi": band_hi,
        "decision": decision,
        "preregistered_decision_rule": (
            "FULL: r_AR1 ∈ [0.5, 1.5] (CI bracket also in same band). "
            "PARTIAL: ∈ [0.2, 0.5] ∪ [1.5, 4.0]. "
            "OUTSIDE: outside [0.2, 4.0]. "
            "INCONCLUSIVE: 95 % CI spans two bands."
        ),
        "per_cell_contribution": per_cell_contrib,
        "sparse_cell_share_of_AR1_term": sparse_share,
        "sparse_cell_dominated": sparse_dominated,
        "sparse_cell_n_threshold": SPARSE_CELL_N_THRESHOLD,
        "bulk_only": {
            "Var_AR1_term": Var_AR1_bulk,
            "r_AR1": r_AR1_bulk,
        } if sparse_dominated else None,
        "linear_correction_term_for_comparison": {
            "value": 1_800_755.50,
            "source": "synthetic Test 3 (commit e4a9036)",
        },
        "AR1_to_linear_ratio": (Var_AR1_term / 1_800_755.50)
            if Var_AR1_term is not None else None,
    }
    (OUT_DIR / "variance_prediction.json").write_text(json.dumps(var_pred_out, indent=2))

    # ── Sensitivity: minimum run length ────────────────────────
    log.info("Sensitivity — minimum run length grid (3, 5, 10)...")
    sens_min_run = {}
    for L in MIN_RUN_LENGTH_SENSITIVITIES:
        rho_L = {}
        n_runs_total = 0
        n_windows_total = 0
        for i, c in enumerate(surviving):
            sums, n_w = per_traj_sums(trajectories, c, 1, L)
            point = pearson_from_sums(sums.sum(axis=0))
            n_pairs = float(sums[:, 0].sum())
            rho_L[c] = {"rho_lag1": point, "n_pairs": int(n_pairs),
                        "effective_n_windows": int(n_w)}
            n_runs_total += int(n_pairs)  # not exactly runs but pairs
            n_windows_total += int(n_w)
        sens_min_run[L] = {
            "min_run_length": L,
            "per_cell": {str(c): rho_L[c] for c in surviving},
            "total_pairs_lag1": n_runs_total,
            "total_qualifying_windows": n_windows_total,
        }
        log.info(f"  L ≥ {L}: total qualifying windows = {n_windows_total:,}")

    # ── Sensitivity: heavy-tail trim ───────────────────────────
    log.info("Sensitivity — heavy-tail trim (top 1 % of Φ per cell)...")
    sens_trim = {}
    kurtosis = {}
    for i, c in enumerate(surviving):
        vals = np.array(cell_phi_pool[c], dtype=np.float64)
        if len(vals) >= 4:
            mu = vals.mean()
            sd = vals.std(ddof=1)
            kurt = float(((vals - mu) ** 4).mean() / (sd ** 4) - 3) if sd > 0 else float("nan")
        else:
            kurt = float("nan")
        kurtosis[c] = kurt
        if len(vals) >= 100:
            trim_threshold = float(np.quantile(vals, HEAVY_TAIL_TRIM_PCT))
        else:
            trim_threshold = float("inf")
        sums = trimmed_sums(trajectories, c, 1, MIN_RUN_LENGTH_PRIMARY, trim_threshold)
        point = pearson_from_sums(sums.sum(axis=0))
        n_pairs = int(sums[:, 0].sum())
        sens_trim[c] = {
            "kurtosis_excess": kurt,
            "trim_threshold": trim_threshold,
            "rho_lag1_trimmed": point,
            "n_pairs_after_trim": n_pairs,
            "rho_lag1_full": rho_results[c][1]["point"],
            "delta_rho": float(point - rho_results[c][1]["point"])
                if np.isfinite(point) and np.isfinite(rho_results[c][1]["point"]) else None,
        }
        log.info(f"  cell {c:>2d}: kurt_ex={kurt:.2f}  ρ_full={rho_results[c][1]['point']:.4f}  "
                 f"ρ_trim={point:.4f}")

    sensitivity_out = {
        "min_run_length_grid": {str(L): sens_min_run[L] for L in MIN_RUN_LENGTH_SENSITIVITIES},
        "heavy_tail_trim_pct": HEAVY_TAIL_TRIM_PCT,
        "per_cell_trim_sensitivity": {str(c): sens_trim[c] for c in surviving},
        "per_cell_kurtosis_excess": {str(c): kurtosis[c] for c in surviving},
    }
    (OUT_DIR / "sensitivity.json").write_text(json.dumps(sensitivity_out, indent=2))

    # ── Summary ────────────────────────────────────────────────
    write_summary({
        "trajectories": n_traj,
        "surviving": surviving,
        "E_visits": E_visits.tolist(),
        "sigma2": sigma2.tolist(),
        "sigma2_n_windows": sigma2_n_windows.tolist(),
        "run_id": run_id,
        "rho_results": rho_results,
        "stat_results": stat_results,
        "rho_point_lag1": rho_point_lag1.tolist(),
        "var_pred": var_pred_out,
        "sensitivity": sensitivity_out,
    })
    log.info("=" * 70)
    log.info("Autocorrelation characterization complete.")


def write_summary(d: dict):
    surviving = d["surviving"]
    L = []
    L.append("# Within-Cell Autocorrelation Characterization (catalog-scale Φ)\n\n")
    L.append("Pre-registered characterization on the catalog-scale per-window "
             "Φ data (commit `fdd1ba3`). Determines whether AR(1) within-cell "
             "autocorrelation, applied to the variance correction term, "
             "closes the gap to catalog-scale Var_emp = "
             f"{d['var_pred']['catalog_Var_emp']:,.0f}.\n\n")
    L.append("Decision rule (pre-registered):\n\n")
    L.append("- **FULL**: r_AR1 ∈ [0.5, 1.5] AND 95 % CI in same band → "
             "AR(1) closed-form prediction is feasible (Fix A extended).\n")
    L.append("- **PARTIAL**: r_AR1 ∈ [0.2, 0.5] ∪ [1.5, 4.0] AND CI in same "
             "band → partial recovery; need additional terms or pivot.\n")
    L.append("- **OUTSIDE**: outside [0.2, 4.0] → AR(1) does not capture; "
             "pivot to bootstrap/simulation.\n")
    L.append("- **INCONCLUSIVE**: CI spans two bands.\n\n")

    vp = d["var_pred"]
    L.append("## Headline result\n\n")
    L.append("| quantity | value |\n|---|---:|\n")
    L.append(f"| Kemeny baseline (catalog π)            | {vp['Kemeny_baseline']:,.2f} |\n")
    L.append(f"| Var_AR1_term = Σ_c E[v]·σ²·(1+ρ)/(1−ρ) | {vp['Var_AR1_term']:,.2f} |\n")
    L.append(f"| Var_pred,full_AR1 = sum                | {vp['Var_pred_full_AR1']:,.2f} |\n")
    L.append(f"| catalog Var_emp                        | {vp['catalog_Var_emp']:,.2f} |\n")
    L.append(f"| **r_AR1**                              | **{vp['r_AR1_point']:.6f}** |\n")
    L.append(f"| **95 % bootstrap CI on r_AR1**         | **[{vp['r_AR1_CI95'][0]:.6f}, {vp['r_AR1_CI95'][1]:.6f}]** |\n")
    L.append(f"| linear correction term (Test 3)        | {vp['linear_correction_term_for_comparison']['value']:,.2f} |\n")
    L.append(f"| AR(1)/linear ratio                     | {vp['AR1_to_linear_ratio']:.2f}× |\n")
    L.append(f"| sparse-cell share of Var_AR1           | {vp['sparse_cell_share_of_AR1_term']*100:.2f} % |\n")
    L.append(f"| sparse-cell dominated?                 | {vp['sparse_cell_dominated']} |\n\n")

    L.append("## Decision\n\n")
    L.append(f"Point r_AR1 = **{vp['r_AR1_point']:.6f}** → band **{vp['decision_band_point']}**.\n\n")
    L.append(f"95 % CI: [{vp['r_AR1_CI95'][0]:.6f}, {vp['r_AR1_CI95'][1]:.6f}] → "
             f"low band {vp['decision_band_ci_lo']}, high band {vp['decision_band_ci_hi']}.\n\n")
    L.append(f"**Decision: {vp['decision']}.**\n\n")

    decision = vp["decision"]
    if decision == "FULL":
        L.append("AR(1) within-cell autocorrelation, applied via the "
                 "(1+ρ)/(1−ρ) variance-of-sum factor, recovers the catalog-"
                 "scale variance gap. Closed-form variance prediction is "
                 "feasible with this extension. Path forward: incorporate "
                 "AR(1) correction into the canonical variance machinery "
                 "(Fix A extended).\n\n")
    elif decision == "PARTIAL":
        L.append("AR(1) correction is partially explanatory but insufficient. "
                 "Path forward: combine with cross-cell correlation modeling "
                 "or pivot to bootstrap/simulation for variance estimation "
                 "(Fix B).\n\n")
    elif decision == "OUTSIDE":
        L.append("AR(1) correction does not capture the variance source. "
                 "Path forward: pivot to bootstrap/simulation; abandon "
                 "closed-form variance prediction.\n\n")
    else:  # INCONCLUSIVE
        L.append("Inconclusive at current data: 95 % CI on r_AR1 spans "
                 "two decision bands. The point estimate falls in band "
                 f"{vp['decision_band_point']} but the CI cannot exclude "
                 "the adjacent band. Further data or a tighter test is "
                 "required before committing to a path.\n\n")

    if vp["sparse_cell_dominated"]:
        bulk = vp["bulk_only"]
        L.append("## Sparse-cell artifact triggered\n\n")
        L.append(f"More than {int(vp['sparse_cell_share_of_AR1_term']*100)} % "
                 f"of Var_AR1_term comes from cells with effective sample "
                 f"size < {SPARSE_CELL_N_THRESHOLD}. The decision rule is "
                 f"sparse-cell-dominated and does not directly apply.\n\n")
        L.append(f"Bulk-only re-run (cells with effective n ≥ "
                 f"{SPARSE_CELL_N_THRESHOLD}):\n\n")
        L.append(f"- Var_AR1_term (bulk only): {bulk['Var_AR1_term']:,.2f}\n")
        L.append(f"- r_AR1 (bulk only):        {bulk['r_AR1']:.6f}\n\n")

    L.append("## Per-cell ρ_c^(1) and contribution to Var_AR1\n\n")
    L.append("| cell | n qual. windows | σ²_c | ρ^(1) | 95 % CI | "
             "(1+ρ)/(1−ρ) | E[visits] | contribution | share |\n")
    L.append("|---:|---:|---:|---:|---|---:|---:|---:|---:|\n")
    for r in vp["per_cell_contribution"]:
        c = r["cell"]
        rho_ci = d["rho_results"][c][1]
        L.append(f"| {c} | {r['n_qualifying_run_windows']:,} | "
                 f"{r['sigma2_c']:.4f} | {r['rho_lag1']:.4f} | "
                 f"[{rho_ci['ci_lo']:.4f}, {rho_ci['ci_hi']:.4f}] | "
                 f"{r['factor_AR1']:.2f} | {r['E_visits']:.4f} | "
                 f"{r['contribution']:,.2f} | "
                 f"{r['share_of_AR1_term']*100:.2f} % |\n")
    L.append("\n")

    L.append("## ρ^(k) for k = 1, 2, 3\n\n")
    L.append("| cell | ρ^(1) | ρ^(2) | ρ^(3) | n_pairs (lag 1) |\n|---:|---:|---:|---:|---:|\n")
    for c in surviving:
        rho1 = d["rho_results"][c][1]; rho2 = d["rho_results"][c][2]; rho3 = d["rho_results"][c][3]
        L.append(f"| {c} | {rho1['point']:.4f} | {rho2['point']:.4f} | "
                 f"{rho3['point']:.4f} | {rho1['n_pairs']:,} |\n")
    L.append("\n")

    L.append("## Stationarity (early/middle/late thirds, lag 1)\n\n")
    L.append("| cell | ρ_early | ρ_middle | ρ_late | non-stationary? | reason |\n|---:|---:|---:|---:|---|---|\n")
    for c in surviving:
        s = d["stat_results"][c]
        L.append(f"| {c} | {s['early']['rho_lag1']:.4f} | "
                 f"{s['middle']['rho_lag1']:.4f} | {s['late']['rho_lag1']:.4f} | "
                 f"{s['non_stationary_flag']} | {s['reason']} |\n")
    L.append("\n")

    sens = d["sensitivity"]
    L.append("## Sensitivity — minimum run length grid\n\n")
    L.append("Pre-registered grid: 3 (primary), 5, 10. Stability across "
             "min-run-length suggests the AR(1) estimate is robust to "
             "short-run noise.\n\n")
    L.append("| cell | ρ^(1) at L≥3 | ρ^(1) at L≥5 | ρ^(1) at L≥10 |\n|---:|---:|---:|---:|\n")
    for c in surviving:
        v3 = sens["min_run_length_grid"]["3"]["per_cell"][str(c)]
        v5 = sens["min_run_length_grid"]["5"]["per_cell"][str(c)]
        v10 = sens["min_run_length_grid"]["10"]["per_cell"][str(c)]
        def _f(x):
            return f"{x:.4f}" if x is not None and np.isfinite(x) else "—"
        L.append(f"| {c} | {_f(v3['rho_lag1'])} | {_f(v5['rho_lag1'])} | "
                 f"{_f(v10['rho_lag1'])} |\n")
    L.append("\n")

    L.append("## Sensitivity — heavy-tail trim (top 1 % of Φ per cell)\n\n")
    L.append("| cell | excess kurt | ρ_full | ρ_trimmed | Δρ |\n|---:|---:|---:|---:|---:|\n")
    for c in surviving:
        s = sens["per_cell_trim_sensitivity"][str(c)]
        kurt = s["kurtosis_excess"]
        rho_full = s["rho_lag1_full"]
        rho_tr = s["rho_lag1_trimmed"]
        dlt = s["delta_rho"]
        def _f(x):
            return f"{x:.4f}" if x is not None and np.isfinite(x) else "—"
        L.append(f"| {c} | {kurt:.2f} | {_f(rho_full)} | {_f(rho_tr)} | "
                 f"{_f(dlt)} |\n")
    L.append("\n")

    L.append("## Caveats (scientific-rigor)\n\n")
    L.append("- AR(1) variance-of-sum formula `(1+ρ)/(1−ρ)` is asymptotic in "
             "n → ∞ visits per cell. For cells with small E[visits], "
             "the finite-sum correction `(1 + ρ − 2ρ(1−ρⁿ)/(n(1−ρ)))` "
             "differs; not used here.\n")
    L.append("- The bootstrap propagates trajectory-level uncertainty "
             "through σ²_c, ρ_c^(1), and the empirical π simultaneously. CI "
             "reflects all three sources of variance.\n")
    L.append("- Per-cell ρ for cells with very small effective sample size "
             "(< 1,000 qualifying-run windows) is unstable; sparse-cell "
             "rule documents whether such cells dominate Var_AR1_term.\n")
    L.append("- The (1+ρ)/(1−ρ) factor diverges as ρ → 1; clipped to "
             "0.999 here to avoid numerical overflow. Cells with ρ very "
             "close to 1 may have inflated contributions that the AR(1) "
             "model struggles to represent quantitatively.\n")
    L.append("- σ²_c is the all-windows variance, matching the linear "
             "correction term in synthetic Test 3 (commit e4a9036). The "
             "all-windows pooling does not condition on within-run "
             "membership; under stationarity this is equivalent to the "
             "marginal AR(1) variance.\n\n")

    L.append("## Companion artifacts\n\n")
    L.append("- `run_identification.json`  — per-cell qualifying-run counts\n")
    L.append("- `autocorrelation.json`     — ρ_c^(k) for k=1,2,3 with bootstrap CIs\n")
    L.append("- `stationarity.json`        — early/middle/late thirds analysis\n")
    L.append("- `variance_prediction.json` — Var_AR1_term, r_AR1, decision, sparse-cell rule\n")
    L.append("- `sensitivity.json`         — min run length grid, heavy-tail trim\n")

    (OUT_DIR / "summary.md").write_text("".join(L))


if __name__ == "__main__":
    main()
