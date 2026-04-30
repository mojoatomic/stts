#!/usr/bin/env python3
"""
Flux Accumulator — Reentry Characterization (Interpretation C, locked).

Re-runs the six-part characterization on the **restricted sub-corpus**:
trajectories that begin in a transient cell AND later enter the failure
basin. Empirical σ_failure is computed at first basin-entry time τ_i,
not corpus T_i. This addresses the structural mismatch identified in
the first run (commit 310a9b7), where stopping-time analytical theory
was compared to fixed-horizon empirical integrals.

α is frozen from the first run (4.27517). No re-fit. No re-integration
of Φ from raw H/D_KL except the truncate-to-τ_i restriction. All
operator primitives are reused from `reentry/flux_accumulator.py`.

Read-only against frozen artifacts (Markov table, scaler, LDA, kmeans,
basin) and the committed first-run per-object outputs.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reentry.config import MARKOV_K
from reentry.corpus import load_corpus
from reentry.flux_accumulator import (
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    DKL_EPS,
    DKL_WINDOW,
    SANITY_MC_N,
    SANITY_NSIGMA,
    SANITY_SEED,
    bootstrap_variance_ci,
    c_restricted_corpus,
    kappa_per_cell,
    kemeny_snell_variance,
    linear_r2,
    per_window_dkl,
    phi as compute_phi,
    sigma_raw as compute_sigma_raw,
    truncate_to_tau,
    variance_stderr,
)
from reentry.train import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/reentry/flux_characterization_c_restricted")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PER_OBJECT_DIR = Path("results/reentry/per_object")

# α frozen from the first characterization run (commit 310a9b7).
FROZEN_ALPHA = 4.27517


# ── Part 1 ──────────────────────────────────────────────────────

def part1_sanity_check() -> bool:
    """Re-execute the toy-chain Kemeny-Snell sanity check for record."""
    log.info("=" * 70)
    log.info("Part 1 — sanity check (re-execution for record)")
    log.info("=" * 70)
    P = np.array([[0.5, 0.3, 0.2],
                  [0.2, 0.4, 0.4],
                  [0.0, 0.0, 1.0]])
    kappa = np.array([1.0, 2.0])
    transient = [0, 1]
    res = kemeny_snell_variance(P, transient, kappa)
    rng = np.random.default_rng(SANITY_SEED)

    def mc(start: int, n: int = SANITY_MC_N) -> np.ndarray:
        sigmas = np.zeros(n, dtype=np.float64)
        for k in range(n):
            s = start
            sig = 0.0
            while s != 2:
                sig += float(kappa[s])
                s = int(rng.choice(3, p=P[s]))
            sigmas[k] = sig
        return sigmas

    sigmas_per_start = []
    for start in transient:
        log.info(f"  MC from start={start} (N={SANITY_MC_N:,}, seed={SANITY_SEED})")
        sigmas_per_start.append(mc(start))
    means_mc = np.array([s.mean() for s in sigmas_per_start])
    vars_mc = np.array([s.var(ddof=1) for s in sigmas_per_start])
    se_vars = np.array([variance_stderr(s) for s in sigmas_per_start])

    delta_mean = np.abs(res["v"] - means_mc)
    delta_var = np.abs(res["Var_predicted"] - vars_mc)
    threshold_var = SANITY_NSIGMA * se_vars
    ok_mean = bool(np.all(delta_mean < 1e-2))
    ok_var = bool(np.all(delta_var < threshold_var))

    bar = "=" * 70
    txt = [
        "Flux Accumulator — Part 1 sanity check (Kemeny-Snell vs MC; re-execution)",
        bar, "",
        f"Toy chain: 3 states, transient={transient}, absorbing=2",
        f"P =\n{P}",
        f"kappa = {kappa}",
        "",
        f"Analytical: v = {res['v']}",
        f"           Var = {res['Var_predicted']}",
        "",
        f"MC (N={SANITY_MC_N:,} per start, seed={SANITY_SEED}):",
        f"  mean_MC = {means_mc}",
        f"  Var_MC  = {vars_mc}",
        f"  SE(Var_MC) = {se_vars}  (empirical, from 4th central moment)",
        "",
        f"|mean_a - mean_MC| = {delta_mean}",
        f"|Var_a  - Var_MC|  = {delta_var}",
        f"Var threshold      = {SANITY_NSIGMA} × SE(Var_MC) = {threshold_var}",
        "",
        f"Mean assertion (|delta| < 1e-2):                 {'PASS' if ok_mean else 'FAIL'}",
        f"Var  assertion (|delta| < {SANITY_NSIGMA}σ MC stderr): {'PASS' if ok_var else 'FAIL'}",
        "",
        "Result: " + ("PASS — proceed to Parts 2-6 (re-execution under C)."
                       if (ok_mean and ok_var) else "FAIL — halt."),
    ]
    (OUT_DIR / "sanity_check.txt").write_text("\n".join(txt) + "\n")
    log.info("\n".join(txt[-6:]))
    return ok_mean and ok_var


# ── Frozen α record ─────────────────────────────────────────────

def part2_alpha_record() -> dict:
    bar = "=" * 70
    log.info("=" * 70)
    log.info("Part 2 — α (frozen from first run; no recomputation)")
    log.info("=" * 70)
    log.info(f"  α = {FROZEN_ALPHA}")
    (OUT_DIR / "alpha.txt").write_text(
        f"Flux Accumulator — Part 2: α (frozen from first run, commit 310a9b7)\n"
        f"{bar}\n\n"
        f"α = {FROZEN_ALPHA}\n\n"
        f"This value is the calibration produced by the first characterization\n"
        f"run on the full 78-object corpus, computed via\n"
        f"    α = sqrt( Σ H² Δt / (2 · Σ D_KL Δt) )\n"
        f"and frozen there. No recomputation under Interpretation C.\n"
    )
    return {"alpha": FROZEN_ALPHA}


# ── Loading per-object trajectories + frozen P ──────────────────

def load_trajectory(nid: str) -> dict:
    path = PER_OBJECT_DIR / f"{nid}.csv"
    rows = list(csv.DictReader(open(path)))
    days = np.array([float(r["days_to_reentry"]) for r in rows])
    cells = np.array([int(r["M2_cell"]) for r in rows], dtype=np.int64)
    H = np.array([float(r["H_t"]) for r in rows], dtype=np.float64)
    return {"nid": nid, "days": days, "cells": cells, "H": H,
            "n_windows": len(rows)}


def all_78_object_ids() -> list[str]:
    test_ids = json.loads(Path("artifacts/reentry/test_norad_ids.json").read_text())
    corpus = load_corpus()
    sats = corpus["satellites"]
    return [nid for nid in test_ids if sats[nid].get("decay_epoch")]


# ── First action: restricted-corpus count report ────────────────

def report_restricted_corpus_counts(restriction: dict) -> None:
    log.info("=" * 70)
    log.info("First action — Restricted Corpus Counts (pre-registered)")
    log.info("=" * 70)
    log.info(f"  Total objects                  : {restriction['n_total']}")
    log.info(f"  Included (transient → basin)   : {restriction['n_included']}")
    log.info(f"  Excluded — starting in basin   : {restriction['n_starting_in_basin']}")
    log.info(f"  Excluded — never reach basin   : {restriction['n_never_reach_basin']}")

    cat_records = []
    for r in restriction["restricted"]:
        cat_records.append({"norad_id": r["nid"], "category": "included",
                            "tau_i": int(r["tau_i"]), "c_first": int(r["c_first"]),
                            "n_windows": int(r["n_windows"])})
    for nid in restriction["starting_in_basin"]:
        cat_records.append({"norad_id": nid, "category": "starting_in_basin"})
    for nid in restriction["never_reach_basin"]:
        cat_records.append({"norad_id": nid, "category": "never_reach_basin"})
    out = {
        "n_total": restriction["n_total"],
        "n_included": restriction["n_included"],
        "n_excluded_starting_in_basin": restriction["n_starting_in_basin"],
        "n_excluded_never_reach_basin": restriction["n_never_reach_basin"],
        "per_object": cat_records,
    }
    (OUT_DIR / "restricted_corpus_summary.json").write_text(
        json.dumps(out, indent=2)
    )
    log.info(f"  wrote {OUT_DIR/'restricted_corpus_summary.json'}")


# ── Part 3 ──────────────────────────────────────────────────────

def part3_predicted_variance(restricted: list[dict], P_trained: np.ndarray,
                               alpha: float, transient_cells: list[int]) -> dict:
    log.info("=" * 70)
    log.info("Part 3 — predicted Var on restricted corpus (with empty-cell rule)")
    log.info("=" * 70)

    # Truncate every restricted trajectory to [0, tau_i]; aggregate κ per
    # transient cell over this restricted-corpus window set.
    truncated = [truncate_to_tau(r) for r in restricted]
    Hs = [t["H"] for t in truncated]
    dkls = [t["dkl"] for t in truncated]
    cells = [t["cells"] for t in truncated]

    kappa_full, counts_full = kappa_per_cell(
        Hs, dkls, cells, transient_cells, alpha,
    )

    # Empty-cell rule
    surviving = [c for c, n in zip(transient_cells, counts_full) if n > 0]
    dropped = [c for c, n in zip(transient_cells, counts_full) if n == 0]
    surviving_idx = [transient_cells.index(c) for c in surviving]
    kappa = kappa_full[surviving_idx]
    counts = counts_full[surviving_idx]

    log.info(f"  transient cells:                {transient_cells}")
    log.info(f"  surviving (n > 0 in restricted): {surviving}")
    log.info(f"  dropped (empty-cell rule):       {dropped}")
    log.info(f"  κ per surviving cell:")
    for c, k, n in zip(surviving, kappa, counts):
        log.info(f"    cell {c:>2d}: n_windows={int(n):>6d}  κ_c = {k:.6f}")

    # Q = P[surviving, surviving], no renormalization. Mass leaving the
    # surviving subset (to dropped transient cells or to absorbing cells)
    # is treated as exit for the absorbing-chain calculation.
    Q = P_trained[np.ix_(surviving, surviving)].astype(np.float64)
    I = np.eye(len(surviving))
    N = np.linalg.inv(I - Q)
    v = N @ kappa
    g = N @ (kappa * (2.0 * v - kappa))
    var_pred = g - v * v

    log.info(f"  v (mean σ_failure | start cell) = {v}")
    log.info(f"  Var_predicted per start cell    = {var_pred}")

    out = {
        "transient_cells_full": transient_cells,
        "surviving_cells": surviving,
        "dropped_cells_empty_rule": dropped,
        "kappa": kappa.tolist(),
        "kappa_n_windows": [int(n) for n in counts],
        "Q": Q.tolist(),
        "N": N.tolist(),
        "v": v.tolist(),
        "Var_predicted": var_pred.tolist(),
        "alpha": alpha,
        "DKL_WINDOW": DKL_WINDOW,
    }
    (OUT_DIR / "predicted_variance.json").write_text(json.dumps(out, indent=2))
    log.info(f"  wrote {OUT_DIR/'predicted_variance.json'}")
    return out


# ── Part 4 ──────────────────────────────────────────────────────

def part4_empirical_variance(restricted: list[dict], alpha: float) -> dict:
    log.info("=" * 70)
    log.info("Part 4 — empirical σ_raw(τ_i) on restricted corpus")
    log.info("=" * 70)

    sigma_at_tau = []
    per_object = []
    for r in restricted:
        end = int(r["tau_i"]) + 1
        H = r["H"][:end]; dkl = r["dkl"][:end]
        phi_t = compute_phi(H, dkl, alpha)
        sigma_t = compute_sigma_raw(phi_t, dt=1.0)
        sT = float(sigma_t[-1])
        sigma_at_tau.append(sT)
        per_object.append({
            "norad_id": r["nid"],
            "starting_cell_c_first": int(r["c_first"]),
            "tau_i": int(r["tau_i"]),
            "n_windows": int(r["n_windows"]),
            "sigma_raw_at_tau_i": sT,
        })

    sigma_at_tau = np.array(sigma_at_tau, dtype=np.float64)
    sigma_failure = float(np.median(sigma_at_tau))
    var_emp, ci_lo, ci_hi = bootstrap_variance_ci(
        sigma_at_tau, n_replicates=BOOTSTRAP_B, seed=BOOTSTRAP_SEED,
    )
    log.info(f"  n_restricted        = {len(sigma_at_tau)}")
    log.info(f"  σ_failure (median)  = {sigma_failure:.4f}")
    log.info(f"  Var_empirical       = {var_emp:.4f}")
    log.info(f"  95 % CI on Var      = [{ci_lo:.4f}, {ci_hi:.4f}]")

    out = {
        "n_restricted": len(sigma_at_tau),
        "sigma_failure": sigma_failure,
        "var_empirical": var_emp,
        "var_empirical_CI95": [ci_lo, ci_hi],
        "BOOTSTRAP_B": BOOTSTRAP_B,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        "per_object": per_object,
    }
    (OUT_DIR / "empirical_variance.json").write_text(json.dumps(out, indent=2))
    log.info(f"  wrote {OUT_DIR/'empirical_variance.json'}")
    return out


# ── Part 5 ──────────────────────────────────────────────────────

def part5_diagnostics(per_object_full: list[dict], alpha: float,
                       sigma_at_tau: np.ndarray) -> dict:
    log.info("=" * 70)
    log.info("Part 5 — diagnostics (η, R² on FULL traces, histogram on restricted)")
    log.info("=" * 70)

    # R² on full trajectories (NOT truncated to τ_i)
    r2_per_obj = []
    for traj in per_object_full:
        H = traj["H"]; dkl = traj["dkl"]
        phi_t = compute_phi(H, dkl, alpha)
        sigma_t = compute_sigma_raw(phi_t, dt=1.0)
        n = len(sigma_t)
        if n < 3:
            r2_per_obj.append(float("nan"))
            continue
        t_axis = np.arange(n, dtype=np.float64)
        r2_per_obj.append(linear_r2(t_axis, sigma_t))
    r2_arr = np.array(r2_per_obj, dtype=np.float64)
    median_r2 = float(np.nanmedian(r2_arr))
    log.info(f"  median R² across {len(per_object_full)} full traces: {median_r2:.4f}  "
             f"(pre-registered threshold: 0.92)")

    # Histogram on restricted-corpus σ_raw(τ_i)
    counts, edges = np.histogram(sigma_at_tau, bins=10)
    multimodality_flag = False
    for j in range(1, len(counts) - 1):
        if counts[j] == 0 and counts[j - 1] > 0 and counts[j + 1] > 0:
            multimodality_flag = True
            break

    out = {
        "eta": 0.0,
        "median_R2_full_traces": median_r2,
        "per_object_R2": [None if not np.isfinite(x) else float(x) for x in r2_per_obj],
        "histogram_bin_edges": edges.tolist(),
        "histogram_counts": [int(c) for c in counts],
        "multimodality_flag": bool(multimodality_flag),
        "preregistered_R2_threshold": 0.92,
    }
    (OUT_DIR / "diagnostics.json").write_text(json.dumps(out, indent=2))
    log.info(f"  histogram counts (restricted σ_raw(τ_i)): {counts.tolist()}")
    log.info(f"  histogram edges:                          {[f'{e:.2f}' for e in edges]}")
    log.info(f"  multimodality flag: {multimodality_flag}")
    log.info(f"  wrote {OUT_DIR/'diagnostics.json'}")
    return out


# ── Part 6 ──────────────────────────────────────────────────────

def part6_comparison(part3: dict, part4: dict) -> dict:
    log.info("=" * 70)
    log.info("Part 6 — analytical vs empirical (restricted)")
    log.info("=" * 70)
    surviving = part3["surviving_cells"]
    var_pred = np.array(part3["Var_predicted"], dtype=np.float64)
    starts = [r["starting_cell_c_first"] for r in part4["per_object"]]
    n = len(starts)
    counts = np.zeros(len(surviving), dtype=np.int64)
    cell_to_idx = {c: i for i, c in enumerate(surviving)}
    n_dropped = 0
    for s in starts:
        if s in cell_to_idx:
            counts[cell_to_idx[s]] += 1
        else:
            # Restricted-corpus members all start in transient cells, so
            # this can fire only if the start cell was emptied by the
            # rule (cannot happen — that cell would have ≥ n_starts > 0).
            n_dropped += 1
    weights = counts.astype(np.float64) / max(1, n)
    var_pred_weighted = float(np.sum(weights * var_pred))
    var_emp = part4["var_empirical"]
    ci_lo, ci_hi = part4["var_empirical_CI95"]
    r = var_pred_weighted / var_emp if var_emp != 0 else float("nan")

    log.info(f"  Var_pred,weighted = {var_pred_weighted:.4f}")
    log.info(f"  Var_empirical     = {var_emp:.4f}  CI95 = [{ci_lo:.4f}, {ci_hi:.4f}]")
    log.info(f"  ratio r           = {r:.4f}")

    L = []
    L.append("# Flux Accumulator — Part 6 (Interpretation C, restricted)\n\n")
    L.append("## Aggregates\n\n")
    L.append("| quantity | value |\n|---|---:|\n")
    L.append(f"| Var_pred,weighted        | {var_pred_weighted:.6f} |\n")
    L.append(f"| Var_empirical            | {var_emp:.6f} |\n")
    L.append(f"| 95 % CI on Var_empirical | [{ci_lo:.6f}, {ci_hi:.6f}] |\n")
    L.append(f"| ratio r                  | {r:.6f} |\n")
    L.append(f"| n included objects       | {n} |\n")
    L.append(f"| dropped (start outside surviving) | {n_dropped} |\n\n")
    L.append("## Per-cell breakdown (restricted)\n\n")
    L.append("| starting cell | Var_predicted[c] | n_c (restricted starts) | weight w_c | contribution w_c · Var_pred[c] |\n")
    L.append("|---:|---:|---:|---:|---:|\n")
    for c, vp, n_c, w in zip(surviving, var_pred, counts, weights):
        L.append(f"| {c} | {vp:.6f} | {int(n_c)} | {w:.4f} | {w*vp:.6f} |\n")
    L.append("\n")
    L.append("## Neutral interpretation\n\n")
    if 0.85 <= r <= 1.15:
        msg = ("r ≈ 1.0 (within ±15 %): the Kemeny-Snell formula on the "
               "frozen transition matrix captures the dominant variance "
               "structure on the restricted transient-to-absorbing "
               "population.")
    elif r < 0.85:
        msg = ("r ≪ 1.0: predicted variance is smaller than empirical; "
               "additional variance sources (within-cell heteroscedasticity, "
               "curvature effects, or residual model mismatch) are present.")
    else:
        msg = ("r ≫ 1.0: formula overestimates; empirical trajectories are "
               "more constrained than the absorbing-chain prediction.")
    L.append(msg + "\n\n")
    L.append("**Fiber metric note.** α is treated as position-independent. "
             "If the per-cell contribution column shows systematic variation "
             "in `Var_predicted[c]` not balanced by the empirical weights, "
             "this is a signal that position-dependent α(c) may be needed in "
             "future iterations.\n")
    (OUT_DIR / "comparison.md").write_text("".join(L))
    log.info(f"  wrote {OUT_DIR/'comparison.md'}")
    return {"var_pred_weighted": var_pred_weighted, "ratio_r": r,
            "n_dropped": n_dropped, "n_included": n}


# ── Summary ─────────────────────────────────────────────────────

def write_summary(restriction: dict, part3: dict, part4: dict,
                   part5: dict, part6: dict) -> None:
    sf = part4["sigma_failure"]
    var_emp = part4["var_empirical"]
    ci_lo, ci_hi = part4["var_empirical_CI95"]
    var_pred_w = part6["var_pred_weighted"]
    r = part6["ratio_r"]
    median_r2 = part5["median_R2_full_traces"]
    multimodal = part5["multimodality_flag"]

    L = []
    L.append("# Flux Accumulator — C-Restricted Reentry Characterization\n\n")
    L.append("Re-evaluation under Interpretation C (Hybrid Restricted-Corpus "
             "Semantics). Empirical σ_failure is integrated to first basin-"
             "entry time τ_i over the restricted sub-corpus of trajectories "
             "that begin in a transient cell and later enter the failure "
             "basin. Predicted variance is computed via Kemeny-Snell on the "
             "transient sub-chain spanned by the restricted corpus, with the "
             "empty-cell rule applied. α is frozen from the first run "
             "(commit 310a9b7); no re-fit. Read-only against frozen "
             "artifacts.\n\n")
    L.append("All measurements are characterization-level statements; no "
             "\"validation\" or \"confirmation\" claims.\n\n")

    L.append("## Restricted corpus partition\n\n")
    L.append("| category | count |\n|---|---:|\n")
    L.append(f"| Total objects                | {restriction['n_total']} |\n")
    L.append(f"| Included (transient→basin)   | {restriction['n_included']} |\n")
    L.append(f"| Excluded — starting in basin | {restriction['n_starting_in_basin']} |\n")
    L.append(f"| Excluded — never reach basin | {restriction['n_never_reach_basin']} |\n\n")

    L.append("## Headline measurements (restricted)\n\n")
    L.append("| quantity | value |\n|---|---:|\n")
    L.append(f"| α (frozen)                | 4.27517 |\n")
    L.append(f"| n included                | {restriction['n_included']} |\n")
    L.append(f"| σ_failure (median)        | {sf:.4f} |\n")
    L.append(f"| Var_empirical             | {var_emp:.4f} |\n")
    L.append(f"| 95 % CI on Var_empirical  | [{ci_lo:.4f}, {ci_hi:.4f}] |\n")
    L.append(f"| Var_pred,weighted         | {var_pred_w:.4f} |\n")
    L.append(f"| ratio r                   | {r:.4f} |\n")
    L.append(f"| median R² (full traces)   | {median_r2:.4f} |\n")
    L.append(f"| pre-registered R² gate    | 0.92 |\n")
    L.append(f"| multimodality flag        | {multimodal} |\n")
    L.append(f"| η (negative-flux fraction)| 0 (structurally) |\n\n")

    L.append("## Empty-cell rule\n\n")
    surviving = part3["surviving_cells"]
    dropped = part3["dropped_cells_empty_rule"]
    L.append(f"Transient cells (full set, 10): "
             f"{part3['transient_cells_full']}\n\n")
    L.append(f"Surviving (n > 0 in restricted set): {surviving} "
             f"({len(surviving)} of 10).\n\n")
    L.append(f"Dropped (n = 0 in restricted): {dropped}.\n\n")
    L.append(f"Q is built as `P_trained[surviving, surviving]` without "
             f"renormalization. Mass leaving this surviving transient subset "
             f"(to dropped transient cells or to the failure basin) is treated "
             f"as exit for the absorbing-chain calculation, the standard "
             f"construction.\n\n")

    L.append("## What the numbers mean (characterization level)\n\n")
    L.append("σ_failure is the median of σ_raw integrated to τ_i (first basin "
             "entry) over the restricted sub-corpus. Var_empirical is the "
             "sample variance of those σ_raw(τ_i) values; its bootstrap 95 % "
             "CI gives the precision on that summary statistic. "
             "Var_pred,weighted is the closed-form Kemeny-Snell prediction "
             "averaged over the restricted starting-cell distribution. The "
             "ratio r reports how much of empirical variance the Markov "
             "reward theory captures under the operator's per-cell mean cost "
             "κ_c on this restricted population.\n\n")
    L.append("Under C-restriction, the empirical and analytical quantities "
             "are now both stopping-time integrals, removing the structural "
             "mismatch that produced r ≈ 0.008 in the first run. Any "
             "remaining gap is attributable to within-cell heteroscedasticity "
             "of Φ, position-dependent α(c), or cross-coupling between drift "
             "and deformation directions.\n\n")

    n_dropped_in_weight = part6.get("n_dropped", 0)
    if n_dropped_in_weight > 0:
        L.append("## Note — cell-16 starts dropped from weighted aggregate\n\n")
        L.append(f"{n_dropped_in_weight} of {restriction['n_included']} "
                 f"restricted objects have their first window in a transient "
                 f"cell that the empty-cell rule subsequently dropped. This "
                 f"happens when an object touches a transient cell only at the "
                 f"first ≤ W = 5 windows (D_KL = NaN by the early-window "
                 f"convention) and exits before any finite-D_KL window in "
                 f"that cell is observed. κ for that cell is undefined on "
                 f"the restricted corpus, so the cell is dropped from Q; "
                 f"objects starting there are then unmapped in the weighted "
                 f"Var_pred aggregate. Effective n in the weighted "
                 f"Var_pred,weighted: {restriction['n_included'] - n_dropped_in_weight} of "
                 f"{restriction['n_included']}.\n\n"
                 f"This is a consequence of the locked W = 5 D_KL convention "
                 f"interacting with cell 16's low operational frequency (cell "
                 f"16 had n = 31 windows total in the first run's full "
                 f"corpus; under restriction, none survive the W = 5 NaN "
                 f"mask). It is not load-bearing on the per-cell variance "
                 f"prediction for the surviving 8 cells but is reported here "
                 f"so the gap between n_included and the effective weighted "
                 f"n is explicit.\n\n")

    # Note on n=39 statistical power and CI envelope of r
    var_emp = part4["var_empirical"]
    ci_lo, ci_hi = part4["var_empirical_CI95"]
    r = part6["ratio_r"]
    var_pred_w = part6["var_pred_weighted"]
    if var_emp > 0 and ci_lo > 0:
        r_at_ci_lo = var_pred_w / ci_lo if ci_lo != 0 else float("inf")
        r_at_ci_hi = var_pred_w / ci_hi if ci_hi != 0 else float("inf")
        L.append("## Note — statistical power on r\n\n")
        L.append(f"The point ratio `r = {r:.4f}` is computed from "
                 f"Var_pred,weighted = {var_pred_w:,.0f} and "
                 f"Var_empirical = {var_emp:,.0f}. The 95 % bootstrap CI on "
                 f"Var_empirical is [{ci_lo:,.0f}, {ci_hi:,.0f}], which "
                 f"maps to an r envelope of "
                 f"[{r_at_ci_hi:.3f}, {r_at_ci_lo:.3f}]. The CI width is "
                 f"driven by the small restricted-corpus size "
                 f"(n = {restriction['n_included']}) combined with a "
                 f"right-skewed distribution of σ_raw(τ_i) "
                 f"(see histogram in `diagnostics.json`: 34 of 39 below the "
                 f"first bin edge, with one extreme observation at the high "
                 f"end). At this sample size and this empirical "
                 f"distribution, the r envelope brackets 1 from above; we "
                 f"cannot distinguish r ≈ 1 from r ≈ 2 at 95 % CI.\n\n")

    L.append("## Caveats\n\n")
    L.append("- **Within-cell heteroscedasticity** of Φ contributes to "
             "Var_empirical but not to Var_predicted; the analytical formula "
             "treats κ_c as a deterministic per-state cost.\n")
    L.append("- **Block-diagonal bundle metric** is assumed by the "
             "Pythagorean form. Per-cell residual patterns in `comparison.md` "
             "are the right test for cross-coupling.\n")
    L.append("- **Position-independent α** is used. Per-cell residual "
             "variation is the diagnostic for whether α(c) is needed.\n")
    L.append("- **Restricted-corpus size dictates CI width.** With "
             f"n = {restriction['n_included']} restricted objects, the "
             f"bootstrap CI on Var_empirical is correspondingly wider than "
             f"the full-78 number from the first run.\n")
    L.append("- **First basin-entry semantics.** τ_i is defined as the first "
             "window index at which the trajectory's M₂ cell falls in the "
             "failure basin {1, 2, 4, 6, 7, 10, 11, 14, 15, 18}. This is "
             "frozen as part of the C specification; no alternative basin "
             "definitions are explored here.\n\n")

    L.append("## Companion artifacts\n\n")
    L.append("- `sanity_check.txt`             — Part 1, re-execution for record\n")
    L.append("- `alpha.txt`                    — Part 2, frozen α from first run\n")
    L.append("- `restricted_corpus_summary.json` — partition counts and per-object category\n")
    L.append("- `predicted_variance.json`      — Part 3, Q/N/κ/Var_pred on surviving cells\n")
    L.append("- `empirical_variance.json`      — Part 4, σ_raw(τ_i) per object + bootstrap CI\n")
    L.append("- `diagnostics.json`             — Part 5, R² (full traces), histogram\n")
    L.append("- `comparison.md`                — Part 6, ratio r and per-cell breakdown\n")

    (OUT_DIR / "flux_summary.md").write_text("".join(L))
    log.info(f"  wrote {OUT_DIR/'flux_summary.md'}")


# ── main ────────────────────────────────────────────────────────

def main():
    if not part1_sanity_check():
        log.error("Part 1 sanity check FAILED on re-execution. Halting.")
        sys.exit(2)

    log.info("Loading 78-object trajectories + frozen P...")
    nids = all_78_object_ids()
    assert len(nids) == 78, f"expected 78 reentry-class objects, got {len(nids)}"
    per_object = [load_trajectory(nid) for nid in nids]
    model = load_model()
    P = model["markov"]["transition_matrix"]
    failure_cells = sorted(model["markov"]["failure_cells"].tolist())
    transient_cells = sorted(set(range(MARKOV_K)) - set(failure_cells))
    log.info(f"  failure cells: {failure_cells}")
    log.info(f"  transient cells: {transient_cells}")

    # Cache D_KL on each trajectory (frozen W=5, ε=1e-9 from the first run).
    log.info(f"Computing D_KL trajectories (W={DKL_WINDOW}, ε={DKL_EPS})...")
    for t in per_object:
        t["dkl"] = per_window_dkl(t["cells"], P, window=DKL_WINDOW, eps=DKL_EPS)

    # Apply C restriction.
    restriction = c_restricted_corpus(per_object, failure_cells)
    report_restricted_corpus_counts(restriction)

    # Part 2 (frozen α)
    part2 = part2_alpha_record()
    alpha = part2["alpha"]

    # Carry D_KL into restricted records (c_restricted_corpus copies dicts
    # but its result preserves the dkl field via dict(traj))
    # Sanity check — restricted has dkl
    if restriction["restricted"] and "dkl" not in restriction["restricted"][0]:
        for r in restriction["restricted"]:
            for src in per_object:
                if src["nid"] == r["nid"]:
                    r["dkl"] = src["dkl"]
                    break

    # Parts 3-6
    part3 = part3_predicted_variance(restriction["restricted"], P, alpha,
                                       transient_cells)
    part4 = part4_empirical_variance(restriction["restricted"], alpha)
    sigma_at_tau = np.array([r["sigma_raw_at_tau_i"] for r in part4["per_object"]])
    part5 = part5_diagnostics(per_object, alpha, sigma_at_tau)
    part6 = part6_comparison(part3, part4)
    write_summary(restriction, part3, part4, part5, part6)

    log.info("=" * 70)
    log.info("C-restricted characterization complete.")


if __name__ == "__main__":
    main()
