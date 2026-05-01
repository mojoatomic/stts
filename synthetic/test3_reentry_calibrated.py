#!/usr/bin/env python3
"""
Test 3 — Reentry-calibrated bridge.

Question: If we use the empirically observed within-cell variance from
the catalog-scale per-window Φ data, does the corrected variance
prediction match the catalog-scale Var_emp = 33,396,043?

Design:
  1. From the catalog-scale per-object trajectories
     (results/reentry/catalog_scale_validation/per_object/, 3,346 CSVs),
     reconstruct per-window Φ using the same convention as
     flux_characterization_catalog_c.py:
        D_KL(t)   per-window via per_window_dkl(W=5, ε=1e-9)
        Φ(t)      = sqrt(H(t)² + 2 α² · D_KL(t))   with α = 4.27517
     Restrict to the C-restricted corpus's within-τ_i windows (matches
     how κ_c was aggregated in the catalog-scale run).
  2. For each transient cell c (10 surviving cells in catalog-scale),
     compute σ²_c,empirical = sample variance (ddof=1) of Φ values
     across all (trajectory, window) pairs whose M2 cell is c.
  3. Apply the analytical correction term using the catalog-scale
     N matrix and π = empirical starting-cell distribution:
        Var_correction = Σ_c E_π[visits to c] · σ²_c,empirical
     where E_π[visits to c] = (Σ_s π(s) · N[s,c]).
  4. Var_pred,corrected = Var_pred,weighted + Var_correction
     (= 147,992 + Var_correction)
     We also report Var_pred,total + Var_correction (with the
     across-start term Var_starts(v)) for completeness.
  5. r_corrected = Var_pred,corrected / Var_emp
     where Var_emp = catalog-scale 33,396,043.

Scientific-rigor: σ²_c,empirical is computed on per-visit Φ values, NOT
on σ_failure values. The correction term assumes within-visit
independence; it does not capture within-cell autocorrelation in Φ.
The independence assumption is documented as a caveat in the summary;
if the catalog Φ trajectory has within-cell autocorrelation, the
linear correction will under- or over-estimate accordingly.
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

OUT_DIR = Path("results/synthetic/test3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PER_OBJECT_DIR = Path("results/reentry/catalog_scale_validation/per_object")
CATALOG_PRED_VAR = Path(
    "results/reentry/flux_characterization_catalog_c/predicted_variance.json"
)
CATALOG_EMP_VAR = Path(
    "results/reentry/flux_characterization_catalog_c/empirical_variance.json"
)
FROZEN_ALPHA = 4.27517


def load_catalog_trajectory(path: Path) -> dict:
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    cells = np.array([int(r["M2_cell"]) for r in rows], dtype=np.int64)
    H = np.array([float(r["H_t"]) for r in rows], dtype=np.float64)
    return {"nid": path.stem, "cells": cells, "H": H, "n_windows": len(rows)}


def main():
    log.info("=" * 70)
    log.info("Test 3 — Reentry-calibrated bridge")
    log.info("=" * 70)

    # Load committed catalog-scale numbers for reference
    pv = json.loads(CATALOG_PRED_VAR.read_text())
    ev = json.loads(CATALOG_EMP_VAR.read_text())
    catalog_var_emp = float(ev["var_empirical"])
    catalog_var_pred_weighted = sum(
        # weighted by starting-cell distribution from catalog data
        # but we need to reconstruct that — use restricted corpus starts
        0.0  # placeholder; recompute below with per-object starts
        for _ in pv["surviving_cells"]
    )
    log.info(f"  Committed catalog Var_emp        = {catalog_var_emp:,.2f}")

    # Per-object starting cells from the catalog C run
    starting_cells = []
    for r in ev["per_object"]:
        starting_cells.append(int(r["starting_cell_c_first"]))
    n_objects = len(starting_cells)
    log.info(f"  n restricted catalog objects     = {n_objects}")

    surviving = pv["surviving_cells"]
    N_committed = np.array(pv["N"], dtype=np.float64)  # 10×10 fundamental matrix
    Q_committed = np.array(pv["Q"], dtype=np.float64)
    kappa_committed = np.array(pv["kappa"], dtype=np.float64)
    var_per_cell_committed = np.array(pv["Var_predicted"], dtype=np.float64)
    v_committed = np.array(pv["v"], dtype=np.float64)
    # Empirical starting-cell distribution π over surviving cells
    cell_idx = {c: i for i, c in enumerate(surviving)}
    pi_counts = np.zeros(len(surviving), dtype=np.float64)
    for s in starting_cells:
        if s in cell_idx:
            pi_counts[cell_idx[s]] += 1.0
    pi = pi_counts / pi_counts.sum() if pi_counts.sum() > 0 else None
    log.info(f"  empirical π over surviving cells:  "
             f"{[(surviving[i], int(pi_counts[i])) for i in range(len(surviving))]}")

    # Recompute Var_pred,weighted from the committed per-cell variance and
    # this empirical π (matches the catalog-scale run's r computation).
    var_pred_weighted = float((pi * var_per_cell_committed).sum())
    var_starts_v = float(((v_committed - (pi * v_committed).sum()) ** 2 * pi).sum())
    var_pred_total = var_pred_weighted + var_starts_v
    log.info(f"  Var_pred,weighted (recomputed)   = {var_pred_weighted:,.2f}")
    log.info(f"  Var_starts(v) (across-start)     = {var_starts_v:,.2f}")
    log.info(f"  Var_pred,total (sum)             = {var_pred_total:,.2f}")

    # E_π[visits to c]
    # E_π[visits to c] = Σ_s π(s) · N[s, c]
    E_visits_pi = pi @ N_committed  # (n_surviving,)
    log.info(f"  E_π[visits to c]:")
    for i, c in enumerate(surviving):
        log.info(f"    cell {c:>2d}: E[visits] = {E_visits_pi[i]:.4f}, "
                 f"κ_c = {kappa_committed[i]:.4f}")

    # ── Compute per-cell within-cell σ²_c empirically ──────────
    log.info("Loading 3,346 catalog-scale per-object trajectories + "
             "computing per-window Φ...")
    model = load_model()
    P = model["markov"]["transition_matrix"]
    failure_cells = sorted(model["markov"]["failure_cells"].tolist())

    csv_paths = sorted(PER_OBJECT_DIR.glob("*.csv"))
    t0 = time.time()
    per_object = []
    for p in csv_paths:
        traj = load_catalog_trajectory(p)
        if traj is None or traj["n_windows"] == 0:
            continue
        per_object.append(traj)
    log.info(f"  loaded {len(per_object)} trajectories in {time.time()-t0:.1f}s")

    log.info("Computing D_KL trajectories (W=5, ε=1e-9)...")
    t0 = time.time()
    for t in per_object:
        t["dkl"] = per_window_dkl(t["cells"], P, window=DKL_WINDOW, eps=DKL_EPS)
    log.info(f"  done in {time.time()-t0:.1f}s")

    # Apply C-restriction
    restriction = c_restricted_corpus(per_object, failure_cells)
    log.info(f"  C-restricted: {restriction['n_included']} included "
             f"(matches catalog C run: {n_objects})")
    if restriction["n_included"] != n_objects:
        log.warning(f"  Mismatch with catalog C run ({n_objects}); "
                    "proceeding with current restriction set")

    # Aggregate per-window Φ values per cell across all restricted
    # trajectories (windows 0..τ_i inclusive, matching κ aggregation).
    log.info("Aggregating per-cell Φ values...")
    per_cell_phi: dict[int, list[float]] = {c: [] for c in surviving}
    nid_to_traj = {t["nid"]: t for t in per_object}
    for r in restriction["restricted"]:
        end = int(r["tau_i"]) + 1
        traj = nid_to_traj[r["nid"]]
        H = traj["H"][:end]
        dkl = traj["dkl"][:end]
        cells = traj["cells"][:end]
        # Drop windows where D_KL is NaN (early-window convention)
        mask = np.isfinite(H) & np.isfinite(dkl)
        Hf = H[mask]; dklf = dkl[mask]; cf = cells[mask]
        phi_t = compute_phi(Hf, dklf, FROZEN_ALPHA)
        for c in surviving:
            in_cell = cf == c
            if in_cell.any():
                per_cell_phi[c].extend(phi_t[in_cell].tolist())

    # Per-cell empirical variance
    sigma2_emp = np.zeros(len(surviving), dtype=np.float64)
    n_per_cell = np.zeros(len(surviving), dtype=np.int64)
    log.info(f"  per-cell within-cell empirical σ²_c (Φ values):")
    log.info(f"    cell  n_windows  mean_Φ      σ²_emp")
    for i, c in enumerate(surviving):
        vals = np.array(per_cell_phi[c], dtype=np.float64)
        n_per_cell[i] = len(vals)
        if len(vals) >= 2:
            sigma2_emp[i] = float(np.var(vals, ddof=1))
        log.info(f"     {c:>2d}  {n_per_cell[i]:>9d}  {float(vals.mean()):.4f}  "
                 f"{sigma2_emp[i]:.4f}")

    # ── Apply correction ───────────────────────────────────────
    Var_correction = float((E_visits_pi * sigma2_emp).sum())
    log.info(f"  Var_correction = Σ_c E_π[visits] · σ²_c,empirical = "
             f"{Var_correction:,.2f}")
    Var_pred_corrected = var_pred_weighted + Var_correction
    Var_pred_total_corrected = var_pred_total + Var_correction
    r_uncorr = var_pred_weighted / catalog_var_emp
    r_corr = Var_pred_corrected / catalog_var_emp
    r_total_corr = Var_pred_total_corrected / catalog_var_emp
    log.info(f"  Var_pred,weighted                    = {var_pred_weighted:,.2f}")
    log.info(f"  Var_pred,corrected = w + correction  = {Var_pred_corrected:,.2f}")
    log.info(f"  Var_pred,total + correction          = {Var_pred_total_corrected:,.2f}")
    log.info(f"  catalog Var_emp                      = {catalog_var_emp:,.2f}")
    log.info(f"  ratio r_uncorrected                   = {r_uncorr:.6f}")
    log.info(f"  ratio r_corrected (brief's form)      = {r_corr:.6f}")
    log.info(f"  ratio r_total_corrected               = {r_total_corr:.6f}")

    # Per-cell contribution table
    contributions = []
    for i, c in enumerate(surviving):
        contrib = E_visits_pi[i] * sigma2_emp[i]
        contributions.append({
            "cell": int(c),
            "kappa_c": float(kappa_committed[i]),
            "sigma2_c_emp": float(sigma2_emp[i]),
            "n_windows": int(n_per_cell[i]),
            "E_visits_pi": float(E_visits_pi[i]),
            "contribution_to_correction": float(contrib),
            "share_of_correction": float(contrib / Var_correction)
                if Var_correction > 0 else 0.0,
        })

    out = {
        "config": {
            "alpha_frozen": FROZEN_ALPHA,
            "DKL_WINDOW": DKL_WINDOW,
            "DKL_EPS": DKL_EPS,
            "catalog_var_emp_source": str(CATALOG_EMP_VAR),
            "catalog_pred_var_source": str(CATALOG_PRED_VAR),
            "n_restricted_objects": n_objects,
            "surviving_cells": surviving,
        },
        "catalog_var_emp": catalog_var_emp,
        "var_pred_weighted_recomputed": var_pred_weighted,
        "var_starts_v": var_starts_v,
        "var_pred_total": var_pred_total,
        "E_visits_pi": E_visits_pi.tolist(),
        "per_cell_sigma2_empirical": sigma2_emp.tolist(),
        "per_cell_n_windows": n_per_cell.tolist(),
        "var_correction": Var_correction,
        "var_pred_corrected": Var_pred_corrected,
        "var_pred_total_corrected": Var_pred_total_corrected,
        "r_uncorrected": r_uncorr,
        "r_corrected": r_corr,
        "r_total_corrected": r_total_corr,
        "per_cell_contribution": contributions,
        "preregistered_decision": (
            "If r_corrected ≈ 1 (within ±10 %), within-cell heteroscedasticity "
            "is the verified failure mode of the canonical specification on "
            "the catalog-scale reentry result. If r_corrected ≠ 1 even with "
            "the full empirical correction, other structural issues are "
            "present."
        ),
    }
    (OUT_DIR / "result.json").write_text(json.dumps(out, indent=2))
    write_summary(out)
    log.info(f"  wrote {OUT_DIR/'result.json'}")
    log.info(f"  wrote {OUT_DIR/'summary.md'}")


def write_summary(out: dict):
    cfg = out["config"]
    L = []
    L.append("# Test 3 — Reentry-Calibrated Bridge\n\n")
    L.append("Tests whether the within-cell heteroscedasticity correction term, "
             "calibrated from the catalog-scale empirical Φ distribution per "
             "cell, closes the gap between Kemeny-Snell prediction and the "
             "catalog-scale Var_emp = 33,396,043.\n\n")
    L.append("Decision rule (from the brief): r_corrected ≈ 1 ⇒ within-cell "
             "heteroscedasticity is the verified failure mode of the "
             "canonical specification on the catalog-scale result.\n\n")

    L.append("## Setup\n\n")
    L.append("| parameter | value |\n|---|---|\n")
    L.append(f"| α (frozen)       | {cfg['alpha_frozen']} |\n")
    L.append(f"| W (D_KL window) | {cfg['DKL_WINDOW']} |\n")
    L.append(f"| ε                 | {cfg['DKL_EPS']:.0e} |\n")
    L.append(f"| n restricted catalog objects | {cfg['n_restricted_objects']} |\n")
    L.append(f"| surviving transient cells    | {cfg['surviving_cells']} |\n\n")

    L.append("## Decomposition\n\n")
    L.append("| quantity | value |\n|---|---:|\n")
    L.append(f"| Var_pred,weighted (recomputed under empirical π) | {out['var_pred_weighted_recomputed']:,.2f} |\n")
    L.append(f"| Var_starts(v) (across-start)                    | {out['var_starts_v']:,.2f} |\n")
    L.append(f"| Var_pred,total                                  | {out['var_pred_total']:,.2f} |\n")
    L.append(f"| Var_correction = Σ_c E_π[visits] · σ²_c,emp     | {out['var_correction']:,.2f} |\n")
    L.append(f"| Var_pred,corrected = weighted + correction      | {out['var_pred_corrected']:,.2f} |\n")
    L.append(f"| Var_pred,total + correction                     | {out['var_pred_total_corrected']:,.2f} |\n")
    L.append(f"| catalog Var_emp                                 | {out['catalog_var_emp']:,.2f} |\n\n")

    L.append("## Ratios\n\n")
    L.append("| ratio | value |\n|---|---:|\n")
    L.append(f"| r_uncorrected = Var_pred,weighted / Var_emp        | {out['r_uncorrected']:.6f} |\n")
    L.append(f"| r_corrected   = (weighted + correction) / Var_emp  | {out['r_corrected']:.6f} |\n")
    L.append(f"| r_total_corrected = (total + correction) / Var_emp | {out['r_total_corrected']:.6f} |\n\n")

    r_corr = out["r_corrected"]
    if abs(r_corr - 1.0) < 0.10:
        verdict = ("**Within-cell heteroscedasticity diagnosis CONFIRMED on "
                   "catalog-scale reentry.** The empirical within-cell variance "
                   "term closes the gap to within ±10 %. The path forward is "
                   "Fix A: extend the canonical operator with a within-cell "
                   "variance term computed from runtime Φ statistics.")
    elif abs(r_corr - 1.0) < 0.5:
        verdict = ("**Diagnosis PARTIALLY supported.** The empirical "
                   "within-cell variance term closes a substantial fraction "
                   "of the gap but does not bring r_corrected to within "
                   "±10 % of unity. Additional structural sources are "
                   "present (within-cell autocorrelation of Φ, non-Markovian "
                   "trajectory structure, or higher-order moments).")
    else:
        verdict = ("**Diagnosis NOT supported on catalog-scale data.** Even "
                   "with the full empirical within-cell variance correction, "
                   "r_corrected is far from 1. Within-cell heteroscedasticity "
                   "alone does not explain the catalog-scale result. The "
                   "actual failure mode is unknown and requires further "
                   "investigation. Candidate sources: within-cell "
                   "autocorrelation of Φ (catalog Φ has empirical kurtosis "
                   "≈ 298, far heavier-tailed than Gaussian); non-Markovian "
                   "trajectory structure (D_P10 measured at 0.033 nats, "
                   "non-zero); structural mismatch between the Markov "
                   "reward model and the per-trajectory σ_raw "
                   "construction.")
    L.append("## Verdict\n\n")
    L.append(verdict + "\n\n")

    L.append("## Per-cell contribution to Var_correction\n\n")
    L.append("| cell | n windows | κ_c | σ²_c,emp | E_π[visits] | contribution | share |\n")
    L.append("|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in out["per_cell_contribution"]:
        L.append(f"| {r['cell']} | {r['n_windows']:,} | {r['kappa_c']:.4f} | "
                 f"{r['sigma2_c_emp']:.4f} | {r['E_visits_pi']:.4f} | "
                 f"{r['contribution_to_correction']:,.2f} | "
                 f"{r['share_of_correction']*100:.2f} % |\n")
    L.append("\n")

    L.append("## Caveats (scientific-rigor)\n\n")
    L.append("- The correction term assumes per-visit Φ values are conditionally "
             "independent given the cell sequence. The empirical Φ within a "
             "cell may be autocorrelated across consecutive visits (a "
             "trajectory crossing cell c twice may have similar Φ both "
             "times because of slow underlying physics). The linear "
             "correction over-estimates Var_correction in that case.\n")
    L.append("- σ²_c,empirical pools across all (trajectory, window) Φ values "
             "in cell c. It does not condition on the trajectory.\n")
    L.append("- The catalog Φ distribution has heavy tails (excess kurtosis "
             "≈ 298 from the catalog-scale run); the Gaussian-moment "
             "assumption underlying the correction (Test 2 untruncated mode) "
             "is approximate. Test 3's r_corrected reflects whether the "
             "linear correction is sufficient for this data, not whether "
             "the underlying distribution is well-behaved.\n")
    (OUT_DIR / "summary.md").write_text("".join(L))


if __name__ == "__main__":
    main()
