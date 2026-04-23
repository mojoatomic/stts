#!/usr/bin/env python3
"""
Validation-set bootstrap with three-pass decay-epoch sensitivity.

Resample per-object rows from the 78-object reference aggregate and
the 3,424-object catalog-scale sample. Produce percentile 95% CIs for
object-level aggregate statistics (lead-time medians, mean/median
max P, false-alarm rate, bimodal-outlier rate). Report each statistic
three times:

  - Full:      all rows, as committed.
  - Excluded:  drop the 12 known catalog-error NORADs (from
               catalog_errors_provenance.csv).
  - Corrected: for CONFIRMED year-off-by-one objects (7 of 12),
               substitute the GCAT DDate for the corpus date; recompute
               lead times relative to the corrected decay_epoch.
               Drop the 5 LIKELY-error objects whose true date falls
               outside local TLE coverage.

Reads existing CSVs; no pipeline re-run required.

Outputs:
  results/reentry/catalog_scale_validation/bootstrap_validation_cis.json
  results/reentry/catalog_scale_validation/bootstrap_validation_cis.md
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reentry.bootstrap_ci import bootstrap_ci, bootstrap_rate_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/reentry/catalog_scale_validation")
OUT_JSON = OUT_DIR / "bootstrap_validation_cis.json"
OUT_MD = OUT_DIR / "bootstrap_validation_cis.md"

PER_OBJECT = OUT_DIR / "per_object.csv"
AGGREGATE_78 = Path("results/reentry/aggregate_summary.csv")
PROVENANCE = OUT_DIR / "catalog_errors_provenance.csv"
GP_HISTORY = Path("data/reentry/gp_history_cache")

N_REPLICATES = 1000
SEED = 20260422


# ── Helpers ──────────────────────────────────────────────────────

def _f(v: str) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return float("nan")


def load_per_object_rows(path: Path, schema: str) -> list[dict]:
    """Load per-object CSV, returning list of dicts with normalized keys.

    `schema` is 'catalog' (3,424-object) or '78' (committed aggregate).
    """
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if schema == "catalog":
                norm = {
                    "norad_id": r["norad_id"].strip(),
                    "decay_date": r["decay_date"].strip(),
                    "max_p": _f(r["max_p_failure"]),
                    "lead_10": _f(r["lead_p10_days"]),
                    "lead_25": _f(r["lead_p25_days"]),
                    "lead_50": _f(r["lead_p50_days"]),
                    "lead_75": _f(r["lead_p75_days"]),
                }
            elif schema == "78":
                norm = {
                    "norad_id": r["norad_id"].strip(),
                    "decay_date": r["decay_epoch"].strip(),
                    "max_p": _f(r["P_forward_max"]),
                    "lead_10": _f(r["lead_0.10"]),
                    "lead_25": _f(r["lead_0.25"]),
                    "lead_50": _f(r["lead_0.50"]),
                    "lead_75": _f(r["lead_0.75"]),
                }
            else:
                raise ValueError(schema)
            rows.append(norm)
    return rows


def load_provenance() -> list[dict]:
    with open(PROVENANCE) as f:
        return list(csv.DictReader(f))


def corrected_lead_times_if_year_off(nid: str, gcat_ddate: str) -> dict[str, float]:
    """For a CONFIRMED year-off-by-one object, recompute lead times
    relative to the GCAT DDate instead of the corpus date.

    Returns a dict with keys lead_10, lead_25, lead_50, lead_75 or NaN
    if the trajectory never crossed the threshold.
    """
    # Load per-window P_forward trajectory from the per-object CSV, if present.
    po = Path("results/reentry/per_object") / f"{nid}.csv"
    if not po.exists():
        # Fall back to catalog-scale per-object dir
        po = OUT_DIR / "per_object" / f"{nid}.csv"
    if not po.exists():
        return {f"lead_{t}": float("nan") for t in (10, 25, 50, 75)}

    # Reconstruct days-to-corrected-decay for each window by re-reading
    # the window's days_to_reentry column and adding back the corpus-vs-GCAT offset.
    # Committed per-object CSVs have columns: t, days_to_reentry, M2_cell, ...
    # We need: nearest TLE epoch -> days before GCAT DDate.
    # days_to_reentry in CSV is days before CORPUS decay. Correction: days_to_gcat = days_to_corpus - delta_days.
    # delta = corpus_date - gcat_date (positive if corpus is later, i.e. GCAT is earlier).

    rows = []
    with open(po) as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "days_to_reentry": _f(r["days_to_reentry"]),
                    "P_forward": _f(r["P_forward"]),
                }
            )
    if not rows:
        return {f"lead_{t}": float("nan") for t in (10, 25, 50, 75)}

    # We don't directly know the corpus_date here; infer from the provenance row.
    prov = {p["norad_id"]: p for p in load_provenance()}
    if nid not in prov:
        return {f"lead_{t}": float("nan") for t in (10, 25, 50, 75)}
    corpus_d = date.fromisoformat(prov[nid]["corpus_decay_date"])
    gcat_d = date.fromisoformat(prov[nid]["gcat_ddate"])
    delta_days = (corpus_d - gcat_d).days  # e.g. -364 for year-off-by-one

    # Corrected days-to-reentry = days_to_corpus - delta_days
    # (If corpus claimed an earlier decay than GCAT, the actual reentry is later,
    # so for a given window the days-to-actual-reentry is LARGER.)
    corrected = [
        {"days_to_reentry": r["days_to_reentry"] - delta_days, "P_forward": r["P_forward"]}
        for r in rows
    ]
    corrected.sort(key=lambda r: -r["days_to_reentry"])  # earliest first

    out = {}
    for th_name, th in [("lead_10", 0.10), ("lead_25", 0.25), ("lead_50", 0.50), ("lead_75", 0.75)]:
        crossed = [r for r in corrected if r["P_forward"] >= th]
        # "First crossed" = earliest window (largest days_to_reentry) where P_forward >= th.
        # corrected is sorted earliest-first, so crossed[0] is it.
        out[th_name] = crossed[0]["days_to_reentry"] if crossed else float("nan")
    return out


def apply_correction(rows: list[dict], provenance: list[dict]) -> list[dict]:
    """Return a NEW list of rows with known-bad decay_epochs corrected and
    lead times recomputed. Drops rows that can't be corrected."""
    confirmed = {
        p["norad_id"]: p for p in provenance
        if p["verdict"] in ("CONFIRMED_year_off_by_one", "CONFIRMED_corpus_error")
    }
    likely = {
        p["norad_id"] for p in provenance
        if p["verdict"] == "LIKELY_error_out_of_TLE_window"
    }
    out = []
    drops = 0
    corrects = 0
    for r in rows:
        nid = r["norad_id"]
        if nid in likely:
            drops += 1
            continue
        if nid in confirmed:
            rc = dict(r)
            corrected = corrected_lead_times_if_year_off(nid, confirmed[nid]["gcat_ddate"])
            rc["lead_10"] = corrected["lead_10"]
            rc["lead_25"] = corrected["lead_25"]
            rc["lead_50"] = corrected["lead_50"]
            rc["lead_75"] = corrected["lead_75"]
            out.append(rc)
            corrects += 1
        else:
            out.append(r)
    log.info(f"  apply_correction: corrected {corrects}, dropped {drops}")
    return out


def apply_exclusion(rows: list[dict], provenance: list[dict]) -> list[dict]:
    """Drop all 12 known catalog-error NORADs."""
    drop = {p["norad_id"] for p in provenance}
    out = [r for r in rows if r["norad_id"] not in drop]
    log.info(f"  apply_exclusion: dropped {len(rows) - len(out)} of {len(rows)}")
    return out


# ── Statistics ───────────────────────────────────────────────────

def stat_block(rows: list[dict], label: str) -> dict:
    """Compute a full block of CIs on a row set."""
    max_p = np.array([r["max_p"] for r in rows if not np.isnan(r["max_p"])])
    leads = {t: np.array([r[f"lead_{t}"] for r in rows if not np.isnan(r[f"lead_{t}"])])
             for t in (10, 25, 50, 75)}

    out = {"label": label, "n_rows": len(rows)}

    # Max P stats
    p, lo, hi = bootstrap_ci(max_p, statistic=np.median, n_replicates=N_REPLICATES, seed=SEED)
    out["median_max_p"] = {"point": p, "ci_lo": lo, "ci_hi": hi, "n": int(len(max_p))}
    p, lo, hi = bootstrap_ci(max_p, statistic=np.mean, n_replicates=N_REPLICATES, seed=SEED)
    out["mean_max_p"] = {"point": p, "ci_lo": lo, "ci_hi": hi, "n": int(len(max_p))}

    # Rates
    mask_05 = (max_p < 0.5).astype(int)
    p, lo, hi = bootstrap_rate_ci(mask_05, n_replicates=N_REPLICATES, seed=SEED)
    out["rate_max_p_below_0.5"] = {"point": p, "ci_lo": lo, "ci_hi": hi, "n": int(len(max_p))}
    mask_25 = (max_p < 0.25).astype(int)
    p, lo, hi = bootstrap_rate_ci(mask_25, n_replicates=N_REPLICATES, seed=SEED)
    out["rate_max_p_below_0.25"] = {"point": p, "ci_lo": lo, "ci_hi": hi, "n": int(len(max_p))}

    # Lead-time medians (resample only objects that crossed the threshold)
    for t in (10, 25, 50, 75):
        vals = leads[t]
        if len(vals) == 0:
            out[f"median_lead_{t}"] = {"point": float("nan"), "ci_lo": float("nan"),
                                       "ci_hi": float("nan"), "n": 0}
            continue
        p, lo, hi = bootstrap_ci(vals, statistic=np.median, n_replicates=N_REPLICATES, seed=SEED)
        out[f"median_lead_{t}"] = {"point": p, "ci_lo": lo, "ci_hi": hi, "n": int(len(vals))}

    return out


# ── Main ─────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading per-object data...")
    rows_catalog = load_per_object_rows(PER_OBJECT, schema="catalog")
    rows_78 = load_per_object_rows(AGGREGATE_78, schema="78")
    provenance = load_provenance()
    log.info(f"  catalog-scale n={len(rows_catalog)}, 78-object n={len(rows_78)}, "
             f"provenance candidates n={len(provenance)}")

    log.info("Bootstrapping CIs (three-pass sensitivity)...")

    out: dict = {"n_replicates": N_REPLICATES, "seed": SEED}

    for label, rows in (("catalog_scale_3424", rows_catalog), ("reference_78", rows_78)):
        log.info(f"  == {label} ==")

        log.info(f"    Full pass (n={len(rows)})")
        full = stat_block(rows, "full")

        log.info(f"    Exclusion pass")
        excl_rows = apply_exclusion(rows, provenance)
        excluded = stat_block(excl_rows, "excluded")

        log.info(f"    Correction pass (year-off substitution, lead recompute)")
        corr_rows = apply_correction(rows, provenance)
        corrected = stat_block(corr_rows, "corrected")

        out[label] = {"full": full, "excluded": excluded, "corrected": corrected}

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"wrote {OUT_JSON}")

    # Sanity: every CI should envelope its point estimate.
    for label in ("catalog_scale_3424", "reference_78"):
        for pass_ in ("full", "excluded", "corrected"):
            block = out[label][pass_]
            for k, v in block.items():
                if isinstance(v, dict) and "point" in v:
                    p, lo, hi = v["point"], v["ci_lo"], v["ci_hi"]
                    if not np.isnan(p) and not np.isnan(lo) and not np.isnan(hi):
                        if not (lo - 1e-9 <= p <= hi + 1e-9):
                            raise SystemExit(
                                f"ENVELOPE FAIL: {label}.{pass_}.{k} "
                                f"point={p} CI=[{lo},{hi}]"
                            )
    log.info("envelope assertions: all passed")

    write_md(out)
    log.info(f"wrote {OUT_MD}")


def _fmt(v: dict) -> str:
    p = v["point"]; lo = v["ci_lo"]; hi = v["ci_hi"]
    if np.isnan(p):
        return "—"
    return f"{p:.4f} [{lo:.4f}, {hi:.4f}]"


def _fmt_lead(v: dict) -> str:
    p = v["point"]; lo = v["ci_lo"]; hi = v["ci_hi"]
    n = v["n"]
    if np.isnan(p):
        return "—"
    return f"{p:.1f} [{lo:.1f}, {hi:.1f}] (n={n})"


def write_md(out: dict) -> None:
    L = []
    L.append("# STTS-Reentry — Validation-Set Bootstrap (three-pass decay-epoch sensitivity)\n\n")
    L.append(
        f"Percentile bootstrap over per-object aggregate statistics, "
        f"B = {out['n_replicates']} replicates, seed {out['seed']}. Three passes:\n\n"
        "- **Full**: all rows as committed.\n"
        "- **Excluded**: drop the 12 known catalog-error NORADs.\n"
        "- **Corrected**: substitute GCAT DDate for the 7 CONFIRMED year-off-by-one "
        "objects; recompute lead times on the corrected reference epoch. Drop the 5 "
        "LIKELY-error objects whose true date falls outside the local TLE window.\n\n"
    )
    for label, human in (("reference_78", "78-object reference (committed aggregate)"),
                         ("catalog_scale_3424", "Catalog-scale stratified sample (n=3,424)")):
        L.append(f"## {human}\n\n")
        for stat_name, display in (
            ("median_max_p", "median max P"),
            ("mean_max_p", "mean max P"),
            ("rate_max_p_below_0.5", "rate max P < 0.5"),
            ("rate_max_p_below_0.25", "rate max P < 0.25 (bimodal-outlier)"),
            ("median_lead_10", "median lead @ P≥0.10 (days)"),
            ("median_lead_25", "median lead @ P≥0.25 (days)"),
            ("median_lead_50", "median lead @ P≥0.50 (days)"),
            ("median_lead_75", "median lead @ P≥0.75 (days)"),
        ):
            L.append(f"**{display}**\n\n")
            L.append("| pass | n | point | 95% CI |\n|---|---:|---:|---:|\n")
            for pass_ in ("full", "excluded", "corrected"):
                block = out[label][pass_]
                v = block.get(stat_name)
                if v is None:
                    continue
                n = v.get("n", "—")
                if stat_name.startswith("median_lead"):
                    cell = _fmt_lead(v)
                else:
                    cell = _fmt(v)
                # Parse the cell back into point/CI for the table column
                p = v["point"]; lo = v["ci_lo"]; hi = v["ci_hi"]
                if np.isnan(p):
                    L.append(f"| {pass_} | {n} | — | — |\n")
                elif stat_name.startswith("median_lead"):
                    L.append(f"| {pass_} | {n} | {p:.2f} | [{lo:.2f}, {hi:.2f}] |\n")
                else:
                    L.append(f"| {pass_} | {n} | {p:.6f} | [{lo:.6f}, {hi:.6f}] |\n")
            # Full vs corrected delta check
            full_v = out[label]["full"].get(stat_name)
            corr_v = out[label]["corrected"].get(stat_name)
            if (full_v and corr_v
                    and not np.isnan(full_v["point"]) and not np.isnan(corr_v["point"])):
                delta = corr_v["point"] - full_v["point"]
                half_ci = (full_v["ci_hi"] - full_v["ci_lo"]) / 2
                in_ci = abs(delta) <= half_ci
                L.append(f"\n_Delta corrected − full = {delta:+.4f}; "
                         f"full CI half-width = {half_ci:.4f}; "
                         f"load-bearing on labels = **{'no' if in_ci else 'YES'}**_\n\n")
            else:
                L.append("\n")
    OUT_MD.write_text("".join(L))


if __name__ == "__main__":
    main()
