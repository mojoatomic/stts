#!/usr/bin/env python3
"""
Post-hoc exact hitting-time and endpoint CDFs per object.

Rather than modifying the energy-state pipeline to swap its reported
P_forward from MC to exact, we compute the exact quantity directly
from the per-object CSVs (which already record M2_cell at each window)
and the frozen transition matrix. This:

- Preserves committed MC-based outputs byte-identically (no pipeline
  code changes);
- Produces the exact paper-headline values (max P_end and max P_hit
  per object at Δt=10, plus full horizon sweep);
- Emits an MC-vs-exact cross-check (within MC stderr confirms the
  exact implementation matches MC at large N).

Outputs:
  results/reentry/catalog_scale_validation/per_object_exact.csv
  results/reentry/catalog_scale_validation/mc_vs_exact.csv
  results/reentry/catalog_scale_validation/post_hoc_exact.md
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reentry.hitting_time import (
    assert_hit_ge_end,
    exact_endpoint_cdf,
    exact_hitting_cdf,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/reentry/catalog_scale_validation")
OUT_CSV = OUT_DIR / "per_object_exact.csv"
MC_VS_EXACT = OUT_DIR / "mc_vs_exact.csv"
OUT_MD = OUT_DIR / "post_hoc_exact.md"

PER_OBJECT_78 = Path("results/reentry/per_object")
PER_OBJECT_CATALOG = OUT_DIR / "per_object"
AGGREGATE_78 = Path("results/reentry/aggregate_summary.csv")
AGGREGATE_CATALOG = OUT_DIR / "per_object.csv"
MARKOV = Path("artifacts/reentry/markov_table.npz")

HORIZONS = list(range(1, 31))
H_IDX_10 = HORIZONS.index(10)


def _f(v: str) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return float("nan")


def max_exact_over_windows(csv_path: Path, P_end, P_hit) -> tuple[float, float, float, dict]:
    """Return (exact_max_P_end_dt10, exact_max_P_hit_dt10, mc_max_P_forward, per_dt_maxes).

    per_dt_maxes is a dict {Δt: max_P_hit_over_windows_at_Δt}.
    """
    cells = []
    mc_pf = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            cells.append(int(r["M2_cell"]))
            mc_pf.append(_f(r["P_forward"]))
    if not cells:
        return float("nan"), float("nan"), float("nan"), {}
    cells_arr = np.array(cells, dtype=np.int64)
    mc_arr = np.array(mc_pf)
    # Exact max: take max over visited cells of P_end[cell, :] etc
    P_end_cells = P_end[cells_arr]  # (n_windows, H)
    P_hit_cells = P_hit[cells_arr]  # (n_windows, H)
    per_dt_maxes = {int(h): float(P_hit_cells[:, i].max())
                    for i, h in enumerate(HORIZONS)}
    return (
        float(P_end_cells[:, H_IDX_10].max()),
        float(P_hit_cells[:, H_IDX_10].max()),
        float(np.nanmax(mc_arr)),
        per_dt_maxes,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Loading frozen P + failure cells...")
    d = np.load(MARKOV)
    P = d["transition_matrix"]
    failure_cells = sorted(d["failure_cells"].tolist())
    log.info(f"  K={P.shape[0]}, failure_cells={failure_cells}")

    log.info("Computing exact P_end, P_hit tables (Δt=1..30)...")
    P_end = exact_endpoint_cdf(P, failure_cells, HORIZONS)
    P_hit = exact_hitting_cdf(P, failure_cells, HORIZONS)
    assert_hit_ge_end(P_hit, P_end)
    log.info("  inequality P_hit >= P_end holds everywhere")

    # Gather all objects to process.
    objects: list[tuple[str, str, Path]] = []
    # 78-object reference:
    for fp in sorted(PER_OBJECT_78.glob("*.csv")):
        objects.append((fp.stem, "reference_78", fp))
    # Catalog-scale per-object CSVs (in catalog_scale_validation/per_object).
    # The catalog-scale run also wrote to results/reentry/per_object for the
    # overlapping NORADs, but we use the committed 78-object dir for those and
    # the catalog_scale dir for the others (3,346 additional). Merge:
    cat_ids = set()
    if PER_OBJECT_CATALOG.exists():
        for fp in sorted(PER_OBJECT_CATALOG.glob("*.csv")):
            if fp.stem not in {o[0] for o in objects}:
                objects.append((fp.stem, "catalog_scale", fp))
                cat_ids.add(fp.stem)
    log.info(f"  {len(objects)} per-object CSVs total ({sum(1 for o in objects if o[1]=='reference_78')} reference + "
             f"{sum(1 for o in objects if o[1]=='catalog_scale')} catalog-scale-only)")

    log.info("Computing exact per-object max P_end, P_hit...")
    rows = []
    diffs = []
    for nid, source, fp in objects:
        pe_max, ph_max, mc_max, per_dt = max_exact_over_windows(fp, P_end, P_hit)
        rows.append({
            "norad_id": nid,
            "source": source,
            "exact_P_end_max_dt10": pe_max,
            "exact_P_hit_max_dt10": ph_max,
            "mc_P_forward_max": mc_max,
            "P_hit_max_dt5": per_dt.get(5, float("nan")),
            "P_hit_max_dt10": per_dt.get(10, float("nan")),
            "P_hit_max_dt15": per_dt.get(15, float("nan")),
            "P_hit_max_dt20": per_dt.get(20, float("nan")),
            "P_hit_max_dt25": per_dt.get(25, float("nan")),
            "P_hit_max_dt30": per_dt.get(30, float("nan")),
        })
        if not np.isnan(mc_max) and not np.isnan(pe_max):
            diffs.append(mc_max - pe_max)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log.info(f"wrote {OUT_CSV} ({len(rows)} rows)")

    # MC-vs-exact diagnostic
    diffs_arr = np.array(diffs)
    with open(MC_VS_EXACT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["norad_id", "source", "mc_P_forward_max", "exact_P_end_max_dt10", "diff"])
        for r in rows:
            if not np.isnan(r["mc_P_forward_max"]) and not np.isnan(r["exact_P_end_max_dt10"]):
                w.writerow([r["norad_id"], r["source"],
                            f"{r['mc_P_forward_max']:.6f}",
                            f"{r['exact_P_end_max_dt10']:.6f}",
                            f"{r['mc_P_forward_max'] - r['exact_P_end_max_dt10']:.6f}"])
    log.info(f"wrote {MC_VS_EXACT}")

    # Diagnostic summary
    if len(diffs_arr) > 0:
        abs_diffs = np.abs(diffs_arr)
        log.info(f"MC vs exact max-over-trajectory diff (N objects = {len(diffs_arr)}):")
        log.info(f"  mean |Δ| = {abs_diffs.mean():.5f}")
        log.info(f"  median |Δ| = {np.median(abs_diffs):.5f}")
        log.info(f"  p95 |Δ| = {np.quantile(abs_diffs, 0.95):.5f}")
        log.info(f"  p99 |Δ| = {np.quantile(abs_diffs, 0.99):.5f}")
        log.info(f"  max |Δ| = {abs_diffs.max():.5f}")
        # Expected MC stderr at p=0.9, N=10000: sqrt(0.9*0.1/10000) = 0.003.
        # Max over trajectory of ~ few hundred windows: each is ~MC-stderr;
        # max-of-N samples is ~ 3x stderr. Within-spec if p99 < 0.01.

    # Summary for the report
    source_stats = {}
    for source in ("reference_78", "catalog_scale"):
        subset = [r for r in rows if r["source"] == source]
        if not subset:
            continue
        pe = np.array([r["exact_P_end_max_dt10"] for r in subset])
        ph = np.array([r["exact_P_hit_max_dt10"] for r in subset])
        mc = np.array([r["mc_P_forward_max"] for r in subset])
        source_stats[source] = {
            "n": len(subset),
            "exact_P_end_median": float(np.median(pe)),
            "exact_P_end_mean": float(np.mean(pe)),
            "exact_P_hit_median": float(np.median(ph)),
            "exact_P_hit_mean": float(np.mean(ph)),
            "mc_P_forward_median": float(np.median(mc)),
            "mc_P_forward_mean": float(np.mean(mc)),
            "n_exact_P_end_lt_0.5": int((pe < 0.5).sum()),
            "n_exact_P_end_lt_0.25": int((pe < 0.25).sum()),
            "n_exact_P_hit_lt_0.5": int((ph < 0.5).sum()),
            "n_exact_P_hit_lt_0.25": int((ph < 0.25).sum()),
            "n_exact_P_hit_eq_1.0": int((ph >= 1.0 - 1e-9).sum()),
        }

    write_md(source_stats, abs_diffs, len(rows))
    log.info(f"wrote {OUT_MD}")


def write_md(source_stats, abs_diffs, n_total):
    L = []
    L.append("# STTS-Reentry — Post-hoc Exact Hitting-Time Per Object\n\n")
    L.append("Exact `max_t P_end[M2_cell[t], Δt=10]` and `max_t P_hit[M2_cell[t], Δt=10]` "
             "per object, computed from per-object CSVs (M2_cell per window) and the "
             "frozen transition matrix. No pipeline code changes; committed MC outputs "
             "are preserved byte-identically.\n\n")
    L.append("## MC-vs-exact cross-check\n\n")
    L.append(f"Objects compared: {len(abs_diffs)} (MC max and exact max, per-trajectory).\n\n")
    if len(abs_diffs) > 0:
        L.append("| statistic | value |\n|---|---:|\n")
        for label, val in [("mean |Δ|", abs_diffs.mean()),
                            ("median |Δ|", float(np.median(abs_diffs))),
                            ("p95 |Δ|", float(np.quantile(abs_diffs, 0.95))),
                            ("p99 |Δ|", float(np.quantile(abs_diffs, 0.99))),
                            ("max |Δ|", float(abs_diffs.max()))]:
            L.append(f"| {label} | {val:.5f} |\n")
        L.append(f"\nExpected MC stderr at p ≈ 0.9, N = 10000 samples: "
                 f"`√(0.9·0.1/10000) ≈ 0.003`. Max-over-trajectory of ~N windows "
                 f"is on the order of a few times stderr; consistent agreement if "
                 f"p99 |Δ| < ~0.01.\n\n")
    L.append("## Per-source summary\n\n")
    for source, s in source_stats.items():
        L.append(f"### {source}\n\n")
        L.append(f"n = {s['n']}\n\n")
        L.append("| metric | exact P_end (endpoint) | exact P_hit (hitting CDF) | MC P_forward |\n")
        L.append("|---|---:|---:|---:|\n")
        L.append(f"| median | {s['exact_P_end_median']:.4f} | {s['exact_P_hit_median']:.4f} | {s['mc_P_forward_median']:.4f} |\n")
        L.append(f"| mean   | {s['exact_P_end_mean']:.4f} | {s['exact_P_hit_mean']:.4f} | {s['mc_P_forward_mean']:.4f} |\n\n")
        L.append("| bucket | exact P_end count | exact P_hit count |\n")
        L.append("|---|---:|---:|\n")
        L.append(f"| max < 0.5 | {s['n_exact_P_end_lt_0.5']} | {s['n_exact_P_hit_lt_0.5']} |\n")
        L.append(f"| max < 0.25 | {s['n_exact_P_end_lt_0.25']} | {s['n_exact_P_hit_lt_0.25']} |\n")
        L.append(f"| max = 1.0 (hit basin) | — | {s['n_exact_P_hit_eq_1.0']} |\n\n")
    OUT_MD.write_text("".join(L))


if __name__ == "__main__":
    main()
