#!/usr/bin/env python3
"""
SWAN-SF Window Stitching — Build single trajectories per active region.

For each target AR, collects all FL and NF 12-hour sliding windows,
deduplicates on timestamp, and produces a single chronological
trajectory of 24 SHARP magnetic parameters.

Usage:
    python solar/stitch_trajectories.py [--data-dir DATA_DIR] [--partition N]

Prerequisites:
    Partition 3 extracted in data/swan-sf/ (see explore_swan_sf.py for download).
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# 24 SHARP magnetic field parameters — the physics channels for STTS
SHARP_PARAMS = [
    "TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP",
    "USFLUX", "TOTFZ", "MEANPOT", "EPSZ", "MEANSHR", "SHRGT45",
    "MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH", "MEANJZH", "TOTFY",
    "MEANJZD", "MEANALP", "TOTFX", "EPSY", "EPSX", "R_VALUE",
]

# Target ARs from exploration report (top-5 by window count in partition 3)
TARGET_ARS = ["3686", "3341", "3688", "4197", "3721"]

CADENCE_MINUTES = 12


def parse_filename(fname: str) -> dict | None:
    """Extract AR number, start/end timestamps, flare class, and source (FL/NF)."""
    base = fname.replace(".csv", "")
    ar_idx = base.find("_ar")
    s_idx = base.find("_s")
    e_idx = base.rfind("_e")
    if ar_idx == -1 or s_idx == -1 or e_idx == -1:
        return None

    ar = base[ar_idx + 3 : s_idx]
    start = base[s_idx + 2 : e_idx]
    end = base[e_idx + 2 :]

    # Flare class: FQ for quiet, or e.g. M1.0, X3.3
    if base.startswith("FQ_"):
        flare_class = "FQ"
        goes_letter = "Q"
    else:
        at_idx = base.find("@")
        flare_class = base[:at_idx] if at_idx != -1 else "?"
        goes_letter = flare_class[0] if flare_class else "?"

    return {
        "ar": ar,
        "start": start,
        "end": end,
        "flare_class": flare_class,
        "goes_letter": goes_letter,
        "filename": fname,
    }


def collect_windows(data_dir: str, partition: int, ar: str) -> list[dict]:
    """Collect all FL and NF window files for a given AR."""
    windows = []
    for subdir, source in [("FL", "FL"), ("NF", "NF")]:
        dirpath = Path(data_dir) / f"partition{partition}" / subdir
        if not dirpath.exists():
            continue
        for fname in os.listdir(dirpath):
            if f"_ar{ar}_" not in fname:
                continue
            info = parse_filename(fname)
            if info is None:
                continue
            info["source"] = source
            info["filepath"] = str(dirpath / fname)
            windows.append(info)

    windows.sort(key=lambda w: w["start"])
    return windows


def read_sharp_rows(filepath: str) -> list[dict]:
    """Read all rows from a SWAN-SF CSV, returning timestamp + 24 SHARP values."""
    rows = []
    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ts = row.get("Timestamp", "")
            if not ts:
                continue
            values = {}
            for p in SHARP_PARAMS:
                raw = row.get(p, "")
                if raw in ("", "None", "nan", "NaN"):
                    values[p] = None
                else:
                    try:
                        values[p] = float(raw)
                    except ValueError:
                        values[p] = None
            rows.append({"timestamp": ts, "values": values})
    return rows


def stitch_ar(data_dir: str, partition: int, ar: str) -> dict:
    """Stitch all windows for an AR into a single deduplicated trajectory."""
    windows = collect_windows(data_dir, partition, ar)
    if not windows:
        return {"ar": ar, "error": "no windows found"}

    # Deduplicate on timestamp, keeping track of source (FL/NF) per timestamp
    ts_data = {}  # timestamp -> {values, source, flare_class}
    ts_source = {}  # timestamp -> source label for NF/FL transition tracking

    for w in windows:
        rows = read_sharp_rows(w["filepath"])
        for row in rows:
            ts = row["timestamp"]
            if ts not in ts_data:
                ts_data[ts] = row["values"]
                ts_source[ts] = w["source"]
            # If already present, skip (verified identical in exploration)

    # Sort chronologically
    sorted_ts = sorted(ts_data.keys())
    n_timestamps = len(sorted_ts)

    if n_timestamps == 0:
        return {"ar": ar, "error": "no timestamps after stitching"}

    # Build trajectory array info
    first_dt = datetime.fromisoformat(sorted_ts[0])
    last_dt = datetime.fromisoformat(sorted_ts[-1])
    span_hours = (last_dt - first_dt).total_seconds() / 3600

    # Null count across SHARP params
    null_count = 0
    total_cells = n_timestamps * len(SHARP_PARAMS)
    for ts in sorted_ts:
        for p in SHARP_PARAMS:
            if ts_data[ts][p] is None:
                null_count += 1

    # Gap detection (gaps > 1 cadence = >12 min)
    gaps = []
    for i in range(1, len(sorted_ts)):
        t0 = datetime.fromisoformat(sorted_ts[i - 1])
        t1 = datetime.fromisoformat(sorted_ts[i])
        delta_min = (t1 - t0).total_seconds() / 60
        if delta_min > CADENCE_MINUTES + 1:  # allow 1-min tolerance
            gaps.append({
                "after_index": i - 1,
                "from": sorted_ts[i - 1],
                "to": sorted_ts[i],
                "gap_minutes": delta_min,
                "missing_steps": int(delta_min / CADENCE_MINUTES) - 1,
            })

    # NF/FL transition points
    transitions = []
    for i in range(1, len(sorted_ts)):
        prev_src = ts_source[sorted_ts[i - 1]]
        curr_src = ts_source[sorted_ts[i]]
        if prev_src != curr_src:
            transitions.append({
                "index": i,
                "timestamp": sorted_ts[i],
                "from": prev_src,
                "to": curr_src,
            })

    # Window-level flare class timeline
    flare_timeline = []
    for w in windows:
        flare_timeline.append({
            "start": w["start"],
            "end": w["end"],
            "class": w["flare_class"],
            "source": w["source"],
        })

    # Unique flare classes observed
    fl_classes = sorted(set(w["flare_class"] for w in windows if w["source"] == "FL"))
    nf_classes = sorted(set(w["flare_class"] for w in windows if w["source"] == "NF"))

    return {
        "ar": ar,
        "shape": (n_timestamps, len(SHARP_PARAMS)),
        "n_timestamps": n_timestamps,
        "n_params": len(SHARP_PARAMS),
        "first_timestamp": sorted_ts[0],
        "last_timestamp": sorted_ts[-1],
        "span_hours": span_hours,
        "span_days": span_hours / 24,
        "null_count": null_count,
        "null_pct": null_count / total_cells * 100 if total_cells > 0 else 0,
        "n_windows_fl": sum(1 for w in windows if w["source"] == "FL"),
        "n_windows_nf": sum(1 for w in windows if w["source"] == "NF"),
        "fl_classes": fl_classes,
        "nf_classes": nf_classes,
        "gaps": gaps,
        "transitions": transitions,
        "trajectory_timestamps": sorted_ts,
        "trajectory_data": ts_data,
        "trajectory_source": ts_source,
    }


def print_report(results: list[dict]):
    """Print formatted stitching report."""
    sep = "=" * 72

    print(f"\n{sep}")
    print("  SWAN-SF WINDOW STITCHING REPORT")
    print(f"  5 Sample Active Regions — Partition 3")
    print(sep)

    for r in results:
        if "error" in r:
            print(f"\nAR {r['ar']}: ERROR — {r['error']}")
            continue

        ar = r["ar"]
        print(f"\n{'─' * 72}")
        print(f"  AR {ar}")
        print(f"{'─' * 72}")

        print(f"\n  Shape: {r['shape']}  ({r['n_timestamps']} timesteps x {r['n_params']} SHARP params)")
        print(f"  Windows:  {r['n_windows_fl']} FL + {r['n_windows_nf']} NF = {r['n_windows_fl'] + r['n_windows_nf']} total")
        print(f"  Span:     {r['first_timestamp']} → {r['last_timestamp']}")
        print(f"            {r['span_hours']:.1f} hours ({r['span_days']:.1f} days)")
        print(f"  Nulls:    {r['null_count']}/{r['n_timestamps'] * r['n_params']} ({r['null_pct']:.2f}%)")
        print(f"  FL classes: {', '.join(r['fl_classes'])}")
        print(f"  NF classes: {', '.join(r['nf_classes'])}")

        # Gaps
        print(f"\n  Gaps (>{CADENCE_MINUTES} min):")
        if not r["gaps"]:
            print(f"    None — continuous trajectory")
        else:
            for g in r["gaps"]:
                hours = g["gap_minutes"] / 60
                print(f"    [{g['after_index']:4d}] {g['from']} → {g['to']}  "
                      f"({hours:.1f}h, ~{g['missing_steps']} missing steps)")

        # NF/FL transitions
        print(f"\n  NF ↔ FL transitions:")
        if not r["transitions"]:
            print(f"    None — single source throughout")
        else:
            # Consolidate: show first and last of each contiguous block
            blocks = []
            current_src = r["trajectory_source"][r["trajectory_timestamps"][0]]
            block_start_idx = 0
            block_start_ts = r["trajectory_timestamps"][0]

            for i in range(1, len(r["trajectory_timestamps"])):
                ts = r["trajectory_timestamps"][i]
                src = r["trajectory_source"][ts]
                if src != current_src:
                    blocks.append({
                        "source": current_src,
                        "start_idx": block_start_idx,
                        "end_idx": i - 1,
                        "start_ts": block_start_ts,
                        "end_ts": r["trajectory_timestamps"][i - 1],
                        "length": i - block_start_idx,
                    })
                    current_src = src
                    block_start_idx = i
                    block_start_ts = ts

            # Final block
            blocks.append({
                "source": current_src,
                "start_idx": block_start_idx,
                "end_idx": len(r["trajectory_timestamps"]) - 1,
                "start_ts": block_start_ts,
                "end_ts": r["trajectory_timestamps"][-1],
                "length": len(r["trajectory_timestamps"]) - block_start_idx,
            })

            for b in blocks:
                label = "FL (flaring)" if b["source"] == "FL" else "NF (quiet)  "
                span_h = (datetime.fromisoformat(b["end_ts"]) -
                          datetime.fromisoformat(b["start_ts"])).total_seconds() / 3600
                print(f"    [{b['start_idx']:4d}–{b['end_idx']:4d}] {label}  "
                      f"{b['start_ts'][:16]} → {b['end_ts'][:16]}  "
                      f"({b['length']} steps, {span_h:.1f}h)")

    # Summary table
    print(f"\n{sep}")
    print("  SUMMARY")
    print(sep)
    print(f"\n  {'AR':>6}  {'Shape':>14}  {'Span':>8}  {'Gaps':>4}  {'Transitions':>11}  {'Nulls':>6}")
    print(f"  {'─'*6}  {'─'*14}  {'─'*8}  {'─'*4}  {'─'*11}  {'─'*6}")
    for r in results:
        if "error" in r:
            continue
        shape_str = f"{r['n_timestamps']}x{r['n_params']}"
        span_str = f"{r['span_days']:.1f}d"
        print(f"  {r['ar']:>6}  {shape_str:>14}  {span_str:>8}  {len(r['gaps']):>4}  "
              f"{len(r['transitions']):>11}  {r['null_pct']:>5.2f}%")

    print(f"\n{sep}\n")


def main():
    parser = argparse.ArgumentParser(description="Stitch SWAN-SF windows into trajectories")
    parser.add_argument("--data-dir", default="data/swan-sf",
                        help="Path to SWAN-SF data directory")
    parser.add_argument("--partition", type=int, default=3,
                        help="Partition number (default: 3)")
    args = parser.parse_args()

    data_dir = args.data_dir
    pdir = Path(data_dir) / f"partition{args.partition}"
    if not pdir.exists():
        print(f"ERROR: {pdir} not found. Extract partition first.")
        sys.exit(1)

    results = []
    for ar in TARGET_ARS:
        print(f"Stitching AR {ar}...", file=sys.stderr)
        result = stitch_ar(data_dir, args.partition, ar)
        results.append(result)

    print_report(results)


if __name__ == "__main__":
    main()
