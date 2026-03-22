#!/usr/bin/env python3
"""
STTS Apophis Case Study — The Canonical Planetary Defense Event

Uses the SAME canonical model (W weights, scaler, LDA, epsilon) trained
on the SAME 200-object corpus from the 2000–2024 CNEOS pool as the
800-object corpus validation in horizons_stts_pipeline.py.

This script does NOT train its own model. It:
  1. Fetches the same CNEOS events and Horizons trajectories
  2. Applies the same seed-42 split to get the same 200 training objects
  3. Trains the canonical model via train_model()
  4. Fetches Apophis's full orbital history
  5. Evaluates Apophis against the frozen canonical model

Both results — the 800-object validation and Apophis — come from one
trained model. One set of W weights. One training corpus.

Usage:
    python3 run_apophis.py

Requires: pip install requests numpy scipy scikit-learn pandas
No JPL authentication required.
"""

import json
import time
import numpy as np
from datetime import datetime, timedelta
from horizons_stts_pipeline import (
    HORIZONS_URL, REQUEST_DELAY,
    OrbitalElements, parse_horizons_elements,
    extract_features, build_dataset,
    train_model,
    fetch_close_approaches, fetch_orbital_elements_history,
    jd_to_horizons_date,
    WINDOW_DAYS, STRIDE_DAYS, WARNING_DAYS,
)
import requests


# ─────────────────────────────────────────────────────────────
# APOPHIS PARAMETERS
# ─────────────────────────────────────────────────────────────

APOPHIS_DESIGNATION = "99942"
DISCOVERY_DATE      = "2004-Jun-19"
FLYBY_DATE          = "2029-Apr-13"
FLYBY_JD            = 2462136.5  # April 13, 2029 TDB (approximate)
FLYBY_DIST_AU       = 0.000253   # 37,800 km from Earth center


def fetch_apophis_full_history():
    """Fetch Apophis orbital elements from discovery to just before 2029 flyby."""
    print(f"Fetching Apophis (99942) from Horizons...")
    print(f"  Range: {DISCOVERY_DATE} to {FLYBY_DATE}")

    params = {
        "format":     "json",
        "COMMAND":    f"'{APOPHIS_DESIGNATION}'",
        "EPHEM_TYPE": "ELEMENTS",
        "CENTER":     "500@10",
        "START_TIME": f"'{DISCOVERY_DATE}'",
        "STOP_TIME":  f"'{FLYBY_DATE}'",
        "STEP_SIZE":  "'1d'",
        "OBJ_DATA":   "NO",
        "MAKE_EPHEM": "YES",
        "OUT_UNITS":  "AU-D",
        "REF_PLANE":  "ECLIPTIC",
        "REF_SYSTEM": "J2000",
        "TP_TYPE":    "ABSOLUTE",
        "ELEM_LABELS": "YES",
        "CSV_FORMAT": "YES",
    }

    r = requests.get(HORIZONS_URL, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()

    if "error" in data:
        raise ValueError(f"Horizons error: {data['error']}")

    elements = parse_horizons_elements(data.get("result", ""))
    print(f"  Retrieved {len(elements)} daily element sets")

    if elements:
        print(f"  First epoch: JD {elements[0].jd:.1f}")
        print(f"  Last epoch:  JD {elements[-1].jd:.1f}")
        print(f"  q range: {elements[0].q:.6f} – {elements[-1].q:.6f} AU")

    return elements


def build_canonical_corpus(verbose=True):
    """
    Build the same 200-object training corpus used by the 1000-object
    validation run. Same CNEOS query, same Horizons fetch, same seed-42
    split. Returns (train_trajs, test_trajs, all_trajectories).
    """
    # Same CNEOS query as horizons_stts_pipeline.py main()
    events = fetch_close_approaches(
        dist_max_au=0.02,
        date_min="2000-01-01",
        date_max="2024-01-01",
        v_inf_max=15.0,
    )

    if not events:
        raise RuntimeError("No events fetched from CNEOS")

    print(f"\nFetching orbital histories from Horizons (365d lookback)...")

    trajectories = []
    failed = 0
    fetch_limit = 1000

    for i, event in enumerate(events[:fetch_limit]):
        # Exclude Apophis from the corpus
        if "99942" in event.designation or "apophis" in event.designation.lower():
            if verbose:
                print(f"  [{i+1:3d}/{min(fetch_limit,len(events))}] "
                      f"{event.designation} — EXCLUDED (Apophis)")
            continue

        if verbose:
            print(f"  [{i+1:3d}/{min(fetch_limit,len(events))}] "
                  f"{event.designation} CA:{event.cd[:10]} "
                  f"dist={event.dist_au:.4f} AU", end="")

        try:
            jd_start = event.jd - 365
            jd_end   = event.jd - 1
            elements = fetch_orbital_elements_history(
                event.designation, jd_start, jd_end, step="1d"
            )
            if len(elements) >= WINDOW_DAYS + 10:
                trajectories.append((elements, event.jd))
                if verbose:
                    print(f" → {len(elements)} epochs OK")
            else:
                if verbose:
                    print(f" → only {len(elements)} epochs, skip")
                failed += 1
        except Exception as ex:
            if verbose:
                print(f" → ERROR: {ex}")
            failed += 1

        time.sleep(REQUEST_DELAY)

    print(f"\n  Usable trajectories: {len(trajectories)}")
    print(f"  Failed/insufficient: {failed}")

    # Same split as horizons_stts_pipeline.py main()
    np.random.seed(42)
    idx = np.random.permutation(len(trajectories))
    n_train = min(200, int(0.8 * len(trajectories)))

    train_trajs = [trajectories[i] for i in idx[:n_train]]
    test_trajs  = [trajectories[i] for i in idx[n_train:]]

    print(f"  Train: {len(train_trajs)}, Test: {len(test_trajs)}")

    return train_trajs, test_trajs, trajectories


def run_full_history_analysis(elements, model):
    """
    Run the frozen canonical model on Apophis's full trajectory.
    Walk through every 30-day window and record when the monitoring
    query fires.
    """
    print("\n=== Full History Analysis ===")
    print(f"  Apophis trajectory: {len(elements)} days")

    scaler = model["scaler"]
    W      = model["W"]
    lda    = model["lda"]
    eps    = model["epsilon"]
    dist_to_basin = model["dist_to_basin"]

    first_fire_jd = None
    first_fire_rul = None
    detection_history = []

    for start in range(0, len(elements) - WINDOW_DAYS + 1, STRIDE_DAYS):
        window = elements[start:start + WINDOW_DAYS]
        rul = FLYBY_JD - window[-1].jd

        if rul < 0:
            break

        feat = extract_features(window, FLYBY_JD)
        x_s = scaler.transform(feat.reshape(1, -1))
        x_w = x_s * W
        proj = lda.transform(x_w).ravel()[0]
        dist = np.mean(np.sort(np.abs(model["basin"] - proj))[:min(5, len(model["basin"]))])

        fired = dist < eps

        detection_history.append({
            "jd": window[-1].jd,
            "rul_days": rul,
            "basin_dist": float(dist),
            "fired": bool(fired),
            "q": window[-1].q,
            "e": window[-1].e,
        })

        if fired and first_fire_jd is None:
            first_fire_jd = window[-1].jd
            first_fire_rul = rul

    if first_fire_jd:
        days_from_j2000 = first_fire_jd - 2451545.0
        fire_date = datetime(2000, 1, 1, 12) + timedelta(days=days_from_j2000)
        print(f"\n  FIRST DETECTION:")
        print(f"    Date:     {fire_date.strftime('%Y-%m-%d')}")
        print(f"    RUL:      {first_fire_rul:.0f} days before 2029 flyby")
        print(f"    Years:    {first_fire_rul/365.25:.1f} years before flyby")
    else:
        print(f"\n  STTS did not fire on Apophis trajectory")

    n_fired = sum(1 for d in detection_history if d["fired"])
    n_total = len(detection_history)
    print(f"\n  Windows evaluated: {n_total}")
    print(f"  Windows fired:     {n_fired} ({100*n_fired/max(1,n_total):.1f}%)")

    return {
        "first_fire_jd": first_fire_jd,
        "first_fire_rul_days": first_fire_rul,
        "n_windows": n_total,
        "n_fired": n_fired,
        "detection_history": detection_history,
    }


def run_arc_length_sensitivity(elements, model):
    """
    Arc-length sensitivity using the frozen canonical model.
    Truncate Apophis history to first N days after discovery,
    check if the monitoring query fires at each truncation.
    """
    print("\n=== Arc-Length Sensitivity ===")
    print("  Truncating Apophis history to first N days after discovery")
    print("  Using frozen canonical model (same as corpus validation)")
    print()

    scaler = model["scaler"]
    W      = model["W"]
    lda    = model["lda"]
    eps    = model["epsilon"]
    basin  = model["basin"]

    arc_lengths = [7, 14, 21, 30, 45, 60, 90, 180, 365, 730, 1825]
    discovery_jd = elements[0].jd
    results = []

    print(f"  {'Arc (days)':>12s}  {'Windows':>8s}  {'Fired':>6s}  "
          f"{'Min Dist':>10s}  {'Final Dist':>10s}  {'RUL at end':>10s}")
    print("  " + "-" * 68)

    for arc in arc_lengths:
        truncated = [e for e in elements if e.jd <= discovery_jd + arc]
        if len(truncated) < WINDOW_DAYS:
            print(f"  {arc:>12d}  insufficient_arc "
                  f"(need >= {WINDOW_DAYS} days for {WINDOW_DAYS}-day window)")
            results.append({
                "arc_days": arc,
                "status": "insufficient_arc",
                "reason": f"arc ({len(truncated)} days) < window size ({WINDOW_DAYS} days)",
            })
            continue

        # Evaluate the last window
        window = truncated[-WINDOW_DAYS:]
        rul = FLYBY_JD - window[-1].jd
        feat = extract_features(window, FLYBY_JD)
        x_s = scaler.transform(feat.reshape(1, -1))
        x_w = x_s * W
        proj = lda.transform(x_w).ravel()[0]
        dist = np.mean(np.sort(np.abs(basin - proj))[:min(5, len(basin))])
        fired = dist < eps

        # Check all windows in the truncated arc
        n_fired = 0
        min_dist = float('inf')
        for start in range(0, len(truncated) - WINDOW_DAYS + 1, max(1, STRIDE_DAYS)):
            w = truncated[start:start + WINDOW_DAYS]
            f = extract_features(w, FLYBY_JD)
            xs = scaler.transform(f.reshape(1, -1))
            xw = xs * W
            p = lda.transform(xw).ravel()[0]
            d = np.mean(np.sort(np.abs(basin - p))[:min(5, len(basin))])
            if d < eps:
                n_fired += 1
            min_dist = min(min_dist, d)

        n_windows = max(1, (len(truncated) - WINDOW_DAYS) // max(1, STRIDE_DAYS) + 1)
        marker = "<<< DETECTED" if n_fired > 0 else ""

        print(f"  {arc:>12d}  {n_windows:>8d}  {n_fired:>6d}  "
              f"{min_dist:>10.4f}  {dist:>10.4f}  {rul:>10.0f}  {marker}")

        results.append({
            "arc_days": arc,
            "n_windows": n_windows,
            "n_fired": n_fired,
            "min_basin_dist": float(min_dist),
            "final_basin_dist": float(dist),
            "fired_final": bool(fired),
            "rul_at_end": float(rul),
        })

    return results


def main():
    print("=" * 60)
    print("STTS Apophis Case Study — Canonical Model")
    print("99942 Apophis — 2029 Close Approach at 0.000253 AU")
    print("Using same model as 800-object corpus validation")
    print("=" * 60)
    print()

    # ── Step 1: Fetch Apophis full history ──────────────────
    elements = fetch_apophis_full_history()
    if not elements:
        print("ERROR: Could not fetch Apophis data")
        return

    # ── Step 2: Build same corpus as 1000-object run ──────
    print("\nStep 2: Build canonical 200-object training corpus")
    print("  (Same CNEOS query, same seed-42 split, Apophis excluded)")
    train_trajs, test_trajs, all_trajs = build_canonical_corpus(verbose=True)

    # ── Step 3: Train canonical model ─────────────────────
    print("\nStep 3: Train canonical model")
    model = train_model(train_trajs, verbose=True)
    if model is None:
        print("ERROR: Failed to train model")
        return

    print(f"\n  Canonical model trained:")
    print(f"    V1 separation: {model['v1_separation']:.1f}x")
    print(f"    V2 Spearman ρ: {model['v2_rho']:.3f}")
    print(f"    Epsilon:       {model['epsilon']:.4f}")
    print(f"    Training F1:   {model['train_f1']:.3f}")

    # ── Step 4: Full history analysis ─────────────────────
    full_results = run_full_history_analysis(elements, model)

    # ── Step 5: Arc-length sensitivity ────────────────────
    arc_results = run_arc_length_sensitivity(elements, model)

    # ── Save results ──────────────────────────────────────
    def jsonify(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {
        "model": {
            "description": "Canonical model: same 200-object training corpus "
                           "as 800-object validation (seed 42, 2000-2024 CNEOS)",
            "train_size": len(train_trajs),
            "v1_separation": model["v1_separation"],
            "v2_rho": model["v2_rho"],
            "epsilon": model["epsilon"],
            "train_f1": model["train_f1"],
        },
        "apophis": {
            "designation": APOPHIS_DESIGNATION,
            "flyby_jd": FLYBY_JD,
            "flyby_dist_au": FLYBY_DIST_AU,
            "n_elements": len(elements),
        },
        "full_history": {
            "first_fire_rul_days": full_results["first_fire_rul_days"] if full_results else None,
            "n_windows": full_results["n_windows"] if full_results else 0,
            "n_fired": full_results["n_fired"] if full_results else 0,
        } if full_results else None,
        "arc_sensitivity": arc_results,
    }

    with open("apophis_stts_results.json", "w") as f:
        json.dump(output, f, indent=2, default=jsonify)

    print("\n" + "=" * 60)
    print("Results saved to apophis_stts_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
