#!/usr/bin/env python3
"""
STTS Apophis Case Study — The Canonical Planetary Defense Event

Fetches the complete orbital history of 99942 Apophis from JPL Horizons
(discovery 2004-06-19 through 2029-04-01) and runs the STTS pipeline
to answer two questions:

  1. At what date does the 2029 close approach first appear in the
     geometric signature? (Full-history analysis)

  2. How soon after discovery could STTS have flagged Apophis?
     (Arc-length sensitivity: truncate to 14, 30, 60, 90, 180, 365 days)

Apophis specifics:
  - Discovered June 19, 2004
  - 2029 close approach: April 13, 2029, at 0.000253 AU
    (closer than geostationary orbit — ~38,000 km from Earth center)
  - Initial calculations showed 2.7% impact probability (later ruled out)
  - Complete orbital history from 2004 to present in Horizons

Usage:
    python3 run_apophis.py

Requires the 250-asteroid pipeline to have run first (uses its trained
LDA + failure basin as the reference corpus).
"""

import json
import time
import numpy as np
import requests
from datetime import datetime, timedelta
from horizons_stts_pipeline import (
    HORIZONS_URL, REQUEST_DELAY,
    OrbitalElements, parse_horizons_elements,
    extract_features, run_pipeline, build_dataset,
    fetch_close_approaches, fetch_orbital_elements_history,
    jd_to_horizons_date,
    WINDOW_DAYS, STRIDE_DAYS, WARNING_DAYS,
)
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


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
        print(f"  a range: {elements[0].a:.6f} – {elements[-1].a:.6f} AU")
        print(f"  e range: {elements[0].e:.6f} – {elements[-1].e:.6f}")
        print(f"  q range: {elements[0].q:.6f} – {elements[-1].q:.6f} AU")

    return elements


def run_full_history_analysis(elements, corpus_trajs):
    """
    Run STTS on full Apophis history.
    Train on the corpus of other NEA close approaches,
    then walk through Apophis's trajectory and detect when
    the 2029 flyby geometry first appears.
    """
    print("\n=== Full History Analysis ===")
    print(f"  Apophis trajectory: {len(elements)} days")
    print(f"  Corpus: {len(corpus_trajs)} other NEA trajectories")

    # Build training set from corpus (other asteroids, not Apophis)
    X_train, r_train, y_train = build_dataset(corpus_trajs)
    print(f"  Training windows: {len(X_train)} "
          f"({y_train.sum():.0f} precursor, {(y_train==0).sum():.0f} nominal)")

    if y_train.sum() < 5:
        print("  ERROR: insufficient precursor windows in corpus")
        return None

    # Fit pipeline on corpus
    W = np.ones(30)
    W[8:10]  *= 3.0   # dq/dt
    W[16:18] *= 3.0   # q distance from 1 AU
    W[18:20] *= 3.0   # dq toward 1 AU
    W[20:22] *= 2.0   # d2q/dt2
    W[24]    *= 3.0   # q slope
    W[25]    *= 4.0   # late/early ratio

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_w = X_scaled * W

    lda = LinearDiscriminantAnalysis(n_components=1, solver='svd')
    lda.fit(X_w, y_train)
    X_proj = lda.transform(X_w).ravel()

    basin = X_proj[y_train == 1]

    # Calibrate epsilon on training data
    thresholds = np.percentile(X_proj, np.linspace(5, 95, 40))
    best_f1, best_eps = 0.0, thresholds[0]
    for eps in thresholds:
        preds = (np.array([np.mean(np.sort(np.abs(basin - p))[:min(5, len(basin))])
                           for p in X_proj]) < eps).astype(int)
        tp = ((preds==1)&(y_train==1)).sum()
        fp = ((preds==1)&(y_train==0)).sum()
        fn = ((preds==0)&(y_train==1)).sum()
        pr = tp/max(1,tp+fp); re = tp/max(1,tp+fn)
        f1 = 2*pr*re/max(1e-10,pr+re)
        if f1 > best_f1:
            best_f1, best_eps = f1, eps

    print(f"  Calibrated ε: {best_eps:.4f} (training F1: {best_f1:.3f})")

    # Walk through Apophis trajectory
    print(f"\n  Walking through Apophis trajectory...")
    first_fire_jd = None
    first_fire_rul = None
    detection_history = []

    for start in range(0, len(elements) - WINDOW_DAYS + 1, STRIDE_DAYS):
        window = elements[start:start + WINDOW_DAYS]
        rul = FLYBY_JD - window[-1].jd  # days until 2029 flyby

        if rul < 0:
            break

        feat = extract_features(window, FLYBY_JD)
        x_s = scaler.transform(feat.reshape(1, -1))
        x_w = x_s * W
        proj = lda.transform(x_w).ravel()[0]
        dist = np.mean(np.sort(np.abs(basin - proj))[:min(5, len(basin))])

        fired = dist < best_eps

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

    # Report
    if first_fire_jd:
        days_from_j2000 = first_fire_jd - 2451545.0
        fire_date = datetime(2000, 1, 1, 12) + timedelta(days=days_from_j2000)
        print(f"\n  FIRST DETECTION:")
        print(f"    Date:     {fire_date.strftime('%Y-%m-%d')}")
        print(f"    RUL:      {first_fire_rul:.0f} days before 2029 flyby")
        print(f"    Years:    {first_fire_rul/365.25:.1f} years before flyby")
    else:
        print(f"\n  STTS did not fire on Apophis trajectory")

    # Summary of detection history
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
        "epsilon": best_eps,
    }


def run_arc_length_sensitivity(elements, corpus_trajs):
    """
    Arc-length sensitivity: truncate Apophis history to N days after
    discovery and check if STTS fires at each truncation.

    Answers: how soon after discovery could STTS have flagged Apophis?
    """
    print("\n=== Arc-Length Sensitivity ===")
    print("  Truncating Apophis history to first N days after discovery")
    print("  At each truncation: does the monitoring query fire?")
    print()

    arc_lengths = [7, 14, 21, 30, 45, 60, 90, 180, 365, 730, 1825]

    # Build corpus-trained pipeline (same as full history)
    X_train, r_train, y_train = build_dataset(corpus_trajs)
    W = np.ones(30)
    W[8:10] *= 3.0; W[16:18] *= 3.0; W[18:20] *= 3.0
    W[20:22] *= 2.0; W[24] *= 3.0; W[25] *= 4.0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_w = X_scaled * W
    lda = LinearDiscriminantAnalysis(n_components=1, solver='svd')
    lda.fit(X_w, y_train)
    X_proj = lda.transform(X_w).ravel()
    basin = X_proj[y_train == 1]

    # Calibrate epsilon
    thresholds = np.percentile(X_proj, np.linspace(5, 95, 40))
    best_f1, best_eps = 0.0, thresholds[0]
    for eps in thresholds:
        dists_t = np.array([np.mean(np.sort(np.abs(basin - p))[:min(5, len(basin))])
                           for p in X_proj])
        preds = (dists_t < eps).astype(int)
        tp = ((preds==1)&(y_train==1)).sum()
        fp = ((preds==1)&(y_train==0)).sum()
        fn = ((preds==0)&(y_train==1)).sum()
        pr = tp/max(1,tp+fp); re = tp/max(1,tp+fn)
        f1 = 2*pr*re/max(1e-10,pr+re)
        if f1 > best_f1:
            best_f1, best_eps = f1, eps

    discovery_jd = elements[0].jd
    results = []

    print(f"  {'Arc (days)':>12s}  {'Windows':>8s}  {'Fired':>6s}  "
          f"{'Min Dist':>10s}  {'Final Dist':>10s}  {'RUL at end':>10s}")
    print("  " + "-" * 68)

    for arc in arc_lengths:
        # Truncate to first `arc` days
        truncated = [e for e in elements if e.jd <= discovery_jd + arc]
        if len(truncated) < WINDOW_DAYS:
            print(f"  {arc:>12d}  insufficient_arc (need >= {WINDOW_DAYS} days for {WINDOW_DAYS}-day window)")
            results.append({"arc_days": arc, "status": "insufficient_arc",
                            "reason": f"arc ({len(truncated)} days) < window size ({WINDOW_DAYS} days)"})
            continue

        # Evaluate the last window
        window = truncated[-WINDOW_DAYS:]
        rul = FLYBY_JD - window[-1].jd
        feat = extract_features(window, FLYBY_JD)
        x_s = scaler.transform(feat.reshape(1, -1))
        x_w = x_s * W
        proj = lda.transform(x_w).ravel()[0]
        dist = np.mean(np.sort(np.abs(basin - proj))[:min(5, len(basin))])
        fired = dist < best_eps

        # Also check all windows in the truncated arc
        n_fired = 0
        min_dist = float('inf')
        for start in range(0, len(truncated) - WINDOW_DAYS + 1, max(1, STRIDE_DAYS)):
            w = truncated[start:start + WINDOW_DAYS]
            f = extract_features(w, FLYBY_JD)
            xs = scaler.transform(f.reshape(1, -1))
            xw = xs * W
            p = lda.transform(xw).ravel()[0]
            d = np.mean(np.sort(np.abs(basin - p))[:min(5, len(basin))])
            if d < best_eps:
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
    print("STTS Apophis Case Study")
    print("99942 Apophis — 2029 Close Approach at 0.000253 AU")
    print("=" * 60)
    print()

    # ── Step 1: Fetch Apophis full history ──────────────────
    elements = fetch_apophis_full_history()
    if not elements:
        print("ERROR: Could not fetch Apophis data")
        return

    # ── Step 2: Build corpus from other NEAs ────────────────
    print("\nStep 2: Build corpus from other NEA close approaches")
    events = fetch_close_approaches(
        dist_max_au=0.02,
        date_min="2005-01-01",
        date_max="2020-01-01",
        v_inf_max=15.0,
    )

    corpus_trajs = []
    fetch_limit = 80  # enough for a solid corpus
    failed = 0
    for i, event in enumerate(events[:fetch_limit]):
        # Skip Apophis itself
        if "99942" in event.designation or "Apophis" in event.designation.lower():
            continue

        try:
            jd_start = event.jd - 365
            jd_end = event.jd - 1
            els = fetch_orbital_elements_history(
                event.designation, jd_start, jd_end, step="1d"
            )
            if len(els) >= WINDOW_DAYS + 10:
                corpus_trajs.append((els, event.jd))
                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{fetch_limit}] {len(corpus_trajs)} usable")
            time.sleep(REQUEST_DELAY)
        except Exception:
            failed += 1

    print(f"  Corpus: {len(corpus_trajs)} NEA trajectories ({failed} failed)")

    if len(corpus_trajs) < 10:
        print("ERROR: Insufficient corpus")
        return

    # ── Step 3: Full history analysis ───────────────────────
    full_results = run_full_history_analysis(elements, corpus_trajs)

    # ── Step 4: Arc-length sensitivity ──────────────────────
    arc_results = run_arc_length_sensitivity(elements, corpus_trajs)

    # ── Save results ────────────────────────────────────────
    def jsonify(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {
        "apophis": {
            "designation": APOPHIS_DESIGNATION,
            "flyby_jd": FLYBY_JD,
            "flyby_dist_au": FLYBY_DIST_AU,
            "n_elements": len(elements),
        },
        "corpus_size": len(corpus_trajs),
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
