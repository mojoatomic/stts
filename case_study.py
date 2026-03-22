"""
Apophis case study: frozen canonical model applied to 99942 Apophis.

Loads serialized artifacts — never calls .fit().
Fetches Apophis orbital history from JPL Horizons, runs full history
analysis and arc-length sensitivity with the frozen model.

Usage:
    python3 case_study.py

Requires: corpus built + model trained
"""

import os
import json
import numpy as np
import requests
from datetime import datetime, timedelta

import config
from train import load_model, md5_file
from horizons_stts_pipeline import (
    HORIZONS_URL,
    parse_horizons_elements,
    extract_features,
)


def fetch_apophis():
    """Fetch Apophis orbital elements from discovery to just before 2029 flyby."""
    print(f"Fetching Apophis ({config.APOPHIS_DESIGNATION}) from Horizons...")
    print(f"  Range: {config.APOPHIS_DISCOVERY_DATE} to {config.APOPHIS_FLYBY_DATE}")

    params = {
        "format": "json",
        "COMMAND": f"'{config.APOPHIS_DESIGNATION}'",
        "EPHEM_TYPE": "ELEMENTS",
        "CENTER": "500@10",
        "START_TIME": f"'{config.APOPHIS_DISCOVERY_DATE}'",
        "STOP_TIME": f"'{config.APOPHIS_FLYBY_DATE}'",
        "STEP_SIZE": "'1d'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "OUT_UNITS": "AU-D",
        "REF_PLANE": "ECLIPTIC",
        "REF_SYSTEM": "J2000",
        "TP_TYPE": "ABSOLUTE",
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


def full_history(elements, model):
    """Walk through full Apophis trajectory with frozen model."""
    print("\n--- Full History Analysis ---")
    print(f"  Apophis trajectory: {len(elements)} days")

    scaler = model["scaler"]
    W = model["W"]
    lda = model["lda"]
    eps = model["epsilon"]
    basin = model["basin"]
    k = config.K_NEIGHBORS
    flyby_jd = config.APOPHIS_FLYBY_JD

    first_fire_jd = None
    first_fire_rul = None
    history = []

    for start in range(
        0, len(elements) - config.WINDOW_DAYS + 1, config.STRIDE_DAYS
    ):
        window = elements[start : start + config.WINDOW_DAYS]
        rul = flyby_jd - window[-1].jd
        if rul < 0:
            break

        feat = extract_features(window, flyby_jd)
        x_s = scaler.transform(feat.reshape(1, -1))
        x_w = x_s * W
        proj = lda.transform(x_w).ravel()[0]
        dist = float(
            np.mean(np.sort(np.abs(basin - proj))[: min(k, len(basin))])
        )
        fired = dist < eps

        history.append({
            "jd": window[-1].jd,
            "rul_days": float(rul),
            "basin_dist": dist,
            "fired": fired,
        })

        if fired and first_fire_jd is None:
            first_fire_jd = window[-1].jd
            first_fire_rul = rul

    n_fired = sum(1 for h in history if h["fired"])

    if first_fire_jd:
        days_from_j2000 = first_fire_jd - 2451545.0
        fire_date = datetime(2000, 1, 1, 12) + timedelta(days=days_from_j2000)
        print(f"  First detection: {fire_date.strftime('%Y-%m-%d')} "
              f"({first_fire_rul:.0f}d / {first_fire_rul/365.25:.1f}y before flyby)")
    else:
        print("  STTS did not fire on Apophis trajectory")

    print(f"  Windows: {len(history)} evaluated, "
          f"{n_fired} fired ({100*n_fired/max(1,len(history)):.1f}%)")

    return {
        "first_fire_jd": first_fire_jd,
        "first_fire_rul_days": float(first_fire_rul) if first_fire_rul else None,
        "n_windows": len(history),
        "n_fired": n_fired,
    }


def arc_sensitivity(elements, model):
    """Arc-length sensitivity with frozen model."""
    print("\n--- Arc-Length Sensitivity ---")
    print("  Using frozen canonical model (same as corpus validation)")

    scaler = model["scaler"]
    W = model["W"]
    lda = model["lda"]
    eps = model["epsilon"]
    basin = model["basin"]
    k = config.K_NEIGHBORS
    flyby_jd = config.APOPHIS_FLYBY_JD
    discovery_jd = elements[0].jd

    arc_lengths = [7, 14, 21, 30, 45, 60, 90, 180, 365, 730, 1825]
    results = []

    print(f"\n  {'Arc':>8s}  {'Windows':>8s}  {'Fired':>6s}  {'Min Dist':>10s}")
    print("  " + "-" * 40)

    for arc in arc_lengths:
        truncated = [e for e in elements if e.jd <= discovery_jd + arc]
        if len(truncated) < config.WINDOW_DAYS:
            print(f"  {arc:>8d}  insufficient_arc")
            results.append({"arc_days": arc, "status": "insufficient_arc"})
            continue

        n_fired = 0
        min_dist = float("inf")

        for start in range(
            0,
            len(truncated) - config.WINDOW_DAYS + 1,
            max(1, config.STRIDE_DAYS),
        ):
            w = truncated[start : start + config.WINDOW_DAYS]
            feat = extract_features(w, flyby_jd)
            xs = scaler.transform(feat.reshape(1, -1))
            xw = xs * W
            p = lda.transform(xw).ravel()[0]
            d = float(
                np.mean(np.sort(np.abs(basin - p))[: min(k, len(basin))])
            )
            if d < eps:
                n_fired += 1
            min_dist = min(min_dist, d)

        n_windows = max(
            1,
            (len(truncated) - config.WINDOW_DAYS) // max(1, config.STRIDE_DAYS) + 1,
        )
        marker = " <<< DETECTED" if n_fired > 0 else ""

        print(
            f"  {arc:>8d}  {n_windows:>8d}  {n_fired:>6d}  "
            f"{min_dist:>10.4f}{marker}"
        )

        results.append({
            "arc_days": arc,
            "n_windows": n_windows,
            "n_fired": n_fired,
            "min_basin_dist": min_dist,
        })

    return results


def main():
    print("=" * 60)
    print("Apophis Case Study — Frozen Canonical Model")
    print("=" * 60)

    # ── Verify Apophis not in training corpus ─────────────
    with open(config.TRAIN_DESIGNATIONS_FILE) as f:
        train_des = json.load(f)
    assert not any(config.APOPHIS_DESIGNATION in d for d in train_des), \
        "Apophis found in training corpus!"
    print(f"Verified: Apophis not in training corpus ({len(train_des)} objects)")

    # ── Load frozen model ─────────────────────────────────
    model = load_model()
    print(f"Loaded model: ε={model['epsilon']:.4f}, "
          f"V1={model['meta']['metrics']['v1_separation']:.1f}x, "
          f"V2 ρ={model['meta']['metrics']['v2_rho']:.3f}")

    # ── Fetch Apophis ─────────────────────────────────────
    elements = fetch_apophis()
    if not elements:
        raise RuntimeError("Could not fetch Apophis data")

    # ── Full history ──────────────────────────────────────
    full = full_history(elements, model)

    # ── Arc-length sensitivity ────────────────────────────
    arcs = arc_sensitivity(elements, model)

    # ── Save results ──────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    results = {
        "config": config.config_snapshot(),
        "artifact_checksums": {
            "scaler": md5_file(config.SCALER_FILE),
            "lda": md5_file(config.LDA_FILE),
            "basin": md5_file(config.BASIN_FILE),
        },
        "apophis": {
            "designation": config.APOPHIS_DESIGNATION,
            "flyby_jd": config.APOPHIS_FLYBY_JD,
            "flyby_dist_au": config.APOPHIS_FLYBY_DIST_AU,
            "n_elements": len(elements),
        },
        "full_history": full,
        "arc_sensitivity": arcs,
    }

    outfile = f"{config.RESULTS_DIR}/case_study.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {outfile}")
    return results


if __name__ == "__main__":
    main()
