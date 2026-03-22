"""
1825-day lookback experiment: does the model fire earlier with longer history?

Loads the frozen canonical model from artifacts/. Re-fetches test object
trajectories with 1825-day lookback (vs 365-day training lookback).
Compares first-fire dates between 365-day and 1825-day histories.

Design:
  - Same frozen model (scaler, LDA, epsilon) — no retraining
  - Same test objects from cached corpus
  - Only change: longer trajectory histories for test evaluation
  - Reports: first-fire date per object, distribution comparison

Usage:
    python3 lookback.py

Requires: corpus built + model trained (python3 run_all.py)
Runtime: ~14 minutes (795 Horizons API calls at 1 req/sec)
"""

import os
import json
import time
import numpy as np

import config
from corpus import load_corpus
from train import load_model, md5_file
from horizons_stts_pipeline import (
    fetch_orbital_elements_history,
    extract_features,
    REQUEST_DELAY,
)

LOOKBACK_EXTENDED = 1825  # 5 years


def run_lookback():
    """Compare 365-day vs 1825-day test evaluation on frozen model."""
    model = load_model()
    corpus = load_corpus()
    trajectories = corpus["trajectories"]
    test_idx = corpus["test_idx"]

    scaler = model["scaler"]
    W = model["W"]
    lda = model["lda"]
    eps = model["epsilon"]
    basin = model["basin"]
    k = config.K_NEIGHBORS

    def dist_to_basin(p):
        dists = np.abs(basin - p)
        return np.mean(np.sort(dists)[:min(k, len(basin))])

    # ── Evaluate 365-day histories (from cached corpus) ───
    print("Evaluating 365-day histories (cached)...")
    results_365 = []
    for idx in test_idx:
        traj = trajectories[idx]
        elements = traj["elements"]
        ca_jd = traj["ca_jd"]
        fire_rul = _first_fire(elements, ca_jd, scaler, W, lda, eps, dist_to_basin)
        results_365.append({
            "designation": traj["designation"],
            "ca_jd": ca_jd,
            "fire_rul": fire_rul,
            "history_days": ca_jd - elements[0].jd if elements else 0,
        })

    detected_365 = sum(1 for r in results_365 if r["fire_rul"] is not None)
    leads_365 = [r["fire_rul"] for r in results_365 if r["fire_rul"] is not None]
    print(f"  Detected: {detected_365}/{len(results_365)}")
    if leads_365:
        lt = np.array(leads_365)
        print(f"  Lead time: mean={lt.mean():.0f}d, median={np.median(lt):.0f}d")

    # ── Re-fetch test objects with 1825-day lookback ──────
    print(f"\nFetching 1825-day histories for {len(test_idx)} test objects...")
    results_1825 = []
    fetch_failed = 0

    for i, idx in enumerate(test_idx):
        traj = trajectories[idx]
        des = traj["designation"]
        ca_jd = traj["ca_jd"]

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{len(test_idx)}] {des}...")

        try:
            elements_ext = fetch_orbital_elements_history(
                des,
                ca_jd - LOOKBACK_EXTENDED,
                ca_jd - 1,
                step="1d",
            )
            if len(elements_ext) >= config.WINDOW_DAYS + 10:
                fire_rul = _first_fire(
                    elements_ext, ca_jd, scaler, W, lda, eps, dist_to_basin
                )
                actual_days = ca_jd - elements_ext[0].jd
                results_1825.append({
                    "designation": des,
                    "ca_jd": ca_jd,
                    "fire_rul": fire_rul,
                    "history_days": actual_days,
                })
            else:
                # Fall back to 365-day cached data
                elements = traj["elements"]
                fire_rul = _first_fire(
                    elements, ca_jd, scaler, W, lda, eps, dist_to_basin
                )
                results_1825.append({
                    "designation": des,
                    "ca_jd": ca_jd,
                    "fire_rul": fire_rul,
                    "history_days": ca_jd - elements[0].jd if elements else 0,
                    "fallback": True,
                })
                fetch_failed += 1
        except Exception:
            # Fall back to 365-day cached data
            elements = traj["elements"]
            fire_rul = _first_fire(
                elements, ca_jd, scaler, W, lda, eps, dist_to_basin
            )
            results_1825.append({
                "designation": des,
                "ca_jd": ca_jd,
                "fire_rul": fire_rul,
                "history_days": ca_jd - elements[0].jd if elements else 0,
                "fallback": True,
            })
            fetch_failed += 1

        time.sleep(REQUEST_DELAY)

    detected_1825 = sum(1 for r in results_1825 if r["fire_rul"] is not None)
    leads_1825 = [r["fire_rul"] for r in results_1825 if r["fire_rul"] is not None]
    fallbacks = sum(1 for r in results_1825 if r.get("fallback"))

    print(f"\n  Detected: {detected_1825}/{len(results_1825)}")
    print(f"  Fetch failures (fell back to 365d): {fallbacks}")
    if leads_1825:
        lt = np.array(leads_1825)
        print(f"  Lead time: mean={lt.mean():.0f}d, median={np.median(lt):.0f}d")

    # ── Comparison ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON: 365-day vs 1825-day lookback")
    print("=" * 60)

    lt365 = np.array(leads_365) if leads_365 else np.array([0.0])
    lt1825 = np.array(leads_1825) if leads_1825 else np.array([0.0])

    print(f"\n{'Metric':<25s} {'365-day':>10s} {'1825-day':>10s}")
    print("-" * 50)
    print(f"{'Detected':<25s} {detected_365:>10d} {detected_1825:>10d}")
    print(f"{'Mean lead (days)':<25s} {lt365.mean():>10.0f} {lt1825.mean():>10.0f}")
    print(f"{'Median lead (days)':<25s} {np.median(lt365):>10.0f} {np.median(lt1825):>10.0f}")
    print(f"{'Min lead (days)':<25s} {lt365.min():>10.0f} {lt1825.min():>10.0f}")
    print(f"{'25th pct (days)':<25s} {np.percentile(lt365,25):>10.0f} {np.percentile(lt1825,25):>10.0f}")
    print(f"{'75th pct (days)':<25s} {np.percentile(lt365,75):>10.0f} {np.percentile(lt1825,75):>10.0f}")
    print(f"{'Max lead (days)':<25s} {lt365.max():>10.0f} {lt1825.max():>10.0f}")

    # Per-object comparison
    gained = same = lost = 0
    for r3, r18 in zip(results_365, results_1825):
        if r3["fire_rul"] is not None and r18["fire_rul"] is not None:
            if r18["fire_rul"] > r3["fire_rul"] + 7:
                gained += 1
            elif r3["fire_rul"] > r18["fire_rul"] + 7:
                lost += 1
            else:
                same += 1

    print(f"\n  Earlier detection (>7d gain): {gained}")
    print(f"  Similar detection (±7d):      {same}")
    print(f"  Later detection (>7d loss):    {lost}")

    # ── Save results ──────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    output = {
        "config": config.config_snapshot(),
        "artifact_checksums": {
            "scaler": md5_file(config.SCALER_FILE),
            "lda": md5_file(config.LDA_FILE),
            "basin": md5_file(config.BASIN_FILE),
        },
        "lookback_days": LOOKBACK_EXTENDED,
        "n_test": len(test_idx),
        "n_fetch_fallback": fallbacks,
        "365_day": {
            "detected": detected_365,
            "mean_lead": round(float(lt365.mean()), 1),
            "median_lead": round(float(np.median(lt365)), 1),
            "min_lead": round(float(lt365.min()), 1),
            "max_lead": round(float(lt365.max()), 1),
            "p25_lead": round(float(np.percentile(lt365, 25)), 1),
            "p75_lead": round(float(np.percentile(lt365, 75)), 1),
        },
        "1825_day": {
            "detected": detected_1825,
            "mean_lead": round(float(lt1825.mean()), 1),
            "median_lead": round(float(np.median(lt1825)), 1),
            "min_lead": round(float(lt1825.min()), 1),
            "max_lead": round(float(lt1825.max()), 1),
            "p25_lead": round(float(np.percentile(lt1825, 25)), 1),
            "p75_lead": round(float(np.percentile(lt1825, 75)), 1),
        },
        "comparison": {
            "gained_gt7d": gained,
            "similar_within7d": same,
            "lost_gt7d": lost,
        },
        "lead_times_365": leads_365,
        "lead_times_1825": [float(x) for x in leads_1825],
    }

    outfile = f"{config.RESULTS_DIR}/lookback.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {outfile}")


def _first_fire(elements, ca_jd, scaler, W, lda, eps, dist_to_basin):
    """Find the first window that fires. Returns RUL in days, or None."""
    if len(elements) < config.WINDOW_DAYS:
        return None

    for start in range(
        0, len(elements) - config.WINDOW_DAYS + 1, config.STRIDE_DAYS
    ):
        window = elements[start : start + config.WINDOW_DAYS]
        rul = ca_jd - window[-1].jd
        if rul < 0:
            break

        feat = extract_features(window, ca_jd)
        x_s = scaler.transform(feat.reshape(1, -1))
        x_w = x_s * W
        proj = lda.transform(x_w).ravel()[0]
        d = dist_to_basin(proj)

        if d < eps:
            return float(rul)

    return None


if __name__ == "__main__":
    run_lookback()
