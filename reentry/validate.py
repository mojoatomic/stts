"""
Evaluate the frozen reentry model on the held-out test set.

Loads serialized artifacts — never calls .fit().
Reports F1, precision, recall, detection lead times, and Wilson CIs.
Verifies V1, V2 on test data. Reports per-satellite detection results.

Usage:
    python reentry/validate.py

Requires: corpus built + model trained
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from reentry.config import (
    CONSECUTIVE_FIRE_THRESHOLD,
    PRECURSOR_DAYS,
    RESULTS_DIR,
    V1_P_VALUE_THRESHOLD,
    V2_SPEARMAN_THRESHOLD,
    WINDOW_SIZE,
    WINDOW_STRIDE_EVAL,
    config_snapshot,
)
from reentry.corpus import load_corpus
from reentry.features import build_feature_matrix
from reentry.train import load_model, md5_file, SCALER_FILE, LDA_FILE, BASIN_FILE


def wilson_ci(successes: int, n: int, z: float = 1.96):
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def validate():
    """Evaluate frozen model on test set."""
    model = load_model()
    corpus = load_corpus()
    satellites = corpus["satellites"]
    test_ids = corpus["test_ids"]
    train_ids = corpus["train_ids"]

    # ── Leakage verification ────────────────────────────────
    overlap = set(train_ids) & set(test_ids)
    assert len(overlap) == 0, f"LEAKAGE: {overlap}"
    storm_overlap = set(test_ids) & set(corpus["storm_ids"])
    assert len(storm_overlap) == 0, f"Storm objects in test: {storm_overlap}"
    print(f"Leakage check passed: {len(test_ids)} test, "
          f"{len(train_ids)} train, 0 overlap")

    # ── Extract test features ───────────────────────────────
    X_test, y_test, days_test, ids_test = build_feature_matrix(
        satellites, test_ids,
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE_EVAL,
    )

    print(f"Test windows: {len(X_test)} "
          f"({(y_test == 1).sum()} precursor, "
          f"{(y_test == 0).sum()} nominal, "
          f"{(y_test == -1).sum()} ambiguous)")

    # ── Project through frozen model ────────────────────────
    scaler = model["scaler"]
    W = model["W"]
    lda = model["lda"]
    eps = model["epsilon"]
    dist_to_basin = model["dist_to_basin"]

    X_scaled = np.nan_to_num(scaler.transform(X_test), nan=0.0, posinf=0.0, neginf=0.0)
    X_w = X_scaled * W
    X_proj = lda.transform(X_w).ravel()
    test_dists = np.array([dist_to_basin(p) for p in X_proj])

    # ── V1: Test set separation ─────────────────────────────
    non_ambig = y_test != -1
    nom_d = test_dists[(y_test == 0)]
    pre_d = test_dists[(y_test == 1)]

    if len(nom_d) > 0 and len(pre_d) > 0:
        v1_sep = float(np.median(nom_d) / (np.median(pre_d) + 1e-10))
        _, v1_p = mannwhitneyu(nom_d, pre_d, alternative="greater")
        v1_p = float(v1_p)
    else:
        v1_sep, v1_p = 0.0, 1.0

    print(f"\nV1 (test): {v1_sep:.1f}x separation (p={v1_p:.2e})")
    print(f"  {'PASS' if v1_p < V1_P_VALUE_THRESHOLD else 'FAIL'}")

    # ── V2: Monotonic approach (test) ───────────────────────
    mask_365 = (days_test < 365) & (y_test != -1)
    if mask_365.sum() > 10:
        v2_rho, v2_p = spearmanr(days_test[mask_365], test_dists[mask_365])
        v2_rho, v2_p = float(v2_rho), float(v2_p)
    else:
        v2_rho, v2_p = 0.0, 1.0

    print(f"V2 (test): rho={v2_rho:.3f} (p={v2_p:.2e})")
    print(f"  {'PASS' if v2_rho > V2_SPEARMAN_THRESHOLD else 'FAIL'}")

    # ── Per-satellite detection ─────────────────────────────
    # Two-population evaluation:
    #   Reentry satellites: should fire (TP if detected, FN if not)
    #   Operational satellites: should NOT fire (TN if quiet, FP if fires)
    tp = 0   # reentry satellite detected
    fn = 0   # reentry satellite missed
    fp = 0   # operational satellite falsely fires
    tn = 0   # operational satellite correctly quiet
    lead_times = []
    per_satellite = []

    # Get the split lists to know which are reentry vs operational
    test_reentry_ids = set(corpus.get("test_reentry_ids", []))
    test_operational_ids = set(corpus.get("test_operational_ids", []))

    for nid in test_ids:
        sat = satellites[nid]
        is_reentry = nid in test_reentry_ids

        # Find this satellite's windows in the test set
        sat_mask = np.array([wid.startswith(f"{nid}:") for wid in ids_test])
        if sat_mask.sum() == 0:
            continue

        sat_dists = test_dists[sat_mask]
        sat_days = days_test[sat_mask]

        # Walk windows chronologically (sorted by days_to_reentry descending)
        sort_idx = np.argsort(sat_days)[::-1]
        sat_dists = sat_dists[sort_idx]
        sat_days = sat_days[sort_idx]

        # Consecutive window requirement: K consecutive windows below epsilon.
        # A genuine approach signal is persistent, not a single-window anomaly.
        first_fire_days = None
        consecutive = 0
        for d, dist in zip(sat_days, sat_dists):
            if dist < eps:
                consecutive += 1
                if consecutive >= CONSECUTIVE_FIRE_THRESHOLD:
                    first_fire_days = float(d)
                    break
            else:
                consecutive = 0

        fired = first_fire_days is not None

        result = {
            "norad_id": nid,
            "object_name": sat.get("object_name"),
            "classification": sat.get("classification", ""),
            "decay_epoch": sat.get("decay_epoch"),
            "n_windows": int(sat_mask.sum()),
            "first_fire_days": first_fire_days,
            "fired": fired,
            "is_reentry": is_reentry,
        }

        if is_reentry:
            if fired:
                tp += 1
                if first_fire_days < 1e10:  # finite lead time
                    lead_times.append(first_fire_days)
                result["outcome"] = "TP"
            else:
                fn += 1
                result["outcome"] = "FN"
        else:
            # Operational satellite
            if fired:
                fp += 1
                result["outcome"] = "FP"
            else:
                tn += 1
                result["outcome"] = "TN"

        per_satellite.append(result)

    n_reentry_test = tp + fn
    n_operational_test = fp + tn
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, n_reentry_test)
    spec = tn / max(1, n_operational_test)  # specificity
    f1 = 2 * prec * rec / max(1e-10, prec + rec)

    # ── Confidence intervals ────────────────────────────────
    recall_ci_lo, recall_ci_hi = wilson_ci(tp, n_reentry_test)
    prec_ci_lo, prec_ci_hi = wilson_ci(tp, tp + fp) if (tp + fp) > 0 else (0.0, 0.0)
    spec_ci_lo, spec_ci_hi = wilson_ci(tn, n_operational_test)

    # F1 CI from recall and precision CIs
    if prec > 0 and rec > 0:
        f1_ci_lo = 2 * prec_ci_lo * recall_ci_lo / max(1e-10, prec_ci_lo + recall_ci_lo) if prec_ci_lo > 0 and recall_ci_lo > 0 else 0.0
        f1_ci_hi = 2 * prec_ci_hi * recall_ci_hi / max(1e-10, prec_ci_hi + recall_ci_hi) if prec_ci_hi > 0 and recall_ci_hi > 0 else 0.0
    else:
        f1_ci_lo, f1_ci_hi = 0.0, 0.0

    lt = np.array(lead_times) if lead_times else np.array([0.0])

    print(f"\nReentry detection: {tp}/{n_reentry_test} "
          f"(Recall={rec:.3f} [{recall_ci_lo:.3f}-{recall_ci_hi:.3f}])")
    print(f"Operational specificity: {tn}/{n_operational_test} "
          f"(Spec={spec:.3f} [{spec_ci_lo:.3f}-{spec_ci_hi:.3f}])")
    print(f"Precision={prec:.3f} [{prec_ci_lo:.3f}-{prec_ci_hi:.3f}]")
    print(f"F1={f1:.3f} [{f1_ci_lo:.3f}-{f1_ci_hi:.3f}]")
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"Lead time: mean={lt.mean():.0f}d, median={np.median(lt):.0f}d, "
          f"min={lt.min():.0f}d, max={lt.max():.0f}d, "
          f"p25={np.percentile(lt, 25):.0f}d, p75={np.percentile(lt, 75):.0f}d")

    # ── Save results ────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "config": config_snapshot(),
        "artifact_checksums": {
            "scaler": md5_file(SCALER_FILE),
            "lda": md5_file(LDA_FILE),
            "basin": md5_file(BASIN_FILE),
        },
        "test_set": {
            "n_reentry": n_reentry_test,
            "n_operational": n_operational_test,
            "n_total": n_reentry_test + n_operational_test,
            "n_windows": len(X_test),
        },
        "validation": {
            "v1_separation": v1_sep,
            "v1_p": v1_p,
            "v1_pass": v1_p < V1_P_VALUE_THRESHOLD,
            "v2_rho": v2_rho,
            "v2_p": v2_p,
            "v2_pass": v2_rho > V2_SPEARMAN_THRESHOLD,
        },
        "detection": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(prec, 4),
            "precision_ci_95": [round(prec_ci_lo, 4), round(prec_ci_hi, 4)],
            "recall": round(rec, 4),
            "recall_ci_95": [round(recall_ci_lo, 4), round(recall_ci_hi, 4)],
            "specificity": round(spec, 4),
            "specificity_ci_95": [round(spec_ci_lo, 4), round(spec_ci_hi, 4)],
            "f1": round(f1, 4),
            "f1_ci_95": [round(f1_ci_lo, 4), round(f1_ci_hi, 4)],
        },
        "lead_time_days": {
            "mean": round(float(lt.mean()), 1),
            "median": round(float(np.median(lt)), 1),
            "min": round(float(lt.min()), 1),
            "max": round(float(lt.max()), 1),
            "p25": round(float(np.percentile(lt, 25)), 1),
            "p75": round(float(np.percentile(lt, 75)), 1),
        },
        "lead_times_raw": lead_times,
        "per_satellite": per_satellite,
    }

    outfile = RESULTS_DIR / "validate.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {outfile}")
    return results


if __name__ == "__main__":
    validate()
