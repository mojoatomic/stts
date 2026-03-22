"""
Evaluate the frozen canonical model on the held-out test set.

Loads serialized artifacts — never calls .fit().
Reports F1, precision, recall, lead times, and confidence intervals.
Verifies no data leakage (designation-level check).

Usage:
    python3 validate.py

Requires: corpus built + model trained
"""

import os
import json
import numpy as np

import config
from corpus import load_corpus
from train import load_model, md5_file
from horizons_stts_pipeline import extract_features


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
    """Evaluate frozen model on test set. Save results with config snapshot."""
    model = load_model()
    corpus = load_corpus()
    trajectories = corpus["trajectories"]
    test_idx = corpus["test_idx"]

    # ── Leakage verification ──────────────────────────────
    with open(config.TRAIN_DESIGNATIONS_FILE) as f:
        train_des = set(json.load(f))

    test_des = [trajectories[i]["designation"] for i in test_idx]
    for d in test_des:
        assert d not in train_des, f"LEAKAGE: {d} in both train and test"
    print(f"Leakage check passed: {len(test_des)} test objects, "
          f"0 overlap with {len(train_des)} training objects")

    # Verify Apophis not in either set
    for d in list(train_des) + test_des:
        assert config.APOPHIS_DESIGNATION not in d, \
            f"Apophis ({d}) found in corpus"

    # ── Evaluate ──────────────────────────────────────────
    scaler = model["scaler"]
    W = model["W"]
    lda = model["lda"]
    eps = model["epsilon"]
    dist_to_basin = model["dist_to_basin"]

    tp = fp = fn = 0
    lead_times = []

    for idx in test_idx:
        traj = trajectories[idx]
        elements = traj["elements"]
        ca_jd = traj["ca_jd"]

        if len(elements) < config.WINDOW_DAYS:
            continue

        fired = False
        fire_rul = None

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
                fired = True
                fire_rul = rul
                break

        if fired and fire_rul and fire_rul > 0:
            tp += 1
            lead_times.append(float(fire_rul))
        else:
            fn += 1

    n_test = tp + fn
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-10, prec + rec)

    # ── Confidence intervals ──────────────────────────────
    # F1 = 2*recall/(1+recall) when precision=1.0 (0 FP)
    # CI on F1 is monotonic transform of CI on recall
    recall_ci_lo, recall_ci_hi = wilson_ci(tp, n_test)
    if fp == 0:
        f1_ci_lo = 2 * recall_ci_lo / (1 + recall_ci_lo) if recall_ci_lo > 0 else 0.0
        f1_ci_hi = 2 * recall_ci_hi / (1 + recall_ci_hi) if recall_ci_hi > 0 else 0.0
    else:
        # With FP > 0, use direct Wilson on F1 as approximation
        f1_ci_lo, f1_ci_hi = wilson_ci(tp, n_test)

    lt = np.array(lead_times) if lead_times else np.array([0.0])

    print(f"\nDetection: {tp}/{n_test} "
          f"(F1={f1:.3f} [{f1_ci_lo:.3f}–{f1_ci_hi:.3f}])")
    print(f"Precision={prec:.3f}, Recall={rec:.3f} "
          f"[{recall_ci_lo:.3f}–{recall_ci_hi:.3f}]")
    print(f"FP={fp}, FN={fn}")
    print(f"Lead time: mean={lt.mean():.0f}d, median={np.median(lt):.0f}d, "
          f"min={lt.min():.0f}d, max={lt.max():.0f}d")

    # ── Save results ──────────────────────────────────────
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    results = {
        "config": config.config_snapshot(),
        "artifact_checksums": {
            "scaler": md5_file(config.SCALER_FILE),
            "lda": md5_file(config.LDA_FILE),
            "basin": md5_file(config.BASIN_FILE),
        },
        "test_set": {
            "n_objects": n_test,
            "designations_file": config.TEST_DESIGNATIONS_FILE,
        },
        "detection": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "f1_ci_95": [round(f1_ci_lo, 4), round(f1_ci_hi, 4)],
            "recall_ci_95": [round(recall_ci_lo, 4), round(recall_ci_hi, 4)],
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
    }

    outfile = f"{config.RESULTS_DIR}/validate.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {outfile}")
    return results


if __name__ == "__main__":
    validate()
