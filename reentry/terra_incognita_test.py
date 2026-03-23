"""
TERRA_INCOGNITA evaluation: February 2022 geomagnetic storm holdout.

Tests whether the STTS framework correctly identifies storm-induced
reentries as outside the operational envelope (OOD). These satellites
decayed within ~2 weeks of launch due to elevated atmospheric density,
NOT deliberate deorbit maneuvers. Their decay trajectories should be
geometrically unlike the deliberate deorbit corpus.

This is a separate result — NOT included in the main F1 metric.

The scientific claim: the framework's OOD detection mechanism
(distance to full training corpus) should flag these objects as
TERRA_INCOGNITA, meaning "the corpus has never seen trajectories
like this." That's the correct operational answer — a system trained
on deliberate deorbits should not confidently predict storm-induced
reentries, and the framework should say so explicitly.

Usage:
    python reentry/terra_incognita_test.py

Requires: corpus built + model trained
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from reentry.config import (
    K_NEIGHBORS,
    OOD_PERCENTILE,
    RESULTS_DIR,
    WINDOW_SIZE,
    WINDOW_STRIDE_EVAL,
    config_snapshot,
)
from reentry.corpus import load_corpus
from reentry.features import build_feature_matrix
from reentry.train import load_model, md5_file, SCALER_FILE, LDA_FILE, BASIN_FILE


def terra_incognita_test():
    """Evaluate storm holdout for OOD detection."""
    model = load_model()
    corpus = load_corpus()
    satellites = corpus["satellites"]
    storm_ids = corpus["storm_ids"]
    train_ids = corpus["train_ids"]

    if not storm_ids:
        print("No storm objects in corpus. Skipping TERRA_INCOGNITA test.")
        return None

    print(f"TERRA_INCOGNITA evaluation: {len(storm_ids)} storm objects")

    # ── Build training corpus projection for OOD baseline ───
    # We need the distribution of corpus distances to set the OOD threshold
    X_train, y_train, _, _ = build_feature_matrix(
        satellites, train_ids,
        window_size=WINDOW_SIZE,
        stride=5,  # coarser stride for efficiency
    )

    scaler = model["scaler"]
    W = model["W"]
    lda = model["lda"]

    # Project training data
    train_mask = y_train != -1
    X_train_clean = X_train[train_mask]
    X_ts = np.nan_to_num(scaler.transform(X_train_clean), nan=0.0, posinf=0.0, neginf=0.0)
    X_tw = X_ts * W
    train_proj = lda.transform(X_tw).ravel()

    # OOD threshold: distance to k-th nearest neighbor in training corpus
    # at the OOD_PERCENTILE of training distances
    k = K_NEIGHBORS

    def corpus_knn_dist(p):
        """Distance to k-th nearest neighbor in training corpus."""
        dists = np.abs(train_proj - p)
        return np.sort(dists)[min(k - 1, len(dists) - 1)]

    train_corpus_dists = np.array([corpus_knn_dist(p) for p in train_proj])
    ood_threshold = float(np.percentile(train_corpus_dists, OOD_PERCENTILE))
    print(f"  OOD threshold (p{OOD_PERCENTILE}): {ood_threshold:.4f}")

    # ── Extract storm features ──────────────────────────────
    X_storm, y_storm, days_storm, ids_storm = build_feature_matrix(
        satellites, storm_ids,
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE_EVAL,
    )

    if len(X_storm) == 0:
        print("  No valid storm windows (insufficient TLE records)")
        return None

    print(f"  Storm windows: {len(X_storm)}")

    # Project storm data through frozen model
    X_ss = np.nan_to_num(scaler.transform(X_storm), nan=0.0, posinf=0.0, neginf=0.0)
    X_sw = X_ss * W
    storm_proj = lda.transform(X_sw).ravel()

    # Compute OOD distances
    storm_corpus_dists = np.array([corpus_knn_dist(p) for p in storm_proj])

    # Also compute basin distances for comparison
    dist_to_basin = model["dist_to_basin"]
    storm_basin_dists = np.array([dist_to_basin(p) for p in storm_proj])

    # ── Per-object OOD analysis ─────────────────────────────
    n_ood = 0
    n_total = 0
    per_object = []

    for nid in storm_ids:
        sat = satellites[nid]
        sat_mask = np.array([wid.startswith(f"{nid}:") for wid in ids_storm])

        if sat_mask.sum() == 0:
            continue

        n_total += 1
        sat_ood_dists = storm_corpus_dists[sat_mask]
        sat_basin_dists = storm_basin_dists[sat_mask]

        # Object is OOD if ANY window exceeds the threshold
        # (conservative: even one OOD window flags the trajectory)
        max_ood_dist = float(np.max(sat_ood_dists))
        mean_ood_dist = float(np.mean(sat_ood_dists))
        is_ood = max_ood_dist > ood_threshold
        frac_ood = float((sat_ood_dists > ood_threshold).mean())

        if is_ood:
            n_ood += 1

        per_object.append({
            "norad_id": nid,
            "object_name": sat["object_name"],
            "decay_epoch": sat["decay_epoch"],
            "n_windows": int(sat_mask.sum()),
            "max_corpus_dist": max_ood_dist,
            "mean_corpus_dist": mean_ood_dist,
            "frac_windows_ood": frac_ood,
            "is_terra_incognita": is_ood,
            "mean_basin_dist": float(np.mean(sat_basin_dists)),
        })

    ood_rate = n_ood / max(1, n_total)

    print(f"\n  TERRA_INCOGNITA detection:")
    print(f"    Storm objects evaluated: {n_total}")
    print(f"    Flagged as OOD: {n_ood}/{n_total} ({ood_rate:.1%})")
    print(f"    OOD threshold: {ood_threshold:.4f}")

    # Distribution comparison
    print(f"\n  Corpus distance distribution:")
    print(f"    Training (nominal+precursor): "
          f"mean={np.mean(train_corpus_dists):.4f}, "
          f"p95={np.percentile(train_corpus_dists, 95):.4f}")
    print(f"    Storm holdout:               "
          f"mean={np.mean(storm_corpus_dists):.4f}, "
          f"p95={np.percentile(storm_corpus_dists, 95):.4f}")

    # ── Save results ────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "config": config_snapshot(),
        "artifact_checksums": {
            "scaler": md5_file(SCALER_FILE),
            "lda": md5_file(LDA_FILE),
            "basin": md5_file(BASIN_FILE),
        },
        "ood_detection": {
            "n_storm_objects": n_total,
            "n_flagged_ood": n_ood,
            "ood_rate": round(ood_rate, 4),
            "ood_threshold": ood_threshold,
            "ood_percentile": OOD_PERCENTILE,
        },
        "distance_distributions": {
            "training_mean": round(float(np.mean(train_corpus_dists)), 4),
            "training_p95": round(float(np.percentile(train_corpus_dists, 95)), 4),
            "storm_mean": round(float(np.mean(storm_corpus_dists)), 4),
            "storm_p95": round(float(np.percentile(storm_corpus_dists, 95)), 4),
        },
        "per_object": per_object,
    }

    outfile = RESULTS_DIR / "terra_incognita.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {outfile}")
    return results


if __name__ == "__main__":
    terra_incognita_test()
