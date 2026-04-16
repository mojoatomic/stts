"""
Matched-FPR lead time analysis.

Primary metric: at false-positive rate FPR_TARGET on healthy 320K baseline,
threshold both curvature and distance. On each held-out 450K trajectory,
measure the frame at which the signal first fires (sustained for
SUSTAINED_WINDOW consecutive above-threshold windows). Lead time =
failure_frame - first_sustained_alarm_frame (larger = earlier detection).

Supplementary metrics:
  - Spearman rho of signal vs RUL
  - Detection at fixed early point (pre-failure windows)

Usage:
    python -m geometry.lead_time
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from geometry.config import (
    FPR_TARGET,
    K_FINAL,
    RESULTS_DIR,
    SUSTAINED_WINDOW,
    config_snapshot,
)


def threshold_at_fpr(healthy_values: np.ndarray, fpr: float, higher_is_alarm: bool) -> float:
    """Find threshold on healthy baseline such that `fpr` fraction of healthy
    values would trigger an alarm.

    If higher_is_alarm: threshold = (1-fpr) percentile, alarm if signal > threshold.
    If lower_is_alarm:  threshold = fpr percentile, alarm if signal < threshold.
    """
    if higher_is_alarm:
        return float(np.quantile(healthy_values, 1.0 - fpr))
    return float(np.quantile(healthy_values, fpr))


def first_sustained_alarm(
    signal: np.ndarray,
    threshold: float,
    sustained: int,
    higher_is_alarm: bool,
) -> int | None:
    """Return the earliest index where signal crosses threshold AND stays
    on the alarm side for the next `sustained` consecutive windows (including
    the crossing window itself).
    """
    if higher_is_alarm:
        above = signal > threshold
    else:
        above = signal < threshold

    n = len(above)
    for i in range(n - sustained + 1):
        if above[i:i + sustained].all():
            return int(i)
    return None


def analyze_probe(probe_results: dict) -> dict:
    baseline = probe_results["baseline"]
    probes = probe_results["probes"]

    # Load the healthy baseline arrays from artifacts
    from geometry.config import ARTIFACTS_DIR
    healthy_curv = np.load(ARTIFACTS_DIR / "baseline_curvature.npy")
    healthy_dist = np.load(ARTIFACTS_DIR / "baseline_distance.npy")

    # Determine alarm direction:
    # Distance: higher = farther from healthy = alarm (higher_is_alarm)
    # Curvature: from sanity checks, kappa rises (toward +0.5) when a test
    # point is out of the manifold. So higher kappa = alarm.
    # Confirm this by comparing means of healthy vs one failing trajectory.
    dist_thresh = threshold_at_fpr(healthy_dist, FPR_TARGET, higher_is_alarm=True)
    curv_thresh = threshold_at_fpr(healthy_curv, FPR_TARGET, higher_is_alarm=True)

    print(f"Matched thresholds at FPR={FPR_TARGET:.2%}:")
    print(f"  distance  > {dist_thresh:.4f}  (healthy mean={healthy_dist.mean():.4f})")
    print(f"  curvature > {curv_thresh:.4f}  (healthy mean={healthy_curv.mean():.4f})")
    print()

    per_replicate = []
    for probe in probes:
        rep = probe["replicate"]
        ff = probe["failure_frame"]
        ends = np.array(probe["end_frames"])
        rul = np.array(probe["rul"])
        curv = np.array(probe["curvature"])
        dist = np.array(probe["distance"])

        # first sustained alarms
        curv_first = first_sustained_alarm(curv, curv_thresh, SUSTAINED_WINDOW, True)
        dist_first = first_sustained_alarm(dist, dist_thresh, SUSTAINED_WINDOW, True)

        # Lead time: frames between first alarm and failure
        # If alarm fires after failure, lead time is negative (detection lag)
        # If alarm never fires, lead time is None
        if ff is not None and curv_first is not None:
            curv_lead_frames = ff - int(ends[curv_first])
        else:
            curv_lead_frames = None

        if ff is not None and dist_first is not None:
            dist_lead_frames = ff - int(ends[dist_first])
        else:
            dist_lead_frames = None

        # Spearman vs RUL (pre-failure windows only, where rul is finite and > 0)
        pre_failure_mask = (rul > 0) & np.isfinite(rul)
        if pre_failure_mask.sum() > 5:
            curv_rho, curv_p = spearmanr(rul[pre_failure_mask], curv[pre_failure_mask])
            dist_rho, dist_p = spearmanr(rul[pre_failure_mask], dist[pre_failure_mask])
            curv_rho, curv_p = float(curv_rho), float(curv_p)
            dist_rho, dist_p = float(dist_rho), float(dist_p)
        else:
            curv_rho = curv_p = dist_rho = dist_p = None

        per_replicate.append({
            "replicate": rep,
            "failure_frame": ff,
            "curv_first_alarm_window": curv_first,
            "curv_first_alarm_frame": int(ends[curv_first]) if curv_first is not None else None,
            "curv_lead_frames": curv_lead_frames,
            "dist_first_alarm_window": dist_first,
            "dist_first_alarm_frame": int(ends[dist_first]) if dist_first is not None else None,
            "dist_lead_frames": dist_lead_frames,
            "curv_spearman_rho_vs_rul": curv_rho,
            "curv_spearman_p": curv_p,
            "dist_spearman_rho_vs_rul": dist_rho,
            "dist_spearman_p": dist_p,
        })

        print(f"rep {rep}: failure_frame={ff}")
        print(f"    curvature: first alarm at frame "
              f"{int(ends[curv_first]) if curv_first is not None else 'NEVER':>5}, "
              f"lead = {curv_lead_frames if curv_lead_frames is not None else 'N/A'} frames, "
              f"rho vs RUL = {curv_rho:+.3f}" if curv_rho is not None else "")
        print(f"    distance:  first alarm at frame "
              f"{int(ends[dist_first]) if dist_first is not None else 'NEVER':>5}, "
              f"lead = {dist_lead_frames if dist_lead_frames is not None else 'N/A'} frames, "
              f"rho vs RUL = {dist_rho:+.3f}" if dist_rho is not None else "")
        print()

    # Aggregate
    curv_leads = [r["curv_lead_frames"] for r in per_replicate
                  if r["curv_lead_frames"] is not None]
    dist_leads = [r["dist_lead_frames"] for r in per_replicate
                  if r["dist_lead_frames"] is not None]

    return {
        "thresholds": {
            "curvature_threshold": curv_thresh,
            "distance_threshold": dist_thresh,
            "fpr_target": FPR_TARGET,
            "sustained_window": SUSTAINED_WINDOW,
        },
        "per_replicate": per_replicate,
        "summary": {
            "mean_curvature_lead_frames": float(np.mean(curv_leads)) if curv_leads else None,
            "mean_distance_lead_frames": float(np.mean(dist_leads)) if dist_leads else None,
            "median_curvature_lead_frames": float(np.median(curv_leads)) if curv_leads else None,
            "median_distance_lead_frames": float(np.median(dist_leads)) if dist_leads else None,
            "n_replicates": len(per_replicate),
            "curvature_fires_first_count": sum(
                1 for r in per_replicate
                if r["curv_lead_frames"] is not None
                and r["dist_lead_frames"] is not None
                and r["curv_lead_frames"] > r["dist_lead_frames"]
            ),
            "distance_fires_first_count": sum(
                1 for r in per_replicate
                if r["curv_lead_frames"] is not None
                and r["dist_lead_frames"] is not None
                and r["dist_lead_frames"] > r["curv_lead_frames"]
            ),
            "tied_count": sum(
                1 for r in per_replicate
                if r["curv_lead_frames"] is not None
                and r["dist_lead_frames"] is not None
                and r["curv_lead_frames"] == r["dist_lead_frames"]
            ),
        },
    }


def main():
    with open(RESULTS_DIR / "probe.json") as f:
        probe_results = json.load(f)

    analysis = analyze_probe(probe_results)

    summary = analysis["summary"]
    print("=" * 60)
    print("Matched-FPR lead time summary")
    print("=" * 60)
    print(f"  n replicates:                   {summary['n_replicates']}")
    print(f"  mean curvature lead (frames):   {summary['mean_curvature_lead_frames']}")
    print(f"  mean distance lead (frames):    {summary['mean_distance_lead_frames']}")
    print(f"  curvature fires first on:       {summary['curvature_fires_first_count']}/{summary['n_replicates']} replicates")
    print(f"  distance fires first on:        {summary['distance_fires_first_count']}/{summary['n_replicates']} replicates")
    print(f"  tied:                           {summary['tied_count']}/{summary['n_replicates']} replicates")
    print()

    analysis["config"] = config_snapshot()
    analysis["k_final"] = K_FINAL

    out_path = RESULTS_DIR / "lead_time.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
