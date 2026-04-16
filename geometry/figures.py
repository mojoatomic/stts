"""Figures for the curvature experiment. All plots 300 DPI, STTS style."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geometry.config import ARTIFACTS_DIR, FIG_DIR, FPR_TARGET, K_FINAL, RESULTS_DIR


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "probe.json") as f:
        probe = json.load(f)
    with open(RESULTS_DIR / "lead_time.json") as f:
        lead = json.load(f)

    healthy_curv = np.load(ARTIFACTS_DIR / "baseline_curvature.npy")
    healthy_dist = np.load(ARTIFACTS_DIR / "baseline_distance.npy")

    curv_thresh = lead["thresholds"]["curvature_threshold"]
    dist_thresh = lead["thresholds"]["distance_threshold"]

    # --- Figure 1: Healthy baseline distributions with matched FPR thresholds ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.hist(healthy_curv, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
    ax.axvline(curv_thresh, color="firebrick", linestyle="--",
               label=f"FPR={FPR_TARGET:.0%} threshold\n({curv_thresh:.3f})")
    ax.set_xlabel("Ollivier-Ricci curvature (local)")
    ax.set_ylabel("Count")
    ax.set_title(f"Healthy 320K baseline — curvature (k={K_FINAL})")
    ax.legend()

    ax = axes[1]
    ax.hist(healthy_dist, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
    ax.axvline(dist_thresh, color="firebrick", linestyle="--",
               label=f"FPR={FPR_TARGET:.0%} threshold\n({dist_thresh:.3f})")
    ax.set_xlabel("Euclidean distance to nearest 320K window")
    ax.set_ylabel("Count")
    ax.set_title("Healthy 320K baseline — distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "healthy_baseline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'healthy_baseline.png'}")

    # --- Figure 2: Per-replicate signal trajectories ---
    fig, axes = plt.subplots(5, 2, figsize=(12, 14), sharex=True)
    for row, p in enumerate(probe["probes"]):
        rep = p["replicate"]
        ff = p["failure_frame"]
        ends = np.array(p["end_frames"])
        curv = np.array(p["curvature"])
        dist = np.array(p["distance"])

        ax = axes[row, 0]
        ax.plot(ends, curv, color="steelblue", linewidth=0.8)
        ax.axhline(curv_thresh, color="firebrick", linestyle="--", linewidth=0.7)
        if ff is not None:
            ax.axvline(ff, color="black", linestyle=":", linewidth=0.8,
                       label=f"failure @ frame {ff}")
        ax.set_ylabel(f"rep {rep}\ncurvature")
        if row == 0:
            ax.set_title("Ollivier-Ricci curvature along trajectory")
        ax.legend(loc="lower right", fontsize=7)

        ax = axes[row, 1]
        ax.plot(ends, dist, color="steelblue", linewidth=0.8)
        ax.axhline(dist_thresh, color="firebrick", linestyle="--", linewidth=0.7)
        if ff is not None:
            ax.axvline(ff, color="black", linestyle=":", linewidth=0.8,
                       label=f"failure @ frame {ff}")
        ax.set_ylabel(f"rep {rep}\ndistance")
        if row == 0:
            ax.set_title("Euclidean distance along trajectory")
        ax.legend(loc="lower right", fontsize=7)

    axes[-1, 0].set_xlabel("frame (ns)")
    axes[-1, 1].set_xlabel("frame (ns)")
    fig.suptitle(
        f"Curvature vs distance — held-out 450K replicates (k={K_FINAL}, FPR={FPR_TARGET:.0%})",
        y=1.00, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "signal_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'signal_trajectories.png'}")

    # --- Figure 3: Lead time comparison ---
    per = lead["per_replicate"]
    reps = [r["replicate"] for r in per]
    curv_leads = [r["curv_lead_frames"] if r["curv_lead_frames"] is not None else 0 for r in per]
    dist_leads = [r["dist_lead_frames"] if r["dist_lead_frames"] is not None else 0 for r in per]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    x = np.arange(len(reps))
    w = 0.35
    ax.bar(x - w/2, curv_leads, w, label="Curvature lead", color="firebrick", alpha=0.7)
    ax.bar(x + w/2, dist_leads, w, label="Distance lead", color="steelblue", alpha=0.7)
    ax.set_xlabel("replicate (450K)")
    ax.set_ylabel("lead time (frames = ns)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"rep {r}" for r in reps])
    ax.set_title(f"Lead time at matched FPR={FPR_TARGET:.0%}")
    ax.legend()

    mean_curv = lead["summary"]["mean_curvature_lead_frames"]
    mean_dist = lead["summary"]["mean_distance_lead_frames"]
    ax.text(0.98, 0.02,
            f"mean curv lead: {mean_curv:.1f}\nmean dist lead: {mean_dist:.1f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round", fc="wheat", alpha=0.6))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lead_time_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'lead_time_comparison.png'}")

    # --- Figure 4: Curvature vs distance scatter (are they essentially the same?) ---
    all_curv = []
    all_dist = []
    for p in probe["probes"]:
        all_curv.extend(p["curvature"])
        all_dist.extend(p["distance"])
    all_curv = np.array(all_curv)
    all_dist = np.array(all_dist)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(all_dist, all_curv, s=3, alpha=0.2, color="steelblue")
    ax.axhline(curv_thresh, color="firebrick", linestyle="--", linewidth=0.8,
               label=f"curv threshold ({curv_thresh:.3f})")
    ax.axvline(dist_thresh, color="firebrick", linestyle="-.", linewidth=0.8,
               label=f"dist threshold ({dist_thresh:.3f})")
    ax.set_xlabel("Euclidean distance to healthy")
    ax.set_ylabel("Ollivier-Ricci curvature")
    ax.set_title("Curvature vs distance — all 450K test windows")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "curvature_vs_distance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'curvature_vs_distance.png'}")


if __name__ == "__main__":
    main()
