"""
STTS Energy State — Aggregate across full reentry test manifest.

Runs the existing H(t), p(t), P(failure, t+Δt) extension against every
test reentry object. Produces per-object CSVs, aggregate summary,
population plots, and framework performance report.

Does NOT modify the Markov table, retrain, or change any computation.
Loads frozen model + Markov artifacts. NEVER calls .fit().

Usage:
    python reentry/energy_state_aggregate.py

Requires: corpus built + model trained (python reentry/run_all.py)
"""

import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy as scipy_entropy

sys.path.insert(0, str(Path(__file__).parent.parent))

from reentry.config import (
    ENTROPY_THRESHOLD,
    ENTROPY_WINDOW,
    K_NEIGHBORS,
    MARKOV_K,
    MC_HORIZON,
    MC_N_SAMPLES,
    RESULTS_DIR,
    WINDOW_SIZE,
    WINDOW_STRIDE_EVAL,
    WINDOW_STRIDE_TRAIN,
)
from reentry.corpus import load_corpus
from reentry.features import build_feature_matrix
from reentry.train import load_model
from reentry.validate import wilson_ci

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PER_OBJECT_DIR = RESULTS_DIR / "per_object"
PLOTS_DIR = RESULTS_DIR / "plots"
AGGREGATE_CSV = RESULTS_DIR / "aggregate_summary.csv"
RESPONSE_FILE = RESULTS_DIR / "response_distribution.txt"

P_THRESHOLDS = [0.1, 0.25, 0.5, 0.75]


def _project(X, scaler, W, lda):
    """Project raw features through frozen F->W->M pipeline. Returns 1D."""
    X_s = np.nan_to_num(scaler.transform(X), nan=0.0, posinf=0.0, neginf=0.0)
    return lda.transform(X_s * W).ravel()


def run_single_object(nid, satellites, scaler, W, lda, kmeans,
                      transition_matrix, cum_trans, failure_arr,
                      mu_nominal, expected_entropy, train_proj, y_tr,
                      k, rng):
    """Run energy state extension for one satellite. Returns summary dict or None."""
    sat = satellites[nid]
    decay_epoch = sat.get("decay_epoch")

    X_sat, y_sat, days_sat, ids_sat = build_feature_matrix(
        satellites, [nid],
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE_EVAL,
    )

    if len(X_sat) == 0:
        return None

    M2 = _project(X_sat, scaler, W, lda)
    n_windows = len(M2)

    # Sort chronologically (days_to_reentry descending = earliest first)
    order = np.argsort(days_sat)[::-1]
    M2 = M2[order]
    days_sat = days_sat[order]

    if not np.all(np.isfinite(days_sat)):
        return None

    M2_cells = kmeans.predict(M2.reshape(-1, 1))

    # H(t)
    V = np.abs(M2 - mu_nominal)
    T = np.zeros(n_windows)
    T[1:] = 0.5 * (M2[1:] - M2[:-1]) ** 2
    H = T + V

    # p(t)
    p_t = np.zeros(n_windows)
    for t in range(n_windows):
        dists = np.abs(train_proj - M2[t])
        nn_idx = np.argpartition(dists, k)[:k]
        p_t[t] = np.mean(y_tr[nn_idx] == 1)

    # Signal separation
    artifact_flags = np.zeros(n_windows, dtype=int)
    for t in range(ENTROPY_WINDOW, n_windows):
        dest_cells = M2_cells[t - ENTROPY_WINDOW + 1: t + 1]
        dest_counts = np.bincount(dest_cells, minlength=MARKOV_K).astype(float)
        dest_counts /= dest_counts.sum()
        obs_H = scipy_entropy(dest_counts)
        cur_cell = M2_cells[t]
        exp_H = expected_entropy[cur_cell]
        if exp_H < 1e-10:
            if obs_H > 1e-10:
                artifact_flags[t] = 1
        elif obs_H > ENTROPY_THRESHOLD * exp_H:
            artifact_flags[t] = 1

    # P(failure, t+Δt)
    P_forward = np.zeros(n_windows)
    for t in range(n_windows):
        states = np.full(MC_N_SAMPLES, M2_cells[t], dtype=int)
        for step in range(MC_HORIZON):
            u = rng.random(MC_N_SAMPLES)
            cdfs = cum_trans[states]
            states = (cdfs < u[:, np.newaxis]).sum(axis=1)
            np.clip(states, 0, MARKOV_K - 1, out=states)
        P_forward[t] = np.mean(np.isin(states, failure_arr))

    # Per-object CSV
    PER_OBJECT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PER_OBJECT_DIR / f"{nid}.csv"
    with open(csv_path, "w") as f:
        f.write("t,days_to_reentry,M2_cell,H_t,p_t,P_forward,artifact_flag\n")
        for t in range(n_windows):
            f.write(
                f"{t},{days_sat[t]:.2f},{M2_cells[t]},"
                f"{H[t]:.6f},{p_t[t]:.4f},{P_forward[t]:.6f},"
                f"{artifact_flags[t]}\n"
            )

    # Summary stats
    n_flags = int(artifact_flags.sum())

    # Lead times: windows before reentry where P(failure) first exceeds threshold
    # days_sat is sorted descending (earliest first), so final window is closest to reentry
    lead_times = {}
    for thresh in P_THRESHOLDS:
        crossed = np.where(P_forward >= thresh)[0]
        if len(crossed) > 0:
            first_idx = crossed[0]
            lead_times[thresh] = float(days_sat[first_idx])
        else:
            lead_times[thresh] = None

    return {
        "norad_id": nid,
        "object_name": sat.get("object_name", ""),
        "decay_epoch": decay_epoch,
        "n_windows": n_windows,
        "H_max": float(H.max()),
        "H_mean": float(H.mean()),
        "H_final": float(H[-1]),
        "p_max": float(p_t.max()),
        "p_mean": float(p_t.mean()),
        "p_final": float(p_t[-1]),
        "P_forward_max": float(P_forward.max()),
        "P_forward_mean": float(P_forward.mean()),
        "P_forward_final": float(P_forward[-1]),
        "artifact_count": n_flags,
        "artifact_pct": round(100.0 * n_flags / n_windows, 2),
        "lead_0.10": lead_times[0.1],
        "lead_0.25": lead_times[0.25],
        "lead_0.50": lead_times[0.5],
        "lead_0.75": lead_times[0.75],
    }


def main():
    # ── Load frozen model and corpus ───────────────────────────
    model = load_model()
    corpus = load_corpus()

    scaler = model["scaler"]
    W = model["W"]
    lda = model["lda"]
    k = K_NEIGHBORS

    if model["kmeans"] is None or model["markov"] is None:
        log.error("Markov artifacts not found. Retrain: python reentry/train.py")
        sys.exit(1)

    kmeans = model["kmeans"]
    markov = model["markov"]
    transition_matrix = markov["transition_matrix"]
    failure_cells = markov["failure_cells"]
    failure_arr = np.array(sorted(set(failure_cells.tolist())))
    mu_nominal = float(markov["mu_nominal"].item())
    expected_entropy = markov["expected_entropy"]
    train_proj = markov["train_proj"]
    y_tr = markov["train_labels"]

    cum_trans = np.cumsum(transition_matrix, axis=1)

    satellites = corpus["satellites"]
    test_reentry_ids = corpus.get("test_reentry_ids", [])

    # ══════════════════════════════════════════════════════════
    # Step 1 — Enumerate test objects
    # ══════════════════════════════════════════════════════════
    log.info("Step 1: Enumerating test reentry objects...")
    log.info("  Test reentry manifest: %d objects", len(test_reentry_ids))

    valid_ids = []
    skipped = []
    for nid in test_reentry_ids:
        sat = satellites.get(nid)
        if sat is None:
            skipped.append((nid, "not in corpus"))
            continue
        records = sat.get("tle_records", [])
        if len(records) < WINDOW_SIZE:
            skipped.append((nid, f"insufficient TLEs ({len(records)} < {WINDOW_SIZE})"))
            continue
        if not sat.get("decay_epoch"):
            skipped.append((nid, "no decay_epoch"))
            continue
        valid_ids.append(nid)

    log.info("  Valid: %d", len(valid_ids))
    log.info("  Skipped: %d", len(skipped))
    for nid, reason in skipped:
        log.info("    %s: %s", nid, reason)

    # ══════════════════════════════════════════════════════════
    # Step 2 + 3 — Per-object run + summary
    # ══════════════════════════════════════════════════════════
    log.info("\nStep 2-3: Running extension on %d objects...", len(valid_ids))
    rng = np.random.RandomState(42)
    summaries = []

    for i, nid in enumerate(valid_ids):
        result = run_single_object(
            nid, satellites, scaler, W, lda, kmeans,
            transition_matrix, cum_trans, failure_arr,
            mu_nominal, expected_entropy, train_proj, y_tr,
            k, rng,
        )
        if result is None:
            skipped.append((nid, "no windows after feature extraction"))
            log.info("  [%d/%d] %s — SKIPPED (no windows)", i + 1, len(valid_ids), nid)
            continue

        summaries.append(result)
        log.info("  [%d/%d] %s (%s) — %d windows, max P=%.4f",
                 i + 1, len(valid_ids), nid,
                 result["object_name"], result["n_windows"],
                 result["P_forward_max"])

    log.info("\n  Completed: %d objects", len(summaries))

    # ── Write aggregate CSV ────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "norad_id", "object_name", "decay_epoch", "n_windows",
        "H_max", "H_mean", "H_final",
        "p_max", "p_mean", "p_final",
        "P_forward_max", "P_forward_mean", "P_forward_final",
        "artifact_count", "artifact_pct",
        "lead_0.10", "lead_0.25", "lead_0.50", "lead_0.75",
    ]
    with open(AGGREGATE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)
    log.info("  Aggregate CSV: %s", AGGREGATE_CSV)

    # ══════════════════════════════════════════════════════════
    # Step 4 — Population plots
    # ══════════════════════════════════════════════════════════
    log.info("\nStep 4: Population plots...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    max_P = np.array([s["P_forward_max"] for s in summaries])
    max_H = np.array([s["H_max"] for s in summaries])
    n_wins = np.array([s["n_windows"] for s in summaries])
    lead_50 = [s["lead_0.50"] for s in summaries]

    # Plot 1: Distribution of max P(failure)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(max_P, bins=30, color="darkred", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Max P(failure, t+\u0394t)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Max P(failure) — {len(summaries)} test reentry objects")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "max_P_forward_distribution.png", dpi=150)
    plt.close(fig)

    # Plot 2: Lead time at 0.5 threshold
    crossed_50 = [v for v in lead_50 if v is not None]
    never_50 = sum(1 for v in lead_50 if v is None)
    fig, ax = plt.subplots(figsize=(10, 6))
    if crossed_50:
        ax.hist(crossed_50, bins=30, color="steelblue", edgecolor="black",
                alpha=0.8, label=f"Crossed 0.5 (n={len(crossed_50)})")
    # "Never crossed" bar at x=-50 (off the natural range)
    if never_50 > 0:
        ax.bar(-50, never_50, width=30, color="gray", edgecolor="black",
               alpha=0.8, label=f"Never crossed (n={never_50})")
    ax.set_xlabel("Days before reentry at first P(failure) \u2265 0.5")
    ax.set_ylabel("Count")
    ax.set_title(f"Lead Time Distribution at P(failure) \u2265 0.5 — {len(summaries)} objects")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lead_time_050_distribution.png", dpi=150)
    plt.close(fig)

    # Plot 3: Distribution of max H(t)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(max_H, bins=30, color="darkblue", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Max H(t)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Max H(t) — {len(summaries)} test reentry objects")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "max_H_distribution.png", dpi=150)
    plt.close(fig)

    # Plot 4: Scatter — max P(failure) vs window count
    crossed_mask = np.array([v is not None for v in lead_50])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(n_wins[crossed_mask], max_P[crossed_mask],
               c="darkred", alpha=0.6, s=30, label="Crossed 0.5", zorder=3)
    ax.scatter(n_wins[~crossed_mask], max_P[~crossed_mask],
               c="gray", alpha=0.6, s=30, label="Never crossed 0.5", zorder=3)
    ax.set_xlabel("Window count")
    ax.set_ylabel("Max P(failure, t+\u0394t)")
    ax.set_title(f"Max P(failure) vs Window Count — {len(summaries)} objects")
    ax.axhline(y=0.5, color="red", ls="--", lw=1, alpha=0.5, label="0.5 threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "max_P_vs_windows.png", dpi=150)
    plt.close(fig)

    log.info("  Plots saved to %s/", PLOTS_DIR)

    # ══════════════════════════════════════════════════════════
    # Step 5 — Response distribution
    # ══════════════════════════════════════════════════════════
    log.info("\nStep 5: Response distribution...")

    never_01 = sum(1 for s in summaries if s["P_forward_max"] < 0.1)
    crossed_05_before_final = 0
    crossed_05_at_or_after_final = 0
    never_05 = 0

    for s in summaries:
        lead = s["lead_0.50"]
        if lead is None:
            never_05 += 1
        elif lead > 0:
            crossed_05_before_final += 1
        else:
            crossed_05_at_or_after_final += 1

    n = len(summaries)
    lines = [
        "STTS Energy State — Response Distribution",
        f"Test reentry objects: {n}",
        "",
        "P(failure) never exceeded 0.1:",
        f"  {never_01} / {n} ({100*never_01/n:.1f}%)",
        "  Framework stayed quiet.",
        "",
        "P(failure) reached 0.5 before final window:",
        f"  {crossed_05_before_final} / {n} ({100*crossed_05_before_final/n:.1f}%)",
        "  Framework crossed threshold with lead time.",
        "",
        "P(failure) only crossed 0.5 at or after final window:",
        f"  {crossed_05_at_or_after_final} / {n} ({100*crossed_05_at_or_after_final/n:.1f}%)",
        "  Framework crossed threshold late.",
        "",
        f"P(failure) never reached 0.5:",
        f"  {never_05} / {n} ({100*never_05/n:.1f}%)",
        "",
        "Skipped objects (incomplete data):",
        f"  {len(skipped)}",
    ]
    for nid, reason in skipped:
        lines.append(f"    {nid}: {reason}")

    report = "\n".join(lines) + "\n"

    with open(RESPONSE_FILE, "w") as f:
        f.write(report)

    log.info("  %s", RESPONSE_FILE)
    log.info("")
    for line in lines:
        log.info("  %s", line)

    log.info("\nDone.")


if __name__ == "__main__":
    main()
