"""
STTS Energy State and Markov Monte Carlo Forward Prediction.

Additive extension to the reentry validation. Computes:
  H(t)              — energy state (drift velocity + basin distance)
  p(t)              — current basin membership probability
  P(failure, t+Dt)  — forward failure probability via Markov MC

Loads the frozen model and Markov artifacts. NEVER calls .fit().
All model fitting is in train.py.

Usage:
    python reentry/energy_state.py

Requires: corpus built + model trained (python reentry/run_all.py)
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from reentry.config import (
    ARTIFACTS_DIR,
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
    config_snapshot,
)
from reentry.corpus import load_corpus
from reentry.features import build_feature_matrix
from reentry.train import (
    load_model, md5_file,
    SCALER_FILE, LDA_FILE, BASIN_FILE, KMEANS_FILE, MARKOV_TABLE_FILE,
)
from reentry.validate import wilson_ci

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RESULTS_CSV = RESULTS_DIR / "energy_state.csv"
RESULTS_PLOT = RESULTS_DIR / "energy_state.png"
RESULTS_JSON = RESULTS_DIR / "energy_state.json"

# ── Metric Semantics ──────────────────────────────────────────
# Required by scientific-rigor: for every reported metric, state
# exactly what question it answers and what it does NOT answer.

METRIC_SEMANTICS = {
    "H_t": {
        "answers": (
            "How energetically displaced is the system from nominal "
            "at time t? Combines positional displacement V = |M2 - "
            "mu_nominal| and drift velocity T = 0.5*(M2(t) - M2(t-1))^2."
        ),
        "does_not_answer": (
            "Whether the system will fail. H is a state descriptor, "
            "not a predictor. High H can indicate displacement without "
            "failure (new operating regime) or rapid but benign drift."
        ),
    },
    "p_t": {
        "answers": (
            "What fraction of the system's k nearest neighbors in "
            "the full training corpus (nominal + precursor) are "
            "precursor (pre-failure) windows?"
        ),
        "does_not_answer": (
            "Future state. p(t) is a present-state measurement. It "
            "says where the system is now relative to known failure "
            "trajectories, not where it will be. With k=5, resolution "
            "is limited to {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}."
        ),
    },
    "P_forward": {
        "answers": (
            "What fraction of N Monte Carlo forward trajectories, "
            "sampled from the Markov transition table at horizon Dt, "
            "terminate in a failure basin cell?"
        ),
        "does_not_answer": (
            "Ground truth failure probability. P(failure) is "
            "conditional on the Markov model being correct — i.e., "
            "P9 (trajectory continuity) holding. If P9 is violated, "
            "the forward probability is unreliable. The Markov table "
            "is also built at training stride, not eval stride."
        ),
    },
    "artifact_flag": {
        "answers": (
            "Does the observed transition entropy at this timestep "
            "exceed the threshold multiple of expected entropy for "
            "the current Markov cell?"
        ),
        "does_not_answer": (
            "Whether the transition is definitely noise. High entropy "
            "relative to expectation is a signal for investigation, "
            "not a definitive noise classification. With a rolling "
            "window of 5, empirical entropy has high variance."
        ),
    },
}


def _project(X, scaler, W, lda):
    """Project raw features through the frozen F->W->M pipeline. Returns 1D."""
    X_s = np.nan_to_num(scaler.transform(X), nan=0.0, posinf=0.0, neginf=0.0)
    return lda.transform(X_s * W).ravel()


def main():
    # ── Load frozen model and corpus ───────────────────────────
    model = load_model()
    corpus = load_corpus()

    scaler = model["scaler"]
    W = model["W"]
    lda = model["lda"]
    epsilon = model["epsilon"]
    k = K_NEIGHBORS

    if model["kmeans"] is None or model["markov"] is None:
        log.error("Markov artifacts not found. Retrain: python reentry/train.py")
        sys.exit(1)

    kmeans = model["kmeans"]
    markov = model["markov"]
    transition_matrix = markov["transition_matrix"]
    failure_cells = markov["failure_cells"]
    failure_set = set(failure_cells.tolist())
    mu_nominal = float(markov["mu_nominal"].item())
    expected_entropy = markov["expected_entropy"]
    train_proj = markov["train_proj"]
    y_tr = markov["train_labels"]
    cell_centers = markov["cell_centers"]

    satellites = corpus["satellites"]
    test_reentry_ids = corpus.get("test_reentry_ids", [])

    log.info("Loaded model + Markov artifacts")
    log.info("  mu_nominal=%.6f  failure_cells=%s  k_cells=%d",
             mu_nominal, sorted(failure_set), MARKOV_K)
    log.info("  Training pool: %d windows (nominal=%d, precursor=%d)",
             len(train_proj), int((y_tr == 0).sum()), int((y_tr == 1).sum()))

    # ── Select test reentry satellite (most windows) ───────────
    log.info("\nSelecting test reentry satellite...")
    best_nid, best_n = None, 0
    for nid in test_reentry_ids:
        sat = satellites[nid]
        n_rec = len(sat.get("tle_records", []))
        n_win = max(0, n_rec - WINDOW_SIZE + 1)
        if n_win > best_n:
            best_n = n_win
            best_nid = nid

    if best_nid is None:
        log.error("No test reentry satellites found.")
        sys.exit(1)

    sat = satellites[best_nid]
    log.info("  NORAD %s  (%s)  decay=%s  ~%d windows",
             best_nid, sat.get("object_name"), sat.get("decay_epoch"), best_n)

    # ── Build test satellite features (stride=1) ───────────────
    log.info("\nExtracting test features (stride=1)...")
    X_sat, y_sat, days_sat, ids_sat = build_feature_matrix(
        satellites, [best_nid],
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE_EVAL,
    )

    M2 = _project(X_sat, scaler, W, lda)
    n_windows = len(M2)
    log.info("  Windows: %d", n_windows)

    # Sort chronologically (days_to_reentry descending = earliest first)
    order = np.argsort(days_sat)[::-1]
    M2 = M2[order]
    days_sat = days_sat[order]
    y_sat = y_sat[order]
    ids_sat = [ids_sat[i] for i in order]

    assert np.all(np.isfinite(days_sat)), "Non-finite days_to_reentry"

    # Assign to Markov cells (no .fit() — uses trained kmeans)
    M2_cells = kmeans.predict(M2.reshape(-1, 1))

    # ══════════════════════════════════════════════════════════
    # Step 1 — H(t)
    # H(t) = T + V
    #   V = |M2(t) - mu_nominal|  (distance from nominal)
    #   T = 0.5 * (M2(t) - M2(t-1))^2  (drift velocity)
    # ══════════════════════════════════════════════════════════
    log.info("\nStep 1: H(t)...")
    V = np.abs(M2 - mu_nominal)
    T = np.zeros(n_windows)
    T[1:] = 0.5 * (M2[1:] - M2[:-1]) ** 2
    H = T + V

    # ══════════════════════════════════════════════════════════
    # Step 2 — p(t)
    # k-NN against full training corpus. p(t) = fraction of k
    # neighbors that are precursor (y=1).
    # ══════════════════════════════════════════════════════════
    log.info("Step 2: p(t)...")
    p_t = np.zeros(n_windows)
    for t in range(n_windows):
        dists = np.abs(train_proj - M2[t])
        nn_idx = np.argpartition(dists, k)[:k]
        p_t[t] = np.mean(y_tr[nn_idx] == 1)

    assert np.all((p_t >= 0) & (p_t <= 1)), "p(t) out of [0, 1]"
    log.info("  p(t) range: [%.4f, %.4f]", p_t.min(), p_t.max())

    # ══════════════════════════════════════════════════════════
    # Step 4 — Signal separation
    # Rolling entropy over ENTROPY_WINDOW transitions. Flag
    # timesteps where observed entropy > ENTROPY_THRESHOLD x
    # expected entropy for the current cell.
    # ══════════════════════════════════════════════════════════
    log.info("Step 4: Signal separation (window=%d, threshold=%.1fx)...",
             ENTROPY_WINDOW, ENTROPY_THRESHOLD)
    artifact_flags = np.zeros(n_windows, dtype=int)
    from scipy.stats import entropy as scipy_entropy

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
                log.info("  t=%d cell=%d obs_entropy=%.4f (deterministic cell)",
                         t, cur_cell, obs_H)
        elif obs_H > ENTROPY_THRESHOLD * exp_H:
            artifact_flags[t] = 1
            log.info("  t=%d cell=%d obs=%.4f exp=%.4f (%.1fx)",
                     t, cur_cell, obs_H, exp_H, obs_H / exp_H)

    log.info("  Flagged: %d / %d", artifact_flags.sum(), n_windows)

    # ══════════════════════════════════════════════════════════
    # Step 5 — P(failure, t+Dt)
    # MC forward sampling from Markov table.
    # Each MC step = one training-stride transition.
    # ══════════════════════════════════════════════════════════
    log.info("Step 5: P(failure, t+Dt) — MC N=%d, Dt=%d ...",
             MC_N_SAMPLES, MC_HORIZON)

    rng = np.random.RandomState(42)
    P_forward = np.zeros(n_windows)
    P_forward_ci_lo = np.zeros(n_windows)
    P_forward_ci_hi = np.zeros(n_windows)

    cum_trans = np.cumsum(transition_matrix, axis=1)
    failure_arr = np.array(sorted(failure_set))

    for t in range(n_windows):
        states = np.full(MC_N_SAMPLES, M2_cells[t], dtype=int)
        for step in range(MC_HORIZON):
            u = rng.random(MC_N_SAMPLES)
            cdfs = cum_trans[states]
            states = (cdfs < u[:, np.newaxis]).sum(axis=1)
            np.clip(states, 0, MARKOV_K - 1, out=states)

        n_in_basin = int(np.isin(states, failure_arr).sum())
        P_forward[t] = n_in_basin / MC_N_SAMPLES
        P_forward_ci_lo[t], P_forward_ci_hi[t] = wilson_ci(
            n_in_basin, MC_N_SAMPLES
        )

        if (t + 1) % 100 == 0 or t == n_windows - 1:
            log.info("  %d / %d", t + 1, n_windows)

    # ══════════════════════════════════════════════════════════
    # Step 6 — Output
    # ══════════════════════════════════════════════════════════
    log.info("\nStep 6: Writing output...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── CSV ────────────────────────────────────────────────────
    header_lines = [
        f"# STTS Energy State — NORAD {best_nid} ({sat.get('object_name')})",
        f"# decay_epoch: {sat.get('decay_epoch')}",
        f"# mu_nominal: {mu_nominal:.6f}",
        f"# Markov cells: {MARKOV_K}  MC samples: {MC_N_SAMPLES}"
        f"  MC horizon: {MC_HORIZON}",
        f"# Dt={MC_HORIZON} windows at training stride="
        f"{WINDOW_STRIDE_TRAIN} TLE records/stride.",
        "# Physical time per MC step is variable: ~2.5 days during"
        " active decay,",
        "# ~5 weeks at operational altitude. Dt=10 steps ~ 25 days"
        " (decay) to ~1 year (operational).",
    ]
    with open(RESULTS_CSV, "w") as f:
        for line in header_lines:
            f.write(line + "\n")
        f.write("t,days_to_reentry,M2_cell,H_t,p_t,"
                "P_forward,P_forward_ci_lo,P_forward_ci_hi,artifact_flag\n")
        for t in range(n_windows):
            f.write(
                f"{t},{days_sat[t]:.2f},{M2_cells[t]},"
                f"{H[t]:.6f},{p_t[t]:.4f},"
                f"{P_forward[t]:.6f},{P_forward_ci_lo[t]:.6f},"
                f"{P_forward_ci_hi[t]:.6f},{artifact_flags[t]}\n"
            )
    log.info("  CSV: %s", RESULTS_CSV)

    # ── Results JSON (scientific-rigor: config + checksums) ─────
    results = {
        "config": config_snapshot(),
        "artifact_checksums": {
            "scaler": md5_file(SCALER_FILE),
            "lda": md5_file(LDA_FILE),
            "basin": md5_file(BASIN_FILE),
            "kmeans": md5_file(KMEANS_FILE),
            "markov_table": md5_file(MARKOV_TABLE_FILE),
        },
        "satellite": {
            "norad_id": best_nid,
            "object_name": sat.get("object_name"),
            "decay_epoch": sat.get("decay_epoch"),
            "n_windows": n_windows,
        },
        "parameters": {
            "mu_nominal": mu_nominal,
            "markov_k": MARKOV_K,
            "mc_n_samples": MC_N_SAMPLES,
            "mc_horizon": MC_HORIZON,
            "mc_horizon_physical": (
                f"{MC_HORIZON} steps x {WINDOW_STRIDE_TRAIN} records/step "
                f"= {MC_HORIZON * WINDOW_STRIDE_TRAIN} TLE records. "
                "~25 days (active decay) to ~350 days (operational)."
            ),
            "entropy_window": ENTROPY_WINDOW,
            "entropy_threshold": ENTROPY_THRESHOLD,
            "k_neighbors": K_NEIGHBORS,
            "failure_cells": sorted(failure_set),
        },
        "summary": {
            "H_range": [round(float(H.min()), 6), round(float(H.max()), 6)],
            "p_t_range": [round(float(p_t.min()), 4),
                          round(float(p_t.max()), 4)],
            "P_forward_range": [round(float(P_forward.min()), 6),
                                round(float(P_forward.max()), 6)],
            "n_artifact_flags": int(artifact_flags.sum()),
        },
        "metric_semantics": METRIC_SEMANTICS,
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    log.info("  JSON: %s", RESULTS_JSON)

    # ── Plot ───────────────────────────────────────────────────
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(
        f"STTS Energy State \u2014 NORAD {best_nid} ({sat.get('object_name')})",
        fontsize=14, fontweight="bold",
    )

    x = days_sat

    for ax in axes:
        ax.axvline(x=0, color="red", ls="--", lw=1.5, alpha=0.8,
                   label="Reentry")

    axes[0].plot(x, H, color="darkblue", lw=0.8)
    axes[0].set_ylabel("H(t)\n(energy)")
    axes[0].set_title("Energy State:  H(t) = T + V")
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].plot(x, p_t, color="darkorange", lw=0.8)
    axes[1].set_ylabel("p(t)\n(basin prob)")
    axes[1].set_title("Current Basin Membership Probability:  p(t)")
    axes[1].set_ylim(-0.05, 1.05)

    axes[2].fill_between(x, P_forward_ci_lo, P_forward_ci_hi,
                         color="darkred", alpha=0.15, label="95% CI")
    axes[2].plot(x, P_forward, color="darkred", lw=0.8)
    axes[2].set_ylabel("P(failure)\n(forward)")
    axes[2].set_title(
        f"Forward Failure Probability:  P(failure, t+\u0394t)  "
        f"[N={MC_N_SAMPLES:,}, \u0394t={MC_HORIZON}]"
    )
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(loc="upper right", fontsize=9)

    axes[3].plot(x, M2_cells, color="teal", lw=0.8, marker=".", ms=1)
    axes[3].set_ylabel("M\u2082 cell")
    axes[3].set_title("Markov Cell Assignment")

    flag_x = x[artifact_flags == 1]
    axes[4].vlines(flag_x, 0, 1, color="purple", lw=0.8, alpha=0.7)
    axes[4].set_ylabel("Artifact\nflag")
    axes[4].set_title(
        "Signal Separation:  Artifact Flags "
        f"(entropy > {ENTROPY_THRESHOLD:.0f}\u00d7 expected)"
    )
    axes[4].set_ylim(-0.1, 1.1)
    axes[4].set_xlabel("Days to Reentry")

    axes[0].invert_xaxis()

    plt.tight_layout()
    fig.savefig(RESULTS_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Plot: %s", RESULTS_PLOT)

    log.info("\nDone. All prior results unchanged.")
    return results


if __name__ == "__main__":
    main()
