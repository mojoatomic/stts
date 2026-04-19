"""
STTS Energy State and Markov Monte Carlo Forward Prediction.

Additive extension to the reentry validation. Computes:
  H(t)              — energy state (drift velocity + basin distance)
  p(t)              — current basin membership probability
  P(failure, t+Dt)  — forward failure probability via Markov MC

Does NOT modify existing pipeline, artifacts, or results.
Loads the frozen model (never calls .fit()).

Usage:
    python reentry/energy_state.py

Requires: corpus built + model trained (python reentry/run_all.py)
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy as scipy_entropy
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent.parent))

from reentry.config import (
    ARTIFACTS_DIR,
    K_NEIGHBORS,
    RESULTS_DIR,
    WINDOW_SIZE,
    WINDOW_STRIDE_EVAL,
    WINDOW_STRIDE_TRAIN,
)
from reentry.corpus import load_corpus
from reentry.features import build_feature_matrix
from reentry.train import load_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────
MARKOV_K = 20            # k-means cells for M2 discretization
MC_N_SAMPLES = 10_000    # forward trajectories per timestep
MC_HORIZON = 10          # forward steps (in training-stride windows)
ENTROPY_WINDOW = 5       # rolling window for signal separation
ENTROPY_THRESHOLD = 2.0  # flag if observed entropy > 2x expected

MARKOV_TABLE_FILE = ARTIFACTS_DIR / "markov_table.npz"
RESULTS_CSV = RESULTS_DIR / "energy_state.csv"
RESULTS_PLOT = RESULTS_DIR / "energy_state.png"


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
    basin = model["basin"]
    epsilon = model["epsilon"]
    dist_to_basin = model["dist_to_basin"]
    k = K_NEIGHBORS

    satellites = corpus["satellites"]
    train_ids = corpus["train_ids"]
    test_reentry_ids = corpus.get("test_reentry_ids", [])

    # ── Build training projections ─────────────────────────────
    # Used for: mu_nominal, p(t) neighbor pool, Markov table
    log.info("Building training projections (stride=%d)...", WINDOW_STRIDE_TRAIN)
    X_train, y_train, _, ids_train = build_feature_matrix(
        satellites, train_ids,
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE_TRAIN,
    )

    # Exclude ambiguous windows (y=-1)
    mask = y_train != -1
    X_tr = X_train[mask]
    y_tr = y_train[mask]
    ids_tr = [ids_train[i] for i in range(len(ids_train)) if mask[i]]

    train_proj = _project(X_tr, scaler, W, lda)

    # mu_nominal: mean of nominal-class (y=0) training projections
    mu_nominal = float(np.mean(train_proj[y_tr == 0]))
    log.info("  mu_nominal = %.6f", mu_nominal)
    log.info("  Training pool: %d windows (nominal=%d, precursor=%d)",
             len(train_proj), (y_tr == 0).sum(), (y_tr == 1).sum())

    # ══════════════════════════════════════════════════════════
    # Step 3 — Markov table
    # ══════════════════════════════════════════════════════════
    log.info("\nStep 3: Building Markov table (k-means k=%d)...", MARKOV_K)

    kmeans = KMeans(n_clusters=MARKOV_K, random_state=42, n_init=10)
    train_cells = kmeans.fit_predict(train_proj.reshape(-1, 1))
    cell_centers = kmeans.cluster_centers_.ravel()

    # Report cell counts
    cell_counts = np.bincount(train_cells, minlength=MARKOV_K)
    log.info("  Cell counts: %s", cell_counts.tolist())

    # Count cell-to-cell transitions within each satellite's trajectory.
    # Only consecutive windows (start indices differ by exactly the
    # training stride) count as a valid transition — no cross-satellite
    # or gap transitions.
    transition_matrix = np.zeros((MARKOV_K, MARKOV_K), dtype=np.float64)

    sat_windows = {}
    for i, wid in enumerate(ids_tr):
        nid, start = wid.rsplit(":", 1)
        start = int(start)
        sat_windows.setdefault(nid, []).append((i, start))

    for nid, windows in sat_windows.items():
        windows.sort(key=lambda x: x[1])
        for j in range(len(windows) - 1):
            idx_curr, start_curr = windows[j]
            idx_next, start_next = windows[j + 1]
            if start_next - start_curr == WINDOW_STRIDE_TRAIN:
                transition_matrix[train_cells[idx_curr],
                                  train_cells[idx_next]] += 1

    # Normalize rows; uniform for zero rows
    row_sums = transition_matrix.sum(axis=1)
    zero_rows = row_sums == 0
    if zero_rows.any():
        log.warning("  WARNING: %d zero rows (cells %s) — uniform fallback.",
                    zero_rows.sum(), np.where(zero_rows)[0].tolist())
        transition_matrix[zero_rows] = 1.0 / MARKOV_K
        row_sums[zero_rows] = 1.0
    transition_matrix /= row_sums[:, np.newaxis]

    # Identify failure basin cells: centroid's k-NN distance to basin < epsilon
    cell_basin_dists = np.array([dist_to_basin(c) for c in cell_centers])
    failure_cells = np.where(cell_basin_dists < epsilon)[0]
    failure_set = set(failure_cells.tolist())
    log.info("  Failure basin cells: %s (of %d total)", sorted(failure_set), MARKOV_K)

    # Serialize
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        MARKOV_TABLE_FILE,
        transition_matrix=transition_matrix,
        cell_centers=cell_centers,
        failure_cells=failure_cells,
        cell_counts=cell_counts,
        mu_nominal=np.array([mu_nominal]),
    )
    log.info("  Saved: %s", MARKOV_TABLE_FILE)

    # Expected entropy per cell (for signal separation)
    expected_entropy = np.array([
        scipy_entropy(transition_matrix[c]) for c in range(MARKOV_K)
    ])

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

    assert np.all(np.isfinite(days_sat)), "Non-finite days_to_reentry on reentry satellite"

    # Assign to Markov cells
    M2_cells = kmeans.predict(M2.reshape(-1, 1))

    # ══════════════════════════════════════════════════════════
    # Step 1 — H(t)
    # ══════════════════════════════════════════════════════════
    log.info("\nStep 1: Computing H(t)...")
    V = np.abs(M2 - mu_nominal)
    T = np.zeros(n_windows)
    T[1:] = 0.5 * (M2[1:] - M2[:-1]) ** 2
    H = T + V

    # ══════════════════════════════════════════════════════════
    # Step 2 — p(t)
    # ══════════════════════════════════════════════════════════
    log.info("Step 2: Computing p(t)...")
    p_t = np.zeros(n_windows)
    for t in range(n_windows):
        dists = np.abs(train_proj - M2[t])
        nn_idx = np.argpartition(dists, k)[:k]
        p_t[t] = np.mean(y_tr[nn_idx] == 1)

    assert np.all((p_t >= 0) & (p_t <= 1)), "p(t) out of [0, 1]"
    log.info("  p(t) range: [%.4f, %.4f]", p_t.min(), p_t.max())

    # ══════════════════════════════════════════════════════════
    # Step 4 — Signal separation
    # ══════════════════════════════════════════════════════════
    log.info("Step 4: Signal separation (entropy, window=%d)...", ENTROPY_WINDOW)
    artifact_flags = np.zeros(n_windows, dtype=int)

    for t in range(ENTROPY_WINDOW, n_windows):
        # Destination cells from the last ENTROPY_WINDOW transitions
        dest_cells = M2_cells[t - ENTROPY_WINDOW + 1: t + 1]
        dest_counts = np.bincount(dest_cells, minlength=MARKOV_K).astype(float)
        dest_counts /= dest_counts.sum()
        obs_H = scipy_entropy(dest_counts)

        cur_cell = M2_cells[t]
        exp_H = expected_entropy[cur_cell]

        if exp_H < 1e-10:
            # Deterministic cell — any observed variation is an artifact
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
    # ══════════════════════════════════════════════════════════
    # Each MC step = one training-stride transition = WINDOW_STRIDE_TRAIN
    # TLE records. Dt=10 steps x 5 records/step = 50 TLE records.
    # At ~2 TLEs/day during active decay:  ~25 days forward.
    # At ~1 TLE/week at operational alt:   ~350 days forward.
    # Physical time is variable — Dt is a record-count horizon.
    log.info("Step 5: P(failure, t+Dt) — MC N=%d, Dt=%d ...",
             MC_N_SAMPLES, MC_HORIZON)

    rng = np.random.RandomState(42)
    P_forward = np.zeros(n_windows)

    # Precompute cumulative transition probabilities
    cum_trans = np.cumsum(transition_matrix, axis=1)

    failure_arr = np.array(sorted(failure_set))

    for t in range(n_windows):
        states = np.full(MC_N_SAMPLES, M2_cells[t], dtype=int)
        for step in range(MC_HORIZON):
            u = rng.random(MC_N_SAMPLES)
            cdfs = cum_trans[states]                    # (N, K)
            states = (cdfs < u[:, np.newaxis]).sum(axis=1)
            np.clip(states, 0, MARKOV_K - 1, out=states)
        P_forward[t] = np.mean(np.isin(states, failure_arr))

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
        f.write("t,days_to_reentry,M2_cell,H_t,p_t,P_forward,artifact_flag\n")
        for t in range(n_windows):
            f.write(
                f"{t},{days_sat[t]:.2f},{M2_cells[t]},"
                f"{H[t]:.6f},{p_t[t]:.4f},{P_forward[t]:.6f},"
                f"{artifact_flags[t]}\n"
            )
    log.info("  CSV: %s", RESULTS_CSV)

    # ── Plot ───────────────────────────────────────────────────
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(
        f"STTS Energy State \u2014 NORAD {best_nid} ({sat.get('object_name')})",
        fontsize=14, fontweight="bold",
    )

    x = days_sat  # x-axis: days to reentry (inverted so time flows L→R)

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

    axes[2].plot(x, P_forward, color="darkred", lw=0.8)
    axes[2].set_ylabel("P(failure)\n(forward)")
    axes[2].set_title(
        f"Forward Failure Probability:  P(failure, t+\u0394t)  "
        f"[N={MC_N_SAMPLES:,}, \u0394t={MC_HORIZON}]"
    )
    axes[2].set_ylim(-0.05, 1.05)

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

    # Invert so time flows left-to-right (reentry on the right)
    axes[0].invert_xaxis()

    plt.tight_layout()
    fig.savefig(RESULTS_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Plot: %s", RESULTS_PLOT)

    log.info("\nDone. All prior results unchanged.")


if __name__ == "__main__":
    main()
