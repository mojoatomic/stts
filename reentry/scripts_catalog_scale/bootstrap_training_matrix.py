#!/usr/bin/env python3
"""
Training-set bootstrap over the 529 satellites that fit the reentry
Markov transition matrix P.

Read-only against frozen model artifacts. Only the training-satellite
*identities* are resampled (with replacement); projection (scaler,
LDA, KMeans, basin, failure_cells) is frozen per the REV 3 plan.

Produces per-replicate values and percentile CIs for:
  lambda_2       — second-largest eigenvalue of P by magnitude
  D_P10          — first-order vs second-order KL aggregate on training cell sequences
  mu_nominal     — mean of frozen LDA projection over nominal windows in the resample
  stationary_pi  — left eigenvector of P at eigenvalue 1, normalized
  P_end[x, Dt]   — endpoint probability of failure basin, for Dt = 1..30
  P_hit[x, Dt]   — hitting-time CDF, for Dt = 1..30

Outputs:
  results/reentry/catalog_scale_validation/bootstrap_training_cis.json
  results/reentry/catalog_scale_validation/bootstrap_training_cis.md
"""
from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reentry.config import MARKOV_K, WINDOW_SIZE, WINDOW_STRIDE_TRAIN
from reentry.corpus import load_corpus
from reentry.features import build_feature_matrix
from reentry.hitting_time import exact_endpoint_cdf, exact_hitting_cdf
from reentry.train import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("results/reentry/catalog_scale_validation")
OUT_JSON = OUT_DIR / "bootstrap_training_cis.json"
OUT_MD = OUT_DIR / "bootstrap_training_cis.md"

N_REPLICATES = 1000
SEED = 20260421
H_MAX = 30
HORIZONS = list(range(1, H_MAX + 1))


# ── Helpers reused across replicates ─────────────────────────────

def transitions_from_sequences(sat_sequences: dict[str, list[tuple[int, int]]],
                                norads: list[str],
                                K: int,
                                stride: int) -> np.ndarray:
    """Count transitions (c_t, c_{t+1}) across a list of satellites.

    Each satellite's sequence is a list of (window_start, cell) pairs,
    assumed sorted by window_start. A pair is counted when
    start_next - start_curr == stride. Matches train.py:210-233.
    """
    N = np.zeros((K, K), dtype=np.int64)
    for nid in norads:
        seq = sat_sequences.get(nid)
        if not seq:
            continue
        for j in range(len(seq) - 1):
            s0, c0 = seq[j]
            s1, c1 = seq[j + 1]
            if s1 - s0 == stride:
                N[c0, c1] += 1
    return N


def triples_from_sequences(sat_sequences, norads, K, stride):
    """Count second-order triples (c_{t-1}, c_t, c_{t+1})."""
    N2 = np.zeros((K, K, K), dtype=np.int64)
    for nid in norads:
        seq = sat_sequences.get(nid)
        if not seq or len(seq) < 3:
            continue
        for j in range(len(seq) - 2):
            s0, c0 = seq[j]
            s1, c1 = seq[j + 1]
            s2, c2 = seq[j + 2]
            if s1 - s0 == stride and s2 - s1 == stride:
                N2[c0, c1, c2] += 1
    return N2


def normalize_rows(N: np.ndarray, K: int) -> np.ndarray:
    """Row-normalize a count matrix; empty rows fall back to uniform (as train.py:229-232)."""
    N = N.astype(np.float64)
    row_sums = N.sum(axis=1)
    P = N.copy()
    for i in range(K):
        if row_sums[i] > 0:
            P[i] /= row_sums[i]
        else:
            P[i] = 1.0 / K
    return P


def lambda2(P: np.ndarray) -> float:
    """Second-largest eigenvalue of P by magnitude. Real (no complex pairs in this chain)."""
    ev = np.linalg.eigvals(P)
    mags = np.sort(np.abs(ev))[::-1]
    return float(mags[1])


def stationary(P: np.ndarray) -> np.ndarray:
    """Left eigenvector at eigenvalue 1, normalized to sum to 1."""
    evals, evecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(evals - 1.0)))
    pi = np.real(evecs[:, idx])
    pi = pi / pi.sum()
    # Numerical negatives -> clip then renormalize
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


def d_p10(N2: np.ndarray, K: int, min_support: int = 10) -> float:
    """D_P10 = sum_{(a,b)} pi(a,b) * KL(P2(.|a,b) || P1(.|b)).

    Matches the Markov-order-baseline computation.
    """
    N1_from_N2 = N2.sum(axis=0)  # N1[b, c] = number of (b -> c) transitions that appeared as the tail of a triple
    # First-order P(c|b) from the N2-derived counts (consistent with the triple-constraint sample).
    row_sums_1 = N1_from_N2.sum(axis=1).astype(np.float64)
    P1 = np.zeros((K, K), dtype=np.float64)
    for b in range(K):
        if row_sums_1[b] > 0:
            P1[b] = N1_from_N2[b] / row_sums_1[b]
    pair_count = N2.sum(axis=2)  # pair_count[a, b]
    total = pair_count.sum()
    if total == 0:
        return float("nan")
    pair_freq = pair_count.astype(np.float64) / total
    aggregate = 0.0
    for a in range(K):
        for b in range(K):
            pc = int(pair_count[a, b])
            if pc < min_support:
                continue
            p2 = N2[a, b, :].astype(np.float64) / pc
            p1 = P1[b]
            kl = 0.0
            for k in range(K):
                if p2[k] <= 0:
                    continue
                if p1[k] <= 0:
                    # inf divergence; skip per task convention
                    kl = float("nan")
                    break
                kl += p2[k] * np.log(p2[k] / p1[k])
            if not np.isnan(kl):
                aggregate += pair_freq[a, b] * kl
    return aggregate


def percentile_ci(arr: np.ndarray, alpha: float = 0.05):
    lo = float(np.quantile(arr, alpha / 2))
    hi = float(np.quantile(arr, 1 - alpha / 2))
    return lo, hi


# ── Main ─────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading frozen model + corpus...")
    model = load_model()
    scaler, W, lda, kmeans = model["scaler"], model["W"], model["lda"], model["kmeans"]
    markov = model["markov"]
    T_frozen = markov["transition_matrix"]
    failure_cells_arr = np.array(sorted(markov["failure_cells"].tolist()))
    log.info(f"  frozen P shape: {T_frozen.shape}, failure cells: {failure_cells_arr.tolist()}")

    corpus = load_corpus()
    train_ids = list(corpus["train_ids"])
    log.info(f"  training satellites: {len(train_ids)}")

    log.info("Reconstructing per-satellite cell sequences under frozen projection...")
    t0 = time.time()
    X_all, y_all, _days, ids_all = build_feature_matrix(
        corpus["satellites"], train_ids,
        window_size=WINDOW_SIZE, stride=WINDOW_STRIDE_TRAIN,
    )
    mask = y_all != -1
    X_tr = X_all[mask]
    ids_tr = [ids_all[i] for i in range(len(ids_all)) if mask[i]]
    y_tr = y_all[mask]
    X_sc = scaler.transform(X_tr)
    X_sc = np.nan_to_num(X_sc, nan=0.0, posinf=0.0, neginf=0.0)
    X_w = X_sc * W
    X_p = lda.transform(X_w).ravel()
    cells = kmeans.predict(X_p.reshape(-1, 1))
    log.info(f"  windows reconstructed: {len(cells)}, elapsed {time.time()-t0:.1f}s")

    # Verify reconstruction matches frozen Markov table (full-train transition matrix).
    sat_sequences = defaultdict(list)
    for i, wid in enumerate(ids_tr):
        nid, start = wid.rsplit(":", 1)
        sat_sequences[nid].append((int(start), int(cells[i])))
    for nid in sat_sequences:
        sat_sequences[nid].sort(key=lambda x: x[0])
    sat_sequences = dict(sat_sequences)

    N_full = transitions_from_sequences(sat_sequences, train_ids, MARKOV_K, WINDOW_STRIDE_TRAIN)
    P_full = normalize_rows(N_full, MARKOV_K)
    max_diff = float(np.abs(P_full - T_frozen).max())
    log.info(f"  Full-train P vs frozen: max |diff| = {max_diff:.3e}")
    if max_diff > 1e-10:
        raise SystemExit(
            f"Reconstruction mismatch (max |diff| = {max_diff:.3e}) vs frozen Markov table. "
            "Cannot proceed with bootstrap."
        )

    # Record the point-estimate quantities on the full training set.
    t0 = time.time()
    lam2_full = lambda2(P_full)
    pi_full = stationary(P_full)
    N2_full = triples_from_sequences(sat_sequences, train_ids, MARKOV_K, WINDOW_STRIDE_TRAIN)
    D_P10_full = d_p10(N2_full, MARKOV_K)
    # mu_nominal on full training set — uses frozen projection; consistent with markov_table.npz.
    mu_nominal_full = float(np.mean(X_p[y_tr == 0]))
    P_end_full = exact_endpoint_cdf(P_full, failure_cells_arr, HORIZONS)
    P_hit_full = exact_hitting_cdf(P_full, failure_cells_arr, HORIZONS)
    log.info(f"  full-training point estimates: lam2={lam2_full:.6f}  "
             f"D_P10={D_P10_full:.6f}  mu_nom={mu_nominal_full:.6f}  "
             f"({time.time()-t0:.2f}s)")

    # Bootstrap.
    log.info(f"Running B={N_REPLICATES} bootstrap replicates (seed={SEED})...")
    rng = np.random.default_rng(SEED)
    n = len(train_ids)
    train_ids_arr = np.array(train_ids)

    # Map NORAD -> (list of (start, cell) pairs) and NORAD -> (list of X_p values, y values)
    sat_X_p = defaultdict(list)
    sat_y = defaultdict(list)
    for i, wid in enumerate(ids_tr):
        nid = wid.split(":")[0]
        sat_X_p[nid].append(X_p[i])
        sat_y[nid].append(y_tr[i])
    for nid in sat_X_p:
        sat_X_p[nid] = np.array(sat_X_p[nid])
        sat_y[nid] = np.array(sat_y[nid])

    lam2_b = np.zeros(N_REPLICATES)
    D_P10_b = np.zeros(N_REPLICATES)
    mu_nom_b = np.zeros(N_REPLICATES)
    pi_b = np.zeros((N_REPLICATES, MARKOV_K))
    P_end_b = np.zeros((N_REPLICATES, MARKOV_K, H_MAX))
    P_hit_b = np.zeros((N_REPLICATES, MARKOV_K, H_MAX))

    t0 = time.time()
    for b in range(N_REPLICATES):
        idx = rng.integers(0, n, size=n)
        resample = train_ids_arr[idx].tolist()

        N_b = transitions_from_sequences(sat_sequences, resample, MARKOV_K, WINDOW_STRIDE_TRAIN)
        P_b = normalize_rows(N_b, MARKOV_K)
        lam2_b[b] = lambda2(P_b)
        pi_b[b] = stationary(P_b)

        N2_b = triples_from_sequences(sat_sequences, resample, MARKOV_K, WINDOW_STRIDE_TRAIN)
        D_P10_b[b] = d_p10(N2_b, MARKOV_K)

        nom_vals = []
        for nid in resample:
            xs = sat_X_p[nid]
            ys = sat_y[nid]
            nom_vals.append(xs[ys == 0])
        if nom_vals:
            mu_nom_b[b] = float(np.concatenate(nom_vals).mean())

        P_end_b[b] = exact_endpoint_cdf(P_b, failure_cells_arr, HORIZONS)
        P_hit_b[b] = exact_hitting_cdf(P_b, failure_cells_arr, HORIZONS)

        if (b + 1) % 100 == 0 or b == N_REPLICATES - 1:
            elapsed = time.time() - t0
            remaining = elapsed / (b + 1) * (N_REPLICATES - b - 1)
            log.info(f"  [{b+1}/{N_REPLICATES}]  elapsed {elapsed:.1f}s  est remaining {remaining:.1f}s")

    # Assemble percentile CIs.
    out = {
        "n_replicates": N_REPLICATES,
        "seed": SEED,
        "n_train_satellites": n,
        "horizons": HORIZONS,
        "frozen_match_max_diff": max_diff,
        "lambda_2": {
            "point": lam2_full,
            "ci_lo": percentile_ci(lam2_b)[0],
            "ci_hi": percentile_ci(lam2_b)[1],
        },
        "D_P10": {
            "point": D_P10_full,
            "ci_lo": percentile_ci(D_P10_b)[0],
            "ci_hi": percentile_ci(D_P10_b)[1],
        },
        "mu_nominal": {
            "point": mu_nominal_full,
            "ci_lo": percentile_ci(mu_nom_b)[0],
            "ci_hi": percentile_ci(mu_nom_b)[1],
        },
        "stationary_pi": {
            "point": pi_full.tolist(),
            "ci_lo": [percentile_ci(pi_b[:, k])[0] for k in range(MARKOV_K)],
            "ci_hi": [percentile_ci(pi_b[:, k])[1] for k in range(MARKOV_K)],
        },
        "P_end": {
            "point": P_end_full.tolist(),
            "ci_lo": np.quantile(P_end_b, 0.025, axis=0).tolist(),
            "ci_hi": np.quantile(P_end_b, 0.975, axis=0).tolist(),
        },
        "P_hit": {
            "point": P_hit_full.tolist(),
            "ci_lo": np.quantile(P_hit_b, 0.025, axis=0).tolist(),
            "ci_hi": np.quantile(P_hit_b, 0.975, axis=0).tolist(),
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"wrote {OUT_JSON}")

    # Write formatted markdown summary.
    write_md(out)
    log.info(f"wrote {OUT_MD}")


def write_md(out: dict) -> None:
    L = []
    L.append("# STTS-Reentry — Training-Set Bootstrap (frozen projection)\n\n")
    L.append(
        f"Percentile bootstrap over {out['n_train_satellites']} training satellites, "
        f"B = {out['n_replicates']} replicates, seed {out['seed']}. "
        "Projection (scaler/LDA/KMeans/basin/failure_cells) held frozen across "
        "replicates; only the set of training NORADs is resampled with "
        "replacement and the transition matrix is rebuilt from their cell "
        "sequences under the frozen projection.\n\n"
    )
    L.append(
        f"Full-training transition matrix reconstruction vs frozen "
        f"`markov_table.npz`: max abs difference "
        f"{out['frozen_match_max_diff']:.3e} (expect ≤ 1e-10).\n\n"
    )

    L.append("## Scalar quantities\n\n")
    L.append("| quantity | point | 95% CI |\n")
    L.append("|---|---:|---:|\n")
    for k in ("lambda_2", "D_P10", "mu_nominal"):
        v = out[k]
        L.append(f"| `{k}` | {v['point']:.6f} | [{v['ci_lo']:.6f}, {v['ci_hi']:.6f}] |\n")
    L.append("\n")

    L.append("## Stationary distribution π\n\n")
    L.append("| cell | π point | 95% CI |\n|---:|---:|---:|\n")
    pi = out["stationary_pi"]
    for k in range(len(pi["point"])):
        L.append(f"| {k} | {pi['point'][k]:.6f} | [{pi['ci_lo'][k]:.6f}, {pi['ci_hi'][k]:.6f}] |\n")
    L.append("\n")

    L.append("## Hitting-time CDF P_hit[x, Δt]\n\n")
    L.append(
        "Point estimate with percentile 95% CI in brackets. Horizon Δt in "
        "Markov steps (1 step = 1 training-stride window = 5 TLE records).\n\n"
    )
    horizons = out["horizons"]
    shown = [1, 5, 10, 15, 20, 25, 30]
    idxs = [horizons.index(h) for h in shown]
    P_hit_point = np.array(out["P_hit"]["point"])
    P_hit_lo = np.array(out["P_hit"]["ci_lo"])
    P_hit_hi = np.array(out["P_hit"]["ci_hi"])
    K = P_hit_point.shape[0]
    L.append("| cell | class | " + " | ".join(f"Δt={h}" for h in shown) + " |\n")
    L.append("|---:|---:|" + "---:|" * len(shown) + "\n")
    failure_cells = np.where(P_hit_point[:, 0] >= 1.0 - 1e-12)[0].tolist()
    for x in range(K):
        tag = "F" if x in failure_cells else "n"
        cells_str = []
        for i in idxs:
            cells_str.append(
                f"{P_hit_point[x, i]:.3f} [{P_hit_lo[x, i]:.3f},{P_hit_hi[x, i]:.3f}]"
            )
        L.append(f"| {x} | {tag} | " + " | ".join(cells_str) + " |\n")
    L.append("\n")

    L.append("## Endpoint probability P_end[x, Δt]\n\n")
    L.append("(For comparison with historical MC P_forward values.)\n\n")
    P_end_point = np.array(out["P_end"]["point"])
    P_end_lo = np.array(out["P_end"]["ci_lo"])
    P_end_hi = np.array(out["P_end"]["ci_hi"])
    L.append("| cell | class | " + " | ".join(f"Δt={h}" for h in shown) + " |\n")
    L.append("|---:|---:|" + "---:|" * len(shown) + "\n")
    for x in range(K):
        tag = "F" if x in failure_cells else "n"
        cells_str = []
        for i in idxs:
            cells_str.append(
                f"{P_end_point[x, i]:.3f} [{P_end_lo[x, i]:.3f},{P_end_hi[x, i]:.3f}]"
            )
        L.append(f"| {x} | {tag} | " + " | ".join(cells_str) + " |\n")
    L.append("\n")

    OUT_MD.write_text("".join(L))


if __name__ == "__main__":
    main()
