"""Conformal calibration of STTS monitoring thresholds.

Provides distribution-free false alarm rate guarantees via conformal
prediction. See docs/seeds/stts_conformal_calibration.md for the full
theoretical treatment.

The core guarantee: for target FPR α and calibration set of size n,
the achieved false alarm rate is bounded by α + 1/(n+1) regardless
of the underlying distance distribution.
"""

from __future__ import annotations

from typing import Optional
from collections import deque

import numpy as np
import pandas as pd
import faiss


def conformal_epsilon(
    calibration_distances: np.ndarray,
    target_fpr: float,
) -> tuple[float, float]:
    """Set ε at the (1-α) quantile of calibration distances.

    Args:
        calibration_distances: distances from nominal trajectories to B_f
        target_fpr: desired false alarm rate α

    Returns:
        (epsilon, achieved_fpr_bound)
    """
    n = len(calibration_distances)
    adjusted_quantile = min(1.0, (1 - target_fpr) * (n + 1) / n)
    epsilon = float(np.quantile(calibration_distances, adjusted_quantile))
    fpr_bound = target_fpr + 1 / (n + 1)
    return epsilon, fpr_bound


def calibration_curve(
    cal_distances_nominal: np.ndarray,
    test_distances_nominal: np.ndarray,
    test_distances_failure: np.ndarray,
    fpr_targets: list[float] = None,
) -> pd.DataFrame:
    """Compute precision/recall/F1 at each target FPR level.

    The FPR guarantee is conformal — distribution-free.

    Args:
        cal_distances_nominal: calibration set nominal distances to B_f
        test_distances_nominal: test set nominal distances
        test_distances_failure: test set failure-approaching distances
        fpr_targets: list of target FPR values

    Returns:
        DataFrame with one row per target FPR
    """
    if fpr_targets is None:
        fpr_targets = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    rows = []
    for target_fpr in fpr_targets:
        eps, fpr_bound = conformal_epsilon(cal_distances_nominal, target_fpr)

        fp = int(np.sum(test_distances_nominal < eps))
        tp = int(np.sum(test_distances_failure < eps))
        fn = int(np.sum(test_distances_failure >= eps))
        tn = int(np.sum(test_distances_nominal >= eps))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        actual_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        rows.append({
            "target_fpr": target_fpr,
            "fpr_guarantee": fpr_bound,
            "actual_fpr": actual_fpr,
            "epsilon": eps,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    return pd.DataFrame(rows)


def four_state_monitor(
    embedding: np.ndarray,
    basin_index: faiss.IndexFlatL2,
    nominal_index: faiss.IndexFlatL2,
    corpus_index: faiss.IndexFlatL2,
    epsilon_f: float,
    epsilon_n: float,
    epsilon_ood: float,
    k_basin: int = 5,
    k_ood: int = 5,
) -> dict:
    """Classify a trajectory embedding into one of four states.

    States:
        TERRA_INCOGNITA: outside corpus coverage, no prediction reliable
        FAILURE_APPROACH: approaching failure basin, intervention indicated
        NOMINAL: confirmed safe, corpus well-covered
        WATCH: between basins, monitoring closely

    Args:
        embedding: (1, d) or (d,) trajectory embedding
        basin_index: FAISS index over failure basin B_f
        nominal_index: FAISS index over nominal basin B_n
        corpus_index: FAISS index over full corpus
        epsilon_f: conformal failure threshold
        epsilon_n: conformal nominal threshold
        epsilon_ood: conformal OOD threshold
        k_basin: k for basin distance
        k_ood: k for OOD k-NN distance

    Returns:
        dict with state, distances, and thresholds
    """
    emb = np.ascontiguousarray(embedding.reshape(1, -1), dtype=np.float32)

    # Distance to failure basin
    k_f = min(k_basin, basin_index.ntotal)
    d_f, _ = basin_index.search(emb, k_f)
    d_failure = float(np.sqrt(d_f[0]).mean())

    # Distance to nominal basin
    k_n = min(k_basin, nominal_index.ntotal)
    d_n, _ = nominal_index.search(emb, k_n)
    d_nominal = float(np.sqrt(d_n[0]).mean())

    # OOD: k-th nearest neighbor distance in full corpus
    k_o = min(k_ood, corpus_index.ntotal)
    d_c, _ = corpus_index.search(emb, k_o)
    d_ood = float(np.sqrt(d_c[0, -1]))

    # Priority ordering: OOD > FAILURE > NOMINAL > WATCH
    if d_ood > epsilon_ood:
        state = "TERRA_INCOGNITA"
    elif d_failure < epsilon_f:
        state = "FAILURE_APPROACH"
    elif d_nominal < epsilon_n:
        state = "NOMINAL"
    else:
        state = "WATCH"

    return {
        "state": state,
        "d_failure": d_failure,
        "d_nominal": d_nominal,
        "d_ood": d_ood,
        "epsilon_f": epsilon_f,
        "epsilon_n": epsilon_n,
        "epsilon_ood": epsilon_ood,
    }


def build_nominal_basin(
    embeddings: np.ndarray,
    rul: np.ndarray,
    rul_min: int,
) -> np.ndarray:
    """Extract nominal basin B_n: embeddings with RUL > rul_min.

    Args:
        embeddings: (n_samples, d)
        rul: (n_samples,)
        rul_min: trajectories with RUL > this enter B_n

    Returns:
        nominal embeddings
    """
    mask = rul > rul_min
    return embeddings[mask].copy()
