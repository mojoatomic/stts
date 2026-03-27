"""Evaluation: intervention windows, verification conditions, baselines."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats


def compute_stts_detection_cycle(
    distances: np.ndarray,
    epsilon: float,
) -> Optional[int]:
    """First cycle index where basin distance drops below epsilon.

    Returns:
        Index into the distances array, or None if never fires.
    """
    below = np.where(distances < epsilon)[0]
    return int(below[0]) if len(below) > 0 else None


def compute_threshold_detection_cycle(
    sensor_data: np.ndarray,
    train_means: np.ndarray,
    train_stds: np.ndarray,
    n_sigma: float = 3.0,
) -> Optional[int]:
    """Simulate threshold monitoring: first cycle any sensor exceeds n-sigma.

    Args:
        sensor_data: (n_cycles, n_sensors) normalized sensor readings
        train_means: (n_sensors,) training mean per sensor
        train_stds: (n_sensors,) training std per sensor
        n_sigma: threshold in standard deviations

    Returns:
        Cycle index of first threshold crossing, or None.
    """
    deviations = np.abs(sensor_data - train_means) / np.maximum(train_stds, 1e-10)
    exceeds = np.any(deviations > n_sigma, axis=1)
    crossings = np.where(exceeds)[0]
    return int(crossings[0]) if len(crossings) > 0 else None


def intervention_window(
    stts_cycle: Optional[int],
    threshold_cycle: Optional[int],
    total_cycles: int,
) -> dict:
    """Compute the intervention window recovered by STTS.

    Returns dict with:
        stts_lead: cycles before failure that STTS fires
        threshold_lead: cycles before failure that threshold fires
        window_recovered: additional cycles STTS provides over threshold
    """
    stts_lead = (total_cycles - stts_cycle) if stts_cycle is not None else 0
    threshold_lead = (total_cycles - threshold_cycle) if threshold_cycle is not None else 0
    return {
        "stts_lead": stts_lead,
        "threshold_lead": threshold_lead,
        "window_recovered": stts_lead - threshold_lead,
        "stts_fired": stts_cycle is not None,
        "threshold_fired": threshold_cycle is not None,
    }


def verify_v1(
    basin_distances_precursor: np.ndarray,
    basin_distances_nominal: np.ndarray,
) -> dict:
    """V1 — Precursor proximity: precursors should be closer to B_f than nominal.

    Returns:
        dict with pass/fail, median distances, and Mann-Whitney U test p-value
    """
    median_precursor = np.median(basin_distances_precursor)
    median_nominal = np.median(basin_distances_nominal)
    stat, pvalue = stats.mannwhitneyu(
        basin_distances_precursor, basin_distances_nominal, alternative="less"
    )
    return {
        "passed": median_precursor < median_nominal,
        "median_precursor": median_precursor,
        "median_nominal": median_nominal,
        "separation_ratio": median_nominal / max(median_precursor, 1e-10),
        "mannwhitney_p": pvalue,
    }


def verify_v2(
    distances: np.ndarray,
    rul: np.ndarray,
) -> dict:
    """V2 — Monotonic approach: distance to B_f should increase with RUL.

    Spearman correlation between distance and RUL should be positive
    (higher RUL = farther from basin).

    Returns:
        dict with Spearman rho, p-value, and pass/fail
    """
    rho, pvalue = stats.spearmanr(distances, rul)
    return {
        "passed": rho > 0 and pvalue < 0.05,
        "spearman_rho": rho,
        "p_value": pvalue,
    }


def verify_v3_ablation(
    features: np.ndarray,
    weights: np.ndarray,
    feature_class_indices: dict[str, np.ndarray],
    fit_and_query_fn: callable,
) -> dict:
    """V3 — Causal traceability: ablate each feature class and measure impact.

    For each feature class, zero it out, re-run the embedding + basin
    distance computation, and measure how much the distance changes.
    The feature class with the largest impact is the primary driver.

    Args:
        features: (n_samples, feature_dim) raw features
        weights: (feature_dim,) weight vector
        feature_class_indices: {class_name: index_array}
        fit_and_query_fn: callable(features, weights) -> distances array

    Returns:
        dict mapping class_name -> relative distance change when ablated
    """
    baseline_distances = fit_and_query_fn(features, weights)
    baseline_mean = np.mean(baseline_distances)

    results = {}
    for cls_name, indices in feature_class_indices.items():
        ablated_weights = weights.copy()
        ablated_weights[indices] = 0.0
        ablated_distances = fit_and_query_fn(features, ablated_weights)
        ablated_mean = np.mean(ablated_distances)
        # Relative change: positive means removing this class pushed
        # trajectories farther from B_f (i.e., this class was important)
        results[cls_name] = (ablated_mean - baseline_mean) / (baseline_mean + 1e-10)

    return results


def calibrate_epsilon(
    train_distances: np.ndarray,
    train_rul: np.ndarray,
    basin_rul_threshold: int,
    target_recall: float = 0.90,
    approach_margin: int = 20,
) -> float:
    """Calibrate epsilon to detect trajectories approaching the failure basin.

    Uses the "approach zone" — trajectories with RUL between basin_rul_threshold
    and basin_rul_threshold + approach_margin — to set epsilon. These are
    trajectories that are near but not yet in B_f, which is the regime where
    early detection matters.

    Args:
        train_distances: (n_train_windows,) distance to B_f
        train_rul: (n_train_windows,) RUL values
        basin_rul_threshold: RUL cutoff defining B_f
        target_recall: desired fraction of approach-zone trajectories detected
        approach_margin: RUL range above basin defining the approach zone

    Returns:
        epsilon threshold value
    """
    # Approach zone: trajectories approaching but not yet in B_f
    approach_mask = (
        (train_rul > basin_rul_threshold)
        & (train_rul <= basin_rul_threshold + approach_margin)
    )
    if approach_mask.sum() == 0:
        # Fallback to precursors in B_f
        approach_mask = train_rul <= basin_rul_threshold

    approach_distances = np.sort(train_distances[approach_mask])
    # epsilon = the distance below which target_recall fraction fall
    idx = int(np.ceil(target_recall * len(approach_distances))) - 1
    idx = max(0, min(idx, len(approach_distances) - 1))
    return float(approach_distances[idx])


def precision_recall_sweep(
    test_final_distances: np.ndarray,
    test_true_rul: np.ndarray,
    warning_rul: int,
    n_thresholds: int = 200,
) -> dict:
    """Sweep epsilon and compute precision/recall at each threshold.

    An engine is a "true positive" if STTS fires (distance < epsilon)
    AND the engine's true RUL <= warning_rul. A "false positive" if
    STTS fires but true RUL > warning_rul.

    Args:
        test_final_distances: (n_engines,) final basin distance per engine
        test_true_rul: (n_engines,) true RUL at end of test sequence
        warning_rul: engines with true RUL <= this should be flagged
        n_thresholds: number of epsilon values to test

    Returns:
        dict with 'epsilons', 'precision', 'recall', 'f1', 'best_epsilon',
              'best_f1', 'best_precision', 'best_recall'
    """
    should_fire = test_true_rul <= warning_rul
    n_positive = should_fire.sum()

    epsilons = np.linspace(
        test_final_distances.min() * 0.5,
        test_final_distances.max() * 1.1,
        n_thresholds,
    )
    precisions = []
    recalls = []
    f1s = []

    for eps in epsilons:
        fired = test_final_distances < eps
        tp = (fired & should_fire).sum()
        fp = (fired & ~should_fire).sum()
        fn = (~fired & should_fire).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    best_idx = np.argmax(f1s)

    return {
        "epsilons": epsilons,
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
        "best_epsilon": float(epsilons[best_idx]),
        "best_f1": float(f1s[best_idx]),
        "best_precision": float(precisions[best_idx]),
        "best_recall": float(recalls[best_idx]),
        "n_positive": int(n_positive),
        "n_negative": int(len(test_true_rul) - n_positive),
    }
