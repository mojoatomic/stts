"""TSBP (Trajectory Similarity Based Prediction) baseline — Wang et al. 2008.

Implements the core TSBP algorithm for comparison with STTS:
1. Build a library of run-to-failure sensor trajectories (training set)
2. For each test engine, find the most similar training trajectory segment
3. Predict RUL from the matched position in the reference trajectory
4. Compute detection metrics using the same framework as STTS evaluation

The comparison answers: does STTS's framework (feature extraction + embedding
+ failure basin) outperform direct trajectory matching on raw sensor data?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def smooth_sensors(data: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply moving average smoothing to sensor data.

    Wang 2008 smoothed sensor data before comparison.
    """
    if window <= 1:
        return data
    kernel = np.ones(window) / window
    smoothed = np.zeros_like(data)
    for col in range(data.shape[1]):
        smoothed[:, col] = np.convolve(data[:, col], kernel, mode='same')
    return smoothed


def tsbp_predict_rul(
    test_trajectory: np.ndarray,
    train_trajectories: dict[int, np.ndarray],
    train_ruls: dict[int, np.ndarray],
    match_window: int = 30,
    top_k: int = 5,
    smooth_window: int = 5,
) -> dict:
    """Predict RUL for a test engine using TSBP.

    For the test engine's last `match_window` cycles, find the most similar
    segment in each training trajectory. Predict RUL from the matched position.

    Args:
        test_trajectory: (n_test_cycles, n_sensors)
        train_trajectories: {uid: (n_cycles, n_sensors)}
        train_ruls: {uid: (n_cycles,) RUL array}
        match_window: number of cycles to match
        top_k: number of best matches to average
        smooth_window: moving average window for smoothing

    Returns:
        dict with predicted_rul, distances to matches, matched positions
    """
    # Smooth
    test_smooth = smooth_sensors(test_trajectory, smooth_window)

    # Extract the tail of the test trajectory
    if len(test_smooth) < match_window:
        match_window = len(test_smooth)
    test_tail = test_smooth[-match_window:]  # (match_window, n_sensors)

    # Compare against every position in every training trajectory
    matches = []
    for uid, train_data in train_trajectories.items():
        train_smooth = smooth_sensors(train_data, smooth_window)
        rul = train_ruls[uid]

        if len(train_smooth) < match_window:
            continue

        # Slide along the training trajectory
        for start in range(len(train_smooth) - match_window + 1):
            segment = train_smooth[start:start + match_window]
            # Euclidean distance between test tail and this segment
            dist = np.sqrt(np.mean((test_tail - segment) ** 2))
            # RUL at the end of this segment
            segment_rul = rul[start + match_window - 1]
            matches.append({
                'train_uid': uid,
                'position': start + match_window - 1,
                'distance': dist,
                'rul_at_match': segment_rul,
            })

    # Sort by distance, take top-k
    matches.sort(key=lambda m: m['distance'])
    top_matches = matches[:top_k]

    # Weighted average RUL prediction (inverse distance weighting)
    if len(top_matches) == 0:
        return {'predicted_rul': None, 'matches': []}

    dists = np.array([m['distance'] for m in top_matches])
    ruls = np.array([m['rul_at_match'] for m in top_matches])

    if np.all(dists == 0):
        predicted_rul = np.mean(ruls)
    else:
        weights = 1.0 / (dists + 1e-10)
        predicted_rul = np.average(ruls, weights=weights)

    return {
        'predicted_rul': float(predicted_rul),
        'min_distance': float(dists[0]),
        'matches': top_matches,
    }


def tsbp_evaluate(
    test_sensors: dict[int, np.ndarray],
    train_sensors: dict[int, np.ndarray],
    train_ruls: dict[int, np.ndarray],
    true_ruls: np.ndarray,
    warning_rul: int = 50,
    match_window: int = 30,
    top_k: int = 5,
) -> dict:
    """Run TSBP on all test engines and compute detection metrics.

    Uses the same positive/negative definition as STTS evaluation:
    an engine should be flagged if true RUL <= warning_rul.

    For TSBP, "fired" = predicted RUL <= warning_rul.
    """
    test_ids = sorted(test_sensors.keys())
    results = []

    for i, uid in enumerate(test_ids):
        pred = tsbp_predict_rul(
            test_sensors[uid], train_sensors, train_ruls,
            match_window=match_window, top_k=top_k,
        )
        true_rul = int(true_ruls[i])
        predicted = pred['predicted_rul']
        results.append({
            'engine_id': uid,
            'true_rul': true_rul,
            'predicted_rul': predicted,
            'error': (predicted - true_rul) if predicted is not None else None,
        })

    df = pd.DataFrame(results)

    # RUL prediction metrics (standard PHM metrics)
    valid = df[df['predicted_rul'].notna()]
    errors = valid['error'].values
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))

    # PHM08 scoring function (asymmetric)
    score = 0
    for e in errors:
        if e < 0:  # early prediction
            score += np.exp(-e / 13) - 1
        else:  # late prediction
            score += np.exp(e / 10) - 1

    # Detection metrics (same framework as STTS)
    should_fire = valid['true_rul'] <= warning_rul
    fired = valid['predicted_rul'] <= warning_rul
    tp = (fired & should_fire).sum()
    fp = (fired & ~should_fire).sum()
    fn = (~fired & should_fire).sum()
    tn = (~fired & ~should_fire).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'rmse': rmse,
        'mae': mae,
        'phm08_score': score,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'per_engine': df,
    }


if __name__ == '__main__':
    import pandas as pd
    from pipeline.data_loader import load_dataset, drop_flat_sensors, normalize_sensors, get_engine_data
    from pipeline.config import ACTIVE_SENSORS

    train_df, test_df, rul_truth = load_dataset('FD001')
    train_df = drop_flat_sensors(train_df)
    test_df = drop_flat_sensors(test_df)
    train_df, test_df, _ = normalize_sensors(train_df, test_df)
    active = [c for c in ACTIVE_SENSORS if c in train_df.columns]
    train_engines = get_engine_data(train_df)
    test_engines = get_engine_data(test_df)
    train_sensors = {uid: df[active].values for uid, df in train_engines.items()}
    train_ruls = {uid: df['rul'].values for uid, df in train_engines.items()}
    test_sensors = {uid: df[active].values for uid, df in test_engines.items()}

    print("Running TSBP baseline on FD001...")
    results = tsbp_evaluate(
        test_sensors, train_sensors, train_ruls, rul_truth,
        warning_rul=50, match_window=30, top_k=5,
    )
    print(f"\nTSBP Results:")
    print(f"  RMSE:       {results['rmse']:.2f}")
    print(f"  MAE:        {results['mae']:.2f}")
    print(f"  PHM08 Score: {results['phm08_score']:.0f}")
    print(f"  Precision:  {results['precision']:.3f}")
    print(f"  Recall:     {results['recall']:.3f}")
    print(f"  F1:         {results['f1']:.3f}")
    print(f"  TP={results['tp']}, FP={results['fp']}, FN={results['fn']}, TN={results['tn']}")

    results['per_engine'].to_csv('results/tsbp_results_FD001.csv', index=False)
    print("\nSaved to results/tsbp_results_FD001.csv")
