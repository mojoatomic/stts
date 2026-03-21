"""STTS validation pipeline — C-MAPSS turbofan engine degradation.

Run: python -m pipeline.run_pipeline

Pipeline order (critical — do not reorder):
  1. Load & normalize sensors (equalize scales)
  2. Extract features F(T) from sliding windows
  3. StandardScaler on raw features (before weighting)
  4. Apply causal weights W (amplifies upstream features)
  5. Optional projection M (PCA/UMAP, NO additional scaling)
  6. Build failure basin B_f, run monitoring query
  7. Evaluate: precision/recall sweep, verification conditions
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.config import (
    DATASET, ACTIVE_SENSORS, WINDOW_SIZE, WINDOW_STRIDE,
    BASIN_RUL_THRESHOLD, BASIN_K, EMBEDDING_DIM, PROJECTION_METHOD,
    WARNING_RUL, THRESHOLD_SIGMA, RESULTS_DIR, USE_CAUSAL_WEIGHTS,
)
from pipeline.data_loader import (
    load_dataset, drop_flat_sensors, normalize_sensors, get_engine_data,
)
from pipeline.feature_extraction import (
    build_feature_matrix, get_feature_class_indices,
)
from pipeline.causal_weighting import build_weight_vector, apply_weights
from pipeline.manifold_projection import fit_scaler, fit_projection, project
from pipeline.failure_basin import (
    build_failure_basin, build_index, distance_to_basin, distance_to_corpus,
)
from pipeline.evaluation import (
    compute_stts_detection_cycle, compute_threshold_detection_cycle,
    intervention_window, verify_v1, verify_v2, calibrate_epsilon,
    precision_recall_sweep,
)
from pipeline import visualization as viz


def prepare_engine_arrays(engine_data: dict[int, pd.DataFrame]):
    """Convert engine DataFrames to numpy arrays of active sensors and RUL."""
    sensors = {}
    ruls = {}
    for uid, df in engine_data.items():
        cols = [c for c in ACTIVE_SENSORS if c in df.columns]
        sensors[uid] = df[cols].values
        if "rul" in df.columns:
            ruls[uid] = df["rul"].values
    return sensors, ruls


def main():
    print(f"=== STTS Validation Pipeline — {DATASET} ===")
    print(f"    Projection: {PROJECTION_METHOD}, d={EMBEDDING_DIM}")
    print(f"    Basin RUL threshold: {BASIN_RUL_THRESHOLD}")
    print(f"    Warning RUL: {WARNING_RUL}")
    print()

    # --- 1. Load and preprocess ---
    print("1. Loading data...")
    train_df, test_df, rul_truth = load_dataset(DATASET)
    train_df = drop_flat_sensors(train_df)
    test_df = drop_flat_sensors(test_df)
    train_df, test_df, sensor_scaler = normalize_sensors(train_df, test_df)

    active_cols = [c for c in ACTIVE_SENSORS if c in train_df.columns]
    n_sensors = len(active_cols)
    print(f"   Active sensors: {n_sensors}")
    print(f"   Train engines: {train_df['unit_id'].nunique()}")
    print(f"   Test engines: {test_df['unit_id'].nunique()}")

    # Training stats for threshold baseline
    train_sensor_means = train_df[active_cols].mean().values
    train_sensor_stds = train_df[active_cols].std().values

    train_engines = get_engine_data(train_df)
    test_engines = get_engine_data(test_df)
    train_sensors, train_ruls = prepare_engine_arrays(train_engines)
    test_sensors, _ = prepare_engine_arrays(test_engines)

    # --- 2. Feature extraction (Stage 1: F) ---
    print("\n2. Extracting features...")
    train_features, train_meta = build_feature_matrix(
        train_sensors, train_ruls, WINDOW_SIZE, WINDOW_STRIDE,
    )
    test_features, test_meta = build_feature_matrix(
        test_sensors, None, WINDOW_SIZE, WINDOW_STRIDE,
    )
    print(f"   Train: {train_features.shape}")
    print(f"   Test:  {test_features.shape}")

    # --- 3. StandardScaler BEFORE causal weighting ---
    print("\n3. Scaling features (before weighting)...")
    feature_scaler = fit_scaler(train_features)
    train_scaled = feature_scaler.transform(train_features)
    test_scaled = feature_scaler.transform(test_features)

    # --- 4. Causal weighting (Stage 2: W) ---
    if USE_CAUSAL_WEIGHTS:
        print("4. Applying causal weights...")
        weights = build_weight_vector(active_cols)
        train_weighted = apply_weights(train_scaled, weights)
        test_weighted = apply_weights(test_scaled, weights)
    else:
        print("4. Causal weights: DISABLED (uniform)")
        weights = np.ones(train_scaled.shape[1])
        train_weighted = train_scaled
        test_weighted = test_scaled

    # --- 5. Optional projection (Stage 3: M) ---
    print(f"5. Projection: {PROJECTION_METHOD}...")
    projector = fit_projection(train_weighted, PROJECTION_METHOD, EMBEDDING_DIM)
    train_embeddings = project(train_weighted, projector)
    test_embeddings = project(test_weighted, projector)
    print(f"   Embedding dim: {train_embeddings.shape[1]}")

    if PROJECTION_METHOD == "pca" and projector is not None:
        explained = projector.explained_variance_ratio_.sum()
        print(f"   PCA explained variance: {explained:.1%}")

    # --- 6. Build failure basin and compute distances ---
    print(f"\n6. Building failure basin (RUL <= {BASIN_RUL_THRESHOLD})...")
    basin = build_failure_basin(
        train_embeddings, train_meta["rul"], BASIN_RUL_THRESHOLD,
    )
    basin_index = build_index(basin)
    corpus_index = build_index(train_embeddings)
    print(f"   Basin size: {basin.shape[0]} embeddings")
    print(f"   Corpus size: {train_embeddings.shape[0]} embeddings")

    # Training distances for calibration and verification
    train_basin_distances = distance_to_basin(
        train_embeddings, basin_index, BASIN_K,
    )

    # Calibrate epsilon on training approach zone
    epsilon = calibrate_epsilon(
        train_basin_distances, train_meta["rul"], BASIN_RUL_THRESHOLD,
    )
    print(f"   Calibrated ε (approach zone): {epsilon:.4f}")

    # --- 7. Verification conditions ---
    print("\n7. Verification Conditions (training data)")

    precursor_mask = train_meta["rul"] <= BASIN_RUL_THRESHOLD
    nominal_mask = train_meta["rul"] > BASIN_RUL_THRESHOLD + 30
    v1 = verify_v1(
        train_basin_distances[precursor_mask],
        train_basin_distances[nominal_mask],
    )
    print(f"   V1 (Precursor proximity): {'PASS' if v1['passed'] else 'FAIL'}")
    print(f"       Median precursor: {v1['median_precursor']:.4f}")
    print(f"       Median nominal:   {v1['median_nominal']:.4f}")
    print(f"       Separation ratio: {v1['median_nominal'] / v1['median_precursor']:.1f}x")
    print(f"       Mann-Whitney p:   {v1['mannwhitney_p']:.2e}")

    v2 = verify_v2(train_basin_distances, train_meta["rul"])
    print(f"   V2 (Monotonic approach): {'PASS' if v2['passed'] else 'FAIL'}")
    print(f"       Spearman ρ:       {v2['spearman_rho']:.4f}")
    print(f"       p-value:          {v2['p_value']:.2e}")

    # --- 8. Test set evaluation ---
    print("\n8. Test Set Evaluation")

    test_engine_ids = sorted(test_engines.keys())
    per_engine_results = []

    for uid in test_engine_ids:
        engine_mask = test_meta["unit_id"] == uid
        engine_embeddings = test_embeddings[engine_mask]
        engine_cycles = test_meta["cycle_end"][engine_mask]

        if len(engine_embeddings) == 0:
            continue

        # Distances
        distances = distance_to_basin(engine_embeddings, basin_index, BASIN_K)
        ood_dists = distance_to_corpus(engine_embeddings, corpus_index, BASIN_K)

        # STTS detection
        stts_idx = compute_stts_detection_cycle(distances, epsilon)

        # Threshold detection
        threshold_idx = compute_threshold_detection_cycle(
            test_sensors[uid], train_sensor_means, train_sensor_stds,
            n_sigma=THRESHOLD_SIGMA,
        )

        true_rul = int(rul_truth[uid - 1])
        n_cycles = len(test_sensors[uid])
        total_life = n_cycles + true_rul

        # Convert window indices to absolute cycle positions
        stts_abs = (n_cycles - len(distances) + stts_idx) if stts_idx is not None else None
        threshold_abs = threshold_idx

        iw = intervention_window(stts_abs, threshold_abs, total_life)

        per_engine_results.append({
            "engine_id": uid,
            "true_rul": true_rul,
            "total_life": total_life,
            "final_basin_dist": float(distances[-1]),
            "min_basin_dist": float(distances.min()),
            "mean_ood": float(np.mean(ood_dists)),
            **iw,
        })

    results_df = pd.DataFrame(per_engine_results)
    RESULTS_DIR.mkdir(exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "test_results.csv", index=False)

    # --- 9. Precision-recall analysis ---
    print("\n9. Precision-Recall Analysis")
    final_dists = results_df["final_basin_dist"].values
    true_ruls = results_df["true_rul"].values

    pr = precision_recall_sweep(final_dists, true_ruls, WARNING_RUL)
    print(f"   Positive class: {pr['n_positive']} engines with RUL <= {WARNING_RUL}")
    print(f"   Negative class: {pr['n_negative']} engines with RUL > {WARNING_RUL}")
    print(f"   Best F1:        {pr['best_f1']:.3f}")
    print(f"   At ε:           {pr['best_epsilon']:.4f}")
    print(f"   Precision:      {pr['best_precision']:.3f}")
    print(f"   Recall:         {pr['best_recall']:.3f}")

    # Also report at the calibrated epsilon
    fired_cal = final_dists < epsilon
    should_fire = true_ruls <= WARNING_RUL
    tp_cal = (fired_cal & should_fire).sum()
    fp_cal = (fired_cal & ~should_fire).sum()
    fn_cal = (~fired_cal & should_fire).sum()
    p_cal = tp_cal / (tp_cal + fp_cal) if (tp_cal + fp_cal) > 0 else 0
    r_cal = tp_cal / (tp_cal + fn_cal) if (tp_cal + fn_cal) > 0 else 0
    print(f"\n   At calibrated ε={epsilon:.4f}:")
    print(f"   Precision: {p_cal:.3f}, Recall: {r_cal:.3f}")
    print(f"   TP={tp_cal}, FP={fp_cal}, FN={fn_cal}")

    # Intervention window analysis (using best F1 epsilon)
    best_eps = pr["best_epsilon"]
    print(f"\n   Intervention windows at best ε={best_eps:.4f}:")
    fired_best = results_df[results_df["final_basin_dist"] < best_eps]
    correct_fires = fired_best[fired_best["true_rul"] <= WARNING_RUL]
    if len(correct_fires) > 0:
        print(f"   True positives: {len(correct_fires)}")
        print(f"   Mean RUL at detection: {correct_fires['true_rul'].mean():.1f}")
        stts_wins = correct_fires[
            correct_fires["stts_fired"] & correct_fires["threshold_fired"]
            & (correct_fires["window_recovered"] > 0)
        ]
        both_fire = correct_fires[
            correct_fires["stts_fired"] & correct_fires["threshold_fired"]
        ]
        if len(both_fire) > 0:
            print(f"   Both STTS & threshold fired: {len(both_fire)}")
            print(f"   STTS earlier: {len(stts_wins)}/{len(both_fire)}")
            print(f"   Mean window recovered: {both_fire['window_recovered'].mean():.1f} cycles")

    # --- 10. Generate plots ---
    print("\n10. Generating plots...")
    viz.plot_embedding_2d(
        train_embeddings, train_meta["rul"], precursor_mask,
        title=f"STTS Embeddings — {DATASET}", filename=f"embedding_{DATASET}",
    )
    viz.plot_verification_v2(
        train_basin_distances, train_meta["rul"], v2["spearman_rho"],
        filename=f"v2_{DATASET}",
    )
    viz.plot_precision_recall(pr, filename=f"precision_recall_{DATASET}")

    # Distance curves for engines near the decision boundary
    interesting_engines = results_df.nsmallest(5, "true_rul")["engine_id"].tolist()
    for uid in interesting_engines:
        engine_mask = test_meta["unit_id"] == uid
        engine_embeddings = test_embeddings[engine_mask]
        engine_cycles = test_meta["cycle_end"][engine_mask]
        if len(engine_embeddings) == 0:
            continue
        distances = distance_to_basin(engine_embeddings, basin_index, BASIN_K)
        stts_idx = compute_stts_detection_cycle(distances, best_eps)
        true_rul = int(rul_truth[uid - 1])
        viz.plot_distance_curve(
            distances, engine_cycles, true_rul, best_eps, uid,
            stts_cycle=stts_idx,
            filename=f"distance_{DATASET}_engine_{uid}",
        )

    print(f"\n   Results saved to {RESULTS_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
