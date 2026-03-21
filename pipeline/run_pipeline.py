"""STTS validation pipeline — C-MAPSS turbofan engine degradation.

Run: python -m pipeline.run_pipeline
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.config import (
    DATASET, ACTIVE_SENSORS, WINDOW_SIZE, WINDOW_STRIDE,
    BASIN_RUL_THRESHOLD, BASIN_K, EMBEDDING_DIM, PROJECTION_METHOD,
    RESULTS_DIR,
)
from pipeline.data_loader import (
    load_dataset, drop_flat_sensors, normalize_sensors, get_engine_data,
)
from pipeline.feature_extraction import (
    build_feature_matrix, get_feature_class_indices,
)
from pipeline.causal_weighting import build_weight_vector, apply_weights
from pipeline.manifold_projection import fit_projection, project
from pipeline.failure_basin import (
    build_failure_basin, build_index, distance_to_basin, distance_to_corpus,
)
from pipeline.evaluation import (
    compute_stts_detection_cycle, compute_threshold_detection_cycle,
    intervention_window, verify_v1, verify_v2, calibrate_epsilon,
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
    print(f"=== STTS Validation Pipeline — {DATASET} ===\n")

    # --- 1. Load and preprocess ---
    print("Loading data...")
    train_df, test_df, rul_truth = load_dataset(DATASET)
    train_df = drop_flat_sensors(train_df)
    test_df = drop_flat_sensors(test_df)
    train_df, test_df, sensor_scaler = normalize_sensors(train_df, test_df)

    active_cols = [c for c in ACTIVE_SENSORS if c in train_df.columns]
    n_sensors = len(active_cols)
    print(f"  Active sensors: {n_sensors}")
    print(f"  Train engines: {train_df['unit_id'].nunique()}")
    print(f"  Test engines: {test_df['unit_id'].nunique()}")

    # Compute per-sensor training statistics for threshold baseline
    train_sensor_means = train_df[active_cols].mean().values
    train_sensor_stds = train_df[active_cols].std().values

    train_engines = get_engine_data(train_df)
    test_engines = get_engine_data(test_df)

    train_sensors, train_ruls = prepare_engine_arrays(train_engines)
    test_sensors, _ = prepare_engine_arrays(test_engines)

    # --- 2. Feature extraction (Stage 1: F) ---
    print("\nExtracting features...")
    train_features, train_meta = build_feature_matrix(
        train_sensors, train_ruls, WINDOW_SIZE, WINDOW_STRIDE,
    )
    test_features, test_meta = build_feature_matrix(
        test_sensors, None, WINDOW_SIZE, WINDOW_STRIDE,
    )
    print(f"  Train feature matrix: {train_features.shape}")
    print(f"  Test feature matrix: {test_features.shape}")

    # --- 3. Causal weighting (Stage 2: W) ---
    print("\nApplying causal weights...")
    weights = build_weight_vector(active_cols)
    train_weighted = apply_weights(train_features, weights)
    test_weighted = apply_weights(test_features, weights)

    # --- 4. Manifold projection (Stage 3: M) ---
    print(f"\nFitting {PROJECTION_METHOD.upper()} projection (d={EMBEDDING_DIM})...")
    projector, proj_scaler = fit_projection(
        train_weighted, PROJECTION_METHOD, EMBEDDING_DIM,
    )
    train_embeddings = project(train_weighted, projector, proj_scaler)
    test_embeddings = project(test_weighted, projector, proj_scaler)
    print(f"  Train embeddings: {train_embeddings.shape}")
    print(f"  Test embeddings: {test_embeddings.shape}")

    if PROJECTION_METHOD == "pca":
        explained = projector.explained_variance_ratio_.sum()
        print(f"  PCA explained variance: {explained:.1%}")

    # --- 5. Build failure basin ---
    print(f"\nBuilding failure basin (RUL <= {BASIN_RUL_THRESHOLD})...")
    basin = build_failure_basin(
        train_embeddings, train_meta["rul"], BASIN_RUL_THRESHOLD,
    )
    basin_index = build_index(basin)
    corpus_index = build_index(train_embeddings)
    print(f"  Basin size: {basin.shape[0]} embeddings")

    # Compute train distances for calibration
    train_basin_distances = distance_to_basin(
        train_embeddings, basin_index, BASIN_K,
    )

    # Calibrate epsilon
    epsilon = calibrate_epsilon(
        train_basin_distances, train_meta["rul"], BASIN_RUL_THRESHOLD,
    )
    print(f"  Calibrated ε: {epsilon:.4f}")

    # --- 6. Verification conditions on training data ---
    print("\n--- Verification Conditions ---")

    # V1: Precursor proximity
    precursor_mask = train_meta["rul"] <= BASIN_RUL_THRESHOLD
    nominal_mask = train_meta["rul"] > BASIN_RUL_THRESHOLD + 30  # well above basin
    v1 = verify_v1(
        train_basin_distances[precursor_mask],
        train_basin_distances[nominal_mask],
    )
    print(f"  V1 (Precursor proximity): {'PASS' if v1['passed'] else 'FAIL'}")
    print(f"      Median precursor dist: {v1['median_precursor']:.4f}")
    print(f"      Median nominal dist:   {v1['median_nominal']:.4f}")
    print(f"      Mann-Whitney p:        {v1['mannwhitney_p']:.2e}")

    # V2: Monotonic approach
    v2 = verify_v2(train_basin_distances, train_meta["rul"])
    print(f"  V2 (Monotonic approach):   {'PASS' if v2['passed'] else 'FAIL'}")
    print(f"      Spearman ρ:            {v2['spearman_rho']:.4f}")
    print(f"      p-value:               {v2['p_value']:.2e}")

    # --- 7. Test set evaluation ---
    print("\n--- Test Set Evaluation ---")

    test_engine_ids = sorted(test_engines.keys())
    results = []

    for uid in test_engine_ids:
        # Get this engine's embeddings
        engine_mask = test_meta["unit_id"] == uid
        engine_embeddings = test_embeddings[engine_mask]
        engine_cycles = test_meta["cycle_end"][engine_mask]

        if len(engine_embeddings) == 0:
            continue

        # Distance to failure basin
        distances = distance_to_basin(engine_embeddings, basin_index, BASIN_K)

        # OOD signal
        ood_distances = distance_to_corpus(engine_embeddings, corpus_index, BASIN_K)

        # STTS detection
        stts_idx = compute_stts_detection_cycle(distances, epsilon)

        # Threshold detection
        engine_sensor_data = test_sensors[uid]
        threshold_idx = compute_threshold_detection_cycle(
            engine_sensor_data, train_sensor_means, train_sensor_stds,
        )

        # True RUL at end of test sequence
        true_rul_at_end = int(rul_truth[uid - 1])  # 0-indexed
        total_life = len(engine_sensor_data) + true_rul_at_end

        # Intervention window (relative to cycles in the sensor data)
        n_cycles = len(engine_sensor_data)
        stts_cycle_abs = (n_cycles - WINDOW_SIZE + stts_idx) if stts_idx is not None else None
        threshold_cycle_abs = threshold_idx

        iw = intervention_window(
            stts_cycle_abs, threshold_cycle_abs, n_cycles + true_rul_at_end,
        )

        results.append({
            "engine_id": uid,
            "true_rul": true_rul_at_end,
            "total_life": total_life,
            **iw,
            "mean_ood": float(np.mean(ood_distances)),
            "final_basin_dist": float(distances[-1]),
        })

    results_df = pd.DataFrame(results)
    RESULTS_DIR.mkdir(exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "test_results.csv", index=False)

    # Summary
    fired = results_df[results_df["stts_fired"]]
    print(f"\n  STTS fired on {len(fired)}/{len(results_df)} test engines")
    if len(fired) > 0:
        print(f"  Mean STTS lead:        {fired['stts_lead'].mean():.1f} cycles")
        print(f"  Mean threshold lead:   {fired['threshold_lead'].mean():.1f} cycles")
        print(f"  Mean window recovered: {fired['window_recovered'].mean():.1f} cycles")
        print(f"  Median window:         {fired['window_recovered'].median():.1f} cycles")

    both_fired = results_df[results_df["stts_fired"] & results_df["threshold_fired"]]
    if len(both_fired) > 0:
        wins = (both_fired["window_recovered"] > 0).sum()
        print(f"\n  STTS earlier than threshold: {wins}/{len(both_fired)} engines")

    # --- 8. Visualizations ---
    print("\nGenerating plots...")

    viz.plot_embedding_2d(
        train_embeddings, train_meta["rul"], precursor_mask,
        title=f"STTS Embeddings — {DATASET}", filename=f"embedding_{DATASET}",
    )
    viz.plot_verification_v2(
        train_basin_distances, train_meta["rul"], v2["spearman_rho"],
        filename=f"v2_{DATASET}",
    )
    viz.plot_intervention_windows(results, filename=f"intervention_{DATASET}")

    # Plot distance curves for a few representative engines
    for uid in test_engine_ids[:5]:
        engine_mask = test_meta["unit_id"] == uid
        engine_embeddings = test_embeddings[engine_mask]
        engine_cycles = test_meta["cycle_end"][engine_mask]
        if len(engine_embeddings) == 0:
            continue
        distances = distance_to_basin(engine_embeddings, basin_index, BASIN_K)
        stts_idx = compute_stts_detection_cycle(distances, epsilon)
        true_rul = int(rul_truth[uid - 1])
        viz.plot_distance_curve(
            distances, engine_cycles, true_rul, epsilon, uid,
            stts_cycle=stts_idx,
            filename=f"distance_{DATASET}_engine_{uid}",
        )

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
