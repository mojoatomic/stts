"""STTS validation pipeline — C-MAPSS cross-validated LDA.

Run: python -m pipeline.run_cmapss

Reproduces the canonical C-MAPSS results from the paper (§6.1):
cross-validated LDA on held-out dataset pairs.

  LDA fit on FD002+FD004 → evaluate FD001, FD003
  LDA fit on FD001+FD003 → evaluate FD002, FD004

This is the proper held-out evaluation: no test data and no evaluation-
dataset training data enters the LDA fit.

Pipeline order:
  1. Load all four sub-datasets, normalize sensors
  2. Extract 264-dim features F(T) from sliding windows
  3. StandardScaler on raw features
  4. W = uniform
  5. M = 1-component LDA on RUL-bucketed classes (cross-validated)
  6. Build failure basin B_f, run monitoring query
  7. Evaluate: V1, V2, precision-recall, TSBP comparison
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from pipeline.config import (
    ACTIVE_SENSORS, DROP_SENSORS, WINDOW_SIZE, WINDOW_STRIDE,
    BASIN_RUL_THRESHOLD, BASIN_K, WARNING_RUL, RUL_CLIP,
)
from pipeline.data_loader import (
    load_dataset, drop_flat_sensors, normalize_sensors,
    normalize_by_regime, get_engine_data,
)
from pipeline.feature_extraction import build_feature_matrix
from pipeline.failure_basin import (
    build_failure_basin, build_index, distance_to_basin,
)
from pipeline.evaluation import (
    verify_v1, verify_v2, calibrate_epsilon, precision_recall_sweep,
)
from pipeline.tsbp_baseline import tsbp_predict_rul, smooth_sensors

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "cmapss"

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
MULTI_CONDITION = {"FD002", "FD004"}

# Cross-validation folds: fit LDA on one pair, evaluate on the other
CV_FOLDS = [
    {"fit": ["FD002", "FD004"], "eval": ["FD001", "FD003"]},
    {"fit": ["FD001", "FD003"], "eval": ["FD002", "FD004"]},
]

N_LDA_CLASSES = 6


def rul_bucket_labels(rul: np.ndarray) -> np.ndarray:
    """Bucket RUL into classes for LDA."""
    boundaries = np.linspace(0, RUL_CLIP, N_LDA_CLASSES + 1)
    labels = np.digitize(rul, boundaries) - 1
    return np.clip(labels, 0, N_LDA_CLASSES - 1)


def load_and_featurize(dataset_name: str) -> dict:
    """Load a C-MAPSS sub-dataset and extract features.

    Returns dict with train features/meta, test features/meta,
    raw test sensor arrays, and RUL truth.
    """
    train_df, test_df, rul_truth = load_dataset(dataset_name)
    train_df = drop_flat_sensors(train_df)
    test_df = drop_flat_sensors(test_df)

    if dataset_name in MULTI_CONDITION:
        train_df, test_df = normalize_by_regime(train_df, test_df)
    else:
        train_df, test_df, _ = normalize_sensors(train_df, test_df)

    active_cols = [c for c in ACTIVE_SENSORS if c in train_df.columns]
    train_engines = get_engine_data(train_df)
    test_engines = get_engine_data(test_df)

    # Raw sensor arrays for TSBP baseline
    train_sensors = {}
    train_ruls_raw = {}
    for uid, df in train_engines.items():
        train_sensors[uid] = df[active_cols].values
        train_ruls_raw[uid] = df["rul"].values

    test_sensors = {}
    for uid, df in test_engines.items():
        test_sensors[uid] = df[active_cols].values

    # Feature extraction
    train_sensor_arrays = {uid: df[active_cols].values for uid, df in train_engines.items()}
    train_rul_arrays = {uid: df["rul"].values for uid, df in train_engines.items()}
    test_sensor_arrays = {uid: df[active_cols].values for uid, df in test_engines.items()}

    train_features, train_meta = build_feature_matrix(
        train_sensor_arrays, train_rul_arrays, WINDOW_SIZE, WINDOW_STRIDE,
    )
    test_features, test_meta = build_feature_matrix(
        test_sensor_arrays, None, WINDOW_SIZE, WINDOW_STRIDE,
    )

    return {
        "name": dataset_name,
        "train_features": train_features,
        "train_meta": train_meta,
        "test_features": test_features,
        "test_meta": test_meta,
        "test_sensors": test_sensors,
        "train_sensors": train_sensors,
        "train_ruls": train_ruls_raw,
        "rul_truth": rul_truth,
        "n_train_engines": len(train_engines),
        "n_test_engines": len(test_engines),
    }


def run_tsbp(data: dict) -> dict:
    """Run TSBP baseline on a dataset, return per-engine predictions."""
    test_engine_ids = sorted(data["test_sensors"].keys())
    predictions = {}
    for uid in test_engine_ids:
        result = tsbp_predict_rul(
            data["test_sensors"][uid],
            data["train_sensors"],
            data["train_ruls"],
            match_window=WINDOW_SIZE,
        )
        predictions[uid] = result["predicted_rul"]
    return predictions


def evaluate_dataset(
    data: dict,
    scaler: StandardScaler,
    lda: LinearDiscriminantAnalysis,
    good_features: np.ndarray,
    fit_source: str,
) -> dict:
    """Evaluate one dataset with a pre-fitted scaler + LDA.

    Returns evaluation metrics.
    """
    ds = data["name"]

    # Transform test features
    test_scaled = scaler.transform(data["test_features"])[:, good_features]
    test_lda = lda.transform(test_scaled)

    # Also transform training features for basin construction
    train_scaled = scaler.transform(data["train_features"])[:, good_features]
    train_lda = lda.transform(train_scaled)

    # Build failure basin from this dataset's training data
    basin = build_failure_basin(train_lda, data["train_meta"]["rul"], BASIN_RUL_THRESHOLD)
    basin_index = build_index(basin)

    # Training verification conditions
    train_distances = distance_to_basin(train_lda, basin_index, BASIN_K)
    precursor_mask = data["train_meta"]["rul"] <= BASIN_RUL_THRESHOLD
    nominal_mask = data["train_meta"]["rul"] > BASIN_RUL_THRESHOLD + 30
    v1 = verify_v1(train_distances[precursor_mask], train_distances[nominal_mask])
    v2 = verify_v2(train_distances, data["train_meta"]["rul"])

    # Calibrate epsilon
    epsilon = calibrate_epsilon(train_distances, data["train_meta"]["rul"], BASIN_RUL_THRESHOLD)

    # Per-engine test evaluation (only engines with enough cycles for windowed features)
    all_test_engine_ids = sorted(data["test_sensors"].keys())
    eval_engine_ids = []
    final_dists = []
    true_ruls = []

    for uid in all_test_engine_ids:
        engine_mask = data["test_meta"]["unit_id"] == uid
        engine_lda = test_lda[engine_mask]
        if len(engine_lda) == 0:
            continue
        dists = distance_to_basin(engine_lda, basin_index, BASIN_K)
        eval_engine_ids.append(uid)
        final_dists.append(float(dists[-1]))
        true_ruls.append(int(data["rul_truth"][uid - 1]))

    final_dists = np.array(final_dists)
    true_ruls = np.array(true_ruls)

    # Precision-recall sweep
    pr = precision_recall_sweep(final_dists, true_ruls, WARNING_RUL)

    # TSBP baseline (aligned to same engine set)
    tsbp_preds = run_tsbp(data)
    tsbp_fired = np.array([tsbp_preds.get(uid, 999) <= WARNING_RUL for uid in eval_engine_ids])
    should_fire = true_ruls <= WARNING_RUL
    tsbp_tp = (tsbp_fired & should_fire).sum()
    tsbp_fp = (tsbp_fired & ~should_fire).sum()
    tsbp_fn = (~tsbp_fired & should_fire).sum()
    tsbp_p = tsbp_tp / max(tsbp_tp + tsbp_fp, 1)
    tsbp_r = tsbp_tp / max(tsbp_tp + tsbp_fn, 1)
    tsbp_f1 = 2 * tsbp_p * tsbp_r / max(tsbp_p + tsbp_r, 1e-10)

    return {
        "dataset": ds,
        "fit_source": fit_source,
        "v1_sep": v1["median_nominal"] / max(v1["median_precursor"], 1e-10),
        "v1_p": v1["mannwhitney_p"],
        "v2_rho": v2["spearman_rho"],
        "v2_p": v2["p_value"],
        "f1": pr["best_f1"],
        "precision": pr["best_precision"],
        "recall": pr["best_recall"],
        "tsbp_f1": tsbp_f1,
        "tsbp_precision": float(tsbp_p),
        "tsbp_recall": float(tsbp_r),
        "n_positive": pr["n_positive"],
        "n_negative": pr["n_negative"],
    }


def main():
    print("=== STTS C-MAPSS Cross-Validated LDA ===")
    print(f"    Datasets: {', '.join(DATASETS)}")
    print(f"    Window: {WINDOW_SIZE} cycles")
    print(f"    Basin RUL: <= {BASIN_RUL_THRESHOLD}")
    print(f"    Warning RUL: <= {WARNING_RUL}")
    print()

    # --- 1. Load and featurize all four datasets ---
    print("1. Loading and featurizing all datasets...")
    all_data = {}
    for ds in DATASETS:
        data = load_and_featurize(ds)
        all_data[ds] = data
        print(f"   {ds}: {data['n_train_engines']} train, {data['n_test_engines']} test, "
              f"train features: {data['train_features'].shape}")
    print()

    # --- 2. Cross-validated LDA ---
    print("2. Cross-validated LDA evaluation")
    print()

    all_results = []
    for fold in CV_FOLDS:
        fit_names = fold["fit"]
        eval_names = fold["eval"]
        fit_label = "+".join(fit_names)
        print(f"--- LDA fit: {fit_label} ---")

        # Pool training features from fit datasets
        fit_features = np.vstack([all_data[ds]["train_features"] for ds in fit_names])
        fit_ruls = np.concatenate([all_data[ds]["train_meta"]["rul"] for ds in fit_names])

        # StandardScaler
        scaler = StandardScaler()
        fit_scaled = scaler.fit_transform(fit_features)

        # Drop near-zero variance features
        feature_std = np.std(fit_scaled, axis=0)
        good_features = feature_std > 1e-8
        fit_scaled = fit_scaled[:, good_features]
        print(f"   Features: {fit_features.shape[1]} -> {good_features.sum()} (after variance filter)")

        # LDA
        labels = rul_bucket_labels(fit_ruls)
        lda = LinearDiscriminantAnalysis(n_components=1, solver="eigen", shrinkage="auto")
        lda.fit(fit_scaled, labels)

        # Evaluate on all four datasets
        for ds_name in DATASETS:
            held_out = ds_name in eval_names
            result = evaluate_dataset(all_data[ds_name], scaler, lda, good_features, fit_label)
            result["held_out"] = held_out
            all_results.append(result)

            marker = " (held-out)" if held_out else ""
            print(f"   {ds_name}{marker}: F1={result['f1']:.3f} (P={result['precision']:.3f}, "
                  f"R={result['recall']:.3f})  TSBP F1={result['tsbp_f1']:.3f}  "
                  f"V1={result['v1_sep']:.1f}x  V2 ρ={result['v2_rho']:.3f}")
        print()

    # --- 3. Summary: held-out results only ---
    print("3. Canonical held-out results (Table for §6.1)")
    print()
    held_out = [r for r in all_results if r["held_out"]]

    print(f"{'Dataset':<8} {'LDA fit':<14} {'F1':>6} {'Prec':>6} {'Rec':>6} {'TSBP F1':>8}")
    print("-" * 52)
    for r in held_out:
        print(f"{r['dataset']:<8} {r['fit_source']:<14} {r['f1']:>6.3f} "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['tsbp_f1']:>8.3f}")

    beats = sum(1 for r in held_out if r["f1"] > r["tsbp_f1"])
    print(f"\nSTTS beats TSBP on {beats}/{len(held_out)} held-out datasets")

    # V1/V2 summary
    print(f"\n{'Dataset':<8} {'V1 sep':>8} {'V1 p':>12} {'V2 ρ':>8} {'V2 p':>12}")
    print("-" * 52)
    for r in held_out:
        print(f"{r['dataset']:<8} {r['v1_sep']:>7.1f}x {r['v1_p']:>12.2e} "
              f"{r['v2_rho']:>8.3f} {r['v2_p']:>12.2e}")

    # --- 4. Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(RESULTS_DIR / "cross_validated_lda.csv", index=False)
    pd.DataFrame(held_out).to_csv(RESULTS_DIR / "held_out_results.csv", index=False)

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
