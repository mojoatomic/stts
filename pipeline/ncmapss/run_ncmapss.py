"""STTS validation pipeline — N-CMAPSS (Chao et al. 2021).

Run: python -m pipeline.ncmapss.run_ncmapss [--datasets DS01 DS03 ...]

Adapts the STTS F->W->M pipeline to the N-CMAPSS dataset.
Only the F stage (feature extraction / data loading) changes:
  - HDF5 format with whitelisted keys (excludes X_v, T to prevent leakage)
  - Cruise-phase filter (TRA stability + altitude) removes flight transients
  - Per-cycle mean aggregation (14 sensor channels)
  - Regime normalization using flight conditions (W) removes operating-point
    variation so features measure degradation, not flight profile

The W stage behavior is identical: USE_CAUSAL_WEIGHTS = False in both
pipelines, so the causal weight config is documentation-only.
The M stage (LDA projection) and evaluation are unchanged.

Pipeline order:
  1. Load HDF5 (whitelisted keys only)
  2. Cruise-phase filter + per-cycle aggregation
  3. Compute RUL thresholds from dev data distribution
  4. Regime normalization (cluster on W, normalize sensors per regime)
  5. Extract windowed features F(T) from cycle-level data
  6. StandardScaler on windowed features
  7. M = 1-component LDA on RUL-bucketed classes
  8. Serialize artifacts (scaler, LDA, config snapshot)
  9. Build failure basin B_f, run monitoring query
  10. Evaluate: V1, V2, precision-recall
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from pipeline.ncmapss import config as ncmapss_config
from pipeline.ncmapss.config import (
    ACTIVE_SENSORS, WINDOW_SIZE, WINDOW_STRIDE,
    BASIN_K, RESULTS_DIR, DATA_DIR, N_LDA_CLASSES,
    SUBDATASETS, ARTIFACTS_DIR, THRESHOLD_SIGMA,
    compute_rul_thresholds,
)
from pipeline.ncmapss.data_loader import (
    load_dataset, normalize_by_regime, get_engine_data,
)
from pipeline.feature_extraction import build_feature_matrix
from pipeline.failure_basin import (
    build_failure_basin, build_index, distance_to_basin,
)
from pipeline.evaluation import (
    verify_v1, verify_v2, calibrate_epsilon, precision_recall_sweep,
)


# ---------------------------------------------------------------------------
# Config snapshot and artifact serialization
# ---------------------------------------------------------------------------

def _config_snapshot() -> dict:
    """Capture the current config state for reproducibility."""
    return {
        "sensor_names": ncmapss_config.SENSOR_NAMES,
        "active_sensors": ncmapss_config.ACTIVE_SENSORS,
        "cycle_agg_function": ncmapss_config.CYCLE_AGG_FUNCTION,
        "cruise_tra_window": ncmapss_config.CRUISE_TRA_WINDOW,
        "cruise_tra_std_max": ncmapss_config.CRUISE_TRA_STD_MAX,
        "cruise_alt_min": ncmapss_config.CRUISE_ALT_MIN,
        "cruise_min_timesteps": ncmapss_config.CRUISE_MIN_TIMESTEPS,
        "n_regimes": ncmapss_config.N_REGIMES,
        "window_size": ncmapss_config.WINDOW_SIZE,
        "window_stride": ncmapss_config.WINDOW_STRIDE,
        "fft_num_freqs": ncmapss_config.FFT_NUM_FREQS,
        "rul_clip_config": ncmapss_config.RUL_CLIP,
        "basin_rul_threshold_config": ncmapss_config.BASIN_RUL_THRESHOLD,
        "warning_rul_config": ncmapss_config.WARNING_RUL,
        "basin_k": ncmapss_config.BASIN_K,
        "threshold_sigma": ncmapss_config.THRESHOLD_SIGMA,
        "n_lda_classes": ncmapss_config.N_LDA_CLASSES,
        "use_causal_weights": ncmapss_config.USE_CAUSAL_WEIGHTS,
        "causal_weight_values": ncmapss_config.CAUSAL_WEIGHT_VALUES,
        "sensor_causal_groups": ncmapss_config.SENSOR_CAUSAL_GROUPS,
        "feature_class_multipliers": ncmapss_config.FEATURE_CLASS_MULTIPLIERS,
    }


def _md5(filepath: Path) -> str:
    """Compute MD5 checksum of a file."""
    h = hashlib.md5()
    h.update(filepath.read_bytes())
    return h.hexdigest()


def save_artifacts(
    scaler: StandardScaler,
    lda: LinearDiscriminantAnalysis,
    good_features: np.ndarray,
    fit_label: str,
    runtime_thresholds: dict,
) -> dict:
    """Serialize fitted scaler and LDA to disk. Return artifact checksums."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = fit_label.replace("+", "_")

    scaler_path = ARTIFACTS_DIR / f"scaler_{prefix}.pkl"
    lda_path = ARTIFACTS_DIR / f"lda_{prefix}.pkl"
    features_path = ARTIFACTS_DIR / f"good_features_{prefix}.npy"

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(lda_path, "wb") as f:
        pickle.dump(lda, f)
    np.save(features_path, good_features)

    checksums = {
        "scaler_md5": _md5(scaler_path),
        "lda_md5": _md5(lda_path),
        "good_features_md5": _md5(features_path),
        "scaler_path": str(scaler_path),
        "lda_path": str(lda_path),
        "good_features_path": str(features_path),
    }
    checksums.update(runtime_thresholds)
    return checksums


# ---------------------------------------------------------------------------
# RUL bucketing and thresholds
# ---------------------------------------------------------------------------

def resolve_thresholds(dev_ruls: np.ndarray) -> dict:
    """Resolve RUL thresholds — use config values if set, else compute."""
    if ncmapss_config.RUL_CLIP is not None:
        rul_clip = ncmapss_config.RUL_CLIP
    else:
        rul_clip = compute_rul_thresholds(dev_ruls)["rul_clip"]

    if ncmapss_config.BASIN_RUL_THRESHOLD is not None:
        basin_rul_threshold = ncmapss_config.BASIN_RUL_THRESHOLD
    else:
        basin_rul_threshold = int(round(0.20 * rul_clip))

    if ncmapss_config.WARNING_RUL is not None:
        warning_rul = ncmapss_config.WARNING_RUL
    else:
        warning_rul = int(round(0.40 * rul_clip))

    return {
        "rul_clip": rul_clip,
        "basin_rul_threshold": basin_rul_threshold,
        "warning_rul": warning_rul,
        "thresholds_source": "data-driven" if ncmapss_config.RUL_CLIP is None else "config",
    }


def rul_bucket_labels(rul: np.ndarray, rul_clip: int) -> np.ndarray:
    """Bucket RUL into classes for LDA."""
    boundaries = np.linspace(0, rul_clip, N_LDA_CLASSES + 1)
    labels = np.digitize(rul, boundaries) - 1
    return np.clip(labels, 0, N_LDA_CLASSES - 1)


# ---------------------------------------------------------------------------
# Load and featurize
# ---------------------------------------------------------------------------

def load_and_featurize(dataset_name: str, rul_clip: int) -> dict:
    """Load an N-CMAPSS sub-dataset and extract features.

    After cruise-phase filtering and mean-only per-cycle aggregation, each
    cycle has 14 sensor channels. Regime normalization removes operating-point
    variation using flight conditions (W), then the windowed feature extractor
    produces time-domain (4), rate (3), frequency (FFT_NUM_FREQS), and
    covariance features over these 14 channels.
    """
    dev_df, test_df = load_dataset(dataset_name, rul_clip=rul_clip)

    # Regime normalization: cluster on W (dev only), normalize sensors per regime
    dev_df, test_df, _ = normalize_by_regime(dev_df, test_df)

    dev_engines = get_engine_data(dev_df)
    test_engines = get_engine_data(test_df)

    # Build sensor arrays — 14 columns per cycle
    cols = [c for c in ACTIVE_SENSORS if c in dev_df.columns]

    dev_sensor_arrays = {
        uid: df[cols].values for uid, df in dev_engines.items()
    }
    dev_rul_arrays = {
        uid: df["rul"].values for uid, df in dev_engines.items()
    }
    test_sensor_arrays = {
        uid: df[cols].values for uid, df in test_engines.items()
    }
    test_rul_arrays = {
        uid: df["rul"].values for uid, df in test_engines.items()
    }

    dev_features, dev_meta = build_feature_matrix(
        dev_sensor_arrays, dev_rul_arrays, WINDOW_SIZE, WINDOW_STRIDE,
    )
    test_features, test_meta = build_feature_matrix(
        test_sensor_arrays, test_rul_arrays, WINDOW_SIZE, WINDOW_STRIDE,
    )

    return {
        "name": dataset_name,
        "dev_features": dev_features,
        "dev_meta": dev_meta,
        "test_features": test_features,
        "test_meta": test_meta,
        "n_dev_engines": len(dev_engines),
        "n_test_engines": len(test_engines),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    data: dict,
    scaler: StandardScaler,
    lda: LinearDiscriminantAnalysis,
    good_features: np.ndarray,
    fit_source: str,
    thresholds: dict,
) -> dict:
    """Evaluate one dataset with a pre-fitted scaler + LDA.

    Uses .transform() only — never .fit(). All fitting happens in main().
    """
    ds = data["name"]
    basin_rul_threshold = thresholds["basin_rul_threshold"]
    warning_rul = thresholds["warning_rul"]

    test_scaled = scaler.transform(data["test_features"])[:, good_features]
    test_lda = lda.transform(test_scaled)

    dev_scaled = scaler.transform(data["dev_features"])[:, good_features]
    dev_lda = lda.transform(dev_scaled)

    # Build failure basin from dev data
    basin = build_failure_basin(dev_lda, data["dev_meta"]["rul"], basin_rul_threshold)
    basin_index = build_index(basin)

    # V1/V2 on dev data
    dev_distances = distance_to_basin(dev_lda, basin_index, BASIN_K)
    precursor_mask = data["dev_meta"]["rul"] <= basin_rul_threshold
    nominal_mask = data["dev_meta"]["rul"] > basin_rul_threshold + 30
    v1 = verify_v1(dev_distances[precursor_mask], dev_distances[nominal_mask])
    v2 = verify_v2(dev_distances, data["dev_meta"]["rul"])

    # Test evaluation — per-engine final distance
    test_engine_ids = sorted(set(data["test_meta"]["unit_id"]))
    final_dists = []
    true_ruls = []

    for uid in test_engine_ids:
        engine_mask = data["test_meta"]["unit_id"] == uid
        engine_lda = test_lda[engine_mask]
        if len(engine_lda) == 0:
            continue
        dists = distance_to_basin(engine_lda, basin_index, BASIN_K)
        final_dists.append(float(dists[-1]))
        engine_ruls = data["test_meta"]["rul"][engine_mask]
        true_ruls.append(int(engine_ruls[-1]))

    final_dists = np.array(final_dists)
    true_ruls = np.array(true_ruls)

    pr = precision_recall_sweep(final_dists, true_ruls, warning_rul)

    return {
        "dataset": ds,
        "fit_source": fit_source,
        "v1_sep": v1["separation_ratio"],
        "v1_p": v1["mannwhitney_p"],
        "v2_rho": v2["spearman_rho"],
        "v2_p": v2["p_value"],
        "f1": pr["best_f1"],
        "precision": pr["best_precision"],
        "recall": pr["best_recall"],
        "n_positive": pr["n_positive"],
        "n_negative": pr["n_negative"],
        "n_dev_engines": data["n_dev_engines"],
        "n_test_engines": data["n_test_engines"],
        **thresholds,
    }


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def discover_datasets() -> list[str]:
    """Find available N-CMAPSS .h5 files in the data directory."""
    from pipeline.ncmapss.data_loader import find_h5_file
    available = []
    for name in SUBDATASETS:
        try:
            find_h5_file(name)
            available.append(name)
        except FileNotFoundError:
            pass
    return available


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="STTS N-CMAPSS pipeline")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Sub-datasets to evaluate (e.g., DS01 DS03). Default: all available.",
    )
    args = parser.parse_args()

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = discover_datasets()

    if not datasets:
        print(f"No N-CMAPSS .h5 files found in {DATA_DIR}")
        print("Download from: https://phm-datasets.s3.amazonaws.com/NASA/"
              "17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip")
        return

    print("=== STTS N-CMAPSS Pipeline ===")
    print(f"    Datasets: {', '.join(datasets)}")
    print(f"    Window: {WINDOW_SIZE} cycles")
    print(f"    Sensors: {len(ACTIVE_SENSORS)} (cruise-filtered, mean-aggregated)")
    print(f"    Cruise filter: TRA std < {ncmapss_config.CRUISE_TRA_STD_MAX}%, "
          f"alt > {ncmapss_config.CRUISE_ALT_MIN} ft")
    print(f"    Regime normalization: {ncmapss_config.N_REGIMES} clusters on W")
    print()

    # --- 1. Load raw RUL to compute thresholds ---
    print("1. Computing RUL thresholds from dev data...")

    # Collect raw dev RULs across all requested datasets for threshold estimation.
    # We load just the first dataset initially to get RUL distribution, then
    # use those thresholds consistently across all datasets.
    first_dev, _ = load_dataset(datasets[0], rul_clip=None)
    raw_dev_ruls = first_dev["rul"].values
    thresholds = resolve_thresholds(raw_dev_ruls)

    print(f"   RUL clip: {thresholds['rul_clip']} "
          f"(source: {thresholds['thresholds_source']})")
    print(f"   Basin RUL threshold: {thresholds['basin_rul_threshold']}")
    print(f"   Warning RUL: {thresholds['warning_rul']}")
    print(f"   Dev RUL range: [{int(raw_dev_ruls.min())}, {int(raw_dev_ruls.max())}], "
          f"median={int(np.median(raw_dev_ruls))}")
    print()

    rul_clip = thresholds["rul_clip"]

    # --- 2. Load and featurize all datasets ---
    print("2. Loading and featurizing datasets...")
    all_data = {}
    for ds in datasets:
        try:
            data = load_and_featurize(ds, rul_clip=rul_clip)
            all_data[ds] = data
            print(f"   {ds}: {data['n_dev_engines']} dev, "
                  f"{data['n_test_engines']} test, "
                  f"dev features: {data['dev_features'].shape}")
        except FileNotFoundError as e:
            print(f"   {ds}: SKIPPED — {e}")
    print()

    if not all_data:
        print("No datasets loaded successfully.")
        return

    # --- 3. Per-dataset LDA (self-fit) ---
    print("3. Per-dataset LDA evaluation (fit on own dev, eval on test)")
    print()

    config_snap = _config_snapshot()
    all_results = []

    for ds_name, data in all_data.items():
        print(f"--- {ds_name} ---")

        # StandardScaler — fitted here, serialized below
        scaler = StandardScaler()
        dev_scaled = scaler.fit_transform(data["dev_features"])

        # Drop near-zero variance features
        feature_std = np.std(dev_scaled, axis=0)
        good_features = feature_std > 1e-8
        dev_scaled = dev_scaled[:, good_features]
        print(f"   Features: {data['dev_features'].shape[1]} -> "
              f"{good_features.sum()} (after variance filter)")

        # LDA — fitted here, serialized below
        labels = rul_bucket_labels(data["dev_meta"]["rul"], rul_clip)
        lda = LinearDiscriminantAnalysis(
            n_components=1, solver="eigen", shrinkage="auto",
        )
        lda.fit(dev_scaled, labels)

        # Serialize artifacts
        artifact_info = save_artifacts(
            scaler, lda, good_features, ds_name, thresholds,
        )

        result = evaluate_dataset(
            data, scaler, lda, good_features, ds_name, thresholds,
        )
        result["artifact_checksums"] = artifact_info
        all_results.append(result)

        print(f"   F1={result['f1']:.3f} (P={result['precision']:.3f}, "
              f"R={result['recall']:.3f})  "
              f"V1={result['v1_sep']:.1f}x  V2 rho={result['v2_rho']:.3f}")
        print()

    # --- 4. Cross-dataset LDA (if multiple datasets available) ---
    if len(all_data) > 1:
        print("4. Cross-dataset LDA (leave-one-out)")
        print()

        ds_names = list(all_data.keys())
        for eval_ds in ds_names:
            fit_names = [d for d in ds_names if d != eval_ds]
            fit_label = "+".join(fit_names)
            print(f"--- LDA fit: {fit_label}, eval: {eval_ds} ---")

            fit_features = np.vstack(
                [all_data[d]["dev_features"] for d in fit_names]
            )
            fit_ruls = np.concatenate(
                [all_data[d]["dev_meta"]["rul"] for d in fit_names]
            )

            scaler = StandardScaler()
            fit_scaled = scaler.fit_transform(fit_features)

            feature_std = np.std(fit_scaled, axis=0)
            good_features = feature_std > 1e-8
            fit_scaled = fit_scaled[:, good_features]

            labels = rul_bucket_labels(fit_ruls, rul_clip)
            lda = LinearDiscriminantAnalysis(
                n_components=1, solver="eigen", shrinkage="auto",
            )
            lda.fit(fit_scaled, labels)

            artifact_info = save_artifacts(
                scaler, lda, good_features, fit_label, thresholds,
            )

            result = evaluate_dataset(
                all_data[eval_ds], scaler, lda, good_features,
                fit_label, thresholds,
            )
            result["held_out"] = True
            result["artifact_checksums"] = artifact_info
            all_results.append(result)

            print(f"   F1={result['f1']:.3f} (P={result['precision']:.3f}, "
                  f"R={result['recall']:.3f})  "
                  f"V1={result['v1_sep']:.1f}x  V2 rho={result['v2_rho']:.3f}")

        print()

    # --- 5. Summary ---
    print("5. Summary")
    print()
    print(f"{'Dataset':<8} {'LDA fit':<20} {'F1':>6} {'Prec':>6} {'Rec':>6} "
          f"{'V1 sep':>8} {'V2 rho':>8}")
    print("-" * 64)
    for r in all_results:
        print(f"{r['dataset']:<8} {r['fit_source']:<20} {r['f1']:>6.3f} "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} "
              f"{r['v1_sep']:>7.1f}x {r['v2_rho']:>8.3f}")

    # --- 6. Save results with config snapshot ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # CSV for quick inspection
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / "ncmapss_results.csv", index=False)

    # JSON with full config snapshot for reproducibility
    results_json = {
        "config_snapshot": config_snap,
        "runtime_thresholds": thresholds,
        "results": all_results,
    }
    with open(RESULTS_DIR / "ncmapss_results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"Artifacts saved to {ARTIFACTS_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
