"""Post-hoc analysis for N-CMAPSS results: per-engine V2, Wilson CI,
wall-clock timing, and LDA loading analysis (V3).

Run: python -m pipeline.ncmapss.analysis [--datasets DS03]
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from pipeline.ncmapss.config import (
    ACTIVE_SENSORS, WINDOW_SIZE, WINDOW_STRIDE,
    BASIN_K, RESULTS_DIR, ARTIFACTS_DIR, N_LDA_CLASSES,
    compute_rul_thresholds,
)
from pipeline.ncmapss.data_loader import (
    load_dataset, normalize_by_regime, get_engine_data,
)
from pipeline.ncmapss.run_ncmapss import (
    resolve_thresholds, rul_bucket_labels,
)
from pipeline.feature_extraction import (
    build_feature_matrix, get_feature_class_indices, get_feature_sensor_mapping,
)
from pipeline.failure_basin import (
    build_failure_basin, build_index, distance_to_basin,
)
from pipeline.evaluation import verify_v2


def wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    More reliable than normal approximation for small n.
    z=1.96 gives 95% CI.
    """
    if n_total == 0:
        return (0.0, 0.0)
    p_hat = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(
        (p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total
    ) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def per_engine_v2(
    lda_projections: np.ndarray,
    meta: dict,
    basin_index,
    basin_k: int,
) -> list[dict]:
    """Compute V2 (Spearman rho) per engine."""
    engine_ids = sorted(set(meta["unit_id"]))
    results = []
    for uid in engine_ids:
        mask = meta["unit_id"] == uid
        engine_lda = lda_projections[mask]
        engine_rul = meta["rul"][mask]
        if len(engine_lda) < 5:
            continue
        dists = distance_to_basin(engine_lda, basin_index, basin_k)
        rho, p = stats.spearmanr(dists, engine_rul)
        results.append({
            "unit_id": int(uid),
            "v2_rho": float(rho),
            "v2_p": float(p),
            "n_windows": len(engine_lda),
            "rul_range": f"[{int(engine_rul.min())}, {int(engine_rul.max())}]",
        })
    return results


def lda_sensor_loadings(
    lda: LinearDiscriminantAnalysis,
    good_features: np.ndarray,
    n_sensors: int,
    sensor_names: list[str],
) -> list[dict]:
    """Map LDA component loadings back to sensors.

    The LDA scalings_ matrix gives the weight of each feature in the
    1D projection. We aggregate absolute loadings per sensor to identify
    which sensors drive the projection (V3 analysis).
    """
    # LDA scalings: (n_features_after_filter, n_components)
    loadings = lda.scalings_[:, 0]  # 1D projection

    # Build full-length loading vector (restore filtered features as 0)
    full_loadings = np.zeros(len(good_features))
    full_loadings[good_features] = loadings

    # Map features to sensors
    sensor_map = get_feature_sensor_mapping(n_sensors, sensor_names)
    feature_classes = get_feature_class_indices(n_sensors)

    # Aggregate absolute loading per sensor
    sensor_loading = {}
    for i, sensor in enumerate(sensor_map):
        if i >= len(full_loadings):
            break
        key = sensor if sensor is not None else "_covariance"
        if key not in sensor_loading:
            sensor_loading[key] = 0.0
        sensor_loading[key] += abs(full_loadings[i])

    # Sort by loading magnitude
    sorted_sensors = sorted(
        sensor_loading.items(), key=lambda x: x[1], reverse=True,
    )

    # Also aggregate by feature class
    class_loading = {}
    for cls_name, indices in feature_classes.items():
        valid = indices[indices < len(full_loadings)]
        class_loading[cls_name] = float(np.sum(np.abs(full_loadings[valid])))

    results = []
    total = sum(v for _, v in sorted_sensors)
    for sensor, loading in sorted_sensors:
        results.append({
            "sensor": sensor,
            "abs_loading": round(loading, 4),
            "pct_total": round(100 * loading / max(total, 1e-10), 1),
        })

    return results, class_loading


def measure_inference_time(
    lda: LinearDiscriminantAnalysis,
    scaler: StandardScaler,
    good_features: np.ndarray,
    basin_index,
    test_features: np.ndarray,
    n_repeats: int = 1000,
) -> dict:
    """Measure wall-clock inference time for a single engine query.

    Times the full inference path: scale -> feature filter -> LDA transform
    -> k-NN basin distance. This is the operational cost per engine per cycle.
    """
    # Use the last window of test features as a representative query
    single_feature = test_features[-1:, :]

    # Warmup
    for _ in range(10):
        scaled = scaler.transform(single_feature)[:, good_features]
        projected = lda.transform(scaled)
        _ = distance_to_basin(projected, basin_index, BASIN_K)

    # Timed runs
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter_ns()
        scaled = scaler.transform(single_feature)[:, good_features]
        projected = lda.transform(scaled)
        _ = distance_to_basin(projected, basin_index, BASIN_K)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)

    times = np.array(times)
    return {
        "n_repeats": n_repeats,
        "mean_ns": int(np.mean(times)),
        "median_ns": int(np.median(times)),
        "p5_ns": int(np.percentile(times, 5)),
        "p95_ns": int(np.percentile(times, 95)),
        "mean_us": round(np.mean(times) / 1000, 1),
        "median_us": round(np.median(times) / 1000, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="N-CMAPSS post-hoc analysis")
    parser.add_argument(
        "--datasets", nargs="+", default=["DS03"],
    )
    args = parser.parse_args()

    for ds_name in args.datasets:
        print(f"=== Analysis: {ds_name} ===")
        print()

        # Load artifacts
        scaler_path = ARTIFACTS_DIR / f"scaler_{ds_name}.pkl"
        lda_path = ARTIFACTS_DIR / f"lda_{ds_name}.pkl"
        gf_path = ARTIFACTS_DIR / f"good_features_{ds_name}.npy"

        if not scaler_path.exists():
            print(f"   No artifacts for {ds_name}. Run the pipeline first.")
            continue

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(lda_path, "rb") as f:
            lda = pickle.load(f)
        good_features = np.load(gf_path)

        # Reload and featurize (same pipeline as run_ncmapss)
        dev_df, test_df = load_dataset(ds_name, rul_clip=None)
        raw_dev_ruls = dev_df["rul"].values
        thresholds = resolve_thresholds(raw_dev_ruls)
        rul_clip = thresholds["rul_clip"]

        dev_df["rul"] = dev_df["rul"].clip(upper=rul_clip)
        test_df["rul"] = test_df["rul"].clip(upper=rul_clip)

        dev_df, test_df, _ = normalize_by_regime(dev_df, test_df)

        dev_engines = get_engine_data(dev_df)
        test_engines = get_engine_data(test_df)
        cols = [c for c in ACTIVE_SENSORS if c in dev_df.columns]

        dev_sensor = {uid: df[cols].values for uid, df in dev_engines.items()}
        dev_ruls = {uid: df["rul"].values for uid, df in dev_engines.items()}
        test_sensor = {uid: df[cols].values for uid, df in test_engines.items()}
        test_ruls = {uid: df["rul"].values for uid, df in test_engines.items()}

        dev_features, dev_meta = build_feature_matrix(
            dev_sensor, dev_ruls, WINDOW_SIZE, WINDOW_STRIDE,
        )
        test_features, test_meta = build_feature_matrix(
            test_sensor, test_ruls, WINDOW_SIZE, WINDOW_STRIDE,
        )

        # Project through fitted scaler + LDA
        dev_scaled = scaler.transform(dev_features)[:, good_features]
        dev_lda = lda.transform(dev_scaled)
        test_scaled = scaler.transform(test_features)[:, good_features]
        test_lda = lda.transform(test_scaled)

        # Build basin
        basin_rul = thresholds["basin_rul_threshold"]
        warning_rul = thresholds["warning_rul"]
        basin = build_failure_basin(dev_lda, dev_meta["rul"], basin_rul)
        basin_index = build_index(basin)

        # --- 1. Per-engine V2 breakdown ---
        print("1. Per-engine V2 (Spearman rho)")
        print()

        dev_v2 = per_engine_v2(dev_lda, dev_meta, basin_index, BASIN_K)
        test_v2 = per_engine_v2(test_lda, test_meta, basin_index, BASIN_K)

        print("   Dev engines:")
        for r in dev_v2:
            print(f"     Unit {r['unit_id']:3d}: rho={r['v2_rho']:+.3f}  "
                  f"p={r['v2_p']:.2e}  n={r['n_windows']}  "
                  f"RUL={r['rul_range']}")

        dev_rhos = [r["v2_rho"] for r in dev_v2]
        print(f"   Dev mean rho: {np.mean(dev_rhos):.3f} "
              f"[{np.min(dev_rhos):.3f}, {np.max(dev_rhos):.3f}]")
        print()

        print("   Test engines:")
        for r in test_v2:
            print(f"     Unit {r['unit_id']:3d}: rho={r['v2_rho']:+.3f}  "
                  f"p={r['v2_p']:.2e}  n={r['n_windows']}  "
                  f"RUL={r['rul_range']}")

        test_rhos = [r["v2_rho"] for r in test_v2]
        if test_rhos:
            print(f"   Test mean rho: {np.mean(test_rhos):.3f} "
                  f"[{np.min(test_rhos):.3f}, {np.max(test_rhos):.3f}]")
        print()

        # --- 2. Wilson CI on F1 ---
        print("2. Wilson CI on F1 (95%)")
        print()

        # Compute at best epsilon from the results
        test_engine_ids = sorted(set(test_meta["unit_id"]))
        final_dists = []
        true_ruls = []
        for uid in test_engine_ids:
            mask = test_meta["unit_id"] == uid
            engine_lda = test_lda[mask]
            if len(engine_lda) == 0:
                continue
            dists = distance_to_basin(engine_lda, basin_index, BASIN_K)
            final_dists.append(float(dists[-1]))
            true_ruls.append(int(test_meta["rul"][mask][-1]))

        final_dists = np.array(final_dists)
        true_ruls = np.array(true_ruls)
        should_fire = true_ruls <= warning_rul
        n_total = len(true_ruls)
        n_positive = should_fire.sum()
        n_negative = n_total - n_positive

        # Best-case: find an epsilon that gives F1=1.0
        # Use median of final distances as threshold
        if n_positive > 0 and n_negative > 0:
            # Find threshold between positive and negative groups
            pos_dists = final_dists[should_fire]
            neg_dists = final_dists[~should_fire]
            best_eps = (pos_dists.max() + neg_dists.min()) / 2
            fired = final_dists < best_eps
        else:
            # All positive or all negative
            fired = np.ones(n_total, dtype=bool)

        tp = (fired & should_fire).sum()
        fp = (fired & ~should_fire).sum()
        fn = (~fired & should_fire).sum()
        tn = (~fired & ~should_fire).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)

        prec_ci = wilson_ci(tp, tp + fp)
        rec_ci = wilson_ci(tp, tp + fn)

        print(f"   n_total={n_total}, n_positive={n_positive}, n_negative={n_negative}")
        print(f"   TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"   Precision: {precision:.3f} [{prec_ci[0]:.3f}, {prec_ci[1]:.3f}]")
        print(f"   Recall:    {recall:.3f} [{rec_ci[0]:.3f}, {rec_ci[1]:.3f}]")
        print(f"   F1:        {f1:.3f}")
        print()

        # --- 3. Wall-clock inference time ---
        print("3. Wall-clock inference time (single engine query)")
        print()

        timing = measure_inference_time(
            lda, scaler, good_features, basin_index, test_features,
        )
        print(f"   Mean:   {timing['mean_us']:.1f} us")
        print(f"   Median: {timing['median_us']:.1f} us")
        print(f"   P5-P95: [{timing['p5_ns']/1000:.1f}, {timing['p95_ns']/1000:.1f}] us")
        print(f"   ({timing['n_repeats']} repeats)")
        print()

        # --- 4. LDA loadings (V3) ---
        print("4. LDA loadings — V3 sensor attribution")
        print()

        n_sensors = len(cols)
        sensor_loadings, class_loadings = lda_sensor_loadings(
            lda, good_features, n_sensors, cols,
        )

        print("   Top sensors by |loading| contribution:")
        for r in sensor_loadings[:10]:
            marker = ""
            if r["sensor"] in ("T48", "T50", "P50"):
                marker = "  <-- LPT/HPT"
            elif r["sensor"] in ("Nf", "Nc"):
                marker = "  <-- speed"
            print(f"     {r['sensor']:<12s} |loading|={r['abs_loading']:.4f}  "
                  f"({r['pct_total']:5.1f}%){marker}")

        print()
        print("   Feature class contributions:")
        total_class = sum(class_loadings.values())
        for cls, load in sorted(class_loadings.items(), key=lambda x: -x[1]):
            print(f"     {cls:<15s} |loading|={load:.4f}  "
                  f"({100*load/max(total_class, 1e-10):5.1f}%)")

        # --- Save analysis ---
        analysis = {
            "dataset": ds_name,
            "thresholds": thresholds,
            "per_engine_v2_dev": dev_v2,
            "per_engine_v2_test": test_v2,
            "dev_v2_summary": {
                "mean_rho": round(np.mean(dev_rhos), 4),
                "min_rho": round(np.min(dev_rhos), 4),
                "max_rho": round(np.max(dev_rhos), 4),
            },
            "test_v2_summary": {
                "mean_rho": round(np.mean(test_rhos), 4) if test_rhos else None,
                "min_rho": round(np.min(test_rhos), 4) if test_rhos else None,
                "max_rho": round(np.max(test_rhos), 4) if test_rhos else None,
            },
            "wilson_ci": {
                "n_total": n_total,
                "n_positive": int(n_positive),
                "n_negative": int(n_negative),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                "precision": precision,
                "precision_ci_95": prec_ci,
                "recall": recall,
                "recall_ci_95": rec_ci,
                "f1": f1,
            },
            "inference_timing": timing,
            "lda_sensor_loadings": sensor_loadings,
            "lda_class_loadings": class_loadings,
        }

        out_path = RESULTS_DIR / f"analysis_{ds_name}.json"
        with open(out_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\nAnalysis saved to {out_path}")
        print()


if __name__ == "__main__":
    main()
