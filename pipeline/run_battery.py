"""STTS validation pipeline — NASA Battery degradation.

Run: python -m pipeline.run_battery

Validates the STTS framework on electrochemical degradation:
Li-ion battery capacity fade from charge/discharge cycling.
Third physical domain after C-MAPSS (thermomechanical) and
PRONOSTIA (vibrational).

Pipeline order (same as C-MAPSS):
  1. Load battery .mat files, extract per-cycle features
  2. Build sliding-window trajectory features F(T)
  3. StandardScaler on raw features
  4. W = uniform (no domain-specific causal chain)
  5. M = 1-component LDA on RUL-bucketed classes
  6. Build failure basin B_f, run monitoring query
  7. Evaluate: cross-validated, verification conditions, P-R sweep
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import scipy.io
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from pipeline.failure_basin import (
    build_failure_basin, build_index, distance_to_basin, distance_to_corpus,
)
from pipeline.evaluation import (
    verify_v1, verify_v2, calibrate_epsilon, precision_recall_sweep,
)

# ── Configuration ──────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "nasa_battery" / "extracted"
RESULTS_DIR = PROJECT_ROOT / "results" / "battery"

# Batteries with clean monotonic degradation (start cap >= 1.5 Ah, fade >= 10%)
CLEAN_BATTERIES = [
    "B0005", "B0006", "B0007", "B0018",  # 24°C
    "B0042", "B0043", "B0044",            # 22°C
    "B0046", "B0047", "B0048",            # 4°C
]

WINDOW_SIZE = 15          # sliding window in discharge cycles
WINDOW_STRIDE = 1
RUL_CLIP = 100            # cap RUL (in discharge cycles)
BASIN_RUL_THRESHOLD = 15  # last 15 discharge cycles = failure basin
BASIN_K = 5
WARNING_RUL = 30          # batteries with RUL <= 30 cycles should be flagged
N_LDA_CLASSES = 6         # RUL bucket count for LDA


# ── Data loading ───────────────────────────────────────────────────────

def load_battery(name: str) -> dict:
    """Load a single battery .mat file, extract discharge cycle data.

    Returns dict with:
        capacity: (n_discharge,) capacity per discharge cycle
        features: (n_discharge, n_features) per-cycle feature vectors
        ambient_temp: ambient temperature in °C
    """
    mat = scipy.io.loadmat(str(DATA_DIR / f"{name}.mat"))
    b = mat[name][0, 0]
    cycles = b["cycle"][0]

    capacities = []
    cycle_features = []
    ambient_temp = None

    for c in cycles:
        ctype = str(c["type"][0])
        if ctype != "discharge":
            continue

        data = c["data"][0, 0]
        if "Capacity" not in data.dtype.names:
            continue

        cap = data["Capacity"].flatten()
        if len(cap) == 0:
            continue

        if ambient_temp is None:
            ambient_temp = int(c["ambient_temperature"].flatten()[0])

        voltage = data["Voltage_measured"].flatten()
        current = data["Current_measured"].flatten()
        temperature = data["Temperature_measured"].flatten()
        time_s = data["Time"].flatten()

        feats = extract_cycle_features(voltage, current, temperature, time_s)
        capacities.append(float(cap[0]))
        cycle_features.append(feats)

    features = np.array(cycle_features)
    capacity = np.array(capacities)

    # Per-battery normalization: express features as change from early-life baseline.
    # This makes features temperature-invariant — captures degradation trajectory
    # shape rather than absolute voltage/temperature/energy levels.
    baseline_n = min(10, len(features))
    baseline_mean = features[:baseline_n].mean(axis=0)
    baseline_std = features[:baseline_n].std(axis=0)
    baseline_std[baseline_std < 1e-10] = 1.0  # avoid division by zero
    features = (features - baseline_mean) / baseline_std

    # Normalize capacity the same way
    cap_baseline = capacity[:baseline_n].mean()
    capacity_normed = capacity / cap_baseline  # fraction of initial capacity

    return {
        "capacity": capacity,
        "capacity_normed": capacity_normed,
        "features": features,
        "ambient_temp": ambient_temp or 24,
    }


def extract_cycle_features(
    voltage: np.ndarray,
    current: np.ndarray,
    temperature: np.ndarray,
    time_s: np.ndarray,
) -> np.ndarray:
    """Extract per-discharge-cycle features.

    Feature classes:
      - Voltage curve shape: mean, std, min, max, range, end voltage,
        voltage at 25%/50%/75% of discharge time (knee detection)
      - Current: mean, std (should be ~constant for CC discharge)
      - Temperature: max, rise (peak - start), mean
      - Energy: integral of V*I over time
      - Duration: total discharge time in seconds
      - Rate features: voltage slope (linear fit), curvature (quadratic coeff)

    Returns (n_features,) vector.
    """
    n = len(voltage)
    if n < 4:
        return np.zeros(20)

    dt = np.diff(time_s)
    duration = time_s[-1] - time_s[0] if len(time_s) > 1 else 0

    # Voltage curve shape
    v_mean = np.mean(voltage)
    v_std = np.std(voltage)
    v_min = np.min(voltage)
    v_max = np.max(voltage)
    v_range = v_max - v_min
    v_end = voltage[-1]

    # Voltage at quartile timepoints (captures knee shape)
    q_idx = [n // 4, n // 2, 3 * n // 4]
    v_q25 = voltage[q_idx[0]]
    v_q50 = voltage[q_idx[1]]
    v_q75 = voltage[q_idx[2]]

    # Current stats
    i_mean = np.mean(np.abs(current))
    i_std = np.std(current)

    # Temperature
    t_max = np.max(temperature)
    t_rise = temperature.max() - temperature[0]
    t_mean = np.mean(temperature)

    # Energy: integral of |V * I| dt
    if len(dt) > 0:
        power = np.abs(voltage[1:] * current[1:])
        energy = np.sum(power * dt)
    else:
        energy = 0.0

    # Voltage slope (linear fit over normalized time)
    t_norm = np.linspace(0, 1, n)
    coeffs = np.polyfit(t_norm, voltage, 2)
    v_curvature = coeffs[0]  # quadratic term — captures knee
    v_slope = coeffs[1]       # linear term — overall rate

    return np.array([
        v_mean, v_std, v_min, v_max, v_range, v_end,
        v_q25, v_q50, v_q75,
        i_mean, i_std,
        t_max, t_rise, t_mean,
        energy, duration,
        v_curvature, v_slope,
        # Derived: voltage drop rate and energy efficiency proxy
        v_range / max(duration, 1.0),  # V/s discharge rate
        energy / max(duration, 1.0),   # average power
    ])


N_CYCLE_FEATURES = 20  # must match extract_cycle_features output length


# ── Trajectory features (windowed) ────────────────────────────────────

def build_trajectory_features(
    cycle_features: np.ndarray,
    capacities: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Build windowed trajectory feature matrix from per-cycle features.

    For each window of consecutive discharge cycles, extract:
      - Time-domain: mean, std of each cycle feature over the window
      - Rate: mean first-derivative of each cycle feature
      - Capacity trend: slope of capacity within window (direct degradation signal)
      - Capacity features: mean, std, min, last within window

    Returns:
        features: (n_windows, feature_dim)
        window_ruls: (n_windows,) RUL at end of each window
    """
    n_cycles = len(cycle_features)
    if n_cycles < window_size:
        return np.empty((0, 0)), np.empty(0)

    # Compute RUL: distance from end (last cycle = RUL 0)
    rul = np.arange(n_cycles - 1, -1, -1, dtype=float)
    rul = np.clip(rul, 0, RUL_CLIP)

    all_features = []
    all_ruls = []

    for start in range(0, n_cycles - window_size + 1, stride):
        end = start + window_size
        window = cycle_features[start:end]  # (window_size, n_cycle_features)
        win_cap = capacities[start:end]

        # Time-domain: mean and std per cycle feature
        td_mean = np.mean(window, axis=0)
        td_std = np.std(window, axis=0)

        # Rate: mean first derivative per cycle feature
        d1 = np.diff(window, axis=0)
        rate_mean = np.mean(d1, axis=0)

        # Capacity trend within window
        t_win = np.arange(window_size, dtype=float)
        cap_slope = np.polyfit(t_win, win_cap, 1)[0]  # linear slope
        cap_mean = np.mean(win_cap)
        cap_std = np.std(win_cap)
        cap_min = np.min(win_cap)
        cap_last = win_cap[-1]

        feat = np.concatenate([
            td_mean,      # 20 features
            td_std,       # 20 features
            rate_mean,    # 20 features
            [cap_slope, cap_mean, cap_std, cap_min, cap_last],  # 5 features
        ])
        all_features.append(feat)
        all_ruls.append(rul[end - 1])

    return np.array(all_features), np.array(all_ruls)


# ── Cross-validation helpers ──────────────────────────────────────────

def rul_bucket_labels(rul: np.ndarray, n_classes: int = N_LDA_CLASSES) -> np.ndarray:
    """Bucket RUL values into classes for LDA."""
    # Evenly spaced buckets from 0 to RUL_CLIP
    boundaries = np.linspace(0, RUL_CLIP, n_classes + 1)
    labels = np.digitize(rul, boundaries) - 1
    return np.clip(labels, 0, n_classes - 1)


def run_fold(
    train_batteries: list[str],
    test_batteries: list[str],
    all_data: dict[str, dict],
) -> dict:
    """Run one fold of cross-validation.

    Fit LDA on train batteries, evaluate on test batteries.
    """
    # Build training feature matrix
    train_feats_list = []
    train_ruls_list = []
    for bname in train_batteries:
        bd = all_data[bname]
        feats, ruls = build_trajectory_features(bd["features"], bd["capacity_normed"])
        if len(feats) == 0:
            continue
        train_feats_list.append(feats)
        train_ruls_list.append(ruls)

    train_features = np.vstack(train_feats_list)
    train_rul = np.concatenate(train_ruls_list)

    # Remove near-constant features (std < 1e-10 after scaling causes LDA singularity)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)

    # Drop features with zero or near-zero variance after scaling
    feature_std = np.std(train_scaled, axis=0)
    good_features = feature_std > 1e-8
    train_scaled = train_scaled[:, good_features]

    # LDA projection (1 component) — regularized to handle near-singular covariance
    labels = rul_bucket_labels(train_rul)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {"error": "insufficient label diversity"}

    lda = LinearDiscriminantAnalysis(
        n_components=1, solver="eigen", shrinkage="auto",
    )
    train_lda = lda.fit_transform(train_scaled, labels)

    # Build failure basin from training data
    basin = build_failure_basin(train_lda, train_rul, BASIN_RUL_THRESHOLD)
    if len(basin) < BASIN_K:
        return {"error": f"basin too small: {len(basin)}"}

    basin_index = build_index(basin)
    corpus_index = build_index(train_lda)

    # Training verification conditions
    train_distances = distance_to_basin(train_lda, basin_index, BASIN_K)
    precursor_mask = train_rul <= BASIN_RUL_THRESHOLD
    nominal_mask = train_rul > BASIN_RUL_THRESHOLD + 20
    v1_train = verify_v1(train_distances[precursor_mask], train_distances[nominal_mask])
    v2_train = verify_v2(train_distances, train_rul)

    # Calibrate epsilon on training data
    epsilon = calibrate_epsilon(train_distances, train_rul, BASIN_RUL_THRESHOLD)

    # Evaluate on test batteries
    test_results = []
    for bname in test_batteries:
        bd = all_data[bname]
        feats, ruls = build_trajectory_features(bd["features"], bd["capacity_normed"])
        if len(feats) == 0:
            continue

        test_scaled = scaler.transform(feats)[:, good_features]
        test_lda = lda.transform(test_scaled)
        test_distances = distance_to_basin(test_lda, basin_index, BASIN_K)

        # V2 per test battery
        v2_test = verify_v2(test_distances, ruls)

        # Final distance and true RUL (for P-R)
        test_results.append({
            "battery": bname,
            "n_windows": len(feats),
            "final_dist": float(test_distances[-1]),
            "true_rul_at_end": float(ruls[-1]),  # should be 0 (run to failure)
            "v2_rho": v2_test["spearman_rho"],
            "v2_p": v2_test["p_value"],
            "distances": test_distances,
            "ruls": ruls,
            "ambient_temp": bd["ambient_temp"],
        })

    return {
        "train_batteries": train_batteries,
        "test_batteries": test_batteries,
        "v1_train": v1_train,
        "v2_train": v2_train,
        "epsilon": epsilon,
        "basin_size": len(basin),
        "train_size": len(train_features),
        "test_results": test_results,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=== STTS Battery Validation Pipeline ===")
    print(f"    Batteries: {len(CLEAN_BATTERIES)}")
    print(f"    Window: {WINDOW_SIZE} cycles, stride {WINDOW_STRIDE}")
    print(f"    Basin RUL: <= {BASIN_RUL_THRESHOLD}")
    print(f"    Warning RUL: <= {WARNING_RUL}")
    print()

    # --- 1. Load all battery data ---
    print("1. Loading battery data...")
    all_data = {}
    for bname in CLEAN_BATTERIES:
        bd = load_battery(bname)
        all_data[bname] = bd
        print(f"   {bname}: {len(bd['capacity'])} discharge cycles, "
              f"cap {bd['capacity'][0]:.3f} -> {bd['capacity'][-1]:.3f} Ah, "
              f"T={bd['ambient_temp']}°C")

    # --- 2. Group by temperature for cross-validation ---
    # Train on two temperature groups, test on the third
    temp_groups = {
        "24C": ["B0005", "B0006", "B0007", "B0018"],
        "22C": ["B0042", "B0043", "B0044"],
        "4C":  ["B0046", "B0047", "B0048"],
    }

    print("\n2. Cross-validation: train on 2 temp groups, test on 1")
    print("   (Tests generalization across operating temperatures)")
    print()

    all_fold_results = []
    for test_group_name, test_batteries in temp_groups.items():
        train_batteries = [b for b in CLEAN_BATTERIES if b not in test_batteries]
        print(f"--- Fold: test={test_group_name} ({len(test_batteries)} batteries), "
              f"train={len(train_batteries)} batteries ---")

        fold = run_fold(train_batteries, test_batteries, all_data)
        if "error" in fold:
            print(f"   ERROR: {fold['error']}")
            continue

        all_fold_results.append(fold)

        print(f"   Train: {fold['train_size']} windows, basin: {fold['basin_size']}")
        v1 = fold["v1_train"]
        print(f"   V1 (train): {'PASS' if v1['passed'] else 'FAIL'} — "
              f"sep={v1['separation_ratio']:.1f}x, "
              f"p={v1['mannwhitney_p']:.2e}")
        v2 = fold["v2_train"]
        print(f"   V2 (train): {'PASS' if v2['passed'] else 'FAIL'} — "
              f"ρ={v2['spearman_rho']:.3f}, p={v2['p_value']:.2e}")
        print(f"   ε (calibrated): {fold['epsilon']:.4f}")

        for tr in fold["test_results"]:
            print(f"   {tr['battery']} (T={tr['ambient_temp']}°C): "
                  f"V2 ρ={tr['v2_rho']:.3f}, "
                  f"final_dist={tr['final_dist']:.4f}")

        print()

    # --- 3. Leave-one-out cross-validation for detection F1 ---
    print("3. Leave-one-battery-out cross-validation for detection performance")
    print()

    loo_results = []
    for test_bname in CLEAN_BATTERIES:
        train_bats = [b for b in CLEAN_BATTERIES if b != test_bname]
        fold = run_fold(train_bats, [test_bname], all_data)
        if "error" in fold:
            continue

        tr = fold["test_results"][0]
        # For run-to-failure batteries, all windows with RUL <= WARNING_RUL should fire
        # We evaluate: does the monitoring query fire before WARNING_RUL?
        dists = tr["distances"]
        ruls = tr["ruls"]
        eps = fold["epsilon"]

        # Check if STTS fires (distance < epsilon) before the battery reaches WARNING_RUL
        fired_mask = dists < eps
        in_warning = ruls <= WARNING_RUL

        # True positives: windows where STTS fires AND RUL <= WARNING_RUL
        tp = (fired_mask & in_warning).sum()
        fp = (fired_mask & ~in_warning).sum()
        fn = (~fired_mask & in_warning).sum()
        tn = (~fired_mask & ~in_warning).sum()

        # Detection: did STTS fire at all before failure?
        detected = np.any(fired_mask & in_warning)

        # First detection cycle (earliest window where fired and RUL <= WARNING_RUL)
        fired_warning_indices = np.where(fired_mask & in_warning)[0]
        first_detection_rul = float(ruls[fired_warning_indices[0]]) if len(fired_warning_indices) > 0 else None

        loo_results.append({
            "battery": test_bname,
            "temp": all_data[test_bname]["ambient_temp"],
            "detected": detected,
            "first_detection_rul": first_detection_rul,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "v2_rho": tr["v2_rho"],
            "n_windows": len(dists),
        })

        status = "DETECTED" if detected else "MISSED"
        det_rul_str = f"at RUL={first_detection_rul:.0f}" if first_detection_rul is not None else ""
        print(f"   {test_bname} (T={all_data[test_bname]['ambient_temp']}°C): "
              f"{status} {det_rul_str}  "
              f"V2 ρ={tr['v2_rho']:.3f}  TP={tp} FP={fp} FN={fn}")

    # Aggregate LOO results
    total_tp = sum(r["tp"] for r in loo_results)
    total_fp = sum(r["fp"] for r in loo_results)
    total_fn = sum(r["fn"] for r in loo_results)
    n_detected = sum(r["detected"] for r in loo_results)
    n_total = len(loo_results)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    print(f"\n   LOO Aggregate:")
    print(f"   Detected: {n_detected}/{n_total} batteries")
    print(f"   Window-level: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    print(f"   TP={total_tp}, FP={total_fp}, FN={total_fn}")

    mean_v2 = np.mean([r["v2_rho"] for r in loo_results])
    print(f"   Mean V2 ρ (test): {mean_v2:.3f}")

    det_ruls = [r["first_detection_rul"] for r in loo_results if r["first_detection_rul"] is not None]
    if det_ruls:
        print(f"   Mean first-detection RUL: {np.mean(det_ruls):.1f} cycles")

    # --- 4. Full-corpus results (all batteries) ---
    print("\n4. Full-corpus verification (all 10 batteries, no holdout)")

    all_feats = []
    all_ruls = []
    battery_ids = []
    for bname in CLEAN_BATTERIES:
        bd = all_data[bname]
        feats, ruls = build_trajectory_features(bd["features"], bd["capacity_normed"])
        if len(feats) == 0:
            continue
        all_feats.append(feats)
        all_ruls.append(ruls)
        battery_ids.extend([bname] * len(feats))

    full_features = np.vstack(all_feats)
    full_rul = np.concatenate(all_ruls)

    scaler = StandardScaler()
    full_scaled = scaler.fit_transform(full_features)
    feature_std = np.std(full_scaled, axis=0)
    good_mask = feature_std > 1e-8
    full_scaled = full_scaled[:, good_mask]

    labels = rul_bucket_labels(full_rul)
    lda = LinearDiscriminantAnalysis(
        n_components=1, solver="eigen", shrinkage="auto",
    )
    full_lda = lda.fit_transform(full_scaled, labels)

    basin = build_failure_basin(full_lda, full_rul, BASIN_RUL_THRESHOLD)
    basin_index = build_index(basin)
    full_distances = distance_to_basin(full_lda, basin_index, BASIN_K)

    precursor_mask = full_rul <= BASIN_RUL_THRESHOLD
    nominal_mask = full_rul > BASIN_RUL_THRESHOLD + 20
    v1_full = verify_v1(full_distances[precursor_mask], full_distances[nominal_mask])
    v2_full = verify_v2(full_distances, full_rul)

    print(f"   Total windows: {len(full_features)}, dim: {full_features.shape[1]}")
    print(f"   Basin size: {len(basin)}")
    print(f"   V1: {'PASS' if v1_full['passed'] else 'FAIL'} — "
          f"sep={v1_full['separation_ratio']:.1f}x, "
          f"p={v1_full['mannwhitney_p']:.2e}")
    print(f"   V2: {'PASS' if v2_full['passed'] else 'FAIL'} — "
          f"ρ={v2_full['spearman_rho']:.3f}, p={v2_full['p_value']:.2e}")

    # --- 5. Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    pd.DataFrame(loo_results).to_csv(RESULTS_DIR / "loo_results.csv", index=False)

    summary = {
        "n_batteries": n_total,
        "n_detected": n_detected,
        "window_precision": precision,
        "window_recall": recall,
        "window_f1": f1,
        "mean_v2_test": mean_v2,
        "v1_full_sep": v1_full["separation_ratio"],
        "v1_full_p": v1_full["mannwhitney_p"],
        "v2_full_rho": v2_full["spearman_rho"],
        "v2_full_p": v2_full["p_value"],
    }
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "summary.csv", index=False)

    print(f"\n   Results saved to {RESULTS_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
