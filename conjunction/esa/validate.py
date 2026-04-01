"""
Evaluate a frozen conjunction model on the held-out test set.

Loads serialized artifacts via load_model(model_name) — never calls .fit().
Reports V1 (test set separation), V2 (monotonic approach per high-risk
event), F1/precision/recall with Wilson CIs. Writes results JSON with
config snapshot and artifact checksums.

V2 implementation: for each high-risk test event, compute cumulative
features at each CDM step (using CDMs 1..k), project through the frozen
model, and compute Spearman ρ between time_to_tca and distance-to-basin.
A positive ρ means distance increases with time_to_tca (farther from basin
when far from TCA, closer when near TCA) — the correct direction.

Metric clarity (per scientific-rigor skill):
  - V1 separation: "Do high-risk test events cluster closer to the failure
    basin than nominal events?" NOT "Can the model predict risk?"
  - V2 ρ: "Does the embedding trajectory approach the basin monotonically
    as CDM updates arrive closer to TCA?" NOT "Does risk increase
    monotonically."
  - F1/precision/recall: "If we threshold on basin proximity, what fraction
    of high-risk events are detected and what fraction of alerts are true?"

Usage:
    python conjunction/validate.py model_a
    python conjunction/validate.py model_b
    python conjunction/validate.py          # runs both

Requires: model trained (python conjunction/train.py)
          features extracted (python conjunction/corpus.py)
"""

import csv
import json
import math
import os
import sys

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from conjunction.esa.train import load_model, md5_file
from conjunction.esa.corpus import (
    extract_event_features,
    load_events,
    safe_float,
    TCA_WINDOW_DAYS,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "conjunction")
TEST_FEATURES = os.path.join(DATA_DIR, "test_features.csv")
TEST_EVENTS = os.path.join(DATA_DIR, "test_events.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "conjunction", "esa", "results")


# ---------------------------------------------------------------------------
# Wilson score CI
# ---------------------------------------------------------------------------

def wilson_ci(successes, n, z=1.96):
    """Wilson score confidence interval for a proportion (95% default)."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def f1_wilson_ci(tp, fp, fn, z=1.96):
    """Wilson CIs for precision, recall, and derived F1 CI."""
    n_prec = tp + fp
    n_rec = tp + fn

    prec = tp / max(1, n_prec)
    rec = tp / max(1, n_rec)

    prec_lo, prec_hi = wilson_ci(tp, n_prec, z) if n_prec > 0 else (0.0, 0.0)
    rec_lo, rec_hi = wilson_ci(tp, n_rec, z) if n_rec > 0 else (0.0, 0.0)

    f1 = 2 * prec * rec / max(1e-10, prec + rec)

    if prec_lo > 0 and rec_lo > 0:
        f1_lo = 2 * prec_lo * rec_lo / (prec_lo + rec_lo)
    else:
        f1_lo = 0.0
    if prec_hi > 0 and rec_hi > 0:
        f1_hi = 2 * prec_hi * rec_hi / (prec_hi + rec_hi)
    else:
        f1_hi = 0.0

    return {
        "precision": prec,
        "precision_ci_95": [round(prec_lo, 4), round(prec_hi, 4)],
        "recall": rec,
        "recall_ci_95": [round(rec_lo, 4), round(rec_hi, 4)],
        "f1": f1,
        "f1_ci_95": [round(f1_lo, 4), round(f1_hi, 4)],
    }


# ---------------------------------------------------------------------------
# Load test features for a specific feature set
# ---------------------------------------------------------------------------

def load_test_features(feature_names):
    """Load test_features.csv, selecting only the given feature columns."""
    X_rows = []
    labels = []
    event_ids = []
    final_risks = []

    with open(TEST_FEATURES, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            features = [float(row[fname]) for fname in feature_names]
            X_rows.append(features)
            labels.append(int(row["label"]))
            event_ids.append(row["event_id"])
            final_risks.append(float(row["final_risk"]))

    return (
        np.array(X_rows, dtype=np.float64),
        np.array(labels, dtype=np.int32),
        event_ids,
        np.array(final_risks),
    )


# ---------------------------------------------------------------------------
# V2: Cumulative embedding trajectory per event
# ---------------------------------------------------------------------------

def compute_v2_per_event(test_events, high_risk_eids, model):
    """Compute V2 (monotonic approach) for each high-risk test event.

    For each high-risk event, builds cumulative feature vectors at each
    CDM step (using CDMs 1..k for k=2..n), projects through the frozen
    model, and computes Spearman ρ between time_to_tca and distance-to-basin.

    Returns list of per-event V2 results.
    """
    scaler = model["scaler"]
    W = model["W"]
    lda = model["lda"]
    dist_to_basin = model["dist_to_basin"]
    feature_names = model["feature_names"]

    v2_results = []

    for eid in high_risk_eids:
        if eid not in test_events:
            continue
        cdms = test_events[eid]

        # Apply TCA window — same filter as corpus.py
        windowed = [c for c in cdms
                    if safe_float(c["time_to_tca"]) <= TCA_WINDOW_DAYS]
        n = len(windowed)

        if n < 5:
            v2_results.append({
                "event_id": eid,
                "n_cdms": n,
                "rho": float("nan"),
                "p_value": float("nan"),
                "n_steps": 0,
                "skipped": True,
                "skip_reason": f"sequence too short ({n} CDMs, need 5+)",
            })
            continue

        # Sort windowed CDMs by time_to_tca descending (chronological)
        windowed.sort(key=lambda r: float(r["time_to_tca"]), reverse=True)

        ttca_at_step = []
        dist_at_step = []

        for k in range(2, n + 1):
            partial_cdms = windowed[:k]
            feat, _ = extract_event_features(partial_cdms)

            x = np.array([feat[f] for f in feature_names], dtype=np.float64)
            x = x.reshape(1, -1)

            # .transform() only — never .fit()
            x_scaled = np.nan_to_num(
                scaler.transform(x), nan=0.0, posinf=0.0, neginf=0.0
            )
            x_w = x_scaled * W
            x_proj = lda.transform(x_w).ravel()[0]

            d = dist_to_basin(x_proj)
            ttca = safe_float(partial_cdms[-1]["time_to_tca"])

            ttca_at_step.append(ttca)
            dist_at_step.append(d)

        ttca_arr = np.array(ttca_at_step)
        dist_arr = np.array(dist_at_step)

        if len(ttca_arr) >= 3:
            rho, p_val = spearmanr(ttca_arr, dist_arr)
            rho, p_val = float(rho), float(p_val)
        else:
            rho, p_val = float("nan"), float("nan")

        v2_results.append({
            "event_id": eid,
            "n_cdms": n,
            "rho": rho,
            "p_value": p_val,
            "n_steps": len(ttca_at_step),
            "skipped": False,
        })

    return v2_results


# ---------------------------------------------------------------------------
# Validate one model
# ---------------------------------------------------------------------------

def validate(model_name):
    """Evaluate a frozen model on the held-out test set."""

    print(f"\n{'#' * 60}")
    print(f"  VALIDATING: {model_name}")
    print(f"{'#' * 60}")

    # ── Load frozen model (never .fit()) ───────────────────────
    model = load_model(model_name)
    meta = model["meta"]
    feature_names = model["feature_names"]
    scaler = model["scaler"]
    W = model["W"]
    lda = model["lda"]
    basin = model["basin"]
    dist_to_basin_fn = model["dist_to_basin"]

    config = meta["config"]
    n_feat = config["n_features"]

    print(f"  Model: {config.get('model_name', model_name)}")
    print(f"  Features: {n_feat}")
    print(f"  Basin: {len(basin)} embeddings, "
          f"mean={meta['diagnostics']['basin_mean']:.4f}, "
          f"std={meta['diagnostics']['basin_std']:.4f}")
    print(f"  Artifact checksums verified.")

    # ── Load test features ─────────────────────────────────────
    X_test, y_test, event_ids, final_risks = load_test_features(feature_names)
    n_test = len(y_test)
    n_pos = int(y_test.sum())
    n_neg = n_test - n_pos

    print(f"\n  Test set: {n_test} events ({n_pos} high-risk, {n_neg} nominal)")

    # ── Project test data — .transform() only ──────────────────
    X_scaled = np.nan_to_num(
        scaler.transform(X_test), nan=0.0, posinf=0.0, neginf=0.0
    )
    X_w = X_scaled * W
    X_proj = lda.transform(X_w).ravel()

    # ── Distances to basin ─────────────────────────────────────
    test_dists = np.array([dist_to_basin_fn(p) for p in X_proj])

    nom_dists = test_dists[y_test == 0]
    hr_dists = test_dists[y_test == 1]

    # ── V1: Test set separation ────────────────────────────────
    median_nominal = float(np.median(nom_dists))
    median_hr = float(np.median(hr_dists))
    v1_sep = median_nominal / max(median_hr, 1e-10)

    _, v1_p = mannwhitneyu(nom_dists, hr_dists, alternative="greater")
    v1_p = float(v1_p)

    print(f"\n  V1 BASIN SEPARATION (TEST)")
    print(f"    Median dist — nominal:   {median_nominal:.6f}")
    print(f"    Median dist — high-risk: {median_hr:.6f}")
    print(f"    Separation: {v1_sep:.1f}x, p={v1_p:.2e} "
          f"{'PASS' if v1_p < 0.001 else 'FAIL'}")

    # ── V2: Monotonic approach per high-risk event ─────────────
    test_events_raw, _ = load_events(TEST_EVENTS, has_true_risk=True)
    high_risk_eids = [eid for i, eid in enumerate(event_ids) if y_test[i] == 1]

    v2_results = compute_v2_per_event(test_events_raw, high_risk_eids, model)

    valid_rhos = [r["rho"] for r in v2_results
                  if not r["skipped"] and r["rho"] == r["rho"]]
    n_valid = len(valid_rhos)
    n_skipped = sum(1 for r in v2_results if r["skipped"])
    n_positive_rho = sum(1 for rho in valid_rhos if rho > 0)
    n_negative_rho = sum(1 for rho in valid_rhos if rho < 0)

    if valid_rhos:
        mean_rho = float(np.mean(valid_rhos))
        median_rho = float(np.median(valid_rhos))
        frac_positive = n_positive_rho / n_valid
    else:
        mean_rho = float("nan")
        median_rho = float("nan")
        frac_positive = 0.0

    print(f"\n  V2 MONOTONIC APPROACH")
    print(f"    Analyzed: {n_valid} events ({n_skipped} skipped)")
    print(f"    Mean ρ:   {mean_rho:+.3f}")
    print(f"    Median ρ: {median_rho:+.3f}")
    print(f"    ρ > 0 (correct): {n_positive_rho}/{n_valid} ({100*frac_positive:.1f}%)")

    # ── Classification ─────────────────────────────────────────
    # Epsilon = 95th percentile of training basin self-distances.
    # For each basin point, compute its k-NN distance to the rest of the
    # basin. The 95th percentile of these distances is the radius that
    # contains 95% of training high-risk embeddings. This adapts to basin
    # shape regardless of mean/std relationship.
    k = config["k_neighbors"]
    basin_self_dists = []
    for i, p in enumerate(basin):
        dists = np.abs(np.delete(basin, i) - p)  # exclude self
        basin_self_dists.append(np.mean(np.sort(dists)[:min(k, len(dists))]))
    epsilon = float(np.percentile(basin_self_dists, 95))

    preds = (test_dists < epsilon).astype(int)
    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())
    tn = int(((preds == 0) & (y_test == 0)).sum())

    metrics = f1_wilson_ci(tp, fp, fn)

    print(f"\n  CLASSIFICATION (ε = {epsilon:.4f})")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"    Precision: {metrics['precision']:.4f} "
          f"[{metrics['precision_ci_95'][0]:.4f}, {metrics['precision_ci_95'][1]:.4f}]")
    print(f"    Recall:    {metrics['recall']:.4f} "
          f"[{metrics['recall_ci_95'][0]:.4f}, {metrics['recall_ci_95'][1]:.4f}]")
    print(f"    F1:        {metrics['f1']:.4f} "
          f"[{metrics['f1_ci_95'][0]:.4f}, {metrics['f1_ci_95'][1]:.4f}]")

    # ── Summary ────────────────────────────────────────────────
    print(f"\n  {'Metric':<30s} {'Value':>12s} {'Status':>8s}")
    print(f"  {'-'*30} {'-'*12} {'-'*8}")
    print(f"  {'V1 separation':<30s} {v1_sep:>12.1f}x {'PASS' if v1_p < 0.001 else 'FAIL':>8s}")
    print(f"  {'V1 p-value':<30s} {v1_p:>12.2e} {'':>8s}")
    print(f"  {'V2 mean ρ':<30s} {mean_rho:>+12.3f} {'PASS' if mean_rho > 0 else 'FAIL':>8s}")
    print(f"  {'V2 median ρ':<30s} {median_rho:>+12.3f} {'':>8s}")
    print(f"  {'V2 fraction ρ>0':<30s} {frac_positive:>11.1%} {'':>8s}")
    print(f"  {'F1':<30s} {metrics['f1']:>12.4f} {'':>8s}")
    print(f"  {'Precision':<30s} {metrics['precision']:>12.4f} {'':>8s}")
    print(f"  {'Recall':<30s} {metrics['recall']:>12.4f} {'':>8s}")

    # ── Save results ───────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {
        "config": config,
        "artifact_checksums": meta["artifact_checksums"],
        "test_set": {
            "n_events": n_test,
            "n_high_risk": n_pos,
            "n_nominal": n_neg,
        },
        "v1_train": meta["v1"],
        "v1": {
            "description": "Basin separation on held-out test set",
            "question_answered": (
                "Do high-risk test events cluster closer to the training "
                "failure basin than nominal test events?"
            ),
            "separation_ratio": round(v1_sep, 2),
            "p_value": v1_p,
            "median_nominal_distance": round(median_nominal, 6),
            "median_high_risk_distance": round(median_hr, 6),
            "passed": v1_p < 0.001,
        },
        "v2": {
            "description": "Monotonic approach per high-risk event via cumulative embedding",
            "question_answered": (
                "Does each high-risk event's embedding trajectory approach "
                "the failure basin monotonically as CDM updates arrive "
                "closer to TCA?"
            ),
            "mean_rho": round(mean_rho, 4),
            "median_rho": round(median_rho, 4),
            "n_positive_rho": n_positive_rho,
            "n_negative_rho": n_negative_rho,
            "n_valid": n_valid,
            "n_skipped": n_skipped,
            "fraction_positive_rho": round(frac_positive, 4),
            "passed": mean_rho > 0 if valid_rhos else False,
            "per_event": [
                {k: v for k, v in r.items()}
                for r in v2_results
            ],
        },
        "classification": {
            "description": "Binary classification using distance-to-basin < epsilon",
            "epsilon": round(epsilon, 6),
            "epsilon_derivation": "95th percentile of training basin self-distances (k-NN)",
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            **metrics,
        },
        "metric_notes": {
            "v1": "Measures geometric separability, not predictive accuracy",
            "v2": "Measures trajectory directionality in embedding space",
            "f1": (
                f"Computed on {n_test} events with {n_pos} positives. "
                f"Wilson 95% CI accounts for small sample size."
            ),
        },
    }

    outfile = os.path.join(RESULTS_DIR, f"validate_{model_name}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {outfile}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else ["model_a", "model_b"]
    for m in models:
        validate(m)
