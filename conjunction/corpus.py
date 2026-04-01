"""
Stage F: Corpus loading and feature extraction for conjunction assessment.

Loads reconstructed CDM time series (from reconstruct_split.py) and extracts
46 features per event across six classes:

  F_risk   (8)  — risk trajectory dynamics
  F_geom  (10)  — miss distance and geometry evolution
  F_cov   (12)  — covariance structure evolution
  F_od     (8)  — orbit determination quality indicators
  F_timing (4)  — inter-CDM temporal structure
  F_cross  (4)  — cross-parameter coupling over the sequence

All rate features are computed as Δvalue/Δt using actual elapsed time
between CDM updates (days), not assuming uniform spacing.

Single-CDM events: rate features = 0, ratio features = 1.0,
  correlation features = 0.0, flagged with single_cdm column.

Data quality handling:
  - Flag 3: c_rcs_estimate dropped entirely (32.5% missing)
  - Flag 4: covariance determinants use log10(abs(value)) (negative dets exist)
  - Flag 6: 11 missing c_sigma_r values forward-filled within event

Label: binary 1 if final_risk >= -5 (ESA red threshold), else 0.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "conjunction")
TRAIN_IN = os.path.join(DATA_DIR, "train_events.csv")
TEST_IN = os.path.join(DATA_DIR, "test_events.csv")
TRAIN_OUT = os.path.join(DATA_DIR, "train_features.csv")
TEST_OUT = os.path.join(DATA_DIR, "test_features.csv")

# ---------------------------------------------------------------------------
# Feature definitions — 46 total
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    # F_risk (8)
    "risk_final",
    "risk_mean",
    "d_risk_dt_mean",
    "d_risk_dt_last",
    "d2_risk_dt2_mean",
    "risk_late_early_ratio",
    "risk_range",
    "risk_monotonicity",
    # F_geom (10)
    "miss_dist_final",
    "d_miss_dist_dt_mean",
    "d_miss_dist_dt_last",
    "mahal_dist_final",
    "d_mahal_dt_mean",
    "rel_pos_r_final",
    "rel_pos_t_final",
    "rel_pos_n_final",
    "miss_dist_late_early_ratio",
    "mahal_late_early_ratio",
    # F_cov (12)
    "t_sigma_r_final",
    "c_sigma_r_final",
    "d_t_sigma_r_dt",
    "d_c_sigma_r_dt",
    "t_cov_det_final",
    "c_cov_det_final",
    "d_t_cov_det_dt",
    "d_c_cov_det_dt",
    "cov_det_ratio_tc",
    "sigma_r_ratio_tc",
    "t_cov_late_early_ratio",
    "c_cov_late_early_ratio",
    # F_od (8)
    "t_obs_used_final",
    "c_obs_used_final",
    "d_c_obs_used_dt",
    "c_weighted_rms_final",
    "c_actual_od_span_final",
    "t_time_lastob_end_final",
    "c_time_lastob_end_final",
    "od_quality_ratio",
    # F_timing (4)
    "inter_cdm_dt_mean",
    "inter_cdm_dt_std",
    "inter_cdm_dt_last",
    "n_cdms",
    # F_cross (4)
    "corr_risk_miss",
    "corr_risk_mahal",
    "corr_risk_c_sigma_r",
    "corr_miss_c_cov_det",
]

N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 46, f"Expected 46 features, got {N_FEATURES}"

# Failure basin threshold: log10(Pc) >= -5 (ESA red screen threshold)
HIGH_RISK_THRESHOLD = -5.0

# TCA window: only use CDMs within this many days of TCA for feature extraction.
# This ensures structural consistency between training events (median 6 CDMs from
# train_data.csv only) and test events (median 21 CDMs from interleaved
# reconstruction). Without this, test events have 3.5x more CDMs spanning the
# full 7-day screening window including early floor-risk CDMs, producing
# features the LDA has never seen.
TCA_WINDOW_DAYS = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_float(val, default=float("nan")):
    """Parse a string to float, returning default for empty/unparseable."""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_log10_abs(val):
    """log10(abs(val)), handling zero and negative (Flag 4).

    Zero determinants (from degenerate covariance matrices) get a floor
    value of -30, consistent with the risk floor in the dataset.
    """
    if val != val:  # NaN
        return float("nan")
    if val == 0.0:
        return -30.0  # floor for degenerate covariance
    return math.log10(abs(val))


def spearman_rank_corr(x, y):
    """Spearman rank correlation for two lists. Returns 0.0 on degenerate input."""
    n = len(x)
    if n < 3:
        return 0.0

    def _rank(vals):
        indexed = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))

    if den_x < 1e-12 or den_y < 1e-12:
        return 0.0
    return num / (den_x * den_y)


def late_early_ratio(values):
    """Mean of second half / mean of first half. Safe for zero denominator.
    Filters NaN values from each half before computing means."""
    n = len(values)
    if n < 2:
        return 1.0
    mid = n // 2
    early_valid = [v for v in values[:mid] if _is_valid(v)]
    late_valid = [v for v in values[mid:] if _is_valid(v)]
    if not early_valid or not late_valid:
        return 1.0
    early_mean = sum(early_valid) / len(early_valid)
    late_mean = sum(late_valid) / len(late_valid)
    denom = abs(early_mean) + 1e-10
    return late_mean / denom


def _is_valid(v):
    """Check if a float value is valid (not NaN, not inf)."""
    return v == v and not math.isinf(v)


def mean_rate(values, times):
    """Mean of Δvalue/Δt across consecutive pairs. Skips NaN/inf values."""
    if len(values) < 2:
        return 0.0
    rates = []
    for i in range(1, len(values)):
        if not _is_valid(values[i]) or not _is_valid(values[i - 1]):
            continue
        dt = times[i] - times[i - 1]
        if abs(dt) < 1e-8:
            continue
        rates.append((values[i] - values[i - 1]) / dt)
    return sum(rates) / len(rates) if rates else 0.0


def last_rate(values, times):
    """Δvalue/Δt for the last consecutive pair with valid values."""
    if len(values) < 2:
        return 0.0
    # Walk backward to find last pair of valid values
    for i in range(len(values) - 1, 0, -1):
        if _is_valid(values[i]) and _is_valid(values[i - 1]):
            dt = times[i] - times[i - 1]
            if abs(dt) < 1e-8:
                continue
            return (values[i] - values[i - 1]) / dt
    return 0.0


def mean_accel(values, times):
    """Mean of Δ²value/Δt² (second derivative). Skips NaN/inf values."""
    if len(values) < 3:
        return 0.0
    rates = []
    for i in range(1, len(values)):
        if not _is_valid(values[i]) or not _is_valid(values[i - 1]):
            continue
        dt = times[i] - times[i - 1]
        if abs(dt) < 1e-8:
            continue
        rates.append(((values[i] - values[i - 1]) / dt, times[i]))
    if len(rates) < 2:
        return 0.0
    accels = []
    for i in range(1, len(rates)):
        dt2 = rates[i][1] - rates[i - 1][1]
        if abs(dt2) < 1e-8:
            continue
        accels.append((rates[i][0] - rates[i - 1][0]) / dt2)
    return sum(accels) / len(accels) if accels else 0.0


# ---------------------------------------------------------------------------
# Core: extract 46 features from one event's CDM sequence
# ---------------------------------------------------------------------------

def extract_event_features(cdms):
    """Extract 46 features from a chronologically-ordered CDM sequence.

    Args:
        cdms: list of dicts, sorted by time_to_tca descending (first CDM
              is earliest, highest time_to_tca). Each dict has all 103
              CDM columns as strings.

    Returns:
        dict mapping feature name -> float value
        single_cdm: bool flag
    """
    n = len(cdms)
    single_cdm = n < 2

    # Parse time axis: time_to_tca in days. CDMs are sorted descending
    # (chronological), so time_to_tca decreases. We use the sequence index
    # as the time axis for rate computation, with Δt = |ttca[i] - ttca[i-1]|.
    ttca = [safe_float(c["time_to_tca"]) for c in cdms]

    # Elapsed time from first CDM (increasing). Used for rate denominators.
    t0 = ttca[0]
    elapsed = [t0 - t for t in ttca]  # increasing from 0

    # --- Parse key columns ---
    risk = [safe_float(c["risk"]) for c in cdms]
    miss_dist = [safe_float(c["miss_distance"]) for c in cdms]
    mahal = [safe_float(c["mahalanobis_distance"]) for c in cdms]
    rel_pos_r = [safe_float(c["relative_position_r"]) for c in cdms]
    rel_pos_t = [safe_float(c["relative_position_t"]) for c in cdms]
    rel_pos_n = [safe_float(c["relative_position_n"]) for c in cdms]

    t_sigma_r = [safe_float(c["t_sigma_r"]) for c in cdms]
    c_sigma_r = [safe_float(c["c_sigma_r"]) for c in cdms]

    # Flag 4: covariance determinants — use log10(abs()) due to negative values
    t_cov_det_raw = [safe_float(c["t_position_covariance_det"]) for c in cdms]
    c_cov_det_raw = [safe_float(c["c_position_covariance_det"]) for c in cdms]
    t_cov_det = [safe_log10_abs(v) for v in t_cov_det_raw]
    c_cov_det = [safe_log10_abs(v) for v in c_cov_det_raw]

    t_obs_used = [safe_float(c["t_obs_used"]) for c in cdms]
    c_obs_used = [safe_float(c["c_obs_used"]) for c in cdms]
    c_wrms = [safe_float(c["c_weighted_rms"]) for c in cdms]
    c_od_span = [safe_float(c["c_actual_od_span"]) for c in cdms]
    t_lastob_end = [safe_float(c["t_time_lastob_end"]) for c in cdms]
    c_lastob_end = [safe_float(c["c_time_lastob_end"]) for c in cdms]
    t_wrms = [safe_float(c["t_weighted_rms"]) for c in cdms]

    # Flag 6: fill missing c_sigma_r within event.
    # Pattern: first 1-2 CDMs have NaN, then valid values follow.
    # Back-fill leading NaN from first valid value, then forward-fill any gaps.
    first_valid = None
    for i in range(n):
        if c_sigma_r[i] == c_sigma_r[i]:  # not NaN
            first_valid = c_sigma_r[i]
            break
    if first_valid is not None:
        for i in range(n):
            if c_sigma_r[i] != c_sigma_r[i]:
                c_sigma_r[i] = first_valid
            else:
                first_valid = c_sigma_r[i]  # update for forward-fill

    # Last CDM values (index -1 = closest to TCA)
    feat = {}

    # === F_risk (8) ===
    feat["risk_final"] = risk[-1]
    feat["risk_mean"] = sum(risk) / n
    feat["d_risk_dt_mean"] = mean_rate(risk, elapsed)
    feat["d_risk_dt_last"] = last_rate(risk, elapsed)
    feat["d2_risk_dt2_mean"] = mean_accel(risk, elapsed)
    feat["risk_late_early_ratio"] = late_early_ratio(risk)
    feat["risk_range"] = max(risk) - min(risk)
    if n >= 2:
        increasing = sum(1 for i in range(1, n) if risk[i] > risk[i - 1])
        feat["risk_monotonicity"] = increasing / (n - 1)
    else:
        feat["risk_monotonicity"] = 0.0

    # === F_geom (10) ===
    feat["miss_dist_final"] = miss_dist[-1]
    feat["d_miss_dist_dt_mean"] = mean_rate(miss_dist, elapsed)
    feat["d_miss_dist_dt_last"] = last_rate(miss_dist, elapsed)
    feat["mahal_dist_final"] = mahal[-1]
    feat["d_mahal_dt_mean"] = mean_rate(mahal, elapsed)
    feat["rel_pos_r_final"] = rel_pos_r[-1]
    feat["rel_pos_t_final"] = rel_pos_t[-1]
    feat["rel_pos_n_final"] = rel_pos_n[-1]
    feat["miss_dist_late_early_ratio"] = late_early_ratio(miss_dist)
    feat["mahal_late_early_ratio"] = late_early_ratio(mahal)

    # === F_cov (12) ===
    feat["t_sigma_r_final"] = t_sigma_r[-1]
    feat["c_sigma_r_final"] = c_sigma_r[-1]
    feat["d_t_sigma_r_dt"] = mean_rate(t_sigma_r, elapsed)
    feat["d_c_sigma_r_dt"] = mean_rate(c_sigma_r, elapsed)
    feat["t_cov_det_final"] = t_cov_det[-1]
    feat["c_cov_det_final"] = c_cov_det[-1]
    feat["d_t_cov_det_dt"] = mean_rate(t_cov_det, elapsed)
    feat["d_c_cov_det_dt"] = mean_rate(c_cov_det, elapsed)

    t_det_last = t_cov_det[-1] if t_cov_det[-1] == t_cov_det[-1] else 0.0
    c_det_last = c_cov_det[-1] if c_cov_det[-1] == c_cov_det[-1] else 0.0
    feat["cov_det_ratio_tc"] = t_det_last - c_det_last  # log space: log(a/b) = log(a) - log(b)

    c_sig_last = c_sigma_r[-1] if c_sigma_r[-1] == c_sigma_r[-1] else 1.0
    feat["sigma_r_ratio_tc"] = t_sigma_r[-1] / (c_sig_last + 1e-10)

    feat["t_cov_late_early_ratio"] = late_early_ratio(t_cov_det)
    feat["c_cov_late_early_ratio"] = late_early_ratio(c_cov_det)

    # === F_od (8) ===
    feat["t_obs_used_final"] = t_obs_used[-1]
    feat["c_obs_used_final"] = c_obs_used[-1]
    feat["d_c_obs_used_dt"] = mean_rate(c_obs_used, elapsed)
    feat["c_weighted_rms_final"] = c_wrms[-1]
    feat["c_actual_od_span_final"] = c_od_span[-1]
    feat["t_time_lastob_end_final"] = t_lastob_end[-1]
    feat["c_time_lastob_end_final"] = c_lastob_end[-1]

    t_wrms_last = t_wrms[-1] if t_wrms[-1] == t_wrms[-1] and t_wrms[-1] != 0 else 1e-10
    feat["od_quality_ratio"] = t_wrms_last / (c_wrms[-1] + 1e-10)

    # === F_timing (4) ===
    if n >= 2:
        intervals = [elapsed[i] - elapsed[i - 1] for i in range(1, n)]
        feat["inter_cdm_dt_mean"] = sum(intervals) / len(intervals)
        if len(intervals) >= 2:
            mean_iv = feat["inter_cdm_dt_mean"]
            feat["inter_cdm_dt_std"] = math.sqrt(
                sum((iv - mean_iv) ** 2 for iv in intervals) / len(intervals)
            )
        else:
            feat["inter_cdm_dt_std"] = 0.0
        feat["inter_cdm_dt_last"] = intervals[-1]
    else:
        feat["inter_cdm_dt_mean"] = 0.0
        feat["inter_cdm_dt_std"] = 0.0
        feat["inter_cdm_dt_last"] = 0.0
    feat["n_cdms"] = float(n)

    # === F_cross (4) ===
    # Spearman correlations over the CDM sequence
    # Filter NaN pairs before computing
    def _clean_pairs(a, b):
        return zip(*[(x, y) for x, y in zip(a, b) if x == x and y == y])

    try:
        rx, ry = _clean_pairs(risk, miss_dist)
        feat["corr_risk_miss"] = spearman_rank_corr(list(rx), list(ry))
    except ValueError:
        feat["corr_risk_miss"] = 0.0

    try:
        rx, ry = _clean_pairs(risk, mahal)
        feat["corr_risk_mahal"] = spearman_rank_corr(list(rx), list(ry))
    except ValueError:
        feat["corr_risk_mahal"] = 0.0

    try:
        rx, ry = _clean_pairs(risk, c_sigma_r)
        feat["corr_risk_c_sigma_r"] = spearman_rank_corr(list(rx), list(ry))
    except ValueError:
        feat["corr_risk_c_sigma_r"] = 0.0

    try:
        rx, ry = _clean_pairs(miss_dist, c_cov_det)
        feat["corr_miss_c_cov_det"] = spearman_rank_corr(list(rx), list(ry))
    except ValueError:
        feat["corr_miss_c_cov_det"] = 0.0

    return feat, single_cdm


# ---------------------------------------------------------------------------
# Event loading and grouping
# ---------------------------------------------------------------------------

def load_events(csv_path, has_true_risk=False):
    """Load CDM events from CSV, grouped by event_id.

    Returns:
        events: dict {event_id: list of CDM dicts sorted by time_to_tca desc}
        labels: dict {event_id: true_risk} (only for test set)
    """
    events = defaultdict(list)
    labels = {}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = row["event_id"]
            events[eid].append(row)
            if has_true_risk and "true_risk" in row and row["true_risk"]:
                labels[eid] = float(row["true_risk"])

    # Sort each event's CDMs by time_to_tca descending (chronological)
    for eid in events:
        events[eid].sort(key=lambda r: float(r["time_to_tca"]), reverse=True)

    return dict(events), labels


def get_final_risk(cdms):
    """Get the risk value from the CDM closest to TCA (last in sorted order)."""
    return safe_float(cdms[-1]["risk"])


# ---------------------------------------------------------------------------
# Feature extraction for a full dataset
# ---------------------------------------------------------------------------

def build_feature_matrix(events, labels=None):
    """Extract features for all events.

    Only CDMs within TCA_WINDOW_DAYS of TCA are used for feature extraction.
    Events with no CDMs in the window are excluded entirely.

    Args:
        events: {event_id: [cdm_dicts]} from load_events
        labels: {event_id: true_risk} for test set, or None for train set
            (train set derives label from final CDM risk value)

    Returns:
        rows: list of dicts ready for CSV output
        excluded: number of events excluded (no CDMs in TCA window)
    """
    rows = []
    excluded = 0

    for eid in sorted(events.keys(), key=int):
        cdms = events[eid]

        # Filter to CDMs within TCA window
        windowed = [c for c in cdms if safe_float(c["time_to_tca"]) <= TCA_WINDOW_DAYS]
        if not windowed:
            excluded += 1
            continue

        # Re-sort windowed CDMs (should already be sorted, but be safe)
        windowed.sort(key=lambda r: float(r["time_to_tca"]), reverse=True)

        feat, single_cdm = extract_event_features(windowed)

        # Determine final risk and binary label
        if labels and eid in labels:
            final_risk = labels[eid]
        else:
            final_risk = get_final_risk(windowed)

        label = 1 if final_risk >= HIGH_RISK_THRESHOLD else 0

        row = {"event_id": eid}
        for fname in FEATURE_NAMES:
            row[fname] = feat[fname]
        row["label"] = label
        row["final_risk"] = final_risk
        row["single_cdm"] = 1 if single_cdm else 0

        rows.append(row)

    return rows, excluded


def write_features(rows, out_path):
    """Write feature matrix to CSV."""
    columns = ["event_id"] + FEATURE_NAMES + ["label", "final_risk", "single_cdm"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(name, rows):
    """Print feature matrix summary statistics."""
    n = len(rows)
    labels = [r["label"] for r in rows]
    pos = sum(labels)
    neg = n - pos
    single = sum(1 for r in rows if r["single_cdm"] == 1)

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Shape: {n} events x {N_FEATURES} features")
    print(f"  Labels: {pos} high-risk (label=1), {neg} nominal (label=0)")
    print(f"  Label rate: {100 * pos / n:.2f}%")
    print(f"  Single-CDM events: {single}")

    # NaN counts
    nan_counts = {fname: 0 for fname in FEATURE_NAMES}
    for row in rows:
        for fname in FEATURE_NAMES:
            v = row[fname]
            if isinstance(v, float) and v != v:
                nan_counts[fname] += 1

    nan_features = {k: v for k, v in nan_counts.items() if v > 0}
    if nan_features:
        print(f"\n  Features with NaN values:")
        for fname, count in sorted(nan_features.items(), key=lambda x: -x[1]):
            print(f"    {fname}: {count} ({100 * count / n:.1f}%)")
    else:
        print(f"\n  NaN values: none")

    # Feature ranges (min/max for non-NaN values)
    print(f"\n  Feature ranges:")
    for fname in FEATURE_NAMES:
        vals = [r[fname] for r in rows if isinstance(r[fname], (int, float)) and r[fname] == r[fname]]
        if vals:
            vmin = min(vals)
            vmax = max(vals)
            print(f"    {fname:35s}  [{vmin:>14.4f}, {vmax:>14.4f}]")
        else:
            print(f"    {fname:35s}  [no valid values]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"TCA window: {TCA_WINDOW_DAYS} days")
    print(f"  Only CDMs with time_to_tca <= {TCA_WINDOW_DAYS} are used\n")

    print("Loading training events...")
    train_events, _ = load_events(TRAIN_IN)
    print(f"  {len(train_events)} events loaded")

    print("Loading test events...")
    test_events, test_labels = load_events(TEST_IN, has_true_risk=True)
    print(f"  {len(test_events)} events loaded, {len(test_labels)} labels")

    print("\nExtracting training features...")
    train_rows, train_excluded = build_feature_matrix(train_events)
    write_features(train_rows, TRAIN_OUT)
    print(f"  Wrote {TRAIN_OUT}")
    print(f"  Retained: {len(train_rows)}, Excluded (no CDMs in window): {train_excluded}")

    print("\nExtracting test features...")
    test_rows, test_excluded = build_feature_matrix(test_events, labels=test_labels)
    write_features(test_rows, TEST_OUT)
    print(f"  Wrote {TEST_OUT}")
    print(f"  Retained: {len(test_rows)}, Excluded (no CDMs in window): {test_excluded}")

    print_summary("TRAINING SET", train_rows)
    print_summary("TEST SET", test_rows)


if __name__ == "__main__":
    main()
