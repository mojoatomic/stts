"""
STTS Orbital Pipeline: F -> W -> M -> monitoring query.

F: Feature extraction from sliding window of orbital states
   - Time-domain: mean/std/min/max of a, e, inc, B*
   - Rate features: decay rate, decay acceleration, B* drift
   - Cross-element covariance structure

W: Causal weighting (physics-informed)
   - Up-weight: decay rate, decay acceleration (primary degradation signal)
   - Down-weight: RAAN, argp (precession angles, not degradation)

M: 1-component LDA projection (same as C-MAPSS approach)

Monitoring: nearest-neighbor distance to failure basin B_f
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, mannwhitneyu
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from orbital_stts.generate_synthetic_corpus import OrbitalState, decay_rate, RE


# ─────────────────────────────────────────────────────────────
# STAGE F: Feature extraction
# ─────────────────────────────────────────────────────────────

WINDOW_DAYS = 14  # sliding window size
STRIDE_DAYS = 1   # stride

def extract_window_features(window: List[OrbitalState]) -> np.ndarray:
    """
    Extract F-stage features from a window of orbital states.
    
    Features (40 total):
      - Time-domain summaries of a, e, inc, B*  [4 × 4 = 16]
      - Rate features: da/dt mean/std, d2a/dt2, dB*/dt  [6]
      - Computed decay rates at each point: mean/std/min/max  [4]
      - Altitude: mean/std/min/max  [4]
      - Cross-element: corr(da/dt, B*), corr(alt, B*), corr(da/dt, alt)  [3]
      - B* trend: linear slope  [1]
      - Eccentricity trend  [1]
      - Decay rate ratio (late/early window)  [1]
      - RUL-independent: inclination mean  [1]
      - Mean RAAN precession rate  [1]
      - Decay acceleration (normalized)  [1]
      - Window length indicator  [1]
    Total: ~40 features
    """
    if len(window) < 3:
        return np.zeros(40)

    # Raw arrays
    a_vals   = np.array([s.a for s in window])
    e_vals   = np.array([s.e for s in window])
    inc_vals = np.array([s.inc for s in window])
    bs_vals  = np.array([s.bstar for s in window])
    alt_vals = a_vals - RE

    # Computed decay rates
    dr_vals  = np.array([abs(decay_rate(alt, bs))
                         for alt, bs in zip(alt_vals, bs_vals)])

    # da/dt from successive observations
    da_dt = np.diff(a_vals)  # km/day
    if len(da_dt) == 0:
        da_dt = np.array([0.0])

    # d2a/dt2 (decay acceleration)
    if len(da_dt) > 1:
        d2a_dt2 = np.diff(da_dt)
    else:
        d2a_dt2 = np.array([0.0])

    # dB*/dt
    dbs_dt = np.diff(bs_vals)
    if len(dbs_dt) == 0:
        dbs_dt = np.array([0.0])

    # Time-domain summaries
    feats = []

    for arr in [a_vals, e_vals, inc_vals, bs_vals]:
        feats.extend([arr.mean(), arr.std() + 1e-10,
                      arr.min(), arr.max()])

    # Rate features
    feats.extend([
        da_dt.mean(),
        da_dt.std() + 1e-10,
        d2a_dt2.mean(),
        d2a_dt2.std() + 1e-10,
        dbs_dt.mean(),
        dbs_dt.std() + 1e-10,
    ])

    # Computed decay rates
    feats.extend([dr_vals.mean(), dr_vals.std() + 1e-10,
                  dr_vals.min(), dr_vals.max()])

    # Altitude summaries
    feats.extend([alt_vals.mean(), alt_vals.std() + 1e-10,
                  alt_vals.min(), alt_vals.max()])

    # Cross-element correlations
    if len(da_dt) > 3:
        bs_trim = bs_vals[:len(da_dt)]
        alt_trim = alt_vals[:len(da_dt)]
        with np.errstate(divide='ignore', invalid='ignore'):
            c1 = np.corrcoef(da_dt, bs_trim)[0, 1]
            c2 = np.corrcoef(alt_trim, bs_trim)[0, 1]
            c3 = np.corrcoef(da_dt, alt_trim)[0, 1]
        feats.extend([
            0.0 if np.isnan(c1) else c1,
            0.0 if np.isnan(c2) else c2,
            0.0 if np.isnan(c3) else c3,
        ])
    else:
        feats.extend([0.0, 0.0, 0.0])

    # B* trend (linear slope)
    if len(bs_vals) > 2:
        t = np.arange(len(bs_vals))
        slope = np.polyfit(t, bs_vals, 1)[0]
        feats.append(slope)
    else:
        feats.append(0.0)

    # Eccentricity trend
    if len(e_vals) > 2:
        t = np.arange(len(e_vals))
        feats.append(np.polyfit(t, e_vals, 1)[0])
    else:
        feats.append(0.0)

    # Decay rate ratio: late window / early window (key precursor signal)
    half = len(dr_vals) // 2
    if half > 0:
        early_rate = dr_vals[:half].mean()
        late_rate  = dr_vals[half:].mean()
        ratio = late_rate / (early_rate + 1e-10)
    else:
        ratio = 1.0
    feats.append(ratio)

    # Inclination (stable, context feature)
    feats.append(inc_vals.mean())

    # RAAN precession rate (J2 effect — context, not degradation)
    feats.append(0.0)  # placeholder; computed separately if needed

    # Decay acceleration normalized by current decay rate
    if len(d2a_dt2) > 0 and abs(da_dt.mean()) > 1e-10:
        feats.append(d2a_dt2.mean() / (abs(da_dt.mean()) + 1e-10))
    else:
        feats.append(0.0)

    # Window completeness
    feats.append(float(len(window)) / WINDOW_DAYS)

    # Pad or trim to exactly 40
    feats = feats[:40]
    while len(feats) < 40:
        feats.append(0.0)

    return np.array(feats, dtype=np.float64)


def extract_trajectory_windows(
    trajectory: List[OrbitalState],
    window_days: int = WINDOW_DAYS,
    stride_days: int = STRIDE_DAYS,
    warning_rul: float = 30.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract all feature windows from a trajectory.
    
    Returns:
        features: (n_windows, n_features)
        ruls:     (n_windows,) days until reentry
        labels:   (n_windows,) 1=precursor (RUL<=warning), 0=nominal
    """
    features, ruls, labels = [], [], []

    for start in range(0, len(trajectory) - window_days + 1, stride_days):
        window = trajectory[start:start + window_days]
        feat = extract_window_features(window)
        # Use the RUL at the END of the window (most recent state)
        rul = window[-1].rul_days
        label = 1 if rul <= warning_rul else 0
        features.append(feat)
        ruls.append(rul)
        labels.append(label)

    if not features:
        return np.empty((0, 40)), np.array([]), np.array([])

    return np.array(features), np.array(ruls), np.array(labels)


# ─────────────────────────────────────────────────────────────
# STAGE W: Causal weighting
# ─────────────────────────────────────────────────────────────

def build_weight_matrix(n_features: int = 40) -> np.ndarray:
    """
    Physics-informed causal weights for orbital degradation features.
    
    High weight: decay rate features, decay acceleration (direct degradation)
    Medium weight: altitude, B* trend (degradation-correlated)
    Low weight: inclination, RAAN-derived features (context, not degradation)
    """
    W = np.ones(n_features)

    # Feature layout (from extract_window_features):
    # [0-15]:  a, e, inc, B* summaries (mean/std/min/max × 4)
    # [16-21]: da/dt mean/std, d2a/dt2 mean/std, dB*/dt mean/std
    # [22-25]: decay rate mean/std/min/max
    # [26-29]: altitude mean/std/min/max
    # [30-32]: cross-element correlations
    # [33]:    B* slope
    # [34]:    e trend
    # [35]:    decay rate ratio (late/early)  ← MOST IMPORTANT
    # [36]:    inclination mean
    # [37]:    RAAN precession rate
    # [38]:    normalized decay acceleration
    # [39]:    window completeness

    # Primary degradation signal: decay rate and acceleration
    W[16:22] *= 3.0   # da/dt, d2a/dt2, dB*/dt
    W[22:26] *= 3.0   # computed decay rates
    W[35]    *= 5.0   # decay rate ratio (late/early) — KEY PRECURSOR
    W[38]    *= 3.0   # normalized decay acceleration

    # Secondary: altitude (directly related to decay)
    W[0:4]   *= 2.0   # semi-major axis summaries
    W[26:30] *= 2.0   # altitude summaries

    # B* trend
    W[12:16] *= 2.0   # B* summaries
    W[33]    *= 2.0   # B* slope

    # Cross-correlations
    W[30:33] *= 1.5

    # Low weight: inclination, RAAN (orbital geometry, not degradation)
    W[8:12]  *= 0.3   # inclination summaries
    W[36]    *= 0.2   # inclination mean
    W[37]    *= 0.1   # RAAN precession

    # Eccentricity (weakly correlated with decay)
    W[4:8]   *= 0.5

    return W


# ─────────────────────────────────────────────────────────────
# STAGE M: LDA projection
# ─────────────────────────────────────────────────────────────

def fit_projection(features: np.ndarray,
                   labels: np.ndarray,
                   weights: np.ndarray) -> Tuple[StandardScaler, np.ndarray, LinearDiscriminantAnalysis]:
    """
    Fit the M-stage projection:
    1. Standardize features (per-feature z-score)
    2. Apply causal weights W
    3. Fit 1-component LDA

    Returns scaler, W, lda for later application.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X_weighted = X_scaled * weights
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_weighted, labels)
    return scaler, weights, lda


def project_features(features: np.ndarray,
                     scaler: StandardScaler,
                     weights: np.ndarray,
                     lda: LinearDiscriminantAnalysis) -> np.ndarray:
    """Apply fitted F->W->M pipeline to new features."""
    X_scaled = scaler.transform(features)
    X_weighted = X_scaled * weights
    return lda.transform(X_weighted).ravel()


# ─────────────────────────────────────────────────────────────
# STTS monitoring
# ─────────────────────────────────────────────────────────────

def build_failure_basin(projected: np.ndarray,
                        labels: np.ndarray) -> np.ndarray:
    """Return projected coordinates of failure precursor windows (label=1)."""
    return projected[labels == 1]


def basin_distance(query: float, basin: np.ndarray, k: int = 5) -> float:
    """Mean distance to k nearest neighbors in the failure basin."""
    if len(basin) == 0:
        return 999.0
    dists = np.abs(basin - query)
    k = min(k, len(basin))
    return np.mean(np.sort(dists)[:k])


# ─────────────────────────────────────────────────────────────
# Verification conditions
# ─────────────────────────────────────────────────────────────

def verify_v1(nominal_dists: np.ndarray,
              precursor_dists: np.ndarray) -> Dict:
    """V1: Basin separation. Nominal >> Precursor distances."""
    median_nominal    = np.median(nominal_dists)
    median_precursor  = np.median(precursor_dists)
    separation = (median_nominal / (median_precursor + 1e-10))
    stat, p = mannwhitneyu(nominal_dists, precursor_dists,
                            alternative='greater')
    return {
        'median_nominal':   median_nominal,
        'median_precursor': median_precursor,
        'separation':       separation,
        'p_value':          p,
        'pass':             separation > 2.0 and p < 0.001
    }


def verify_v2(ruls: np.ndarray, dists: np.ndarray) -> Dict:
    """V2: Monotonic approach. Distance decreases as RUL decreases."""
    # Only use windows approaching failure (RUL < 200)
    mask = ruls < 200
    if mask.sum() < 5:
        return {'rho': 0.0, 'p_value': 1.0, 'pass': False}
    rho, p = spearmanr(ruls[mask], dists[mask])
    return {
        'rho':     rho,
        'p_value': p,
        'pass':    rho > 0.5 and p < 0.05
    }


# ─────────────────────────────────────────────────────────────
# Full evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_detection(
    test_trajectories: List[List[OrbitalState]],
    basin: np.ndarray,
    scaler: StandardScaler,
    weights: np.ndarray,
    lda: LinearDiscriminantAnalysis,
    epsilon: float,
    warning_rul: float = 30.0
) -> Dict:
    """
    Evaluate detection performance on test trajectories.
    For each trajectory: walk forward, record when STTS fires.
    """
    true_positives  = 0
    false_positives = 0
    true_negatives  = 0
    false_negatives = 0
    detection_leads = []  # RUL at time of detection for TPs

    for traj in test_trajectories:
        is_reentry = traj[-1].rul_days < 50  # terminal reentry trajectory

        fired = False
        fire_rul = None

        # Walk forward through trajectory
        for start in range(0, len(traj) - WINDOW_DAYS + 1, STRIDE_DAYS):
            window = traj[start:start + WINDOW_DAYS]
            feat = extract_window_features(window)
            proj = project_features(feat.reshape(1, -1), scaler, weights, lda)[0]
            dist = basin_distance(proj, basin)
            rul  = window[-1].rul_days

            if dist < epsilon and not fired:
                fired = True
                fire_rul = rul
                break  # first detection

        if is_reentry:
            if fired and fire_rul is not None and fire_rul > 0:
                true_positives += 1
                detection_leads.append(fire_rul)
            else:
                false_negatives += 1
        else:
            if fired:
                false_positives += 1
            else:
                true_negatives += 1

    total = true_positives + false_negatives
    precision = true_positives / max(1, true_positives + false_positives)
    recall    = true_positives / max(1, total)
    f1 = (2 * precision * recall / max(1e-10, precision + recall))

    return {
        'f1':          f1,
        'precision':   precision,
        'recall':      recall,
        'tp':          true_positives,
        'fp':          false_positives,
        'fn':          false_negatives,
        'tn':          true_negatives,
        'mean_lead':   np.mean(detection_leads) if detection_leads else 0.0,
        'median_lead': np.median(detection_leads) if detection_leads else 0.0,
    }


def sweep_epsilon(basin: np.ndarray,
                  all_dists: np.ndarray,
                  all_labels: np.ndarray) -> float:
    """Find epsilon that maximises F1 on training data."""
    thresholds = np.percentile(all_dists, np.linspace(5, 95, 50))
    best_f1, best_eps = 0.0, thresholds[0]

    for eps in thresholds:
        preds = (all_dists < eps).astype(int)
        tp = ((preds == 1) & (all_labels == 1)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1   = 2 * prec * rec / max(1e-10, prec + rec)
        if f1 > best_f1:
            best_f1, best_eps = f1, eps

    return best_eps


if __name__ == "__main__":
    from orbital_stts.generate_synthetic_corpus import generate_corpus

    print("=== STTS Orbital Pipeline Test ===\n")

    # Generate corpus
    print("Generating corpus...")
    reentry, nominal = generate_corpus(n_reentry=120, n_nominal=120,
                                        seed=42, verbose=True)

    # Train/test split: 80/20
    n_re_train = int(0.8 * len(reentry))
    n_no_train = int(0.8 * len(nominal))
    re_train, re_test = reentry[:n_re_train], reentry[n_re_train:]
    no_train, no_test = nominal[:n_no_train], nominal[n_no_train:]

    print(f"\nSplit: {n_re_train} reentry train, {len(re_test)} test")

    # Extract training windows
    print("\nExtracting features...")
    train_feats, train_ruls, train_labels = [], [], []

    for traj in re_train + no_train:
        f, r, l = extract_trajectory_windows(traj)
        if len(f):
            train_feats.append(f)
            train_ruls.append(r)
            train_labels.append(l)

    X_train = np.vstack(train_feats)
    y_train = np.concatenate(train_labels)
    r_train = np.concatenate(train_ruls)

    print(f"Training windows: {len(X_train)} "
          f"({y_train.sum()} precursor, {(y_train==0).sum()} nominal)")

    # Fit pipeline
    print("\nFitting F->W->M pipeline...")
    W = build_weight_matrix(X_train.shape[1])
    scaler, W, lda = fit_projection(X_train, y_train, W)

    # Project training data
    X_proj = project_features(X_train, scaler, W, lda)

    # Build failure basin
    basin = build_failure_basin(X_proj, y_train)
    print(f"Failure basin: {len(basin)} precursor embeddings")

    # Compute distances for all training windows
    print("Computing basin distances...")
    all_dists = np.array([basin_distance(p, basin) for p in X_proj])

    # Verify V1 and V2
    print("\n=== Verification Conditions ===")
    v1 = verify_v1(all_dists[y_train == 0], all_dists[y_train == 1])
    v2 = verify_v2(r_train, all_dists)

    print(f"V1 (Basin Separation):")
    print(f"  Nominal median distance:    {v1['median_nominal']:.4f}")
    print(f"  Precursor median distance:  {v1['median_precursor']:.4f}")
    print(f"  Separation ratio:           {v1['separation']:.1f}x")
    print(f"  p-value:                    {v1['p_value']:.2e}")
    print(f"  PASS: {v1['pass']}")

    print(f"\nV2 (Monotonic Approach):")
    print(f"  Spearman ρ(RUL, distance):  {v2['rho']:.4f}")
    print(f"  p-value:                    {v2['p_value']:.2e}")
    print(f"  PASS: {v2['pass']}")

    # Calibrate epsilon
    eps = sweep_epsilon(basin, all_dists, y_train)
    print(f"\nCalibrated epsilon: {eps:.4f}")

    # Evaluate on test set
    print("\n=== Test Set Detection Performance ===")
    test_trajs = re_test + no_test
    results = evaluate_detection(test_trajs, basin, scaler, W, lda, eps)

    print(f"F1:        {results['f1']:.3f}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall:    {results['recall']:.3f}")
    print(f"TP/FP/FN:  {results['tp']}/{results['fp']}/{results['fn']}")
    print(f"Mean detection lead time:   {results['mean_lead']:.1f} days before reentry")
    print(f"Median detection lead time: {results['median_lead']:.1f} days before reentry")

    # Compare against TIP message baseline
    print(f"\n=== Comparison vs TIP Messages ===")
    print(f"TIP first message: T-4 days")
    print(f"STTS mean lead:    T-{results['mean_lead']:.0f} days")
    if results['mean_lead'] > 4:
        advantage = results['mean_lead'] - 4
        print(f"STTS advantage:    +{advantage:.0f} days earlier warning")
    else:
        print(f"STTS advantage:    None over TIP on synthetic data")
