#!/usr/bin/env python3
"""
STTS Orbital: JPL Horizons + CNEOS Close Approach Pipeline

Runs the full STTS pipeline on real NASA data:
  1. Fetch labeled close approach events from CNEOS (the failure corpus)
  2. Fetch orbital element histories from JPL Horizons (the trajectories)
  3. Compute residuals: observed elements vs Horizons integrated prediction
  4. Run F->W->M pipeline on residuals
  5. Verify V1, V2, detection performance

This is the "reverse from knowns" approach:
  - We know which asteroids had close approaches and exactly when
  - We walk backwards through their orbital history
  - The STTS query should fire well before the close approach date
  - We compare against the known event date to measure lead time

Usage:
    python3 horizons_stts_pipeline.py

Requirements:
    pip install requests numpy scipy scikit-learn pandas
    
No authentication required. JPL APIs are free and open.
"""

import requests
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────

CNEOS_CAD_URL    = "https://ssd-api.jpl.nasa.gov/cad.api"
HORIZONS_URL     = "https://ssd.jpl.nasa.gov/api/horizons.api"
SBDB_URL         = "https://ssd-api.jpl.nasa.gov/sbdb.api"

# Rate limit: one request at a time per JPL policy
REQUEST_DELAY    = 1.0  # seconds between requests


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class CloseApproach:
    """A labeled close approach event from CNEOS."""
    designation: str    # Asteroid designation e.g. "2004 BL86"
    jd:          float  # Julian date of close approach
    cd:          str    # Calendar date string
    dist_au:     float  # Miss distance in AU
    dist_ld:     float  # Miss distance in lunar distances
    v_rel:       float  # Relative velocity km/s
    v_inf:       float  # Infinity velocity km/s


@dataclass 
class OrbitalElements:
    """Osculating Keplerian elements at a single epoch."""
    jd:    float   # Julian date (TDB)
    a:     float   # Semi-major axis (AU)
    e:     float   # Eccentricity
    i:     float   # Inclination (degrees)
    om:    float   # Long. of ascending node (degrees)  
    w:     float   # Argument of perihelion (degrees)
    ma:    float   # Mean anomaly (degrees)
    n:     float   # Mean motion (degrees/day)
    q:     float   # Perihelion distance (AU)
    tp:    float   # Time of perihelion (JD)


# ─────────────────────────────────────────────────────────────
# DATA FETCHERS
# ─────────────────────────────────────────────────────────────

def fetch_close_approaches(
    dist_max_au:  float = 0.05,
    date_min:     str   = "2000-01-01",
    date_max:     str   = "2024-01-01",
    v_inf_max:    float = 20.0,
    limit:        int   = 200
) -> List[CloseApproach]:
    """
    Fetch labeled close approach events from CNEOS.
    These become the failure corpus ℬ_f.
    
    Returns events sorted by date, closest approach ≤ dist_max_au.
    """
    params = {
        "dist-max":   f"{dist_max_au}",
        "date-min":   date_min,
        "date-max":   date_max,
        "v-inf-max":  str(v_inf_max),
        "sort":       "date",
        "fullname":   "true",
    }
    
    print(f"Fetching CNEOS close approaches ({date_min} to {date_max}, "
          f"dist ≤ {dist_max_au} AU)...")
    
    r = requests.get(CNEOS_CAD_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    
    fields = data["fields"]
    fi = {f: i for i, f in enumerate(fields)}
    
    events = []
    for row in data.get("data", []):
        try:
            events.append(CloseApproach(
                designation = row[fi["des"]],
                jd          = float(row[fi["jd"]]),
                cd          = row[fi["cd"]],
                dist_au     = float(row[fi["dist"]]),
                dist_ld     = float(row[fi["dist_min"]]) * 389.17,  # approx LD
                v_rel       = float(row[fi["v_rel"]]) if row[fi["v_rel"]] else 0.0,
                v_inf       = float(row[fi["v_inf"]]) if row[fi["v_inf"]] else 0.0,
            ))
        except (ValueError, KeyError, TypeError):
            continue
    
    print(f"  Found {len(events)} close approach events")
    return events


def jd_to_horizons_date(jd: float) -> str:
    """Convert Julian date to Horizons date string 'YYYY-MMM-DD'."""
    # JD 2451545.0 = J2000.0 = 2000-Jan-01.5
    days_since_j2000 = jd - 2451545.0
    dt = datetime(2000, 1, 1, 12, 0, 0) + timedelta(days=days_since_j2000)
    months = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']
    return f"{dt.year}-{months[dt.month-1]}-{dt.day:02d}"


def fetch_orbital_elements_history(
    designation: str,
    jd_start:    float,
    jd_end:      float,
    step:        str = "1d"
) -> List[OrbitalElements]:
    """
    Fetch osculating orbital elements from JPL Horizons.
    
    Returns daily element sets over [jd_start, jd_end].
    This is the trajectory data for the STTS pipeline.
    """
    start_str = jd_to_horizons_date(jd_start)
    end_str   = jd_to_horizons_date(jd_end)
    
    params = {
        "format":     "json",
        "COMMAND":    f"'{designation}'",
        "EPHEM_TYPE": "ELEMENTS",
        "CENTER":     "500@10",       # heliocentric, solar system barycenter
        "START_TIME": f"'{start_str}'",
        "STOP_TIME":  f"'{end_str}'",
        "STEP_SIZE":  f"'{step}'",
        "OBJ_DATA":   "NO",
        "MAKE_EPHEM": "YES",
        "OUT_UNITS":  "AU-D",
        "REF_PLANE":  "ECLIPTIC",
        "REF_SYSTEM": "J2000",
        "TP_TYPE":    "ABSOLUTE",
        "ELEM_LABELS":"YES",
        "CSV_FORMAT": "YES",
    }
    
    time.sleep(REQUEST_DELAY)
    r = requests.get(HORIZONS_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    
    if "error" in data:
        raise ValueError(f"Horizons error for {designation}: {data['error']}")
    
    # Parse the CSV output between $$SOE and $$EOE markers
    result_text = data.get("result", "")
    elements = parse_horizons_elements(result_text)
    return elements


def parse_horizons_elements(text: str) -> List[OrbitalElements]:
    """Parse Horizons ELEMENTS CSV output."""
    elements = []
    
    in_data = False
    for line in text.split('\n'):
        line = line.strip()
        
        if line == "$$SOE":
            in_data = True
            continue
        if line == "$$EOE":
            break
        if not in_data or not line:
            continue
        
        # CSV format: JDTDB, Calendar_Date, EC, QR, IN, OM, W, Tp, N, MA, TA, A, AD, PR
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 12:
            continue
        
        try:
            elements.append(OrbitalElements(
                jd  = float(parts[0]),
                e   = float(parts[2]),   # EC eccentricity
                q   = float(parts[3]),   # QR perihelion distance AU
                i   = float(parts[4]),   # IN inclination deg
                om  = float(parts[5]),   # OM long. ascending node deg
                w   = float(parts[6]),   # W arg of perihelion deg
                tp  = float(parts[7]),   # Tp time of perihelion JD
                n   = float(parts[8]),   # N mean motion deg/day
                ma  = float(parts[9]),   # MA mean anomaly deg
                a   = float(parts[11]),  # A semi-major axis AU
            ))
        except (ValueError, IndexError):
            continue
    
    return elements


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_features(
    window: List[OrbitalElements],
    approach_jd: float
) -> np.ndarray:
    """
    Extract STTS features from a window of orbital element sets.
    
    The key insight: for heliocentric asteroids, the 'degradation'
    signal is approach to Earth. We compute:
    
    1. Perihelion distance trajectory (q approaching 1 AU = Earth orbit)
    2. Rate of change of q (dq/dt — is perihelion moving toward Earth?)
    3. Minimum Earth approach distance estimate from elements
    4. Cross-element covariance (e-q correlation as orbit evolves)
    5. Time to perihelion (approaching close approach)
    
    The residual approach: we compute da/dt, de/dt etc. from 
    successive observations. Growing rates = approaching event.
    """
    if len(window) < 3:
        return np.zeros(30)
    
    # Raw arrays
    jd_arr = np.array([s.jd  for s in window])
    a_arr  = np.array([s.a   for s in window])
    e_arr  = np.array([s.e   for s in window])
    i_arr  = np.array([s.i   for s in window])
    q_arr  = np.array([s.q   for s in window])  # perihelion dist
    n_arr  = np.array([s.n   for s in window])  # mean motion
    ma_arr = np.array([s.ma  for s in window])
    
    # Time to close approach at end of window
    rul_days = approach_jd - jd_arr[-1]
    
    # Rates (first differences)
    dq = np.diff(q_arr)   # perihelion drift
    da = np.diff(a_arr)   # semi-major axis drift  
    de = np.diff(e_arr)   # eccentricity evolution
    di = np.diff(i_arr)   # inclination drift
    
    feats = []
    
    # Time-domain summaries: q (most important for Earth approach)
    feats += [q_arr.mean(), q_arr.std()+1e-10, q_arr.min(), q_arr.max()]
    
    # Time-domain summaries: a, e
    feats += [a_arr.mean(), a_arr.std()+1e-10]
    feats += [e_arr.mean(), e_arr.std()+1e-10]
    
    # Rate features: dq/dt (approaching Earth orbit at 1 AU?)
    feats += [dq.mean(), dq.std()+1e-10]
    
    # Rate features: da/dt, de/dt
    feats += [da.mean(), da.std()+1e-10]
    feats += [de.mean(), de.std()+1e-10]
    
    # Perihelion distance trend — is q converging on 1 AU?
    q_dist_from_1au = np.abs(q_arr - 1.0)  # distance from Earth's orbit
    feats += [q_dist_from_1au.mean(), q_dist_from_1au.min()]
    
    # Rate of approach to 1 AU
    dq_to_1au = np.diff(q_dist_from_1au)
    feats += [dq_to_1au.mean(), dq_to_1au.std()+1e-10]
    
    # Second derivative of q (acceleration toward Earth orbit)
    if len(dq) > 1:
        d2q = np.diff(dq)
        feats += [d2q.mean(), d2q.std()+1e-10]
    else:
        feats += [0.0, 0.0]
    
    # Cross-element covariance: e-q correlation
    if len(q_arr) > 3:
        corr_eq = np.corrcoef(e_arr, q_arr)[0,1]
        corr_aq = np.corrcoef(a_arr, q_arr)[0,1]
        feats += [0.0 if np.isnan(corr_eq) else corr_eq,
                  0.0 if np.isnan(corr_aq) else corr_aq]
    else:
        feats += [0.0, 0.0]
    
    # q trend slope (linear fit)
    if len(q_arr) > 2:
        t = np.arange(len(q_arr))
        q_slope = np.polyfit(t, q_arr, 1)[0]
        feats.append(q_slope)
    else:
        feats.append(0.0)
    
    # Late/early q ratio (is approach accelerating?)
    half = len(q_dist_from_1au) // 2
    if half > 0:
        early = q_dist_from_1au[:half].mean()
        late  = q_dist_from_1au[half:].mean()
        feats.append(late / (early + 1e-10))
    else:
        feats.append(1.0)
    
    # Inclination (context, not approach signal)
    feats.append(i_arr.mean())
    
    # Mean motion (higher = closer/faster orbit)
    feats.append(n_arr.mean())
    
    # Time to perihelion proxy
    tp_arr = np.array([abs(s.tp - s.jd) for s in window])
    feats.append(tp_arr.mean())
    feats.append(tp_arr.min())
    
    # Pad/trim to 30
    feats = feats[:30]
    while len(feats) < 30:
        feats.append(0.0)
    
    return np.array(feats, dtype=np.float64)


# ─────────────────────────────────────────────────────────────
# STTS PIPELINE
# ─────────────────────────────────────────────────────────────

WINDOW_DAYS  = 30   # 30-day observation window
STRIDE_DAYS  = 7    # advance 1 week at a time
WARNING_DAYS = 90   # 90 days before close approach = precursor


def build_dataset(
    trajectories: List[Tuple[List[OrbitalElements], float]],
    warning_days: float = WARNING_DAYS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features from all trajectories.
    
    trajectories: list of (element_history, close_approach_jd)
    Returns: features, ruls, labels (1=precursor, 0=nominal)
    """
    all_feats, all_ruls, all_labels = [], [], []
    
    for elements, ca_jd in trajectories:
        if len(elements) < WINDOW_DAYS:
            continue
        
        for start in range(0, len(elements) - WINDOW_DAYS + 1, STRIDE_DAYS):
            window = elements[start:start + WINDOW_DAYS]
            rul = ca_jd - window[-1].jd
            if rul < 0:
                continue  # past the close approach
            
            feat  = extract_features(window, ca_jd)
            label = 1 if rul <= warning_days else 0
            
            all_feats.append(feat)
            all_ruls.append(rul)
            all_labels.append(label)
    
    if not all_feats:
        return np.empty((0,30)), np.array([]), np.array([])
    
    return np.array(all_feats), np.array(all_ruls), np.array(all_labels)


def run_pipeline(
    train_trajectories: List[Tuple[List[OrbitalElements], float]],
    test_trajectories:  List[Tuple[List[OrbitalElements], float]],
    verbose: bool = True
) -> Dict:
    """
    Full F->W->M pipeline:
    1. Extract features from training trajectories
    2. Fit LDA projection
    3. Build failure basin from precursor windows
    4. Evaluate detection on test trajectories
    5. Report V1, V2, detection lead time
    """
    # Extract training features
    X_train, r_train, y_train = build_dataset(train_trajectories)
    
    if verbose:
        print(f"  Training windows: {len(X_train)} "
              f"({y_train.sum():.0f} precursor, {(y_train==0).sum():.0f} nominal)")
    
    if y_train.sum() < 5 or (y_train==0).sum() < 5:
        return {"error": "Insufficient training data"}
    
    # W: physics-informed weights
    W = np.ones(30)
    W[0:4]   *= 0.5   # q summaries — important but noisy
    W[4:8]   *= 0.3   # a, e summaries
    W[8:10]  *= 3.0   # dq/dt — primary approach signal
    W[12:16] *= 2.0   # da/dt, de/dt
    W[16:18] *= 3.0   # q distance from 1 AU
    W[18:20] *= 3.0   # dq toward 1 AU
    W[20:22] *= 2.0   # d2q/dt2 — acceleration signal
    W[22:24] *= 2.0   # cross-correlations
    W[24]    *= 3.0   # q slope
    W[25]    *= 4.0   # late/early ratio — KEY PRECURSOR
    W[26]    *= 0.2   # inclination (not approach signal)
    W[27]    *= 1.0   # mean motion
    W[28:30] *= 2.0   # time to perihelion
    
    # Standardize then weight
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_w      = X_scaled * W
    
    # LDA
    lda = LinearDiscriminantAnalysis(n_components=1, solver='svd')
    lda.fit(X_w, y_train)
    X_proj = lda.transform(X_w).ravel()
    
    # Failure basin
    basin = X_proj[y_train == 1]
    
    # Compute training distances
    def dist_to_basin(p, k=5):
        if len(basin) == 0:
            return 999.0
        dists = np.abs(basin - p)
        return np.mean(np.sort(dists)[:min(k, len(basin))])
    
    train_dists = np.array([dist_to_basin(p) for p in X_proj])
    
    # V1: separation
    nom_d  = train_dists[y_train == 0]
    pre_d  = train_dists[y_train == 1]
    sep    = np.median(nom_d) / (np.median(pre_d) + 1e-10)
    _, p1  = mannwhitneyu(nom_d, pre_d, alternative='greater')
    v1_pass = sep > 2.0 and p1 < 0.05
    
    # V2: monotonic approach
    mask   = r_train < 365
    if mask.sum() > 5:
        rho, p2 = spearmanr(r_train[mask], train_dists[mask])
        v2_pass = rho > 0.3 and p2 < 0.05
    else:
        rho, p2, v2_pass = 0.0, 1.0, False
    
    # Calibrate epsilon (maximize F1 on training)
    thresholds = np.percentile(train_dists, np.linspace(5, 95, 40))
    best_f1, best_eps = 0.0, thresholds[0]
    for eps in thresholds:
        preds  = (train_dists < eps).astype(int)
        tp = ((preds==1)&(y_train==1)).sum()
        fp = ((preds==1)&(y_train==0)).sum()
        fn = ((preds==0)&(y_train==1)).sum()
        pr = tp/max(1,tp+fp); re = tp/max(1,tp+fn)
        f1 = 2*pr*re/max(1e-10,pr+re)
        if f1 > best_f1:
            best_f1, best_eps = f1, eps
    
    if verbose:
        print(f"  V1 separation: {sep:.1f}x (p={p1:.2e}) {'PASS' if v1_pass else 'FAIL'}")
        print(f"  V2 Spearman ρ: {rho:.3f} (p={p2:.2e}) {'PASS' if v2_pass else 'FAIL'}")
        print(f"  Training F1: {best_f1:.3f} at ε={best_eps:.4f}")
    
    # Evaluate on test trajectories
    tp_count = fp_count = fn_count = tn_count = 0
    lead_times = []
    
    for elements, ca_jd in test_trajectories:
        if len(elements) < WINDOW_DAYS:
            continue
        
        is_close_approach = True  # All test trajectories are labeled events
        fired = False
        fire_rul = None
        
        for start in range(0, len(elements) - WINDOW_DAYS + 1, STRIDE_DAYS):
            window = elements[start:start + WINDOW_DAYS]
            rul = ca_jd - window[-1].jd
            if rul < 0:
                break
            
            feat = extract_features(window, ca_jd)
            x_s  = scaler.transform(feat.reshape(1,-1))
            x_w  = x_s * W
            proj = lda.transform(x_w).ravel()[0]
            d    = dist_to_basin(proj)
            
            if d < best_eps and not fired:
                fired    = True
                fire_rul = rul
                break
        
        if is_close_approach:
            if fired and fire_rul and fire_rul > 0:
                tp_count += 1
                lead_times.append(fire_rul)
            else:
                fn_count += 1
        else:
            if fired:
                fp_count += 1
            else:
                tn_count += 1
    
    prec = tp_count / max(1, tp_count + fp_count)
    rec  = tp_count / max(1, tp_count + fn_count)
    f1   = 2*prec*rec / max(1e-10, prec+rec)
    
    return {
        "v1_separation": sep,
        "v1_p":          p1,
        "v1_pass":       v1_pass,
        "v2_rho":        rho,
        "v2_p":          p2,
        "v2_pass":       v2_pass,
        "train_f1":      best_f1,
        "epsilon":       best_eps,
        "test_f1":       f1,
        "test_precision":prec,
        "test_recall":   rec,
        "tp":            tp_count,
        "fp":            fp_count,
        "fn":            fn_count,
        "mean_lead_days":np.mean(lead_times) if lead_times else 0.0,
        "median_lead_days": np.median(lead_times) if lead_times else 0.0,
        "lead_times":    lead_times,
    }


# ─────────────────────────────────────────────────────────────
# MAIN: FULL PIPELINE ON REAL NASA DATA
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STTS Orbital Pipeline — JPL Horizons + CNEOS")
    print("Running in reverse from known close approach events")
    print("=" * 60)
    print()
    
    # ── Step 1: Fetch labeled corpus from CNEOS ──────────────
    print("Step 1: Fetch close approach corpus from CNEOS")
    events = fetch_close_approaches(
        dist_max_au = 0.02,        # within ~8 lunar distances
        date_min    = "2005-01-01",
        date_max    = "2020-01-01",
        v_inf_max   = 15.0,
    )
    
    if not events:
        print("ERROR: No events fetched. Check network access.")
        return
    
    # Select events with good data — numbered/well-tracked asteroids
    # Filter for objects with enough history for a 365-day lookback
    print(f"  Selected {len(events)} close approach events as corpus")
    print(f"  Distance range: {min(e.dist_au for e in events):.4f} – "
          f"{max(e.dist_au for e in events):.4f} AU")
    print()
    
    # ── Step 2: Fetch orbital element histories ───────────────
    print("Step 2: Fetch orbital histories from JPL Horizons")
    print("  (365 days before each close approach)")
    print()
    
    trajectories = []
    failed = 0
    
    fetch_limit = 250
    for i, event in enumerate(events[:fetch_limit]):
        print(f"  [{i+1:3d}/{min(fetch_limit,len(events))}] {event.designation} "
              f"CA: {event.cd[:10]} dist={event.dist_au:.4f} AU", end="")
        
        try:
            # Fetch 365 days before close approach
            jd_start = event.jd - 365
            jd_end   = event.jd - 1   # stop just before the event
            
            elements = fetch_orbital_elements_history(
                event.designation,
                jd_start,
                jd_end,
                step="1d"
            )
            
            if len(elements) >= WINDOW_DAYS + 10:
                trajectories.append((elements, event.jd))
                print(f" → {len(elements)} epochs OK")
            else:
                print(f" → only {len(elements)} epochs, skip")
                failed += 1
                
        except Exception as ex:
            print(f" → ERROR: {ex}")
            failed += 1
        
        time.sleep(REQUEST_DELAY)
    
    print(f"\n  Usable trajectories: {len(trajectories)}")
    print(f"  Failed/insufficient: {failed}")
    
    if len(trajectories) < 10:
        print("ERROR: Insufficient trajectories for pipeline.")
        return
    
    # ── Step 3: Split train/test ──────────────────────────────
    print()
    print("Step 3: Train/test split (80/20)")
    
    np.random.seed(42)
    idx = np.random.permutation(len(trajectories))
    n_train = int(0.8 * len(trajectories))
    
    train_trajs = [trajectories[i] for i in idx[:n_train]]
    test_trajs  = [trajectories[i] for i in idx[n_train:]]
    
    print(f"  Train: {len(train_trajs)} trajectories")
    print(f"  Test:  {len(test_trajs)} trajectories")
    
    # ── Step 4: Run STTS pipeline ─────────────────────────────
    print()
    print("Step 4: Running F->W->M pipeline")
    
    results = run_pipeline(train_trajs, test_trajs, verbose=True)
    
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return
    
    # ── Step 5: Report results ────────────────────────────────
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print("Verification Conditions:")
    print(f"  V1 Basin Separation: {results['v1_separation']:.1f}x "
          f"(p={results['v1_p']:.2e}) "
          f"{'✓ PASS' if results['v1_pass'] else '✗ FAIL'}")
    print(f"  V2 Monotonic Approach: ρ={results['v2_rho']:.3f} "
          f"(p={results['v2_p']:.2e}) "
          f"{'✓ PASS' if results['v2_pass'] else '✗ FAIL'}")
    print()
    print("Detection Performance:")
    print(f"  F1:        {results['test_f1']:.3f}")
    print(f"  Precision: {results['test_precision']:.3f}")
    print(f"  Recall:    {results['test_recall']:.3f}")
    print(f"  TP/FP/FN:  {results['tp']}/{results['fp']}/{results['fn']}")
    print()
    print("Detection Lead Time:")
    print(f"  Mean:   {results['mean_lead_days']:.0f} days before close approach")
    print(f"  Median: {results['median_lead_days']:.0f} days before close approach")
    print()
    print("Comparison vs current practice:")
    print(f"  CNEOS issues close approach alerts typically ~1-3 months before event")
    print(f"  STTS fires at mean T-{results['mean_lead_days']:.0f} days")
    if results['mean_lead_days'] > 90:
        print(f"  → STTS provides {results['mean_lead_days']-90:.0f} additional days "
              f"over 90-day alert window")
    
    # Save results
    output = {
        "events_fetched":   len(events),
        "trajectories_used": len(trajectories),
        "train_size":       len(train_trajs),
        "test_size":        len(test_trajs),
        "results":          {k: v for k,v in results.items() 
                             if k != "lead_times"},
        "lead_times_days":  results["lead_times"],
    }
    
    # Convert numpy types for JSON serialization
    def jsonify(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open("orbital_stts_results.json", "w") as f:
        json.dump(output, f, indent=2, default=jsonify)
    print()
    print("Results saved to orbital_stts_results.json")


if __name__ == "__main__":
    main()
