"""
Synthetic orbital corpus generator for STTS orbital pipeline.
Calibrated against: ISS, Mir, Soyuz RB, CZ-4B RB, Iridium.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

RE = 6371.0
MU = 398600.4418

# Empirical decay model: log(rate) = C0 + C1*log(B*) + C2*alt
DECAY_C0 = 10.565
DECAY_C1 =  1.208
DECAY_C2 = -0.00707


@dataclass
class OrbitalState:
    epoch_jd: float
    a: float        # semi-major axis (km)
    e: float        # eccentricity
    inc: float      # inclination (deg)
    raan: float     # RAAN (deg)
    argp: float     # arg of perigee (deg)
    bstar: float    # drag term (1/Re)
    norad_id: int
    rul_days: float # 9999 = nominal


def decay_rate(altitude_km: float, bstar: float) -> float:
    """Decay rate in km/day (negative). Calibrated to real objects."""
    alt = max(120.0, min(800.0, altitude_km))
    return -np.exp(DECAY_C0 + DECAY_C1 * np.log(max(1e-7, bstar)) + DECAY_C2 * alt)


def estimate_rul(altitude_km: float, bstar: float) -> float:
    """Days until altitude drops below 120 km."""
    a, days = altitude_km, 0.0
    while a > 120.0 and days < 5000:
        a += decay_rate(a, bstar)
        days += 1.0
    return days


def raan_rate(a_km, e, inc_deg):
    """RAAN precession rate deg/day from J2."""
    J2 = 1.08263e-3
    n = np.sqrt(MU / a_km**3) * 86400 / (2 * np.pi)
    p = a_km * (1 - e**2)
    return -1.5 * n * J2 * (RE / p)**2 * np.cos(np.radians(inc_deg)) * 360


def simulate_reentry_object(norad_id, initial_altitude_km,
                             inclination_deg, bstar_base,
                             solar_factor=1.0, rng=None):
    if rng is None:
        rng = np.random.RandomState(norad_id)

    a = RE + initial_altitude_km
    e = rng.uniform(0.0001, 0.003)
    inc = inclination_deg + rng.normal(0, 0.3)
    raan_val = rng.uniform(0, 360)
    argp = rng.uniform(0, 360)
    jd = 2458850.0

    states = []
    step = 0
    while (a - RE) > 120.0 and step < 5000:
        solar = 1.0 + 0.25 * np.sin(2 * np.pi * step / 365)
        bstar = max(1e-7, bstar_base * solar_factor * solar * (1 + rng.normal(0, 0.05)))
        e = max(1e-5, e * (1 - 5e-4))
        raan_val = (raan_val + raan_rate(a, e, inc)) % 360
        argp = (argp + 0.15 + rng.normal(0, 0.05)) % 360

        states.append(OrbitalState(
            epoch_jd=jd + step,
            a=a + rng.normal(0, 0.03),
            e=max(1e-5, e + rng.normal(0, 5e-7)),
            inc=inc + rng.normal(0, 0.0005),
            raan=raan_val,
            argp=argp,
            bstar=bstar,
            norad_id=norad_id,
            rul_days=0.0
        ))
        a += decay_rate(a - RE, bstar)
        step += 1

    for i, s in enumerate(states):
        s.rul_days = float(len(states) - i)
    return states


def simulate_nominal_object(norad_id, altitude_km, inclination_deg,
                              bstar_base, n_days=180, rng=None):
    if rng is None:
        rng = np.random.RandomState(norad_id + 50000)

    a = RE + altitude_km
    e = rng.uniform(0.0001, 0.002)
    inc = inclination_deg + rng.normal(0, 0.3)
    raan_val = rng.uniform(0, 360)
    argp = rng.uniform(0, 360)
    jd = 2458850.0

    states = []
    for step in range(n_days):
        bstar = max(1e-7, bstar_base * (1 + rng.normal(0, 0.03)))
        e = max(1e-5, e * (1 - 1e-4))
        raan_val = (raan_val + raan_rate(a, e, inc)) % 360
        argp = (argp + 0.1 + rng.normal(0, 0.05)) % 360

        states.append(OrbitalState(
            epoch_jd=jd + step,
            a=a + rng.normal(0, 0.03),
            e=max(1e-5, e + rng.normal(0, 5e-7)),
            inc=inc + rng.normal(0, 0.0005),
            raan=raan_val,
            argp=argp,
            bstar=bstar,
            norad_id=norad_id,
            rul_days=9999.0
        ))
        a += decay_rate(a - RE, bstar)
    return states


def generate_corpus(n_reentry=150, n_nominal=150, seed=42, verbose=True):
    rng = np.random.RandomState(seed)
    INCLS = [28.5, 51.6, 65.0, 82.0, 97.8, 98.7]
    WGTS  = [0.15, 0.35, 0.15, 0.10, 0.15, 0.10]

    reentry = []
    if verbose:
        print(f"Generating {n_reentry} reentry trajectories...")
    for i in range(n_reentry):
        r = np.random.RandomState(seed + i)
        obj = r.choice(['rocket_body', 'payload', 'debris'], p=[0.30, 0.40, 0.30])
        if obj == 'rocket_body':
            alt, bs = r.uniform(250, 400), r.uniform(5e-4, 3e-3)
        elif obj == 'payload':
            alt, bs = r.uniform(300, 500), r.uniform(1e-4, 8e-4)
        else:
            alt, bs = r.uniform(200, 450), r.uniform(1e-4, 5e-3)

        inc = r.choice(INCLS, p=WGTS) + r.normal(0, 2)
        traj = simulate_reentry_object(
            10000 + i, alt, inc, bs,
            solar_factor=r.uniform(0.8, 1.4), rng=r
        )
        if len(traj) >= 30:
            reentry.append(traj)

    nominal = []
    if verbose:
        print(f"Generating {n_nominal} nominal trajectories...")
    for i in range(n_nominal):
        r = np.random.RandomState(seed + n_reentry + i)
        obj = r.choice(['operational', 'high_alt', 'polar'], p=[0.50, 0.30, 0.20])
        if obj == 'operational':
            alt, bs = r.uniform(450, 650), r.uniform(5e-5, 2e-4)
            inc = r.choice([28.5, 51.6], p=[0.4, 0.6]) + r.normal(0, 1)
        elif obj == 'high_alt':
            alt, bs = r.uniform(700, 900), r.uniform(1e-5, 8e-5)
            inc = r.uniform(0, 180)
        else:
            alt, bs = r.uniform(500, 700), r.uniform(5e-5, 3e-4)
            inc = r.choice([97.8, 98.7]) + r.normal(0, 1)

        traj = simulate_nominal_object(20000 + i, alt, inc, bs, rng=r)
        nominal.append(traj)

    if verbose:
        lens = [len(t) for t in reentry]
        print(f"  Reentry: {len(reentry)} objects, "
              f"avg {np.mean(lens):.0f} days [{np.min(lens)}-{np.max(lens)}]")
        print(f"  Nominal: {len(nominal)} objects, 180 days each")

    return reentry, nominal


if __name__ == "__main__":
    print("=== Decay Model Calibration ===")
    cases = [
        (420, 2e-4, "ISS",     0.066),
        (386, 3e-4, "Mir",     0.150),
        (350, 8e-4, "Soyuz RB",0.500),
        (300, 1.5e-3,"CZ-4B",  2.000),
        (780, 5e-5, "Iridium", 0.001),
    ]
    for alt, bs, label, target in cases:
        rate = abs(decay_rate(alt, bs))
        err = abs(rate - target) / target * 100
        print(f"  {label:10s} {alt}km: {rate:.4f} km/day "
              f"(target {target:.3f}, err {err:.0f}%)")

    print("\n=== Lifetime Estimates ===")
    for alt, bs, label in [
        (550, 1e-4, "Stable 550km"),
        (450, 2e-4, "ISS-like 450km"),
        (400, 3e-4, "Decaying 400km"),
        (350, 5e-4, "Decaying 350km"),
        (300, 1e-3, "RB 300km"),
        (250, 3e-3, "RB 250km"),
    ]:
        rul = estimate_rul(alt, bs)
        note = f"{rul/365:.1f} yr" if rul > 365 else f"{rul:.0f} d"
        print(f"  {label:20s}: {note}")

    print("\n=== Test Corpus (20+20) ===")
    re, nom = generate_corpus(20, 20, seed=42)
    t = re[0]
    print(f"Sample: {t[0].a-RE:.0f}km -> {t[-1].a-RE:.0f}km "
          f"over {len(t)} days, "
          f"decay x{abs(decay_rate(t[-1].a-RE,t[-1].bstar)/decay_rate(t[0].a-RE,t[0].bstar)):.1f}")
