"""
Test the field/geodesic decomposition on existing mechanism data.

Hypothesis from theoretical discussion:
  M2 (centroid drift)        ~ geodesic displacement (how state moves)
  M1 (participation ratio)   ~ curvature-source (how manifold is shaped)
  M5 (mean abs correlation)  ~ curvature-source
  M6 (subspace angle)        ~ curvature-source

Prediction: across the 9 domains, some should be geodesic-dominant
(flat STTS captures the signal) and some should be curvature-source-dominant
(flat STTS misses structure). Bearing and solar — where flat STTS shows
weak M2 — should fall in the curvature-source-dominant group.

This is a free test: it uses only the existing mechanism_discovery.json
from the 9-domain experiment. No new computation of curvature needed.

Usage:
    python -m geometry.field_geodesic_test
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geometry.config import FIG_DIR, RESULTS_DIR


MECH_PATH = Path(__file__).parent.parent / "experiments" / "mechanism_discovery" / "mechanism_discovery.json"

# Decomposition labels from the theoretical proposal
GEODESIC_MECHANISMS = ["m2_centroid_distance"]
CURVATURE_SOURCE_MECHANISMS = [
    "m1_participation_ratio",
    "m5_mean_abs_correlation",
    "m6_subspace_angle",
]
# Left out: M3 (velocity — also geodesic-like but weak everywhere),
# M4 (mean variance — baseline/Λop-like), M7 (LDA fluctuation — meta).


def signal_coherence(values: list[float]) -> dict:
    """Assess whether per-trajectory rho values agree in direction.

    Incoherent = range crosses zero and/or std > 0.25. The domain-mean
    is then aggregating over qualitatively different behaviors and isn't
    a reliable estimate of a domain-wide phenomenon.
    """
    vals = np.array(values)
    n = len(vals)
    mean = float(vals.mean())
    std = float(vals.std())
    rng = (float(vals.min()), float(vals.max()))
    # SE of the mean (small n => wide interval)
    se = std / max(1, np.sqrt(n))
    # Bootstrap-free 95% CI from ~1.96*se, floor/ceil clipped to [-1, 1]
    ci = (max(-1.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se))
    crosses_zero = rng[0] < 0 < rng[1] or ci[0] < 0 < ci[1]
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "range": rng,
        "se": float(se),
        "ci_95": ci,
        "crosses_zero": bool(crosses_zero),
        "incoherent": bool(crosses_zero or std > 0.25),
    }


def domain_scores(correlation_tables: dict) -> dict:
    """For each domain, compute:
      G = |mean rho_vs_rul of M2|  (geodesic strength)
      K = max |mean rho_vs_rul| over curvature-source mechanisms
      K_which = which curvature-source mechanism wins
      + coherence flags: is the mean itself a reliable estimate?
    """
    out = {}
    for domain, table in correlation_tables.items():
        rho = table["rho_vs_rul"]
        m2 = rho[GEODESIC_MECHANISMS[0]]
        g = abs(m2["mean"])
        g_coherence = signal_coherence(m2["values"])

        curv_abs = {
            m: abs(rho[m]["mean"]) for m in CURVATURE_SOURCE_MECHANISMS
        }
        k_which = max(curv_abs, key=curv_abs.get)
        k = curv_abs[k_which]
        k_coherence = signal_coherence(rho[k_which]["values"])

        out[domain] = {
            "geodesic_score": g,
            "geodesic_coherence": g_coherence,
            "curvature_source_score": k,
            "curvature_winner": k_which,
            "curvature_coherence": k_coherence,
            "m2_signed": float(m2["mean"]),
            "ratio_G_over_K": g / max(k, 1e-6),
            "dominant": "geodesic" if g > k else "curvature_source",
            "mixed": 0.8 <= g / max(k, 1e-6) <= 1.25,
            "m1_score": abs(rho["m1_participation_ratio"]["mean"]),
            "m5_score": abs(rho["m5_mean_abs_correlation"]["mean"]),
            "m6_score": abs(rho["m6_subspace_angle"]["mean"]),
            # Reliability: is the DOMAIN MEAN of M2 a meaningful estimate?
            # Requires: (1) per-trajectory M2 rho values don't cross zero
            # (sign coherence — trajectories agree on direction), and
            # (2) n >= 5 (basic sample size).
            # Note: does NOT require K coherence — we care whether the
            # dominance call is reliable, and that's about whether G is real.
            "reliable": (
                not g_coherence["crosses_zero"]
                and g_coherence["n"] >= 5
            ),
        }
    return out


def main():
    with open(MECH_PATH) as f:
        data = json.load(f)

    scores = domain_scores(data["correlation_tables"])

    # Order domains for display. Bearing (PRONOSTIA) was removed 2026-04-16:
    # six accelerated-rig bearings can't support any cross-domain geometric
    # claim. Re-introduce only with a real-operating-conditions dataset.
    domain_order = [
        "protein", "cmapss_fd001", "battery", "orbital",
        "ncmapss_ds03", "ncmapss_ds02", "reentry",
        "solar",
    ]

    n_domains = sum(1 for d in domain_order if d in scores)
    print("=" * 78)
    print(f"Field/geodesic decomposition across {n_domains} domains (bearing excluded)")
    print("=" * 78)
    print(f"  G = |mean rho of M2 vs RUL|  (geodesic strength)")
    print(f"  K = max |mean rho| over M1/M5/M6  (curvature-source strength)")
    print()
    print(f"{'domain':<18s} {'n':>4s} {'G':>8s} {'K':>8s} {'G/K':>7s} {'M2 std':>8s} "
          f"{'M2 cross0':>10s} {'reliable':>10s}")
    print("-" * 82)
    for d in domain_order:
        if d not in scores:
            continue
        s = scores[d]
        n = s["geodesic_coherence"]["n"]
        m2_std = s["geodesic_coherence"]["std"]
        m2_cross = "yes" if s["geodesic_coherence"]["crosses_zero"] else "no"
        rel = "YES" if s["reliable"] else "no"
        print(f"{d:<18s} {n:>4d} {s['geodesic_score']:>8.3f} {s['curvature_source_score']:>8.3f} "
              f"{s['ratio_G_over_K']:>7.2f} {m2_std:>8.3f} {m2_cross:>10s} {rel:>10s}")

    # Group classification — but only among reliable domains
    reliable_domains = [d for d, s in scores.items() if s["reliable"]]
    unreliable_domains = [d for d, s in scores.items() if not s["reliable"]]

    geodesic_dom = [d for d in reliable_domains
                    if scores[d]["dominant"] == "geodesic" and not scores[d]["mixed"]]
    curv_dom = [d for d in reliable_domains
                if scores[d]["dominant"] == "curvature_source" and not scores[d]["mixed"]]
    mixed = [d for d in reliable_domains if scores[d]["mixed"]]

    print()
    print("Classification (RELIABLE domains only — coherent M2 AND n >= 5):")
    print(f"  Geodesic-dominant:         {sorted(geodesic_dom)}")
    print(f"  Curvature-source-dominant: {sorted(curv_dom)}")
    print(f"  Mixed (G ~ K):             {sorted(mixed)}")
    print()
    print(f"Unreliable (incoherent per-trajectory signal, cannot classify):")
    print(f"  {sorted(unreliable_domains)}")

    # --- Test the specific prediction on reliable domains only ---
    print()
    print("Prediction check: 'flat-STTS-weak domains are curvature-source-dominant'")
    for d in ["solar"]:
        if d not in scores:
            continue
        s = scores[d]
        if not s["reliable"]:
            print(f"  {d}: UNRELIABLE "
                  f"(n={s['geodesic_coherence']['n']}, M2 std={s['geodesic_coherence']['std']:.3f}, "
                  f"M2 range crosses zero={s['geodesic_coherence']['crosses_zero']}) "
                  f"— result cannot be used to test the hypothesis")
        else:
            verdict = "CONFIRMED" if s["dominant"] == "curvature_source" else "REJECTED"
            if s["mixed"]:
                verdict = "MIXED"
            print(f"  {d}: RELIABLE. dominant={s['dominant']}, "
                  f"G={s['geodesic_score']:.3f}, K={s['curvature_source_score']:.3f} → {verdict}")

    # --- Figure: G vs K scatter, labelled per domain ---
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7))
    gs = [scores[d]["geodesic_score"] for d in domain_order if d in scores]
    ks = [scores[d]["curvature_source_score"] for d in domain_order if d in scores]
    labels = [d for d in domain_order if d in scores]
    colors = [
        "firebrick" if scores[d]["dominant"] == "curvature_source" else "steelblue"
        for d in labels
    ]
    # Reliable domains: filled markers; unreliable: hollow + hatched
    for x, y, lab, c in zip(gs, ks, labels, colors):
        s = scores[lab]
        if s["reliable"]:
            ax.scatter(x, y, c=c, s=110, edgecolor="black", zorder=3)
        else:
            ax.scatter(x, y, facecolors="none", edgecolors=c, s=110,
                       linewidth=1.8, linestyle="--", zorder=3)
        suffix = "" if s["reliable"] else " (unreliable)"
        ax.annotate(f"{lab}{suffix}", (x, y),
                    textcoords="offset points", xytext=(7, 6), fontsize=8.5)

    lim = max(max(gs), max(ks)) * 1.05
    ax.plot([0, lim], [0, lim], color="gray", linestyle="--", linewidth=0.8, label="G = K")
    ax.fill_between([0, lim], [0, lim], [lim, lim], color="firebrick", alpha=0.07)
    ax.fill_between([0, lim], [0, 0], [0, lim], color="steelblue", alpha=0.07)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Geodesic score: |rho(M2, RUL)|")
    ax.set_ylabel("Curvature-source score: max |rho(M1/M5/M6, RUL)|")
    ax.set_title("Field/geodesic decomposition across 9 STTS domains")
    ax.text(0.72, 0.04, "geodesic-dominant", transform=ax.transAxes,
            color="steelblue", ha="center", fontsize=10, fontweight="bold")
    ax.text(0.18, 0.93, "curvature-source-\ndominant", transform=ax.transAxes,
            color="firebrick", ha="center", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "field_geodesic_decomposition.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {FIG_DIR / 'field_geodesic_decomposition.png'}")

    # --- Save JSON ---
    out = {
        "hypothesis": (
            "M2 (centroid drift) ~ geodesic displacement. "
            "M1/M5/M6 (PR, correlation, rotation) ~ curvature sources. "
            "Domains stratify by dominance."
        ),
        "scores": scores,
        "classification": {
            "geodesic_dominant": sorted(geodesic_dom),
            "curvature_source_dominant": sorted(curv_dom),
            "mixed": sorted(mixed),
        },
        "prediction_check": {
            d: {
                "dominant": scores[d]["dominant"],
                "mixed": scores[d]["mixed"],
                "verdict": (
                    "CONFIRMED" if scores[d]["dominant"] == "curvature_source"
                    and not scores[d]["mixed"]
                    else ("MIXED" if scores[d]["mixed"] else "REJECTED")
                ),
            } for d in ["solar"] if d in scores
        },
    }
    with open(RESULTS_DIR / "field_geodesic_test.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {RESULTS_DIR / 'field_geodesic_test.json'}")


if __name__ == "__main__":
    main()
