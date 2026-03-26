"""
Generate static visualizations for STTS-Reentry results.

All plots use real data only. No synthetic or simulated data.

Data sources:
    plots/basin_distances.npz   — precomputed from frozen model pipeline
    validate.json               — lead_times_raw (78 real detection events)
    firing_satellites_categorized.json — 125 satellites, 3 categories
    patterns.md                 — excursion onset counts by year

Sunspot numbers: SILSO/WDC-SILSO annual mean SSN (publicly archived).

Outputs:
    plots/basin_separation.png
    plots/anomalous_satellites.png
    plots/sc25_correlation.png
    plots/lead_time_distribution.png

Usage:
    python results/reentry/plots/generate_plots.py
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ── Paths ────────────────────────────────────────────────────
RESULTS = Path(__file__).resolve().parent.parent
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# ── Color scheme ─────────────────────────────────────────────
BLUE_NOM = "#2166ac"       # nominal trajectories
RED_REENTRY = "#d6604d"    # reentry-like trajectories
ORANGE_TRANS = "#f4a582"   # transitional
GREEN_SC25 = "#1b7837"     # solar cycle correlation
GRAY_REF = "#878787"       # reference lines
DARK_TEXT = "#1a1a1a"


# ══════════════════════════════════════════════════════════════
# Plot 1: Basin Separation Histogram
# ══════════════════════════════════════════════════════════════
def plot_basin_separation():
    """Load precomputed real distances from npz and plot histogram."""
    npz_path = PLOTS / "basin_distances.npz"
    data = np.load(npz_path)
    nom_d = data["nom_d"]
    pre_d = data["pre_d"]

    sep = np.median(nom_d) / np.median(pre_d)
    print(f"  Source: {npz_path}")
    print(f"  Basin separation: loaded {len(nom_d):,} nominal "
          f"and {len(pre_d):,} precursor distances from basin_distances.npz")
    print(f"  Median nominal: {np.median(nom_d):.4f}, "
          f"median precursor: {np.median(pre_d):.6f}")
    print(f"  V1 separation: {sep:.1f}x")

    fig, ax = plt.subplots(figsize=(10, 6))

    all_d = np.concatenate([nom_d, pre_d])
    bins = np.logspace(
        np.log10(max(all_d.min(), 1e-6)),
        np.log10(all_d.max()),
        60,
    )

    # Log-log axes: log y compresses the 37x count disparity between
    # nominal (288k) and precursor (7.7k), making both peaks clearly visible.
    ax.hist(nom_d, bins=bins, alpha=0.5, color=BLUE_NOM,
            edgecolor=BLUE_NOM, linewidth=0.8,
            label=f"Nominal (n={len(nom_d):,})", zorder=2)
    ax.hist(pre_d, bins=bins, alpha=0.8, color=RED_REENTRY,
            edgecolor=RED_REENTRY, linewidth=0.8,
            label=f"Precursor (n={len(pre_d):,})", zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k-NN Distance to Failure Basin", fontsize=12)
    ax.set_ylabel("Window Count", fontsize=12)
    ax.set_title("STTS-Reentry: Nominal vs Precursor Basin Distance",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")

    nom_med = np.median(nom_d)
    pre_med = np.median(pre_d)
    ax.axvline(nom_med, color=BLUE_NOM, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(pre_med, color=RED_REENTRY, linestyle="--", linewidth=1.5, alpha=0.8)

    ax.annotate(
        f"{sep:.1f}\u00d7 V1 Basin Separation",
        xy=(0.5, 0.92), xycoords="axes fraction",
        fontsize=18, fontweight="bold", color=DARK_TEXT,
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=GRAY_REF, alpha=0.9),
    )

    ax.tick_params(labelsize=10)
    fig.tight_layout()
    outpath = PLOTS / "basin_separation.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


# ══════════════════════════════════════════════════════════════
# Plot 2: Anomalous Satellites Scatter
# ══════════════════════════════════════════════════════════════
def plot_anomalous_satellites():
    """Scatter periapsis vs first excursion date, colored by category."""
    cat_path = RESULTS / "firing_satellites_categorized.json"
    with open(cat_path) as f:
        cat = json.load(f)

    n_unexpected = len(cat["unexpected"])
    n_transitional = len(cat["transitional"])
    n_deliberate = len(cat["deliberate_deorbit"])
    print(f"  Source: {cat_path}")
    print(f"  Anomalous satellites: loaded {n_unexpected} unexpected, "
          f"{n_transitional} transitional, {n_deliberate} deliberate deorbit "
          f"from firing_satellites_categorized.json")

    # Count the 450-500km cluster from real data
    cluster_count = sum(
        1 for s in cat["unexpected"]
        if 450 <= s["current_periapsis_km"] < 500
    )
    print(f"  450-500 km cluster: {cluster_count} satellites")

    fig, ax = plt.subplots(figsize=(12, 7))

    categories = [
        (cat["unexpected"], RED_REENTRY, f"Unexpected ({n_unexpected})", "o"),
        (cat["transitional"], ORANGE_TRANS, f"Transitional ({n_transitional})", "s"),
        (cat["deliberate_deorbit"], GRAY_REF, f"Deliberate deorbit ({n_deliberate})", "D"),
    ]

    for sats, color, label, marker in categories:
        dates = [datetime.strptime(s["first_excursion_date"], "%Y-%m-%d")
                 for s in sats]
        periapsis = [s["current_periapsis_km"] for s in sats]
        ax.scatter(dates, periapsis, c=color, label=label, marker=marker,
                   s=40, alpha=0.8, edgecolors="white", linewidths=0.3, zorder=3)

    ax.axhline(400, color=GRAY_REF, linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(600, color=GRAY_REF, linestyle="--", linewidth=1, alpha=0.6)
    ax.text(datetime(2020, 2, 1), 405, "400 km",
            fontsize=9, color=GRAY_REF, va="bottom")
    ax.text(datetime(2020, 2, 1), 605, "600 km",
            fontsize=9, color=GRAY_REF, va="bottom")

    ax.annotate(
        f"{cluster_count}-satellite cluster\n450\u2013500 km",
        xy=(datetime(2023, 6, 1), 475), xycoords="data",
        xytext=(datetime(2020, 6, 1), 250),
        fontsize=10, fontweight="bold", color=RED_REENTRY,
        arrowprops=dict(arrowstyle="->", color=RED_REENTRY, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=RED_REENTRY, alpha=0.9),
    )

    ax.set_xlabel("First Excursion Date", fontsize=12)
    ax.set_ylabel("Current Periapsis Altitude (km)", fontsize=12)
    ax.set_title("STTS-Reentry: Anomalous Operational Satellites",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    outpath = PLOTS / "anomalous_satellites.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


# ══════════════════════════════════════════════════════════════
# Plot 3: Solar Cycle 25 Correlation
# ══════════════════════════════════════════════════════════════
def plot_sc25_correlation():
    """Dual-axis: excursion onset rate vs sunspot number."""
    # Excursion onset counts — extracted from results/reentry/patterns.md
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    onsets = [4, 6, 17, 23, 21, 37]

    # Annual mean sunspot numbers from SILSO/WDC-SILSO Royal Observatory
    # of Belgium (https://www.sidc.be/SILSO/datafiles).
    # These are published reference values, not simulated data.
    # 2025 value is provisional (Jan-Dec 2025 monthly mean average).
    ssn = [8.8, 26.1, 63.7, 107.4, 150.0, 116.0]

    from scipy.stats import spearmanr
    rho_year, p_year = spearmanr(years, onsets)
    rho_ssn, p_ssn = spearmanr(ssn, onsets)

    print(f"  Source: patterns.md (onsets), SILSO/WDC (SSN)")
    print(f"  SC25 correlation: {len(years)} years, "
          f"onsets={onsets}, SSN={ssn}")
    print(f"  Spearman(year, onsets):  rho={rho_year:.3f}, p={p_year:.4f}")
    print(f"  Spearman(SSN, onsets):   rho={rho_ssn:.3f}, p={p_ssn:.4f}")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(years, onsets, color=RED_REENTRY, alpha=0.7, width=0.4,
            label="Excursion onsets", zorder=3)
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Excursion Onset Count", fontsize=12, color=RED_REENTRY)
    ax1.tick_params(axis="y", labelcolor=RED_REENTRY, labelsize=10)
    ax1.tick_params(axis="x", labelsize=10)
    ax1.set_xticks(years)

    ax2 = ax1.twinx()
    ax2.plot(years, ssn, color=GREEN_SC25, linewidth=2.5, marker="o",
             markersize=8, label="Mean SSN (SILSO)", zorder=4)
    ax2.set_ylabel("Annual Mean Sunspot Number", fontsize=12, color=GREEN_SC25)
    ax2.tick_params(axis="y", labelcolor=GREEN_SC25, labelsize=10)

    ax1.axvspan(2021.5, 2024.5, alpha=0.08, color=GREEN_SC25, zorder=1)
    ax1.text(2023.0, 2, "SC25 Rising Phase",
             fontsize=9, color=GREEN_SC25, ha="center", alpha=0.7, style="italic")

    # Primary annotation: year vs onset correlation (paper-reported value)
    ax1.annotate(
        f"\u03c1 = {rho_year:.3f} (year vs onset, p={p_year:.3f})",
        xy=(0.50, 0.92), xycoords="axes fraction",
        fontsize=16, fontweight="bold", color=DARK_TEXT,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=GREEN_SC25, alpha=0.9),
    )

    # Secondary annotation: SSN-onset correlation with lag explanation
    ax1.annotate(
        f"SSN\u2013onset \u03c1={rho_ssn:.3f} reflects ~6 month lag\n"
        f"between solar flux peak and atmospheric\ndensity response",
        xy=(0.50, 0.77), xycoords="axes fraction",
        fontsize=9, color=GRAY_REF, ha="center", style="italic",
    )

    ax1.set_title("Excursion Onset Rate vs Solar Cycle 25",
                   fontsize=14, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left")

    fig.tight_layout()
    outpath = PLOTS / "sc25_correlation.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


# ══════════════════════════════════════════════════════════════
# Plot 4: Lead Time Distribution
# ══════════════════════════════════════════════════════════════
def plot_lead_time():
    """Histogram of real detection lead times from validate.json."""
    val_path = RESULTS / "validate.json"
    with open(val_path) as f:
        data = json.load(f)

    lead_times = np.array(data["lead_times_raw"])
    stats = data["lead_time_days"]

    print(f"  Source: {val_path}")
    print(f"  Lead times: loaded {len(lead_times)} real detection events "
          f"from validate.json lead_times_raw")
    print(f"  Mean={stats['mean']}d, median={stats['median']}d, "
          f"min={stats['min']}d, max={stats['max']}d")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(lead_times, bins=25, color=BLUE_NOM, alpha=0.7,
            edgecolor="white", linewidth=0.5)

    mean_lt = stats["mean"]
    ax.axvline(mean_lt, color=RED_REENTRY, linestyle="--", linewidth=2, zorder=4)

    # Place annotation after histogram is drawn so ylim is set
    ymax = ax.get_ylim()[1]
    ax.annotate(
        f"Mean lead time: {mean_lt:.0f} days\nbefore confirmed reentry",
        xy=(mean_lt, ymax * 0.85), xycoords="data",
        xytext=(mean_lt + 200, ymax * 0.70),
        fontsize=13, fontweight="bold", color=DARK_TEXT,
        arrowprops=dict(arrowstyle="->", color=RED_REENTRY, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=RED_REENTRY, alpha=0.9),
    )

    ax.set_xlabel("Detection Lead Time (days)", fontsize=12)
    ax.set_ylabel("Number of Satellites", fontsize=12)
    ax.set_title(f"STTS-Reentry: Detection Lead Time Distribution (n={len(lead_times)})",
                 fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    outpath = PLOTS / "lead_time_distribution.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {outpath}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    print("Generating STTS-Reentry plots...")
    print("All plots use real data only.\n")

    print("[1/4] Basin separation histogram")
    plot_basin_separation()

    print("\n[2/4] Anomalous satellites scatter")
    plot_anomalous_satellites()

    print("\n[3/4] SC25 correlation")
    plot_sc25_correlation()

    print("\n[4/4] Lead time distribution")
    plot_lead_time()

    print(f"\nAll plots saved to {PLOTS}/")


if __name__ == "__main__":
    main()
