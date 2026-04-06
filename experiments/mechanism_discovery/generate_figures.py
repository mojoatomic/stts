"""Generate figures for the Geometric Mechanism Discovery Experiment.

Run: python experiments/mechanism_discovery/generate_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXP_DIR = PROJECT_ROOT / "experiments" / "mechanism_discovery"
FIG_DIR = EXP_DIR / "figures"

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MECHANISM_KEYS = [
    ("m1_participation_ratio", "M1: Participation Ratio"),
    ("m2_centroid_distance", "M2: Centroid Drift"),
    ("m3_trajectory_velocity", "M3: Trajectory Velocity"),
    ("m4_mean_variance", "M4: Distribution Contraction"),
    ("m5_mean_abs_correlation", "M5: Correlation Tightening"),
    ("m6_subspace_angle", "M6: Subspace Rotation"),
    ("m7_lda_fluctuation", "M7: Critical Fluctuations"),
]

MECH_SHORT = ["M1: PR", "M2: Drift", "M3: Vel", "M4: Var",
              "M5: Corr", "M6: Rot", "M7: Fluct"]

ENGINE_COLORS = plt.cm.tab10.colors


def load_trajectories(domain: str) -> dict:
    """Load per-engine trajectories from npz."""
    data = np.load(EXP_DIR / "trajectories.npz", allow_pickle=True)
    engines = {}
    prefix = f"{domain}__"
    for key in data.files:
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].split("__", 1)
        if len(parts) != 2:
            continue
        eid, field = parts
        if eid not in engines:
            engines[eid] = {}
        engines[eid][field] = data[key]
    return engines


def load_results():
    with open(EXP_DIR / "mechanism_discovery.json") as f:
        return json.load(f)


# -----------------------------------------------------------------------
# Figure 1: Mechanism vs RUL (DS03 dev)
# -----------------------------------------------------------------------

def fig1_mechanism_vs_rul(domain="ncmapss_ds03"):
    engines = load_trajectories(domain)
    dev_engines = {k: v for k, v in engines.items() if k.startswith("dev_")}

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for idx, (mech_key, mech_name) in enumerate(MECHANISM_KEYS):
        ax = axes[idx]
        for i, (eid, eng) in enumerate(sorted(dev_engines.items())):
            rul = eng["rul"]
            vals = eng[mech_key]
            valid = np.isfinite(vals)
            color = ENGINE_COLORS[i % len(ENGINE_COLORS)]
            label = eid.replace("dev_", "U")
            ax.plot(rul[valid], vals[valid], color=color, alpha=0.7,
                    linewidth=1.2, label=label)
        ax.set_xlabel("RUL")
        ax.set_ylabel(mech_name.split(": ")[1])
        ax.set_title(mech_name)
        ax.invert_xaxis()

    # Hide last subplot, put legend there
    axes[7].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[7].legend(handles, labels, loc="center", fontsize=10, ncol=2,
                   title="Dev Engines")

    fig.suptitle(f"Mechanism Metrics vs RUL — {domain}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"mechanism_vs_rul_{domain.split('_')[-1]}.png")
    plt.close(fig)
    print(f"   Saved mechanism_vs_rul_{domain.split('_')[-1]}.png")


# -----------------------------------------------------------------------
# Figure 2: Mechanism vs LDA (DS03 dev)
# -----------------------------------------------------------------------

def fig2_mechanism_vs_lda(domain="ncmapss_ds03"):
    engines = load_trajectories(domain)
    dev_engines = {k: v for k, v in engines.items() if k.startswith("dev_")}

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for idx, (mech_key, mech_name) in enumerate(MECHANISM_KEYS):
        ax = axes[idx]
        for i, (eid, eng) in enumerate(sorted(dev_engines.items())):
            lda = eng["lda_score"]
            vals = eng[mech_key]
            valid = np.isfinite(vals)
            color = ENGINE_COLORS[i % len(ENGINE_COLORS)]
            ax.scatter(vals[valid], lda[valid], color=color, alpha=0.4, s=8)
        ax.set_xlabel(mech_name.split(": ")[1])
        ax.set_ylabel("LDA Score")
        ax.set_title(mech_name)

    axes[7].axis("off")

    fig.suptitle(f"Mechanism Metrics vs LDA Score — {domain}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"mechanism_vs_lda_{domain.split('_')[-1]}.png")
    plt.close(fig)
    print(f"   Saved mechanism_vs_lda_{domain.split('_')[-1]}.png")


# -----------------------------------------------------------------------
# Figure 3: Unit 10 mechanisms
# -----------------------------------------------------------------------

def fig3_unit10():
    engines = load_trajectories("ncmapss_ds03")
    test_engines = {k: v for k, v in engines.items() if k.startswith("test_")}

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for idx, (mech_key, mech_name) in enumerate(MECHANISM_KEYS):
        ax = axes[idx]
        for eid, eng in sorted(test_engines.items()):
            rul = eng["rul"]
            vals = eng[mech_key]
            valid = np.isfinite(vals)
            is_u10 = "10" in eid
            style = {"color": "#b2182b", "linewidth": 2.5, "linestyle": "--",
                     "zorder": 10, "label": "Unit 10"} if is_u10 else {
                "color": "gray", "linewidth": 1.0, "alpha": 0.5,
                "label": eid.replace("test_", "U")}
            ax.plot(rul[valid], vals[valid], **style)
        ax.set_xlabel("RUL")
        ax.set_ylabel(mech_name.split(": ")[1])
        ax.set_title(mech_name)
        ax.invert_xaxis()

    axes[7].axis("off")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#b2182b", linewidth=2.5, linestyle="--",
               label="Unit 10"),
        Line2D([0], [0], color="gray", linewidth=1.0, alpha=0.5,
               label="Other test engines"),
    ]
    axes[7].legend(handles=legend_elements, loc="center", fontsize=12)

    fig.suptitle("Unit 10 vs Other Test Engines — All Mechanisms", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mechanism_unit10.png")
    plt.close(fig)
    print("   Saved mechanism_unit10.png")


# -----------------------------------------------------------------------
# Figure 4: Eigenvalue spectrum early vs late
# -----------------------------------------------------------------------

def fig4_eigenvalue_spectrum():
    engines = load_trajectories("ncmapss_ds03")
    # Pick a representative dev engine (longest trajectory)
    dev_engines = {k: v for k, v in engines.items() if k.startswith("dev_")}
    rep_eid = max(dev_engines, key=lambda k: len(dev_engines[k]["rul"]))
    eng = dev_engines[rep_eid]

    # We need raw sensor windows — reload from data
    from pipeline.ncmapss.data_loader import load_dataset, normalize_by_regime, get_engine_data
    from pipeline.ncmapss.config import ACTIVE_SENSORS, WINDOW_SIZE

    dev_df, _ = load_dataset("DS03", rul_clip=None)
    dev_df, _, _ = normalize_by_regime(dev_df, dev_df.copy())
    dev_engs = get_engine_data(dev_df)
    cols = [c for c in ACTIVE_SENSORS if c in dev_df.columns]

    uid = int(rep_eid.replace("dev_", ""))
    sensor_data = dev_engs[uid][cols].values

    # Early life window (first)
    early_win = sensor_data[:WINDOW_SIZE]
    early_cov = np.cov(early_win, rowvar=False)
    early_eig = np.sort(np.linalg.eigvalsh(early_cov))[::-1]
    early_eig = early_eig[early_eig > 0]

    # End of life window (last)
    late_win = sensor_data[-WINDOW_SIZE:]
    late_cov = np.cov(late_win, rowvar=False)
    late_eig = np.sort(np.linalg.eigvalsh(late_cov))[::-1]
    late_eig = late_eig[late_eig > 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(range(1, len(early_eig) + 1), early_eig, "o-",
                color="#2166ac", label=f"Early life (RUL={int(eng['rul'][0])})")
    ax.semilogy(range(1, len(late_eig) + 1), late_eig, "s-",
                color="#b2182b", label=f"End of life (RUL={int(eng['rul'][-1])})")
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title(f"Eigenvalue Spectrum — DS03 Unit {uid}")
    ax.legend()
    fig.savefig(FIG_DIR / "eigenvalue_spectrum_ds03.png")
    plt.close(fig)
    print("   Saved eigenvalue_spectrum_ds03.png")


# -----------------------------------------------------------------------
# Figure 5: Mechanism ranking bar chart — ALL DOMAINS side by side
# -----------------------------------------------------------------------

def fig5_mechanism_ranking():
    results = load_results()
    tables = results["correlation_tables"]

    domains = list(tables.keys())
    domain_labels = {
        "ncmapss_ds03": "N-CMAPSS DS03",
        "ncmapss_ds02": "N-CMAPSS DS02",
        "cmapss_fd001": "C-MAPSS FD001",
        "battery": "Battery",
        "bearing": "Bearing",
    }

    n_domains = len(domains)
    fig, axes = plt.subplots(1, n_domains, figsize=(4.5 * n_domains, 6), sharey=True)
    if n_domains == 1:
        axes = [axes]

    for ax_idx, domain in enumerate(domains):
        ax = axes[ax_idx]
        rho_vals = []
        r_vals = []
        names = []

        for mech_key, mech_name in MECHANISM_KEYS:
            rho = tables[domain]["rho_vs_rul"].get(mech_key, {})
            r = tables[domain]["r_vs_lda"].get(mech_key, {})
            rho_mean = abs(rho["mean"]) if rho.get("mean") is not None else 0
            r_mean = abs(r["mean"]) if r.get("mean") is not None else 0
            rho_vals.append(rho_mean)
            r_vals.append(r_mean)
            names.append(mech_name.split(": ")[1])

        y = np.arange(len(names))
        h = 0.35

        ax.barh(y + h/2, rho_vals, h, label="|ρ| vs RUL", color="#2166ac")
        ax.barh(y - h/2, r_vals, h, label="|r| vs LDA", color="#b2182b")
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel("Correlation Magnitude")
        ax.set_title(domain_labels.get(domain, domain))
        ax.set_xlim(0, 1.05)
        if ax_idx == 0:
            ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Mechanism Ranking Across Domains", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mechanism_ranking.png")
    plt.close(fig)
    print("   Saved mechanism_ranking.png")


# -----------------------------------------------------------------------
# Figure 6: Collinearity heatmap
# -----------------------------------------------------------------------

def fig6_collinearity():
    results = load_results()

    domains = list(results["collinearity"].keys())
    n = len(domains)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    domain_labels = {
        "ncmapss_ds03": "N-CMAPSS DS03",
        "ncmapss_ds02": "N-CMAPSS DS02",
        "cmapss_fd001": "C-MAPSS FD001",
        "battery": "Battery",
        "bearing": "Bearing",
    }

    for ax_idx, domain in enumerate(domains):
        ax = axes[ax_idx]
        cmat = np.array(results["collinearity"][domain])

        im = ax.imshow(cmat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
        ax.set_xticks(range(7))
        ax.set_xticklabels(MECH_SHORT, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(7))
        ax.set_yticklabels(MECH_SHORT, fontsize=8)
        ax.set_title(domain_labels.get(domain, domain), fontsize=11)

        for i in range(7):
            for j in range(7):
                val = cmat[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    fig.colorbar(im, ax=axes, shrink=0.8, label="Pearson r")
    fig.suptitle("Inter-Mechanism Collinearity", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mechanism_collinearity.png")
    plt.close(fig)
    print("   Saved mechanism_collinearity.png")


# -----------------------------------------------------------------------
# Figure 7: Critical fluctuations (M7)
# -----------------------------------------------------------------------

def fig7_critical_fluctuations():
    engines = load_trajectories("ncmapss_ds03")
    dev_engines = {k: v for k, v in engines.items() if k.startswith("dev_")}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, (eid, eng) in enumerate(sorted(dev_engines.items())):
        rul = eng["rul"]
        fluct = eng["m7_lda_fluctuation"]
        ac = eng["m7_autocorr_lag1"]
        valid_f = np.isfinite(fluct)
        valid_a = np.isfinite(ac)
        color = ENGINE_COLORS[i % len(ENGINE_COLORS)]
        label = eid.replace("dev_", "U")

        ax1.plot(rul[valid_f], fluct[valid_f], color=color, alpha=0.7,
                 linewidth=1.2, label=label)
        ax2.plot(rul[valid_a], ac[valid_a], color=color, alpha=0.7,
                 linewidth=1.2, label=label)

    ax1.set_xlabel("RUL")
    ax1.set_ylabel("LDA Score Variance")
    ax1.set_title("M7: LDA Fluctuation vs RUL")
    ax1.invert_xaxis()
    ax1.legend(fontsize=8, ncol=2)

    ax2.set_xlabel("RUL")
    ax2.set_ylabel("Lag-1 Autocorrelation")
    ax2.set_title("M7: Autocorrelation vs RUL")
    ax2.invert_xaxis()

    fig.suptitle("Critical Fluctuations — DS03 Dev Engines", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "critical_fluctuations_ds03.png")
    plt.close(fig)
    print("   Saved critical_fluctuations_ds03.png")


# -----------------------------------------------------------------------
# Figure 8: Scaling power law
# -----------------------------------------------------------------------

def fig8_scaling_powerlaw():
    engines = load_trajectories("ncmapss_ds03")
    dev_engines = {k: v for k, v in engines.items() if k.startswith("dev_")}

    # Pick a representative engine
    rep_eid = max(dev_engines, key=lambda k: len(dev_engines[k]["rul"]))
    eng = dev_engines[rep_eid]
    uid = rep_eid.replace("dev_", "")

    rul = eng["rul"]
    pr = eng["m1_participation_ratio"]
    valid = np.isfinite(pr) & (rul > 0) & (pr > 0)
    r, p = rul[valid].astype(float), pr[valid].astype(float)

    # Fit all three models
    log_r, log_p = np.log(r), np.log(p)

    # Power law
    alpha, log_c = np.polyfit(log_r, log_p, 1)
    pred_power = np.exp(log_c) * r ** alpha

    # Linear
    a_lin, b_lin = np.polyfit(r, p, 1)
    pred_lin = a_lin * r + b_lin

    # Exponential
    b_exp, log_a_exp = np.polyfit(r, log_p, 1)
    pred_exp = np.exp(log_a_exp) * np.exp(b_exp * r)

    # R² in original space
    def r2(y, yhat):
        ss_res = np.sum((y - yhat)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - ss_res / max(ss_tot, 1e-15)

    r2_pow = r2(p, pred_power)
    r2_lin = r2(p, pred_lin)
    r2_exp_val = r2(p, pred_exp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Log-log plot
    sort_idx = np.argsort(r)
    ax1.scatter(r, p, c="#999999", s=10, alpha=0.5, label="Data")
    ax1.plot(r[sort_idx], pred_power[sort_idx], "-", color="#2166ac",
             linewidth=2, label=f"Power: α={alpha:.3f}, R²={r2_pow:.3f}")
    ax1.plot(r[sort_idx], pred_lin[sort_idx], "--", color="#b2182b",
             linewidth=2, label=f"Linear: R²={r2_lin:.3f}")
    ax1.plot(r[sort_idx], pred_exp[sort_idx], ":", color="#1b7837",
             linewidth=2, label=f"Exp: R²={r2_exp_val:.3f}")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("RUL (log)")
    ax1.set_ylabel("Participation Ratio (log)")
    ax1.set_title(f"Log-Log: PR vs RUL — DS03 Unit {uid}")
    ax1.legend(fontsize=9)

    # Original space
    ax2.scatter(r, p, c="#999999", s=10, alpha=0.5, label="Data")
    ax2.plot(r[sort_idx], pred_power[sort_idx], "-", color="#2166ac",
             linewidth=2, label=f"Power: R²={r2_pow:.3f}")
    ax2.plot(r[sort_idx], pred_lin[sort_idx], "--", color="#b2182b",
             linewidth=2, label=f"Linear: R²={r2_lin:.3f}")
    ax2.plot(r[sort_idx], pred_exp[sort_idx], ":", color="#1b7837",
             linewidth=2, label=f"Exp: R²={r2_exp_val:.3f}")
    ax2.set_xlabel("RUL")
    ax2.set_ylabel("Participation Ratio")
    ax2.set_title(f"Original Space: PR vs RUL — DS03 Unit {uid}")
    ax2.legend(fontsize=9)

    fig.suptitle("Scaling Test — Competing Fits", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scaling_powerlaw_ds03.png")
    plt.close(fig)
    print("   Saved scaling_powerlaw_ds03.png")


# -----------------------------------------------------------------------

def main():
    print("Generating figures...")
    print()

    fig1_mechanism_vs_rul("ncmapss_ds03")
    fig2_mechanism_vs_lda("ncmapss_ds03")
    fig3_unit10()
    fig4_eigenvalue_spectrum()
    fig5_mechanism_ranking()
    fig6_collinearity()
    fig7_critical_fluctuations()
    fig8_scaling_powerlaw()

    print()
    print("Done. All figures in experiments/mechanism_discovery/figures/")


if __name__ == "__main__":
    main()
