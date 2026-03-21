"""Visualization of STTS pipeline results."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

from pipeline.config import RESULTS_DIR


def _save(fig, name: str):
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(RESULTS_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_2d(
    embeddings: np.ndarray,
    rul: np.ndarray,
    basin_mask: np.ndarray,
    title: str = "Trajectory Embeddings",
    filename: str = "embedding_2d",
):
    """2D scatter of embeddings colored by RUL, failure basin highlighted."""

    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
    else:
        coords = embeddings

    fig, ax = plt.subplots(figsize=(10, 8))

    # Nominal trajectories
    nominal = ~basin_mask
    sc = ax.scatter(
        coords[nominal, 0], coords[nominal, 1],
        c=rul[nominal], cmap="viridis", s=3, alpha=0.5, label="Nominal"
    )
    plt.colorbar(sc, ax=ax, label="RUL (cycles)")

    # Failure basin
    ax.scatter(
        coords[basin_mask, 0], coords[basin_mask, 1],
        c="red", s=5, alpha=0.3, label=f"Failure basin (B_f)"
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend()
    _save(fig, filename)


def plot_distance_curve(
    distances: np.ndarray,
    cycles: np.ndarray,
    true_rul_at_end: int,
    epsilon: float,
    engine_id: int,
    stts_cycle: Optional[int] = None,
    threshold_cycle: Optional[int] = None,
    filename: Optional[str] = None,
):
    """Distance to B_f over time for a single test engine."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(cycles, distances, "b-", linewidth=1, label="d(φ(T), B_f)")
    ax.axhline(y=epsilon, color="r", linestyle="--", alpha=0.7, label=f"ε = {epsilon:.2f}")

    if stts_cycle is not None:
        ax.axvline(x=cycles[stts_cycle], color="green", linestyle="-",
                   alpha=0.7, label=f"STTS alert (cycle {cycles[stts_cycle]})")
    if threshold_cycle is not None:
        ax.axvline(x=cycles[threshold_cycle], color="orange", linestyle="-",
                   alpha=0.7, label=f"Threshold alert (cycle {cycles[threshold_cycle]})")

    # Mark estimated failure point
    failure_cycle = cycles[-1] + true_rul_at_end
    ax.axvline(x=failure_cycle, color="black", linestyle=":", alpha=0.5,
               label=f"Est. failure (cycle {failure_cycle})")

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Distance to failure basin")
    ax.set_title(f"Engine {engine_id} — STTS Monitoring Query")
    ax.legend(fontsize=8)
    ax.set_xlim(cycles[0], max(cycles[-1], failure_cycle) + 5)
    _save(fig, filename or f"distance_engine_{engine_id}")


def plot_intervention_windows(
    results: list[dict],
    filename: str = "intervention_windows",
):
    """Box plot of intervention window sizes across test engines."""
    windows = [r["window_recovered"] for r in results if r["stts_fired"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(windows, vert=True)
    ax.set_ylabel("Intervention window (cycles)")
    ax.set_title(f"STTS Intervention Window (n={len(windows)} engines)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    _save(fig, filename)


def plot_verification_v2(
    distances: np.ndarray,
    rul: np.ndarray,
    rho: float,
    filename: str = "verification_v2",
):
    """Scatter of distance vs RUL — should show positive correlation."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(rul, distances, s=2, alpha=0.3)
    ax.set_xlabel("RUL (cycles)")
    ax.set_ylabel("Distance to B_f")
    ax.set_title(f"V2 Verification: Distance vs RUL (Spearman ρ = {rho:.3f})")
    _save(fig, filename)


def plot_feature_ablation(
    ablation_results: dict[str, float],
    filename: str = "verification_v3",
):
    """Bar chart of V3 ablation — which feature class drives basin proximity."""
    classes = list(ablation_results.keys())
    impacts = [ablation_results[c] * 100 for c in classes]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in impacts]
    ax.barh(classes, impacts, color=colors)
    ax.set_xlabel("Relative distance change when ablated (%)")
    ax.set_title("V3 Verification: Feature Class Importance")
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    _save(fig, filename)


def plot_precision_recall(
    pr_results: dict,
    filename: str = "precision_recall",
):
    """Precision-recall curve with F1-optimal point marked."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # PR curve
    ax1.plot(pr_results["recall"], pr_results["precision"], "b-", linewidth=1.5)
    ax1.scatter(
        [pr_results["best_recall"]], [pr_results["best_precision"]],
        c="red", s=100, zorder=5,
        label=f"Best F1={pr_results['best_f1']:.2f}\n"
              f"P={pr_results['best_precision']:.2f}, R={pr_results['best_recall']:.2f}\n"
              f"ε={pr_results['best_epsilon']:.2f}",
    )
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title(f"Precision-Recall (n+={pr_results['n_positive']}, n-={pr_results['n_negative']})")
    ax1.legend(fontsize=9)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    # F1 vs epsilon
    ax2.plot(pr_results["epsilons"], pr_results["f1"], "g-", linewidth=1.5)
    ax2.axvline(x=pr_results["best_epsilon"], color="red", linestyle="--",
                label=f"Best ε={pr_results['best_epsilon']:.2f}")
    ax2.set_xlabel("ε (distance threshold)")
    ax2.set_ylabel("F1 score")
    ax2.set_title("F1 vs ε")
    ax2.legend()

    fig.tight_layout()
    _save(fig, filename)
