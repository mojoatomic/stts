"""
Synthetic absorbing-Markov-chain construction and sampling utilities for
the Flux Accumulator validation session.

Two modes:
  - Deterministic per-cell rewards (Test 1):  cumulative reward = Σ κ_{c(v)}
  - Heteroscedastic per-cell rewards (Test 2): cumulative reward = Σ Φ_v
                                                where Φ_v ~ N(κ_c, σ_c²) truncated at 0

The transition matrix Q (8×8 substochastic) and the per-cell mean reward
vector κ are loaded from the n=39 reentry C-restricted artifact
`results/reentry/flux_characterization_c_restricted/predicted_variance.json`
(commit 1bd283c). This is the canonical source for these synthetic tests.

Read-only against the loaded artifact. Reuses the canonical
`kemeny_snell_variance` from `reentry/flux_accumulator.py` so the formula
under test is the exact one used in the reentry pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


SOURCE_PATH = Path(
    "results/reentry/flux_characterization_c_restricted/predicted_variance.json"
)


# ── Load chain artifacts ────────────────────────────────────────

def load_chain() -> dict:
    """Load Q, κ, surviving cells, and the committed Kemeny-Snell outputs.

    Returns dict with keys: cells (list[int]), Q (n×n ndarray), kappa
    (n,), N_committed, v_committed, Var_committed (for byte-identity
    cross-check). All arrays float64.
    """
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {SOURCE_PATH}. "
            f"This file is produced by commit 1bd283c."
        )
    d = json.loads(SOURCE_PATH.read_text())
    Q = np.array(d["Q"], dtype=np.float64)
    kappa = np.array(d["kappa"], dtype=np.float64)
    cells = list(d["surviving_cells"])
    if Q.shape != (len(cells), len(cells)):
        raise ValueError(
            f"Q shape {Q.shape} does not match surviving cells {len(cells)}"
        )
    if kappa.shape != (len(cells),):
        raise ValueError(
            f"kappa shape {kappa.shape} does not match surviving cells {len(cells)}"
        )
    return {
        "cells": cells,
        "Q": Q,
        "kappa": kappa,
        "N_committed": np.array(d["N"], dtype=np.float64),
        "v_committed": np.array(d["v"], dtype=np.float64),
        "Var_committed": np.array(d["Var_predicted"], dtype=np.float64),
        "alpha_used_at_calibration": float(d["alpha"]),
    }


# ── Trajectory sampling ─────────────────────────────────────────

def _build_cum_table(Q: np.ndarray) -> np.ndarray:
    """Augment the substochastic Q with an absorbing column.

    Returns a row-stochastic matrix of shape (n_transient, n_transient + 1)
    where column n_transient represents the absorbing state, and row sums
    are 1 by construction.
    """
    n = Q.shape[0]
    P_ext = np.zeros((n, n + 1), dtype=np.float64)
    P_ext[:, :n] = Q
    P_ext[:, n] = 1.0 - Q.sum(axis=1)
    # Numerical clean-up: ensure non-negativity
    P_ext = np.clip(P_ext, 0.0, 1.0)
    P_ext /= P_ext.sum(axis=1, keepdims=True)
    return P_ext


def sample_trajectories(Q: np.ndarray,
                         start_cells: np.ndarray,
                         seed: int = 20260430,
                         max_steps: int = 100_000) -> dict:
    """Sample one trajectory per entry in `start_cells`. Returns a dict:

      - sigma_visit_counts:   (n_traj, n_transient) int  — visits to each
                              transient state per trajectory
      - lengths:              (n_traj,) int  — number of transient visits
                              before absorption (= trajectory length)
      - absorbed:             (n_traj,) bool — all True in normal use
      - paths:                None (not stored — too large for n=100k)

    Vectorised over trajectories: at each step, alive trajectories are
    advanced together via Q-row-indexed CDF inversion.
    """
    n_t = Q.shape[0]
    n_traj = len(start_cells)
    P_ext = _build_cum_table(Q)
    cum = np.cumsum(P_ext, axis=1)            # (n_t, n_t+1)
    cum[:, -1] = 1.0                          # numerical guard

    rng = np.random.default_rng(seed)
    states = np.asarray(start_cells, dtype=np.int64).copy()
    alive = np.ones(n_traj, dtype=bool)
    visit_counts = np.zeros((n_traj, n_t), dtype=np.int64)
    lengths = np.zeros(n_traj, dtype=np.int64)
    # Reward κ accrues at the state on entry; the starting state counts
    # as visit #1.
    np.add.at(visit_counts, (np.arange(n_traj), states), 1)
    lengths += 1

    for step in range(1, max_steps):
        if not alive.any():
            break
        idx_alive = np.where(alive)[0]
        cur = states[idx_alive]
        u = rng.random(len(idx_alive))
        # For each alive trajectory, find first j such that cum[cur, j] >= u
        cdf = cum[cur]                          # (n_alive, n_t + 1)
        # Use searchsorted within each row
        next_st = (cdf < u[:, None]).sum(axis=1)
        absorbed_now = next_st == n_t
        alive_after = ~absorbed_now
        # Update non-absorbed
        survivors = idx_alive[alive_after]
        new_states = next_st[alive_after]
        states[survivors] = new_states
        np.add.at(visit_counts, (survivors, new_states), 1)
        lengths[survivors] += 1
        # Mark absorbed
        alive[idx_alive[absorbed_now]] = False

    absorbed = ~alive
    return {
        "visit_counts": visit_counts,
        "lengths": lengths,
        "absorbed": absorbed,
        "n_traj": n_traj,
        "n_transient": n_t,
        "max_steps_reached": int(lengths.max()),
        "max_steps_limit": max_steps,
    }


def deterministic_sigma(visit_counts: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """σ_failure = Σ_c visit_counts[i, c] · κ_c  (per trajectory).

    Equivalent to the path-dependent reward sum when reward at state c is
    deterministic and equal to κ_c, paid on entry.
    """
    return visit_counts.astype(np.float64) @ kappa


def heteroscedastic_sigma(visit_counts: np.ndarray, kappa: np.ndarray,
                           sigma_c: np.ndarray, seed: int,
                           truncate: bool = True) -> np.ndarray:
    """σ_failure with each visit's reward drawn independently from a
    truncated-normal Φ ~ N(κ_c, σ_c²) (clipped at 0).

    Vectorised: for each cell c with total visits N_c across all
    trajectories, draw N_c iid samples from N(κ_c, σ_c²), distribute back
    to the per-trajectory contributions in arrival order. The order of
    draws within a trajectory does not affect the sum, so a simple
    cell-batched draw is sufficient and exactly equivalent to per-visit
    sampling.

    Returns σ_failure per trajectory (n_traj,).
    """
    rng = np.random.default_rng(seed)
    n_traj, n_t = visit_counts.shape
    total = np.zeros(n_traj, dtype=np.float64)
    for c in range(n_t):
        if sigma_c[c] <= 0:
            total += visit_counts[:, c] * kappa[c]
            continue
        N_c = int(visit_counts[:, c].sum())
        if N_c == 0:
            continue
        draws = rng.normal(loc=kappa[c], scale=sigma_c[c], size=N_c)
        if truncate:
            # Truncate at 0 (single rejection pass + clip). For
            # high σ_c/κ_c, this systematically inflates the mean
            # (E[N(κ,σ²) | X ≥ 0] > κ) and reduces the variance below
            # σ_c² (clipped tail). The analytical correction term
            # uses untruncated σ_c², so a like-for-like comparison
            # requires `truncate=False`. The brief specifies truncation
            # to preserve non-negativity; we report both modes so the
            # truncation contribution is separable from any formula
            # mismatch.
            neg = draws < 0
            if neg.any():
                draws[neg] = rng.normal(loc=kappa[c], scale=sigma_c[c], size=int(neg.sum()))
                draws = np.clip(draws, 0.0, None)
        # Distribute draws to trajectories in proportion to visit counts.
        # Cumulative-sum trick: positions allocated to traj i are
        # [start_i, start_i + visits_i).
        starts = np.zeros(n_traj + 1, dtype=np.int64)
        starts[1:] = np.cumsum(visit_counts[:, c])
        for i in range(n_traj):
            if visit_counts[i, c] == 0:
                continue
            total[i] += draws[starts[i]:starts[i + 1]].sum()
    return total


# ── Empirical per-starting-cell statistics ─────────────────────

def per_start_cell_stats(start_cells: np.ndarray,
                          sigma_failure: np.ndarray,
                          n_transient: int) -> dict:
    """Mean and ddof=1 variance of σ_failure stratified by starting cell."""
    out = {}
    for c in range(n_transient):
        mask = start_cells == c
        n = int(mask.sum())
        if n < 2:
            out[c] = {"n": n, "mean": float("nan"), "var": float("nan")}
            continue
        out[c] = {
            "n": n,
            "mean": float(sigma_failure[mask].mean()),
            "var": float(sigma_failure[mask].var(ddof=1)),
        }
    return out


# ── Bootstrap CI for a ratio of variances ──────────────────────

def bootstrap_var_ratio_ci(sigma_failure: np.ndarray,
                            var_pred: float,
                            n_replicates: int = 10_000,
                            seed: int = 20260501,
                            alpha: float = 0.05) -> tuple[float, float, float]:
    """Bootstrap CI for r = var_pred / Var_emp by resampling
    sigma_failure with replacement and recomputing Var_emp.

    Returns (point, ci_lo, ci_hi).
    """
    sigma_failure = np.asarray(sigma_failure, dtype=np.float64)
    sigma_failure = sigma_failure[np.isfinite(sigma_failure)]
    n = len(sigma_failure)
    if n < 4 or var_pred <= 0:
        return float("nan"), float("nan"), float("nan")
    point = float(var_pred / np.var(sigma_failure, ddof=1))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_replicates, n))
    samples = sigma_failure[idx]
    repl_var = np.var(samples, axis=1, ddof=1)
    repl_r = var_pred / repl_var
    return point, float(np.quantile(repl_r, alpha / 2)), float(np.quantile(repl_r, 1 - alpha / 2))
