"""
Exact hitting-time and endpoint CDFs for a discrete-time Markov chain.

The reentry extension's forward failure probability is a function of a
20x20 transition matrix P and a failure-cell set F. For this size,
exact computation via matrix powers is microseconds; Monte Carlo is
reserved for diagnostic cross-check.

Two quantities:
  P_end[x, Dt] = P(X_Dt in F | X_0 = x)
               = [P^Dt . 1_F]_x   where 1_F[c] = 1 if c in F else 0
  P_hit[x, Dt] = P(tau_F <= Dt | X_0 = x)   (first-passage CDF)
               = 1 for x in F (already absorbed)
               = 1 - [Q^Dt . 1]_x for x not in F, where Q is P
                 restricted to non-failure rows and columns (sub-stochastic;
                 row sums = 1 - P(exit to F)).

Inequality (checked): P_hit[x, Dt] >= P_end[x, Dt] for all x, Dt. Strict
whenever the basin is not perfectly absorbing.
"""

from __future__ import annotations

import numpy as np


def exact_endpoint_cdf(P: np.ndarray, failure_cells, horizons) -> np.ndarray:
    """P_end[x, Dt] = P(X_Dt in F | X_0 = x), computed exactly.

    Parameters
    ----------
    P : (K, K) ndarray, row-stochastic.
    failure_cells : iterable of int, cell indices in F.
    horizons : sequence of positive int, values of Dt.

    Returns
    -------
    (K, len(horizons)) ndarray, P_end[x, i] = P(X_{horizons[i]} in F | X_0 = x).
    """
    P = np.asarray(P, dtype=np.float64)
    K = P.shape[0]
    if P.shape != (K, K):
        raise ValueError(f"P must be square, got shape {P.shape}")
    horizons = list(horizons)
    if any(h < 1 for h in horizons):
        raise ValueError("horizons must be >= 1")
    indic = np.zeros(K, dtype=np.float64)
    indic[list(failure_cells)] = 1.0

    # Iteratively apply P: v_t = P . v_{t-1}, starting from indic = 1_F.
    # Then P_end[x, t] = v_t[x].
    H_max = max(horizons)
    v = indic.copy()
    out = np.zeros((K, len(horizons)), dtype=np.float64)
    h_set = {h: i for i, h in enumerate(horizons)}
    for t in range(1, H_max + 1):
        v = P @ v
        if t in h_set:
            out[:, h_set[t]] = v
    return out


def exact_hitting_cdf(P: np.ndarray, failure_cells, horizons) -> np.ndarray:
    """P_hit[x, Dt] = P(tau_F <= Dt | X_0 = x), computed exactly.

    For x in F: 1 for all Dt >= 1 (already in basin at Dt=0, so trivially
    tau_F = 0 <= Dt; hitting CDF at Dt = 0 is 1 too but we report only
    Dt >= 1 for symmetry with P_end).

    For x not in F: 1 - [Q^Dt . 1]_x, where Q = P[non-F rows, non-F cols]
    and 1 is the all-ones vector on non-F indices.

    Parameters
    ----------
    P : (K, K) ndarray, row-stochastic.
    failure_cells : iterable of int, cell indices in F.
    horizons : sequence of positive int, values of Dt.

    Returns
    -------
    (K, len(horizons)) ndarray.
    """
    P = np.asarray(P, dtype=np.float64)
    K = P.shape[0]
    if P.shape != (K, K):
        raise ValueError(f"P must be square, got shape {P.shape}")
    horizons = list(horizons)
    if any(h < 1 for h in horizons):
        raise ValueError("horizons must be >= 1")

    F = sorted(set(int(c) for c in failure_cells))
    NF = [c for c in range(K) if c not in set(F)]
    if not NF:
        # Pathological: all cells are failure cells. Hitting probability = 1 everywhere.
        return np.ones((K, len(horizons)), dtype=np.float64)

    # Q = P restricted to non-failure rows and columns, shape (|NF|, |NF|).
    Q = P[np.ix_(NF, NF)].astype(np.float64)

    H_max = max(horizons)
    # v_t[i] = [Q^t . 1]_{NF[i]} = P(X_t in NF for all steps <= t, and
    #                               X_0 = NF[i])
    # Starting at t=0: v_0 = 1 (all ones), then v_t = Q . v_{t-1}.
    v = np.ones(len(NF), dtype=np.float64)
    out = np.zeros((K, len(horizons)), dtype=np.float64)
    # x in F rows are 1 at all horizons.
    out[F, :] = 1.0
    h_set = {h: i for i, h in enumerate(horizons)}
    for t in range(1, H_max + 1):
        v = Q @ v
        if t in h_set:
            # For x in NF, P(tau_F <= t | X_0 = x) = 1 - v[NF.index(x)].
            col = h_set[t]
            for i_nf, x in enumerate(NF):
                out[x, col] = 1.0 - v[i_nf]
    return out


def assert_hit_ge_end(P_hit: np.ndarray, P_end: np.ndarray, tol: float = 1e-12) -> None:
    """P_hit[x, Dt] >= P_end[x, Dt] for all x, Dt. Any sample in F at Dt
    was in F at some step <= Dt, so it's counted in P_hit."""
    gap = P_hit - P_end
    worst = float(gap.min())
    if worst < -tol:
        idx = np.unravel_index(int(gap.argmin()), gap.shape)
        raise AssertionError(
            f"P_hit < P_end at {idx}: P_hit={P_hit[idx]:.6g}, "
            f"P_end={P_end[idx]:.6g}, gap={worst:.3e}"
        )


def leakage_by_cell(P_hit: np.ndarray, P_end: np.ndarray) -> np.ndarray:
    """(P_hit - P_end) — per-cell, per-horizon basin leakage.

    Zero for absorbing basins. Positive for chains where failure cells
    transition back to nominal cells.
    """
    return P_hit - P_end
