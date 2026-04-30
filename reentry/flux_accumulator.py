"""
Flux Accumulator (𝒜) — degradation-flux operator for STTS.

Canonical Pythagorean form:

    Φ(t) = sqrt( H(t)^2  +  2 α^2 · D_KL(t) )
    σ_raw(t) = ∫_0^t Φ(s) ds      (discretised as Σ_t Φ(t) Δt)

Components:
    H(t)      drift component on the LDA-projection base manifold
              (existing pipeline; non-negative under current
              implementation, V = |M2 − μ_nom|, T = ½(ΔM2)^2)
    D_KL(t)   deformation magnitude on the transition fiber: KL between
              the rolling W-window empirical destination distribution
              and the trained transition row at the current cell
    α         single global scalar (frozen after Part 2 calibration);
              chosen to L^2-equate H and the Fisher-Rao length
              δ_FR = α · sqrt(2 D_KL).

Pre-registered choices (do not adjust based on results):
    W       = 5 windows  (matches existing signal-separator buffer)
    EPS     = 1e-9       (smoothing for empirical zeros against
                          trained-row nonzero entries)

Closed-form variance prediction (Kemeny-Snell on transient submatrix):
    Q  = P[transient, transient]                     (10×10 here)
    N  = (I − Q)^{-1}                                fundamental matrix
    v  = N κ                                         expected σ_failure
    Var_i[σ_failure] = [N (κ ⊙ (2v − κ))]_i − v_i^2

Per-cell mean cost (κ) is aggregated **Pythagorean-then-mean**:

    κ_c = mean over (t,i) with M2_i(t) ∈ c of   sqrt(H_i(t)^2 + 2 α^2 D_KL_i(t))

Mean-then-Pythagorean is incorrect by Jensen's inequality and would
inflate κ. The aggregation order matters.

This module is read-only against the frozen Markov table and the
existing pipeline outputs. It does not call .fit() on any sklearn
artifact.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

# ── Pre-registered constants ────────────────────────────────────

DKL_WINDOW = 5            # rolling buffer for empirical destination distribution
DKL_EPS = 1e-9            # smoothing for log argument
SANITY_MC_N = 1_000_000   # Monte Carlo samples for Part 1 sanity check
SANITY_NSIGMA = 5.0       # gate width on Var sanity check, in MC standard
                          # errors estimated empirically from the same sample.
                          # Replaces the brief's pre-registered absolute
                          # tolerance of 1e-2, which was implicitly calibrated
                          # against a Gaussian σ assumption that does not hold
                          # for absorbing-chain reward sums (excess kurtosis
                          # ratio ≈ 8.5 on the toy chain). See Note 1 in
                          # flux_summary.md for the brief-defect-and-fix.
SANITY_SEED = 20260429    # MC RNG seed (Part 1)
BOOTSTRAP_B = 1_000       # bootstrap replicates (Part 4)
BOOTSTRAP_SEED = 20260430 # bootstrap RNG seed (Part 4)


# ── Per-window D_KL ─────────────────────────────────────────────

def per_window_dkl(
    cells: np.ndarray,
    P_trained: np.ndarray,
    window: int = DKL_WINDOW,
    eps: float = DKL_EPS,
) -> np.ndarray:
    """KL divergence per window between rolling-W empirical destination
    distribution and the trained transition row at the current cell.

    Parameters
    ----------
    cells : (n_windows,) int array of M2 cell assignments per window
    P_trained : (K, K) row-stochastic transition matrix
    window : rolling buffer size (default 5)
    eps : additive smoothing for log argument (default 1e-9)

    Returns
    -------
    dkl : (n_windows,) float array. NaN for the first `window` windows
          where < `window` transitions are available.
    """
    n = len(cells)
    K = P_trained.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for t in range(window, n):
        # The W transitions ending at t: cells[t-window..t] gives W+1 cells,
        # (W) transitions, destinations = cells[t-window+1..t+1]?
        # Convention: at window t, the transition that produced t is
        # cells[t-1] -> cells[t]. The last W transitions ending at t are
        # cells[t-W] -> cells[t-W+1], ..., cells[t-1] -> cells[t].
        # Empirical destinations of those W transitions:
        dests = cells[t - window + 1: t + 1]  # length W
        counts = np.bincount(dests, minlength=K).astype(np.float64)
        emp = counts / counts.sum()
        # Trained row at the *current* cell c(t)
        trained = P_trained[cells[t]].astype(np.float64)
        # KL(emp || trained) with smoothing: replace 0/0 limit by 0.
        # Apply ε to both numerator and denominator inside the log to keep
        # the divergence finite and continuous in the empirical zeros.
        kl = 0.0
        for j in range(K):
            p = emp[j]
            q = trained[j]
            if p <= 0:
                continue
            kl += p * np.log((p + eps) / (q + eps))
        out[t] = kl
    return out


# ── Operator integrand and integral ─────────────────────────────

def phi(H: np.ndarray, dkl: np.ndarray, alpha: float) -> np.ndarray:
    """Φ(t) = sqrt(H(t)^2 + 2 α^2 · D_KL(t)).

    NaN propagates: any window with NaN dkl produces NaN Φ. The driver
    is responsible for excluding NaN windows from σ_raw integration.
    """
    return np.sqrt(np.asarray(H, dtype=np.float64) ** 2
                   + 2.0 * alpha * alpha * np.asarray(dkl, dtype=np.float64))


def sigma_raw(phi_t: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """σ_raw(t) = Σ_{s ≤ t} Φ(s) · Δt.

    NaN-safe: NaN entries are skipped (treated as zero contribution).
    Returns the running cumulative sum.
    """
    finite = np.where(np.isfinite(phi_t), phi_t, 0.0)
    return np.cumsum(finite * dt)


# ── α calibration ───────────────────────────────────────────────

def calibrate_alpha(per_object_H: list[np.ndarray],
                    per_object_dkl: list[np.ndarray],
                    dt: float = 1.0) -> float:
    """Global α from L^2-equality on H and the Fisher-Rao length.

    α = sqrt( Σ_i Σ_t H_i(t)^2 Δt  /  ( 2 · Σ_i Σ_t D_KL_i(t) Δt ) )

    NaN entries (e.g. early windows where D_KL undefined) are excluded
    from BOTH numerator and denominator integrals via a shared finite
    mask per object. This keeps α the L^2 ratio over the windows where
    both quantities are defined.
    """
    sum_H2 = 0.0
    sum_dkl = 0.0
    for H, dkl in zip(per_object_H, per_object_dkl):
        H = np.asarray(H, dtype=np.float64)
        dkl = np.asarray(dkl, dtype=np.float64)
        m = np.isfinite(H) & np.isfinite(dkl)
        sum_H2 += np.sum((H[m] ** 2) * dt)
        sum_dkl += np.sum(dkl[m] * dt)
    if sum_dkl <= 0:
        raise ValueError("Σ D_KL is non-positive; cannot calibrate α.")
    return float(np.sqrt(sum_H2 / (2.0 * sum_dkl)))


# ── Per-cell mean cost (κ) — Pythagorean-then-mean ──────────────

def kappa_per_cell(per_object_H: list[np.ndarray],
                    per_object_dkl: list[np.ndarray],
                    per_object_cells: list[np.ndarray],
                    transient_cells: Iterable[int],
                    alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """κ_c = mean over windows in cell c of sqrt(H^2 + 2 α^2 · D_KL).

    Pythagorean-then-mean. Mean-then-Pythagorean would understate the
    typical magnitude by Jensen's inequality (concavity of sqrt).

    Returns
    -------
    kappa : (len(transient_cells),) float array, in the order given
            by the iterable transient_cells
    counts : (len(transient_cells),) int array of window counts per cell
    """
    transient_list = list(transient_cells)
    n_t = len(transient_list)
    cell_to_idx = {c: i for i, c in enumerate(transient_list)}
    sum_phi = np.zeros(n_t, dtype=np.float64)
    counts = np.zeros(n_t, dtype=np.int64)
    for H, dkl, cells in zip(per_object_H, per_object_dkl, per_object_cells):
        H = np.asarray(H, dtype=np.float64)
        dkl = np.asarray(dkl, dtype=np.float64)
        cells = np.asarray(cells, dtype=np.int64)
        m = np.isfinite(H) & np.isfinite(dkl)
        if not m.any():
            continue
        Hf = H[m]; dklf = dkl[m]; cf = cells[m]
        phif = np.sqrt(Hf ** 2 + 2.0 * alpha * alpha * dklf)
        for c in transient_list:
            mask = cf == c
            if mask.any():
                sum_phi[cell_to_idx[c]] += float(phif[mask].sum())
                counts[cell_to_idx[c]] += int(mask.sum())
    kappa = np.zeros(n_t, dtype=np.float64)
    for i in range(n_t):
        if counts[i] > 0:
            kappa[i] = sum_phi[i] / counts[i]
    return kappa, counts


# ── Kemeny-Snell variance ───────────────────────────────────────

def kemeny_snell_variance(P: np.ndarray,
                          transient_cells: Iterable[int],
                          kappa: np.ndarray) -> dict:
    """Closed-form mean and variance of accumulated cost to absorption.

    Inputs
    ------
    P : (K, K) row-stochastic transition matrix (transient + absorbing)
    transient_cells : iterable of transient cell indices, length = T
                      same order as kappa
    kappa : (T,) per-state mean cost

    Returns
    -------
    dict with N (T×T fundamental), v = Nκ (mean σ_failure | start cell),
    Var_predicted = N(κ ⊙ (2v − κ)) − v⊙v.

    Derivation note (independently re-derived, conditioning on first
    transition with reward κ_i paid at state i):
        E_i[σ²] satisfies (I − Q) g = κ⊙κ + 2·κ⊙(Qv)
        Q v = v − κ            (since v = Nκ)
        ⇒ g = N (κ⊙κ + 2 κ⊙(v − κ)) = N (κ ⊙ (2v − κ))
        Var_i[σ] = g_i − v_i^2
    """
    transient_list = list(transient_cells)
    T = len(transient_list)
    Q = P[np.ix_(transient_list, transient_list)].astype(np.float64)
    I = np.eye(T)
    N = np.linalg.inv(I - Q)
    v = N @ kappa
    g = N @ (kappa * (2.0 * v - kappa))
    var = g - v * v
    return {"N": N, "v": v, "g": g, "Var_predicted": var, "Q": Q}


# ── Bootstrap CI for variance (vectorised over replicates) ──────

def bootstrap_variance_ci(values: np.ndarray,
                          n_replicates: int = BOOTSTRAP_B,
                          seed: int = BOOTSTRAP_SEED,
                          alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap CI for sample variance (ddof=1) of `values`.

    Returns (point, ci_lo, ci_hi).
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(np.var(values, ddof=1))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_replicates, n))
    samples = values[idx]
    repl = np.var(samples, axis=1, ddof=1)
    return point, float(np.quantile(repl, alpha / 2)), float(np.quantile(repl, 1 - alpha / 2))


# ── R^2 of linear fit (per object) ──────────────────────────────

def c_restricted_corpus(per_object_trajectories: list[dict],
                          failure_cells: Iterable[int]) -> dict:
    """Partition the 78-object set under Interpretation C — Hybrid
    Restricted-Corpus Semantics.

    For each trajectory, classify:
      - INCLUDED: first window's cell is transient AND a later window
        falls in the failure basin. Compute first basin-entry index
        τ_i (smallest t ≥ 1 with cells[t] ∈ F).
      - EXCLUDED, starting-in-basin: cells[0] ∈ F.
      - EXCLUDED, never-reach-basin: cells[0] ∉ F but no later window
        is in F within the observation horizon.

    Each trajectory dict must contain at minimum keys: 'nid', 'cells'
    (1-D int array). The returned 'restricted' list copies the input
    dict and adds 'tau_i' and 'c_first' fields.
    """
    F = set(int(c) for c in failure_cells)
    restricted: list[dict] = []
    starting_in_basin: list[str] = []
    never_reach_basin: list[str] = []

    for traj in per_object_trajectories:
        cells = np.asarray(traj["cells"], dtype=np.int64)
        if len(cells) == 0:
            continue
        c_first = int(cells[0])
        if c_first in F:
            starting_in_basin.append(traj["nid"])
            continue
        tau_i = None
        for t in range(1, len(cells)):
            if int(cells[t]) in F:
                tau_i = t
                break
        if tau_i is None:
            never_reach_basin.append(traj["nid"])
            continue
        out = dict(traj)
        out["tau_i"] = int(tau_i)
        out["c_first"] = c_first
        restricted.append(out)

    return {
        "restricted": restricted,
        "starting_in_basin": starting_in_basin,
        "never_reach_basin": never_reach_basin,
        "n_total": len(per_object_trajectories),
        "n_included": len(restricted),
        "n_starting_in_basin": len(starting_in_basin),
        "n_never_reach_basin": len(never_reach_basin),
    }


def truncate_to_tau(traj: dict) -> dict:
    """Return a copy of `traj` with H, dkl, cells truncated to indices
    [0, tau_i] inclusive. Requires `tau_i` field set by
    `c_restricted_corpus`."""
    if "tau_i" not in traj:
        raise ValueError(f"trajectory {traj.get('nid','?')} has no tau_i")
    end = int(traj["tau_i"]) + 1
    out = dict(traj)
    out["cells"] = np.asarray(traj["cells"])[:end]
    out["H"] = np.asarray(traj["H"])[:end]
    if "dkl" in traj:
        out["dkl"] = np.asarray(traj["dkl"])[:end]
    if "days" in traj:
        out["days"] = np.asarray(traj["days"])[:end]
    out["n_windows"] = end
    return out


def variance_stderr(values: np.ndarray) -> float:
    """Asymptotic standard error of the sample variance (ddof=1) from
    the same MC sample, using SE(S^2) ≈ sqrt((μ_4 − μ_2^2 (N-3)/(N-1)) / N)
    where μ_k is the k-th central moment of `values`.

    For Gaussian samples this reduces to sqrt(2)·σ²/sqrt(N). For
    heavy-tailed samples (e.g. absorbing-chain reward sums with rare
    long trajectories) the SE is dominated by the excess fourth moment.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 4:
        return float("nan")
    mu = values.mean()
    diff = values - mu
    mu2 = float((diff ** 2).mean())
    mu4 = float((diff ** 4).mean())
    var_of_var = (mu4 - mu2 * mu2 * (n - 3) / (n - 1)) / n
    if var_of_var < 0:
        # Numerical edge case: clip to zero
        return 0.0
    return float(np.sqrt(var_of_var))


def linear_r2(t: np.ndarray, y: np.ndarray) -> float:
    """Coefficient of determination of the OLS line y = β₀ + β₁ t."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(t) < 2:
        return float("nan")
    finite = np.isfinite(t) & np.isfinite(y)
    if finite.sum() < 2:
        return float("nan")
    t = t[finite]
    y = y[finite]
    # OLS:
    t_mean = t.mean()
    y_mean = y.mean()
    Stt = float(((t - t_mean) ** 2).sum())
    if Stt == 0:
        return float("nan")
    beta1 = float(((t - t_mean) * (y - y_mean)).sum() / Stt)
    beta0 = float(y_mean - beta1 * t_mean)
    yhat = beta0 + beta1 * t
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y_mean) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot
