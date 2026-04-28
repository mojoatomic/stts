"""
Percentile bootstrap confidence intervals for object-level aggregate
statistics. Companion to `validate.wilson_ci` (analytic Wilson CI for
a proportion); style matched to that module.

Design notes
------------
- NaN-safe: NaN entries in `values` are dropped before resampling.
- Vectorised: one `rng.integers((B, n))` call, no Python loop over B.
- Seeded: default seed 20260420 (plan REV 3); override per call.
- Percentile interval: alpha/2 and 1 - alpha/2 quantiles of the
  bootstrap-replicate statistics. Symmetric by construction, so
  `ci_lo <= point <= ci_hi` holds unless the sample is degenerate.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _drop_nan(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {values.shape}")
    if values.dtype.kind in ("f", "c"):
        return values[~np.isnan(values)]
    return values


def bootstrap_ci(
    values,
    statistic: Callable[[np.ndarray], float] = np.median,
    n_replicates: int = 1000,
    seed: int = 20260420,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for `statistic` over `values`.

    Parameters
    ----------
    values : 1-D array-like. NaNs are dropped before resampling.
    statistic : callable mapping 1-D ndarray to float. Default median.
    n_replicates : number of bootstrap samples (B).
    seed : PRNG seed for reproducibility.
    alpha : two-sided significance level (0.05 = 95% CI).

    Returns
    -------
    (point, ci_lo, ci_hi) where point = statistic(values) and
    (ci_lo, ci_hi) are the alpha/2 and 1-alpha/2 percentiles of the
    bootstrap distribution of `statistic`.
    """
    values = _drop_nan(values)
    n = values.shape[0]
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(statistic(values))

    rng = np.random.default_rng(seed)
    # Single vectorised resample: shape (B, n) of integer indices.
    idx = rng.integers(0, n, size=(n_replicates, n))
    resamples = values[idx]  # shape (B, n)
    # Apply the statistic row-wise. For np.median / np.mean, pass axis=1.
    # For generic callables, fall back to a Python list comprehension
    # (still fast for B=1000 and small n).
    if statistic is np.median:
        repl = np.median(resamples, axis=1)
    elif statistic is np.mean:
        repl = np.mean(resamples, axis=1)
    else:
        repl = np.array([statistic(resamples[b]) for b in range(n_replicates)])
    lo = float(np.quantile(repl, alpha / 2))
    hi = float(np.quantile(repl, 1 - alpha / 2))
    return point, lo, hi


def bootstrap_rate_ci(
    mask,
    n_replicates: int = 1000,
    seed: int = 20260420,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a proportion mask.sum()/len(mask).

    Provided for envelope consistency with other bootstrap numbers in a
    report. Use `validate.wilson_ci` when only a single proportion CI is
    wanted and the analytic form is preferable.
    """
    mask = np.asarray(mask).astype(np.int64)
    n = mask.shape[0]
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(mask.sum()) / n
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_replicates, n))
    resamples = mask[idx]  # (B, n) of 0/1
    repl = resamples.mean(axis=1)
    lo = float(np.quantile(repl, alpha / 2))
    hi = float(np.quantile(repl, 1 - alpha / 2))
    return point, lo, hi
