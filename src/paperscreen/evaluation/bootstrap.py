"""Cluster bootstrap over evaluation windows, paired differences, and Holm correction."""

from __future__ import annotations

import numpy as np


def _indices(n_units: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_units, size=(n_boot, n_units))


def mean_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Mean of per-window values with a percentile cluster-bootstrap CI (windows are resampled).

    With fewer than two windows there is no between-window variation to resample, so the interval is
    undefined (NaN) rather than a misleading zero-width interval around the single value.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(v) < 2:
        return float(v.mean()), float("nan"), float("nan")
    boots = v[_indices(len(v), n_boot, seed)].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def paired_difference(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> dict[str, float]:
    """Paired window-level difference ``mean(a - b)`` with CI and a two-sided bootstrap p-value.

    Windows where either value is missing are dropped from both. The p-value is the share of
    bootstrap means on the other side of zero (doubled); it cannot go below ``1/(n_boot+1)``.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[ok]
    if len(d) < 2:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"),
                "n_windows": int(len(d))}
    boots = d[_indices(len(d), n_boot, seed)].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "p": float(min(1.0, max(p, 1 / (n_boot + 1)))), "n_windows": int(len(d))}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone). NaNs are passed through."""
    items = [(k, p) for k, p in pvals.items() if not np.isnan(p)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out = {k: float("nan") for k in pvals}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
