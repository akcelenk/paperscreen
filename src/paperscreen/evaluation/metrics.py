"""Ranking metrics with seeded random tie-breaking (so ties never favour label order)."""

from __future__ import annotations

import numpy as np


def order(scores: np.ndarray, seed: int = 0) -> np.ndarray:
    """Indices from best to worst; ties are broken by a seeded random key, not input order."""
    s = np.asarray(scores, dtype=float)
    tie = np.random.default_rng(seed).random(len(s))
    return np.lexsort((tie, -s))


def average_precision(y: np.ndarray, scores: np.ndarray, seed: int = 0) -> float:
    y = np.asarray(y)
    if (y == 1).sum() == 0:
        return float("nan")
    ranked = y[order(scores, seed)] == 1
    hits = np.cumsum(ranked)
    prec = hits / (np.arange(len(ranked)) + 1)
    return float(prec[ranked].mean())


def precision_at_k(y: np.ndarray, scores: np.ndarray, k: int, seed: int = 0) -> float:
    y = np.asarray(y)
    if len(y) == 0:
        return float("nan")
    k = min(k, len(y))
    return float((y[order(scores, seed)][:k] == 1).mean())


def ndcg_at_k(y: np.ndarray, scores: np.ndarray, k: int, seed: int = 0) -> float:
    y = (np.asarray(y) == 1).astype(float)
    if y.sum() == 0:
        return float("nan")
    k = min(k, len(y))
    disc = 1.0 / np.log2(np.arange(k) + 2)
    dcg = float((y[order(scores, seed)][:k] * disc).sum())
    ideal = float((np.sort(y)[::-1][:k] * disc).sum())
    return dcg / ideal


def window_metrics(y, scores, k: int = 10, ndcg_k: int = 20, seed: int = 0) -> dict[str, float]:
    return {
        "ap": average_precision(y, scores, seed),
        "p_at_k": precision_at_k(y, scores, k, seed),
        "ndcg": ndcg_at_k(y, scores, ndcg_k, seed),
        "n": int(len(y)),
        "n_pos": int((np.asarray(y) == 1).sum()),
    }
