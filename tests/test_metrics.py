import numpy as np
import pytest

from paperscreen.evaluation.bootstrap import holm, mean_ci, paired_difference
from paperscreen.evaluation.metrics import average_precision, ndcg_at_k, order, precision_at_k, window_metrics


def test_known_values():
    y, s = np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1])
    assert average_precision(y, s) == pytest.approx((1 + 2 / 3) / 2)
    assert precision_at_k(y, s, 2) == 0.5
    assert ndcg_at_k(y, s, 4) == pytest.approx(1.5 / (1 + 1 / np.log2(3)))
    assert np.isnan(average_precision(np.zeros(3), np.ones(3)))
    assert np.isnan(ndcg_at_k(np.zeros(3), np.ones(3), 3))


def test_perfect_and_worst_rankings():
    y = np.array([0, 0, 1, 1, 0])
    assert average_precision(y, y.astype(float)) == 1.0
    assert average_precision(y, 1 - y.astype(float)) < 0.5


def test_ties_do_not_follow_input_order():
    y = np.array([1] * 5 + [0] * 5)  # positives first in input
    aps = [average_precision(y, np.zeros(10), seed=s) for s in range(200)]
    assert 0.5 < np.mean(aps) < 0.75 and np.std(aps) > 0  # ~chance, not the optimistic 1.0
    assert order(np.zeros(4), seed=1).tolist() != list(range(4)) or order(np.zeros(4), seed=2).tolist() != list(range(4))


def test_window_metrics_keys():
    m = window_metrics(np.array([1, 0, 0]), np.array([0.2, 0.9, 0.1]), k=2, ndcg_k=3)
    assert m["n"] == 3 and m["n_pos"] == 1 and 0 < m["ap"] < 1


def test_mean_ci_contains_mean_and_is_seeded():
    v = np.array([0.2, 0.4, 0.5, 0.6, 0.3, 0.45])
    m, lo, hi = mean_ci(v, n_boot=1000, seed=3)
    assert lo <= m <= hi and (m, lo, hi) == mean_ci(v, n_boot=1000, seed=3)
    assert np.isnan(mean_ci(np.array([np.nan]))[0])


def test_paired_difference():
    a = np.array([0.6, 0.7, 0.65, 0.8, 0.7, 0.75])
    same = paired_difference(a, a.copy(), n_boot=500)
    assert same["diff"] == 0 and same["p"] == 1.0
    big = paired_difference(a, a - 0.2, n_boot=500)
    assert big["diff"] == pytest.approx(0.2) and big["lo"] > 0 and big["p"] < 0.01
    dropped = paired_difference(np.array([1, np.nan, 3.0]), np.array([0, 1, 1.0]), n_boot=100)
    assert dropped["n_windows"] == 2


def test_holm_matches_hand_calculation():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    assert np.isnan(holm({"x": float("nan"), "y": 0.5})["x"])


def test_single_window_has_no_interval():
    m, lo, hi = mean_ci(np.array([0.7]), n_boot=100)
    assert m == 0.7 and lo != lo and hi != hi  # NaN, never a zero-width interval
