"""Pure-function tests for the metrics math — no DB."""

import math
import random

from overlaat import metrics_db as m


def test_pct():
    assert m._pct([], 0.5) is None
    assert m._pct([1, 2, 3, 4, 5], 0.5) == 3
    assert m._pct([5, 3, 1, 4, 2], 0.0) == 1
    assert m._pct([1, 2, 3, 4, 5], 1.0) == 5
    assert m._pct([None, 2, 4], 0.5) in (2, 4)  # None filtered out


def test_mean_concurrency_alone_and_empty():
    assert m._mean_concurrency(0, 10, [(0, 10)]) == 1.0
    assert m._mean_concurrency(0, 10, []) == 0.0
    assert m._mean_concurrency(5, 5, [(0, 10)]) == 0.0  # zero-width window


def test_mean_concurrency_overlap():
    assert m._mean_concurrency(0, 10, [(0, 10), (0, 10)]) == 2.0
    assert m._mean_concurrency(0, 10, [(0, 10), (5, 10)]) == 1.5  # half overlap


def test_concintegral_matches_naive():
    rng = random.Random(42)
    intervals = [(s, s + rng.uniform(0.1, 5)) for s in (rng.uniform(0, 20) for _ in range(40))]
    ci = m._ConcIntegral(intervals)
    for _ in range(50):
        a = rng.uniform(0, 20)
        b = a + rng.uniform(0.1, 5)
        assert math.isclose(
            ci.mean(a, b), m._mean_concurrency(a, b, intervals), rel_tol=1e-9, abs_tol=1e-9
        )


def test_concintegral_empty():
    assert m._ConcIntegral([]).mean(0, 10) == 0.0


def test_bucket_concurrency():
    res = m._bucket_concurrency([(0, 5)], 0, 5, 3)
    assert res == [1.0, 0.0, 0.0]
    res2 = m._bucket_concurrency([(0, 10)], 0, 5, 3)
    assert res2[0] == 1.0 and res2[1] == 1.0
