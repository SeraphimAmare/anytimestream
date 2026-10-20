"""The deadline guarantee, tested against random and adversarial timings.

run: python -m pytest -q tests
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anytimestream import BudgetTable, InfeasibleDeadline, ReservationScheduler, run_token  # noqa: E402


def random_case(rng):
    L = int(rng.integers(1, 40))
    m = rng.uniform(1e-5, 2e-4, L)
    head = float(rng.uniform(0, 1e-4))
    D = (m.sum() + head) * float(rng.uniform(1.0, 6.0))
    w = rng.uniform(0, 3, L)
    w[rng.integers(L)] += 0.1                       # never all zero
    kappa = float(rng.uniform(0, 3e-5))
    return L, m, head, D, w, kappa


def random_arrivals(rng, L, scale):
    out = []
    for _ in range(L):
        n = int(rng.integers(0, 300))
        a = rng.pareto(1.2, n) * scale              # heavy-tailed flash latency
        a[rng.random(n) < 0.05] = np.inf            # some reads never come back
        out.append(a)
    return out


def test_deadline_always_met():
    rng = np.random.default_rng(0)
    worst = -np.inf
    for _ in range(5000):
        L, m, head, D, w, kappa = random_case(rng)
        s = ReservationScheduler(m, D, w, kappa=kappa, head=head)
        mand = m * rng.uniform(0, 1, L)             # real mandatory time within its bound
        tr = run_token(s, mand, random_arrivals(rng, L, float(rng.uniform(1e-6, 1e-3))), start=float(rng.uniform(0, 5)))
        assert tr.met, (tr.end - tr.start, D)
        for lt in tr.layers:
            assert lt.end <= lt.cutoff + 1e-12
        worst = max(worst, (tr.end - tr.start) / D)
    assert worst <= 1 + 1e-9


def test_mandatory_at_worst_case_and_dead_flash():
    m = np.full(12, 1e-4)
    s = ReservationScheduler(m, 1.31e-3, kappa=1e-5, head=1e-4)
    tr = run_token(s, m, [np.full(50, np.inf)] * 12)
    assert tr.met and all(lt.optional_done == 0 for lt in tr.layers)


def test_infeasible_deadline_rejected():
    with pytest.raises(InfeasibleDeadline):
        ReservationScheduler(np.full(4, 1e-3), 3.9e-3)


def test_generous_deadline_does_everything():
    m = np.full(6, 1e-4)
    s = ReservationScheduler(m, 1.0, kappa=1e-6)
    arr = [np.linspace(0, 1e-4, 100)] * 6
    tr = run_token(s, m, arr)
    assert all(lt.optional_done == 100 for lt in tr.layers)


def test_weights_move_optional_work():
    m = np.full(4, 1e-4)
    arr = [np.zeros(10_000)] * 4                    # flash is instant: compute-bound
    even = run_token(ReservationScheduler(m, 1e-3, kappa=1e-6), m, arr)
    tilted = run_token(ReservationScheduler(m, 1e-3, [4, 1, 1, 1], kappa=1e-6), m, arr)
    assert tilted.layers[0].optional_done > 1.5 * even.layers[0].optional_done
    assert tilted.met and even.met


def test_budget_table_lookup(tmp_path):
    t = BudgetTable([0.2, 0.1], [[0.4, 0.2, 0.0], [0.2, 0.1, 0.0]], mode="compute")
    assert np.allclose(t.lookup(0.15), [0.3, 0.15, 0.0])
    assert np.allclose(t.lookup(0.9), [0.4, 0.2, 0.0])     # clamped
    assert np.isclose(t.weights(0.1).mean(), 1.0)
    p = tmp_path / "t.json"
    t.save(p)
    assert np.allclose(BudgetTable.load(p).lookup(0.1), t.lookup(0.1))
