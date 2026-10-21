"""Stall-free guarantee: the token-level reservation composed with the layer-level scheduler.

run: python -m pytest -q tests
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anytimestream import ReservationScheduler, run_token  # noqa: E402
from anytimestream.stream import PriceRule, TokenStreamScheduler  # noqa: E402


def stream(rng, policy_rule, n_tokens=200):
    L = int(rng.integers(2, 30))
    m = rng.uniform(1e-5, 1e-4, L)
    head = float(rng.uniform(0, 5e-5))
    M = m.sum() + head
    P = M * float(rng.uniform(1.0, 4.0))
    rule, lam = None, None
    if policy_rule:
        rule = PriceRule([0, 500, 1000, 2000], [1.0, 0.4, 0.2, 0.1])
        lam = rule.fit_lambda(rng.lognormal(np.log(2e-7), 1.0, 1000), max(P - M, 1e-12))
    ts = TokenStreamScheduler(P, M, startup=float(rng.uniform(0, 3)) * P, rule=rule, lam=lam)
    ts.begin(0.0)
    now, w = 0.0, rng.uniform(0.1, 2, L)
    for t in range(n_tokens):
        tau = float(rng.lognormal(np.log(2e-7), 1.5))
        D = ts.token_deadline(t, now, tau)
        sched = ReservationScheduler(m, D * (1 + 1e-12), w, kappa=1e-7, head=head)
        arr = [rng.pareto(1.0, int(rng.integers(0, 400))) * tau * 50 for _ in range(L)]
        tr = run_token(sched, m * rng.uniform(0, 1, L), arr, start=now)
        assert tr.end <= ts.due(t) + 1e-9, (t, tr.end, ts.due(t))
        now = tr.end


@pytest.mark.parametrize("seed", range(40))
def test_never_stalls(seed):
    rng = np.random.default_rng(seed)
    stream(rng, policy_rule=bool(seed % 2))


def test_rejects_impossible_rate():
    with pytest.raises(ValueError):
        TokenStreamScheduler(period=1.0, mandatory=1.5)


def test_price_buys_more_when_flash_is_fast():
    rule = PriceRule([0, 500, 1000, 2000], [1.0, 0.4, 0.2, 0.1])
    fast, slow = rule.neurons(5.0, [1e-4, 1e-3])
    assert fast > slow
