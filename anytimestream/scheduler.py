"""Reservation scheduler: every token finishes by its deadline, whatever flash does.

Each layer's work has two parts:

* mandatory: attention plus the FFN neurons whose weights are already resident in RAM. Its
  worst-case time m_l is known ahead of time, because it never touches flash.
* optional: FFN neurons whose weights must be streamed from flash. Reads are issued
  asynchronously; at the layer's cutoff, reads that have not arrived are abandoned (LeanStream's
  cutoff flag), and an arrived neuron is computed only if its compute (at most `kappa`) fits.

Invariant (checked by tests/test_scheduler.py): when layer l starts at time s_l,

    s_l + sum_{j >= l} m_j  <=  start + deadline - head

It holds at l = 0 when the deadline is feasible, and each layer preserves it, because the
layer's cutoff never exceeds s_l + m_l + slack_l, where slack_l is the time beyond all remaining
mandatory work. Spare time is split across layers by `weights` (a budget table row): layer l
gets slack_l * w_l / sum_{j >= l} w_j.
"""
from dataclasses import dataclass, field

import numpy as np


class InfeasibleDeadline(ValueError):
    pass


@dataclass
class LayerTrace:
    start: float
    end: float
    cutoff: float
    optional_done: int
    optional_arrived: int


@dataclass
class TokenTrace:
    start: float
    deadline: float
    layers: list = field(default_factory=list)
    end: float = 0.0

    @property
    def met(self):
        return self.end <= self.start + self.deadline + 1e-12


class ReservationScheduler:
    def __init__(self, mandatory, deadline, weights=None, kappa=0.0, head=0.0):
        self.m = np.asarray(mandatory, dtype=float)
        self.L = len(self.m)
        self.D, self.kappa, self.head = float(deadline), float(kappa), float(head)
        w = np.ones(self.L) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != (self.L,) or (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be non-negative, one per layer, not all zero")
        self.w = w
        self.w_suffix = np.cumsum(w[::-1])[::-1]
        self.m_suffix = np.concatenate([np.cumsum(self.m[::-1])[::-1], [0.0]])
        if self.m_suffix[0] + self.head > self.D:
            raise InfeasibleDeadline(
                f"deadline {self.D:.6g}s is shorter than the mandatory work {self.m_suffix[0] + self.head:.6g}s")
        self.start = None

    def begin(self, now):
        self.start = float(now)

    def slack(self, l, now):
        """Time left after reserving every remaining layer's mandatory work and the LM head."""
        return self.start + self.D - self.head - now - self.m_suffix[l]

    def cutoff(self, l, now):
        """Absolute time by which layer l (started at `now`) must finish its optional work."""
        s = max(0.0, self.slack(l, now))
        share = self.w[l] / self.w_suffix[l] if self.w_suffix[l] > 0 else 1.0
        return now + self.m[l] + s * share


def run_token(sched, mandatory_actual, arrivals, start=0.0, head_actual=None):
    """Simulate one token.

    mandatory_actual[l]: real mandatory time of layer l (must be <= sched.m[l]).
    arrivals[l]: arrival delays (seconds after the layer's mandatory part ends) of that layer's
                 optional neurons, in issue order; np.inf for a read that never returns.
    Returns a TokenTrace.
    """
    sched.begin(start)
    tr = TokenTrace(start=start, deadline=sched.D)
    t = start
    for l in range(sched.L):
        s_l = t
        cut = sched.cutoff(l, s_l)
        t += mandatory_actual[l]
        mand_end = t
        done = arrived = 0
        for a in np.sort(np.asarray(arrivals[l], dtype=float)):
            ready = mand_end + a
            if ready > cut:
                break                      # cutoff reached: abandon this and every later read
            arrived += 1
            t = max(t, ready)
            if t + sched.kappa > cut:
                break
            t += sched.kappa
            done += 1
        tr.layers.append(LayerTrace(s_l, t, cut, done, arrived))
    t += sched.head if head_actual is None else head_actual
    tr.end = t
    return tr
