"""Token-level scheduling against a consumption deadline (stall-free streaming).

A reader or a text-to-speech engine consumes token t at time  due_t = start + startup + t * period.
A stall is a token that is not ready when it is due. Time a token does not use is banked.

Reservation rule, one level above ReservationScheduler: before token t starts at time `now`,

    optional_t <= due_t - now - mandatory

and the token's layer schedule gets the deadline  mandatory + optional_t.  If the true mandatory
time of every token is at most `mandatory` and `mandatory <= period`, no token ever stalls:
token t finishes by due_t, so token t+1 starts with at least period - mandatory >= 0 of room.

How much of that room to spend is the policy. The price policy buys flash neurons while their
marginal quality is worth their current price: with K(n) the expected KL after fetching n
neurons (a convex calibration curve) and tau the current flash time per neuron, it fetches the n
that minimises K(n) + lam * tau * n. lam is chosen once so that the average time per token equals
the period, so slow-flash tokens borrow time that fast-flash tokens bank.
"""
import numpy as np


def convex_hull_curve(n_points, k_points, size=801):
    """Lower convex hull of a measured K(n) curve on a fine grid, for a well-defined price rule."""
    grid = np.linspace(0, float(n_points[-1]), size)
    kg = np.interp(grid, n_points, k_points)
    hull = [0]
    for i in range(1, size):
        while len(hull) >= 2:
            i0, i1 = hull[-2], hull[-1]
            if (kg[i1] - kg[i0]) * (grid[i] - grid[i0]) >= (kg[i] - kg[i0]) * (grid[i1] - grid[i0]):
                hull.pop()
            else:
                break
        hull.append(i)
    return grid, np.interp(grid, grid[hull], kg[hull])


class PriceRule:
    def __init__(self, n_points, k_points):
        self.grid, self.k = convex_hull_curve(np.asarray(n_points, float), np.asarray(k_points, float))

    def neurons(self, lam, tau):
        tau = np.atleast_1d(np.asarray(tau, float))
        return self.grid[np.argmin(self.k[None, :] + lam * tau[:, None] * self.grid[None, :], axis=1)]

    def fit_lambda(self, taus, time_per_token):
        """lam such that the mean of n(lam, tau) * tau over `taus` equals `time_per_token`."""
        taus = np.asarray(taus, float)
        lo, hi = -40.0, 10.0
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if (self.neurons(np.exp(mid), taus) * taus).mean() > time_per_token else (lo, mid)
        return float(np.exp((lo + hi) / 2))


class TokenStreamScheduler:
    def __init__(self, period, mandatory, startup=0.0, rule=None, lam=None):
        if mandatory > period:
            raise ValueError("mandatory time per token exceeds the consumption period: stalls are unavoidable")
        self.P, self.M, self.startup = float(period), float(mandatory), float(startup)
        self.rule, self.lam = rule, lam
        self.start = None

    def begin(self, now):
        self.start = float(now)

    def due(self, t):
        return self.start + self.startup + self.M + t * self.P

    def optional_budget(self, t, now, tau=None):
        """Optional (flash) time token t may use if it starts at `now`; never violates the reservation."""
        room = max(0.0, self.due(t) - now - self.M)
        if self.rule is None or tau is None:
            return room
        return min(room, float(self.rule.neurons(self.lam, tau)[0]) * tau)

    def token_deadline(self, t, now, tau=None):
        """Relative deadline to hand to the layer-level ReservationScheduler for token t."""
        return self.M + self.optional_budget(t, now, tau)
