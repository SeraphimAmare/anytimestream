"""Per-layer budget table: how a model's spare FFN budget should be split across its layers.

Built once per model (scripts/build_budget_table.py) from a measured fragility profile. Each row
maps an average budget (share of neurons, or share of non-resident neurons fetched from flash)
to per-layer shares. At run time the scheduler only needs `weights(budget)`.
"""
import json

import numpy as np


class BudgetTable:
    def __init__(self, budgets, shares, mode="flash", model="", meta=None):
        order = np.argsort(budgets)
        self.budgets = np.asarray(budgets, dtype=float)[order]
        self.shares = np.asarray(shares, dtype=float)[order]
        if self.shares.ndim != 2 or len(self.budgets) != len(self.shares):
            raise ValueError("shares must be one row of per-layer values per budget")
        self.mode, self.model, self.meta = mode, model, meta or {}

    @property
    def layers(self):
        return self.shares.shape[1]

    def lookup(self, budget):
        """Per-layer shares at `budget`, linearly interpolated between rows, clamped at the ends."""
        b = float(np.clip(budget, self.budgets[0], self.budgets[-1]))
        return np.array([np.interp(b, self.budgets, self.shares[:, l]) for l in range(self.layers)])

    def weights(self, budget):
        """Shares normalised to mean 1: the form ReservationScheduler takes."""
        s = self.lookup(budget)
        return s / s.mean()

    def save(self, path):
        json.dump({"model": self.model, "mode": self.mode, "budgets": self.budgets.tolist(),
                   "shares": self.shares.tolist(), "meta": self.meta}, open(path, "w"), indent=1)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(d["budgets"], d["shares"], d.get("mode", "flash"), d.get("model", ""), d.get("meta"))
