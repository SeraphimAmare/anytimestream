"""Demo: waiting for every weight vs the reservation scheduler, under heavy-tailed flash latency.

Uses the layer weights from a budget table if given. Timings are illustrative, not measured.
usage: python scripts/demo_guarantee.py [--table results/budget_table_12L.json] [--out results/demo_guarantee.png]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anytimestream import BudgetTable, ReservationScheduler, run_token  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--table", default=None)
ap.add_argument("--budget", type=float, default=0.1)
ap.add_argument("--tokens", type=int, default=2000)
ap.add_argument("--out", default="results/demo_guarantee.png")
args = ap.parse_args()

L, n_opt = 12, 70                                  # 12 layers; 10% of a 768-neuron layer to fetch
m, kappa, head = np.full(L, 60e-6), 2e-6, 30e-6     # seconds: mandatory per layer, compute per neuron, LM head
w = BudgetTable.load(args.table).weights(args.budget) if args.table else None
rng = np.random.default_rng(0)

wait_all, reserved, kept = [], {}, {}
for D in (1.0e-3, 2.0e-3, 4.0e-3, 6.0e-3):
    s = ReservationScheduler(m, D, w, kappa=kappa, head=head)
    ends, done = [], []
    for _ in range(args.tokens):
        arr = [rng.lognormal(np.log(80e-6), 0.6, n_opt) for _ in range(L)]
        tr = run_token(s, m, arr)
        ends.append(tr.end)
        done.append(sum(x.optional_done for x in tr.layers) / (L * n_opt))
        if D == 1.0e-3:
            wait_all.append(sum(m) + head + sum(a.max() + n_opt * kappa for a in arr))
    reserved[D], kept[D] = np.array(ends), float(np.mean(done))

wait_all = np.array(wait_all)
print(f"{'deadline':>9} | {'wait for all: missed':>21} | {'reservation: missed':>20} | optional work done")
for D in reserved:
    print(f"{D * 1e3:7.1f}ms | {np.mean(wait_all > D):21.1%} | {np.mean(reserved[D] > D + 1e-12):20.1%} | {kept[D]:.1%}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(wait_all * 1e3, bins=60, alpha=0.6, label="wait for every weight")
    ax.hist(reserved[2.0e-3] * 1e3, bins=30, alpha=0.8, label="reservation, 2 ms deadline")
    ax.axvline(2.0, color="k", ls="--", lw=1)
    ax.set(xlabel="time per token (ms, illustrative)", ylabel="tokens")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print("saved", args.out)
except ImportError:
    pass
