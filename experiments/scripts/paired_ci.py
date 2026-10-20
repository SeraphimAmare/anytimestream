"""Paired comparisons with 95% bootstrap intervals over evaluation sequences.

usage: paired_ci.py RESULTS_DIR
"""
import os
import sys

import numpy as np
import torch

d = torch.load(os.path.join(sys.argv[1], "per_token_kl.pt"))
T, kl = d["seq_len"], d["kl"]
rng = np.random.default_rng(0)


def per_seq(x):
    return x.view(-1, T).mean(1).numpy()


def compare(a, b, n=2000):
    """Relative KL reduction of policy a against policy b (positive = a is better), with a 95% interval."""
    sa, sb = per_seq(kl[a]), per_seq(kl[b])
    point = 1 - sa.mean() / sb.mean()
    idx = rng.integers(0, len(sa), (n, len(sa)))
    boot = 1 - sa[idx].mean(1) / sb[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, lo, hi


pairs = []
for key in kl:
    sec, budget, policy = key.split("|")
    if policy != "uniform":
        base = f"{sec}|{budget}|uniform"
        if base in kl:
            pairs.append((key, base))
    if policy.startswith("price") or policy.startswith("capped") or policy == "per-token oracle":
        st = f"{sec}|{budget}|static split"
        if st in kl:
            pairs.append((key, st))
print(f"{'comparison':<62} {'KL reduction':>12} {'95% interval':>18}")
for a, b in pairs:
    p, lo, hi = compare(a, b)
    print(f"{a + '  vs  ' + b.split('|')[2]:<62} {p:+12.1%}   [{lo:+.1%}, {hi:+.1%}]")
