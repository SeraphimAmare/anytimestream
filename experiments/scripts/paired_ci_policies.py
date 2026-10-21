"""Paired bootstrap 95% intervals for the change in mean (and p90 / p99) KL between two policies.

Resamples evaluation sequences (not tokens), since tokens in a sequence are correlated.
usage: paired_ci_policies.py PER_TOKEN_KL.pt SEQ_LEN BASE OTHER [OTHER...]
"""
import sys

import numpy as np
import torch

path, T, base, others = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
d = torch.load(path)
d = d.get("kl", d)
b = d[base].numpy().reshape(-1, T)
rng = np.random.default_rng(0)
idx = rng.integers(0, b.shape[0], (2000, b.shape[0]))
for o in others:
    x = d[o].numpy().reshape(-1, T)
    for name, f in (("mean", np.mean), ("p90", lambda a: np.quantile(a, 0.9)), ("p99", lambda a: np.quantile(a, 0.99))):
        point = 1 - f(x) / f(b)
        boot = np.array([1 - f(x[i]) / f(b[i]) for i in idx])
        lo, hi = np.quantile(boot, [0.025, 0.975])
        print(f"{o:>15} vs {base}: {name:>4} KL reduction {point:+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]")
