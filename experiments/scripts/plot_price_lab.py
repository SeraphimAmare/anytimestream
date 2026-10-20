"""Plot price_lab.py results: layer gains, the damage law, and policy comparisons.

usage: plot_price_lab.py RESULTS_DIR
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1]
r = json.load(open(os.path.join(d, "results.json")))
law = r["law"]
L = r["layers"]

fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
ax = axes[0]
ax.semilogy(range(L), law["gains"], "o-")
ax.set(xlabel="layer", ylabel="gain G_l (KL per unit rho^2)", title=f"Layer gains (law R^2 {law['r2']:.2f})")
ax2 = ax.twinx()
ax2.plot(range(L), law["slopes"], "s--", color="tab:orange")
ax2.axhline(2, color="tab:orange", lw=0.6)
ax2.set_ylabel("fitted slope (theory 2)", color="tab:orange")

for ax, key, cost in ((axes[1], "compute", "kept"), (axes[2], "flash", "fetch")):
    rows = r[key]
    base = {x["budget"]: x["kl"] for x in rows if x["policy"] == "uniform"}
    for name in dict.fromkeys(x["policy"] for x in rows):
        pts = [x for x in rows if x["policy"] == name]
        ax.plot([x["budget"] for x in pts], [100 * (1 - x["kl"] / base[x["budget"]]) for x in pts], "o-", label=name)
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel=f"budget (share of neurons {'computed' if key == 'compute' else 'fetched from flash'})",
           ylabel="KL reduction vs uniform (%)", title=key)
    ax.legend(fontsize=8)
fig.suptitle(r["model"])
fig.tight_layout()
fig.savefig(os.path.join(d, "price_lab.png"), dpi=130)
print("saved", os.path.join(d, "price_lab.png"))
