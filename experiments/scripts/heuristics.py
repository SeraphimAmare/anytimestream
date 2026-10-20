"""Is the calibrated split better than simple rules? Compare against two heuristics at equal compute:
'protect ends' (first and last layer get twice the uniform share) and 'damage-proportional' (shares
proportional to the KL when only that layer is sparsified to 20%).

usage: heuristics.py MODEL CALIB_FILE EVAL_FILE FOLLOWUP_DIR   ("wikitext" works as with followup.py)
"""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import price_lab as P

class A: pass
model_path, calib_file, eval_file, fdir = sys.argv[1:5]
torch.set_num_threads(2)
fu = json.load(open(os.path.join(fdir, "followup.json")))
args = A(); args.model, args.device = model_path, None
model, tok, device = P.load(args)
evalx = P.token_chunks(tok, None if eval_file == "wikitext" else eval_file, "test", fu.get("eval_seqs", 32), 256)
lab = P.Lab(model, list(evalx.split(4)), device)
L = lab.L
damage = np.array(fu["single_layer_kl_at_20pct"])
rows = []
for B in (0.1, 0.2, 0.3):
    ends = np.ones(L); ends[0] = ends[-1] = 2.0
    ends = np.minimum(ends * B * L / ends.sum(), 1.0)
    dmg = np.minimum(damage / damage.sum() * B * L, 1.0)
    sqrt_dmg = np.sqrt(damage); sqrt_dmg = np.minimum(sqrt_dmg / sqrt_dmg.sum() * B * L, 1.0)
    plans = {"uniform": [B] * L, "static split": fu["shares"][f"compute_{B}"], "protect ends": list(ends),
             "damage-proportional": list(dmg), "sqrt damage-proportional": list(sqrt_dmg)}
    base = None
    for name, frac in plans.items():
        r = lab.run({"kind": "topk", "frac": [float(x) for x in frac]})
        s = P.summarize(r, name, B); rows.append(s)
        base = s["kl"] if name == "uniform" else base
        print(f"{B:.0%} {name:>26}: KL {s['kl']:.4f} ({1 - s['kl'] / base:+.1%} vs uniform)  kept {s['kept']:.3f}")
json.dump(rows, open(os.path.join(fdir, "heuristics.json"), "w"), indent=1)
torch.save({"seq_len": 256, "kl": P.PER_TOKEN}, os.path.join(fdir, "heuristics_per_token_kl.pt"))
