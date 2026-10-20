"""Follow-up checks on price_lab.py results.

1. Single-layer damage: KL when only one layer keeps 20% of its neurons (an intuitive sensitivity measure).
2. Where the static split moves the budget: per-layer shares at each budget.
3. Correlation-corrected gains: dropped pieces add up larger than their squares suggest, by a
   layer-dependent factor; fold that factor into the gains and re-split.
4. The per-token oracle at exactly the same average budget (the main run let it overshoot slightly).
5. Layer weights for the deadline simulator (anytime_sim.py --weights).

usage: followup.py MODEL CALIB_FILE EVAL_FILE PRICE_LAB_DIR OUT_DIR   (use "wikitext" for both files with a
Hugging Face model: WikiText-2 train calibrates, test evaluates)
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import price_lab as P


class A:
    pass


model_path, calib_file, eval_file, lab_dir, out_dir = sys.argv[1:6]
os.makedirs(out_dir, exist_ok=True)
torch.set_num_threads(2)
torch.manual_seed(0)
prev = json.load(open(os.path.join(lab_dir, "results.json")))
args = A()
args.model, args.device = model_path, None
model, tok, device = P.load(args)
pick = lambda f: None if f == "wikitext" else f
n_cal, n_ev = prev["calib_tokens"] // 256, prev["eval_tokens"] // 256
calib = P.token_chunks(tok, pick(calib_file), "train", n_cal, 256)
evalx = P.token_chunks(tok, pick(eval_file), "test", n_ev, 256)
split = lambda x: list(x.split(4))
lab = P.Lab(model, split(calib), device)
L, d_ff = lab.L, lab.d_ff
gains, orth = prev["law"]["gains"], prev["law"]["orth_ratio"]
lab.gain = torch.tensor(gains, dtype=torch.float32, device=device)
samples, sample_ids, hot = lab.dense_stats()
resident = torch.zeros(L, d_ff, dtype=torch.bool)
resident.scatter_(1, hot.topk(int(round(0.1 * d_ff)), dim=1).indices, True)
is_res = [resident[l][sample_ids[l]] for l in range(L)]
lab.set_data(split(evalx))
out = {"eval_seqs": n_ev}

single = []
for l in range(L):
    single.append(lab.run({l: {"kind": "topk", "frac": [0.2] * L}})["kl"].mean().item())
out["single_layer_kl_at_20pct"] = single
print("single-layer KL at 20%:", " ".join(f"{x:.4f}" for x in single))

gains_c = [g * o for g, o in zip(gains, orth)]
rows, shares_out = [], {}
for B in (0.1, 0.2, 0.3):
    sh = P.static_split(samples, gains, B)
    sh_c = P.static_split(samples, gains_c, B)
    shares_out[f"compute_{B}"] = list(sh)
    shares_out[f"compute_corrected_{B}"] = list(sh_c)
    ru = lab.run({"kind": "topk", "frac": [B] * L})
    rs = lab.run({"kind": "topk", "frac": list(sh)})
    rc = lab.run({"kind": "topk", "frac": list(sh_c)})
    lo, hi = 0.6 * B, 1.1 * B
    for _ in range(9):
        mid = (lo + hi) / 2
        ro = lab.run({"kind": "thr_tok", "tok_lams": lab.token_prices(mid)})
        got = ro["kept"].mean().item()
        if abs(got - B) < 0.0015:
            break
        lo, hi = (lo, mid) if got > B else (mid, hi)
    for name, r in (("uniform", ru), ("static split", rs), ("static split, corrected gains", rc),
                    ("per-token oracle, matched", ro)):
        rows.append(P.summarize(r, name, B))
    base = rows[-4]["kl"]
    for r in rows[-4:]:
        print(f"{B:.0%} {r['policy']:>30}: KL {r['kl']:.4f} ({1 - r['kl'] / base:+.1%} vs uniform)  kept {r['kept']:.3f} max {r['kept_max']:.3f}")
out["compute"] = rows

sh_f = P.static_split(samples, gains, 0.1, exclude=is_res)
shares_out["flash_0.1"] = list(sh_f)
out["shares"] = shares_out
w = np.array(sh_f) / np.mean(sh_f)
json.dump({"model": model_path, "weights": list(w), "source": "static split of flash fetches at a 10% budget"},
          open(os.path.join(out_dir, "layer_weights.json"), "w"), indent=1)
print("flash 10% shares:", " ".join(f"{x:.3f}" for x in sh_f))
print("compute 20% shares:", " ".join(f"{x:.3f}" for x in shares_out["compute_0.2"]))
json.dump(out, open(os.path.join(out_dir, "followup.json"), "w"), indent=1)
torch.save({"seq_len": 256, "kl": P.PER_TOKEN}, os.path.join(out_dir, "per_token_kl.pt"))
print("wrote", out_dir)
