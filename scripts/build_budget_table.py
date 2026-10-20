"""Build the per-layer budget table for a model (step 2 of AnytimeStream).

Measures each layer's fragility (the damage law KL ~ G_l * rho_l^2), then, at each budget, finds
the per-layer shares a single cost-weighted price gives. In flash mode the 10% most often used
neurons per layer are treated as resident and cost nothing, so the budget counts flash reads only.

usage:
  python scripts/build_budget_table.py --model models/gutenberg-12L --calib-file data/gutenberg/austen-persuasion.txt --out results/budget_table_12L.json
  python scripts/build_budget_table.py --model Qwen/Qwen2.5-0.5B --out results/budget_table_qwen05.json   (WikiText-2)
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "scripts"))
import price_lab as P  # noqa: E402
from anytimestream import BudgetTable  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--calib-file", default=None, help="text file; default WikiText-2 train")
ap.add_argument("--calib-seqs", type=int, default=16)
ap.add_argument("--seq-len", type=int, default=256)
ap.add_argument("--resident", type=float, default=0.1, help="share of neurons per layer kept in RAM")
ap.add_argument("--budgets", default="0.05,0.1,0.15,0.2,0.3,0.4,0.5")
ap.add_argument("--device", default=None)
ap.add_argument("--out", required=True)
args = ap.parse_args()

t0 = time.time()
torch.manual_seed(0)
model, tok, device = P.load(args)
calib = P.token_chunks(tok, args.calib_file, "train", args.calib_seqs, args.seq_len)
lab = P.Lab(model, list(calib.split(4)), device)
L, d_ff = lab.L, lab.d_ff
gains, slopes, orth, r2 = P.fit_law(lab)
lab.gain = torch.tensor(gains, device=device, dtype=torch.float32)
samples, sample_ids, hot = lab.dense_stats()
resident = torch.zeros(L, d_ff, dtype=torch.bool)
resident.scatter_(1, hot.topk(int(round(args.resident * d_ff)), dim=1).indices, True)
is_res = [resident[l][sample_ids[l]] for l in range(L)]

budgets = [float(b) for b in args.budgets.split(",")]
meta = {"gains": gains, "slopes": slopes, "law_r2": r2, "resident": args.resident, "calib_tokens": calib.numel(),
        "d_ff": d_ff}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
for mode, exclude in (("compute", None), ("flash", is_res)):
    rows = [P.static_split(samples, gains, b, exclude=exclude) for b in budgets]
    t = BudgetTable(budgets, rows, mode=mode, model=args.model, meta=meta)
    path = args.out if mode == "flash" else args.out.replace(".json", "_compute.json")
    t.save(path)
    print(f"{mode} table -> {path}")
    for b, r in zip(budgets, rows):
        print(f"  {b:4.0%}: " + " ".join(f"{x:.2f}" for x in r))
print(f"law R^2 {r2:.3f}, {time.time() - t0:.0f}s")
