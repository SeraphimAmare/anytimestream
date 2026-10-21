"""residency.py: which neurons should live in RAM? (designing the mandatory part)

In a deadline-guaranteed runtime the RAM-resident neurons are the mandatory part: they are always
computed, never wait on flash, and set the quality floor. This compares ways to pick them at the
same total RAM capacity (a share of all FFN neurons):

  uniform-freq    each layer keeps its most frequently active neurons, same count per layer
                  (a fixed per-layer cache budget, as in LeanStream's and PowerInfer-2's evaluations)
  global-freq     most frequently active neurons pooled across layers (activation-frequency impact)
  uniform-damage  same count per layer, ranked by expected damage G_l * E[(s/||r||)^2]
  global-damage   ranked by expected damage pooled across layers (ours: minimises the predicted
                  KL of the mandatory-only model under the additive damage law)

Each is evaluated mandatory-only and with a flash budget (neurons fetched per token), the flash
reads split either uniformly or with the cache-aware static split that excludes resident neurons.

usage: residency.py --model M [--calib-file a.txt --eval-file b.txt] --law results.json --out DIR
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import price_lab as P  # noqa: E402


class Acc:
    """Accumulates per-neuron sums of the captured value, layer by layer, across forward passes."""

    def __init__(self, L, d_ff):
        self.L, self.sum, self.n, self.i = L, torch.zeros(L, d_ff, dtype=torch.float64), 0, 0

    def append(self, x):
        l = self.i % self.L
        self.sum[l] += x.double().sum(0).cpu()
        if l == 0:
            self.n += x.shape[0]
        self.i += 1


def neuron_damage(lab):
    acc = Acc(lab.L, lab.d_ff)
    lab.store, lab.plan, lab.collect, lab.capture = acc, {}, False, True
    with torch.no_grad():
        for b in lab.batches:
            lab._logits(b)
    lab.capture = False
    del lab.store
    return (acc.sum / acc.n).float()


def top_mask(score, count_per_layer=None, total=None):
    L, d = score.shape
    m = torch.zeros(L, d, dtype=torch.bool)
    if count_per_layer is not None:
        m.scatter_(1, score.topk(count_per_layer, dim=1).indices, True)
    else:
        flat = score.flatten()
        m.view(-1)[flat.topk(total).indices] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib-file", default=None)
    ap.add_argument("--eval-file", default=None)
    ap.add_argument("--calib-seqs", type=int, default=16)
    ap.add_argument("--eval-seqs", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--law", required=True, help="results.json from price_lab.py (layer gains)")
    ap.add_argument("--capacities", default="0.05,0.1,0.2")
    ap.add_argument("--flash", default="0,0.05,0.1")
    ap.add_argument("--strategies", default="uniform-freq,global-freq,uniform-damage,global-damage")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4, help="sequences per forward pass")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    model, tok, device = P.load(args)
    calib = P.token_chunks(tok, args.calib_file, "train", args.calib_seqs, args.seq_len)
    evalx = P.token_chunks(tok, args.eval_file, "test", args.eval_seqs, args.seq_len)
    lab = P.Lab(model, list(calib.split(args.batch)), device)
    L, d_ff = lab.L, lab.d_ff
    gains = json.load(open(args.law))["law"]["gains"]
    lab.gain = torch.tensor(gains, dtype=torch.float32, device=device)

    samples, sample_ids, hot = lab.dense_stats()
    dmg = neuron_damage(lab)
    lab.set_data(list(evalx.split(args.batch)))
    print(f"{args.model}: {L} layers x {d_ff}; calib {calib.numel()} tok, eval {evalx.numel()} tok; stats {time.time() - t0:.0f}s",
          flush=True)

    rows, layer_counts = [], {}
    for C in [float(c) for c in args.capacities.split(",")]:
        total, per = int(round(C * L * d_ff)), int(round(C * d_ff))
        masks = {"uniform-freq": top_mask(hot, per), "global-freq": top_mask(hot, total=total),
                 "uniform-damage": top_mask(dmg, per), "global-damage": top_mask(dmg, total=total)}
        for share in (0.25, 0.5):                      # hybrid: part of RAM by damage, the rest by frequency
            dm = top_mask(dmg, total=int(round(share * total)))
            fr = hot.masked_fill(dm, -1.0)
            masks[f"hybrid-{share:g}"] = dm | top_mask(fr, int(round((1 - share) * per)))
        masks = {k: v for k, v in masks.items() if k in args.strategies.split(",")}
        for name, mask in masks.items():
            layer_counts[f"{C}|{name}"] = (mask.float().mean(1)).tolist()
            lab.resident = mask.to(device)
            is_res = [mask[l][sample_ids[l]] for l in range(L)]
            for f in [float(x) for x in args.flash.split(",")]:
                splits = {"uniform": [f] * L} if f > 0 else {"none": [0.0] * L}
                if f > 0:
                    splits["cache-aware"] = list(P.static_split(samples, gains, f, exclude=is_res))
                for sname, frac in splits.items():
                    r = lab.run({"kind": "res_topk", "frac": frac})
                    row = {"capacity": C, "resident": name, "flash": f, "split": sname, "kl": r["kl"].mean().item(),
                           "kl_p90": r["kl"].quantile(0.9).item(), "top1": r["agree"].mean().item(),
                           "fetch": r["fetch"].mean().item(), "kept": r["kept"].mean().item()}
                    rows.append(row)
                    P.PER_TOKEN[f"{C}|{name}|{f}|{sname}"] = r["kl"].clone()
                    print(f"C={C:.2f} {name:>15} flash={f:.2f} {sname:>11}: KL {row['kl']:.4f}  top1 {row['top1']:.3f}  "
                          f"fetch {row['fetch']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump({"model": args.model, "rows": rows, "layer_counts": layer_counts, "gains": gains,
               "eval_tokens": evalx.numel()}, open(os.path.join(args.out, "residency.json"), "w"), indent=1)
    torch.save({"kl": P.PER_TOKEN, "seq_len": args.seq_len}, os.path.join(args.out, "per_token_kl.pt"))
    print("wrote", args.out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
