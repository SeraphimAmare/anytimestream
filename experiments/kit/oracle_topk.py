"""Phase 2: the quality ceiling of sparse FFNs, and which layers are most sensitive.

Mode 1 (always): keep the true top-k fraction of neurons in EVERY layer (an oracle that knows
the real activations) and measure quality vs the dense model. This is the 'performance profile'
of the anytime computation: quality as a function of how much optional work you complete.

Mode 2 (--sensitivity): sparsify ONE layer at a time (others dense) and record the damage.
Normalized, these become per-layer weights for the 'weighted' policy in anytime_sim.py.

Outputs (in --out): oracle_curve.csv/.png, and with --sensitivity: sensitivity.json/.png

Example:
  python oracle_topk.py --model Qwen/Qwen2.5-1.5B --sensitivity
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from common import (compare_logits, decoder_layers, down_col_norms, eval_batches, ffn_hidden,
                    load_model, neuron_scores, pick_device, swap_mlps)


class OracleTopK(nn.Module):
    """Computes only the true top-`frac` neurons of each token (the ideal activation pattern)."""

    def __init__(self, mlp, frac):
        super().__init__()
        self.mlp, self.frac = mlp, frac
        self.col = down_col_norms(mlp)

    def forward(self, x):
        h = ffn_hidden(self.mlp, x)
        if self.frac >= 1.0:
            return self.mlp.down_proj(h)
        k = max(1, round(self.frac * h.shape[-1]))
        idx = neuron_scores(h, self.col).topk(k, dim=-1).indices
        return self.mlp.down_proj(h * torch.zeros_like(h).scatter_(-1, idx, 1))


@torch.no_grad()
def evaluate(model, batches, refs, device, make):
    kl, agree, nll, n = 0.0, 0.0, 0.0, 0
    with swap_mlps(model, make):
        for b, ref in zip(batches, refs):
            ids = b.to(device)
            m = compare_logits(ref.to(device), model(input_ids=ids, use_cache=False).logits, ids)
            kl += m["kl"].sum().item()
            agree += m["agree"].sum().item()
            nll += m["nll"].sum().item()
            n += m["kl"].numel()
    return {"kl": kl / n, "top1_agree": agree / n, "ppl": math.exp(nll / (n - len(batches)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--fracs", default="0.05,0.1,0.15,0.2,0.3,0.5,1.0")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--sens-frac", type=float, default=0.1, help="fraction kept in the single sparsified layer")
    ap.add_argument("--n-seqs", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/oracle")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)
    model, tok = load_model(args.model, device)
    batches = eval_batches(tok, args.n_seqs, args.seq_len, args.text_file, model.config.vocab_size)
    with torch.no_grad():  # dense reference logits, kept on CPU in half precision to save memory
        refs = [model(input_ids=b.to(device), use_cache=False).logits.to(torch.float16).cpu() for b in batches]

    fracs = [float(f) for f in args.fracs.split(",")]
    rows = []
    for f in fracs:
        r = evaluate(model, batches, refs, device, lambda i, m, f=f: OracleTopK(m, f))
        rows.append({"frac": f, **r})
        print(f"keep {f:5.0%} of neurons in every layer: KL={r['kl']:.4f}  top1={r['top1_agree']:.3f}  ppl={r['ppl']:.3f}")
    with open(os.path.join(args.out, "oracle_curve.csv"), "w") as fh:
        fh.write("frac,kl,top1_agree,ppl\n")
        for r in rows:
            fh.write(f"{r['frac']},{r['kl']},{r['top1_agree']},{r['ppl']}\n")

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot([r["frac"] for r in rows], [r["kl"] for r in rows], "o-", label="KL to dense")
    ax1.set_xlabel("fraction of FFN neurons computed (oracle top-k, every layer)")
    ax1.set_ylabel("mean per-token KL(dense || sparse)")
    ax1.set_yscale("log")
    ax2 = ax1.twinx()
    ax2.plot([r["frac"] for r in rows], [r["top1_agree"] for r in rows], "s--", color="tab:orange")
    ax2.set_ylabel("top-1 agreement with dense", color="tab:orange")
    ax1.set_title("Performance profile: quality vs optional work done")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "oracle_curve.png"), dpi=150)

    if args.sensitivity:
        L = len(decoder_layers(model))
        dkl = []
        for l in range(L):
            r = evaluate(model, batches, refs, device,
                         lambda i, m, l=l: OracleTopK(m, args.sens_frac) if i == l else m)
            dkl.append(r["kl"])
            print(f"  only layer {l:2d} sparsified to {args.sens_frac:.0%}: KL={r['kl']:.5f}")
        mean = sum(dkl) / L
        weights = [max(d, 1e-9) / max(mean, 1e-12) for d in dkl]  # normalized to mean 1
        with open(os.path.join(args.out, "sensitivity.json"), "w") as fh:
            json.dump({"model": args.model, "sens_frac": args.sens_frac, "delta_kl": dkl, "weights": weights}, fh, indent=1)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(range(L), weights)
        ax.set_xlabel("layer")
        ax.set_ylabel("relative sensitivity (mean = 1)")
        ax.set_title(f"Damage from sparsifying one layer to {args.sens_frac:.0%}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "sensitivity.png"), dpi=150)
        print(f"Wrote {args.out}/sensitivity.json (feed it to anytime_sim.py --weights)")


if __name__ == "__main__":
    main()
