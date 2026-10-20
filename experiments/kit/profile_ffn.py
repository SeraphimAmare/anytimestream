"""Phase 1: how concentrated is the FFN's work, and which neurons are 'hot'?

Outputs (in --out):
  freq.pt            per-layer activation frequency of every neuron (used by anytime_sim.py to pick pinned neurons)
  summary.json       numbers behind the plots
  concentration.png  relative FFN-output error vs fraction of neurons kept (per layer)
  frequency.png      sorted activation frequency for a few layers (flat = no hot set, steep = strong hot set)

Example:
  python profile_ffn.py --model Qwen/Qwen2.5-1.5B --n-seqs 8 --seq-len 256
  python profile_ffn.py --model tiny            # smoke test, no download
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from common import collect_ffn_stats, eval_batches, load_model, pick_device

KEEP_FRACS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--n-seqs", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--active-frac", type=float, default=0.2,
                    help="a neuron counts as 'active' for a token if it is in that token's top fraction by score")
    ap.add_argument("--text-file", default=None, help="use your own text instead of WikiText-2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/profile")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)
    model, tok = load_model(args.model, device)
    batches = eval_batches(tok, args.n_seqs, args.seq_len, args.text_file, model.config.vocab_size)

    freq, err = collect_ffn_stats(model, batches, device, args.active_frac, KEEP_FRACS)
    L, d_ff = freq.shape
    torch.save({"freq": freq, "active_frac": args.active_frac, "model": args.model}, os.path.join(args.out, "freq.pt"))

    # How much of the active set is predictable from frequency alone? If you pinned the
    # most frequent 10% of neurons, what share of all (token, active-neuron) pairs would hit?
    sorted_f, _ = freq.sort(dim=1, descending=True)
    pin10 = max(1, round(0.1 * d_ff))
    pinned_hit = (sorted_f[:, :pin10].sum(1) / freq.sum(1)).tolist()

    summary = {
        "model": args.model, "layers": L, "d_ff": d_ff, "active_frac": args.active_frac,
        "keep_fracs": list(KEEP_FRACS),
        "rel_err_per_layer": err.tolist(),
        "rel_err_mean": err.mean(0).tolist(),
        "pinned_top10pct_hit_rate_per_layer": pinned_hit,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print(f"\n{args.model}: {L} layers, d_ff={d_ff}")
    print("Mean relative FFN-output error when keeping only the top fraction of neurons (by |h|*||w_down||):")
    for f_, e in zip(KEEP_FRACS, err.mean(0).tolist()):
        print(f"  keep {f_:4.0%}: {e:.3f}")
    print(f"Pinning the 10% most frequently active neurons covers on average {sum(pinned_hit)/L:.0%} "
          f"of active-neuron occurrences (min layer {min(pinned_hit):.0%}, max {max(pinned_hit):.0%}).")

    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = plt.get_cmap("viridis")
    for l in range(L):
        ax.plot(KEEP_FRACS, err[l].tolist(), color=cmap(l / max(1, L - 1)), lw=0.8, alpha=0.7)
    ax.plot(KEEP_FRACS, err.mean(0).tolist(), color="black", lw=2, label="mean over layers")
    ax.set_xlabel("fraction of FFN neurons kept (top by score)")
    ax.set_ylabel("relative FFN output error")
    ax.set_title("Concentration (dark = early layers, yellow = late)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "concentration.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    for l in sorted({0, L // 4, L // 2, (3 * L) // 4, L - 1}):
        ax.plot(torch.linspace(0, 1, d_ff).tolist(), sorted_f[l].tolist(), label=f"layer {l}")
    ax.set_xlabel("neurons, sorted by activation frequency (fraction of d_ff)")
    ax.set_ylabel("fraction of tokens where neuron is active")
    ax.set_title("Hot neurons")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "frequency.png"), dpi=150)
    print(f"Wrote {args.out}/freq.pt, summary.json, concentration.png, frequency.png")


if __name__ == "__main__":
    main()
