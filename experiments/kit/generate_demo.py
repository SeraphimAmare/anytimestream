"""See it: generate from the same prompt with the dense model and with the simulated anytime
runtime under a per-token deadline (free-running, so errors can compound across tokens).

Example:
  python generate_demo.py --model Qwen/Qwen2.5-1.5B --prompt "The key idea of real-time scheduling is" \
      --weights results/oracle/sensitivity.json --policy weighted --deadline-ms 20
"""
import argparse
import json
import math

import torch

from anytime_sim import HW, AnytimeSim, derive_costs, make_policy, pinned_from_freq
from common import collect_ffn_stats, load_model, model_dims, pick_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--prompt", default="The key idea of real-time scheduling is")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--deadline-ms", type=float, default=None, help="default: 2x the minimum feasible deadline")
    ap.add_argument("--policy", default="slack")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--k-frac", type=float, default=0.2)
    ap.add_argument("--pinned-frac", type=float, default=0.1)
    ap.add_argument("--cache-frac", type=float, default=0.1)
    ap.add_argument("--pred-noise", type=float, default=0.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    model, tok = load_model(args.model, device)
    dims = model_dims(model.config)
    costs = derive_costs(HW(), dims)
    L, nb = dims["n_layers"], dims["d_ff"]
    Kb = max(1, round(args.k_frac * nb))
    m = costs["attn"] + costs["kappa_neuron"] * 3 * Kb + costs["kappa_block"]
    d_min = L * m + costs["head"]
    D = args.deadline_ms / 1e3 if args.deadline_ms else 2 * d_min

    if tok is not None:
        ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
    else:
        ids = torch.randint(0, dims["vocab"], (1, 16), device=device)
    freq = torch.load(args.profile)["freq"] if args.profile else collect_ffn_stats(model, [ids.cpu()], device, args.k_frac)[0]
    pinned = pinned_from_freq(freq, 1, nb, args.pinned_frac)
    weights = json.load(open(args.weights))["weights"] if args.weights else None
    policy = make_policy(args.policy, L, D - costs["head"], m, weights)

    gen = dict(max_new_tokens=args.max_new_tokens, do_sample=False, attention_mask=torch.ones_like(ids))
    if tok is not None:
        gen["pad_token_id"] = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        dense = model.generate(ids, **gen)
        sim = AnytimeSim(model, costs, policy, D, args.k_frac, pinned, args.cache_frac, 1, args.pred_noise)
        sim.persist_cache = True  # each generate() step is one token; the DRAM cache must carry over
        with sim.attach(model):
            anytime = model.generate(ids, **gen)

    show = (lambda x: tok.decode(x[0, ids.shape[1]:], skip_special_tokens=True)) if tok else (lambda x: x[0, ids.shape[1]:].tolist())
    s = sim.summary()
    print(f"\nDeadline {D*1e3:.2f} ms/token (minimum feasible {d_min*1e3:.2f}), policy '{args.policy}'")
    print(f"\n--- dense ---\n{show(dense)}\n\n--- anytime ---\n{show(anytime)}\n")
    print(f"misses {s['miss_rate']:.1%}, p50 {s['lat_p50_ms']:.2f} ms, max {s['lat_max_ms']:.2f} ms, "
          f"coverage of true top-k mass {s['coverage']:.3f} (prompt tokens included)")


if __name__ == "__main__":
    main()
