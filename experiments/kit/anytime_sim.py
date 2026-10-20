"""Phases 3-4: co-simulate a deadline-aware, anytime sparse-FFN runtime.

The real model runs in PyTorch, so every quality number is exact. Alongside it, a virtual
clock models flash I/O and GPU time for each generated token, layer by layer, and decides
which FFN neurons would have arrived before that layer's cutoff. Neurons that miss the
cutoff are simply not computed (their term in the FFN sum is dropped).

Timeline per token and layer l (see the diagram in our chat):
  - stale phase   : while layer l-1's FFN runs, flash fetches layer l's blocks ranked by a
                    prediction made from the residual *before* layer l-1's FFN
  - refined phase : during layer l's attention, re-rank using the residual after layer l-1
  - fresh phase   : after attention, rank by (optionally noisy) true scores; fetch until the cutoff
  The GPU computes pinned/cached blocks predicted active plus everything that arrived in time.

Policies (the research lives here; add your own by subclassing Policy):
  nocut    : wait for every predicted-active block (LeanStream-style: fast on average, no deadline)
  static   : fixed per-layer cutoffs computed offline
  slack    : reserve mandatory time for all remaining layers, give this layer a share of the rest
  weighted : like slack, but shares follow per-layer sensitivity weights (oracle_topk.py --sensitivity)
  adaptive : weighted, plus each token's share follows how much predicted mass that layer is missing

Hardware numbers are placeholders. Calibrate them (fio for flash, profiling for GPU) before
trusting absolute milliseconds; relative comparisons between policies are meaningful earlier.

Examples:
  python anytime_sim.py --model tiny                                   # smoke test
  python anytime_sim.py --model Qwen/Qwen2.5-1.5B --profile results/profile/freq.pt \
      --weights results/oracle/sensitivity.json --deadlines auto
"""
import argparse
import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (collect_ffn_stats, compare_logits, decoder_layers, down_col_norms, eval_batches,
                    ffn_hidden, load_model, model_dims, neuron_scores, pick_device, swap_mlps)


# ============================================================================ hardware model

@dataclass
class HW:
    flash_seq_gbps: float = 4.0    # large sequential read bandwidth, GB/s (UFS 4.0 / NVMe class)
    flash_half_kb: float = 16.0    # request size (KB) that gets half the sequential bandwidth
    dram_gbps: float = 100.0       # effective memory bandwidth for resident weights, GB/s
    weight_bits: float = 4.5       # bits per weight including quantization scales
    group: int = 1                 # neurons per I/O block (gate row + up row + down column each)
    layer_overhead_us: float = 20.0  # per-layer fixed cost: kernel launches, sync, norms


def derive_costs(hw, dims):
    """Seconds per operation, derived from model shapes and the hardware model."""
    bpw = hw.weight_bits / 8
    d, heads, kv, hd = dims["d_model"], dims["heads"], dims["kv_heads"], dims["head_dim"]
    neuron_bytes = 3 * d * bpw
    block_bytes = neuron_bytes * hw.group
    kb = block_bytes / 1024
    eff_bw = hw.flash_seq_gbps * 1e9 * kb / (kb + hw.flash_half_kb)   # small reads get less bandwidth
    attn_params = d * heads * hd * 2 + d * kv * hd * 2                 # q, o and k, v projections
    return {
        "neuron_bytes": neuron_bytes, "block_bytes": block_bytes, "flash_eff_gbps": eff_bw / 1e9,
        "tau": block_bytes / eff_bw,                                    # flash time per block
        "kappa_neuron": neuron_bytes / (hw.dram_gbps * 1e9),            # GPU time per neuron computed
        "kappa_block": block_bytes / (hw.dram_gbps * 1e9),
        "attn": attn_params * bpw / (hw.dram_gbps * 1e9) + hw.layer_overhead_us * 1e-6,
        "head": dims["vocab"] * d * bpw / (hw.dram_gbps * 1e9),
    }


# ============================================================================ policies

class Policy:
    """Chooses the absolute cutoff time for layer l of the current token, given the GPU is
    ready to start layer l at time s_l. D is the token budget excluding the LM head, m is a
    worst-case bound on one layer's mandatory time (attention + computing resident blocks).
    ctx["missing"] (only filled if needs_ctx) is the share of this layer's predicted-active
    mass that is NOT resident, i.e. how much this layer stands to gain from flash time."""
    name = "base"
    needs_ctx = False

    def __init__(self, L, D, m, weights=None):
        self.L, self.D, self.m = L, D, m
        self.w = np.ones(L) if weights is None else np.asarray(weights, dtype=float)
        self.w_suffix = np.cumsum(self.w[::-1])[::-1]          # sum of w_j for j >= l

    def cutoff(self, l, s_l, ctx):
        raise NotImplementedError

    def observe(self, l, mean_missing):
        """Called once per layer per sequence with the average ctx['missing'] seen."""


class NoCut(Policy):
    name = "nocut"

    def cutoff(self, l, s_l, ctx):
        return math.inf


class Static(Policy):
    name = "static"

    def cutoff(self, l, s_l, ctx):
        return (l + 1) * self.D / self.L


class Slack(Policy):
    """Deadline guarantee by construction: after this layer, the time left always covers the
    mandatory work of every remaining layer (invariant proved by induction over layers)."""
    name = "slack"

    def share(self, l, ctx):
        return self.w[l] / self.w_suffix[l]

    def cutoff(self, l, s_l, ctx):
        slack = self.D - s_l - (self.L - l) * self.m          # time beyond all remaining mandatory work
        if slack <= 0:
            return s_l + self.m                               # mandatory only
        return s_l + self.m + slack * self.share(l, ctx)


class Adaptive(Slack):
    """Certainty-equivalent reference policy: this layer's share grows with how much predicted
    mass it is actually missing right now, compared with what later layers typically miss
    (running averages). Same guarantee as Slack, since only the share changes.
    Improving on this (value curves, DP / MPC lookup tables) is the research."""
    name = "adaptive"
    needs_ctx = True

    def __init__(self, L, D, m, weights=None, ema=0.3):
        super().__init__(L, D, m, weights)
        self.mu = np.ones(L)        # expected missing share per layer; starts pessimistic
        self.ema = ema

    def share(self, l, ctx):
        mine = self.w[l] * ctx["missing"]
        later = float((self.w[l + 1:] * self.mu[l + 1:]).sum())
        return mine / (mine + later) if mine + later > 0 else 1.0

    def observe(self, l, mean_missing):
        self.mu[l] = (1 - self.ema) * self.mu[l] + self.ema * mean_missing


def make_policy(name, L, D, m, weights):
    if name == "nocut":
        return NoCut(L, D, m)
    if name == "static":
        return Static(L, D, m)
    if name == "slack":
        return Slack(L, D, m)
    if name in ("weighted", "adaptive"):
        if weights is None:
            raise ValueError(f"'{name}' needs --weights (run oracle_topk.py --sensitivity first)")
        p = Slack(L, D, m, weights) if name == "weighted" else Adaptive(L, D, m, weights)
        p.name = name
        return p
    raise ValueError(f"unknown policy {name}")


# ============================================================================ co-simulator

class AnytimeMLP(nn.Module):
    def __init__(self, sim, l, mlp):
        super().__init__()
        self.mlp, self.l = mlp, l
        self._sim = [sim]  # stored in a list so nn.Module does not try to register it

    def forward(self, x):
        return self._sim[0].run_layer(self.l, self.mlp, x)


class AnytimeSim:
    def __init__(self, model, costs, policy, deadline, k_frac, pinned, cache_frac, group, noise,
                 io_jitter=0.0, seed=0):
        self.io_jitter = io_jitter                           # flash slowdown: tau * U[1, 1 + io_jitter]
        self.rng = np.random.default_rng(seed)
        self.layers = decoder_layers(model)
        self.L = len(self.layers)
        self.mlps = [ly.mlp for ly in self.layers]
        self.norms = [ly.post_attention_layernorm for ly in self.layers]
        self.col = [down_col_norms(m) for m in self.mlps]
        self.d_ff = self.col[0].numel()
        self.g = group
        self.nb = math.ceil(self.d_ff / group)
        self.Kb = max(1, round(k_frac * self.nb))            # target active blocks per token per layer
        self.c, self.policy, self.deadline = costs, policy, deadline
        self.pinned = pinned                                 # np.bool_ [L, nb]
        self.cap = int(round(cache_frac * self.nb))          # per-layer LRU capacity in blocks
        self.noise, self.seed, self.gen = noise, seed, None
        self.persist_cache, self.in_cache, self.clock = False, None, 0
        self.lat, self.bytes = [], 0.0
        self.n = np.zeros(self.L)
        self.cov = np.zeros(self.L)
        self.fetched = np.zeros(self.L)
        self.recall = {k: np.zeros(self.L) for k in ("stale", "refined", "fresh")}

    # ------------------------------------------------------------------ plumbing
    @contextmanager
    def attach(self, model):
        self.rmid = [None] * self.L
        handles = []

        def grab_rmid(l):
            def hook(mod, args):
                self.rmid[l] = args[0]   # residual entering layer l's FFN (after attention)
            return hook

        for l, ly in enumerate(self.layers):
            handles.append(ly.post_attention_layernorm.register_forward_pre_hook(grab_rmid(l)))

        def grab_r0(mod, args, kwargs):
            self.r0 = args[0] if args else kwargs["hidden_states"]

        handles.append(self.layers[0].register_forward_pre_hook(grab_r0, with_kwargs=True))
        try:
            with swap_mlps(model, lambda i, m: AnytimeMLP(self, i, m)):
                yield self
        finally:
            for h in handles:
                h.remove()

    def _scores(self, j, residual):
        """What layer j's scores would be if its FFN input came from `residual` (a stale guess)."""
        return neuron_scores(ffn_hidden(self.mlps[j], self.norms[j](residual)), self.col[j]).reshape(-1, self.d_ff)

    def _noisy(self, s):
        """Multiplicative log-normal noise: a stand-in for an imperfect learned predictor."""
        if self.noise <= 0:
            return s
        return s * torch.exp(self.noise * torch.randn(s.shape, generator=self.gen, device=s.device))

    def _blocks(self, s):
        if self.g == 1:
            return s
        pad = self.nb * self.g - self.d_ff
        return (F.pad(s, (0, pad)) if pad else s).view(s.shape[0], self.nb, self.g).sum(-1)

    def _start_sequence(self, n_tokens, device):
        """Called at layer 0 of every forward pass. Timing restarts for each token; the DRAM
        cache restarts too, unless persist_cache is set (use that with model.generate(), where
        each forward pass is one new token and the cache must carry over)."""
        self.t_gpu = np.zeros(n_tokens)    # when the GPU finishes the previous layer
        self.io_free = np.zeros(n_tokens)  # where the flash channel's timeline stands
        self.f_prev = np.zeros(n_tokens)   # when the previous layer's attention finished
        if not self.persist_cache or self.in_cache is None:
            self.in_cache = np.zeros((self.L, self.nb), dtype=bool)
            self.stamp = np.full((self.L, self.nb), -1, dtype=np.int64)
            self.clock = 0
        if self.gen is None:
            self.gen = torch.Generator(device=device).manual_seed(self.seed)

    # ------------------------------------------------------------------ one layer, all tokens
    @torch.no_grad()
    def run_layer(self, l, mlp, x):
        B, T, _ = x.shape
        h = ffn_hidden(mlp, x)
        s_true = neuron_scores(h, self.col[l]).reshape(B * T, -1)
        if l == 0:
            self._start_sequence(B * T, x.device)
            early = self._noisy(self._scores(0, self.r0))
            stale = refined = early
        else:
            stale, refined = self.next_pred
        keep = self._decide(l, stale, refined, self._noisy(s_true), s_true)
        if self.g > 1:
            keep = keep.repeat_interleave(self.g, dim=1)[:, : self.d_ff]
        y = mlp.down_proj(h * keep.reshape(B, T, -1).to(h.dtype))
        if l + 1 < self.L:  # predictions for the next layer, from this layer's stale and refined residuals
            r = self.rmid[l]
            self.next_pred = (self._noisy(self._scores(l + 1, r)), self._noisy(self._scores(l + 1, r + y)))
        return y

    def _decide(self, l, stale, refined, fresh, s_true):
        N, Kb, g = s_true.shape[0], self.Kb, self.g
        tau, kap_b, kap_n, attn = self.c["tau"], self.c["kappa_block"], self.c["kappa_neuron"], self.c["attn"]
        tb = self._blocks(s_true)
        true_top = tb.topk(Kb, dim=-1)
        top_mass = true_top.values.sum(-1).clamp_min(1e-30)
        ranks = []
        for name, p in (("stale", stale), ("refined", refined), ("fresh", fresh)):
            top = self._blocks(p).topk(Kb, dim=-1)
            self.recall[name][l] += (tb.gather(1, top.indices).sum(-1) / top_mass).sum().item()
            ranks.append(top.indices.cpu().numpy())
            if name == "refined":
                refined_vals = top.values.cpu().numpy()
        tb_np, tt_np, mass_np = tb.cpu().numpy(), true_top.indices.cpu().numpy(), top_mass.cpu().numpy()

        keep = np.zeros((N, self.nb), dtype=bool)
        pinned, in_cache, stamp = self.pinned[l], self.in_cache[l], self.stamp[l]
        ctx, missing_sum = {}, 0.0
        for t in range(N):
            s_l = self.t_gpu[t]
            f_l = s_l + attn
            resident = pinned | in_cache
            if self.policy.needs_ctx:   # what the best prediction available at s_l says is missing
                v = refined_vals[t]
                ctx["missing"] = float(v[~resident[ranks[1][t]]].sum() / max(v.sum(), 1e-30))
                missing_sum += ctx["missing"]
            arrive_by = self.policy.cutoff(l, s_l, ctx) - kap_b  # last block must land in time to compute
            if self.io_jitter > 0:
                tau = self.c["tau"] * (1 + self.io_jitter * self.rng.random())
            fetched = np.zeros(self.nb, dtype=bool)
            u, last, n_spec = self.io_free[t], -math.inf, 0
            phases = ((ranks[0][t], max(self.f_prev[t], u), s_l, True),   # stale
                      (ranks[1][t], s_l, f_l, True),                         # refined
                      (ranks[2][t], f_l, arrive_by, False))                  # fresh
            for rank, p0, p1, speculative in phases:
                u = max(u, p0)
                if u >= p1:
                    continue
                cand = rank[~(resident[rank] | fetched[rank])]
                n = len(cand) if math.isinf(p1) else min(len(cand), int((p1 - u) / tau))
                if speculative:
                    n = min(n, Kb - n_spec)                      # buffer cap on speculative fetches
                    n_spec += max(n, 0)
                if n > 0:
                    fetched[cand[:n]] = True
                    u += n * tau
                    last = u
            fr = ranks[2][t]
            comp = fetched.copy()
            comp[fr[resident[fr]]] = True                        # resident blocks the fresh predictor wants
            e_l = f_l + kap_n * g * int(comp.sum())
            if last > -math.inf:
                e_l = max(e_l, last + kap_b)
            self.t_gpu[t], self.f_prev[t], self.io_free[t] = e_l, f_l, u
            keep[t] = comp
            if self.cap > 0:                                     # per-layer LRU; pinned blocks never enter
                stamp[comp & ~pinned] = self.clock + t
                in_cache |= fetched & ~pinned
                over = int(in_cache.sum()) - self.cap
                if over > 0:
                    idx = np.flatnonzero(in_cache)
                    in_cache[idx[np.argpartition(stamp[idx], over - 1)[:over]]] = False
            nf = int(fetched.sum())
            self.bytes += nf * self.c["block_bytes"]
            self.fetched[l] += nf
            tt = tt_np[t]
            self.cov[l] += tb_np[t][tt][comp[tt]].sum() / mass_np[t]
        self.n[l] += N
        if l == self.L - 1:
            self.clock += N
        if self.policy.needs_ctx:
            self.policy.observe(l, missing_sum / N)
        if l == self.L - 1:
            self.lat.extend((self.t_gpu + self.c["head"]).tolist())
        return torch.from_numpy(keep).to(s_true.device)

    def summary(self):
        lat = np.asarray(self.lat)
        n = np.maximum(self.n, 1)
        return {
            "miss_rate": float((lat > self.deadline * (1 + 1e-9)).mean()),
            "lat_p50_ms": float(np.percentile(lat, 50) * 1e3),
            "lat_p99_ms": float(np.percentile(lat, 99) * 1e3),
            "lat_max_ms": float(lat.max() * 1e3),
            "mb_per_token": self.bytes / max(len(lat), 1) / 1e6,
            "coverage": float((self.cov / n).mean()),
            "coverage_per_layer": (self.cov / n).tolist(),
            "fetched_blocks_per_layer": (self.fetched / n).tolist(),
            "recall": {k: (v / n).tolist() for k, v in self.recall.items()},
        }


# ============================================================================ experiment driver

def pinned_from_freq(freq, group, nb, frac):
    L, d_ff = freq.shape
    f = freq.float()
    if group > 1:
        f = F.pad(f, (0, nb * group - d_ff)).view(L, nb, group).sum(-1)
    pinned = np.zeros((L, nb), dtype=bool)
    k = int(round(frac * nb))
    if k > 0:
        np.put_along_axis(pinned, f.topk(k, dim=1).indices.numpy(), True, axis=1)
    return pinned


@torch.no_grad()
def run_policy(model, batches, refs, device, costs, policy, deadline, args, pinned):
    sim = AnytimeSim(model, costs, policy, deadline, args.k_frac, pinned, args.cache_frac,
                     args.group, args.pred_noise, args.io_jitter, args.seed)
    kl = agree = nll = 0.0
    n = 0
    with sim.attach(model):
        for b, ref in zip(batches, refs):
            ids = b.to(device)
            m = compare_logits(ref.to(device), model(input_ids=ids, use_cache=False).logits, ids)
            kl += m["kl"].sum().item()
            agree += m["agree"].sum().item()
            nll += m["nll"].sum().item()
            n += m["kl"].numel()
    out = {"kl": kl / n, "top1_agree": agree / n, "ppl": math.exp(nll / (n - len(batches)))}
    out.update(sim.summary())
    return out, np.asarray(sim.lat)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--policies", default="static,slack,weighted,adaptive")
    ap.add_argument("--deadlines", default="auto", help="comma-separated ms per token, or 'auto'")
    ap.add_argument("--k-frac", type=float, default=0.2, help="target fraction of FFN neurons treated as active")
    ap.add_argument("--pinned-frac", type=float, default=0.1, help="fraction of each layer's blocks pinned in DRAM")
    ap.add_argument("--cache-frac", type=float, default=0.1, help="per-layer LRU cache size, as a fraction of blocks")
    ap.add_argument("--group", type=int, default=1, help="neurons per I/O block")
    ap.add_argument("--pred-noise", type=float, default=0.0, help="log-normal noise on all predictions (0 = oracle predictor)")
    ap.add_argument("--io-jitter", type=float, default=0.0, help="flash slowdown per token-layer: tau * U[1, 1+j]")
    ap.add_argument("--flash-gbps", type=float, default=4.0)
    ap.add_argument("--flash-half-kb", type=float, default=16.0)
    ap.add_argument("--dram-gbps", type=float, default=100.0)
    ap.add_argument("--bits", type=float, default=4.5)
    ap.add_argument("--layer-overhead-us", type=float, default=20.0)
    ap.add_argument("--profile", default=None, help="freq.pt from profile_ffn.py (else estimated from one sequence)")
    ap.add_argument("--weights", default=None, help="sensitivity.json from oracle_topk.py --sensitivity")
    ap.add_argument("--n-seqs", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/sim")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)
    model, tok = load_model(args.model, device)
    dims = model_dims(model.config)
    batches = eval_batches(tok, args.n_seqs, args.seq_len, args.text_file, dims["vocab"], args.seed)

    hw = HW(args.flash_gbps, args.flash_half_kb, args.dram_gbps, args.bits, args.group, args.layer_overhead_us)
    costs = derive_costs(hw, dims)
    L = dims["n_layers"]
    nb = math.ceil(dims["d_ff"] / args.group)
    Kb = max(1, round(args.k_frac * nb))
    m = costs["attn"] + costs["kappa_neuron"] * args.group * 3 * Kb + costs["kappa_block"]
    d_min = L * m + costs["head"]
    d_worst = d_min + L * Kb * costs["tau"]
    ffn_resident_mb = L * nb * (args.pinned_frac + args.cache_frac) * costs["block_bytes"] / 1e6
    print(f"\nModel {args.model}: {L} layers, d_ff={dims['d_ff']}, {nb} blocks/layer, {Kb} active blocks/token/layer")
    print(f"Block {costs['block_bytes']/1024:.1f} KB -> effective flash {costs['flash_eff_gbps']:.2f} GB/s, "
          f"{costs['tau']*1e6:.2f} us per block; attention {costs['attn']*1e3:.3f} ms/layer; LM head {costs['head']*1e3:.3f} ms")
    print(f"Resident FFN memory (pinned + cache): {ffn_resident_mb:.0f} MB")
    print(f"Minimum feasible deadline (mandatory work only): {d_min*1e3:.2f} ms/token; "
          f"no-cache worst case: {d_worst*1e3:.1f} ms/token")

    if args.profile:
        freq = torch.load(args.profile)["freq"]
    else:
        print("No --profile given: estimating hot neurons from the first sequence.")
        freq, _ = collect_ffn_stats(model, batches[:1], device, args.k_frac)
    pinned = pinned_from_freq(freq, args.group, nb, args.pinned_frac)
    weights = json.load(open(args.weights))["weights"] if args.weights else None

    if args.deadlines == "auto":
        deadlines = np.geomspace(1.15 * d_min, d_worst, 6).tolist()
    else:
        deadlines = [float(x) / 1e3 for x in args.deadlines.split(",")]
    policies = [p for p in args.policies.split(",") if p and p != "nocut"]
    if weights is None:
        for p in ("weighted", "adaptive"):
            if p in policies:
                print(f"Skipping '{p}' (needs --weights from oracle_topk.py --sensitivity).")
                policies.remove(p)

    with torch.no_grad():
        refs = [model(input_ids=b.to(device), use_cache=False).logits.to(torch.float16).cpu() for b in batches]

    rows = []
    base, base_lat = run_policy(model, batches, refs, device, costs,
                                NoCut(L, math.inf, m), math.inf, args, pinned)
    ladder = base["recall"]
    header = f"{'deadline':>9} {'policy':>9} {'KL':>9} {'top1':>6} {'ppl':>8} {'miss':>6} {'p50ms':>8} {'p99ms':>8} {'MB/tok':>7} {'cover':>6}"
    print("\n" + header + "\n" + "-" * len(header))
    for D in deadlines:
        r = dict(base, miss_rate=float((base_lat > D * (1 + 1e-9)).mean()))
        rows.append({"deadline_ms": D * 1e3, "policy": "nocut", **r})
        for p in policies:
            res, _ = run_policy(model, batches, refs, device, costs,
                                make_policy(p, L, D - costs["head"], m, weights), D, args, pinned)
            rows.append({"deadline_ms": D * 1e3, "policy": p, **res})
        for r in rows[-(1 + len(policies)):]:
            print(f"{r['deadline_ms']:9.2f} {r['policy']:>9} {r['kl']:9.5f} {r['top1_agree']:6.3f} {r['ppl']:8.3f} "
                  f"{r['miss_rate']:6.1%} {r['lat_p50_ms']:8.2f} {r['lat_p99_ms']:8.2f} {r['mb_per_token']:7.2f} {r['coverage']:6.3f}")

    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump({"args": vars(args), "hw": asdict(hw), "costs": costs, "rows": rows}, fh, indent=1)

    names = ["nocut"] + policies
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for p in names:
        pts = [r for r in rows if r["policy"] == p]
        x = [r["deadline_ms"] for r in pts]
        a1.plot(x, [r["kl"] for r in pts], "o-", label=p)
        a2.plot(x, [r["miss_rate"] for r in pts], "o-", label=p)
    for a in (a1, a2):
        a.set_xscale("log")
        a.set_xlabel("per-token deadline (ms)")
        a.legend()
    a1.set_yscale("log")
    a1.set_ylabel("mean KL to dense model")
    a1.set_title("Quality vs deadline")
    a2.set_ylabel("deadline miss rate")
    a2.set_title("Deadline misses")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "quality_vs_deadline.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    for k in ("stale", "refined", "fresh"):
        ax.plot(range(L), ladder[k], "o-", ms=3, label=k)
    ax.set_xlabel("layer")
    ax.set_ylabel("share of true top-k mass in predicted top-k")
    ax.set_title(f"Prediction ladder (noise={args.pred_noise})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "prediction_ladder.png"), dpi=150)
    print(f"\nWrote {args.out}/results.json, quality_vs_deadline.png, prediction_ladder.png")


if __name__ == "__main__":
    main()
