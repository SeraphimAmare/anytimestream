"""price_lab.py: should each layer get the same share of a token's FFN budget?

A neuron's contribution to its layer's FFN output has size s = |h| * ||W_down column||. Skipping it
perturbs the layer output by roughly s, and to second order the damage to the model's final
next-token distribution is G_l * (s / ||r_l||)^2, where r_l is the residual stream the FFN adds into
and G_l is a per-layer gain measured once. That gives every neuron a value per unit of cost, and a
single price lambda can then decide what to compute in every layer of every token.

Experiments (all quality numbers are KL divergence to the dense model, per token):
  law       one layer sparsified at a time: does KL grow with rho^2 (slope 2), and how much do gains differ?
  additive  all layers sparsified: does sum_l G_l rho_l^2 predict the total?
  compute   equal average compute: uniform top-k, static per-layer split, per-token price, price without
            gains, and a price policy capped so no token ever exceeds the static budget (hard deadline)
  flash     10% hot neurons resident and free; the budget is neurons fetched from flash per token
  predicted the flash experiment with neurons chosen from an early guess instead of true scores

Gains, resident sets and static splits come from a calibration text; everything is evaluated on a
separate text. Only the scalar price is tuned on the evaluation text, and only to hit the budget.

  python price_lab.py --model Qwen/Qwen2.5-0.5B             # WikiText-2 train (calibrate) / test (evaluate)
  python price_lab.py --model path/to/model --calib-file a.txt --eval-file b.txt
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROBE_FRACS = (0.1, 0.2, 0.4)
REF_TOPK = int(os.environ.get("REF_TOPK", "0"))   # >0: keep only the reference's top-k tokens (large vocabularies)


def kl_to_ref(ref, lp):
    """KL(reference || current) per token. ref is a full log-prob tensor, or (top-k log-probs, top-k ids):
    then the KL is that of the two distributions coarsened to the top-k tokens plus one 'rest' bucket,
    a lower bound that is tight when the top-k covers almost all of the reference mass."""
    if not isinstance(ref, tuple):
        return (ref.exp() * (ref - lp)).sum(-1)
    lpk, idx = ref
    lqk = lp.gather(-1, idx)
    pk = lpk.exp()
    p_rest = (1 - pk.sum(-1)).clamp_min(1e-12)
    q_rest = (1 - lqk.exp().sum(-1)).clamp_min(1e-12)
    return (pk * (lpk - lqk)).sum(-1) + p_rest * (p_rest.log() - q_rest.log())


def ref_of(logits):
    lp = F.log_softmax(logits, -1)
    if REF_TOPK > 0:
        v, i = lp.topk(REF_TOPK, dim=-1)
        return (v.cpu(), i.cpu())
    return lp.cpu()
PER_TOKEN, SECTION = {}, [""]


def get_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--calib-file", default=None)
    ap.add_argument("--eval-file", default=None)
    ap.add_argument("--calib-seqs", type=int, default=8)
    ap.add_argument("--eval-seqs", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4, help="sequences per forward pass")
    ap.add_argument("--budgets", default="0.1,0.2,0.3", help="average share of FFN neurons computed")
    ap.add_argument("--flash-budgets", default="0.05,0.1,0.2", help="average share fetched from flash")
    ap.add_argument("--resident", type=float, default=0.1, help="share of each layer's neurons kept in DRAM")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="skip the no-gain price, oracle and capped-price rows")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/price_lab")
    return ap.parse_args()


def load(args):
    import transformers
    from packaging import version
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    kw = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(args.model, **{kw: dtype}).to(device).eval()
    return model, AutoTokenizer.from_pretrained(args.model), device


def token_chunks(tokenizer, path, split, n, T):
    if path:
        text = open(path, encoding="utf-8", errors="ignore").read()
    else:
        from datasets import load_dataset
        text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    avail = ids.numel() // T
    if avail < n:
        raise ValueError(f"{path or split}: only {avail} chunks of {T} tokens")
    starts = np.linspace(0, (avail - 1) * T, n).astype(int)
    return torch.stack([ids[s:s + T] for s in starts])


def masked_topk(value, cand, take):
    """Keep, per row, the `take` highest-value entries among candidates (take may differ per row)."""
    vm = value.masked_fill(~cand, float("-inf"))
    srt = vm.sort(dim=-1, descending=True).values
    idx = (take - 1).clamp(min=0).unsqueeze(-1)
    thr = srt.gather(-1, idx)
    return cand & (vm >= thr) & (take > 0).unsqueeze(-1)


class Lab:
    """Runs the model with every layer's FFN under a selection policy and records what it cost."""

    def __init__(self, model, batches, device):
        self.model, self.device = model, device
        self.layers = model.model.layers
        self.L = len(self.layers)
        self.col = [ly.mlp.down_proj.weight.float().norm(dim=0) for ly in self.layers]
        self.d_ff = self.col[0].numel()
        self.gain = torch.ones(self.L, device=device)
        self.resident = torch.zeros(self.L, self.d_ff, dtype=torch.bool, device=device)
        self.plan, self.collect, self.samples = {}, False, None
        self.r_mid, self.r_pre = [None] * self.L, [None] * self.L
        for l, ly in enumerate(self.layers):
            ly.post_attention_layernorm.register_forward_pre_hook(self._grab(self.r_mid, l))
            ly.register_forward_pre_hook(self._grab_pre(l), with_kwargs=True)
            ly.mlp.forward = self._ffn(l, ly)
        self.set_data(batches)

    @staticmethod
    def _grab(store, l):
        def hook(module, args):
            store[l] = args[0]
        return hook

    def _grab_pre(self, l):
        def hook(module, args, kwargs):
            self.r_pre[l] = args[0] if args else kwargs["hidden_states"]
        return hook

    def set_data(self, batches):
        self.batches = batches
        self.plan = {}
        self.ref = [ref_of(self._logits(b)) for b in batches]

    @torch.no_grad()
    def _logits(self, b):
        return self.model(input_ids=b.to(self.device), use_cache=False).logits.float()

    def _ffn(self, l, ly):
        mlp, norm = ly.mlp, ly.post_attention_layernorm

        def forward(x):
            h = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
            p = self.plan.get(l)
            if p is None and not self.collect and not getattr(self, "capture", False):
                return mlp.down_proj(h)
            hf = h.reshape(-1, h.shape[-1])
            rn = self.r_mid[l].reshape(-1, self.r_mid[l].shape[-1]).float().norm(dim=-1).clamp_min(1e-6)
            s = hf.abs().float() * self.col[l]
            if self.collect:
                self._sample(l, s, rn)
            if getattr(self, "capture", False):
                self.store.append(self.gain[l] * (s / rn[:, None]) ** 2)
            if p is None:
                return mlp.down_proj(h)
            if p.get("predict"):
                rp = self.r_pre[l]
                xp = norm.forward(rp)
                hp = mlp.act_fn(mlp.gate_proj(xp)) * mlp.up_proj(xp)
                sel = hp.reshape(-1, hp.shape[-1]).abs().float() * self.col[l]
                sel_rn = rp.reshape(-1, rp.shape[-1]).float().norm(dim=-1).clamp_min(1e-6)
            else:
                sel, sel_rn = s, rn
            g = self.gain[l] if p.get("gain", True) else 1.0
            value = g * (sel / sel_rn[:, None]) ** 2
            keep = self._decide(l, p, value)
            out_f = mlp.down_proj(hf * keep.to(hf.dtype))
            full = mlp.down_proj(hf)
            delta = (full - out_f).float()
            self.rho[l] = delta.norm(dim=-1) / rn
            dropped = (s.pow(2) * (~keep)).sum(-1)
            self.orth[l] = (delta.pow(2).sum(-1) / dropped.clamp_min(1e-12))
            self.cost_kept += keep.sum(-1)
            self.cost_fetch += (keep & ~self.resident[l]).sum(-1)
            return out_f.reshape(h.shape[:-1] + (out_f.shape[-1],))

        return forward

    def _decide(self, l, p, value):
        kind, n_tok = p["kind"], value.shape[0]
        res = self.resident[l].unsqueeze(0).expand_as(value)
        if kind == "topk":
            k = int(round(p["frac"][l] * self.d_ff))
            take = torch.full((n_tok,), k, device=value.device)
            return masked_topk(value, torch.ones_like(res), take)
        if kind == "thr":
            return value >= p["lam"]
        if kind == "thr_tok":
            return value >= self.cur_tok_lam[:, None]
        if kind == "res_topk":
            k = int(round(p["frac"][l] * self.d_ff))
            take = torch.full((n_tok,), k, device=value.device)
            return res | masked_topk(value, ~res, take)
        if kind == "res_thr":
            return res | ((value >= p["lam"]) & ~res)
        if kind in ("cap", "res_cap"):
            cand = (value >= p["lam"]) & (~res if kind == "res_cap" else torch.ones_like(res))
            left = p["cap"] - self.used
            if l < self.L - 1:
                allow = torch.floor(left * p["w"][l] / p["w_suffix"][l])
            else:
                allow = left
            take = torch.minimum(allow, cand.sum(-1).float()).long()
            keep = masked_topk(value, cand, take)
            self.used += keep.sum(-1)
            return keep | res if kind == "res_cap" else keep
        raise ValueError(kind)

    def _sample(self, l, s, rn, per_batch=50_000):
        rel = (s / rn[:, None]).flatten()
        idx = torch.randint(0, rel.numel(), (min(per_batch, rel.numel()),), generator=self.gen, device=rel.device)
        self.samples[l].append(rel[idx].cpu())
        self.sample_ids[l].append((idx % self.d_ff).cpu())
        self.hot[l] += (s >= s.quantile(0.8, dim=-1, keepdim=True)).float().sum(0).cpu()

    @torch.no_grad()
    def dense_stats(self):
        """Samples of relative contributions s/||r|| per layer, and how often each neuron is in its token's top 20%."""
        self.collect, self.plan = True, {}
        self.samples, self.sample_ids = [[] for _ in range(self.L)], [[] for _ in range(self.L)]
        self.gen = torch.Generator(device=self.device).manual_seed(0)
        self.hot = [torch.zeros(self.d_ff) for _ in range(self.L)]
        for b in self.batches:
            self._logits(b)
        self.collect = False
        return ([torch.cat(x) for x in self.samples], [torch.cat(x) for x in self.sample_ids],
                torch.stack(self.hot))

    @torch.no_grad()
    def token_prices(self, budget):
        """For each token, the price at which exactly `budget` of its neurons (all layers together) are worth
        computing, read from the dense pass. Used only as an oracle diagnostic."""
        prices = []
        for b in self.batches:
            self.store, self.plan, self.collect = [], {}, False
            self.capture = True
            self._logits(b)
            self.capture = False
            vals = torch.cat(self.store, dim=1)
            k = int(round(budget * vals.shape[1]))
            prices.append(vals.topk(k, dim=1).values[:, -1].clone())
            del self.store
        return prices

    @torch.no_grad()
    def run(self, plan):
        self.plan = {l: plan for l in range(self.L)} if "kind" in plan else plan
        kls, agree, rhos, orths, kept, fetch = [], [], [], [], [], []
        tok_lams = plan.get("tok_lams") if "kind" in plan else None
        for i, (b, ref) in enumerate(zip(self.batches, self.ref)):
            if tok_lams is not None:
                self.cur_tok_lam = tok_lams[i]
            self.rho, self.orth = {}, {}
            n = b.numel()
            self.used = torch.zeros(n, device=self.device)
            self.cost_kept = torch.zeros(n, device=self.device)
            self.cost_fetch = torch.zeros(n, device=self.device)
            ref = tuple(x.to(self.device) for x in ref) if isinstance(ref, tuple) else ref.to(self.device)
            lp = F.log_softmax(self._logits(b), -1)
            kls.append(kl_to_ref(ref, lp).flatten().clamp_min(0))
            top = ref[1][..., 0] if isinstance(ref, tuple) else ref.argmax(-1)
            agree.append((top == lp.argmax(-1)).float().flatten())
            rhos.append(torch.stack([self.rho.get(l, torch.zeros_like(kls[-1])) for l in range(self.L)]))
            orths.append(torch.stack([self.orth.get(l, torch.ones_like(kls[-1])) for l in range(self.L)]))
            kept.append(self.cost_kept / (self.L * self.d_ff))
            fetch.append(self.cost_fetch / (self.L * self.d_ff))
        self.plan = {}
        return {"kl": torch.cat(kls).cpu(), "agree": torch.cat(agree).cpu(), "rho": torch.cat(rhos, 1).cpu(),
                "orth": torch.cat(orths, 1).cpu(), "kept": torch.cat(kept).cpu(), "fetch": torch.cat(fetch).cpu()}


def summarize(r, name, budget):
    kl = r["kl"]
    PER_TOKEN[f"{SECTION[0]}|{budget}|{name}"] = kl.clone()
    return {"policy": name, "budget": budget, "kl": kl.mean().item(), "kl_p90": kl.quantile(0.9).item(),
            "top1": r["agree"].mean().item(), "kept": r["kept"].mean().item(),
            "kept_max": r["kept"].max().item(), "fetch": r["fetch"].mean().item(),
            "fetch_p95": r["fetch"].quantile(0.95).item(), "fetch_max": r["fetch"].max().item()}


def calibrate(lab, make, target, key, init, iters=16):
    """Bisection on log(lambda), started around an estimate, until the realized average cost hits the target."""
    lo, hi = math.log(init) - 5, math.log(init) + 5
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        r = lab.run(make(math.exp(mid)))
        got = r[key].mean().item()
        if best is None or abs(got - target) < abs(best[1] - target):
            best = (math.exp(mid), got, r)
        if abs(got - target) < 0.002:
            break
        lo, hi = (mid, hi) if got > target else (lo, mid)
    return best


def fit_law(lab):
    """One layer sparsified at a time: gain G_l and slope of log KL against log rho."""
    gains, slopes, orth, xs, ys = [], [], [], [], []
    for l in range(lab.L):
        lr_all, lk_all = [], []
        for f in PROBE_FRACS:
            r = lab.run({l: {"kind": "topk", "frac": [f] * lab.L}})
            rho, kl = r["rho"][l], r["kl"]
            ok = (kl > 1e-10) & (rho > 1e-8)
            lr_all.append(rho[ok].log())
            lk_all.append(kl[ok].log())
            if f == 0.2:
                orth.append(r["orth"][l].median().item())
        lr, lk = torch.cat(lr_all), torch.cat(lk_all)
        gains.append((lk - 2 * lr).mean().exp().item())
        slopes.append((((lr - lr.mean()) * (lk - lk.mean())).sum() / ((lr - lr.mean()) ** 2).sum()).item())
        xs.append(math.log(gains[-1]) + 2 * lr)
        ys.append(lk)
    x, y = torch.cat(xs), torch.cat(ys)
    r2 = (1 - ((y - x) ** 2).sum() / ((y - y.mean()) ** 2).sum()).item()
    return gains, slopes, orth, r2


def price_at(samples, gains, budget, exclude=None):
    """The single price at which, on the calibration text, the chosen share of neurons is worth computing.
    Returns that price and the share of each layer's neurons it selects (resident neurons excluded if given)."""
    vals = []
    for l, (g, x) in enumerate(zip(gains, samples)):
        v = g * x.pow(2)
        if exclude is not None:
            v = v.masked_fill(exclude[l], float("-inf"))
        vals.append(v)
    pooled = torch.cat(vals).numpy()
    lam = float(np.quantile(pooled, 1 - budget))
    return lam, np.array([(v >= lam).float().mean().item() for v in vals])


def static_split(samples, gains, budget, exclude=None):
    """Per-layer shares a single price would give on average, rescaled so their mean is exactly the budget."""
    _, shares = price_at(samples, gains, budget, exclude)
    shares = shares * budget * len(shares) / max(shares.sum(), 1e-12)
    return np.minimum(shares, 1.0)


def main():
    args = get_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    model, tok, device = load(args)
    calib = token_chunks(tok, args.calib_file, "train", args.calib_seqs, args.seq_len)
    evalx = token_chunks(tok, args.eval_file, "test", args.eval_seqs, args.seq_len)
    split = lambda x: list(x.split(args.batch))
    lab = Lab(model, split(calib), device)
    L, d_ff = lab.L, lab.d_ff
    out = {"model": args.model, "layers": L, "d_ff": d_ff, "eval_tokens": evalx.numel(),
           "calib_tokens": calib.numel(), "calib_file": args.calib_file, "eval_file": args.eval_file}
    print(f"{args.model}: {L} layers, d_ff={d_ff}, calibration {calib.numel()} tokens, evaluation {evalx.numel()} tokens")

    gains, slopes, orth, r2 = fit_law(lab)
    lab.gain = torch.tensor(gains, device=device, dtype=torch.float32)
    out["law"] = {"gains": gains, "slopes": slopes, "orth_ratio": orth, "r2": r2}
    print(f"\nLAW  R^2 {r2:.3f}   median slope {np.median(slopes):.2f} (theory 2)   gain spread {max(gains) / min(gains):.1f}x")
    for l in range(L):
        print(f"  layer {l:2d}  gain {gains[l]:10.4g}  slope {slopes[l]:5.2f}  ||delta||^2 / sum s^2 = {orth[l]:.2f}")

    samples, sample_ids, hot = lab.dense_stats()
    k_res = int(round(args.resident * d_ff))
    resident = torch.zeros(L, d_ff, dtype=torch.bool)
    resident.scatter_(1, hot.topk(k_res, dim=1).indices, True)
    is_res = [resident[l][sample_ids[l]] for l in range(L)]
    ones = [1.0] * L

    lab.set_data(split(evalx))
    SECTION[0] = "compute"
    budgets = [float(b) for b in args.budgets.split(",")]
    rows, add = [], []
    for B in budgets:
        r = lab.run({"kind": "topk", "frac": [B] * L})
        pred = (lab.gain.cpu()[:, None] * r["rho"].pow(2)).sum(0)
        ok = (r["kl"] > 1e-10) & (pred > 1e-12)
        add.append({"budget": B, "measured": r["kl"].mean().item(), "predicted": pred.mean().item(),
                    "log_corr": torch.corrcoef(torch.stack([r["kl"][ok].log(), pred[ok].log()]))[0, 1].item()})
        rows.append(summarize(r, "uniform", B))
        shares = static_split(samples, gains, B)
        rows.append(summarize(lab.run({"kind": "topk", "frac": list(shares)}), "static split", B))
        init = price_at(samples, gains, B)[0]
        lam, _, r = calibrate(lab, lambda x: {"kind": "thr", "lam": x}, B, "kept", init)
        rows.append(summarize(r, "price", B) | {"lam": lam})
        if args.quick:
            continue
        init = price_at(samples, ones, B)[0]
        _, _, r = calibrate(lab, lambda x: {"kind": "thr", "lam": x, "gain": False}, B, "kept", init)
        rows.append(summarize(r, "price, no gains", B))
        r = lab.run({"kind": "thr_tok", "tok_lams": lab.token_prices(B)})
        rows.append(summarize(r, "per-token oracle", B))
        w = np.maximum(shares, 1e-6)
        w_suffix = np.cumsum(w[::-1])[::-1].copy()
        for m in (0.25, 0.5, 1.0):
            plan = {"kind": "cap", "lam": lam * m, "cap": B * L * d_ff, "w": w, "w_suffix": w_suffix}
            rows.append(summarize(lab.run(plan), f"capped price x{m}", B))
    out["additive"] = add
    out["compute"] = rows
    print("\nADDITIVE  (all layers at a uniform share)")
    for a in add:
        print(f"  {a['budget']:.0%}: measured KL {a['measured']:.4f}  predicted {a['predicted']:.4f}  per-token log-corr {a['log_corr']:.2f}")
    print_table("COMPUTE  (budget = average share of FFN neurons computed)", rows, "kept")

    lab.resident = resident.to(device)
    SECTION[0] = "flash"
    rows = []
    out["resident_only"] = summarize(lab.run({"kind": "res_topk", "frac": [0.0] * L}), "resident only", 0.0)
    for Fb in [float(b) for b in args.flash_budgets.split(",")]:
        rows.append(summarize(lab.run({"kind": "res_topk", "frac": [Fb] * L}), "uniform", Fb))
        shares = static_split(samples, gains, Fb, exclude=is_res)
        rows.append(summarize(lab.run({"kind": "res_topk", "frac": list(shares)}), "static split", Fb))
        init = price_at(samples, gains, Fb, exclude=is_res)[0]
        lam, _, r = calibrate(lab, lambda x: {"kind": "res_thr", "lam": x}, Fb, "fetch", init)
        rows.append(summarize(r, "price", Fb) | {"lam": lam})
        w = np.maximum(shares, 1e-6)
        w_suffix = np.cumsum(w[::-1])[::-1].copy()
        for m in (() if args.quick else (0.25, 0.5, 1.0)):
            plan = {"kind": "res_cap", "lam": lam * m, "cap": Fb * L * d_ff, "w": w, "w_suffix": w_suffix}
            rows.append(summarize(lab.run(plan), f"capped price x{m}", Fb))
        if abs(Fb - 0.1) < 1e-9:
            SECTION[0] = "predicted"
            prows = [summarize(lab.run({"kind": "res_topk", "frac": [Fb] * L, "predict": True}), "uniform", Fb),
                     summarize(lab.run({"kind": "res_topk", "frac": list(shares), "predict": True}), "static split", Fb)]
            plam, _, r = calibrate(lab, lambda x: {"kind": "res_thr", "lam": x, "predict": True}, Fb, "fetch", lam)
            prows.append(summarize(r, "price", Fb) | {"lam": plam})
            for m in (() if args.quick else (0.25, 0.5, 1.0)):
                plan = {"kind": "res_cap", "lam": plam * m, "cap": Fb * L * d_ff, "w": w, "w_suffix": w_suffix,
                        "predict": True}
                prows.append(summarize(lab.run(plan), f"capped price x{m}", Fb))
            out["predicted"] = prows
            SECTION[0] = "flash"
    out["flash"] = rows
    print(f"\nRESIDENT ONLY ({args.resident:.0%} hot neurons, nothing fetched): KL {out['resident_only']['kl']:.4f}")
    print_table("FLASH  (budget = average share of FFN neurons fetched from flash; resident ones are free)", rows, "fetch")
    if "predicted" in out:
        print_table("PREDICTED  (flash budget 10%, neurons chosen from an early guess)", out["predicted"], "fetch")
    out["minutes"] = (time.time() - t0) / 60
    json.dump(out, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    torch.save({"seq_len": args.seq_len, "kl": PER_TOKEN}, os.path.join(args.out, "per_token_kl.pt"))
    print(f"\nWrote {args.out}/results.json in {out['minutes']:.1f} min")


def print_table(title, rows, cost):
    print(f"\n{title}")
    print(f"  {'budget':>6} {'policy':>20} {'KL':>9} {'vs uniform':>10} {'top-1':>7} {cost + ' mean':>10} {cost + ' max':>9}")
    base = {}
    for r in rows:
        if r["policy"] == "uniform":
            base[r["budget"]] = r["kl"]
        gain = 1 - r["kl"] / base[r["budget"]]
        print(f"  {r['budget']:6.0%} {r['policy']:>20} {r['kl']:9.4f} {gain:+10.1%} {r['top1']:7.1%} "
              f"{r[cost]:10.3f} {r[cost + '_max']:9.3f}")


if __name__ == "__main__":
    main()
