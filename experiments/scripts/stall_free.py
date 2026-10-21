"""stall_free.py: deadlines from how fast the user consumes tokens, not a fixed per-token limit.

A reader or a text-to-speech engine consumes one token every P seconds after a short start-up
delay. A token is late (a stall) only if it is not ready when it is due, so time a token does not
use can be banked for later tokens. Every policy below keeps the reservation rule

    start mandatory work of token t only if it can finish before t is due
    (it always can: the period P is at least the mandatory time M)

so none of them ever stalls. They differ in how many neurons each token fetches from flash:

  fixed      every token gets exactly one period (a per-token deadline, as in AnytimeStream v1)
  greedy     every token may spend all banked time
  steady     ours: aim for the same number of fetched neurons every token, paying more time when
             flash is slow and banking time when it is fast; never spend time the reservation forbids
  price      ours: with K(n) the calibration curve of KL against neurons fetched (convex hull), each
             token fetches the n minimising K(n) + lambda * tau_t * n, i.e. it buys neurons while
             their marginal quality is worth their current flash price; lambda is set so the average
             time per token equals the period, and the reservation rule caps every token
  unified    ours: one price for quality per second across tokens. A neuron of token t is fetched
             if its predicted value is worth mu * tau_t, so hard tokens and fast-flash moments get
             more neurons; the count is capped by the reservation; layers split by the static table
  unified-layers  the same counts, split across layers by each token's own values
  wait-all   no cutoff: every token fetches the neurons it needs (n_target), however long that takes,
             like a streaming runtime without a deadline (stalls are counted, not prevented)

Flash time per neuron follows a bursty two-state process (normal / contended by another app) with
log-normal jitter. Timing is simulated; the quality of every token is measured on the real model
with exactly the neurons that its policy could afford.

usage: stall_free.py --model M --law results.json [--calib-file a --eval-file b] --out DIR
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import price_lab as P  # noqa: E402
from residency import neuron_damage, top_mask  # noqa: E402


class TakeLab(P.Lab):
    """Lab with one more policy: per-token, per-layer fetch counts on top of the resident set."""

    def _decide(self, l, p, value):
        if p["kind"] == "res_take":
            res = self.resident[l].unsqueeze(0).expand_as(value)
            return res | P.masked_topk(value, ~res, self.cur_take[l].to(value.device))
        return super()._decide(l, p, value)

    @torch.no_grad()
    def run_takes(self, takes):
        self.plan = {l: {"kind": "res_take"} for l in range(self.L)}
        kls, agree, fetch = [], [], []
        for b, ref, tk in zip(self.batches, self.ref, takes):
            self.cur_take = tk
            self.rho, self.orth = {}, {}
            n = b.numel()
            self.used = torch.zeros(n, device=self.device)
            self.cost_kept = torch.zeros(n, device=self.device)
            self.cost_fetch = torch.zeros(n, device=self.device)
            ref = tuple(x.to(self.device) for x in ref) if isinstance(ref, tuple) else ref.to(self.device)
            lp = F.log_softmax(self._logits(b), -1)
            kls.append(P.kl_to_ref(ref, lp).flatten().clamp_min(0))
            top = ref[1][..., 0] if isinstance(ref, tuple) else ref.argmax(-1)
            agree.append((top == lp.argmax(-1)).float().flatten())
            fetch.append(self.cost_fetch / (self.L * self.d_ff))
        self.plan = {}
        return torch.cat(kls).cpu(), torch.cat(agree).cpu(), torch.cat(fetch).cpu()


def flash_times(rng, n, tau0, p_enter, p_exit, slow, jitter):
    """Per-token flash time per neuron: two-state Markov (normal / contended) times log-normal jitter."""
    state, out = 0, np.empty(n)
    for t in range(n):
        state = (1 if rng.random() < p_enter else 0) if state == 0 else (0 if rng.random() < p_exit else 1)
        out[t] = tau0 * (slow if state else 1.0) * rng.lognormal(-0.5 * jitter ** 2, jitter)
    return out


def simulate(policy, tau, M, Pd, startup, n_target, n_cap, n_price=None):
    """One token stream. Returns fetched-neuron counts, finish times and stall count."""
    now, n_out, stalls, finish = 0.0, np.zeros(len(tau), dtype=int), 0, np.zeros(len(tau))
    for t, tt in enumerate(tau):
        due = startup + t * Pd
        allowed = max(0.0, due - now - M)          # reservation: mandatory must still fit before `due`
        if policy == "fixed":
            opt = min(allowed, Pd - M)
        elif policy == "greedy":
            opt = allowed
        elif policy == "steady":
            opt = min(allowed, n_target * tt)
        elif policy == "price":
            opt = min(allowed, n_price[t] * tt)
        elif policy == "wait-all":
            opt = n_target * tt                    # fetch what is needed, no cutoff, no reservation
        n = min(n_cap, int(opt / tt))
        spent = M + (n * tt if policy != "fixed" else opt)   # fixed always uses its whole period
        now = max(now, due - Pd) if policy == "fixed" else now
        now += spent
        if now > due + 1e-12:
            stalls += 1
        n_out[t], finish[t] = n, now
    return n_out, finish, stalls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib-file", default=None)
    ap.add_argument("--eval-file", default=None)
    ap.add_argument("--calib-seqs", type=int, default=16)
    ap.add_argument("--eval-seqs", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--law", required=True)
    ap.add_argument("--capacity", type=float, default=0.1, help="RAM-resident share of all FFN neurons")
    ap.add_argument("--resident", default="uniform-freq", choices=["uniform-freq", "global-damage", "hybrid-0.25", "hybrid-0.5"])
    ap.add_argument("--mandatory", type=float, default=0.4, help="mandatory time per token, in periods")
    ap.add_argument("--target", type=float, default=0.1, help="mean flash fetch under 'fixed', share of all neurons")
    ap.add_argument("--cap", type=float, default=0.4, help="most a token can fetch (predicted-active), share")
    ap.add_argument("--startup", type=float, default=2.0, help="first token due after this many periods")
    ap.add_argument("--slow", type=float, default=4.0)
    ap.add_argument("--p-enter", type=float, default=0.1)
    ap.add_argument("--p-exit", type=float, default=0.3)
    ap.add_argument("--jitter", type=float, default=0.3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4, help="sequences per forward pass")
    ap.add_argument("--no-layer-split", action="store_true", help="skip 'unified-layers' (saves memory)")
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
    lab = TakeLab(model, list(calib.split(args.batch)), device)
    L, d_ff, T = lab.L, lab.d_ff, args.seq_len
    gains = json.load(open(args.law))["law"]["gains"]
    lab.gain = torch.tensor(gains, dtype=torch.float32, device=device)
    samples, sample_ids, hot = lab.dense_stats()
    per, total = int(round(args.capacity * d_ff)), int(round(args.capacity * L * d_ff))
    if args.resident == "uniform-freq":
        mask = top_mask(hot, per)
    elif args.resident == "global-damage":
        mask = top_mask(neuron_damage(lab), total=total)
    else:                                              # hybrid: a share of RAM by damage, the rest by frequency
        share = float(args.resident.split("-")[1])
        dm = top_mask(neuron_damage(lab), total=int(round(share * total)))
        mask = dm | top_mask(hot.masked_fill(dm, -1.0), int(round((1 - share) * per)))
    lab.resident = mask.to(device)
    is_res = [mask[l][sample_ids[l]] for l in range(L)]
    w = np.array(P.static_split(samples, gains, args.target, exclude=is_res))
    w = w / w.sum()                                   # per-layer share of each token's fetches
    room = (d_ff - mask.sum(1)).numpy()               # non-resident neurons per layer
    N = L * d_ff
    levels = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
    curve = []
    for f in levels:
        n = int(f * N)
        tk = [torch.from_numpy(np.minimum(np.floor(n * w), room).astype(np.int64))[:, None].expand(L, b.numel()).clone()
              for b in lab.batches]
        curve.append(lab.run_takes(tk)[0].mean().item())
    # quality curve K(n): piecewise-linear in n through the measured points, replaced by its lower
    # convex hull so that the price rule (minimise K(n) + lambda * tau * n) is well defined
    grid = np.linspace(0, levels[-1] * N, 801)
    Kg = np.interp(grid, levels * N, curve)
    hull = [0]
    for i in range(1, len(grid)):
        while len(hull) >= 2:
            i0, i1 = hull[-2], hull[-1]
            if (Kg[i1] - Kg[i0]) * (grid[i] - grid[i0]) >= (Kg[i] - Kg[i0]) * (grid[i1] - grid[i0]):
                hull.pop()
            else:
                break
        hull.append(i)
    Kc = np.interp(grid, grid[hull], Kg[hull])
    print("calibration curve K(n):", " ".join(f"{f:.2f}:{k:.4f}" for f, k in zip(levels, curve)), flush=True)
    lab.set_data(list(evalx.split(args.batch)))
    # per-token neuron values on the evaluation text (idealized predictor, as in price_lab.py);
    # resident neurons are free and excluded. Sorted descending per token.
    lab.store, lab.capture = [], True
    vals, srt = [], []
    with torch.no_grad():
        for b in lab.batches:
            lab.store = []
            lab._logits(b)
            v = torch.stack(lab.store, 1)                        # [tok, L, d_ff]
            v = v.masked_fill(lab.resident.unsqueeze(0), 0.0).cpu()
            lab.store = []
            for c in v.split(64):                                # sort in chunks to bound memory
                srt.append(c.flatten(1).sort(dim=1, descending=True).values.numpy())
            if not args.no_layer_split:
                vals.append(v)
            del v
    lab.capture = False
    del lab.store
    V = torch.cat(vals) if vals else None                        # [n_tok, L, d_ff] (float32)
    S = np.concatenate(srt)                                      # [n_tok, L*d_ff], descending
    del srt
    n_target, n_cap = args.target * N, int(args.cap * N)
    rng = np.random.default_rng(0)
    n_seq = evalx.shape[0]
    taus = [flash_times(rng, T, 1.0, args.p_enter, args.p_exit, args.slow, args.jitter) for _ in range(n_seq)]
    mean_tau = float(np.mean(np.concatenate(taus)))
    tau_scale = 1.0
    Pd = n_target * mean_tau / (1 - args.mandatory)   # 'fixed' spends (1 - mandatory) of each period on flash
    M = args.mandatory * Pd
    print(f"{L}x{d_ff}, resident {args.resident} {args.capacity:.0%}; period {Pd:.3g}, mandatory {M}, "
          f"mean flash time/neuron {mean_tau:.3g}, target fetch {n_target:.0f} of {N}", flush=True)

    allt = np.concatenate(taus)
    def n_of(lam, tt):
        tt = np.atleast_1d(tt)
        return grid[np.argmin(Kc[None, :] + lam * tt[:, None] * grid[None, :], axis=1)]
    lo, hi = -40.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (n_of(np.exp(mid), allt) * allt).mean() > Pd - M else (lo, mid)
    lam = float(np.exp((lo + hi) / 2))
    flat_tau = np.concatenate(taus)                                    # token order = sequence-major, like S
    def n_unified(mu):
        thr = (mu * flat_tau).astype(np.float32)
        cnt = np.concatenate([(S[i:i + 128] >= thr[i:i + 128, None]).sum(1) for i in range(0, len(S), 128)])
        return np.minimum(n_cap, cnt)
    lo2, hi2 = -40.0, 10.0
    for _ in range(80):
        mid = (lo2 + hi2) / 2
        lo2, hi2 = (mid, hi2) if (n_unified(np.exp(mid)) * flat_tau).mean() > Pd - M else (lo2, mid)
    mu = float(np.exp((lo2 + hi2) / 2))
    n_uni = n_unified(mu).reshape(n_seq, T)
    print(f"unified price mu {mu:.3g}: mean desired fetch {n_uni.mean():.0f}", flush=True)
    print(f"price lambda {lam:.3g}: mean desired fetch {n_of(lam, allt).mean():.0f}", flush=True)
    out = {"args": vars(args), "period": Pd, "curve": {"levels": levels.tolist(), "kl": curve},
           "lambda": lam, "rows": []}
    per_token = {}
    policies = ["fixed", "greedy", "steady", "price", "unified"] + ([] if args.no_layer_split else ["unified-layers"])
    for policy in policies + ["wait-all"]:
        counts, stalls, lateness, fins = [], 0, [], []
        for s in range(n_seq):
            want = n_uni[s] if policy.startswith("unified") else n_of(lam, taus[s])
            n, fin, st = simulate("price" if policy.startswith("unified") else policy, taus[s] * tau_scale, M, Pd,
                                  args.startup * Pd, n_target, n_cap, want)
            counts.append(n)
            fins.append(fin)
            stalls += st
            due = args.startup * Pd + np.arange(T) * Pd
            lateness.append(np.maximum(0, fin - due) / Pd)
        counts = np.stack(counts)                      # [seq, T]
        takes = []
        flat = counts.reshape(-1)
        if policy == "unified-layers":
            # each token splits its fetches across layers by its own values: its n-th largest value is the cut
            cut = S[np.arange(len(flat)), np.maximum(flat - 1, 0)]
            cutt = torch.from_numpy(cut).float()
            per_layer = torch.cat([(V[i:i + 128] >= cutt[i:i + 128, None, None]).sum(2) for i in range(0, len(V), 128)]).T.numpy()
            per_layer = np.where(flat[None, :] > 0, per_layer, 0).astype(np.int64)
        else:
            per_layer = np.minimum(np.floor(flat[None, :] * w[:, None]), room[:, None]).astype(np.int64)  # [L, seq*T]
        for bi in range(0, n_seq, args.batch):
            sl = slice(bi * T, min(n_seq, bi + args.batch) * T)
            takes.append(torch.from_numpy(per_layer[:, sl]))
        kl, agree, fetch = lab.run_takes(takes)
        late = np.concatenate(lateness)
        row = {"policy": policy, "kl": kl.mean().item(), "kl_p90": kl.quantile(0.9).item(),
               "kl_p99": kl.quantile(0.99).item(), "top1": agree.mean().item(), "fetch": fetch.mean().item(),
               "fetch_p5": float(np.quantile(flat / N, 0.05)), "stall_rate": stalls / (n_seq * T),
               "max_late_periods": float(late.max())}
        out["rows"].append(row)
        out.setdefault("trace", {})[policy] = {"fetch": (counts[0] / N).tolist(), "finish": (fins[0] / Pd).tolist(),
                                               "kl": kl[:T].tolist()}
        out.setdefault("trace_tau", taus[0].tolist())
        per_token[policy] = kl
        print(f"{policy:>8}: KL {row['kl']:.4f} p90 {row['kl_p90']:.4f} p99 {row['kl_p99']:.4f} top1 {row['top1']:.3f} "
              f"fetch {row['fetch']:.3f} (p5 {row['fetch_p5']:.3f}) stalls {row['stall_rate']:.1%} "
              f"max late {row['max_late_periods']:.1f}P ({time.time() - t0:.0f}s)", flush=True)
    json.dump(out, open(os.path.join(args.out, "stall_free.json"), "w"), indent=1)
    torch.save(per_token, os.path.join(args.out, "per_token_kl.pt"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
