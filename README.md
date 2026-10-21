# AnytimeStream

**Stall-free anytime decoding for flash-streamed on-device LLMs.**
Undergraduate research by Seraphim Amare at George Mason University, with Prof. Shuochao Yao.
It builds on the lab's LeanStream (MobiCom 2026) and on Yao et al.'s real-time DNNs as imprecise
computations.

People read or listen to generated text at a steady pace, so a token only has to be ready when the
person reaches it. AnytimeStream treats decoding as a hierarchical imprecise computation:

- **mandatory:** the FFN neurons already in RAM, which never wait on flash
- **optional:** the neurons streamed from flash, fetched while time remains

A two-level reservation guarantees that no token is late, whatever flash does. Within that
guarantee, one price for quality decides which neurons are worth fetching, across tokens and layers.

## What is new, and what is not

A literature check (October 2026) found that splitting the budget across layers by sensitivity is
prior art (TEAL, ICLR 2025; SpAx and TopK-Guided, 2026), so it is used here as an ingredient, not
claimed. The contributions are:

1. **Consumption deadlines with a hard guarantee** (`anytimestream/stream.py` + `scheduler.py`). Before
   token t starts, its optional time is capped at its due time minus the mandatory time. If the
   mandatory time fits in one consumption period, no token stalls, even when flash reads are slow,
   lost or non-preemptive. `tests/` checks both levels against random and heavy-tailed timings.
2. **One price for quality across tokens** (`experiments/scripts/stall_free.py`, policy `unified`). A
   neuron of token t is fetched if its predicted value is worth `mu * tau_t`, its current flash cost.
   Hard tokens get more neurons and easy ones bank time. A fixed per-token deadline cannot do this.
3. **Designing the mandatory part** (`experiments/scripts/residency.py`). Choosing what stays in RAM
   by expected damage rather than activation frequency sets the quality floor the guarantee can promise.

## Results so far

**12-layer, 10.1M-parameter Qwen2-architecture model, tested on *Alice*.** The deadline and flash
model are simulated, and quality (KL to the dense model) is measured on the model itself.

| Test | Result |
|---|---|
| Unified price vs fixed per-token deadline, same average time per token | p99 KL -17.0% (95% CI -10.6 to -21.7%), p90 -9.8%, mean -3.1%; 0 stalls for both |
| No cutoff, same average time (LeanStream-style) | Late on 41% of tokens, up to 38 periods behind |
| Gain vs burstiness of flash (slowdown 1x / 2x / 4x / 8x) | p99 -1% / -8% / -18% / -23% |
| Gain vs budget (5% / 10% / 20% of neurons fetched) | p99 -7% / -18% / -39% |
| RAM residency, 10% of neurons, RAM only | KL 3.44 by frequency, 0.79 by damage, 0.83 hybrid (half each) |
| Same, plus 10% flash reads | 0.0692 by frequency, 0.0757 by damage, 0.0698 hybrid |

**Qwen2.5-0.5B, WikiText-2** (32 evaluation sequences in 4 shards; `experiments/runs/`).

| Test | Result |
|---|---|
| Unified price vs fixed per-token deadline | p99 KL -7.1% (95% CI -1.2 to -13.4%); mean -1.5% and p90 +1.0%, both not significant; 0 stalls |
| No cutoff, same average time | Late on 44% of tokens, up to 28 periods behind; mean KL 17% worse than the fixed deadline |
| Damage law (one layer cut at a time) | Loose per token: R² 0.16, median slope 1.39 (theory 2); gains rise with depth, last layer highest |
| Fixed per-layer split vs equal shares (known idea) | -5.2% / -5.3% / -1.0% KL at 10% / 20% / 30% compute |
| RAM residency, 10% of neurons, RAM only | KL 8.53 by frequency, 4.97 hybrid: both too weak to promise useful quality |

On the real model the guarantee holds and the tail improves, but less than on the toy model, because
the damage law ranks neurons only roughly. The next steps are better per-token neuron values, and a
stronger mandatory part, such as a compressed copy of each layer in RAM.

Caveats: flash timing is simulated with a bursty two-state model, the predictor is idealized (true
neuron values), and quality is measured teacher-forced. `ROADMAP.md` lists how each gets tested.

## Layout

| Path | What it is |
|---|---|
| `anytimestream/scheduler.py` | Layer-level reservation scheduler: mandatory first, reads dropped at each layer's cutoff |
| `anytimestream/stream.py` | Token-level reservation against consumption deadlines, and the price rule |
| `anytimestream/budget_table.py` | Per-layer budget table (the known per-layer split) |
| `tests/` | Guarantee tests for both levels (`python -m pytest -q tests`) |
| `experiments/scripts/stall_free.py` | Policies under reader deadlines, quality measured on the model |
| `experiments/scripts/residency.py` | What to keep in RAM, and the guaranteed floor |
| `experiments/scripts/price_lab.py`, `followup.py`, ... | October layer-budget experiments (damage law, splits, oracle) |
| `experiments/runs/` | The exact command chains used for the 12-layer sweep and the Qwen runs |
| `results/` | Every number above: JSON summaries and per-token KL |

## Quick start

```bash
python -m pip install torch transformers tokenizers datasets numpy pytest
python -m pytest -q tests
python experiments/scripts/stall_free.py --model Qwen/Qwen2.5-0.5B --law results/price_lab_qwen05/results.json \
    --calib-seqs 8 --eval-seqs 8 --batch 2 --out results/stall_free_qwen05
```

For Qwen's 151,936-token vocabulary, set `REF_TOPK=1024`. The reference distributions are then kept
as their top 1,024 tokens plus a rest bucket, which keeps memory low.
