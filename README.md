# AnytimeStream

**Deadline-guaranteed decoding for flash-streamed LLMs, with sensitivity-budgeted layers.**
Undergraduate research by Seraphim Amare at George Mason University, with Prof. Shuochao Yao.
It builds on the lab's LeanStream (MobiCom 2026).

LeanStream makes on-device LLMs fast on average by streaming feed-forward weights from flash.
AnytimeStream makes every token meet a hard deadline. It then spends the leftover time where it
buys the most quality: across layers, in proportion to how fragile each layer is, measured once per model.

## The three parts

1. **Reservation guarantee** (`anytimestream/scheduler.py`). Each layer's mandatory work uses only
   weights already in RAM, and time for every remaining layer's mandatory work is reserved first.
   Flash reads that miss a layer's cutoff are abandoned. So no token is late, however slow flash is
   or however wrong the predictor is. `tests/test_scheduler.py` checks this against 5,000 random cases
   with heavy-tailed and lost reads.
2. **Per-layer budget table** (`anytimestream/budget_table.py`, `scripts/build_budget_table.py`).
   Each layer's fragility is measured once, using the damage law KL ≈ G_l·ρ_l². That gives a split
   of the spare budget at each budget level, stored as a small JSON table.
3. **Cache-aware costs.** In flash mode, neurons already resident in RAM cost nothing. Only flash
   reads count against the budget.

Not claimed: re-splitting the budget for every token. In the experiments below it lost to the fixed
table, even as an oracle with perfect hindsight.

## Results so far (10.1M-parameter, 12-layer Qwen2-architecture model)

| Test | Result |
|---|---|
| One layer at a time cut to 20% | Layer 0 is 18x as fragile as layer 2; the last layer is 9x |
| Damage law | KL grows as ρ^2 (fitted exponent 1.93 to 2.18 in 11 of 12 layers) |
| Fixed split vs equal shares | 5 to 10% lower KL at equal compute; 8 to 24% at equal flash reads |
| Per-token re-split (oracle) | 3 to 4% worse than the fixed split |
| Deadline simulator | 12 to 27% lower KL at deadlines up to 2.2 ms, 0 missed deadlines; waiting for every weight missed over 99% up to 4.2 ms |

Caveats: this is a toy model, KL is measured teacher-forced, the simulator's timings are not
calibrated, and the predictor is idealized. In 2 of 3 compute-only tests, a simple rule (double the
first and last layers) beat the split. See `ROADMAP.md` for how each caveat gets tested.

## Layout

| Path | What it is |
|---|---|
| `anytimestream/` | The runtime pieces: reservation scheduler, budget table |
| `scripts/build_budget_table.py` | Builds a model's budget table (compute and flash modes) |
| `scripts/demo_guarantee.py` | Waiting for every weight vs reservation, under heavy-tailed flash (illustrative timings) |
| `tests/` | Deadline-guarantee tests (`python -m pytest -q tests`) |
| `experiments/scripts/` | The October 2026 layer-budget experiments (damage law, splits, oracle, heuristics, CIs) |
| `experiments/kit/` | Starter kit: FFN profiler, top-k oracle, deadline co-simulator |
| `results/` | Every number above: JSON summaries, per-token KL, plots, budget table |

## Quick start (Windows PowerShell, venv active)

```powershell
python -m pip install torch transformers datasets matplotlib numpy packaging tokenizers pytest
python -m pytest -q tests
python scripts/demo_guarantee.py --table results/budget_table_12L.json
```

Build a table for a real model (GPU or Colab; WikiText-2 is used automatically):

```bash
python scripts/build_budget_table.py --model Qwen/Qwen2.5-0.5B --out results/budget_table_qwen05.json
```

To reproduce the toy-model experiments, follow `experiments/README.md`.
