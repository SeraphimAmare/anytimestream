# Anytime LLM starter kit

Research starter code for deadline-guaranteed anytime decoding on flash-backed sparse LLMs
(extending the LeanStream speculate-and-refine design with per-token deadlines).

Everything runs in PyTorch on one GPU. Quality numbers come from real forward passes; timing
comes from a virtual-clock model of flash I/O and GPU compute that you calibrate later. That split
lets you study the scheduling problem before writing any low-level I/O or kernel code.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # install the CUDA build of torch for your GPU first if needed
python profile_ffn.py --model tiny --device cpu   # 10-second smoke test, no download
```

Qwen2.5 models download directly. Llama and Mistral require accepting their license on Hugging Face,
then `hf auth login` (older versions: `huggingface-cli login`).
Memory: Qwen2.5-1.5B needs a GPU with about 8 GB; 7B models in bf16 want about 24 GB.

## Run order

```bash
M=Qwen/Qwen2.5-1.5B
python profile_ffn.py  --model $M                                   # 1. sparsity + hot neurons
python oracle_topk.py  --model $M --sensitivity                     # 2. quality ceiling + layer weights
python anytime_sim.py  --model $M --profile results/profile/freq.pt \
                       --weights results/oracle/sensitivity.json    # 3. policies vs deadlines
python generate_demo.py --model $M --weights results/oracle/sensitivity.json --policy weighted
```

Add `--model tiny --device cpu` to any script for a quick check that the code path works.

## What to look for

| Output | Question it answers |
|---|---|
| `profile/concentration.png` | How fast does FFN output error fall as you keep more neurons? Sets a sane sparsity. |
| `profile/frequency.png` | Is there a hot set worth pinning in DRAM? Steep curves say yes. |
| `oracle/oracle_curve.png` | The quality ceiling at each sparsity: no runtime can beat the oracle at equal work. |
| `oracle/sensitivity.png` | Do layers differ in how much damage skipping causes? Feeds the `weighted` policy. |
| `sim/prediction_ladder.png` | How much better do predictions get from stale to refined to fresh context? |
| `sim/quality_vs_deadline.png` | Quality per deadline, and misses. `slack`/`weighted`/`adaptive` never miss by construction. |

## Policies (in `anytime_sim.py`)

- `nocut`: waits for every predicted-active block (LeanStream-like: best quality, no deadline).
- `static`: fixed per-layer cutoffs.
- `slack`: reserves mandatory time for all remaining layers, gives this layer a share of the rest.
  Zero misses whenever the deadline is feasible, regardless of flash speed or predictor accuracy.
- `weighted`: `slack` with per-layer sensitivity weights.
- `adaptive`: `weighted`, plus each token's share follows how much predicted mass the layer is missing.

Add your own by subclassing `Policy`; only `cutoff()` (and optionally `observe()`) is needed.

## Modeling assumptions (know these before trusting numbers)

- Hardware constants are placeholders. Calibrate flash with `fio` (bandwidth vs request size and
  queue depth) and GPU times with a profiler, then set `--flash-gbps`, `--flash-half-kb`, `--dram-gbps`.
- Default evaluation is teacher-forced: each position is simulated as one decode step with the true
  prefix. `generate_demo.py` runs free-running generation (set `sim.persist_cache = True`).
- Caches are per-layer LRU; I/O runs at most one layer ahead; predictions come in three rungs
  (stale, refined, fresh) rather than LeanStream's continuous refinement; KV-cache read time and
  prefill are not modeled separately.
- Stale and refined predictions use the real next-layer weights on older residuals: an idealized
  predictor that isolates context staleness. `--pred-noise` adds predictor error.
- `--group` packs neurons contiguously, not by co-activation.
- The guarantee assumes the per-layer mandatory bound holds on your hardware (measure it) and that
  the GPU computes a block faster than flash delivers one.

## Natural next steps

1. Better controllers: value-curve allocation, or an offline DP / MPC lookup table.
2. A mandatory floor for non-resident neurons (low-bit or low-rank fallback).
3. A learned predictor instead of oracle-plus-noise.
4. Co-activation clustering for `--group`.
5. Calibration, then a real I/O prototype, then integration into the LeanStream runtime.
