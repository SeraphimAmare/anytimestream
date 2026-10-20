# Layer budget experiments (October 2026)

Question: when a token can only afford part of its FFN work, should every layer get the same share?
Answer on a 12-layer model trained here: no. A fixed per-layer split beats equal shares by 5 to 24% in KL,
and re-deciding the split per token does not help, even for an oracle. Full write-up: the Road Guide doc.

## Files

| Path | What it is |
|---|---|
| `experiments/scripts/prep_data.py` | Builds a 4,096-token BPE tokenizer from 16 Gutenberg books; holds out Alice and Persuasion |
| `experiments/scripts/train_lm.py` | Trains the 10.1M-parameter, 12-layer Qwen2-architecture model (40 min on 2 CPU cores) |
| `experiments/scripts/price_lab.py` | Main experiments: damage law, additivity, equal-compute and equal-flash comparisons, predicted scores |
| `experiments/scripts/followup.py` | Single-layer damage, the per-layer shares, the per-token oracle at matched budget, simulator weights |
| `experiments/scripts/heuristics.py` | The calibrated split against simple rules (protect first and last layers, damage-proportional) |
| `experiments/scripts/paired_ci.py` | Paired 95% bootstrap intervals over evaluation passages |
| `experiments/scripts/plot_price_lab.py` | Quick matplotlib overview of `price_lab.py` results |
| `experiments/kit/` | The starter kit, updated: float32 on CPU, tolerant text reading. `anytime_sim.py` is the deadline simulator |
| `results/` | Every number in the guide: JSON summaries, per-token KL files, plots |

## Reproduce on your computer (PowerShell, from this folder, venv active)

```powershell
python -m pip install torch transformers datasets matplotlib numpy packaging tokenizers
Invoke-WebRequest -Uri https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip -OutFile gutenberg.zip
Expand-Archive gutenberg.zip -DestinationPath data
python experiments/scripts/prep_data.py data/gutenberg data/tok
python experiments/scripts/train_lm.py data/tok models/gutenberg-12L 40
python experiments/scripts/price_lab.py --model models/gutenberg-12L --calib-file data/gutenberg/austen-persuasion.txt --eval-file data/gutenberg/carroll-alice.txt --calib-seqs 16 --eval-seqs 32 --out results/price_lab_12L
python experiments/scripts/followup.py models/gutenberg-12L data/gutenberg/austen-persuasion.txt data/gutenberg/carroll-alice.txt results/price_lab_12L results/followup_12L
python experiments/scripts/heuristics.py models/gutenberg-12L data/gutenberg/austen-persuasion.txt data/gutenberg/carroll-alice.txt results/followup_12L
python experiments/scripts/paired_ci.py results/price_lab_12L
```

Deadline simulator with the calibrated split (from `experiments/kit/`):

```powershell
cd experiments/kit
python profile_ffn.py --model ../../models/gutenberg-12L --text-file ../../data/gutenberg/austen-persuasion.txt --n-seqs 16 --seq-len 256 --out ../../results/kit_profile
python anytime_sim.py --model ../../models/gutenberg-12L --text-file ../../data/gutenberg/carroll-alice.txt --n-seqs 4 --seq-len 256 --profile ../../results/kit_profile/freq.pt --weights ../../results/followup_12L/layer_weights.json --policies slack,weighted --out ../../results/sim_static_split
```

Training takes about 40 minutes and the experiments about 10 on a laptop CPU. Results will differ slightly
from mine because training is not bit-for-bit reproducible across machines.

## Run it on a real model (GPU or Google Colab)

WikiText-2 is used automatically: the train split calibrates, the test split evaluates.

```bash
python experiments/scripts/price_lab.py --model Qwen/Qwen2.5-0.5B --calib-seqs 8 --eval-seqs 16 --out results/price_lab_qwen05
python experiments/scripts/followup.py Qwen/Qwen2.5-0.5B wikitext wikitext results/price_lab_qwen05 results/followup_qwen05
python experiments/scripts/heuristics.py Qwen/Qwen2.5-0.5B wikitext wikitext results/followup_qwen05
python experiments/scripts/paired_ci.py results/price_lab_qwen05
```

Memory note: the reference distributions are kept in CPU RAM (eval tokens x vocabulary x 4 bytes). With
Qwen's 151,936-token vocabulary, 16 evaluation passages of 256 tokens need about 2.5 GB.
