#!/bin/sh
# Qwen2.5-0.5B stall-free in 4 shards of 8 sequences (32 total), pooled for confidence intervals
export REF_TOPK=1024
cd /home/claude/anytimestream
for k in 0 1 2 3; do
python3 experiments/scripts/stall_free.py --model /home/claude/hf/Qwen2.5-0.5B --calib-seqs 8 --eval-seqs 8 --nshards 4 --shard $k --seq-len 256 --batch 2 --no-layer-split --law results/price_lab_qwen05/results.json --resident uniform-freq --threads 2 --out results/stall_free_qwen05_shard$k > logs/stall_free_qwen05_shard$k.log 2>&1
done
echo DONE > logs/qwen_shards_done
