#!/bin/sh
export REF_TOPK=1024
cd /home/claude/anytimestream
python3 experiments/scripts/stall_free.py --model /home/claude/hf/Qwen2.5-0.5B --calib-seqs 8 --eval-seqs 8 --seq-len 256 --batch 2 --no-layer-split --law results/price_lab_qwen05/results.json --resident uniform-freq --threads 2 --out results/stall_free_qwen05 > logs/stall_free_qwen05.log 2>&1
echo DONE > logs/qwen_stall_done
