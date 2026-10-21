#!/bin/sh
# Qwen2.5-0.5B: residency and stall-free, using the law rebuilt from the price_lab log
export REF_TOPK=1024
Q=/home/claude/hf/Qwen2.5-0.5B
cd /home/claude/anytimestream
python3 experiments/scripts/residency.py --model $Q --calib-seqs 8 --eval-seqs 12 --seq-len 256 --batch 2 --capacities 0.1 --flash 0,0.05,0.1 --strategies uniform-freq,global-freq,global-damage,hybrid-0.5 --law results/price_lab_qwen05/results.json --threads 2 --out results/residency_qwen05 > logs/residency_qwen05.log 2>&1
python3 experiments/scripts/stall_free.py --model $Q --calib-seqs 8 --eval-seqs 8 --seq-len 256 --batch 2 --law results/price_lab_qwen05/results.json --resident uniform-freq --threads 2 --out results/stall_free_qwen05 > logs/stall_free_qwen05.log 2>&1
echo V2DONE > logs/qwen_v2_done
