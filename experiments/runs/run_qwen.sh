#!/bin/sh
export REF_TOPK=1024
export BATCH=2
Q=/home/claude/hf/Qwen2.5-0.5B
cd /home/claude/anytimestream
python3 experiments/scripts/price_lab.py --model $Q --calib-seqs 8 --eval-seqs 12 --seq-len 256 --batch 2 --threads 2 --quick --out results/price_lab_qwen05 > logs/price_lab_qwen05.log 2>&1 || exit 1
python3 experiments/scripts/followup.py $Q wikitext wikitext results/price_lab_qwen05 results/followup_qwen05 > logs/followup_qwen05.log 2>&1
python3 experiments/scripts/residency.py --model $Q --calib-seqs 8 --eval-seqs 12 --seq-len 256 --law results/price_lab_qwen05/results.json --batch 2 --threads 2 --out results/residency_qwen05 > logs/residency_qwen05.log 2>&1
python3 experiments/scripts/stall_free.py --model $Q --calib-seqs 8 --eval-seqs 8 --seq-len 256 --law results/price_lab_qwen05/results.json --resident uniform-freq --batch 2 --threads 2 --out results/stall_free_qwen05 > logs/stall_free_qwen05.log 2>&1
echo ALLDONE >> logs/price_lab_qwen05.log
