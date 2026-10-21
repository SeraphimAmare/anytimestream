#!/bin/sh
R=/home/claude/research
cd /home/claude/anytimestream
run() { name=$1; shift; python3 experiments/scripts/stall_free.py --model $R/models/gutenberg-12L --calib-file $R/data/gutenberg/austen-persuasion.txt --eval-file $R/data/gutenberg/carroll-alice.txt --law results/price_lab_12L/results.json --resident uniform-freq --threads 1 --out results/stall_free_sweep/$name "$@" > logs/sweep_$name.log 2>&1; }
run slow1 --slow 1
run slow2 --slow 2
run slow8 --slow 8
run startup0 --startup 0
run startup8 --startup 8
run target05 --target 0.05
run target20 --target 0.2
run mand20 --mandatory 0.2
run mand70 --mandatory 0.7
echo SWEEPDONE > logs/sweep_done
