#!/bin/bash

# random-walk baseline (search width 1) on both tasks, 20 seeds

# usage: bash 01_run__baseline.sh <model>
model=$1

for seed in $(seq 1 20); do
    echo "Run baseline: FP - seed ${seed} - ${model} ..."
    python -u run__search.py \
        --prompt set_containment__target_fp_01.json \
        --model "$model" \
        --n-steps 25600 \
        --search-width 1 \
        --early-stop \
        --seed $seed \
        --gpu 0 \
        --save-dir "results/baseline" \
        --log-every 256
done

for seed in $(seq 1 20); do
    echo "Run baseline: FN - seed ${seed} - ${model} ..."
    python -u run__search.py \
        --prompt set_containment__target_fn_01.json \
        --model "$model" \
        --n-steps 25600 \
        --search-width 1 \
        --early-stop \
        --seed $seed \
        --gpu 0 \
        --save-dir "results/baseline" \
        --log-every 256
done
