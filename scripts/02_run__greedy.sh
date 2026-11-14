#!/bin/bash

# greedy search on both tasks, 20 seeds, optimizing over the given number of
# orderings

# usage: bash 02_run__greedy.sh <model> <n_permutations>
model=$1
n_permutations=$2

for seed in $(seq 1 20); do
    echo "Run search: FP - seed ${seed} - permutations ${n_permutations} - ${model} ..."
    python -u run__search.py \
        --prompt set_containment__target_fp_01.json \
        --model "$model" \
        --early-stop \
        --n-permutations $n_permutations \
        --seed $seed \
        --gpu 0 \
        --save-dir "results/greedy"
done

for seed in $(seq 1 20); do
    echo "Run search: FN - seed ${seed} - permutations ${n_permutations} - ${model} ..."
    python -u run__search.py \
        --prompt set_containment__target_fn_01.json \
        --model "$model" \
        --early-stop \
        --n-permutations $n_permutations \
        --seed $seed \
        --gpu 0 \
        --save-dir "results/greedy"
done
