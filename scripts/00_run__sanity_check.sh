#!/bin/bash

# short search on both tasks to check that a model works with the code

# usage: bash 00_run__sanity_check.sh <model>
model=$1

echo "Run sanity check: FP - ${model} ..."
python -u run__search.py \
    --prompt set_containment__target_fp_01.json \
    --model "$model" \
    --n-steps 3 \
    --save-dir "results/sanity_check" \
    --seed 1

echo "Run sanity check: FN - ${model} ..."
python -u run__search.py \
    --prompt set_containment__target_fn_01.json \
    --model "$model" \
    --n-steps 3 \
    --save-dir "results/sanity_check" \
    --seed 1
