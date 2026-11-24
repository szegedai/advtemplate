#!/bin/bash

# run all experiments for all models, in order (transfer needs the search
# results first)

# usage: bash scripts/run_all.sh (from the repository root)

models=(
    Qwen/Qwen2.5-7B-Instruct
    Qwen/Qwen2.5-14B-Instruct
    Qwen/Qwen2.5-32B-Instruct
    meta-llama/Llama-3.1-8B-Instruct
    mistralai/Mistral-Small-24B-Instruct-2501
)

# sanity check
for model in "${models[@]}"; do
    bash scripts/00_run__sanity_check.sh "$model"
done

# baseline
for model in "${models[@]}"; do
    bash scripts/01_run__baseline.sh "$model"
done

# search (1 permutation)
for model in "${models[@]}"; do
    bash scripts/02_run__greedy.sh "$model" 1
done

# search (4 permutations)
for model in "${models[@]}"; do
    bash scripts/02_run__greedy.sh "$model" 4
done
