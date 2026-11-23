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

prompts=(
    set_containment__target_fn_01.json
    set_containment__target_fp_01.json
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

# transferability
for model in "${models[@]}"; do
    for source in results/greedy results/baseline; do
        bash scripts/03_run__transfer.sh "$model" "$source"
    done
done

# word scan
for model in "${models[@]}"; do
    for prompt in "${prompts[@]}"; do
        bash scripts/04_run__word_scan.sh "$model" "$prompt"
    done
done
