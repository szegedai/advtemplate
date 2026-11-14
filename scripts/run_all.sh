#!/bin/bash

# run all experiments for all models, in order (transfer needs the search
# results first)

models=(
    Qwen/Qwen2.5-7B-Instruct
    Qwen/Qwen2.5-14B-Instruct
    Qwen/Qwen2.5-32B-Instruct
    meta-llama/Llama-3.1-8B-Instruct
    mistralai/Mistral-Small-24B-Instruct-2501
)

# sanity check
for model in "${models[@]}"; do
    sbatch 00_run__sanity_check.sbatch "$model"
done

# baseline
for model in "${models[@]}"; do
    sbatch 01_run__baseline.sbatch "$model"
done

# search (1 permutation)
for model in "${models[@]}"; do
    sbatch 02_run__greedy.sbatch "$model" 1
done

# search (4 permutations)
for model in "${models[@]}"; do
    sbatch 02_run__greedy.sbatch "$model" 4
done

# transferability
for model in "${models[@]}"; do
    for source in results/greedy results/baseline; do
        sbatch 03_run__transfer.sbatch "$model" "$source"
    done
done

# word scan
for model in "${models[@]}"; do
    for prompt in set_containment__target_fn_01.json set_containment__target_fp_01.json; do
        sbatch 04_run__word_scan.sbtach "$model" "$prompt"
    done
done
