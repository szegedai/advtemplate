#!/bin/bash

# evaluate the word sets found in a results folder on another model, over all 24
# orderings

# usage: bash 03_run__transfer.sh <model> <source>
model=$1
source=$2

echo "Run transfer: ${model} - source ${source} ..."
python -u run__transfer.py \
    --model "$model" \
    --source "$source" \
    --gpu 0 \
    --save-dir "results/transfer"
