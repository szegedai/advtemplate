#!/bin/bash

# score every vocabulary word as the query in random word sets, to measure the
# base error rate

# usage: bash 04_run__word_scan.sh <model> <prompt>
model=$1
prompt=$2

echo "Run word scan: ${model} - prompt ${prompt} ..."
python -u run__word_scan.py \
    --model "$model" \
    --prompt "$prompt" \
    --gpu 0 \
    --save-dir "results/word_scan"
