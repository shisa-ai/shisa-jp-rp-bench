#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p ../results

# Loop through all score files
for score_file in scores/*_rp_bench_scores.jsonl; do
    # Extract the model name from the filename
    model_name=$(basename "$score_file" _rp_bench_scores.jsonl)
    
    # Create the target directory
    target_dir="../results/${model_name}"
    mkdir -p "$target_dir"
    
    # Copy the file with the new name
    target_file="${target_dir}/score-shisa-jp-rp-bench.jsonl"
    echo "Copying $score_file to $target_file"
    cp "$score_file" "$target_file"
done

echo "Done copying all score files to multieval results directories"
