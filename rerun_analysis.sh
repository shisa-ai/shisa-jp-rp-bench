#!/bin/bash

# Create analysis directory if it doesn't exist
mkdir -p analysis

# Loop through all answer files
for answer_file in scores/*_rp_bench_answers.jsonl; do
    # Extract the model name from the filename
    model_name=$(basename "$answer_file" _rp_bench_answers.jsonl)
    
    # Create the analysis filename
    analysis_file="analysis/${model_name}.Nexusflow__Athene-V2-Chat.jsonl"
    
    # Copy the file
    echo "Copying $answer_file to $analysis_file"
    cp "$answer_file" "$analysis_file"
    
    # Run the analyzer
    echo "Running analysis for $model_name..."
    python choix_analyzer.py --target-model "$model_name" --judge-model "Nexusflow/Athene-V2-Chat"
    echo "----------------------------------------"
done
