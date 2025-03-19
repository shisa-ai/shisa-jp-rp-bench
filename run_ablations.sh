#!/bin/bash

# Get all models from the /fsx2/outputs/ directory
model_paths=()
while IFS= read -r model_path; do
    model_paths+=("$model_path")
done < <(find /fsx2/outputs/ -type d -maxdepth 1 -mindepth 1)

echo "Found ${#model_paths[@]} models to evaluate:"
for path in "${model_paths[@]}"; do
    echo "- $(basename "$path") (path: $path)"
done
echo "----------------------------------------"

# Array of models that need ultra-low context
ultra_low_context_models=(
    "augmxnt/shisa-7b-v1"
    "augmxnt/shisa-gamma-7b-v1"
    "elyza/ELYZA-japanese-Llama-2-7b-instruct"
    "Deepreneur/blue-lizard"
    "DataPilot/ArrowPro-7B-KUJIRA"
    "tokyotech-llm/Swallow-7b-instruct-v0.1"
)

# Run all models
for model_path in "${model_paths[@]}"; do
    model_name=$(basename "$model_path")
    
    echo "Starting benchmark for model: $model_name"
    echo "Path: $model_path"
    echo "----------------------------------------"
    
    # Check if this model needs ultra-low context
    if [[ " ${ultra_low_context_models[@]} " =~ " ${model_name} " ]]; then
        echo "Using ultra-low context (4096) for this model"
        MODEL="$model_path" ULTRA_LOW_CONTEXT="true" sbatch generate_conversation_data.slurm --ultra-low-context
    else
        echo "Using standard context settings"
        MODEL="$model_path" LOW_CONTEXT="true" sbatch generate_conversation_data.slurm --low-context
    fi
    
    echo "Submitted job for model: $model_name"
    echo "----------------------------------------"
    sleep 5
done 