#!/bin/bash

# Array of models to evaluate
models=(
    #SakanaAI/TinySwallow-1.5B-Instruct"
    #"meta-llama/Meta-Llama-3.1-8B-Instruct"
    #"shisa-ai/shisa-v1-llama3-8b"
    #"allenai/Llama-3.1-Tulu-3-8B"
    #"shisa-ai/shisa-v1-llama3-8b"
    #"cyberagent/Mistral-Nemo-Japanese-Instruct-2408"
    #"tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
    #"microsoft/phi-4"
    #"cyberagent/calm3-22b-chat"
    #"Aratako/calm3-22b-RP-v2" 
    "tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3"
    #"weblab-GENIAC/Tanuki-8B-dpo-v1.0"
)
#I will also copy in what's in /home/wooser-shisa/remote-mount/meti/eval/shisa-jp-rp-bench/conversations

# Run each model
for model in "${models[@]}"; do
    echo "Running benchmark for model: $model"
    MODEL="$model" LOW_CONTEXT="true" srun run_japanese_rp_bench.slurm --low-context
    echo "Completed benchmark for model: $model"
    echo "----------------------------------------"
done
