#!/bin/bash

# Default values
MODEL="${MODEL:-}"  # Use empty string if MODEL is not set
LOW_CONTEXT="${LOW_CONTEXT:-false}"  # Default to false if not set
OPENAI_URL="${OPENAI_URL:-}"  # API URL for the model
JUDGE_URL="${JUDGE_URL:-http://athenev2/v1}"  # Default judge API URL
JUDGE_MODEL="${JUDGE_MODEL:-Nexusflow/Athene-V2-Chat}"  # Default judge model

# Validate required arguments
if [ -z "$MODEL" ] || [ -z "$OPENAI_URL" ]; then
    echo "Error: Required environment variables are missing"
    echo "Usage: MODEL=<model_name> OPENAI_URL=<api_url> [JUDGE_URL=<judge_url>] [LOW_CONTEXT=true] [JUDGE_MODEL=model_name] ./$0"
    echo "Example:"
    echo "  MODEL=mistral OPENAI_URL=http://localhost:8000/v1 ./$0"
    exit 1
fi

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Starting eval script."
log "Generating conversation data..."
# Set environment variables for API endpoints
export OPENAI_COMPATIBLE_API_KEY="cat"
export OPENAI_COMPATIBLE_API_URL="$OPENAI_URL"
export JUDGE_OPENAI_COMPATIBLE_API_KEY="cat"
export JUDGE_OPENAI_COMPATIBLE_API_URL="$JUDGE_URL"

# Initialize and activate conda/mamba
source /fsx/ubuntu/miniforge3/etc/profile.d/conda.sh
source /fsx/ubuntu/miniforge3/etc/profile.d/mamba.sh
mamba activate shisa-rp-bench

# Create temporary config with model name substituted
envsubst < ./configs/simple_config.yaml > ./configs/temp_config.yaml

# Run the benchmark
if [ "$LOW_CONTEXT" = "true" ]; then
    log "Running conversation generator with low context..."
    japanese-rp-bench --config ./configs/temp_config.yaml --low-context
else
    japanese-rp-bench --config ./configs/temp_config.yaml
fi
log "Successfully generated conversation data. Generating shootout data..."
python generate_shootout_data.py --target-model "$MODEL"
log "Successfully generated shootout data. Evaluating results with Athene..."
python conversation_comparer_any_model.py --base-url "$JUDGE_URL" --judge-model-name "$JUDGE_MODEL" --test-model-name "$MODEL"
log "Successfully evaluated results. Running Bradley-Terry comparision..."
python choix_analyzer.py --target-model "$MODEL" --judge-model "$JUDGE_MODEL"

# Clean up
rm ./configs/temp_config.yaml