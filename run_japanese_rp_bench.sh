#!/bin/bash

# Default values
MODEL="${MODEL:-}"  # Use empty string if MODEL is not set
LOW_CONTEXT="${LOW_CONTEXT:-false}"  # Default to false if not set
OPENAI_URL="${OPENAI_URL:-}"  # API URL for the model
OPENAI_COMPATIBLE_API_KEY="${OPENAI_COMPATIBLE_API_KEY:-x}"  # API Key for the model
JUDGE_URL="${JUDGE_URL:-http://athenev2/v1}"  # Default judge API URL
JUDGE_MODEL="${JUDGE_MODEL:-athene-v2}"  # Default judge model

# Validate required arguments
if [ -z "$MODEL" ] || [ -z "$OPENAI_URL" ]; then
    echo "Error: Required environment variables are missing"
    echo "Usage: MODEL=<model_name> OPENAI_URL=<api_url> [JUDGE_URL=<judge_url>] [LOW_CONTEXT=true] [JUDGE_MODEL=model_name] ./$0"
    echo "Example:"
    echo "  MODEL=mistral OPENAI_URL=http://localhost:8000/v1 $0"
    exit 1
fi

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Starting eval script."
log "Generating conversation data..."
# Set environment variables for API endpoints
# We pass this in...
# export OPENAI_COMPATIBLE_API_KEY="x" 
export OPENAI_COMPATIBLE_API_URL="$OPENAI_URL"
export JUDGE_OPENAI_COMPATIBLE_API_KEY="x"
export JUDGE_OPENAI_COMPATIBLE_API_URL="$JUDGE_URL"

# Initialize and activate conda/mamba
source /fsx/ubuntu/miniforge3/etc/profile.d/conda.sh
source /fsx/ubuntu/miniforge3/etc/profile.d/mamba.sh
mamba activate shisa-jp-rp-bench


### We use this to make running multiple jobs in parallel possible

# Generate a unique identifier for this job
JOB_ID="${SLURM_JOB_ID:-$$}"  # Use SLURM job ID if available, otherwise use PID
TIMESTAMP=$(date +%s)
UNIQUE_ID="${JOB_ID}_${TIMESTAMP}"

# Create job-specific directories
TEMP_DIR="./tmp_${UNIQUE_ID}"
mkdir -p "$TEMP_DIR"
mkdir -p "${TEMP_DIR}/configs"

# Create temporary config with model name substituted
envsubst < ./configs/simple_config.yaml > "${TEMP_DIR}/configs/temp_config.yaml"

###

# log "Clearing curator cache before evaluation..."
# rm -rf ~/.cache/curator 2>/dev/null || true
export CURATOR_CACHE_DIR="${TEMP_DIR}/curator_cache"
mkdir -p "$CURATOR_CACHE_DIR"

# Run the benchmark
if [ "$LOW_CONTEXT" = "true" ]; then
    log "Running conversation generator with low context..."
    japanese-rp-bench --config "${TEMP_DIR}/configs/temp_config.yaml" --low-context
else
    japanese-rp-bench --config "${TEMP_DIR}/configs/temp_config.yaml"
fi

log "Successfully generated conversation data. Generating shootout data..."
log "> python generate_shootout_data.py --target-model $MODEL --temp-dir $TEMP_DIR"
python generate_shootout_data.py --target-model "$MODEL" --temp-dir "$TEMP_DIR"

log "Successfully generated shootout data. Evaluating results with Athene..."
log "> python conversation_comparer_any_model.py --base-url $JUDGE_URL --judge-model-name $JUDGE_MODEL --test-model-name $MODEL --temp-dir $TEMP_DIR"
python conversation_comparer_any_model.py --base-url "$JUDGE_URL" --judge-model-name "$JUDGE_MODEL" --test-model-name "$MODEL" --temp-dir "$TEMP_DIR"

log "Successfully evaluated results. Running Bradley-Terry comparision..."
log "> python choix_analyzer.py --target-model $MODEL --judge-model $JUDGE_MODEL --temp-dir $TEMP_DIR"
python choix_analyzer.py --target-model "$MODEL" --judge-model "$JUDGE_MODEL" --temp-dir "$TEMP_DIR"

# Clean up - we leave our temp folders for now...
# rm ./configs/temp_config.yaml