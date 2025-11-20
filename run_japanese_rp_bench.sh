#!/bin/bash

# Default values
MODEL="${MODEL:-}"  # Use empty string if MODEL is not set
LOW_CONTEXT="${LOW_CONTEXT:-false}"  # Default to false if not set
OPENAI_URL="${OPENAI_URL:-}"  # API URL for the model
OPENAI_COMPATIBLE_API_KEY="${OPENAI_COMPATIBLE_API_KEY:-EMPTY}"  # API Key for the model
JUDGE_URL="${JUDGE_URL:-http://athenev2/v1}"  # Default judge API URL (for external judge endpoints, if used)
JUDGE_MODEL="${JUDGE_MODEL:-gemini/gemini-2.0-flash}"  # Default judge model label
JUDGE_OPENAI_COMPATIBLE_API_KEY="${JUDGE_OPENAI_COMPATIBLE_API_KEY:-EMPTY}"

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
export OPENAI_COMPATIBLE_API_KEY
export OPENAI_COMPATIBLE_API_URL="$OPENAI_URL"
export JUDGE_OPENAI_COMPATIBLE_API_KEY
export JUDGE_OPENAI_COMPATIBLE_API_URL="$JUDGE_URL"


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

# Run the benchmark (conversation generation only)
if [ "$LOW_CONTEXT" = "true" ]; then
    log "Running conversation generator with low context..."
    japanese-rp-bench --config "${TEMP_DIR}/configs/temp_config.yaml" --low-context
else
    japanese-rp-bench --config "${TEMP_DIR}/configs/temp_config.yaml"
fi

# Skip absolute evaluation if requested (multieval runner handles judging)
if [ "$SKIP_ABSOLUTE_EVAL" != "true" ]; then
    log "Running absolute evaluation with Gemini..."
    log "> CURATOR_DISABLE_CACHE=true python conversation_judge_absolute_evaluator.py --judge-model-name gemini/gemini-2.0-flash --target-model $MODEL"
    CURATOR_DISABLE_CACHE=true python conversation_judge_absolute_evaluator.py --judge-model-name gemini/gemini-2.0-flash --target-model "$MODEL"
else
    log "Skipping absolute evaluation (SKIP_ABSOLUTE_EVAL=true)"
fi

# Clean up - we leave our temp folders for now...
# rm ./configs/temp_config.yaml
