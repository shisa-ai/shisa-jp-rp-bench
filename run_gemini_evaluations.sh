#!/bin/bash

# Directory containing conversation files
CONVERSATIONS_DIR="conversations"

# Check if the conversations directory exists
if [ ! -d "$CONVERSATIONS_DIR" ]; then
    echo "Error: Conversations directory not found!"
    exit 1
fi

# Find all conversation files (assuming they are .jsonl files)
conversation_files=()
while IFS= read -r file; do
    conversation_files+=("$file")
done < <(find "$CONVERSATIONS_DIR" -type f -name "*.jsonl")

echo "Found ${#conversation_files[@]} conversation files to evaluate:"
for file in "${conversation_files[@]}"; do
    echo "- $(basename "$file")"
done
echo "----------------------------------------"

# Run evaluation for each conversation file
for file in "${conversation_files[@]}"; do
    echo "Starting evaluation for: $(basename "$file")"
    echo "Path: $file"
    echo "----------------------------------------"
    
    python judge_conversations.py \
        --judge-model-name gemini/gemini-2.0-flash \
        --target-model "$file"
    
    echo "Completed evaluation for: $(basename "$file")"
    echo "----------------------------------------"
    sleep 2
done

echo "All evaluations completed!" 
