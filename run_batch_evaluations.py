import os
import json
import subprocess
import click
import pandas as pd
from collections import defaultdict
import re

def extract_scores_from_file(file_path):
    """Extract scores from a result file."""
    scores = defaultdict(list)
    model_name = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            
            # Get model name if not already set
            if not model_name:
                model_name = result.get('llm')
            
            # Parse the evaluation
            evaluation = result.get('evaluation', '')
            
            # Try to extract JSON from the evaluation
            try:
                # Look for JSON-like content between curly braces
                start_idx = evaluation.find('{')
                end_idx = evaluation.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = evaluation[start_idx:end_idx]
                    # Clean the JSON string
                    import re
                    clean_json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
                    eval_data = json.loads(clean_json_str)
                    
                    # Collect scores for each category
                    for category, score in eval_data.items():
                        if isinstance(score, (int, float)) and category != "Evaluation Reason":
                            scores[category].append(score)
            except:
                continue
    
    # Calculate average scores
    avg_scores = {}
    all_scores = []
    
    for category, category_scores in scores.items():
        if category_scores:
            avg_scores[category] = sum(category_scores) / len(category_scores)
            all_scores.extend(category_scores)
    
    # Calculate overall average
    if all_scores:
        avg_scores['Overall'] = sum(all_scores) / len(all_scores)
    
    return model_name, avg_scores

@click.command()
@click.option('--judge-model-name', '-j', required=True, help='Model name to use for judging the conversations')
@click.option('--num-conversations', '-n', default=20, help='Number of conversations to evaluate per model (max 20)')
def main(judge_model_name, num_conversations):
    """Run evaluations for all models in base_conversations and rank them."""
    # Get all conversation files from base_conversations
    base_dir = "conversations"
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} directory not found")
        return
    
    # Find all JSONL files in the base_conversations directory
    conversation_files = [f for f in os.listdir(base_dir) if f.endswith('.jsonl')]
    
    
    print(f"Found {len(conversation_files)} conversation files to evaluate")
    
    # Create absolute_analysis directory if it doesn't exist
    os.makedirs("absolute_analysis", exist_ok=True)
    
    # Process each file
    for i, file_name in enumerate(conversation_files):
        model_name = os.path.splitext(file_name)[0].replace("_shisa-ai-shisa-rp-bench-testset", "")
        print(f"\n[{i+1}/{len(conversation_files)}] Evaluating {model_name}")
        
        # Check if results already exist
        safe_judge_name = judge_model_name.replace("/", "__")
        safe_model_name = model_name.replace("/", "__")
        result_file = os.path.join("absolute_analysis", f"{safe_model_name}.{safe_judge_name}.jsonl")
        
        if os.path.exists(result_file):
            print(f"  Results already exist at {result_file}, skipping evaluation")
        else:
            # Run the evaluation script
            file_path = os.path.join(base_dir, file_name)
            cmd = [
                "python",
                "judge_conversations.py",
                "--target-model",
                file_path,
                "--judge-model-name",
                judge_model_name,
                "--num-conversations",
                str(num_conversations),
            ]

            print(f"  Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=False)
    
    # Collect results from absolute_analysis directory
    print("\nCollecting results...")
    results = []
    
    analysis_dir = "absolute_analysis"
    safe_judge_name = judge_model_name.replace("/", "__")
    result_files = [f for f in os.listdir(analysis_dir) if f.endswith(f".{safe_judge_name}.jsonl")]
    
    for file_name in result_files:
        file_path = os.path.join(analysis_dir, file_name)
        model_name, scores = extract_scores_from_file(file_path)
        
        if model_name and 'Overall' in scores:
            results.append({
                'Model': model_name.replace("-fsx2-outputs-",""),
                'Overall Score': scores['Overall'],
                **{k: v for k, v in scores.items() if k != 'Overall'}
            })
    
    # Sort results by overall score (descending)
    results.sort(key=lambda x: x['Overall Score'], reverse=True)
    
    # Display ranking
    print("\n" + "="*80)
    print(f"MODEL RANKINGS (Judge: {judge_model_name})")
    print("="*80)
    
    # Create a DataFrame for better display
    df = pd.DataFrame(results)
    
    # Reorder columns to put Overall Score first
    cols = ['Model', 'Overall Score']
    other_cols = [col for col in df.columns if col not in cols]
    df = df[cols + other_cols]
    
    # Display the table
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    
    # Save the results to a CSV file
    output_file = f"model_rankings_{safe_judge_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nRankings saved to {output_file}")


if __name__ == "__main__":
    main() 
