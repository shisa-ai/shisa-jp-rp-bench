import os
import json
import hashlib
from io import StringIO
from datasets import load_dataset, Dataset
import click
from bespokelabs import curator
import numpy as np
import shutil

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_single_conversation(conv_data, dataset_row):
    """Format a single conversation into a markdown document with settings."""
    output = StringIO()
    
    # Write all settings under main settings heading
    output.write("# 設定\n\n")
    
    # Define the columns and their Japanese titles
    columns = {
        'id': 'データのid',
        'genre': 'ロールプレイのジャンル',
        'world_setting': 'ロールプレイの世界観設定',
        'scene_setting': 'ロールプレイのシーン設定',
        'user_setting': 'ロールプレイのユーザー側キャラクター設定',
        'assistant_setting': 'ロールプレイのアシスタント側キャラクター設定',
        'dialogue_tone': 'ロールプレイの対話のトーン',
        'first_user_input': 'ロールプレイの最初のユーザー発話',
        'response_format': 'ロールプレイの応答形式'
    }
    
    # Write each column title and content
    for col, jp_title in columns.items():
        output.write(f"## {jp_title}\n")
        output.write(f"{dataset_row[col]}\n\n")
    
    # Write separator for conversation
    output.write("---\n\n")
    
    # Write conversation
    output.write("# 会話\n\n")
    for i, message in enumerate(conv_data.get("conversation_history", [])):
        header = "### User" if i % 2 == 0 else "### Assistant"
        output.write(f"{header}\n{message}\n\n")
    
    # Add final separator
    output.write("\n---\n")
    
    return output.getvalue()

def generate_formatted_conversations(target_file, num_conversations=None):
    """Generate formatted conversations from the target file."""
    # Determine the correct directory and file path
    if os.path.exists(target_file):
        # If the full path was provided and exists, use it directly
        file_path = target_file
        # Extract directory and filename
        directory = os.path.dirname(target_file)
        filename = os.path.basename(target_file)
    else:
        # Check in conversations directory
        conversations_dir = "conversations"
        file_path = os.path.join(conversations_dir, target_file)
        directory = conversations_dir
        filename = target_file
        
        # If not found in conversations, check in base_conversations
        if not os.path.exists(file_path):
            base_conversations_dir = "base_conversations"
            file_path = os.path.join(base_conversations_dir, target_file)
            directory = base_conversations_dir
            filename = target_file
            
            # If still not found, raise an error
            if not os.path.exists(file_path):
                raise click.BadParameter(f"File {target_file} not found in either {conversations_dir} or {base_conversations_dir}")
    
    # Load the dataset for settings
    dataset = load_dataset("shisa-ai/shisa-rp-bench-testset")["train"]
    
    # Load conversations from the target file
    conversations = load_jsonl(file_path)
    
    # If num_conversations is specified, limit the number of conversations
    if num_conversations is not None:
        conversations = conversations[:num_conversations]
        dataset = dataset.select(range(min(len(conversations), len(dataset))))
    
    # Get model name from file name
    model_name = os.path.splitext(filename)[0].replace("_shisa-ai-shisa-rp-bench-testset", "")
    
    # Format all conversations
    formatted_conversations = []
    
    for idx, conv in enumerate(conversations):
        if idx >= len(dataset):
            break
            
        settings = dataset[idx]
        
        # Format the conversation
        formatted_data = format_single_conversation(conv, settings)
        
        # Create conversation data
        conv_data = {
            "id": hashlib.md5(f"{filename}_{idx}".encode()).hexdigest(),
            "llm": model_name,
            "settings": settings,
            "formatted_data": formatted_data
        }
        
        formatted_conversations.append(conv_data)
    
    if not formatted_conversations:
        raise click.BadParameter(f"No conversations found in {file_path}")
        
    return formatted_conversations


class ConversationJudgeAbsolute(curator.LLM):
    """Evaluates a single LLM conversation in absolute terms."""

    def prompt(self, input: dict) -> str:
        """Generate a prompt for absolute evaluation using the template from eval_prompt_SFW.txt."""
        with open("prompts/eval_prompt_SFW.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()
        return prompt_template.replace("{{formatted_data}}", input["formatted_data"])

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input data into the desired output format."""
        return {
            "id": input["id"],
            "llm": input["llm"],
            "formatted_data": input["formatted_data"],
            "evaluation": response
        }


def parse_evaluation_json(evaluation_text):
    """Parse the JSON evaluation response and return a structured dictionary."""
    try:
        # Try to parse the entire response as JSON
        evaluation_data = json.loads(evaluation_text)
        return evaluation_data
    except json.JSONDecodeError:
        # If that fails, try to extract JSON from the text
        try:
            # Look for JSON-like content between curly braces
            start_idx = evaluation_text.find('{')
            end_idx = evaluation_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = evaluation_text[start_idx:end_idx]
                # Clean the JSON string
                import re
                clean_json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
                return json.loads(clean_json_str)
        except (json.JSONDecodeError, ValueError):
            pass
    
    # If all parsing attempts fail, return empty dict
    return {}


@click.command()
@click.option('--target-model', required=True, help='Target model name or file path to generate formatted conversations for.')
@click.option('--judge-model-name', '-j', required=True, help='Model name to use for judging the conversations')
@click.option('--num-conversations', '-n', default=None, type=int, help='Number of conversations to evaluate (default: all)')
@click.option('--temp-dir', help='Temporary directory for job-specific files')
def main(target_model, judge_model_name, num_conversations, temp_dir):
    """Format and evaluate conversations using an LLM as judge.

    Takes a target model file, formats the conversations, and evaluates them.
    """
    # Check if the input is a file path or a model name
    if target_model.endswith('.jsonl'):
        # It's already a file path
        target_file = target_model
    else:
        # It's a model name, transform it into the target file path
        target_file = target_model.replace('/', '-') + '_shisa-ai-shisa-rp-bench-testset.jsonl'
    
    print(f"Processing file: {target_file}")
    
    # Generate the formatted conversations
    formatted_conversations = generate_formatted_conversations(target_file, num_conversations)
    
    print(f"Formatted {len(formatted_conversations)} conversations for model: {formatted_conversations[0]['llm']}")
    
    # Configure backend for the judge
    backend = "litellm"
    backend_params = {
        "max_requests_per_minute": 5000,
        "max_tokens_per_minute": 1000000,
        "max_concurrent_requests": 128,
    }  

    # Initialize the judge
    judge = ConversationJudgeAbsolute(
        model_name=judge_model_name,
        backend=backend,  
        backend_params=backend_params,
    )

    # Create a dataset with all formatted conversations
    conversations_dataset = Dataset.from_list(formatted_conversations)
    
    # Process all conversations at once using curator
    print(f"Evaluating {len(formatted_conversations)} conversations...")
    results = judge(conversations_dataset)
    
    # Define the categories we're tracking
    categories = [
        "Roleplay Adherence",
        "Consistency",
        "Contextual Understanding",
        "Expressiveness",
        "Creativity",
        "Naturalness of Japanese",
        "Enjoyment of the Dialogue",
        "Appropriateness of Turn-Taking"
    ]
    
    # Initialize arrays to store scores for each category
    category_scores = {category: [] for category in categories}
    all_scores = []
    
    # Process each result
    for i, result in enumerate(results):
        print(f"\nProcessing result {i+1}/{len(results)}:")
        
        # Parse the evaluation
        eval_data = parse_evaluation_json(result["evaluation"])
        
        if not eval_data:
            print(f"  Could not parse evaluation for conversation {i+1}")
            continue
        
        # Collect scores for each category
        for category in categories:
            if category in eval_data:
                score = eval_data[category]
                category_scores[category].append(score)
                all_scores.append(score)
                print(f"  {category}: {score}")
    
    # Calculate and print average scores
    print("\n" + "="*80)
    print("OVERALL EVALUATION RESULTS")
    print("="*80)
    
    print("\nCategory Averages:")
    print("-" * 40)
    print(f"{'Category':<30} | {'Score':<5} | {'Count':<5}")
    print("-" * 40)
    
    for category in categories:
        scores = category_scores[category]
        if scores:
            avg = sum(scores) / len(scores)
            print(f"{category:<30} | {avg:.2f} | {len(scores)}")
    
    # Calculate overall average
    if all_scores:
        overall_avg = sum(all_scores) / len(all_scores)
        print("-" * 40)
        print(f"{'Overall Average':<30} | {overall_avg:.2f} | {len(all_scores)}")
    
    print("="*80)
    
    # Save results to a file if requested
    if temp_dir:
        output_dir = os.path.join(temp_dir, "analysis")
        os.makedirs(output_dir, exist_ok=True)
        scores_dir = os.path.join(temp_dir, "scores")
        os.makedirs(scores_dir, exist_ok=True)
    else:
        os.makedirs("analysis", exist_ok=True)
        output_dir = "analysis"
        os.makedirs("scores", exist_ok=True)
        scores_dir = "scores"
    
    # Create a filename for the results
    model_name = formatted_conversations[0]['llm']
    safe_model_name = model_name.replace("/", "__")
    safe_judge_name = judge_model_name.replace("/", "__")
    output_file = os.path.join(output_dir, f"{safe_model_name}.{safe_judge_name}.jsonl")
    scores_file = os.path.join(scores_dir, f"{safe_model_name}_rp_bench_scores.json")
    
    # Copy the original conversation file to scores directory with new naming convention
    original_file = target_file
    if not os.path.isabs(original_file):
        # Check in conversations directory first
        if os.path.exists(os.path.join("conversations", original_file)):
            original_file = os.path.join("conversations", original_file)
        # Then check in base_conversations if not found
        elif os.path.exists(os.path.join("base_conversations", original_file)):
            original_file = os.path.join("base_conversations", original_file)
    
    # Define the destination file path
    dest_file = os.path.join(scores_dir, f"{safe_model_name}_rp_bench_scores.jsonl")
    
    # Copy the file
    if os.path.exists(original_file):
        shutil.copy2(original_file, dest_file)
        print(f"Copied original conversations to {dest_file}")
    else:
        print(f"Warning: Could not find original file {original_file} to copy")
    
    # Save the results
    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Save category averages and overall average to scores file
    scores_data = {
        "model_name": model_name,
        "judge_model_name": judge_model_name,
        "category_averages": {},
        "overall_average": None,
        "sample_count": len(formatted_conversations),
        "evaluated_count": sum(1 for cat in category_scores.values() if cat)
    }
    
    # Add category averages
    for category in categories:
        scores = category_scores[category]
        if scores:
            scores_data["category_averages"][category] = sum(scores) / len(scores)
    
    # Add overall average
    if all_scores:
        scores_data["overall_average"] = sum(all_scores) / len(all_scores)
    
    # Write scores to file
    with open(scores_file, "w", encoding="utf-8") as f:
        json.dump(scores_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Scores summary saved to {scores_file}")


if __name__ == "__main__":
    main() 
