import os
import json
from itertools import combinations
import hashlib
from io import StringIO
from datasets import load_dataset
import click

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_conversation(conv_data):
    """Format a single conversation into markdown."""
    settings = conv_data.get("settings", {})
    conversation = conv_data.get("conversation", [])
    
    md_lines = []
    
    # Add settings
    if "世界観設定" in settings:
        md_lines.append(f"世界観設定:\n{settings['世界観設定']}\n")
    if "キャラクター設定" in settings:
        md_lines.append(f"キャラクター設定:\n{settings['キャラクター設定']}\n")
    
    # Add conversation
    md_lines.append("会話:")
    for turn in conversation:
        speaker = turn.get("speaker", "")
        text = turn.get("text", "")
        md_lines.append(f"{speaker}: {text}")
    
    return "\n".join(md_lines)

def format_conversation_pair(conv_a, conv_b, dataset_row):
    """Format a pair of conversations into a single markdown document.
    Creates a structured format with clear separation between conversations
    and properly formatted messages."""
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
    
    # Write separator for conversations
    output.write("---\n\n")
    
    # Write first conversation
    output.write("# 会話A\n\n")
    for i, message in enumerate(conv_a.get("conversation_history", [])):
        header = "### User" if i % 2 == 0 else "### Assistant"
        output.write(f"{header}\n{message}\n\n")
    
    # Add separator between conversations
    output.write("\n---\n\n")
    
    # Write second conversation
    output.write("# 会話B\n\n")
    for i, message in enumerate(conv_b.get("conversation_history", [])):
        header = "### User" if i % 2 == 0 else "### Assistant"
        output.write(f"{header}\n{message}\n\n")
    
    # Add final separator
    output.write("\n---\n")
    
    return output.getvalue()

def write_pair_settings(settings, file_a, file_b):
    """Format the settings for the conversation pair."""
    return {
        "id": hashlib.md5(f"{file_a}_{file_b}".encode()).hexdigest(),
        "llm_a": os.path.splitext(file_a)[0].replace("_Aratako-Japanese-RP-Bench-testdata-SFW", ""),
        "llm_b": os.path.splitext(file_b)[0].replace("_Aratako-Japanese-RP-Bench-testdata-SFW", ""),
        "settings": settings
    }

def generate_conversation_pairs(target_file=None):
    conversations_dir = "conversations"
    output_file = "latest_conversation_pairs.jsonl" if target_file else "all_conversation_pairs.jsonl"
    
    # Load the dataset for settings
    dataset = load_dataset("Aratako/Japanese-RP-Bench-testdata-SFW")
    
    # Get all JSONL files
    jsonl_files = [f for f in os.listdir(conversations_dir) if f.endswith('.jsonl')]
    
    if target_file:
        if target_file not in jsonl_files:
            raise click.BadParameter(f"File {target_file} not found in {conversations_dir}")
        # Generate pairs only for the target file
        other_files = [f for f in jsonl_files if f != target_file]
        pairs = [(target_file, other_file) for other_file in other_files]
    else:
        # Generate all possible pairs using combinations
        pairs = list(combinations(jsonl_files, 2))
    
    print(f"Found {len(jsonl_files)} files, generating {len(pairs)} pairs...")
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_a, file_b in pairs:
            path_a = os.path.join(conversations_dir, file_a)
            path_b = os.path.join(conversations_dir, file_b)
            
            data_a = load_jsonl(path_a)
            data_b = load_jsonl(path_b)
            
            # For each conversation in the files
            for i, (conv_a, conv_b) in enumerate(zip(data_a, data_b)):
                # Get corresponding dataset row
                dataset_row = dataset['train'][i]
                
                # Create settings with unique ID and model names
                settings = write_pair_settings(conv_a.get("settings", {}), file_a, file_b)
                
                # Format both conversations into a single markdown document
                formatted_data = format_conversation_pair(conv_a, conv_b, dataset_row)
                
                # Combine into final format
                comparison_data = {
                    "id": settings["id"],
                    "llm_a": settings["llm_a"],
                    "llm_b": settings["llm_b"],
                    "settings": settings["settings"],
                    "formatted_data": formatted_data
                }
                
                # Write to output file
                out_f.write(json.dumps(comparison_data, ensure_ascii=False) + '\n')
                
    print(f"Generated pairs have been written to {output_file}")

@click.command()
@click.option('--target-model', '-m', help='Generate pairs only for this model against all others')
def main(target_model):
    """Generate conversation pairs for evaluation."""
    print("Generating conversation pairs...")
    if target_model:
        # Transform the model name into the target file path
        target_file = target_model.replace('/', '-') + '_Aratako-Japanese-RP-Bench-testdata-SFW.jsonl'
    else:
        target_file = None
    generate_conversation_pairs(target_file)

if __name__ == "__main__":
    main()
