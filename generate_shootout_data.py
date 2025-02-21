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

def generate_conversation_pairs(target_file=None, generate_base=False):
    base_conversations_dir = "base_conversations"
    conversations_dir = "conversations"
    output_file = "base_conversation_pairs.jsonl" if generate_base else "latest_conversation_pairs.jsonl"
    
    # Number of rows to use from each conversation file and dataset
    # Maximum is 30 as that's the total number of conversations per file
    rows_to_use = 15
    
    # Add warning and confirmation for base_conversation_pairs.jsonl
    if output_file == "base_conversation_pairs.jsonl":
        print("\nWARNING: You are about to overwrite base_conversation_pairs.jsonl. These hold the pairs for all the models you'll be comparing against, and this could cause the program to stop working.")
        confirmation = input("Are you sure you want to continue? (yes/no): ")
        if confirmation.lower() != "yes":
            print("Operation cancelled.")
            return
    
    # Load the dataset for settings
    dataset = load_dataset("Aratako/Japanese-RP-Bench-testdata-SFW")["train"]
    #Take the specified number of rows
    dataset = dataset.select(range(rows_to_use))

    if target_file:
        # Check if target file exists in conversations directory
        if not os.path.exists(os.path.join(conversations_dir, target_file)):
            raise click.BadParameter(f"File {target_file} not found in {conversations_dir}")
        # Get all files from base_conversations to compare against
        base_files = [f for f in os.listdir(base_conversations_dir) if f.endswith('.jsonl')]
        pairs = [(target_file, base_file) for base_file in base_files]
        print(f"Comparing {target_file} against {len(base_files)} files from {base_conversations_dir}")
    else:
        # Determine which directory to use and get JSONL files
        working_dir = base_conversations_dir if generate_base else conversations_dir
        jsonl_files = [f for f in os.listdir(working_dir) if f.endswith('.jsonl')]
        pairs = list(combinations(jsonl_files, 2))
    
    # Process pairs and write to output
    total_pairs = 0
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_a, file_b in pairs:
            if target_file:
                # Load target file from conversations and comparison file from base_conversations
                convs_a = load_jsonl(os.path.join(conversations_dir, file_a))[:rows_to_use]
                convs_b = load_jsonl(os.path.join(base_conversations_dir, file_b))[:rows_to_use]
            else:
                # Load both files from the same working directory
                convs_a = load_jsonl(os.path.join(working_dir, file_a))[:rows_to_use]
                convs_b = load_jsonl(os.path.join(working_dir, file_b))[:rows_to_use]
            
            # Generate pairs for each conversation in the files
            for idx, (conv_a, conv_b) in enumerate(zip(convs_a, convs_b)):
                # Get settings from dataset - use idx since we're going through conversations in order
                settings = dataset[idx]
                
                # Format and write the pair
                pair_data = write_pair_settings(settings, file_a, file_b)
                # Add index to id to make it unique for each conversation pair
                pair_data['id'] = hashlib.md5(f"{file_a}_{file_b}_{idx}".encode()).hexdigest()
                pair_data['formatted_data'] = format_conversation_pair(conv_a, conv_b, settings)
                out_f.write(json.dumps(pair_data, ensure_ascii=False) + '\n')
                total_pairs += 1
    
    print(f"Generated {total_pairs} total pairs written to {output_file}")

@click.command()
@click.option('--target-model', help='Target model to generate pairs for. If not specified, pairs will be generated between all models.')
@click.option('--generate-base', is_flag=True, help='Generate base conversation pairs. This will overwrite base_conversation_pairs.jsonl')
def main(target_model, generate_base):
    """Generate conversation pairs for evaluation."""
    if generate_base:
        generate_conversation_pairs(generate_base=True)
    else:
        if not target_model:
            raise click.UsageError("Either --target-model or --generate-base must be specified")
        # Transform the model name into the target file path
        target_file = target_model.replace('/', '-') + '_Aratako-Japanese-RP-Bench-testdata-SFW.jsonl'
        generate_conversation_pairs(target_file)

if __name__ == "__main__":
    main()
