import os
import json
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

def generate_formatted_conversations(target_file, temp_dir=None):
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
    
    output_file = "formatted_conversations.jsonl"
    
    # Number of rows to use from each conversation file and dataset
    # Maximum is 30 as that's the total number of conversations per file
    rows_to_use = 20
    
    # Load the dataset for settings
    dataset = load_dataset("Aratako/Japanese-RP-Bench-testdata-SFW")["train"]
    # Take the specified number of rows
    dataset = dataset.select(range(rows_to_use))
    
    # Load conversations from the target file
    conversations = load_jsonl(file_path)[:rows_to_use]
    
    # Get model name from file name
    model_name = os.path.splitext(filename)[0].replace("_Aratako-Japanese-RP-Bench-testdata-SFW", "")
    
    # Process conversations and write to output
    total_conversations = 0
    first_conversation_formatted = None
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for idx, conv in enumerate(conversations):
            # Get settings from dataset - use idx since we're going through conversations in order
            settings = dataset[idx]
            
            # Format the conversation
            formatted_data = format_single_conversation(conv, settings)
            
            # Save the first conversation for printing
            if idx == 0:
                first_conversation_formatted = formatted_data
            
            # Create conversation data
            conv_data = {
                "id": hashlib.md5(f"{filename}_{idx}".encode()).hexdigest(),
                "llm": model_name,
                "settings": settings,
                "formatted_data": formatted_data
            }
            
            out_f.write(json.dumps(conv_data, ensure_ascii=False) + '\n')
            total_conversations += 1
    
    print(f"Generated {total_conversations} formatted conversations written to {output_file}")
    print(f"Using file: {file_path}")
    print(f"Model name extracted: {model_name}")
    
    # Print the first conversation for verification
    if first_conversation_formatted:
        print("\nFirst conversation preview:")
        print("=" * 80)
        print(first_conversation_formatted)
        print("=" * 80)

@click.command()
@click.option('--target-model', required=True, help='Target model name or file path to generate formatted conversations for.')
@click.option('--temp-dir', help='Temporary directory for job-specific files')
def main(target_model, temp_dir):
    """Generate formatted conversations for absolute evaluation."""
    # Check if the input is a file path or a model name
    if target_model.endswith('.jsonl'):
        # It's already a file path
        target_file = target_model
    else:
        # It's a model name, transform it into the target file path
        target_file = target_model.replace('/', '-') + '_Aratako-Japanese-RP-Bench-testdata-SFW.jsonl'
    
    generate_formatted_conversations(target_file, temp_dir=temp_dir)

if __name__ == "__main__":
    main() 