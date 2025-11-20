import os
import json
import hashlib
import click
from datasets import Dataset, load_dataset, concatenate_datasets
import re
import random

# Define the prompt as a constant at the top of the file, but with a placeholder for genre
RP_PROMPT_TEMPLATE = """以下の構造で詳細な日本語ロールプレイングシナリオを作成してください：

{{
"genre": "{genre}",
"tag": "全年齢 or R-15",
"world_setting": "[日本語で3〜5文の世界観の説明を提供]",
"scene_setting": "[日本語で具体的なシナリオ/プロット設定を説明]",
"user_setting": "[ユーザーのキャラクターについての詳細情報を日本語で提供。名前、年齢、性別、種族、職業、性格特性、外見、話し方、バックストーリー、目標/動機を含む。ネストされたJSONではなく、テキスト形式で記述してください]",
"assistant_setting": "[AIアシスタントのキャラクターについての詳細情報を日本語で提供。名前、年齢、性別、種族、職業、性格特性、外見、話し方、バックストーリー、目標/動機を含む。ネストされたJSONではなく、テキスト形式で記述してください]",
"dialogue_tone": "[会話の希望するトーンを日本語で説明]",
"first_user_input": "[ロールプレイを開始するためのユーザーキャラクターからの最初のメッセージを提供]"
}}"""

# List of genres to choose from
GENRES = ["ファンタジー", "ホラー", "恋愛", "SF", "ミステリー", "コメディ", "歴史", "学園", "職場", "日常", "アクション", "異世界"]


def merge_and_upload_datasets(generated_data_path):
    """
    Merge our generated RP situations with the existing benchmark dataset
    and upload to HuggingFace.
    
    Args:
        generated_data_path: Path to our generated RP situations JSON file
    """
    print("Loading datasets...")
    
    # Load our generated data
    with open(generated_data_path, 'r', encoding='utf-8') as f:
        generated_data = json.load(f)
    
    # Extract the parsed responses
    processed_data = []
    for item in generated_data:
        if item.get('parsed_response'):
            entry = {
                'genre': item['parsed_response'].get('genre', ''),
                'tag': item['parsed_response'].get('tag', ''),
                'world_setting': item['parsed_response'].get('world_setting', ''),
                'scene_setting': item['parsed_response'].get('scene_setting', ''),
                'user_setting': item['parsed_response'].get('user_setting', ''),
                'assistant_setting': item['parsed_response'].get('assistant_setting', ''),
                'dialogue_tone': item['parsed_response'].get('dialogue_tone', ''),
                'first_user_input': item['parsed_response'].get('first_user_input', ''),
            }
            processed_data.append(entry)
    
    # Create a dataset from our processed data
    our_dataset = Dataset.from_list(processed_data)
    
    # Load the existing benchmark dataset
    existing_dataset = load_dataset("Aratako/Japanese-RP-Bench-testdata-SFW")
    if isinstance(existing_dataset, dict):
        existing_dataset = existing_dataset['train']  # Assuming it has a 'train' split
    
    # Add response_format column with random selection
    response_formats = [
        "キャラ名「発話」（状況や動作の描写）のような形式でキャラクターのセリフと動作や状況を描写",
        "動作や状況の描写は行わず、括弧内でキャラクターのセリフのみを描写",
        "動作や状況の描写は行わず、鍵括弧なしでキャラクターのセリフのみを描写"
    ]
    
    our_dataset = our_dataset.add_column(
        'response_format', 
        [random.choice(response_formats) for _ in range(len(our_dataset))]
    )
    
    # Merge datasets
    # First, ensure column compatibility
    existing_columns = set(existing_dataset.column_names)
    our_columns = set(our_dataset.column_names)
    
    # Add any missing columns to our dataset
    for col in existing_columns - our_columns:
        if col == 'id':
            # Skip adding 'id' column here, we'll handle it after merging
            continue
        our_dataset = our_dataset.add_column(col, [''] * len(our_dataset))
    
    # Add any missing columns to existing dataset
    for col in our_columns - existing_columns:
        if col == 'id':
            # Skip adding 'id' column here, we'll handle it after merging
            continue
        existing_dataset = existing_dataset.add_column(col, [''] * len(existing_dataset))
    
    # Remove 'id' column from both datasets before merging
    if 'id' in our_dataset.column_names:
        our_dataset = our_dataset.remove_columns(['id'])
    if 'id' in existing_dataset.column_names:
        existing_dataset = existing_dataset.remove_columns(['id'])
    
    # Concatenate datasets
    merged_dataset = concatenate_datasets([existing_dataset, our_dataset])
    
    # Now assign unique IDs to the entire merged dataset
    merged_dataset = merged_dataset.add_column('id', list(range(len(merged_dataset))))
    
    # Upload to HuggingFace
    print(f"Uploading merged dataset with {len(merged_dataset)} examples...")
    merged_dataset.push_to_hub("shisa-ai/shisa-rp-bench-testset", private=True)
    
    print("Dataset successfully merged and uploaded!")


@click.command()
@click.option('--model-name', '-m', required=True, help='Model name to use for generating RP situations')
@click.option('--num-situations', '-n', default=20, help='Number of RP situations to generate')
@click.option('--output-file', '-o', default='rp_situations.json', help='Output JSON file path')
@click.option('--merge-and-upload', '-u', is_flag=True, help='Merge with existing dataset and upload to HF')
def main(model_name, num_situations, output_file, merge_and_upload):
    """Generate roleplay situations using a fixed prompt.

    Uses the specified model to generate a set number of roleplay situations
    and saves them to a JSON file.
    """
    print(f"Generating {num_situations} RP situations using model: {model_name}")
    
    # NOTE: The original implementation used an external orchestration
    # framework here. That dependency has been removed from this repo.
    # For now we only support the downstream merge/upload path assuming
    # `output_file` already contains generated situations.

    if os.path.exists(output_file):
        print(f"Found existing situations at {output_file}; skipping generation.")
    else:
        print("Automatic RP situation generation has been removed; please provide a"
              f" pre-generated JSON file at {output_file} if you want to"
              " use merge/upload functionality.")
        results_list = []

    # Save results to a file (no-op if we just reuse an existing file)
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)
    
    print(f"\nGenerated {len(results_list)} RP situations")
    print(f"Results saved to {output_file}")
    
    # Optionally merge with existing dataset and upload
    if merge_and_upload:
        merge_and_upload_datasets(output_file)


if __name__ == "__main__":
    main() 
