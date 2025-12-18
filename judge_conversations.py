"""Japanese RP Bench absolute evaluator using the native Google GenAI SDK.

This script:
- Uses the `google-genai` client for evaluation
- Evaluates formatted RP conversations with Gemini
- Runs in parallel with a thread pool and simple retry/backoff
"""

import os
import json
import hashlib
import random
import time
import threading
from io import StringIO
from datasets import load_dataset, Dataset
import click
import concurrent.futures
from tqdm import tqdm
import google.genai as genai
from google.genai import types as genai_types


EVALUATION_SCHEMA = genai_types.Schema(
    type="object",
    additional_properties=False,
    required=[
        "Evaluation Reason",
        "Roleplay Adherence",
        "Consistency",
        "Contextual Understanding",
        "Expressiveness",
        "Creativity",
        "Naturalness of Japanese",
        "Enjoyment of the Dialogue",
        "Appropriateness of Turn-Taking",
    ],
    properties={
        "Evaluation Reason": genai_types.Schema(type="string"),
        "Roleplay Adherence": genai_types.Schema(type="integer", minimum=1, maximum=5),
        "Consistency": genai_types.Schema(type="integer", minimum=1, maximum=5),
        "Contextual Understanding": genai_types.Schema(type="integer", minimum=1, maximum=5),
        "Expressiveness": genai_types.Schema(type="integer", minimum=1, maximum=5),
        "Creativity": genai_types.Schema(type="integer", minimum=1, maximum=5),
        "Naturalness of Japanese": genai_types.Schema(type="integer", minimum=1, maximum=5),
        "Enjoyment of the Dialogue": genai_types.Schema(type="integer", minimum=1, maximum=5),
        "Appropriateness of Turn-Taking": genai_types.Schema(type="integer", minimum=1, maximum=5),
    },
)


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
            # Use the original dataset/conversation ID when available so that
            # downstream artifacts can join against `*_rp_bench_answers.jsonl`.
            "id": str(conv.get("id") or settings.get("id") or idx),
            "llm": model_name,
            "settings": settings,
            "formatted_data": formatted_data
        }

        formatted_conversations.append(conv_data)

    if not formatted_conversations:
        raise click.BadParameter(f"No conversations found in {file_path}")

    return formatted_conversations


class ConversationJudgeAbsolute:
    """Evaluates LLM conversations using native Google Generative AI SDK."""

    def __init__(self, model_name: str, api_key: str, concurrency_limit: int = 40):
        """Initialize the judge.

        Args:
            model_name: Gemini model name (e.g., "gemini-2.0-flash")
            api_key: Google API key
            concurrency_limit: Maximum concurrent API requests
        """
        self.model_name = model_name
        self.concurrency_limit = concurrency_limit
        self.semaphore = threading.Semaphore(concurrency_limit)

        # Initialize Google GenAI client
        self.client = genai.Client(api_key=api_key)

        # Generation config (deterministic, JSON output)
        safety_settings = [
            genai_types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
            ),
            genai_types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
            ),
            genai_types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
            ),
            genai_types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
            ),
        ]

        self.generation_config = genai_types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=8192,
            response_mime_type="application/json",
            response_schema=EVALUATION_SCHEMA,
            safety_settings=safety_settings,
            thinking_config=genai_types.ThinkingConfig(include_thoughts=False),
        )

        # Load prompt template
        with open("prompts/eval_prompt_SFW.txt", "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        # Track stats
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_thought_tokens = 0
        self.failed_items = []
        self.lock = threading.Lock()

    def prompt(self, input: dict) -> str:
        """Generate a prompt for absolute evaluation."""
        return self.prompt_template.replace("{{formatted_data}}", input["formatted_data"])

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input data."""
        return {
            "id": input["id"],
            "llm": input["llm"],
            "formatted_data": input["formatted_data"],
            "evaluation": response
        }

    def evaluate_item(self, item: dict) -> dict:
        """Evaluates a single item with retry logic and concurrency control."""
        # Add jitter to spread out requests
        time.sleep(random.uniform(0.1, 0.5))

        with self.semaphore:
            prompt_text = self.prompt(item)

            max_retries = 5
            base_delay = 1

            for attempt in range(max_retries + 1):
                try:
                    # Generate response
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt_text,
                        config=self.generation_config,
                    )

                    # Check if response was blocked
                    if not response.text:
                        if hasattr(response, 'prompt_feedback'):
                            raise ValueError(f"Response blocked: {response.prompt_feedback}")
                        raise ValueError("Empty response from API")

                    # Prefer schema-parsed output when available; fall back to raw text.
                    parsed_payload = getattr(response, "parsed", None)
                    if parsed_payload is not None:
                        if hasattr(parsed_payload, "model_dump"):
                            parsed_payload = parsed_payload.model_dump()
                        elif hasattr(parsed_payload, "dict"):
                            parsed_payload = parsed_payload.dict()
                    if isinstance(parsed_payload, dict):
                        response_text = json.dumps(parsed_payload, ensure_ascii=False)
                    else:
                        response_text = response.text

                    # Track token usage (if available)
                    usage = getattr(response, "usage_metadata", None)
                    if usage is not None:
                        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                        candidates_tokens = getattr(usage, "candidates_token_count", 0) or 0
                        thoughts_tokens = getattr(usage, "thoughts_token_count", 0) or 0
                        with self.lock:
                            self.total_input_tokens += prompt_tokens
                            self.total_output_tokens += candidates_tokens
                            self.total_thought_tokens += thoughts_tokens

                    parsed_result = self.parse(item, response_text)
                    return parsed_result

                except Exception as e:
                    error_msg = f"API error: {type(e).__name__}: {str(e)}"
                    if attempt == max_retries:
                        # Track failed item
                        failed_item = {
                            "id": item.get("id", "unknown"),
                            "llm": item.get("llm", "unknown"),
                            "error": error_msg,
                            "attempts": max_retries + 1
                        }
                        with self.lock:
                            self.failed_items.append(failed_item)
                        print(f"Failed to process item {item.get('llm', 'unknown')} after {max_retries + 1} attempts: {error_msg}")
                        return None  # Return None for failed items
                    delay = base_delay * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed for {item.get('llm', 'unknown')}: {error_msg}. Retrying in {delay}s...")
                    time.sleep(delay)

    def __call__(self, dataset: list, max_workers: int = None) -> list:
        """Process the dataset in parallel and return a list of evaluation results."""
        if max_workers is None:
            max_workers = self.concurrency_limit

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(executor.map(self.evaluate_item, dataset), total=len(dataset), desc="Evaluating")
            )

        # Print stats
        total = self.total_input_tokens + self.total_output_tokens + self.total_thought_tokens
        print(
            f"\nToken usage: {self.total_input_tokens:,} input + "
            f"{self.total_output_tokens:,} output + "
            f"{self.total_thought_tokens:,} thoughts = {total:,} total"
        )
        if self.failed_items:
            print(f"Failed items: {len(self.failed_items)}")

        return results


def parse_evaluation_json(evaluation_text):
    """Parse the JSON evaluation response and return a structured dictionary."""
    if isinstance(evaluation_text, dict):
        return evaluation_text
    try:
        # Try to parse the entire response as JSON
        evaluation_data = json.loads(evaluation_text)
        return evaluation_data
    except json.JSONDecodeError:
        # If that fails, try to extract JSON from markdown code blocks
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', evaluation_text, re.DOTALL)
        if json_match:
            try:
                evaluation_data = json.loads(json_match.group(1))
                return evaluation_data
            except json.JSONDecodeError:
                pass

        # If we still can't parse it, try to find any JSON object
        json_match = re.search(r'\{.*\}', evaluation_text, re.DOTALL)
        if json_match:
            try:
                evaluation_data = json.loads(json_match.group(0))
                return evaluation_data
            except json.JSONDecodeError:
                pass

    return None


@click.command()
@click.option(
    '--judge-model-name',
    '-j',
    required=True,
    help='Gemini model name to use for judging (e.g., gemini/gemini-2.0-flash)',
)
@click.option('--target-model', '-t', required=True, help='Target model name or file path')
@click.option('-n', '--num-conversations', type=int, help='Number of conversations to evaluate (default: all)')
@click.option('--temp-dir', help='Temporary directory for job-specific files')
@click.option('--max-workers', default=40, help='Number of worker threads')
@click.option('--concurrency-limit', default=40, help='Max concurrent API requests')
@click.option('--api-key-env', default='GEMINI_API_KEY', help='Environment variable name for API key')
def main(judge_model_name, target_model, num_conversations, temp_dir, max_workers, concurrency_limit, api_key_env):
    """Format and evaluate conversations using Gemini as judge.

    Takes a target model file, formats the conversations, and evaluates them.
    """
    # Get API key
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise click.BadParameter(f"API key not found in environment variable: {api_key_env}")

    # Strip gemini/ prefix if present
    if judge_model_name.startswith("gemini/"):
        judge_model_name = judge_model_name[7:]

    # Check if the input is a file path or a model name
    if target_model.endswith('.jsonl'):
        target_file = target_model
    else:
        # It's a model name, transform it into the target file path
        target_file = target_model.replace('/', '-') + '_shisa-ai-shisa-rp-bench-testset.jsonl'

    print(f"Processing file: {target_file}")

    # Generate the formatted conversations
    formatted_conversations = generate_formatted_conversations(target_file, num_conversations)

    print(f"Formatted {len(formatted_conversations)} conversations for model: {formatted_conversations[0]['llm']}")

    # Initialize the judge
    judge = ConversationJudgeAbsolute(
        model_name=judge_model_name,
        api_key=api_key,
        concurrency_limit=concurrency_limit,
    )

    # Process all conversations
    print(f"Evaluating {len(formatted_conversations)} conversations...")
    results = judge(formatted_conversations, max_workers=max_workers)

    # Filter out None results from failed items
    results = [r for r in results if r is not None]

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
    parsed_count = 0
    judgements = []
    for i, result in enumerate(results):
        print(f"\nProcessing result {i+1}/{len(results)}:")

        # Parse the evaluation
        raw_evaluation = result.get("evaluation", "")
        eval_data = parse_evaluation_json(raw_evaluation)

        if not eval_data:
            print(f"  Could not parse evaluation for conversation {i+1}")
            judgements.append(
                {
                    "id": result.get("id", ""),
                    "llm": result.get("llm", ""),
                    "evaluation": None,
                    "raw_evaluation": raw_evaluation,
                }
            )
            continue
        parsed_count += 1
        judgements.append(
            {
                "id": result.get("id", ""),
                "llm": result.get("llm", ""),
                "evaluation": eval_data,
                "raw_evaluation": raw_evaluation,
            }
        )

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

    # Save results to a file
    if temp_dir:
        scores_dir = os.path.join(temp_dir, "scores")
        os.makedirs(scores_dir, exist_ok=True)
    else:
        os.makedirs("scores", exist_ok=True)
        scores_dir = "scores"

    # Create filenames for the results
    model_name = formatted_conversations[0]['llm']
    safe_model_name = model_name.replace("/", "__")
    safe_judge_name = judge_model_name.replace("/", "__")
    scores_file = os.path.join(scores_dir, f"{safe_model_name}_rp_bench_scores.json")

    # Define the destination file path for the original conversations
    dest_file = os.path.join(scores_dir, f"{safe_model_name}_rp_bench_answers.jsonl")

    # Save the original conversations from the target file directly.
    # We mirror the same resolution logic as generate_formatted_conversations()
    # so that both manual and programmatic invocations can locate the data.
    if target_model.endswith('.jsonl') and os.path.exists(target_model):
        source_file = target_model
    else:
        # Try the plain filename first
        if os.path.exists(target_file):
            source_file = target_file
        else:
            # Then look under conversations/ and base_conversations/
            conversations_dir = "conversations"
            base_conversations_dir = "base_conversations"

            candidate = os.path.join(conversations_dir, target_file)
            if os.path.exists(candidate):
                source_file = candidate
            else:
                candidate = os.path.join(base_conversations_dir, target_file)
                if os.path.exists(candidate):
                    source_file = candidate
                else:
                    raise click.BadParameter(
                        f"File {target_file} not found in either {conversations_dir} or {base_conversations_dir}"
                    )

    conversations = load_jsonl(source_file)
    with open(dest_file, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"Original conversations saved to {dest_file}")

    # Save category averages and overall average to scores file
    scores_data = {
        "model_name": model_name,
        "judge_model_name": judge_model_name,
        "category_averages": {},
        "overall_average": None,
        "sample_count": len(formatted_conversations),
        "evaluated_count": parsed_count,
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

    print(f"\nResults saved to {scores_file}")

    # Save per-conversation judgements (including judge reasoning) for inspection/reprocessing.
    judgements_file = os.path.join(scores_dir, f"{safe_model_name}_rp_bench_judgements.jsonl")
    with open(judgements_file, "w", encoding="utf-8") as f:
        for record in judgements:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Per-conversation judgements saved to {judgements_file}")


if __name__ == "__main__":
    main()
