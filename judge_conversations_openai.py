#!/usr/bin/env python3
"""
OpenAI-compatible judge for Japanese RP Bench conversations.
This script runs in the shisa-jp-rp-bench conda environment.
"""
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from openai import OpenAI


def parse_eval_json(text: str) -> Dict[str, Any]:
    """Parse evaluation JSON from LLM response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start_idx = text.find("{")
        end_idx = text.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            clean_json = re.sub(r"[\x00-\x1F\x7F]", "", json_str)
            try:
                return json.loads(clean_json)
            except json.JSONDecodeError:
                return {}
        return {}


def format_conversation(conv_data: Dict[str, Any], dataset_row: Dict[str, Any]) -> str:
    """Format conversation data for judge evaluation."""
    columns = {
        "id": "データのid",
        "genre": "ロールプレイのジャンル",
        "world_setting": "ロールプレイの世界観設定",
        "scene_setting": "ロールプレイのシーン設定",
        "user_setting": "ロールプレイのユーザー側キャラクター設定",
        "assistant_setting": "ロールプレイのアシスタント側キャラクター設定",
        "dialogue_tone": "ロールプレイの対話のトーン",
        "first_user_input": "ロールプレイの最初のユーザー発話",
        "response_format": "ロールプレイの応答形式",
    }

    output: List[str] = ["# 設定\n\n"]
    for col, jp_title in columns.items():
        output.append(f"## {jp_title}\n")
        output.append(f"{dataset_row.get(col, '')}\n\n")

    output.append("---\n\n# 会話\n\n")
    for i, message in enumerate(conv_data.get("conversation_history", [])):
        header = "### User" if i % 2 == 0 else "### Assistant"
        output.append(f"{header}\n{message}\n\n")
    output.append("\n---\n")
    return "".join(output)


def main():
    parser = argparse.ArgumentParser(description="Judge RP conversations with OpenAI-compatible endpoint")
    parser.add_argument("--conversation-file", required=True, help="Path to conversation JSONL file")
    parser.add_argument("--prompt-file", required=True, help="Path to judge prompt template")
    parser.add_argument("--output-dir", required=True, help="Directory to save judgements and scores")
    parser.add_argument("--model", required=True, help="Judge model name")
    parser.add_argument("--base-url", required=True, help="Judge API base URL")
    parser.add_argument("--api-key", default="EMPTY", help="Judge API key")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to judge")
    parser.add_argument("--max-workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--cache-dir", help="HuggingFace cache directory")

    args = parser.parse_args()

    # Load prompt template
    prompt_template = Path(args.prompt_file).read_text()

    # Load dataset
    cache_dir = args.cache_dir if args.cache_dir else None
    dataset = load_dataset(
        "shisa-ai/shisa-rp-bench-testset",
        split="train",
        cache_dir=cache_dir,
    )

    # Load conversations
    conversations: List[Dict[str, Any]] = []
    with open(args.conversation_file) as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    if not conversations:
        print(f"Error: No conversations found in {args.conversation_file}")
        return 1

    # Limit samples if requested
    if args.max_samples is not None:
        conversations = conversations[:args.max_samples]

    sample_count = min(len(conversations), len(dataset))
    conversations = conversations[:sample_count]

    print(f"Judging {len(conversations)} conversations with {args.model}")
    print(f"Using {args.max_workers} workers")

    # Evaluation categories
    categories = [
        "Roleplay Adherence",
        "Consistency",
        "Contextual Understanding",
        "Expressiveness",
        "Creativity",
        "Naturalness of Japanese",
        "Enjoyment of the Dialogue",
        "Appropriateness of Turn-Taking",
    ]

    # Setup output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract model name for output filenames
    model_safe = args.conversation_file.split("/")[-1].split("_")[0]
    judgements_path = output_dir / f"{model_safe}_rp_bench_judgements.jsonl"
    scores_path = output_dir / f"{model_safe}_rp_bench_scores.json"

    # Judge conversations
    results: List[Dict[str, Any]] = []

    def judge_one(idx: int, conv_data: Dict[str, Any]) -> Dict[str, Any]:
        dataset_row = dataset[idx]
        formatted_data = format_conversation(conv_data, dataset_row)
        prompt = prompt_template.replace("{{formatted_data}}", formatted_data)

        client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
        )
        content = response.choices[0].message.content or ""
        evaluation = parse_eval_json(content)
        return {
            "id": conv_data.get("id", idx),
            "evaluation": evaluation,
            "evaluation_text": content,
        }

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(judge_one, idx, conv_data)
            for idx, conv_data in enumerate(conversations)
        ]
        for future in as_completed(futures):
            try:
                results.append(future.result())
                print(f"Completed {len(results)}/{len(conversations)}", end="\r")
            except Exception as exc:
                print(f"\nWarning: Judge request failed: {exc}")

    print(f"\nJudged {len(results)} conversations")

    if not results:
        print("Error: No judgements produced")
        return 1

    # Calculate scores
    category_scores: Dict[str, List[float]] = {cat: [] for cat in categories}
    all_scores: List[float] = []
    evaluated_count = 0

    with judgements_path.open("w") as f:
        for item in results:
            eval_data = item.get("evaluation") or {}
            if isinstance(eval_data, dict):
                has_score = False
                for category in categories:
                    raw = eval_data.get(category)
                    if isinstance(raw, (int, float)):
                        val = float(raw)
                        category_scores[category].append(val)
                        all_scores.append(val)
                        has_score = True
                if has_score:
                    evaluated_count += 1
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    category_averages = {
        category: (sum(scores) / len(scores) if scores else None)
        for category, scores in category_scores.items()
    }
    overall_average = (sum(all_scores) / len(all_scores)) if all_scores else None

    scores_payload = {
        "overall_average": overall_average,
        "category_averages": category_averages,
        "sample_count": sample_count,
        "evaluated_count": evaluated_count,
    }
    scores_path.write_text(json.dumps(scores_payload, ensure_ascii=False, indent=2))

    print(f"Saved judgements to {judgements_path}")
    print(f"Saved scores to {scores_path}")
    print(f"Overall average: {overall_average:.2f}" if overall_average else "Overall average: N/A")

    return 0


if __name__ == "__main__":
    exit(main())
