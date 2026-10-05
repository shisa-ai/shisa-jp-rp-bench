"""Shared HTTPX judging, scenario matching, and durable result artifacts."""
from concurrent.futures import ThreadPoolExecutor
import json
from importlib.resources import files
from pathlib import Path

import click

from .client import ChatError, _validate_options
from .artifacts import score_paths, write_json, write_jsonl
from .data import index_by_id, normalize_id, load_dataset_wrapper
from .models import create_client
from .utils import EVALUATION_CATEGORIES, parse_evaluation


def default_rubric():
    return files("japanese_rp_bench").joinpath("evaluation_prompt.txt").read_text(encoding="utf-8")



def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_conversation(conversation, settings):
    columns = {
        "id": "データのid", "genre": "ロールプレイのジャンル", "world_setting": "ロールプレイの世界観設定",
        "scene_setting": "ロールプレイのシーン設定", "user_setting": "ロールプレイのユーザー側キャラクター設定",
        "assistant_setting": "ロールプレイのアシスタント側キャラクター設定", "dialogue_tone": "ロールプレイの対話のトーン",
        "first_user_input": "ロールプレイの最初のユーザー発話", "response_format": "ロールプレイの応答形式",
    }
    parts = ["# 設定\n\n"]
    for key, label in columns.items():
        parts.append(f"## {label}\n{settings.get(key, '')}\n\n")
    parts.append("---\n\n# 会話\n\n")
    for index, message in enumerate(conversation.get("conversation_history", [])):
        parts.append(f"### {'User' if index % 2 == 0 else 'Assistant'}\n{message}\n\n")
    return ''.join(parts)


def target_identity(conversations):
    identities = set()
    for conversation in conversations:
        identity = conversation.get("target_model_name")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("Every conversation requires a nonempty target_model_name")
        identities.add(identity)
    if not identities:
        raise ValueError("No conversations found")
    if len(identities) != 1:
        raise ValueError("Mixed target model identities in conversation file")
    return identities.pop()


def prepare_conversations(source, *, dataset_path=None, max_samples=None, cache_dir=None):
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise ValueError("max_samples must be a positive integer")
    conversations = load_jsonl(source)
    index_by_id(conversations, label="conversations")
    target_identity(conversations)
    if max_samples is not None:
        conversations = conversations[:max_samples]
    if not conversations:
        raise ValueError("No conversations found")
    needs_dataset = any(not isinstance(record.get("settings"), dict) for record in conversations)
    settings_by_id = {}
    if needs_dataset:
        if not dataset_path:
            raise ValueError("An explicit dataset is required when embedded settings are absent")
        dataset = load_dataset_wrapper(str(dataset_path), split="train", cache_dir=cache_dir)
        settings_by_id = index_by_id(dataset, label="settings")
    prepared = []
    for conversation in conversations:
        identifier = normalize_id(conversation["id"])
        settings = conversation.get("settings")
        if not isinstance(settings, dict):
            if identifier not in settings_by_id:
                raise ValueError(f"Unknown conversation ID in settings: {identifier}")
            settings = settings_by_id[identifier]
        if normalize_id(settings.get("id")) != identifier:
            raise ValueError(f"Embedded settings ID does not match conversation {identifier}")
        history = conversation.get("conversation_history")
        if not isinstance(history, list) or not history or not all(isinstance(message, str) for message in history):
            raise ValueError(f"Conversation {identifier} must have a list of text messages")
        prepared.append({"id": identifier, "settings": settings, "formatted_data": format_conversation(conversation, settings), "answer": conversation})
    return prepared


def judge_conversations(records, *, client, judge_model, output_dir="scores", max_workers=10, prompt_template=None, request_options=None):
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    if not records:
        raise ValueError("No conversations to judge")
    model_name = target_identity([record["answer"] for record in records])
    options = {**getattr(client, "default_options", {}), **(request_options or {})}
    options.setdefault("temperature", 0)
    if "max_tokens" not in options and "max_completion_tokens" not in options:
        options["max_tokens"] = 8192
    _validate_options(options)
    paths = score_paths(model_name, judge_model, output_dir)
    write_json(paths["scores"], {"model_name": model_name, "judge_model_name": judge_model,
                               "category_averages": dict.fromkeys(EVALUATION_CATEGORIES), "overall_average": None,
                               "sample_count": len(records), "evaluated_count": 0, "failed_count": len(records), "status": "incomplete"})
    # Persist source input before any paid calls; each completed judgment is checkpointed.
    write_jsonl(paths["answers"], [record["answer"] for record in records])
    completed = {}
    write_jsonl(paths["judgements"], [])

    rubric = prompt_template or default_rubric()

    def evaluate(record):
        judgement = {"id": record["id"], "llm": model_name, "judge_model_name": judge_model,
                     "settings": record["settings"], "evaluation": None, "raw_evaluation": None, "error": None}
        if "{{formatted_data}}" in rubric:
            messages = [{"role": "user", "content": rubric.replace("{{formatted_data}}", record["formatted_data"])}]
        else:
            messages = [{"role": "system", "content": rubric}, {"role": "user", "content": record["formatted_data"]}]
        try:
            raw = client.complete(judge_model, messages, **options)
            judgement["raw_evaluation"] = raw
            judgement["evaluation"] = parse_evaluation(raw)
        except Exception as exc:
            judgement["error"] = str(exc) if isinstance(exc, ChatError) else f"{type(exc).__name__}: judging failed; raw response retained when available"
        return judgement

    from concurrent.futures import as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate, record): index for index, record in enumerate(records)}
        for future in as_completed(futures):
            completed[futures[future]] = future.result()
            write_jsonl(paths["judgements"], [completed[index] for index in sorted(completed)])
    judgements = [completed[index] for index in range(len(records))]
    valid = [record["evaluation"] for record in judgements if record["evaluation"] is not None]
    averages = {category: sum(value[category] for value in valid) / len(valid) if valid else None for category in EVALUATION_CATEGORIES}
    summary = {"model_name": model_name, "judge_model_name": judge_model, "category_averages": averages,
               "overall_average": sum(averages.values()) / len(averages) if valid else None,
               "sample_count": len(records), "evaluated_count": len(valid), "failed_count": len(records) - len(valid),
               "status": "complete" if len(valid) == len(records) else "incomplete"}
    write_json(paths["scores"], summary)
    return summary


@click.command()
@click.option("--judge-model", required=True)
@click.option("--conversation-file", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--max-samples", type=click.IntRange(min=1))
@click.option("--dataset", "dataset_path", help="Scenario dataset repository or local JSON/JSONL file")
@click.option("--prompt-file", type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", default="scores", type=click.Path(file_okay=False))
@click.option("--max-workers", default=10, type=click.IntRange(min=1))
@click.option("--base-url")
@click.option("--api-key", help="Prefer --api-key-env to avoid exposing a key in process arguments")
@click.option("--api-key-env")
@click.option("--timeout", default=120.0, type=click.FloatRange(min=0, min_open=True))
@click.option("--max-retries", default=2, type=click.IntRange(min=0))
@click.option("--cache-dir", type=click.Path(file_okay=False))
@click.option("--request-options", default="{}", help="JSON object of endpoint-specific generation options")
def main(judge_model, conversation_file, max_samples, dataset_path, prompt_file, output_dir,
         max_workers, base_url, api_key, api_key_env, timeout, max_retries, cache_dir, request_options):
    """Judge roleplay conversations through an OpenAI-compatible HTTP endpoint."""
    client = None
    try:
        options = json.loads(request_options)
        if not isinstance(options, dict):
            raise ValueError("request-options must be a JSON object")
        if {"model", "messages", "stream"} & options.keys():
            raise ValueError("request-options cannot override model, messages, or stream")
        records = prepare_conversations(conversation_file, dataset_path=dataset_path, max_samples=max_samples, cache_dir=cache_dir)
        rubric = Path(prompt_file).read_text(encoding="utf-8") if prompt_file else None
        client = create_client(base_url=base_url, api_key=api_key, api_key_env=api_key_env, timeout=timeout, max_retries=max_retries)
        summary = judge_conversations(records, client=client, judge_model=judge_model,
                                      output_dir=output_dir, max_workers=max_workers, prompt_template=rubric, request_options=options)
        click.echo(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        if summary["failed_count"]:
            raise click.ClickException(f"Judging incomplete: {summary['failed_count']} of {summary['sample_count']} conversations failed")
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        if client is not None:
            client.close()
