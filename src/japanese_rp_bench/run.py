"""Generate roleplay conversations through authenticated HTTP clients."""
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from japanese_rp_bench.artifacts import safe_name, write_jsonl as _write_jsonl
from japanese_rp_bench.client import ChatError, validate_completion_policy
from japanese_rp_bench.data import index_by_id, load_dataset_wrapper
from japanese_rp_bench.models import create_client, generate_completion
from japanese_rp_bench.prompts import construct_system_prompts
from japanese_rp_bench.utils import setup_logging


CLIENT_FIELDS = {"model_name", "base_url", "api_key", "api_key_env", "timeout", "max_retries", "request_options", "token_limit_ceiling", "max_token_retries", "strip_think_tags"}
CONFIG_FIELDS = {"dataset_repo", "dataset_split", "cache_dir", "max_turns", "max_workers", "max_samples", "output_dir"} | {
    f"{role}_{field}" for role in ("target", "user") for field in CLIENT_FIELDS
}


class GenerationError(ChatError):
    """An API failure with the completed turns retained for diagnostics."""

    def __init__(self, error, role, history, metadata):
        super().__init__(str(error), status_code=error.status_code, completion=error.completion)
        self.context = {"role": role, "history_index": len(history),
                        "conversation_history": list(history), "generation_metadata": list(metadata)}


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    config = dict(config)
    unknown = set(config) - CONFIG_FIELDS
    if unknown:
        fields = ", ".join(sorted(str(field) for field in unknown))
        raise ValueError(f"Unknown configuration fields: {fields}")
    for name in ("dataset_repo", "target_model_name", "user_model_name"):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    for name, default in (("max_turns", 5), ("max_workers", 1)):
        config[name] = _positive_integer(config.get(name, default), name)
    if config.get("max_samples") is not None:
        _positive_integer(config["max_samples"], "max_samples")
    config.setdefault("dataset_split", "train")
    config.setdefault("cache_dir", None)
    config.setdefault("output_dir", ".")
    if not isinstance(config["output_dir"], str) or not config["output_dir"]:
        raise ValueError("output_dir must be a nonempty path string")
    for role in ("target", "user"):
        options = config.get(f"{role}_request_options", {})
        if not isinstance(options, dict):
            raise ValueError(f"{role}_request_options must be a mapping")
        if {"model", "messages", "stream", "api_key", "authorization", "headers", "base_url"} & options.keys():
            raise ValueError(f"{role}_request_options cannot override routing/authentication fields")
        try:
            json.dumps(options, allow_nan=False)
        except (ValueError, TypeError):
            raise ValueError(f"{role}_request_options must contain finite JSON values") from None
        for token_field in ("max_tokens", "max_completion_tokens"):
            if token_field in options:
                _positive_integer(options[token_field], f"{role}_{token_field}")
        if "max_tokens" in options and "max_completion_tokens" in options:
            raise ValueError(f"{role}_request_options must use only one token-limit field")
        ceiling = config.get(f"{role}_token_limit_ceiling")
        validate_completion_policy(ceiling, config.get(f"{role}_max_token_retries", 2),
                                   config.get(f"{role}_strip_think_tags", False))
        initial = options.get("max_completion_tokens", options.get("max_tokens", 1024))
        if ceiling is not None and initial > ceiling:
            raise ValueError(f"{role}_token_limit_ceiling must cover the initial token limit")
        timeout = config.get(f"{role}_timeout", 120)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"{role}_timeout must be a positive finite number")
        _positive_integer(config.get(f"{role}_max_retries", 2), f"{role}_max_retries", 0)
        for suffix in ("base_url", "api_key", "api_key_env"):
            value = config.get(f"{role}_{suffix}")
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{role}_{suffix} must be a nonempty string")
        endpoint = config.get(f"{role}_base_url")
        if endpoint is not None:
            try:
                url = httpx.URL(endpoint)
            except (httpx.InvalidURL, TypeError):
                raise ValueError(f"{role}_base_url must be a valid HTTP(S) URL") from None
            if url.scheme not in ("http", "https") or not url.host or url.userinfo or url.query or url.fragment:
                raise ValueError(f"{role}_base_url must be HTTP(S), without credentials, query or fragment")
        key = config.get(f"{role}_api_key")
        if key is not None and ("\n" in key or "\r" in key):
            raise ValueError(f"{role}_api_key must be a single-line string")
        key_env = config.get(f"{role}_api_key_env")
        if key is None and key_env and not os.getenv(key_env):
            raise ValueError(f"Required API key environment variable {key_env} is not set")
    return config


def _client_options(config, role):
    return {key: config.get(f"{role}_{key}", default) for key, default in (
        ("base_url", None), ("api_key", None), ("api_key_env", None),
        ("timeout", 120), ("max_retries", 2), ("request_options", {}),
        ("token_limit_ceiling", None), ("max_token_retries", 2), ("strip_think_tags", False),
    )}


def generate_conversation(test_case, idx, config, target_client, user_client, logger):
    """The conversation loop shared by serial and threaded execution."""
    assistant_prompt, user_prompt, first_input = construct_system_prompts(test_case)
    history = [first_input]
    metadata = []
    def respond(client, role, prompt, messages):
        try:
            completion = generate_completion(client, config[f"{role}_model_name"], prompt, messages)
        except ChatError as error:
            raise GenerationError(error, role, history, metadata) from error
        metadata.append({"role": role, "history_index": len(history), "completion": completion.metadata()})
        history.append(completion.content)
    for turn in range(config["max_turns"]):
        logger.info("Processing test case %s, turn %s", idx + 1, turn + 1)
        if turn:
            user_messages = [{"role": "user", "content": "対話開始"}]
            user_messages.extend({"role": "assistant" if i % 2 == 0 else "user", "content": text}
                                 for i, text in enumerate(history))
            respond(user_client, "user", user_prompt, user_messages)
        messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": text}
                    for i, text in enumerate(history)]
        respond(target_client, "target", assistant_prompt, messages)
    return {"target_model_name": config["target_model_name"], "user_model_name": config["user_model_name"],
            "id": test_case["id"], "conversation_history": history, "settings": dict(test_case), "generation_metadata": metadata}


def conversation_output_path(config):
    """Return the collision-safe generated artifact path without running inference."""
    source = config["dataset_repo"]
    local = Path(source).is_file()
    dataset_identity = str(Path(source).resolve()) if local else source
    dataset_label = Path(source).stem if local else source
    identity = {"target": config["target_model_name"], "user": config["user_model_name"],
                "dataset": dataset_identity, "split": config.get("dataset_split", "train"),
                "max_turns": config.get("max_turns", 5), "max_samples": config.get("max_samples")}
    for role in ("target", "user"):
        identity[f"{role}_base_url"] = (config.get(f"{role}_base_url") or
            os.getenv("OPENAI_COMPATIBLE_API_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1")
        identity[f"{role}_request_options"] = config.get(f"{role}_request_options", {})
        for key, default in (("token_limit_ceiling", None), ("max_token_retries", 2), ("strip_think_tags", False)):
            identity[f"{role}_{key}"] = config.get(f"{role}_{key}", default)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
    stem = f"{safe_name(config['target_model_name'])}_{safe_name(dataset_label)}__{digest}"
    return Path(config.get("output_dir", ".")) / "conversations" / f"{stem}.jsonl"


def generate_conversations(config) -> Path:
    """Generate, checkpoint, and return a conversation JSONL artifact path."""
    config = _validate_config(config)
    logger = setup_logging()
    dataset = list(load_dataset_wrapper(config["dataset_repo"], split=config["dataset_split"], cache_dir=config["cache_dir"]))
    index_by_id(dataset, label="scenarios")
    if not dataset:
        raise ValueError("Dataset must contain at least one scenario")
    if config.get("max_samples") is not None:
        dataset = dataset[:config["max_samples"]]
    for test_case in dataset:
        try:
            json.dumps(test_case, allow_nan=False)
        except (ValueError, TypeError):
            raise ValueError("Every scenario must contain finite JSON values") from None
        fields = ("tag", "genre", "world_setting", "scene_setting", "user_setting",
                  "assistant_setting", "dialogue_tone", "response_format", "first_user_input")
        if any(not isinstance(test_case.get(field), str) for field in fields):
            raise ValueError("Every scenario requires string roleplay settings and first_user_input")
        if not test_case["first_user_input"].strip():
            raise ValueError("Every scenario requires a nonempty first_user_input")
        construct_system_prompts(test_case)
    conversations_path = conversation_output_path(config)
    failures_path = Path(config["output_dir"]) / "generation_failures" / conversations_path.name
    for path in (conversations_path, failures_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    clients = []
    conversations, failures = [], []
    try:
        def create(role):
            client = create_client(**_client_options(config, role))
            clients.append(client)
            return client
        target = create("target")
        user = create("user")
        # Clear prior run artifacts only after setup has succeeded.
        _write_jsonl(conversations_path, [])
        _write_jsonl(failures_path, [])
        generation_errors = []
        completed = {}
        def consume(index, test_case, operation):
            try:
                completed[index] = operation()
                conversations[:] = [completed[key] for key in sorted(completed)]
            except Exception as error:
                generation_errors.append(error)
                failure = {"stage": "generation", "id": test_case["id"], "error": type(error).__name__}
                if isinstance(error, GenerationError):
                    failure.update(error.context, error_detail=str(error))
                if isinstance(error, ChatError) and error.completion is not None:
                    failure.update(error_detail=str(error), completion=error.completion.metadata(),
                                   partial_content=error.completion.content)
                failures.append(failure)
            _write_jsonl(conversations_path, conversations)
            _write_jsonl(failures_path, failures)
        def generate(index, test_case):
            return generate_conversation(test_case, index, config, target, user, logger)
        if config["max_workers"] == 1:
            for idx, test_case in enumerate(dataset):
                consume(idx, test_case, lambda idx=idx, test_case=test_case: generate(idx, test_case))
        else:
            with ThreadPoolExecutor(max_workers=config["max_workers"]) as executor:
                futures = {executor.submit(generate, idx, case): (idx, case)
                           for idx, case in enumerate(dataset)}
                for future in as_completed(futures):
                    idx, test_case = futures[future]
                    consume(idx, test_case, future.result)
        if generation_errors:
            raise generation_errors[0]
        return conversations_path
    finally:
        closed = set()
        for client in reversed(clients):
            if id(client) in closed:
                continue
            closed.add(id(client))
            close = getattr(client, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    logger.warning("Failed to close inference client")
