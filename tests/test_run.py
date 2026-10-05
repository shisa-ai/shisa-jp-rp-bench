"""Offline generation integration tests with real prompts and files."""
import copy
import json
import logging
from pathlib import Path

import pytest

from japanese_rp_bench import run
from japanese_rp_bench.client import Completion

@pytest.fixture
def scenario():
    return {
        "id": "scene-1", "tag": "SFW", "genre": "冒険", "world_setting": "森",
        "scene_setting": "朝", "user_setting": "旅人", "assistant_setting": "案内人",
        "dialogue_tone": "親切", "response_format": "台詞のみ", "first_user_input": "こんにちは",
    }


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return {
        "dataset_repo": "org/data", "dataset_split": "test", "cache_dir": None,
        "target_model_name": "org/target",
        "user_model_name": "org/user",
        "max_turns": 2,
    }


def prepare(monkeypatch, scenario):
    monkeypatch.setattr(run, "load_dataset_wrapper", lambda *a, **k: [copy.deepcopy(scenario)])
    monkeypatch.setattr(run, "create_client", lambda **k: object())
    calls = []
    def generate(client, name, system, conversations, **kwargs):
        calls.append((name, copy.deepcopy(conversations), kwargs, system))
        return Completion("続けて" if name == "org/user" else "案内します")
    monkeypatch.setattr(run, "generate_completion", generate)
    return calls


def make_output_dirs():
    Path("conversations").mkdir()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def _single_artifact(directory, *, failure=False):
    paths = [path for path in Path(directory).glob("*.jsonl")
             if path.name.endswith("_failures.jsonl") == failure]
    assert len(paths) == 1, paths
    return paths[0]


def conversation_path():
    return _single_artifact("conversations")


def failure_path():
    return _single_artifact("generation_failures")


def test_conversation_generation_alternates_character_roles(monkeypatch, config, scenario):
    calls = prepare(monkeypatch, scenario)
    result = run.generate_conversation(scenario, 0, config, "target", "user", logging.getLogger("test"))
    result.pop("generation_metadata")  # Diagnostics are checked at the HTTP boundary in test_reasoning.
    assert result == {"id": "scene-1", "target_model_name": "org/target", "user_model_name": "org/user",
                      "conversation_history": ["こんにちは", "案内します", "続けて", "案内します"], "settings": scenario}
    assert calls[0][1] == [{"role": "user", "content": "こんにちは"}]
    assert calls[1][1] == [{"role": "user", "content": "対話開始"},
        {"role": "assistant", "content": "こんにちは"}, {"role": "user", "content": "案内します"}]
    assert calls[2][1] == [{"role": "user", "content": "こんにちは"},
        {"role": "assistant", "content": "案内します"}, {"role": "user", "content": "続けて"}]
    assert all(call[2] == {} for call in calls)
    assert all("### あなたがなりきる人物の設定:\n案内人" in calls[i][3] for i in (0, 2))
    assert "### あなたがなりきる人物の設定:\n旅人" in calls[1][3]


@pytest.mark.parametrize("workers", [1, 2])
def test_generation_only_writes_jsonl(monkeypatch, config, scenario, workers):
    calls = prepare(monkeypatch, scenario)
    config["max_workers"] = workers
    make_output_dirs()
    output = run.generate_conversations(config)
    assert output == conversation_path()
    assert len(calls) == 3
    assert all("### あなたがなりきる人物の設定:\n案内人" in calls[i][3] for i in (0, 2))
    assert "### あなたがなりきる人物の設定:\n旅人" in calls[1][3]
    assert read_jsonl(conversation_path())[0]["conversation_history"] == [
        "こんにちは", "案内します", "続けて", "案内します"]
    assert not Path("evaluations").exists()


@pytest.mark.parametrize("workers", [1, 2])
def test_generation_preserves_dataset_order(monkeypatch, config, scenario, workers):
    prepare(monkeypatch, scenario)
    monkeypatch.setattr(run, "load_dataset_wrapper", lambda *a, **k: [dict(scenario, id="b"), dict(scenario, id="a")])
    config["max_workers"] = workers
    make_output_dirs()
    run.generate_conversations(config)
    assert [row["id"] for row in read_jsonl(conversation_path())] == ["b", "a"]


@pytest.mark.parametrize("failure_at", ["load_dataset_wrapper", "create_client", "generate_completion"])
def test_dependency_failures_are_propagated(monkeypatch, config, scenario, failure_at):
    prepare(monkeypatch, scenario)
    make_output_dirs()
    def fail(*args, **kwargs):
        raise RuntimeError("dependency failed")
    monkeypatch.setattr(run, failure_at, fail)
    with pytest.raises(RuntimeError, match="dependency failed"):
        run.generate_conversations(config)


@pytest.mark.parametrize("field,value", [("max_turns", 0), ("max_turns", True), ("max_workers", -1),
    ("max_samples", 0), ("max_turns", "2"),
    ("target_model_name", ""), ("target_inference_method", "google_api"),
    ("target_request_options", []), ("typo_key", "unexpected")])
def test_invalid_config_fails_before_loading_clients(monkeypatch, config, scenario, field, value):
    calls = prepare(monkeypatch, scenario)
    loaded = []
    monkeypatch.setattr(run, "create_client", lambda *a, **k: loaded.append(a))
    config[field] = value
    with pytest.raises(ValueError):
        run.generate_conversations(config)
    assert not loaded
    assert not calls


@pytest.mark.parametrize("configuration", [None, [], "text"])
def test_configuration_must_be_a_mapping(configuration):
    with pytest.raises(ValueError, match="configuration"):
        run.generate_conversations(configuration)


def test_role_http_options_propagate_and_clients_close(monkeypatch, config, scenario):
    from unittest.mock import Mock
    prepare(monkeypatch, scenario)
    loaded = []
    clients = []
    def load(**kwargs):
        client = Mock()
        clients.append(client)
        loaded.append(kwargs)
        return client
    monkeypatch.setattr(run, "create_client", load)
    config.update(target_base_url="https://target.test/v1", target_api_key_env="TARGET_TEST_KEY",
                  user_base_url="https://user.test/v1", user_api_key="test-user-secret", user_request_options={"max_tokens": 8},
                  target_timeout=9, target_max_retries=0)
    monkeypatch.setenv("TARGET_TEST_KEY", "test-target-secret")
    run.generate_conversations(config)
    assert loaded[0]["base_url"] == "https://target.test/v1"
    assert loaded[0]["api_key_env"] == "TARGET_TEST_KEY"
    assert loaded[0]["timeout"] == 9
    assert loaded[0]["max_retries"] == 0
    assert loaded[1]["api_key"] == "test-user-secret"
    assert loaded[1]["request_options"] == {"max_tokens": 8}
    for client in clients:
        client.close.assert_called_once_with()
    for path in [*Path("conversations").glob("*.jsonl"), *Path("generation_failures").glob("*.jsonl")]:
        assert "test-user-secret" not in path.read_text()
        assert "test-target-secret" not in path.read_text()


@pytest.mark.parametrize("failure", ["load", "generate"])
def test_clients_close_on_failure(monkeypatch, config, scenario, failure):
    from unittest.mock import Mock
    prepare(monkeypatch, scenario)
    clients = []
    def load(*args, **kwargs):
        if failure == "load" and clients:
            raise RuntimeError("client construction failed")
        client = Mock()
        clients.append(client)
        return client
    monkeypatch.setattr(run, "create_client", load)
    def fail(*args, **kwargs):
        raise RuntimeError("inference failed")
    if failure == "generate":
        monkeypatch.setattr(run, "generate_completion", fail)
    with pytest.raises(RuntimeError):
        run.generate_conversations(config)
    assert clients
    for client in clients:
        client.close.assert_called_once_with()


def test_max_samples_limits_before_generation_and_keeps_settings(monkeypatch, config, scenario):
    calls = prepare(monkeypatch, scenario)
    monkeypatch.setattr(run, "load_dataset_wrapper", lambda *a, **k: [scenario, dict(scenario, id="second")])
    config.update(max_samples=1, max_workers=2)
    output = run.generate_conversations(config)
    assert output == conversation_path()
    assert len(calls) == 3
    records = read_jsonl(conversation_path())
    assert len(records) == 1
    assert records[0]["settings"] == scenario


@pytest.mark.parametrize("records_kind", ["duplicate", "missing", "empty"])
def test_invalid_scenario_ids_fail_before_inference(monkeypatch, config, scenario, records_kind):
    calls = prepare(monkeypatch, scenario)
    missing = dict(scenario)
    missing.pop("id")
    records = {"duplicate": [scenario, scenario], "missing": [missing], "empty": []}[records_kind]
    monkeypatch.setattr(run, "load_dataset_wrapper", lambda *a, **k: records)
    with pytest.raises(ValueError):
        run.generate_conversations(config)
    assert not calls


@pytest.mark.parametrize("field,value", [("target_inference_method", []), ("judge_inference_methods", [[]]),
    ("target_request_options", {"max_tokens": float("nan")}), ("target_base_url", "not-a-url"),
    ("target_request_options", {"headers": {"Authorization": "Bearer secret"}})])
def test_invalid_http_configuration_is_rejected_before_client_creation(monkeypatch, config, scenario, field, value):
    prepare(monkeypatch, scenario)
    config[field] = value
    loaded = []
    monkeypatch.setattr(run, "create_client", lambda **k: loaded.append(k))
    with pytest.raises(ValueError):
        run.generate_conversations(config)
    assert not loaded


@pytest.mark.parametrize("field,value", [("first_user_input", None), ("first_user_input", ""), ("assistant_setting", {}), ("extra_metadata", float("nan"))])
def test_invalid_scenario_fields_fail_before_paid_generation(monkeypatch, config, scenario, field, value):
    scenario[field] = value
    calls = prepare(monkeypatch, scenario)
    with pytest.raises(ValueError, match="scenario"):
        run.generate_conversations(config)
    assert not calls


@pytest.mark.parametrize("workers", [1, 2])
def test_partial_generation_failure_retains_other_completed_cases(monkeypatch, config, scenario, workers):
    prepare(monkeypatch, scenario)
    config.update(max_workers=workers, max_turns=1)
    records = [dict(scenario, id="bad", first_user_input="fail"), dict(scenario, id="good", first_user_input="succeed")]
    monkeypatch.setattr(run, "load_dataset_wrapper", lambda *a, **k: records)
    def generate(*args, **kwargs):
        if args[3][0]["content"] == "fail":
            raise RuntimeError("request failed")
        return Completion("success")
    monkeypatch.setattr(run, "generate_completion", generate)
    with pytest.raises(RuntimeError, match="request failed"):
        run.generate_conversations(config)
    assert [item["id"] for item in read_jsonl(conversation_path())] == ["good"]
    assert read_jsonl(failure_path()) == [{"stage": "generation", "id": "bad", "error": "RuntimeError"}]


def test_local_dataset_through_real_httpx_adapters(monkeypatch, config, scenario):
    """Exercise the complete runner; replace only HTTP transport, never inference."""
    import httpx
    from japanese_rp_bench import models
    from japanese_rp_bench.client import ChatClient

    Path("scenarios.jsonl").write_text(json.dumps(scenario, ensure_ascii=False) + "\n", encoding="utf-8")
    config.update(dataset_repo="scenarios.jsonl", max_workers=2,
                  target_base_url="https://target.test/v1", target_api_key="target-secret",
                  user_base_url="https://user.test/v1", user_api_key="user-secret",
                  target_request_options={"max_tokens": 64}, user_request_options={"max_tokens": 32})
    requests, clients = [], []

    def respond(request):
        payload = json.loads(request.content)
        requests.append((request.url.host, request.headers["Authorization"], payload))
        text = "HTTPX経由の応答"
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    def client_factory(*args, **kwargs):
        client = ChatClient(*args, transport=httpx.MockTransport(respond), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(models, "ChatClient", client_factory)
    run.generate_conversations(config)
    assert [(host, auth) for host, auth, _ in requests] == [
        ("target.test", "Bearer target-secret"), ("user.test", "Bearer user-secret"),
        ("target.test", "Bearer target-secret"),
    ]
    assert [payload["max_tokens"] for _, _, payload in requests] == [64, 32, 64]
    result = read_jsonl(conversation_path())[0]
    assert result["conversation_history"] == ["こんにちは", "HTTPX経由の応答", "HTTPX経由の応答", "HTTPX経由の応答"]
    assert result["settings"] == scenario
    assert all(client._http.is_closed for client in clients)


def test_artifact_paths_separate_models_datasets_and_users(config):
    base = dict(config)
    cases = [base, dict(base, target_model_name="org-target"), dict(base, dataset_repo="org-data"),
             dict(base, user_model_name="other-user")]
    conversation_paths = [run.conversation_output_path(case) for case in cases]
    assert len(set(conversation_paths)) == 4
    path = conversation_paths[0]
    assert path.name.startswith("org%2Ftarget_org%2Fdata__")
    assert path.parent == Path("conversations")


def test_local_dataset_same_basename_does_not_collide(config, tmp_path):
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "data.jsonl").write_text("{}\n")
    first = run.conversation_output_path(dict(config, dataset_repo=str(tmp_path / "a/data.jsonl")))
    second = run.conversation_output_path(dict(config, dataset_repo=str(tmp_path / "b/data.jsonl")))
    assert first != second


@pytest.mark.parametrize("change", [{"max_turns": 1},
    {"target_base_url": "https://alternate.test/v1"}, {"user_base_url": "https://alternate.test/v1"},
    {"target_request_options": {"temperature": 0.1}}, {"user_request_options": {"max_tokens": 99}},
    {"max_samples": 1}])
def test_generation_ablations_have_distinct_artifacts(config, change):
    assert run.conversation_output_path(config) != run.conversation_output_path(dict(config, **change))


def test_parallel_checkpoints_completed_cases_before_slow_first_case(monkeypatch, config, scenario):
    from threading import Event
    prepare(monkeypatch, scenario)
    config.update(max_workers=2, max_turns=1)
    records = [dict(scenario, id="slow", first_user_input="slow"), dict(scenario, id="fast", first_user_input="fast")]
    monkeypatch.setattr(run, "load_dataset_wrapper", lambda *a, **k: records)
    checkpoint_saved = Event()
    original_write = run._write_jsonl
    def write(path, items):
        original_write(path, items)
        if path.parent.name == "conversations" and [item["id"] for item in items] == ["fast"]:
            checkpoint_saved.set()
    def generate(*args, **kwargs):
        if args[3][0]["content"] == "slow" and not checkpoint_saved.wait(timeout=2):
            raise RuntimeError("Completed fast case was not checkpointed")
        return Completion("response")
    monkeypatch.setattr(run, "_write_jsonl", write)
    monkeypatch.setattr(run, "generate_completion", generate)
    run.generate_conversations(config)
    assert checkpoint_saved.is_set()
    assert [record["id"] for record in read_jsonl(conversation_path())] == ["slow", "fast"]


@pytest.mark.parametrize("role", ["target", "user"])
@pytest.mark.parametrize("suffix,value", [
    ("request_options", {"max_tokens": 20, "max_completion_tokens": 20}),
    ("request_options", {"max_completion_tokens": 0}),
    ("timeout", 0), ("timeout", float("inf")), ("timeout", True),
    ("base_url", ""), ("base_url", "https://example.test/\x00"),
    ("api_key", "secret\r\ninjected: header"), ("api_key_env", "RUNNER_MISSING_TEST_KEY"),
])
def test_each_role_rejects_unusable_http_options_before_loading(monkeypatch, config, scenario, role, suffix, value):
    from unittest.mock import Mock
    prepare(monkeypatch, scenario)
    config[f"{role}_{suffix}"] = value
    monkeypatch.delenv("RUNNER_MISSING_TEST_KEY", raising=False)
    load = Mock()
    monkeypatch.setattr(run, "create_client", load)
    with pytest.raises(ValueError):
        run.generate_conversations(config)
    load.assert_not_called()
    assert not Path("conversations").exists()


@pytest.mark.parametrize("field,value", [("output_dir", ""), ("output_dir", [])])
def test_invalid_artifact_configuration_prevents_inference(monkeypatch, config, scenario, field, value):
    from unittest.mock import Mock
    prepare(monkeypatch, scenario)
    config[field] = value
    load = Mock()
    monkeypatch.setattr(run, "create_client", load)
    with pytest.raises(ValueError, match=field):
        run.generate_conversations(config)
    load.assert_not_called()


def test_client_close_failure_does_not_mask_generation_failure_or_skip_other_clients(monkeypatch, config, scenario, caplog):
    from unittest.mock import Mock
    prepare(monkeypatch, scenario)
    target = Mock()
    user = Mock()
    user.close.side_effect = RuntimeError("sensitive cleanup details")
    monkeypatch.setattr(run, "create_client", Mock(side_effect=[target, user]))
    monkeypatch.setattr(run, "generate_completion", Mock(side_effect=RuntimeError("generation unavailable")))
    with pytest.raises(RuntimeError, match="generation unavailable"):
        run.generate_conversations(config)
    target.close.assert_called_once_with()
    user.close.assert_called_once_with()
    assert "Failed to close inference client" in caplog.text
    assert "sensitive cleanup details" not in caplog.text
    assert read_jsonl(failure_path()) == [{"stage": "generation", "id": "scene-1", "error": "RuntimeError"}]


@pytest.mark.parametrize("field,value", [
    ("target_inference_method", "httpx"), ("user_inference_method", "openai_api"),
    ("judge_inference_methods", ["openai_compatible_api"]),
    ("judge_inference_method", "vllm"), ("inference_method", "transformers"),
    ("tensor_parallel_size", 1), ("no_judge", True), ("low_context", True),
    ("ultra_low_context", True), ("judge_model_names", ["judge"]),
    ("judge_max_attempts", 3), ("evaluation_prompt_file", "rubric.txt"),
])
def test_unknown_fields_require_explicit_config_migration(monkeypatch, config, scenario, field, value):
    from unittest.mock import Mock
    prepare(monkeypatch, scenario)
    config[field] = value
    create = Mock()
    monkeypatch.setattr(run, "create_client", create)
    with pytest.raises(ValueError, match="Unknown configuration"):
        run.generate_conversations(config)
    create.assert_not_called()


@pytest.mark.parametrize("role", ["target", "user"])
def test_explicit_role_key_takes_precedence_over_missing_named_environment(monkeypatch, config, scenario, role):
    import httpx
    from japanese_rp_bench import models
    from japanese_rp_bench.client import ChatClient

    prepare(monkeypatch, scenario)
    # Use the real client creation and inference functions at the HTTP boundary.
    monkeypatch.setattr(run, "create_client", models.create_client)
    monkeypatch.setattr(run, "generate_completion", models.generate_completion)
    monkeypatch.delenv("RUNNER_ABSENT_AUTH_ENV", raising=False)
    config.update(max_turns=2)
    config[f"{role}_api_key"] = "explicit-role-key"
    config[f"{role}_api_key_env"] = "RUNNER_ABSENT_AUTH_ENV"
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append((payload["model"], request.headers.get("Authorization")))
        text = "response"
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    monkeypatch.setattr(models, "ChatClient", lambda **options: ChatClient(transport=httpx.MockTransport(respond), **options))
    run.generate_conversations(config)
    model_name = {"target": "org/target", "user": "org/user"}[role]
    matching_requests = [authorization for model, authorization in requests if model == model_name]
    assert matching_requests
    assert all(authorization == "Bearer explicit-role-key" for authorization in matching_requests)
