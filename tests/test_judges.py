"""Standalone judging through the real HTTPX boundary and artifact contract."""
import json
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner

from japanese_rp_bench.artifacts import safe_name, score_paths
from japanese_rp_bench.client import ChatClient
from japanese_rp_bench import judging


CATEGORIES = ["Roleplay Adherence", "Consistency", "Contextual Understanding", "Expressiveness", "Creativity", "Naturalness of Japanese", "Enjoyment of the Dialogue", "Appropriateness of Turn-Taking"]


def evaluation(score=4):
    return {"Evaluation Reason": "自然な会話", **dict.fromkeys(CATEGORIES, score)}


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records), encoding='utf-8')
    return path


def scenario(identifier, world="村"):
    return {"id": identifier, "world_setting": world, "assistant_setting": "船長", "user_setting": "旅人"}


def conversation(identifier, embedded=True):
    value = {"target_model_name": "target", "id": identifier, "conversation_history": ["こんにちは", "ようこそ"]}
    if embedded:
        value["settings"] = scenario(identifier)
    return value


@pytest.fixture
def http_client():
    clients = []
    def create(responses):
        pending = iter(responses)
        requests = []
        def respond(request):
            requests.append(json.loads(request.content))
            outcome = next(pending)
            if isinstance(outcome, BaseException):
                raise outcome
            if isinstance(outcome, int):
                return httpx.Response(outcome, json={"error": {"message": "failure"}})
            return httpx.Response(200, json={"choices": [{"message": {"content": outcome}}]})
        client = ChatClient("https://judge.invalid/v1", api_key="offline-key", max_retries=0, transport=httpx.MockTransport(respond))
        clients.append(client)
        return client, requests
    yield create
    for client in clients:
        client.close()


def test_artifact_namespace_distinguishes_model_and_judge(tmp_path):
    first = score_paths("org/model_a", "judge/a", tmp_path)
    second = score_paths("org/model_b", "judge/a", tmp_path)
    third = score_paths("org/model_a", "judge/b", tmp_path)
    assert first["scores"] != second["scores"] != third["scores"]
    assert set(first) == {"scores", "answers", "judgements"}
    assert safe_name("org/model") != safe_name("org%2Fmodel")
    assert all(path.is_relative_to(tmp_path) for path in score_paths("..", "../judge", tmp_path).values())




def test_prepare_joins_ids_and_preserves_zero(tmp_path):
    source = write_jsonl(tmp_path / "source.jsonl", [conversation(2, False), conversation(0, False)])
    dataset = write_jsonl(tmp_path / "dataset.jsonl", [scenario(0, "海"), scenario(2, "山")])
    prepared = judging.prepare_conversations(source, dataset_path=dataset)
    assert [record["id"] for record in prepared] == ["2", "0"]
    assert [record["settings"]["world_setting"] for record in prepared] == ["山", "海"]
    assert [record["answer"]["id"] for record in prepared] == [2, 0]


def test_embedded_settings_avoid_dataset_download(tmp_path, monkeypatch):
    source = write_jsonl(tmp_path / "source.jsonl", [conversation(0)])
    monkeypatch.setattr(judging, "load_dataset_wrapper", lambda *a, **k: pytest.fail("Unexpected dataset download"))
    assert judging.prepare_conversations(source)[0]["settings"]["assistant_setting"] == "船長"


@pytest.mark.parametrize("records", [[conversation(1), conversation("1")], [{"conversation_history": []}], [dict(conversation(1), settings=scenario(2))]])
def test_prepare_rejects_duplicate_missing_or_mismatched_ids(tmp_path, records):
    with pytest.raises(ValueError):
        judging.prepare_conversations(write_jsonl(tmp_path / "source.jsonl", records))


def test_prepare_rejects_unknown_id(tmp_path):
    source = write_jsonl(tmp_path / "source.jsonl", [conversation(9, False)])
    dataset = write_jsonl(tmp_path / "dataset.jsonl", [scenario(1)])
    with pytest.raises(ValueError, match="(?i)(unknown|missing|not found)"):
        judging.prepare_conversations(source, dataset_path=dataset)


def test_prepare_rejects_duplicate_dataset_ids(tmp_path):
    source = write_jsonl(tmp_path / "source.jsonl", [conversation(1, False)])
    dataset = write_jsonl(tmp_path / "dataset.jsonl", [scenario(1), scenario("1")])
    with pytest.raises(ValueError, match="(?i)duplicate"):
        judging.prepare_conversations(source, dataset_path=dataset)


def test_engine_averages_complete_evaluations_and_saves_exact_subset(tmp_path, http_client):
    originals = [conversation(0), conversation(1), conversation(2)]
    source = write_jsonl(tmp_path / "org_model.jsonl", originals)
    prepared = judging.prepare_conversations(source, max_samples=2)
    client, requests = http_client([json.dumps(evaluation(2)), json.dumps(evaluation(4))])
    summary = judging.judge_conversations(prepared, client=client, judge_model="judge/a", output_dir=tmp_path / "out", max_workers=1)
    assert summary["overall_average"] == 3
    assert summary["evaluated_count"] == summary["sample_count"] == 2
    assert summary["failed_count"] == 0
    paths = score_paths("target", "judge/a", tmp_path / "out")
    assert [json.loads(line) for line in paths["answers"].read_text().splitlines()] == originals[:2]
    records = [json.loads(line) for line in paths["judgements"].read_text().splitlines()]
    assert [record["id"] for record in records] == ["0", "1"]
    assert records[0]["evaluation"]["Evaluation Reason"] == "自然な会話"
    rendered = "\n".join(message["content"] for message in requests[0]["messages"])
    assert "Consistency" in rendered
    assert "船長" in rendered
    assert "{{formatted_data}}" not in rendered
    assert "offline-key" not in ''.join(path.read_text() for path in paths.values())


@pytest.mark.parametrize("invalid", ['{"Consistency":"five"}', "not JSON", json.dumps(evaluation(True)), json.dumps(evaluation(6)), json.dumps({**evaluation(), "Evaluation Reason": ""})])
def test_invalid_response_persists_raw_and_does_not_affect_averages(tmp_path, http_client, invalid):
    source = write_jsonl(tmp_path / "source.jsonl", [conversation(0), conversation(1)])
    client, _ = http_client([json.dumps(evaluation(4)), invalid])
    result = judging.judge_conversations(judging.prepare_conversations(source), client=client, judge_model="judge", output_dir=tmp_path, max_workers=1)
    assert result["overall_average"] == 4
    assert result["evaluated_count"] == result["failed_count"] == 1
    assert result["status"] == "incomplete"
    records = [json.loads(line) for line in score_paths("target", "judge", tmp_path)["judgements"].read_text().splitlines()]
    assert records[1]["raw_evaluation"] == invalid
    assert records[1]["evaluation"] is None
    assert records[1]["error"]


def test_http_failure_is_recorded_with_source_id(tmp_path, http_client):
    source = write_jsonl(tmp_path / "source.jsonl", [conversation(0)])
    client, _ = http_client([503])
    result = judging.judge_conversations(judging.prepare_conversations(source), client=client, judge_model="judge", output_dir=tmp_path)
    assert result["failed_count"] == 1
    assert result["overall_average"] is None
    record = json.loads(score_paths("target", "judge", tmp_path)["judgements"].read_text())
    assert record["id"] == "0"
    assert record["error"]


def test_cli_uses_shared_artifacts(tmp_path, monkeypatch, http_client):
    source = write_jsonl(tmp_path / "org_model.jsonl", [conversation(0)])
    client, requests = http_client([json.dumps(evaluation())])
    monkeypatch.setattr(judging, "create_client", lambda **k: client)
    result = CliRunner().invoke(judging.main, ["--conversation-file", str(source), "--judge-model", "judge", "--output-dir", str(tmp_path / "out"), "--base-url", "https://judge.invalid/v1"])
    assert result.exit_code == 0, result.output
    assert score_paths("target", "judge", tmp_path / "out")["scores"].exists()
    assert requests[0]["model"] == "judge"




def test_incomplete_cli_exits_nonzero_after_persisting_error(tmp_path, monkeypatch, http_client):
    source = write_jsonl(tmp_path / "source.jsonl", [conversation(0)])
    client, _ = http_client(["invalid"])
    monkeypatch.setattr(judging, "create_client", lambda **k: client)
    result = CliRunner().invoke(judging.main, ["--judge-model", "judge", "--conversation-file", str(source), "--output-dir", str(tmp_path / "out")])
    assert result.exit_code != 0
    assert score_paths("target", "judge", tmp_path / "out")["judgements"].exists()


@pytest.mark.parametrize("flag,value", [("--max-workers", "0"), ("--timeout", "0"), ("--max-retries", "-1"), ("--max-samples", "0")])
def test_cli_rejects_invalid_limits_before_creating_client(tmp_path, monkeypatch, flag, value):
    source = write_jsonl(tmp_path / "source.jsonl", [conversation(0)])
    monkeypatch.setattr(judging, "create_client", lambda *a, **k: pytest.fail("No call for invalid config"))
    result = CliRunner().invoke(judging.main, ["--judge-model", "judge", "--conversation-file", str(source), flag, value])
    assert result.exit_code == 2
    assert flag in result.output


def test_request_options_override_defaults_without_conflicting_token_limits(tmp_path, http_client):
    source = write_jsonl(tmp_path / "source.jsonl", [conversation(0)])
    client, requests = http_client([json.dumps(evaluation())])
    judging.judge_conversations(judging.prepare_conversations(source), client=client, judge_model="judge", output_dir=tmp_path,
                               request_options={"max_completion_tokens": 9000, "temperature": 1, "chat_template_kwargs": {"enable_thinking": False}})
    assert requests[0]["max_completion_tokens"] == 9000
    assert "max_tokens" not in requests[0]
    assert requests[0]["temperature"] == 1
    assert requests[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_judging_defaults_have_sufficient_output_budget(tmp_path, http_client):
    source = write_jsonl(tmp_path / "source.jsonl", [conversation(0)])
    client, requests = http_client([json.dumps(evaluation())])
    judging.judge_conversations(judging.prepare_conversations(source), client=client, judge_model="judge", output_dir=tmp_path)
    assert requests[0]["max_tokens"] == 8192
    assert requests[0]["temperature"] == 0


def test_interrupted_judging_does_not_leave_old_successful_scores(tmp_path, http_client):
    source = write_jsonl(tmp_path / "source.jsonl", [conversation(0)])
    client, _ = http_client([KeyboardInterrupt()])
    paths = score_paths("target", "judge", tmp_path)
    paths["scores"].parent.mkdir(parents=True)
    paths["scores"].write_text('{"status":"complete","overall_average":5}')
    with pytest.raises(KeyboardInterrupt):
        judging.judge_conversations(judging.prepare_conversations(source), client=client, judge_model="judge", output_dir=tmp_path)
    assert json.loads(paths["scores"].read_text())["status"] == "incomplete"


def test_atomic_write_failure_preserves_previous_artifact(tmp_path, monkeypatch):
    from japanese_rp_bench.artifacts import atomic_write
    path = tmp_path / "scores.json"
    path.write_text("previous")
    def fail(*args):
        raise OSError("disk failure")
    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="disk failure"):
        atomic_write(path, "replacement")
    assert path.read_text() == "previous"
    assert not list(tmp_path.glob('.pending-*'))


def test_unexpected_api_exception_does_not_persist_secret(tmp_path, http_client):
    source = write_jsonl(tmp_path / "source.jsonl", [conversation(0)])
    client, _ = http_client([ValueError("secret-key-from-client")])
    result = judging.judge_conversations(judging.prepare_conversations(source), client=client, judge_model="judge", output_dir=tmp_path)
    assert result["failed_count"] == 1
    assert "secret-key-from-client" not in score_paths("target", "judge", tmp_path)["judgements"].read_text()


@pytest.mark.parametrize("limit", [True, 1.5, "2", 0, -1])
def test_prepare_rejects_invalid_sample_limits(tmp_path, limit):
    source = write_jsonl(tmp_path / "source.jsonl", [conversation(0)])
    with pytest.raises(ValueError, match="positive integer"):
        judging.prepare_conversations(source, max_samples=limit)


def test_empty_conversation_history_is_rejected(tmp_path):
    source = write_jsonl(tmp_path / "source.jsonl", [{**conversation(0), "conversation_history": []}])
    with pytest.raises(ValueError, match="(?i)(history|message|conversation)"):
        judging.prepare_conversations(source)


def test_default_rubric_uses_full_packaged_benchmark_prompt(tmp_path, http_client):
    source = write_jsonl(tmp_path / "source.jsonl", [conversation(0)])
    client, requests = http_client([json.dumps(evaluation())])
    judging.judge_conversations(judging.prepare_conversations(source), client=client, judge_model="judge", output_dir=tmp_path)
    prompt = '\n'.join(message["content"] for message in requests[0]["messages"])
    assert "Perfectly adheres to the roleplay settings" in prompt
    assert "5. **Creativity**" in prompt






def test_judge_cli_accepts_huggingface_dataset_identifier(tmp_path, monkeypatch, http_client):
    source = write_jsonl(tmp_path / "source.jsonl", [conversation(0, embedded=False)])
    datasets = []
    def load(repo, **kwargs):
        datasets.append(repo)
        return [scenario(0)]
    monkeypatch.setattr(judging, "load_dataset_wrapper", load)
    client, _ = http_client([json.dumps(evaluation())])
    monkeypatch.setattr(judging, "create_client", lambda **k: client)
    result = CliRunner().invoke(judging.main, ["--judge-model", "judge", "--conversation-file", str(source), "--dataset", "org/scenarios", "--output-dir", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert datasets == ["org/scenarios"]


def test_prepare_rejects_mixed_target_model_identities(tmp_path):
    source = write_jsonl(tmp_path / "source.jsonl", [
        {**conversation(0), "target_model_name": "model-a"},
        {**conversation(1), "target_model_name": "model-b"},
    ])
    with pytest.raises(ValueError, match="(?i)mixed.*model"):
        judging.prepare_conversations(source)


@pytest.mark.parametrize("identity", [True, "", "  ", {}])
def test_prepare_rejects_invalid_declared_target_model_identity(tmp_path, identity):
    source = write_jsonl(tmp_path / "source.jsonl", [{**conversation(0), "target_model_name": identity}])
    with pytest.raises(ValueError, match="(?i)model"):
        judging.prepare_conversations(source)


@pytest.mark.parametrize("missing_index", [0, 1])
def test_every_source_record_requires_target_identity_even_outside_subset(tmp_path, missing_index):
    records = [conversation(0), conversation(1)]
    del records[missing_index]["target_model_name"]
    source = write_jsonl(tmp_path / "source.jsonl", records)
    with pytest.raises(ValueError, match="target_model_name"):
        judging.prepare_conversations(source, max_samples=1)


def test_missing_embedded_settings_requires_explicit_dataset(tmp_path, monkeypatch):
    source = write_jsonl(tmp_path / "source.jsonl", [conversation(0, embedded=False)])
    monkeypatch.setattr(judging, "load_dataset_wrapper", lambda *a, **k: pytest.fail("No implicit dataset download"))
    with pytest.raises(ValueError, match="(?i)explicit.*dataset|dataset.*required"):
        judging.prepare_conversations(source)


@pytest.mark.parametrize("flag", ["--model", "--judge-model-name", "-j", "--target-model", "-t", "--target-model-name", "--num-conversations", "-n", "--temp-dir", "--concurrency-limit"])
def test_cli_rejects_removed_aliases(tmp_path, monkeypatch, flag):
    source = write_jsonl(tmp_path / "source.jsonl", [conversation(0)])
    monkeypatch.setattr(judging, "create_client", lambda **k: pytest.fail("No API client for obsolete flags"))
    result = CliRunner().invoke(judging.main, ["--conversation-file", str(source), "--judge-model", "judge", flag, "1"])
    assert result.exit_code == 2
    assert "No such option" in result.output
    assert flag in result.output


def test_cli_does_not_resolve_model_to_conversation_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_jsonl(tmp_path / "conversations/target_shisa-ai-shisa-rp-bench-testset.jsonl", [conversation(0)])
    monkeypatch.setattr(judging, "create_client", lambda **k: pytest.fail("No API client for guessed source"))
    result = CliRunner().invoke(judging.main, ["--conversation-file", "target", "--judge-model", "judge"])
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_cli_default_output_uses_metadata_not_filename(tmp_path, monkeypatch, http_client):
    monkeypatch.chdir(tmp_path)
    source = write_jsonl(tmp_path / "misleading.jsonl", [conversation(0)])
    client, _ = http_client([json.dumps(evaluation())])
    monkeypatch.setattr(judging, "create_client", lambda **k: client)
    result = CliRunner().invoke(judging.main, ["--conversation-file", str(source), "--judge-model", "judge"])
    assert result.exit_code == 0, result.output
    assert score_paths("target", "judge", "scores")["scores"].exists()
    assert not score_paths("misleading", "judge", "scores")["scores"].exists()
