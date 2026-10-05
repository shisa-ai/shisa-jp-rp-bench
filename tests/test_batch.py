"""Offline batch aggregation and command-line orchestration contracts."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
from click.testing import CliRunner

from japanese_rp_bench import batch
from japanese_rp_bench.artifacts import safe_name, score_paths

CATEGORIES = ("Roleplay Adherence", "Consistency", "Contextual Understanding", "Expressiveness", "Creativity", "Naturalness of Japanese", "Enjoyment of the Dialogue", "Appropriateness of Turn-Taking")


def evaluation(score=3, **updates):
    return {**dict.fromkeys(CATEGORIES, score), "Evaluation Reason": "具体的な評価", **updates}


@pytest.fixture
def batch_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "conversations").mkdir()
    return tmp_path


def source_file(root, model):
    path = root / "conversations" / (safe_name(model) + "__run.jsonl")
    path.write_text(json.dumps({"id": 0, "target_model_name": model, "conversation_history": ["質問", "回答"]}) + "\n")
    return path


def write_stale_scores(root, model, judge):
    path = score_paths(model, judge, root / "scores")["scores"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model_name": model, "judge_model_name": judge, "status": "complete",
        "sample_count": 1, "evaluated_count": 1, "failed_count": 0,
        "overall_average": 5, "category_averages": dict.fromkeys(CATEGORIES, 5)}))


def publish_scores(command, score=3, **changes):
    from japanese_rp_bench.artifacts import score_paths
    source = Path(command[command.index("--conversation-file") + 1])
    model = json.loads(source.read_text().splitlines()[0])["target_model_name"]
    judge = command[command.index("--judge-model") + 1]
    directory = command[command.index("--output-dir") + 1]
    paths = score_paths(model, judge, directory)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    paths["scores"].write_text(json.dumps({"model_name": model, "judge_model_name": judge,
        "category_averages": dict.fromkeys(CATEGORIES, score), "overall_average": score,
        "sample_count": 1, "evaluated_count": 1, "failed_count": 0, "status": "complete", **changes}))
    return subprocess.CompletedProcess(command, 0)


def test_batch_reads_only_fresh_requested_models_and_propagates_configuration(batch_workspace, monkeypatch):
    root = batch_workspace
    for model in ("org/high", "org/low"):
        source_file(root, model)
    write_stale_scores(root, "stale", "org/judge")
    observed = []
    def run(command, **kwargs):
        observed.append(command)
        assert command[:4] == [sys.executable, "-m", "japanese_rp_bench", "judge"]
        assert command[command.index("--base-url") + 1] == "http://offline.invalid/v1"
        assert command[command.index("--api-key-env") + 1] == "TEST_KEY"
        assert command[command.index("--dataset") + 1] == "scenarios.jsonl"
        return publish_scores(command, 4 if "org%2Fhigh" in command[command.index("--conversation-file") + 1] else 2)
    monkeypatch.setattr(batch.subprocess, "run", run)
    result = CliRunner().invoke(batch.main, ["--judge-model", "org/judge", "--base-url", "http://offline.invalid/v1", "--api-key-env", "TEST_KEY", "--dataset", "scenarios.jsonl"])
    assert result.exit_code == 0, result.output
    with next(root.glob("model_rankings_*.csv")).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["Model"] for row in rows] == ["org/high", "org/low"]
    assert [float(row["Overall Score"]) for row in rows] == [4, 2]
    assert len(observed) == 2
    assert all(Path(command[command.index("--output-dir") + 1]) != root / "scores" for command in observed)


def test_failed_evaluation_cannot_succeed_using_stale_results(batch_workspace, monkeypatch):
    source_file(batch_workspace, "new")
    write_stale_scores(batch_workspace, "new", "judge")
    monkeypatch.setattr(batch.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 42))
    result = CliRunner().invoke(batch.main, ["--judge-model", "judge"])
    assert result.exit_code != 0
    assert "failed" in result.output.lower()
    assert not list(batch_workspace.glob("model_rankings_*.csv"))


@pytest.mark.parametrize("changes", [{"status": "incomplete"}, {"evaluated_count": 0}, {"model_name": "wrong"}, {"judge_model_name": "wrong"}, {"overall_average": float("nan")}, {"category_averages": {"Consistency": 3}}])
def test_batch_rejects_incomplete_or_mismatched_artifact(batch_workspace, monkeypatch, changes):
    source_file(batch_workspace, "model")
    monkeypatch.setattr(batch.subprocess, "run", lambda command, **kwargs: publish_scores(command, **changes))
    result = CliRunner().invoke(batch.main, ["--judge-model", "judge"])
    assert result.exit_code != 0
    assert not list(batch_workspace.glob("model_rankings_*.csv"))


def test_empty_batch_finishes_with_header_only_csv(batch_workspace):
    result = CliRunner().invoke(batch.main, ["--judge-model", "judge"])
    assert result.exit_code == 0, result.output
    with next(batch_workspace.glob("model_rankings_*.csv")).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["Model", "Overall Score"]
        assert list(reader) == []


def test_csv_preserves_japanese_quotes_commas_newlines_and_column_order(batch_workspace, monkeypatch):
    names = ['組織/model,"引用"\n改行', "org/alpha"]
    for name in names:
        source_file(batch_workspace, name)
    monkeypatch.setattr(batch.subprocess, "run", lambda command, **kwargs: publish_scores(command, 3))
    result = CliRunner().invoke(batch.main, ["--judge-model", "judge"])
    assert result.exit_code == 0, result.output
    with next(batch_workspace.glob("model_rankings_*.csv")).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["Model", "Overall Score", *CATEGORIES]
        rows = list(reader)
    assert [row["Model"] for row in rows] == sorted(names)
    assert all(float(row["Overall Score"]) == 3 for row in rows)
    assert all(float(row[category]) == 3 for row in rows for category in CATEGORIES)


def test_batch_cli_does_not_import_pandas_for_csv_output(tmp_path):
    (tmp_path / "conversations").mkdir()
    bootstrap = """import builtins, runpy, sys
# Package dataset loading legitimately has pandas as a transitive dependency.
# Initialize that boundary first, then prohibit batch code importing pandas.
import japanese_rp_bench.data
real_import = builtins.__import__
def import_without_pandas(name, *args, **kwargs):
    if name == 'pandas' or name.startswith('pandas.'):
        raise ModuleNotFoundError('pandas intentionally unavailable')
    return real_import(name, *args, **kwargs)
builtins.__import__ = import_without_pandas
sys.argv = ['japanese_rp_bench', 'batch', '--judge-model', 'judge']
runpy.run_module('japanese_rp_bench', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", bootstrap], cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "model_rankings_judge.csv").read_text().strip() == "Model,Overall Score"


def test_missing_conversations_directory_is_failed_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(batch.main, ["--judge-model", "judge"])
    assert result.exit_code != 0


def test_batch_integrates_current_httpx_judge_and_artifacts(batch_workspace, monkeypatch):
    import httpx
    from japanese_rp_bench import judging
    from japanese_rp_bench.client import ChatClient
    from japanese_rp_bench.artifacts import score_paths
    source = source_file(batch_workspace, "org/model")
    record = json.loads(source.read_text())
    record["settings"] = {"id": 0, "world_setting": "学校"}
    source.write_text(json.dumps(record) + "\n")
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(evaluation(4))}}]})
    monkeypatch.setattr(judging, "create_client", lambda **kwargs: ChatClient("http://offline.invalid/v1", transport=httpx.MockTransport(respond)))
    def run(command, **kwargs):
        assert command[:4] == [sys.executable, "-m", "japanese_rp_bench", "judge"]
        result = CliRunner().invoke(judging.main, command[4:])
        assert result.exit_code == 0, result.output
        return subprocess.CompletedProcess(command, result.exit_code)
    monkeypatch.setattr(batch.subprocess, "run", run)
    result = CliRunner().invoke(batch.main, ["--judge-model", "org/judge"])
    assert result.exit_code == 0, result.output
    paths = score_paths("org/model", "org/judge", batch_workspace / "scores")
    assert json.loads(paths["scores"].read_text())["overall_average"] == 4
    assert json.loads(paths["judgements"].read_text())["evaluation"] == evaluation(4)
    assert requests[0]["model"] == "org/judge"
    assert "学校" in "\n".join(message["content"] for message in requests[0]["messages"])


def test_batch_forwards_reasoning_and_budget_controls_to_real_judge(batch_workspace, monkeypatch):
    import httpx
    from japanese_rp_bench import models, judging
    from japanese_rp_bench.client import ChatClient
    source = source_file(batch_workspace, 'target')
    row = json.loads(source.read_text()); row['settings'] = {'id': 0}
    source.write_text(json.dumps(row) + '\n')
    requests = []
    def respond(request):
        body = json.loads(request.content); requests.append(body)
        if body['max_tokens'] == 8:
            return httpx.Response(200, json={'choices': [{'finish_reason': 'length', 'message': {'content': None, 'reasoning_content': 'thinking'}}]})
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': '<think>trace</think>' + json.dumps(evaluation(4))}}]})
    monkeypatch.setattr(models, 'ChatClient', lambda **kwargs: ChatClient(**kwargs, transport=httpx.MockTransport(respond)))
    def subprocess_run(command, **kwargs):
        result = CliRunner().invoke(judging.main, command[4:])
        assert result.exit_code == 0, result.output
        return subprocess.CompletedProcess(command, result.exit_code)
    monkeypatch.setattr(batch.subprocess, 'run', subprocess_run)
    result = CliRunner().invoke(batch.main, ['--judge-model', 'judge', '--request-options', '{"max_tokens":8,"enable_thinking":true}',
        '--token-limit-ceiling', '16', '--max-token-retries', '1', '--strip-think-tags', '--timeout', '240', '--max-workers', '1'])
    assert result.exit_code == 0, result.output
    assert [r['max_tokens'] for r in requests] == [8, 16]
    assert all(r['enable_thinking'] is True for r in requests)
    saved = json.loads(score_paths('target', 'judge', batch_workspace / 'scores')['judgements'].read_text())
    assert saved['evaluation'] == evaluation(4)
    assert saved['completion']['reasoning'] == 'trace'


@pytest.mark.parametrize('options', ['[]', '{bad', '{"max_tokens":0}'])
def test_invalid_batch_request_options_do_not_publish_empty_rankings(batch_workspace, options):
    result = CliRunner().invoke(batch.main, ['--judge-model', 'judge', '--request-options', options])
    assert result.exit_code != 0
    assert not list(batch_workspace.glob('model_rankings_*.csv'))
