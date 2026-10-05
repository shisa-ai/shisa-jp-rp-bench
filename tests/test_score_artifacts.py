"""Package score commands maintain absolute artifacts without inference."""
import json
import subprocess
import sys

import pytest
from click.testing import CliRunner
from japanese_rp_bench import scores

from japanese_rp_bench.artifacts import score_paths
from japanese_rp_bench.utils import EVALUATION_CATEGORIES


def run_cli(operation, tmp_path):
    return subprocess.run([sys.executable, "-m", "japanese_rp_bench", "scores", operation,
        "--scores-dir", str(tmp_path / "scores"), "--results-dir", str(tmp_path / "export")],
        cwd=tmp_path, capture_output=True, text=True, timeout=10)


@pytest.fixture(params=["cli", "in_process"])
def artifact_runner(request):
    if request.param == "cli":
        return run_cli
    def invoke(operation, tmp_path):
        result = CliRunner().invoke(scores.main, [operation, "--scores-dir", str(tmp_path / "scores"), "--results-dir", str(tmp_path / "export")])
        return subprocess.CompletedProcess(operation, result.exit_code, result.output, result.output)
    return invoke


def test_copy_current_scores_keeps_model_and_judge_namespaces(tmp_path, artifact_runner):
    for judge, value in (("judge/one", 3), ("judge/two", 4)):
        paths = score_paths("org/model", judge, tmp_path / "scores")
        paths["scores"].parent.mkdir(parents=True)
        paths["scores"].write_text(json.dumps({"model_name": "org/model", "judge_model_name": judge,
            "status": "complete", "overall_average": value, "category_averages": dict.fromkeys(EVALUATION_CATEGORIES, value)}))
    result = artifact_runner("export", tmp_path)
    assert result.returncode == 0, result.stderr
    files = sorted((tmp_path / "export").glob("*/*/score-shisa-jp-rp-bench.jsonl"))
    assert len(files) == 2
    assert {json.loads(path.read_text())["overall_average"] for path in files} == {3, 4}


def test_rerun_reaggregates_current_judgements_without_inference(tmp_path, artifact_runner):
    paths = score_paths("org/model", "org/judge", tmp_path / "scores")
    paths["judgements"].parent.mkdir(parents=True)
    rows = [{"id": i, "llm": "org/model", "judge_model_name": "org/judge",
             "evaluation": {**dict.fromkeys(EVALUATION_CATEGORIES, value), "Evaluation Reason": "理由"}} for i, value in enumerate((2, 4))]
    paths["judgements"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    paths["answers"].write_text('{"id":0}\n{"id":1}\n')
    result = artifact_runner("reaggregate", tmp_path)
    assert result.returncode == 0, result.stderr
    summary = json.loads(paths["scores"].read_text())
    assert summary["overall_average"] == 3
    assert summary["evaluated_count"] == 2
    assert summary["status"] == "complete"


def test_empty_exports_create_no_literal_glob_directories(tmp_path, artifact_runner):
    (tmp_path / "scores").mkdir()
    result = artifact_runner("export", tmp_path)
    assert result.returncode == 0, result.stderr
    assert not list((tmp_path / "export").rglob("*"))


def test_reaggregation_keeps_invalid_results_incomplete(tmp_path, artifact_runner):
    paths = score_paths("model", "judge", tmp_path / "scores")
    paths["judgements"].parent.mkdir(parents=True)
    original = json.dumps({"id": 0, "llm": "model", "judge_model_name": "judge", "evaluation": {"Consistency": True}}) + "\n"
    paths["judgements"].write_text(original)
    paths["answers"].write_text('{"id":0}\n')
    result = artifact_runner("reaggregate", tmp_path)
    assert result.returncode != 0
    summary = json.loads(paths["scores"].read_text())
    assert summary["status"] == "incomplete"
    assert summary["overall_average"] is None
    assert paths["judgements"].read_text() == original


def test_interrupted_judgements_do_not_become_complete(tmp_path, artifact_runner):
    paths = score_paths("model", "judge", tmp_path / "scores")
    paths["judgements"].parent.mkdir(parents=True)
    paths["answers"].write_text('{"id":0}\n{"id":1}\n')
    paths["scores"].write_text(json.dumps({"model_name": "model", "judge_model_name": "judge", "sample_count": 2, "status": "incomplete"}))
    paths["judgements"].write_text(json.dumps({"id": 0, "llm": "model", "judge_model_name": "judge", "evaluation": {**dict.fromkeys(EVALUATION_CATEGORIES, 4), "Evaluation Reason": "理由"}}) + "\n")
    result = artifact_runner("reaggregate", tmp_path)
    assert result.returncode != 0
    summary = json.loads(paths["scores"].read_text())
    assert summary["sample_count"] == 2
    assert summary["evaluated_count"] == 1
    assert summary["failed_count"] == 1
    assert summary["status"] == "incomplete"


@pytest.mark.parametrize("kind", ["duplicate", "unknown", "mixed", "invalid_count", "empty", "malformed"])
def test_reaggregation_rejects_invalid_identity_or_counts(tmp_path, kind):
    paths = score_paths("model", "judge", tmp_path / "scores")
    paths["judgements"].parent.mkdir(parents=True)
    record = {"id": 0, "llm": "model", "judge_model_name": "judge", "evaluation": {**dict.fromkeys(EVALUATION_CATEGORIES, 4), "Evaluation Reason": "理由"}}
    records = [record]
    paths["answers"].write_text('{"id":0}\n{"id":1}\n')
    if kind == "duplicate":
        records += [record]
    if kind == "mixed":
        records += [{**record, "id": 1, "llm": "other"}]
    if kind == "unknown":
        paths["answers"].write_text('{"id":99}\n')
    if kind == "invalid_count":
        paths["scores"].write_text('{"sample_count":0}')
    text = "" if kind == "empty" else "not-json" if kind == "malformed" else "".join(json.dumps(row) + "\n" for row in records)
    paths["judgements"].write_text(text)
    with pytest.raises(ValueError):
        scores.reaggregate_scores(tmp_path / "scores")


def test_artifact_cli_missing_directory_reports_error(tmp_path):
    result = CliRunner().invoke(scores.main, ["export", "--scores-dir", str(tmp_path / "missing")])
    assert result.exit_code != 0
    assert "directory not found" in result.output


def test_export_skips_incomplete_absolute_scores(tmp_path):
    paths = score_paths("model", "judge", tmp_path / "scores")
    paths["scores"].parent.mkdir(parents=True)
    paths["scores"].write_text(json.dumps({"model_name": "model", "judge_model_name": "judge", "status": "incomplete", "overall_average": 4}))
    assert scores.export_scores(tmp_path / "scores", tmp_path / "export") == 0
    assert not (tmp_path / "export").exists()


def test_reaggregation_requires_explicit_judge_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "judge")
    paths = score_paths("model", "judge", tmp_path)
    paths["judgements"].parent.mkdir(parents=True)
    record = {"id": 0, "llm": "model", "evaluation": {**dict.fromkeys(EVALUATION_CATEGORIES, 3), "Evaluation Reason": "理由"}}
    paths["judgements"].write_text(json.dumps(record) + "\n")
    paths["answers"].write_text('{"id":0}\n')
    with pytest.raises(ValueError, match="judge_model_name"):
        scores.reaggregate_scores(tmp_path)
    assert not paths["scores"].exists()


def test_reaggregation_requires_source_answers_to_establish_completeness(tmp_path):
    paths = score_paths("model", "judge", tmp_path)
    paths["judgements"].parent.mkdir(parents=True)
    record = {"id": 0, "llm": "model", "judge_model_name": "judge", "evaluation": {**dict.fromkeys(EVALUATION_CATEGORIES, 3), "Evaluation Reason": "理由"}}
    paths["judgements"].write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="answers"):
        scores.reaggregate_scores(tmp_path)
    assert not paths["scores"].exists()


@pytest.mark.parametrize("operation", ["export", "reaggregate"])
def test_artifact_metadata_must_match_namespace(tmp_path, operation):
    paths = score_paths("model", "judge", tmp_path)
    paths["scores"].parent.mkdir(parents=True)
    if operation == "export":
        paths["scores"].write_text(json.dumps({"model_name": "other", "judge_model_name": "judge", "status": "complete", "overall_average": 4}))
        with pytest.raises(ValueError, match="namespace"):
            scores.export_scores(tmp_path, tmp_path / "export")
    else:
        paths["judgements"].write_text('{"id":0,"llm":"other","judge_model_name":"judge","evaluation":null}\n')
        with pytest.raises(ValueError, match="namespace"):
            scores.reaggregate_scores(tmp_path)


