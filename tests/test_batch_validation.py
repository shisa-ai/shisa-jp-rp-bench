"""Reject corrupt inputs and incomplete artifacts before publishing rankings."""
import json
import subprocess

import pytest
from click.testing import CliRunner

from japanese_rp_bench import batch
from japanese_rp_bench.artifacts import score_paths
from japanese_rp_bench.utils import EVALUATION_CATEGORIES


@pytest.mark.parametrize('rows', ['', 'null\n', '{bad}\n',
    '{"id":0,"target_model_name":3}\n',
    '{"id":0}\n', '{"id":0,"target_model_name":"   "}\n',
    '{"id":0,"target_model_name":"one"}\n{"id":1}\n',
    '{"id":0,"target_model_name":"one"}\n{"id":1,"target_model_name":"two"}\n'])
def test_unusable_conversation_files_fail_before_subprocess(tmp_path, monkeypatch, rows):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / 'conversations'
    source.mkdir()
    (source / 'input.jsonl').write_text(rows)
    monkeypatch.setattr(batch.subprocess, 'run', lambda *a, **k: pytest.fail('Invalid input reached judge'))
    result = CliRunner().invoke(batch.main, ['--judge-model', 'judge'])
    assert result.exit_code != 0
    assert 'Error:' in result.output
    assert not list(tmp_path.glob('model_rankings*'))


def test_duplicate_model_runs_require_separate_batches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / 'conversations'
    source.mkdir()
    for name in ('standard', 'low'):
        (source / (name + '.jsonl')).write_text('{"id":0,"target_model_name":"same"}\n')
    monkeypatch.setattr(batch.subprocess, 'run', lambda *a, **k: pytest.fail('Duplicate model reached judge'))
    result = CliRunner().invoke(batch.main, ['--judge-model', 'judge'])
    assert result.exit_code != 0
    assert 'duplicate model identity' in result.output


def test_complete_summary_without_source_artifacts_cannot_publish(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'conversations').mkdir()
    (tmp_path / 'conversations/model.jsonl').write_text('{"id":0,"target_model_name":"model"}\n')
    def run(command, **kwargs):
        directory = command[command.index('--output-dir') + 1]
        path = score_paths('model', 'judge', directory)['scores']
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'model_name': 'model', 'judge_model_name': 'judge',
            'status': 'complete', 'sample_count': 1, 'evaluated_count': 1, 'failed_count': 0,
            'category_averages': dict.fromkeys(EVALUATION_CATEGORIES, 3), 'overall_average': 3}))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(batch.subprocess, 'run', run)
    result = CliRunner().invoke(batch.main, ['--judge-model', 'judge'])
    assert result.exit_code != 0
    assert 'artifacts missing' in result.output
    assert not list(tmp_path.glob('model_rankings*'))
    assert not score_paths('model', 'judge', tmp_path / 'scores')['scores'].exists()


def test_summary_overall_must_match_categories(tmp_path):
    path = tmp_path / 'scores.json'
    path.write_text(json.dumps({'model_name': 'model', 'judge_model_name': 'judge',
        'status': 'complete', 'sample_count': 1, 'evaluated_count': 1, 'failed_count': 0,
        'category_averages': dict.fromkeys(EVALUATION_CATEGORIES, 3), 'overall_average': 4}))
    with pytest.raises(ValueError, match='differs'):
        batch.load_score_summary(path, 'model', 'judge', 1)


