"""The installed/module CLI owns the supported workflow."""
import json
from pathlib import Path
import subprocess
import sys

import httpx
import pytest
import yaml
from click.testing import CliRunner


def test_module_entrypoint_exposes_supported_commands():
    result = subprocess.run([sys.executable, '-m', 'japanese_rp_bench', '--help'], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    for command in ('generate', 'judge', 'batch', 'scores'):
        assert command in result.stdout


@pytest.mark.parametrize('body', ['[not: valid', '[]'])
def test_invalid_generation_yaml_is_a_clean_cli_failure(tmp_path, body):
    from japanese_rp_bench.cli import main
    config = tmp_path / 'config.yaml'
    config.write_text(body)
    result = CliRunner().invoke(main, ['generate', '--config', str(config)])
    assert result.exit_code != 0
    assert 'Error:' in result.output
    assert not (tmp_path / 'conversations').exists()


def test_generate_then_judge_through_single_cli(tmp_path, monkeypatch):
    from japanese_rp_bench import models
    from japanese_rp_bench.cli import main
    from japanese_rp_bench.utils import EVALUATION_CATEGORIES
    scenario = json.loads((Path(__file__).resolve().parents[1] / 'configs/smoke_scenario.json').read_text())
    dataset = tmp_path / 'scenarios.json'
    dataset.write_text(json.dumps(scenario))
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'dataset_repo': str(dataset), 'target_model_name': 'target',
        'user_model_name': 'partner', 'target_api_key_env': 'CLI_TEST_KEY', 'user_api_key_env': 'CLI_TEST_KEY',
        'target_request_options': {'max_tokens': 64}, 'user_request_options': {'max_tokens': 64},
        'max_turns': 2, 'max_workers': 1, 'output_dir': str(tmp_path / 'run')}))
    monkeypatch.setenv('CLI_TEST_KEY', 'dummy-key')
    requests = []
    def handle(request):
        assert request.headers['authorization'] == 'Bearer dummy-key'
        payload = json.loads(request.content)
        requests.append(payload)
        content = json.dumps({**dict.fromkeys(EVALUATION_CATEGORIES, 4), 'Evaluation Reason': '自然な対話'}) if payload['model'] == 'judge' else 'こんにちは'
        return httpx.Response(200, json={'choices': [{'message': {'content': content}}]})
    client_class = models.ChatClient
    monkeypatch.setattr(models, 'ChatClient', lambda **kwargs: client_class(**kwargs, transport=httpx.MockTransport(handle)))
    result = CliRunner().invoke(main, ['generate', '--config', str(config), '--max-samples', '1', '--max-workers', '2'])
    assert result.exit_code == 0, result.output
    conversation = Path(result.output.strip())
    records = [json.loads(line) for line in conversation.read_text().splitlines()]
    assert len(records) == 1 and len(records[0]['conversation_history']) == 4
    assert not (tmp_path / 'run/evaluations').exists()
    assert all(request['max_tokens'] == 64 for request in requests)
    result = CliRunner().invoke(main, ['judge', '--conversation-file', str(conversation),
        '--judge-model', 'judge', '--api-key-env', 'CLI_TEST_KEY', '--output-dir', str(tmp_path / 'scores')])
    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / 'scores/target/judge/scores.json').read_text())
    assert summary['status'] == 'complete' and summary['overall_average'] == 4
    assert summary['sample_count'] == 1
    assert requests[-1]['model'] == 'judge'
