"""Reasoning is diagnostic data; only complete final answers enter the benchmark."""
import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from japanese_rp_bench.client import ChatClient, ChatError

MESSAGES = [{'role': 'user', 'content': '答えて'}]


def reply(content='回答', *, reasoning=None, finish='stop', **extra):
    return {'choices': [{'finish_reason': finish, 'message': {'content': content, 'reasoning_content': reasoning, **extra}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30,
                      'completion_tokens_details': {'reasoning_tokens': 17}}}


def make_client(handler, **kwargs):
    return ChatClient('https://example.test/v1', transport=httpx.MockTransport(handler), **kwargs)


@pytest.mark.parametrize('retry_first', [False, True])
@pytest.mark.parametrize('malformation', ['content', 'finish', 'choices'])
def test_malformed_response_retains_known_usage_in_failure_attempts(retry_first, malformation):
    malformed = reply(reasoning='private-trace')
    malformed['usage']['total_tokens'] = 50
    malformed['usage']['secret'] = 'must-not-retain'
    if malformation == 'content':
        malformed['choices'][0]['message']['content'] = 42
    elif malformation == 'finish':
        malformed['choices'][0]['finish_reason'] = 42
    else:
        malformed['choices'] = []
    bodies = iter([reply(finish='length'), malformed] if retry_first else [malformed])
    with make_client(lambda _: httpx.Response(200, json=next(bodies)), token_limit_ceiling=16) as client:
        with pytest.raises(ChatError, match='malformed') as exc:
            client.complete_with_details('model', MESSAGES, max_tokens=8)
    completion = exc.value.completion
    assert completion is not None
    assert completion.usage['total_tokens'] == 50
    assert [a['usage']['total_tokens'] for a in completion.attempts] == ([30, 50] if retry_first else [50])
    assert [a['token_limit'] for a in completion.attempts] == ([8, 16] if retry_first else [8])
    assert 'must-not-retain' not in json.dumps(completion.metadata())
    assert 'private-trace' not in str(exc.value)


def test_truncated_final_text_is_never_accepted_as_a_complete_answer():
    with make_client(lambda _: httpx.Response(200, json=reply('途中', finish='length'))) as client:
        with pytest.raises(ChatError, match='token limit'):
            client.complete('model', MESSAGES, max_tokens=20)


def test_reasoning_and_usage_are_separate_from_final_content():
    with make_client(lambda _: httpx.Response(200, json=reply(reasoning='検討内容'))) as client:
        result = client.complete_with_details('model', MESSAGES)
    assert result.content == '回答'
    assert result.reasoning == '検討内容'
    assert result.finish_reason == 'stop'
    assert result.usage['completion_tokens_details']['reasoning_tokens'] == 17


@pytest.mark.parametrize('field', ['reasoning_content', 'reasoning'])
def test_reasoning_only_response_is_diagnosed_without_echoing_the_trace(field):
    body = reply(None)
    body['choices'][0]['message'][field] = 'private-trace'
    with make_client(lambda _: httpx.Response(200, json=body)) as client:
        with pytest.raises(ChatError, match='reasoning without a final answer') as exc:
            client.complete('model', MESSAGES)
    assert 'private-trace' not in str(exc.value)
    assert exc.value.completion.reasoning == 'private-trace'


def test_content_blocks_do_not_mix_thinking_into_answer():
    body = reply([{'type': 'thinking', 'thinking': 'one'}, {'type': 'reasoning', 'text': 'two'},
                  {'type': 'text', 'text': '回答'}, {'type': 'image', 'text': 'not text'}])
    with make_client(lambda _: httpx.Response(200, json=body)) as client:
        result = client.complete_with_details('model', MESSAGES)
    assert result.content == '回答'
    assert result.reasoning == 'one\ntwo'


@pytest.mark.parametrize('enabled,expected', [(False, '<think>考える</think>回答'), (True, '回答')])
def test_inline_thinking_extraction_is_explicit(enabled, expected):
    with make_client(lambda _: httpx.Response(200, json=reply('<think>考える</think>回答')),
                     strip_think_tags=enabled) as client:
        result = client.complete_with_details('model', MESSAGES)
    assert result.content == expected
    assert result.reasoning == ('考える' if enabled else None)


def test_unclosed_thinking_block_is_not_a_final_answer():
    with make_client(lambda _: httpx.Response(200, json=reply('<think>考え続ける')), strip_think_tags=True) as client:
        with pytest.raises(ChatError, match='reasoning without a final answer'):
            client.complete('model', MESSAGES)


@pytest.mark.parametrize('field', ['max_tokens', 'max_completion_tokens'])
def test_budget_retry_resends_same_messages_and_preserves_reasoning_controls(field):
    requests = []
    options = {field: 8, 'reasoning_effort': 'high', 'enable_thinking': True, 'temperature': None}
    def respond(request):
        body = json.loads(request.content); requests.append(body)
        return httpx.Response(200, json=reply(None, reasoning='still thinking', finish='length')
                              if len(requests) < 3 else reply('最終回答', reasoning='finished'))
    with make_client(respond, default_options=options, token_limit_ceiling=24, max_token_retries=2) as client:
        result = client.complete_with_details('model', MESSAGES)
        assert client.default_options == options
    assert result.content == '最終回答'
    assert [r[field] for r in requests] == [8, 16, 24]
    assert all(r['messages'] == MESSAGES and r['reasoning_effort'] == 'high' and r['enable_thinking'] is True for r in requests)
    assert all('temperature' not in r and 'token_limit_ceiling' not in r for r in requests)
    assert [a['token_limit'] for a in result.attempts] == [8, 16, 24]
    assert sum(a['usage']['completion_tokens'] for a in result.attempts) == 60


@pytest.mark.parametrize('ceiling,retries,expected', [(None, 2, [8]), (16, 9, [8, 16]), (64, 1, [8, 16]), (64, 0, [8])])
def test_budget_growth_requires_opt_in_and_obeys_both_limits(ceiling, retries, expected):
    seen = []
    def respond(request):
        seen.append(json.loads(request.content)['max_tokens'])
        return httpx.Response(200, json=reply(None, reasoning='private-trace', finish='length'))
    with make_client(respond, token_limit_ceiling=ceiling, max_token_retries=retries) as client:
        with pytest.raises(ChatError, match='token limit') as exc:
            client.complete('model', MESSAGES, max_tokens=8)
    assert seen == expected
    assert len(exc.value.completion.attempts) == len(expected)
    assert 'private-trace' not in str(exc.value)


@pytest.mark.parametrize('kwargs', [{'token_limit_ceiling': 0}, {'token_limit_ceiling': True},
    {'max_token_retries': -1}, {'max_token_retries': 1.5}, {'strip_think_tags': 'yes'}])
def test_invalid_reasoning_policy_fails_before_transport(kwargs):
    with pytest.raises(ValueError):
        make_client(lambda _: pytest.fail('invalid policy reached API'), **kwargs)


@pytest.mark.parametrize('options', [{}, {'max_tokens': 65}])
def test_retry_ceiling_requires_an_initial_budget_within_the_ceiling(options):
    with make_client(lambda _: pytest.fail('invalid budget reached API'), token_limit_ceiling=64) as client:
        with pytest.raises(ValueError):
            client.complete('model', MESSAGES, **options)


def test_shared_client_does_not_leak_budget_growth_between_threads():
    def respond(request):
        body = json.loads(request.content)
        limited = body['model'] == 'needs-more' and body['max_tokens'] == 8
        return httpx.Response(200, json=reply(None, reasoning='more', finish='length') if limited else reply(body['model']))
    with make_client(respond, default_options={'max_tokens': 8}, token_limit_ceiling=16) as client:
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda model: client.complete_with_details(model, MESSAGES), ['needs-more', 'small']))
    assert [r.content for r in results] == ['needs-more', 'small']
    assert [[a['token_limit'] for a in r.attempts] for r in results] == [[8, 16], [8]]


@pytest.mark.parametrize('finish', ['tool_calls', 'content_filter'])
def test_non_answer_finish_reasons_are_not_scored(finish):
    with make_client(lambda _: httpx.Response(200, json=reply('partial', finish=finish))) as client:
        with pytest.raises(ChatError):
            client.complete('model', MESSAGES)


def test_generation_keeps_thinking_out_of_history_and_saves_per_turn_usage(tmp_path, monkeypatch):
    from pathlib import Path
    from japanese_rp_bench import models, run
    scenario = Path(__file__).resolve().parents[1] / 'configs/smoke_scenario.json'
    requests = []
    def respond(request):
        body = json.loads(request.content); requests.append(body)
        limited = body['max_tokens'] == 8
        return httpx.Response(200, json=reply(None, reasoning='private-trace', finish='length')
                              if limited else reply('会話', reasoning='private-trace'))
    monkeypatch.setattr(models, 'ChatClient', lambda **kwargs: ChatClient(**kwargs, transport=httpx.MockTransport(respond)))
    config = {'dataset_repo': str(scenario), 'target_model_name': 'target', 'user_model_name': 'partner',
              'max_turns': 2, 'max_workers': 2, 'output_dir': str(tmp_path)}
    for role in ('target', 'user'):
        config.update({f'{role}_request_options': {'max_tokens': 8, 'enable_thinking': True},
                       f'{role}_token_limit_ceiling': 16})
    path = run.generate_conversations(config)
    record = json.loads(path.read_text())
    assert record['conversation_history'][1:] == ['会話', '会話', '会話']
    assert all('private-trace' not in json.dumps(r['messages']) for r in requests)
    assert [x['role'] for x in record['generation_metadata']] == ['target', 'user', 'target']
    assert [x['history_index'] for x in record['generation_metadata']] == [1, 2, 3]
    assert all(len(x['completion']['attempts']) == 2 for x in record['generation_metadata'])
    assert run.conversation_output_path(config) != run.conversation_output_path(dict(config, target_token_limit_ceiling=32))


def test_generation_failure_saves_reasoning_exhaustion_diagnostics(tmp_path, monkeypatch):
    from pathlib import Path
    from japanese_rp_bench import models, run
    config = {'dataset_repo': str(Path(__file__).resolve().parents[1] / 'configs/smoke_scenario.json'),
              'target_model_name': 'target', 'user_model_name': 'partner', 'max_turns': 1, 'output_dir': str(tmp_path)}
    monkeypatch.setattr(models, 'ChatClient', lambda **kwargs: ChatClient(**kwargs, transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=reply(None, reasoning='still thinking', finish='length')))))
    with pytest.raises(ChatError):
        run.generate_conversations(config)
    failure = json.loads(next((tmp_path / 'generation_failures').glob('*.jsonl')).read_text())
    assert failure['completion']['finish_reason'] == 'length'
    assert failure['completion']['usage']['completion_tokens_details']['reasoning_tokens'] == 17
    assert 'token limit' in failure['error_detail']


def test_judge_cli_retries_thinking_exhaustion_and_records_metadata(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from japanese_rp_bench import models
    from japanese_rp_bench.cli import main
    from japanese_rp_bench.utils import EVALUATION_CATEGORIES
    source = tmp_path / 'conversations.jsonl'
    source.write_text(json.dumps({'id': 0, 'target_model_name': 'target', 'settings': {'id': 0},
                                 'conversation_history': ['質問', '回答']}) + '\n')
    evaluation = {**dict.fromkeys(EVALUATION_CATEGORIES, 4), 'Evaluation Reason': '理由'}
    requests = []
    def respond(request):
        body = json.loads(request.content); requests.append(body)
        return httpx.Response(200, json=reply(None, reasoning='考える', finish='length') if len(requests) == 1
                              else reply(json.dumps(evaluation), reasoning='採点の検討'))
    monkeypatch.setattr(models, 'ChatClient', lambda **kwargs: ChatClient(**kwargs, transport=httpx.MockTransport(respond)))
    result = CliRunner().invoke(main, ['judge', '--conversation-file', str(source), '--judge-model', 'judge',
        '--output-dir', str(tmp_path / 'scores'), '--request-options', '{"max_completion_tokens":8,"reasoning_effort":"high"}',
        '--token-limit-ceiling', '16', '--max-token-retries', '1', '--strip-think-tags'])
    assert result.exit_code == 0, result.output
    saved = json.loads((tmp_path / 'scores/target/judge/judgements.jsonl').read_text())
    assert saved['evaluation'] == evaluation
    assert saved['raw_evaluation'] == json.dumps(evaluation)
    assert saved['completion']['reasoning'] == '採点の検討'
    assert [x['token_limit'] for x in saved['completion']['attempts']] == [8, 16]


@pytest.mark.parametrize('key,value', [('token_limit_ceiling', False), ('max_token_retries', -1),
                                     ('strip_think_tags', 'false'), ('token_limit_ceiling', 1)])
def test_invalid_role_budget_policy_rejected_before_dataset_loading(monkeypatch, key, value):
    from japanese_rp_bench import run
    monkeypatch.setattr(run, 'load_dataset_wrapper', lambda *a, **k: pytest.fail('invalid configuration loaded data'))
    with pytest.raises(ValueError):
        run.generate_conversations({'dataset_repo': 'data', 'target_model_name': 't', 'user_model_name': 'u',
                                    f'user_{key}': value})


def test_budget_retry_transport_failure_retains_prior_usage(monkeypatch):
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(200, json=reply(None, reasoning='trace', finish='length')) if len(calls) == 1 else httpx.Response(401, text='secret-sentinel')
    with make_client(respond, token_limit_ceiling=16) as client:
        with pytest.raises(ChatError) as exc:
            client.complete('model', MESSAGES, max_tokens=8)
    assert exc.value.status_code == 401
    assert 'secret-sentinel' not in str(exc.value)
    assert exc.value.completion.attempts[0]['usage']['completion_tokens'] == 20


def test_invalid_judge_budget_does_not_overwrite_existing_artifacts(tmp_path):
    from japanese_rp_bench import judging
    from japanese_rp_bench.artifacts import score_paths
    paths = score_paths('target', 'judge', tmp_path)
    paths['scores'].parent.mkdir(parents=True)
    paths['scores'].write_text('preserve me')
    records = [{'id': '0', 'settings': {'id': 0}, 'formatted_data': 'conversation', 'answer': {'target_model_name': 'target'}}]
    with make_client(lambda _: pytest.fail('bad budget reached API'), token_limit_ceiling=8) as client:
        with pytest.raises(ValueError):
            judging.judge_conversations(records, client=client, judge_model='judge', output_dir=tmp_path)
    assert paths['scores'].read_text() == 'preserve me'


@pytest.mark.parametrize('body', [reply([], reasoning='only trace'), reply([None, {'type': 'thinking', 'thinking': ' ' }]),
                                 {'choices': [{'message': {'content': 'x'}, 'finish_reason': 42}]}])
def test_invalid_or_reasoning_only_blocks_fail(body):
    with make_client(lambda _: httpx.Response(200, json=body)) as client:
        with pytest.raises(ChatError):
            client.complete('model', MESSAGES)


def test_usage_metadata_is_numeric_and_does_not_copy_unknown_upstream_fields():
    body = reply('text');body['usage'] = {'prompt_tokens': True, 'completion_tokens': -1, 'total_tokens': 30,
        'secret': 'secret-sentinel', 'prompt_tokens_details': {'cached_tokens': 7, 'secret': 'secret-sentinel'},
        'completion_tokens_details': {'reasoning_tokens': '20', 'audio_tokens': 2}}
    with make_client(lambda _: httpx.Response(200, json=body)) as client:
        result = client.complete_with_details('model', MESSAGES)
    assert result.usage == {'total_tokens': 30, 'prompt_tokens_details': {'cached_tokens': 7}, 'completion_tokens_details': {'audio_tokens': 2}}
    assert 'secret-sentinel' not in json.dumps(result.metadata())


def test_mid_sentence_think_tags_are_literal_even_when_extraction_enabled():
    content = '台詞: <think>引用</think>'
    with make_client(lambda _: httpx.Response(200, json=reply(content)), strip_think_tags=True) as client:
        assert client.complete('model', MESSAGES) == content


def test_judging_failure_retains_partial_answer_and_reasoning_without_scoring_it(tmp_path):
    from japanese_rp_bench import judging
    from japanese_rp_bench.artifacts import score_paths
    records = [{'id': '0', 'settings': {'id': 0}, 'formatted_data': 'conversation', 'answer': {'target_model_name': 'target'}}]
    with make_client(lambda _: httpx.Response(200, json=reply('partial JSON', reasoning='trace', finish='length'))) as client:
        summary = judging.judge_conversations(records, client=client, judge_model='judge', output_dir=tmp_path)
    assert summary['status'] == 'incomplete' and summary['evaluated_count'] == 0
    saved = json.loads(score_paths('target', 'judge', tmp_path)['judgements'].read_text())
    assert saved['evaluation'] is None
    assert saved['raw_evaluation'] == 'partial JSON'
    assert saved['completion']['reasoning'] == 'trace'
    assert saved['completion']['finish_reason'] == 'length'


@pytest.mark.parametrize('terminal_status', [200, 401])
def test_later_turn_failure_preserves_completed_usage_and_failing_role(tmp_path, monkeypatch, terminal_status):
    from pathlib import Path
    from japanese_rp_bench import models, run
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, json=reply('回答', reasoning='first turn reasoning'))
        return httpx.Response(200, json=reply(None, reasoning='unfinished', finish='length')) if terminal_status == 200 else httpx.Response(401)
    monkeypatch.setattr(models, 'ChatClient', lambda **kwargs: ChatClient(**kwargs, transport=httpx.MockTransport(respond)))
    config = {'dataset_repo': str(Path(__file__).resolve().parents[1] / 'configs/smoke_scenario.json'),
              'target_model_name': 'target', 'user_model_name': 'partner', 'max_turns': 2, 'output_dir': str(tmp_path)}
    with pytest.raises(ChatError):
        run.generate_conversations(config)
    failure = json.loads(next((tmp_path / 'generation_failures').glob('*.jsonl')).read_text())
    assert failure['role'] == 'user' and failure['history_index'] == 2
    assert failure['conversation_history'][-1] == '回答'
    assert failure['generation_metadata'][0]['completion']['reasoning'] == 'first turn reasoning'
    assert failure['generation_metadata'][0]['completion']['usage']['total_tokens'] == 30
    assert not next((tmp_path / 'conversations').glob('*.jsonl')).read_text()
