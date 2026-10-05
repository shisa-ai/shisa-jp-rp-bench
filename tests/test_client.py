"""Real HTTPX requests through an in-memory transport, never live inference."""
import json

import httpx
import pytest

from japanese_rp_bench.client import ChatClient, ChatError


def response(text='返答', **message):
    return {'choices': [{'message': {'content': text, **message}}]}


def client_for(handler, **kwargs):
    return ChatClient('https://example.test/openai/v1/', transport=httpx.MockTransport(handler), **kwargs)


def test_authenticated_request_preserves_messages_and_options():
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=response(' 答え '))
    messages = [{'role': 'system', 'content': '役割'}, {'role': 'user', 'content': 'こんにちは'}]
    with client_for(handle, api_key='private-test-key', default_options={'temperature': 0.2}) as client:
        assert client.complete('model', messages, max_tokens=64) == '答え'
    req = requests[0]
    assert req.url == 'https://example.test/openai/v1/chat/completions'
    assert req.headers['authorization'] == 'Bearer private-test-key'
    assert json.loads(req.content) == {'model': 'model', 'messages': messages, 'stream': False, 'temperature': 0.2, 'max_tokens': 64}
    assert messages == [{'role': 'system', 'content': '役割'}, {'role': 'user', 'content': 'こんにちは'}]
    assert 'private-test-key' not in repr(client)


def test_keyless_local_endpoint_omits_authorization():
    def handle(request):
        assert 'authorization' not in request.headers
        return httpx.Response(200, json=response())
    with client_for(handle) as client:
        assert client.complete('model', [{'role': 'user', 'content': 'hello'}]) == '返答'


@pytest.mark.parametrize('status', [408, 429, 500, 503])
def test_transient_status_retries_then_succeeds(monkeypatch, status):
    calls, sleeps = [], []
    monkeypatch.setattr('japanese_rp_bench.client.time.sleep', sleeps.append)
    def handle(request):
        calls.append(request)
        return httpx.Response(status, headers={'Retry-After': '0.1'}, json={'error': 'retry'}) if len(calls) < 3 else httpx.Response(200, json=response())
    with client_for(handle, max_retries=2) as client:
        assert client.complete('model', [{'role': 'user', 'content': 'hi'}]) == '返答'
    assert len(calls) == 3
    assert sleeps == [0.1, 0.1]


def test_retry_budget_is_finite_and_error_does_not_echo_key(monkeypatch):
    monkeypatch.setattr('japanese_rp_bench.client.time.sleep', lambda _: None)
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(503, text='secret-sentinel echoed by upstream')
    with client_for(handle, api_key='secret-sentinel', max_retries=2) as client:
        with pytest.raises(ChatError, match='503') as exc:
            client.complete('model', [{'role': 'user', 'content': 'hi'}])
    assert len(calls) == 3
    assert 'secret-sentinel' not in str(exc.value)
    assert exc.value.status_code == 503


@pytest.mark.parametrize('status', [400, 401, 403, 404, 302])
def test_permanent_errors_are_not_retried_or_redirected(status):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(status, headers={'Location': 'https://other.invalid'}, text='private upstream body')
    with client_for(handle) as client:
        with pytest.raises(ChatError, match=str(status)):
            client.complete('model', [{'role': 'user', 'content': 'hi'}])
    assert len(calls) == 1


@pytest.mark.parametrize('error_type', [httpx.ConnectError, httpx.ReadTimeout])
def test_transport_failures_retry_and_are_sanitized(monkeypatch, error_type):
    monkeypatch.setattr('japanese_rp_bench.client.time.sleep', lambda _: None)
    calls = []
    def handle(request):
        calls.append(request)
        raise error_type('secret-sentinel', request=request)
    with client_for(handle, max_retries=1) as client:
        with pytest.raises(ChatError) as exc:
            client.complete('model', [{'role': 'user', 'content': 'hi'}])
    assert len(calls) == 2
    assert 'secret-sentinel' not in str(exc.value)
    assert exc.value.__cause__ is None


@pytest.mark.parametrize('body', [{}, {'choices': []}, {'choices': [None]}, {'choices': [{'message': None}]}, response(None), response(''), response(3), response([{'image': 'x'}])])
def test_malformed_or_empty_completions_fail_explicitly(body):
    with client_for(lambda req: httpx.Response(200, json=body)) as client:
        with pytest.raises(ChatError):
            client.complete('model', [{'role': 'user', 'content': 'hi'}])


def test_invalid_json_response_is_sanitized():
    with client_for(lambda req: httpx.Response(200, text='secret-sentinel')) as client:
        with pytest.raises(ChatError) as exc:
            client.complete('model', [{'role': 'user', 'content': 'hi'}])
    assert 'secret-sentinel' not in str(exc.value)


@pytest.mark.parametrize(('content','refusal','expected'), [([{'type':'text','text':'一'}, {'type':'text','text':'二'}],None,'一二'), (None,'拒否','拒否')])
def test_text_blocks_and_refusals_are_normalized(content, refusal, expected):
    with client_for(lambda req: httpx.Response(200, json=response(content, refusal=refusal))) as client:
        assert client.complete('model', [{'role':'user','content':'hi'}]) == expected


@pytest.mark.parametrize('url', ['', 'ftp://bad', 'https://user:password@example.test/v1', 'https://example.test/v1?api_key=bad', 'https://example.test/v1#fragment'])
def test_invalid_base_url_rejected(url):
    with pytest.raises(ValueError):
        ChatClient(url)


@pytest.mark.parametrize('kwargs', [{'timeout': 0}, {'timeout': float('nan')}, {'timeout': float('inf')}, {'max_retries': -1}, {'max_retries': True}, {'max_retries': 1.5}])
def test_invalid_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        ChatClient('https://example.test/v1', **kwargs)


@pytest.mark.parametrize('name', ['model', 'messages', 'stream', 'api_key', 'headers', 'base_url'])
def test_default_options_cannot_override_routing_or_credentials(name):
    with pytest.raises(ValueError):
        ChatClient('https://example.test/v1', default_options={name: 'invalid'})


def test_timeout_is_attached_to_real_request():
    def handle(request):
        assert request.extensions['timeout']['read'] == 17
        return httpx.Response(200, json=response())
    with client_for(handle, timeout=17) as client:
        client.complete('model', [{'role':'user','content':'hi'}])


@pytest.mark.parametrize(('model','messages'), [('', [{'role':'user','content':'hi'}]), ('model', []), ('model', [{'role':'bogus','content':'hi'}]), ('model', [{'role':'user','content':None}])])
def test_invalid_message_inputs_fail_before_request(model, messages):
    with client_for(lambda req: pytest.fail('Invalid input reached transport')) as client:
        with pytest.raises(ValueError):
            client.complete(model, messages)


@pytest.mark.parametrize('options', [[], {'temperature': float('nan')}, {'extra': object()},
    {'max_tokens': 0}, {'max_completion_tokens': True},
    {'max_tokens': 20, 'max_completion_tokens': 20}])
def test_invalid_options_fail_before_request(options):
    with pytest.raises(ValueError):
        client_for(lambda req: pytest.fail('Invalid options reached transport'), default_options=options)


@pytest.mark.parametrize('key', [123, 'key\nsecret', 'key\rsecret'])
def test_invalid_api_key_rejected(key):
    with pytest.raises(ValueError, match='single-line'):
        ChatClient('https://example.test/v1', api_key=key)


@pytest.mark.parametrize('url', [None, 'https://example.test:bad/v1'])
def test_malformed_url_is_validation_error(url):
    with pytest.raises(ValueError, match='Invalid API base URL'):
        ChatClient(url)


@pytest.mark.parametrize(('header', 'expected'), [('NaN', 1), ('-1', 1), ('bad', 1), ('9999', 10)])
def test_retry_after_invalid_values_fall_back_and_large_values_are_bounded(monkeypatch, header, expected):
    sleeps, calls = [], []
    monkeypatch.setattr('japanese_rp_bench.client.time.sleep', sleeps.append)
    def handle(request):
        calls.append(request)
        return httpx.Response(429, headers={'Retry-After': header}) if len(calls) == 1 else httpx.Response(200, json=response())
    with client_for(handle) as client:
        assert client.complete('model', [{'role': 'user', 'content': 'hi'}]) == '返答'
    assert sleeps == [expected]
