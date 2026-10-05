"""Client creation and response generation use the shared HTTPX transport."""
import json
from types import SimpleNamespace

import httpx
import pytest
from japanese_rp_bench import models
from japanese_rp_bench.client import ChatClient, Completion


def test_create_configured_httpx_client(monkeypatch):
    created = []
    def factory(*args, **kwargs):
        created.append((args, kwargs))
        return object()
    monkeypatch.setattr(models, 'ChatClient', factory)
    monkeypatch.setenv('TEST_ROLE_KEY', 'role-secret')
    client = models.create_client(base_url='https://example.test/v1', api_key_env='TEST_ROLE_KEY', timeout=12, max_retries=1, request_options={'max_tokens': 32})
    assert client is not None
    assert created == [((), {'base_url':'https://example.test/v1','api_key':'role-secret','timeout':12,'max_retries':1,'default_options':{'max_tokens':32},'token_limit_ceiling':None,'max_token_retries':2,'strip_think_tags':False})]


def test_explicit_api_key_and_environment_fallbacks(monkeypatch):
    created = []
    monkeypatch.setattr(models, 'ChatClient', lambda **kwargs: created.append(kwargs))
    monkeypatch.setenv('OPENAI_COMPATIBLE_API_URL', 'https://example.test/v1')
    monkeypatch.setenv('OPENAI_COMPATIBLE_API_KEY', 'fallback-key')
    models.create_client(api_key='explicit')
    assert created[-1]['base_url'] == 'https://example.test/v1'
    assert created[-1]['api_key'] == 'explicit'
    models.create_client()
    assert created[-1]['api_key'] == 'fallback-key'


def test_named_key_must_exist(monkeypatch):
    monkeypatch.delenv('TEST_MISSING_KEY', raising=False)
    with pytest.raises(ValueError, match='TEST_MISSING_KEY'):
        models.create_client(api_key_env='TEST_MISSING_KEY')


def test_shisa_key_is_not_sent_to_default_openai_host(monkeypatch):
    monkeypatch.setenv('SHISA_API_KEY', 'shisa-secret')
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    created = []
    monkeypatch.setattr(models, 'ChatClient', lambda **kwargs: created.append(kwargs))
    models.create_client()
    assert created[-1]['api_key'] is None
    models.create_client(base_url='https://api.shisa.ai/openai/v1')
    assert created[-1]['api_key'] == 'shisa-secret'


def test_generation_sends_full_message_history():
    calls = []
    def handle(req):
        calls.append(json.loads(req.content))
        return httpx.Response(200,json={'choices':[{'message':{'content':'返答'}}]})
    history = [{'role':'user','content':'こんにちは'}]
    with ChatClient('https://example.test/v1', transport=httpx.MockTransport(handle)) as client:
        assert models.generate_response(client,'m','役割',history) == '返答'
    assert calls == [{'model':'m','messages':[{'role':'system','content':'役割'},*history],'stream':False,'temperature':0.7,'max_tokens':1024}]
    assert history == [{'role':'user','content':'こんにちは'}]


def test_model_options_control_token_field_without_name_guessing(monkeypatch):
    captured = []
    client = SimpleNamespace(default_options={'max_completion_tokens':1234,'temperature':1}, complete_with_details=lambda *args,**kwargs: captured.append(kwargs) or Completion('response'))
    models.generate_response(client,'o1-anything','s',[{'role':'user','content':'u'}])
    assert captured == [{'max_completion_tokens':1234,'temperature':1}]
    monkeypatch.setenv('JP_RP_MAX_TOKENS','55')
    models.generate_response(client,'m','s',[{'role':'user','content':'u'}])
    assert captured[-1] == {'max_completion_tokens':1234,'temperature':1}


def test_public_api_import_does_not_load_dataset_dependencies():
    """API-only consumers must work without the dataset stack installed."""
    import os
    from pathlib import Path
    import subprocess
    import sys

    script = '''import builtins
original_import = builtins.__import__
def import_api_only(name, *args, **kwargs):
    if name.split('.')[0] in {'datasets', 'pandas'}:
        raise ModuleNotFoundError('Dataset dependencies are intentionally unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = import_api_only
from japanese_rp_bench import create_client, generate_response
assert all(callable(function) for function in (create_client, generate_response))
'''
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run([sys.executable, "-c", script], env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
