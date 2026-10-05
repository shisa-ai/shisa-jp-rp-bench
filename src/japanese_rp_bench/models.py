"""Authenticated API client creation and roleplay response generation."""
import os
from urllib.parse import urlsplit

from .client import ChatClient

def create_client(*, base_url=None, api_key=None, api_key_env=None, timeout=120,
                  max_retries=2, request_options=None, token_limit_ceiling=None,
                  max_token_retries=2, strip_think_tags=False):
    """Create one pooled HTTPX client; callers close it after their requests."""
    endpoint = base_url or os.getenv('OPENAI_COMPATIBLE_API_URL') or os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1'
    if api_key is None:
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f'API key not found in environment variable: {api_key_env}')
        else:
            api_key = os.getenv('OPENAI_COMPATIBLE_API_KEY') or os.getenv('OPENAI_API_KEY')
            # Never forward this service-specific fallback to unrelated hosts.
            if api_key is None and urlsplit(endpoint).hostname == 'api.shisa.ai':
                api_key = os.getenv('SHISA_API_KEY')
    return ChatClient(base_url=endpoint, api_key=api_key, timeout=timeout,
                      max_retries=max_retries, default_options=request_options,
                      token_limit_ceiling=token_limit_ceiling, max_token_retries=max_token_retries,
                      strip_think_tags=strip_think_tags)


def generate_response(client, model_name, system_prompt, conversations):
    return generate_completion(client, model_name, system_prompt, conversations).content


def generate_completion(client, model_name, system_prompt, conversations):
    """Generate a final answer with separate reasoning and per-attempt metadata."""
    options = dict(client.default_options)
    options.setdefault("temperature", 0.7)
    if "max_tokens" not in options and "max_completion_tokens" not in options:
        options["max_tokens"] = 1024
    messages = [{"role": "system", "content": system_prompt}, *conversations]
    return client.complete_with_details(model_name, messages, **options)
