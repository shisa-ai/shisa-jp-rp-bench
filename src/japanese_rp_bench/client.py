"""Minimal pooled HTTP transport for OpenAI-compatible chat completions."""
from copy import deepcopy
import math
import time
from typing import Any

import httpx


class ChatError(RuntimeError):
    """Sanitized request failure: never includes credentials or upstream bodies."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


_RESERVED = {'model', 'messages', 'stream', 'api_key', 'authorization', 'headers', 'base_url'}


def _validate_options(options):
    if not isinstance(options, dict) or _RESERVED.intersection(options):
        raise ValueError('Request options must be a mapping without routing/authentication fields')
    # Validate serializability without including potentially sensitive values in errors.
    import json
    try:
        json.dumps(options, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError('Request options must contain finite JSON values') from None
    for field in ('max_tokens', 'max_completion_tokens'):
        if field in options and (type(options[field]) is not int or options[field] <= 0):
            raise ValueError(f'{field} must be a positive integer')
    if 'max_tokens' in options and 'max_completion_tokens' in options:
        raise ValueError('Specify only one token-limit field')


class ChatClient:
    """One thread-safe HTTPX pool per endpoint; callers own its lifetime."""

    def __init__(self, base_url: str, api_key: str | None = None, *,
                 timeout: float = 120, max_retries: int = 2,
                 transport: httpx.BaseTransport | None = None,
                 default_options: dict[str, Any] | None = None):
        try:
            url = httpx.URL(base_url)
        except (TypeError, httpx.InvalidURL):
            raise ValueError('Invalid API base URL') from None
        if url.scheme not in ('http', 'https') or not url.host or url.userinfo or url.query or url.fragment:
            raise ValueError('API base URL must be HTTP(S), without credentials, query, or fragment')
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be a positive finite number')
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError('max_retries must be a nonnegative integer')
        if api_key is not None and (not isinstance(api_key, str) or '\n' in api_key or '\r' in api_key):
            raise ValueError('API key must be a single-line string')
        self.default_options = deepcopy(default_options if default_options is not None else {})
        _validate_options(self.default_options)
        self.max_retries = max_retries
        self._url = str(url).rstrip('/') + '/chat/completions'
        headers = {'Accept': 'application/json'}
        if api_key:
            headers['Authorization'] = 'Bearer ' + api_key
        self._http = httpx.Client(headers=headers, timeout=timeout, transport=transport,
                                  follow_redirects=False)

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def complete(self, model: str, messages: list[dict[str, str]], **options) -> str:
        if not isinstance(model, str) or not model.strip():
            raise ValueError('model must be a nonempty string')
        if not isinstance(messages, list) or not messages:
            raise ValueError('messages must be a nonempty list')
        for message in messages:
            if not isinstance(message, dict) or message.get('role') not in ('system', 'developer', 'user', 'assistant', 'tool') or not isinstance(message.get('content'), str):
                raise ValueError('Each message requires a supported role and string content')
        merged = {**self.default_options, **options}
        _validate_options(merged)
        payload = {**merged, 'model': model, 'messages': deepcopy(messages), 'stream': False}
        for attempt in range(self.max_retries + 1):
            try:
                result = self._http.post(self._url, json=payload)
            except httpx.TransportError:
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 10))
                    continue
                raise ChatError('Chat transport failed after bounded retries') from None
            if result.is_success:
                return self._text(result)
            if (result.status_code in (408, 429) or result.status_code >= 500) and attempt < self.max_retries:
                try:
                    delay = float(result.headers.get('Retry-After', 2 ** attempt))
                    if not math.isfinite(delay) or delay < 0:
                        raise ValueError
                except ValueError:
                    delay = 2 ** attempt
                time.sleep(min(delay, 10))
                continue
            raise ChatError(f'Chat request failed (HTTP {result.status_code})', status_code=result.status_code)
        raise AssertionError('unreachable')

    @staticmethod
    def _text(result: httpx.Response) -> str:
        try:
            payload = result.json()
            message = payload['choices'][0]['message']
            content = message.get('content')
            if content is None:
                content = message.get('refusal')
            if isinstance(content, list):
                content = ''.join(p['text'] for p in content if isinstance(p, dict) and isinstance(p.get('text'), str))
            if not isinstance(content, str) or not content.strip():
                raise ValueError
            return content.strip()
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise ChatError('Chat endpoint returned an empty or malformed completion') from None
