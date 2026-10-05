"""Minimal pooled HTTP transport for OpenAI-compatible chat completions."""
from copy import deepcopy
from dataclasses import dataclass, field
import math
import time
from typing import Any

import httpx


class ChatError(RuntimeError):
    """Sanitized request failure: never includes credentials or upstream bodies."""

    def __init__(self, message: str, *, status_code: int | None = None, completion=None):
        super().__init__(message)
        self.status_code = status_code
        self.completion = completion


@dataclass
class Completion:
    """Final answer plus endpoint-provided diagnostics, never conversation history."""

    content: str
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    attempts: list[dict] = field(default_factory=list)

    def metadata(self):
        return {"reasoning": self.reasoning, "finish_reason": self.finish_reason,
                "usage": deepcopy(self.usage), "attempts": deepcopy(self.attempts)}


class CompletionError(ChatError):
    """A response arrived, but it did not contain a complete final answer."""

    def __init__(self, message, completion):
        super().__init__(message, completion=completion)


def validate_completion_policy(token_limit_ceiling=None, max_token_retries=2, strip_think_tags=False):
    if token_limit_ceiling is not None and (type(token_limit_ceiling) is not int or token_limit_ceiling < 1):
        raise ValueError('token_limit_ceiling must be a positive integer')
    if type(max_token_retries) is not int or max_token_retries < 0:
        raise ValueError('max_token_retries must be a nonnegative integer')
    if type(strip_think_tags) is not bool:
        raise ValueError('strip_think_tags must be a boolean')


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


def _validate_token_budget(options, ceiling):
    limit = options.get('max_completion_tokens', options.get('max_tokens'))
    if ceiling is not None and (limit is None or limit > ceiling):
        raise ValueError('Token retries require an initial token limit no larger than token_limit_ceiling')


class ChatClient:
    """One thread-safe HTTPX pool per endpoint; callers own its lifetime."""

    def __init__(self, base_url: str, api_key: str | None = None, *,
                 timeout: float = 120, max_retries: int = 2,
                 transport: httpx.BaseTransport | None = None,
                 default_options: dict[str, Any] | None = None,
                 token_limit_ceiling: int | None = None, max_token_retries: int = 2,
                 strip_think_tags: bool = False):
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
        validate_completion_policy(token_limit_ceiling, max_token_retries, strip_think_tags)
        self.token_limit_ceiling = token_limit_ceiling
        self.max_token_retries = max_token_retries
        self.strip_think_tags = strip_think_tags
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
        return self.complete_with_details(model, messages, **options).content

    def complete_with_details(self, model: str, messages: list[dict[str, str]], **options) -> Completion:
        if not isinstance(model, str) or not model.strip():
            raise ValueError('model must be a nonempty string')
        if not isinstance(messages, list) or not messages:
            raise ValueError('messages must be a nonempty list')
        for message in messages:
            if not isinstance(message, dict) or message.get('role') not in ('system', 'developer', 'user', 'assistant', 'tool') or not isinstance(message.get('content'), str):
                raise ValueError('Each message requires a supported role and string content')
        merged = {**self.default_options, **options}
        _validate_options(merged)
        # Explicit null suppresses defaults for endpoints that reject sampling controls.
        for name in ('temperature', 'top_p'):
            if merged.get(name, ...) is None:
                merged.pop(name)
        token_field = 'max_completion_tokens' if 'max_completion_tokens' in merged else 'max_tokens'
        _validate_token_budget(merged, self.token_limit_ceiling)
        payload = {**merged, 'model': model, 'messages': deepcopy(messages), 'stream': False}
        attempts = []
        for budget_attempt in range(self.max_token_retries + 1):
            try:
                completion = self._completion(self._request(payload))
            except ChatError as error:
                if error.completion is not None:
                    failed = error.completion
                    failed.attempts = [*attempts, {'token_limit': payload.get(token_field),
                                                 'finish_reason': failed.finish_reason,
                                                 'usage': deepcopy(failed.usage)}]
                elif attempts:
                    error.completion = completion
                raise
            attempts.append({'token_limit': payload.get(token_field), 'finish_reason': completion.finish_reason,
                             'usage': deepcopy(completion.usage)})
            completion.attempts = list(attempts)
            if completion.finish_reason == 'length':
                if (self.token_limit_ceiling is not None and budget_attempt < self.max_token_retries
                        and payload[token_field] < self.token_limit_ceiling):
                    payload[token_field] = min(payload[token_field] * 2, self.token_limit_ceiling)
                    continue
                raise CompletionError('Chat completion reached its token limit before finishing', completion)
            if completion.finish_reason not in (None, 'stop'):
                raise CompletionError('Chat completion did not finish with a final answer', completion)
            if not completion.content:
                if completion.reasoning:
                    raise CompletionError('Chat endpoint returned reasoning without a final answer', completion)
                raise CompletionError('Chat endpoint returned an empty or malformed completion', completion)
            return completion
        raise AssertionError('unreachable')

    def _request(self, payload):
        for attempt in range(self.max_retries + 1):
            try:
                result = self._http.post(self._url, json=payload)
            except httpx.TransportError:
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 10))
                    continue
                raise ChatError('Chat transport failed after bounded retries') from None
            if result.is_success:
                return result
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

    def _completion(self, result: httpx.Response) -> Completion:
        diagnostics = Completion(content='')
        try:
            payload = result.json()
            diagnostics.usage = _usage(payload.get('usage'))
            choice = payload['choices'][0]
            message = choice['message']
            content = message.get('content')
            reasoning = []
            for name in ('reasoning_content', 'reasoning'):
                value = message.get(name)
                if isinstance(value, str) and value.strip():
                    reasoning.append(value.strip())
                    break
            if content is None:
                content = message.get('refusal')
            if isinstance(content, list):
                texts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    kind = part.get('type', 'text')
                    if kind in ('thinking', 'reasoning'):
                        value = part.get('thinking', part.get('text'))
                        if isinstance(value, str) and value.strip():
                            reasoning.append(value.strip())
                    elif kind in ('text', 'output_text') and isinstance(part.get('text'), str):
                        texts.append(part['text'])
                content = ''.join(texts)
            if content is not None and not isinstance(content, str):
                raise ValueError
            content = (content or '').strip()
            if self.strip_think_tags:
                while content.startswith('<think>'):
                    trace, separator, content = content[len('<think>'):].partition('</think>')
                    if trace.strip():
                        reasoning.append(trace.strip())
                    content = content.strip() if separator else ''
            finish = choice.get('finish_reason')
            if finish is not None and not isinstance(finish, str):
                raise ValueError
            return Completion(content=content, reasoning='\n'.join(reasoning) or None,
                              finish_reason=finish, usage=diagnostics.usage)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise ChatError('Chat endpoint returned an empty or malformed completion',
                            completion=diagnostics if diagnostics.usage else None) from None


def _usage(value):
    """Keep only numeric token accounting, never arbitrary upstream metadata."""
    if not isinstance(value, dict):
        return {}
    result = {key: value[key] for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')
              if type(value.get(key)) is int and value[key] >= 0}
    for field, keys in (('completion_tokens_details', ('reasoning_tokens', 'audio_tokens',
                         'accepted_prediction_tokens', 'rejected_prediction_tokens')),
                        ('prompt_tokens_details', ('cached_tokens', 'audio_tokens'))):
        details = value.get(field)
        if isinstance(details, dict):
            result[field] = {key: details[key] for key in keys
                             if type(details.get(key)) is int and details[key] >= 0}
    return result
