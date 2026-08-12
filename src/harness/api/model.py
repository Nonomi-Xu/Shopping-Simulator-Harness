"""Small, dependency-free client for the DeepSeek Chat Completions API.

Credentials are read from ``DEEPSEEK_API_KEY``. ``DEEPSEEK_API`` is also
accepted for compatibility with older local configuration, but callers can
always pass ``api_key`` explicitly.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


class DeepSeekError(RuntimeError):
    """Base exception raised by this module."""


class DeepSeekConfigurationError(DeepSeekError):
    """Raised when required client configuration is missing or invalid."""


class DeepSeekAPIError(DeepSeekError):
    """Raised when DeepSeek rejects a request or returns invalid data."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _api_key_from_environment() -> str | None:
    return os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API")


def _error_message(error: HTTPError) -> str:
    """Extract a useful provider message without including request headers."""
    try:
        body = error.read().decode("utf-8", errors="replace")
        payload = json.loads(body)
        detail = payload.get("error") if isinstance(payload, Mapping) else None
        if isinstance(detail, Mapping) and detail.get("message"):
            return str(detail["message"])
        if body.strip():
            return body.strip()[:500]
    except (OSError, ValueError, TypeError):
        pass
    return str(error.reason or "request failed")


class DeepSeekClient:
    """Synchronous client for DeepSeek's OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        resolved_key = api_key or _api_key_from_environment()
        if not resolved_key or not resolved_key.strip():
            raise DeepSeekConfigurationError(
                "DeepSeek API key is required; set DEEPSEEK_API_KEY or pass api_key"
            )
        if not base_url or not base_url.strip():
            raise DeepSeekConfigurationError("base_url must not be empty")
        if timeout <= 0:
            raise DeepSeekConfigurationError("timeout must be positive")

        self._api_key = resolved_key.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self._opener = opener or urlopen

    @property
    def chat_completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def create_chat_completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str = DEFAULT_MODEL,
        **parameters: Any,
    ) -> dict[str, Any]:
        """Send a non-streaming chat-completion request and return raw JSON.

        Extra keyword arguments are forwarded to DeepSeek, so options such as
        ``max_tokens``, ``temperature``, ``tools`` and ``thinking`` can be used.
        """
        if not model or not model.strip():
            raise ValueError("model must not be empty")
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise TypeError("messages must be a sequence of mappings")
        if not messages:
            raise ValueError("messages must not be empty")

        normalized_messages: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise TypeError(f"messages[{index}] must be a mapping")
            normalized_messages.append(dict(message))

        if parameters.get("stream") is True:
            raise ValueError("streaming responses are not supported by this client")
        reserved = {"model", "messages"}.intersection(parameters)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"reserved request parameter(s): {names}")

        payload = {
            "model": model.strip(),
            "messages": normalized_messages,
            "stream": False,
            **parameters,
        }
        request = Request(
            self.chat_completions_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "shopping-simulator-harness/1.0",
            },
            method="POST",
        )

        try:
            with self._opener(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = _error_message(error)
            raise DeepSeekAPIError(
                f"DeepSeek API returned HTTP {error.code}: {detail}",
                status_code=error.code,
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            reason = getattr(error, "reason", error)
            raise DeepSeekAPIError(f"DeepSeek API request failed: {reason}") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DeepSeekAPIError("DeepSeek API returned invalid JSON") from error

        if not isinstance(result, dict):
            raise DeepSeekAPIError("DeepSeek API response must be a JSON object")
        return result

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str = DEFAULT_MODEL,
        **parameters: Any,
    ) -> str:
        """Return the text content of the first response choice."""
        response = self.create_chat_completion(
            messages,
            model=model,
            **parameters,
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise DeepSeekAPIError("DeepSeek API response is missing choices")
        first_choice = choices[0]
        message = (
            first_choice.get("message")
            if isinstance(first_choice, Mapping)
            else None
        )
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            raise DeepSeekAPIError(
                "DeepSeek API response is missing choices[0].message.content"
            )
        return content


def call_deepseek(
    prompt: str,
    *,
    system_prompt: str | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
    **parameters: Any,
) -> str:
    """Convenience function for a single-turn DeepSeek request."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string when provided")
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    client = DeepSeekClient(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )
    return client.chat(messages, model=model, **parameters)
