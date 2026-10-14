"""DeepSeek Chat Completions implementation of the model-provider contract.

DeepSeek speaks the OpenAI Chat Completions protocol, so the transport and
normalization logic lives in :mod:`src.harness.providers.openai_compat`.  This
module only supplies DeepSeek's identity, credential policy, and error types.
"""

from __future__ import annotations

import os
from typing import Any

from .base import (
    ProviderAPIError,
    ProviderConfigurationError,
    ProviderError,
    ProviderResponseError,
)
from .openai_compat import OpenAICompatibleProvider

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

# Compatibility name: catching DeepSeekError catches every provider-level error.
DeepSeekError = ProviderError

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DeepSeekAPIError",
    "DeepSeekClient",
    "DeepSeekConfigurationError",
    "DeepSeekError",
    "DeepSeekProvider",
    "DeepSeekResponseError",
    "call_deepseek",
]


class DeepSeekConfigurationError(ProviderConfigurationError):
    """DeepSeek cannot run with the supplied local configuration."""


class DeepSeekAPIError(ProviderAPIError):
    """DeepSeek rejected or could not complete a request."""


class DeepSeekResponseError(ProviderResponseError):
    """DeepSeek returned malformed or contract-incompatible JSON."""


def _api_key_from_environment() -> str | None:
    return os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API")


class DeepSeekProvider(OpenAICompatibleProvider):
    """Synchronous, non-streaming DeepSeek model provider."""

    provider_name = "deepseek"
    label = "DeepSeek"
    require_api_key = True
    api_key_env_names = ("DEEPSEEK_API_KEY", "DEEPSEEK_API")

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
        default_parameters=None,
        opener=None,
        sleeper=None,
        request_observer=None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            default_parameters=default_parameters,
            opener=opener,
            sleeper=sleeper,
            request_observer=request_observer,
        )

    @property
    def missing_api_key_message(self) -> str:
        return "DeepSeek API key is required; set DEEPSEEK_API_KEY or pass api_key"

    def _api_key_from_environment(self) -> str | None:
        return _api_key_from_environment()

    def _configuration_error(self, message: str) -> ProviderConfigurationError:
        return DeepSeekConfigurationError(message)

    def _api_error(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> ProviderAPIError:
        return DeepSeekAPIError(
            message,
            status_code=status_code,
            retryable=retryable,
        )

    def _response_error(self, message: str) -> ProviderResponseError:
        return DeepSeekResponseError(message)


# Historical name retained for callers that used the original client directly.
DeepSeekClient = DeepSeekProvider


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
    """Compatibility helper for a single textual DeepSeek request."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError(
                "system_prompt must be a non-empty string when provided"
            )
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    provider = DeepSeekProvider(
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout=timeout,
    )
    return provider.chat(messages, **parameters)
