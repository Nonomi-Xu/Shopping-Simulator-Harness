"""Local Ollama runtime as a model provider, replacing paid remote calls.

Ollama exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint on
localhost, so this provider reuses the shared Chat Completions implementation
and only changes the identity, the credential policy (a local server needs no
key), and the parameters Ollama does not understand.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request

from .base import (
    ProviderAPIError,
    ProviderConfigurationError,
    ProviderResponseError,
)
from .openai_compat import OpenAICompatibleProvider, default_transport_opener

DEFAULT_BASE_URL = "http://localhost:11434/v1"
#: Small, tool-calling-capable default that fits comfortably in 16 GB unified memory.
DEFAULT_MODEL = "qwen2.5:7b-instruct"

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "OllamaAPIError",
    "OllamaConfigurationError",
    "OllamaProvider",
    "OllamaResponseError",
    "list_local_models",
]


class OllamaConfigurationError(ProviderConfigurationError):
    """Ollama cannot run with the supplied local configuration."""


class OllamaAPIError(ProviderAPIError):
    """The local Ollama server rejected or could not complete a request."""


class OllamaResponseError(ProviderResponseError):
    """Ollama returned malformed or contract-incompatible JSON."""


def list_local_models(
    base_url: str = DEFAULT_BASE_URL,
    *,
    timeout: float = 5.0,
) -> list[str]:
    """Return the model tags the local Ollama server has pulled.

    Raises :class:`OllamaConfigurationError` when the server is unreachable so
    callers get an actionable message instead of a bare connection error.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    request = Request(f"{root}/api/tags", method="GET")
    try:
        with default_transport_opener()(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise OllamaConfigurationError(
            f"Ollama tag listing failed with HTTP {error.code} at {root}/api/tags"
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise OllamaConfigurationError(
            f"cannot reach the Ollama server at {root}; start it with "
            f"'ollama serve' (or 'brew services start ollama')"
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OllamaConfigurationError(
            f"Ollama returned invalid JSON from {root}/api/tags"
        ) from error
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return []
    names = []
    for entry in models:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            names.append(entry["name"])
    return names


class OllamaProvider(OpenAICompatibleProvider):
    """Synchronous, non-streaming provider backed by a local Ollama server."""

    provider_name = "ollama"
    label = "Ollama"
    # A local server does not authenticate; the base class sends a placeholder
    # bearer token because some OpenAI-shaped clients require the header.
    require_api_key = False
    api_key_env_names = ("OLLAMA_API_KEY",)
    placeholder_api_key = "ollama"
    tolerate_missing_usage_counters = True
    connection_hint = (
        "; start the local runtime with 'ollama serve'"
        " (or 'brew services start ollama') and pull a model with 'ollama pull <model>'"
    )
    # DeepSeek-only switch that Ollama has no concept of.  ``reasoning_effort``
    # and ``think`` are deliberately *not* stripped: Ollama uses them to control
    # thinking mode on hybrid-reasoning models such as Qwen3, so callers must be
    # able to pass them through.
    unsupported_parameters = frozenset({"thinking"})

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
        default_parameters: dict[str, Any] | None = None,
        opener=None,
        sleeper=None,
        request_observer=None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            # Local generation on Apple silicon is slower than a hosted API,
            # so the default budget is larger than DeepSeek's 60 seconds.
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            default_parameters=default_parameters,
            opener=opener,
            sleeper=sleeper,
            request_observer=request_observer,
        )

    def _configuration_error(self, message: str) -> ProviderConfigurationError:
        return OllamaConfigurationError(message)

    def _api_error(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> ProviderAPIError:
        return OllamaAPIError(
            message,
            status_code=status_code,
            retryable=retryable,
        )

    def _response_error(self, message: str) -> ProviderResponseError:
        return OllamaResponseError(message)
