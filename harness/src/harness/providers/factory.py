"""Single place that maps a provider name to defaults and a concrete client.

Keeping the table here means the CLI, the benchmark runner, and the branch
experiment/episode drivers all resolve ``deepseek`` and ``ollama`` identically.
"""

from __future__ import annotations

from typing import Any

from .deepseek import (
    DEFAULT_BASE_URL as DEEPSEEK_DEFAULT_BASE_URL,
    DEFAULT_MODEL as DEEPSEEK_DEFAULT_MODEL,
    DeepSeekProvider,
)
from .ollama import (
    DEFAULT_BASE_URL as OLLAMA_DEFAULT_BASE_URL,
    DEFAULT_MODEL as OLLAMA_DEFAULT_MODEL,
    OllamaProvider,
)

#: Supported backend identifiers, in help-text order.
PROVIDER_NAMES = ("deepseek", "ollama")

#: Default model per backend.
DEFAULT_MODELS: dict[str, str] = {
    "deepseek": DEEPSEEK_DEFAULT_MODEL,
    "ollama": OLLAMA_DEFAULT_MODEL,
}

#: Default Chat Completions endpoint per backend.
DEFAULT_BASE_URLS: dict[str, str] = {
    "deepseek": DEEPSEEK_DEFAULT_BASE_URL,
    "ollama": OLLAMA_DEFAULT_BASE_URL,
}

#: Default request timeout per backend; local generation is slower.
DEFAULT_TIMEOUTS: dict[str, float] = {
    "deepseek": 60.0,
    "ollama": 120.0,
}

#: Whether the backend needs a credential before it can run.
REQUIRES_API_KEY: dict[str, bool] = {
    "deepseek": True,
    "ollama": False,
}


def provider_class(name: str):
    """Return the provider class implementing ``name``."""
    if name == "ollama":
        return OllamaProvider
    if name == "deepseek":
        return DeepSeekProvider
    raise ValueError(
        f"unknown provider {name!r}; expected one of {', '.join(PROVIDER_NAMES)}"
    )


def build_provider(
    name: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    max_retries: int = 2,
    retry_backoff: float = 0.5,
    default_parameters: dict[str, Any] | None = None,
    request_observer=None,
):
    """Build a configured provider, filling unset values from the backend table."""
    cls = provider_class(name)
    return cls(
        model=model or DEFAULT_MODELS[name],
        base_url=base_url or DEFAULT_BASE_URLS[name],
        api_key=api_key,
        timeout=DEFAULT_TIMEOUTS[name] if timeout is None else timeout,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        default_parameters=default_parameters,
        request_observer=request_observer,
    )
