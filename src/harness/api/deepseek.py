"""Backward-compatible DeepSeek imports.

New code should import from :mod:`src.harness.providers.deepseek`.
"""

from ..providers.deepseek import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DeepSeekAPIError,
    DeepSeekClient,
    DeepSeekConfigurationError,
    DeepSeekError,
    DeepSeekProvider,
    DeepSeekResponseError,
    call_deepseek,
)

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
