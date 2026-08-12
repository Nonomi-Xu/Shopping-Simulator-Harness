"""Model-provider contracts and built-in provider implementations."""

from .base import (
    AssistantTurn,
    ChatMessage,
    ModelProvider,
    ProviderAPIError,
    ProviderConfigurationError,
    ProviderError,
    ProviderResponseError,
    TokenUsage,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    canonical_json,
)
from .deepseek import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DeepSeekAPIError,
    DeepSeekConfigurationError,
    DeepSeekError,
    DeepSeekProvider,
    DeepSeekResponseError,
)

__all__ = [
    "AssistantTurn",
    "ChatMessage",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DeepSeekAPIError",
    "DeepSeekConfigurationError",
    "DeepSeekError",
    "DeepSeekProvider",
    "DeepSeekResponseError",
    "ModelProvider",
    "ProviderAPIError",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderResponseError",
    "TokenUsage",
    "ToolCall",
    "ToolChoice",
    "ToolDefinition",
    "canonical_json",
]
