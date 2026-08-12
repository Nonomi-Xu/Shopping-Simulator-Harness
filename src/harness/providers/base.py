"""Provider-neutral contracts for model-driven agent turns."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


MessageRole = Literal["system", "user", "assistant", "tool"]
ToolChoice: TypeAlias = (
    Literal["none", "auto", "required"] | Mapping[str, Any] | None
)


class ProviderError(RuntimeError):
    """Base error raised by every model provider."""


class ProviderConfigurationError(ProviderError):
    """The provider cannot run with the supplied local configuration."""


class ProviderAPIError(ProviderError):
    """The remote provider rejected or could not complete a request."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class ProviderResponseError(ProviderError):
    """The provider returned a response that violates the contract."""


class ToolCall(BaseModel):
    """One parsed function call emitted by a model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=64)
    arguments: dict[str, Any]
    raw_arguments: str = Field(min_length=1)

    def as_assistant_payload(self) -> dict[str, Any]:
        """Return the OpenAI-compatible assistant tool-call representation."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.raw_arguments,
            },
        }


class ChatMessage(BaseModel):
    """Provider-neutral message used by the Agent Runner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: MessageRole
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    reasoning_content: str | None = None

    @model_validator(mode="after")
    def _validate_role_fields(self) -> ChatMessage:
        if self.role in ("system", "user"):
            if self.content is None:
                raise ValueError(f"{self.role} messages require content")
            if self.tool_calls or self.tool_call_id or self.reasoning_content:
                raise ValueError(
                    f"{self.role} messages cannot contain tool or reasoning fields"
                )
        elif self.role == "assistant":
            if self.tool_call_id is not None:
                raise ValueError("assistant messages cannot contain tool_call_id")
            if self.content is None and not self.tool_calls:
                raise ValueError(
                    "assistant messages require content or at least one tool call"
                )
        else:
            if self.content is None:
                raise ValueError("tool messages require content")
            if not self.tool_call_id:
                raise ValueError("tool messages require tool_call_id")
            if self.tool_calls or self.reasoning_content:
                raise ValueError(
                    "tool messages cannot contain tool_calls or reasoning_content"
                )
        return self

    def as_provider_payload(self) -> dict[str, Any]:
        """Return an OpenAI-compatible chat message payload."""
        payload: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.name is not None:
            payload["name"] = self.name
        if self.role == "assistant":
            if self.tool_calls:
                payload["tool_calls"] = [
                    call.as_assistant_payload() for call in self.tool_calls
                ]
            if self.reasoning_content is not None:
                payload["reasoning_content"] = self.reasoning_content
        elif self.role == "tool":
            payload["tool_call_id"] = self.tool_call_id
        return payload


class ToolDefinition(BaseModel):
    """One function tool exposed to a model provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    description: str = Field(min_length=1)
    parameters: dict[str, Any]
    strict: bool = False

    @classmethod
    def from_action_model(
        cls,
        *,
        name: str,
        description: str,
        action_model: type[BaseModel],
        strict: bool = False,
    ) -> ToolDefinition:
        """Build a tool directly from the authoritative Pydantic action model."""
        if not isinstance(action_model, type) or not issubclass(
            action_model,
            BaseModel,
        ):
            raise TypeError("action_model must be a Pydantic BaseModel class")
        return cls(
            name=name,
            description=description,
            parameters=action_model.model_json_schema(),
            strict=strict,
        )

    def as_provider_payload(self) -> dict[str, Any]:
        function: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        if self.strict:
            function["strict"] = True
        return {"type": "function", "function": function}


class TokenUsage(BaseModel):
    """Normalized token accounting returned by a provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    cache_miss_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)


class AssistantTurn(BaseModel):
    """Normalized result of one provider completion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = Field(min_length=1)
    usage: TokenUsage | None = None
    response_id: str | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_output(self) -> AssistantTurn:
        if self.content is None and not self.tool_calls:
            raise ValueError("assistant turn requires content or tool calls")
        return self

    def as_message(self) -> ChatMessage:
        """Convert the turn into the assistant message for the next request."""
        return ChatMessage(
            role="assistant",
            content=self.content,
            reasoning_content=self.reasoning_content,
            tool_calls=self.tool_calls,
        )


@runtime_checkable
class ModelProvider(Protocol):
    """Minimal synchronous model interface consumed by an Agent Runner."""

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolDefinition] = (),
        tool_choice: ToolChoice = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> AssistantTurn: ...


def canonical_json(value: Mapping[str, Any]) -> str:
    """Serialize tool arguments deterministically for message replay."""
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
