"""DeepSeek Chat Completions implementation of the model-provider contract."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from .base import (
    AssistantTurn,
    ChatMessage,
    ProviderAPIError,
    ProviderConfigurationError,
    ProviderError,
    ProviderResponseError,
    TokenUsage,
    ToolCall,
    ToolChoice,
    ToolDefinition,
)


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
_RETRYABLE_STATUS_CODES = {408, 409, 429}
_RESERVED_PARAMETERS = {"model", "messages", "stream", "tools", "tool_choice"}

# Compatibility name: catching DeepSeekError catches every provider-level error.
DeepSeekError = ProviderError


class DeepSeekConfigurationError(ProviderConfigurationError):
    """DeepSeek cannot run with the supplied local configuration."""


class DeepSeekAPIError(ProviderAPIError):
    """DeepSeek rejected or could not complete a request."""


class DeepSeekResponseError(ProviderResponseError):
    """DeepSeek returned malformed or contract-incompatible JSON."""


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


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DeepSeekResponseError(f"DeepSeek usage.{field} must be a non-negative integer")
    return value


class DeepSeekProvider:
    """Synchronous, non-streaming DeepSeek model provider."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
        default_parameters: Mapping[str, Any] | None = None,
        opener: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        resolved_key = api_key or _api_key_from_environment()
        if not resolved_key or not resolved_key.strip():
            raise DeepSeekConfigurationError(
                "DeepSeek API key is required; set DEEPSEEK_API_KEY or pass api_key"
            )
        if not isinstance(model, str) or not model.strip():
            raise DeepSeekConfigurationError("model must not be empty")
        if not isinstance(base_url, str) or not base_url.strip():
            raise DeepSeekConfigurationError("base_url must not be empty")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise DeepSeekConfigurationError("timeout must be positive")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise DeepSeekConfigurationError("max_retries must be an integer")
        if max_retries < 0:
            raise DeepSeekConfigurationError("max_retries must not be negative")
        if (
            isinstance(retry_backoff, bool)
            or not isinstance(retry_backoff, (int, float))
            or retry_backoff < 0
        ):
            raise DeepSeekConfigurationError("retry_backoff must not be negative")

        defaults = dict(default_parameters or {})
        reserved = _RESERVED_PARAMETERS.intersection(defaults)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise DeepSeekConfigurationError(
                f"reserved default parameter(s): {names}"
            )

        self._api_key = resolved_key.strip()
        self._model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = max_retries
        self.retry_backoff = float(retry_backoff)
        self.default_parameters = defaults
        self._opener = opener or urlopen
        self._sleeper = sleeper or time.sleep

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def model(self) -> str:
        return self._model

    @property
    def chat_completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolDefinition] = (),
        tool_choice: ToolChoice = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> AssistantTurn:
        """Generate and normalize one model turn."""
        normalized_messages = self._normalize_messages(messages)
        normalized_tools = self._normalize_tools(tools)
        request_parameters = dict(self.default_parameters)
        call_parameters = dict(parameters or {})
        reserved = _RESERVED_PARAMETERS.intersection(call_parameters)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"reserved completion parameter(s): {names}")
        request_parameters.update(call_parameters)
        if normalized_tools:
            request_parameters["tools"] = [
                tool.as_provider_payload() for tool in normalized_tools
            ]
        elif tool_choice not in (None, "none"):
            raise ValueError("tool_choice requires at least one tool")
        if tool_choice is not None:
            request_parameters["tool_choice"] = (
                dict(tool_choice)
                if isinstance(tool_choice, Mapping)
                else tool_choice
            )

        response = self.create_chat_completion(
            [message.as_provider_payload() for message in normalized_messages],
            model=self.model,
            **request_parameters,
        )
        return self._parse_completion(response)

    def create_chat_completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str | None = None,
        **parameters: Any,
    ) -> dict[str, Any]:
        """Send a raw non-streaming Chat Completions request."""
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise TypeError("messages must be a sequence of mappings")
        if not messages:
            raise ValueError("messages must not be empty")
        normalized_messages: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise TypeError(f"messages[{index}] must be a mapping")
            normalized_messages.append(dict(message))
        selected_model = self.model if model is None else str(model).strip()
        if not selected_model:
            raise ValueError("model must not be empty")
        if parameters.get("stream") is True:
            raise ValueError("streaming responses are not supported")
        reserved = {"model", "messages"}.intersection(parameters)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"reserved request parameter(s): {names}")

        payload = {
            "model": selected_model,
            "messages": normalized_messages,
            "stream": False,
            **parameters,
        }
        try:
            encoded_payload = json.dumps(
                payload,
                ensure_ascii=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("DeepSeek request payload is not JSON serializable") from exc

        request = Request(
            self.chat_completions_url,
            data=encoded_payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "shopping-simulator-harness/1.0",
            },
            method="POST",
        )
        return self._send(request)

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str | None = None,
        **parameters: Any,
    ) -> str:
        """Compatibility helper returning the first textual response."""
        response = self.create_chat_completion(
            messages,
            model=model,
            **parameters,
        )
        turn = self._parse_completion(response)
        if turn.content is None:
            raise DeepSeekResponseError(
                "DeepSeek response contains tool calls but no text content"
            )
        return turn.content

    def _send(self, request: Request) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
            except HTTPError as error:
                retryable = (
                    error.code in _RETRYABLE_STATUS_CODES or error.code >= 500
                )
                api_error = DeepSeekAPIError(
                    f"DeepSeek API returned HTTP {error.code}: {_error_message(error)}",
                    status_code=error.code,
                    retryable=retryable,
                )
                if retryable and attempt < self.max_retries:
                    self._wait_before_retry(attempt)
                    continue
                raise api_error from error
            except (URLError, TimeoutError, OSError) as error:
                reason = getattr(error, "reason", error)
                api_error = DeepSeekAPIError(
                    f"DeepSeek API request failed: {reason}",
                    retryable=True,
                )
                if attempt < self.max_retries:
                    self._wait_before_retry(attempt)
                    continue
                raise api_error from error
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise DeepSeekResponseError(
                    "DeepSeek API returned invalid JSON"
                ) from error

            if not isinstance(result, dict):
                raise DeepSeekResponseError(
                    "DeepSeek API response must be a JSON object"
                )
            return result
        raise AssertionError("retry loop exhausted without returning or raising")

    def _wait_before_retry(self, attempt: int) -> None:
        delay = self.retry_backoff * (2**attempt)
        if delay > 0:
            self._sleeper(delay)

    @staticmethod
    def _normalize_messages(
        messages: Sequence[ChatMessage],
    ) -> list[ChatMessage]:
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise TypeError("messages must be a sequence of ChatMessage values")
        if not messages:
            raise ValueError("messages must not be empty")
        normalized = []
        for index, message in enumerate(messages):
            if not isinstance(message, ChatMessage):
                raise TypeError(f"messages[{index}] must be a ChatMessage")
            normalized.append(message)
        return normalized

    @staticmethod
    def _normalize_tools(
        tools: Sequence[ToolDefinition],
    ) -> list[ToolDefinition]:
        if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
            raise TypeError("tools must be a sequence of ToolDefinition values")
        normalized = []
        names: set[str] = set()
        for index, tool in enumerate(tools):
            if not isinstance(tool, ToolDefinition):
                raise TypeError(f"tools[{index}] must be a ToolDefinition")
            if tool.name in names:
                raise ValueError(f"duplicate tool definition: {tool.name}")
            names.add(tool.name)
            normalized.append(tool)
        return normalized

    def _parse_completion(self, response: Mapping[str, Any]) -> AssistantTurn:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise DeepSeekResponseError("DeepSeek response is missing choices")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise DeepSeekResponseError("DeepSeek choices[0] must be an object")
        finish_reason = choice.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise DeepSeekResponseError(
                "DeepSeek response is missing choices[0].finish_reason"
            )
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise DeepSeekResponseError(
                "DeepSeek response is missing choices[0].message"
            )

        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise DeepSeekResponseError("DeepSeek message.content must be a string or null")
        reasoning_content = message.get("reasoning_content")
        if reasoning_content is not None and not isinstance(reasoning_content, str):
            raise DeepSeekResponseError(
                "DeepSeek message.reasoning_content must be a string or null"
            )
        tool_calls = self._parse_tool_calls(message.get("tool_calls"))
        if finish_reason == "tool_calls" and not tool_calls:
            raise DeepSeekResponseError(
                "DeepSeek finish_reason is tool_calls but no tool calls were returned"
            )
        if content is None and not tool_calls:
            raise DeepSeekResponseError(
                "DeepSeek response contains neither content nor tool calls"
            )

        response_model = response.get("model")
        if not isinstance(response_model, str) or not response_model:
            response_model = self.model
        response_id = response.get("id")
        if response_id is not None and not isinstance(response_id, str):
            raise DeepSeekResponseError("DeepSeek response.id must be a string")
        metadata = {
            key: response[key]
            for key in ("created", "system_fingerprint", "object")
            if key in response
        }
        try:
            return AssistantTurn(
                provider=self.name,
                model=response_model,
                content=content,
                reasoning_content=reasoning_content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=self._parse_usage(response.get("usage")),
                response_id=response_id,
                provider_metadata=metadata,
            )
        except ValidationError as exc:
            raise DeepSeekResponseError(
                f"DeepSeek response failed normalized schema validation: {exc}"
            ) from exc

    @staticmethod
    def _parse_tool_calls(value: object) -> list[ToolCall]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise DeepSeekResponseError("DeepSeek message.tool_calls must be an array")
        calls = []
        seen_ids: set[str] = set()
        for index, raw_call in enumerate(value):
            if not isinstance(raw_call, Mapping):
                raise DeepSeekResponseError(
                    f"DeepSeek tool_calls[{index}] must be an object"
                )
            if raw_call.get("type") != "function":
                raise DeepSeekResponseError(
                    f"DeepSeek tool_calls[{index}].type must be function"
                )
            call_id = raw_call.get("id")
            function = raw_call.get("function")
            if not isinstance(call_id, str) or not call_id:
                raise DeepSeekResponseError(
                    f"DeepSeek tool_calls[{index}].id must be a non-empty string"
                )
            if call_id in seen_ids:
                raise DeepSeekResponseError(f"duplicate DeepSeek tool call ID: {call_id}")
            seen_ids.add(call_id)
            if not isinstance(function, Mapping):
                raise DeepSeekResponseError(
                    f"DeepSeek tool_calls[{index}].function must be an object"
                )
            name = function.get("name")
            raw_arguments = function.get("arguments")
            if not isinstance(name, str) or not name:
                raise DeepSeekResponseError(
                    f"DeepSeek tool_calls[{index}].function.name is invalid"
                )
            if not isinstance(raw_arguments, str) or not raw_arguments:
                raise DeepSeekResponseError(
                    f"DeepSeek tool_calls[{index}].function.arguments is invalid"
                )
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise DeepSeekResponseError(
                    f"DeepSeek tool {name} returned invalid JSON arguments"
                ) from exc
            if not isinstance(arguments, dict):
                raise DeepSeekResponseError(
                    f"DeepSeek tool {name} arguments must decode to an object"
                )
            try:
                calls.append(
                    ToolCall(
                        id=call_id,
                        name=name,
                        arguments=arguments,
                        raw_arguments=raw_arguments,
                    )
                )
            except ValidationError as exc:
                raise DeepSeekResponseError(
                    f"DeepSeek tool_calls[{index}] failed validation: {exc}"
                ) from exc
        return calls

    @staticmethod
    def _parse_usage(value: object) -> TokenUsage | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise DeepSeekResponseError("DeepSeek usage must be an object")
        details = value.get("completion_tokens_details")
        if details is not None and not isinstance(details, Mapping):
            raise DeepSeekResponseError(
                "DeepSeek usage.completion_tokens_details must be an object"
            )
        cached = value.get("prompt_cache_hit_tokens")
        cache_miss = value.get("prompt_cache_miss_tokens")
        reasoning = details.get("reasoning_tokens") if details else None
        return TokenUsage(
            input_tokens=_non_negative_int(
                value.get("prompt_tokens"),
                field="prompt_tokens",
            ),
            output_tokens=_non_negative_int(
                value.get("completion_tokens"),
                field="completion_tokens",
            ),
            total_tokens=_non_negative_int(
                value.get("total_tokens"),
                field="total_tokens",
            ),
            cached_input_tokens=(
                None
                if cached is None
                else _non_negative_int(cached, field="prompt_cache_hit_tokens")
            ),
            cache_miss_input_tokens=(
                None
                if cache_miss is None
                else _non_negative_int(
                    cache_miss,
                    field="prompt_cache_miss_tokens",
                )
            ),
            reasoning_tokens=(
                None
                if reasoning is None
                else _non_negative_int(reasoning, field="reasoning_tokens")
            ),
        )


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
