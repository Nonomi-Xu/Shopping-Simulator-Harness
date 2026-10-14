"""Shared Chat Completions implementation used by every HTTP model provider.

DeepSeek and the local Ollama runtime both speak the OpenAI Chat Completions
protocol, so request building, transport, retry, and response normalization
live here once.  Subclasses supply the identity (name/label), the credential
policy, and their own error types.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from pydantic import ValidationError

from .base import (
    AssistantTurn,
    ChatMessage,
    ProviderAPIError,
    ProviderConfigurationError,
    ProviderResponseError,
    TokenUsage,
    ToolCall,
    ToolChoice,
    ToolDefinition,
)

_RETRYABLE_STATUS_CODES = {408, 409, 429}
_RESERVED_PARAMETERS = {"model", "messages", "stream", "tools", "tool_choice"}

#: Redirect codes whose semantics require replaying the original POST body.
_REDIRECT_STATUS_CODES = frozenset({307, 308})
#: Bounded hops; the initial request is hop zero.
_MAX_REDIRECTS = 3
#: Effective port for a scheme, so ``https://host`` and ``https://host:443``
#: compare as one origin instead of being refused as different ones.
_DEFAULT_SCHEME_PORTS = {"http": 80, "https": 443}
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _origin(url: str) -> tuple[str, str, int] | None:
    """Return the normalized ``(scheme, host, effective_port)`` of a URL."""
    try:
        parsed = urlsplit(url)
        scheme = (parsed.scheme or "").lower()
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        return None
    if scheme not in _DEFAULT_SCHEME_PORTS or not host:
        return None
    return (scheme, host, _DEFAULT_SCHEME_PORTS[scheme] if port is None else port)


def _is_loopback(host: str) -> bool:
    return host in _LOOPBACK_HOSTS or host.startswith("127.")


class _LoopbackDirectProxyHandler(ProxyHandler):
    """Honour system proxies, but never send loopback traffic through them.

    A system HTTP proxy exists to reach remote APIs; it cannot serve a local
    model runtime. macOS reports ``127.0.0.1`` as proxy-eligible (only the
    literal name ``localhost`` is exempt), so a local provider configured with
    the equally common ``http://127.0.0.1:11434/v1`` was routed to the proxy and
    failed with a misleading "cannot reach the server" error.
    """

    def proxy_open(self, req, proxy, type):
        if _is_loopback((urlsplit(req.full_url).hostname or "").lower()):
            return None
        return super().proxy_open(req, proxy, type)


def default_transport_opener():
    """Return the opener used when the caller does not inject one."""
    return build_opener(_LoopbackDirectProxyHandler()).open


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


def _non_negative_int(
    value: object,
    *,
    field: str,
    label: str,
    error_factory: Callable[[str], ProviderResponseError],
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise error_factory(f"{label} usage.{field} must be a non-negative integer")
    return value


class OpenAICompatibleProvider:
    """Synchronous, non-streaming Chat Completions provider.

    Subclasses set :attr:`provider_name` and :attr:`label`, and override the
    ``_*_error`` factories so callers can catch provider-specific exceptions.
    """

    provider_name = "openai-compatible"
    label = "OpenAI-compatible"

    #: When false the provider talks to a keyless server and injects a
    #: placeholder token so the ``Authorization`` header stays well-formed.
    require_api_key = True
    #: Environment variables consulted for the credential, in order.
    api_key_env_names: tuple[str, ...] = ()
    #: Placeholder credential used when :attr:`require_api_key` is false.
    placeholder_api_key = "not-required"

    #: Request parameters this backend rejects; silently dropped before sending.
    unsupported_parameters: frozenset[str] = frozenset()
    #: Message fields this backend rejects; silently dropped before sending.
    unsupported_message_fields: frozenset[str] = frozenset()
    #: Remote APIs always report token counters; local runtimes often omit them.
    tolerate_missing_usage_counters = False
    #: Appended to transport failures to explain how to start a local backend.
    connection_hint = ""
    #: Appended when the backend answered too slowly instead of not at all.
    timeout_hint = ""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str,
        base_url: str,
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
        default_parameters: Mapping[str, Any] | None = None,
        opener: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] | None = None,
        request_observer=None,
    ) -> None:
        resolved_key = api_key or self._api_key_from_environment()
        if self.require_api_key and (not resolved_key or not resolved_key.strip()):
            raise self._configuration_error(
                self.missing_api_key_message
            )
        if not isinstance(model, str) or not model.strip():
            raise self._configuration_error("model must not be empty")
        if not isinstance(base_url, str) or not base_url.strip():
            raise self._configuration_error("base_url must not be empty")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise self._configuration_error("timeout must be positive")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise self._configuration_error("max_retries must be an integer")
        if max_retries < 0:
            raise self._configuration_error("max_retries must not be negative")
        if (
            isinstance(retry_backoff, bool)
            or not isinstance(retry_backoff, (int, float))
            or retry_backoff < 0
        ):
            raise self._configuration_error("retry_backoff must not be negative")

        defaults = dict(default_parameters or {})
        reserved = _RESERVED_PARAMETERS.intersection(defaults)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise self._configuration_error(f"reserved default parameter(s): {names}")

        self.request_observer = request_observer
        self._api_key = (
            resolved_key.strip() if resolved_key and resolved_key.strip()
            else self.placeholder_api_key
        )
        self._model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = max_retries
        self.retry_backoff = float(retry_backoff)
        self.default_parameters = defaults
        self._opener = opener or default_transport_opener()
        self._sleeper = sleeper or time.sleep

    # -- identity ---------------------------------------------------------

    @property
    def name(self) -> str:
        return self.provider_name

    @property
    def model(self) -> str:
        return self._model

    @property
    def chat_completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    @property
    def missing_api_key_message(self) -> str:
        names = " or ".join(self.api_key_env_names) or "the environment"
        return f"{self.label} API key is required; set {names} or pass api_key"

    # -- credential and error hooks ---------------------------------------

    def _api_key_from_environment(self) -> str | None:
        for name in self.api_key_env_names:
            value = os.environ.get(name)
            if value:
                return value
        return None

    def _configuration_error(self, message: str) -> ProviderConfigurationError:
        return ProviderConfigurationError(message)

    def _api_error(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> ProviderAPIError:
        return ProviderAPIError(
            message,
            status_code=status_code,
            retryable=retryable,
        )

    def _response_error(self, message: str) -> ProviderResponseError:
        return ProviderResponseError(message)

    # -- completion -------------------------------------------------------

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
        if self.unsupported_parameters:
            for name in sorted(self.unsupported_parameters):
                request_parameters.pop(name, None)
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
            [self._message_payload(message) for message in normalized_messages],
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
            raise ValueError(
                f"{self.label} request payload is not JSON serializable"
            ) from exc

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
            raise self._response_error(
                f"{self.label} response contains tool calls but no text content"
            )
        return turn.content

    def _message_payload(self, message: ChatMessage) -> dict[str, Any]:
        """Serialize one message, honouring backend-specific field limits."""
        payload = message.as_provider_payload()
        if self.unsupported_message_fields:
            for name in self.unsupported_message_fields:
                payload.pop(name, None)
        return payload

    # -- transport --------------------------------------------------------

    def _send(self, request: Request) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            if self.request_observer is not None:
                self.request_observer("request", {"attempt": attempt + 1, "url": request.full_url,
                                                  "body": json.loads(request.data)})
            try:
                with self._open_with_redirects(request) as response:
                    result = json.loads(response.read().decode("utf-8"))
            except HTTPError as error:
                if self.request_observer is not None:
                    self.request_observer("error", {"type": "HTTPError", "status": error.code, "attempt": attempt + 1})
                retryable = (
                    error.code in _RETRYABLE_STATUS_CODES or error.code >= 500
                )
                api_error = self._api_error(
                    f"{self.label} API returned HTTP {error.code}: {_error_message(error)}",
                    status_code=error.code,
                    retryable=retryable,
                )
                if retryable and attempt < self.max_retries:
                    self._wait_before_retry(attempt)
                    continue
                raise api_error from error
            except (URLError, TimeoutError, OSError) as error:
                if self.request_observer is not None:
                    self.request_observer("error", {"type": type(error).__name__, "message": str(error), "attempt": attempt + 1})
                reason = getattr(error, "reason", error)
                # A timeout means the server is reachable but slow, so advising
                # the operator to start it would be actively misleading.
                timed_out = isinstance(error, TimeoutError) or "timed out" in str(
                    reason
                ).lower()
                hint = self.timeout_hint if timed_out else self.connection_hint
                api_error = self._api_error(
                    f"{self.label} API request failed: {reason}{hint}",
                    retryable=True,
                )
                if attempt < self.max_retries:
                    self._wait_before_retry(attempt)
                    continue
                raise api_error from error
            except HTTPException as error:
                # IncompleteRead/BadStatusLine are not OSError subclasses, so an
                # interrupted response used to escape as a non-provider failure
                # and end the run instead of being retried as a transport fault.
                if self.request_observer is not None:
                    self.request_observer("error", {"type": type(error).__name__, "message": str(error), "attempt": attempt + 1})
                api_error = self._api_error(
                    f"{self.label} API returned a truncated or malformed HTTP "
                    f"response: {type(error).__name__}: {error}{self.connection_hint}",
                    retryable=True,
                )
                if attempt < self.max_retries:
                    self._wait_before_retry(attempt)
                    continue
                raise api_error from error
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                # The HTTP envelope failed to decode, which is a gateway or
                # transport fault. The model's own tool-argument JSON is a
                # separate contract and stays non-retryable in _parse_tool_calls.
                if self.request_observer is not None:
                    self.request_observer("error", {"type": type(error).__name__, "attempt": attempt + 1})
                api_error = self._api_error(
                    f"{self.label} API returned an undecodable response envelope",
                    retryable=True,
                )
                if attempt < self.max_retries:
                    self._wait_before_retry(attempt)
                    continue
                raise api_error from error

            if self.request_observer is not None:
                self.request_observer("response", {"attempt": attempt + 1, "body": result})
            if not isinstance(result, dict):
                raise self._response_error(
                    f"{self.label} API response must be a JSON object"
                )
            return result
        raise AssertionError("retry loop exhausted without returning or raising")

    def _open_with_redirects(self, request):
        """Replay the POST across bounded HTTP 307/308 redirects.

        A redirect to another origin is still followed, but the ``Authorization``
        header is dropped first so the credential never reaches a host the caller
        did not configure. Plaintext downgrades and non-HTTP(S) targets are
        refused, and every refusal is recorded with its reason so a transport
        failure can be diagnosed from the audit trail alone.
        """
        origin = _origin(request.full_url)
        for redirect_count in range(_MAX_REDIRECTS + 1):
            try:
                return self._opener(request, timeout=self.timeout)
            except HTTPError as error:
                if error.code not in _REDIRECT_STATUS_CODES:
                    raise
                location = error.headers.get("Location") if error.headers else None
                target = urljoin(request.full_url, location or "")
                target_origin = _origin(target)
                reason = self._redirect_refusal(
                    request.full_url, location, target, target_origin, redirect_count
                )
                if reason is not None:
                    self._observe_transport("redirect_refused", {
                        "status": error.code, "from": request.full_url,
                        "to": target or None, "reason": reason})
                    error.close()
                    raise
                headers = dict(request.header_items())
                cross_origin = target_origin != origin
                dropped = cross_origin and any(
                    name.lower() == "authorization" for name in headers)
                if dropped:
                    headers = {name: value for name, value in headers.items()
                               if name.lower() != "authorization"}
                self._observe_transport("redirect", {
                    "status": error.code, "from": request.full_url, "to": target,
                    "cross_origin": cross_origin, "credentials_dropped": dropped})
                error.close()
                request = Request(target, data=request.data, headers=headers,
                                  method=request.get_method())
        raise AssertionError("redirect loop exhausted")

    @staticmethod
    def _redirect_refusal(
        current_url: str,
        location: str | None,
        target: str,
        target_origin: tuple[str, str, int] | None,
        redirect_count: int,
    ) -> str | None:
        """Return why a redirect must not be followed, or None to follow it."""
        if not location:
            return "missing_location"
        if redirect_count >= _MAX_REDIRECTS:
            return "redirect_limit"
        if target_origin is None:
            return "unsupported_scheme"
        parsed = urlsplit(target)
        if parsed.username or parsed.password:
            return "credentials_in_location"
        scheme, host, _ = target_origin
        source_origin = _origin(current_url)
        if source_origin is None:
            return "unsupported_scheme"
        if scheme == "http" and source_origin[0] == "https":
            return "plaintext_downgrade"
        if (
            target_origin != source_origin
            and scheme == "http"
            and not _is_loopback(host)
        ):
            return "cross_origin_plaintext"
        return None

    def _observe_transport(self, kind: str, payload: Mapping[str, Any]) -> None:
        if self.request_observer is not None:
            self.request_observer(kind, dict(payload))

    def _wait_before_retry(self, attempt: int) -> None:
        delay = self.retry_backoff * (2**attempt)
        if delay > 0:
            self._sleeper(delay)

    # -- validation -------------------------------------------------------

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

    # -- response normalization ------------------------------------------

    def _parse_completion(self, response: Mapping[str, Any]) -> AssistantTurn:
        label = self.label
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise self._response_error(f"{label} response is missing choices")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise self._response_error(f"{label} choices[0] must be an object")
        finish_reason = choice.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise self._response_error(
                f"{label} response is missing choices[0].finish_reason"
            )
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise self._response_error(
                f"{label} response is missing choices[0].message"
            )

        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise self._response_error(f"{label} message.content must be a string or null")
        reasoning_content = message.get("reasoning_content")
        if reasoning_content is not None and not isinstance(reasoning_content, str):
            raise self._response_error(
                f"{label} message.reasoning_content must be a string or null"
            )
        tool_calls = self._parse_tool_calls(message.get("tool_calls"))
        if finish_reason == "tool_calls" and not tool_calls:
            raise self._response_error(
                f"{label} finish_reason is tool_calls but no tool calls were returned"
            )
        if content is None and not tool_calls:
            raise self._response_error(
                f"{label} response contains neither content nor tool calls"
            )

        response_model = response.get("model")
        if not isinstance(response_model, str) or not response_model:
            response_model = self.model
        response_id = response.get("id")
        if response_id is not None and not isinstance(response_id, str):
            raise self._response_error(f"{label} response.id must be a string")
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
            raise self._response_error(
                f"{label} response failed normalized schema validation: {exc}"
            ) from exc

    def _parse_tool_calls(self, value: object) -> list[ToolCall]:
        label = self.label
        if value is None:
            return []
        if not isinstance(value, list):
            raise self._response_error(f"{label} message.tool_calls must be an array")
        calls = []
        seen_ids: set[str] = set()
        for index, raw_call in enumerate(value):
            if not isinstance(raw_call, Mapping):
                raise self._response_error(
                    f"{label} tool_calls[{index}] must be an object"
                )
            if raw_call.get("type") != "function":
                raise self._response_error(
                    f"{label} tool_calls[{index}].type must be function"
                )
            call_id = raw_call.get("id")
            function = raw_call.get("function")
            if not isinstance(call_id, str) or not call_id:
                raise self._response_error(
                    f"{label} tool_calls[{index}].id must be a non-empty string"
                )
            if call_id in seen_ids:
                raise self._response_error(f"duplicate {label} tool call ID: {call_id}")
            seen_ids.add(call_id)
            if not isinstance(function, Mapping):
                raise self._response_error(
                    f"{label} tool_calls[{index}].function must be an object"
                )
            name = function.get("name")
            raw_arguments = function.get("arguments")
            if not isinstance(name, str) or not name:
                raise self._response_error(
                    f"{label} tool_calls[{index}].function.name is invalid"
                )
            # Some local runtimes return arguments as an already-decoded object.
            if isinstance(raw_arguments, Mapping):
                arguments = dict(raw_arguments)
                raw_arguments = json.dumps(arguments, ensure_ascii=False)
            elif not isinstance(raw_arguments, str) or not raw_arguments:
                raise self._response_error(
                    f"{label} tool_calls[{index}].function.arguments is invalid"
                )
            else:
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    raise self._response_error(
                        f"{label} tool {name} returned invalid JSON arguments"
                    ) from exc
            if not isinstance(arguments, dict):
                raise self._response_error(
                    f"{label} tool {name} arguments must decode to an object"
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
                raise self._response_error(
                    f"{label} tool_calls[{index}] failed validation: {exc}"
                ) from exc
        return calls

    def _parse_usage(self, value: object) -> TokenUsage | None:
        label = self.label
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise self._response_error(f"{label} usage must be an object")
        details = value.get("completion_tokens_details")
        if details is not None and not isinstance(details, Mapping):
            raise self._response_error(
                f"{label} usage.completion_tokens_details must be an object"
            )
        cached = value.get("prompt_cache_hit_tokens")
        cache_miss = value.get("prompt_cache_miss_tokens")
        reasoning = details.get("reasoning_tokens") if details else None
        # Local runtimes commonly omit counters; treat missing as zero while
        # still rejecting malformed values.
        def counter(raw: object, *, field: str) -> int:
            if raw is None and self.tolerate_missing_usage_counters:
                return 0
            return _non_negative_int(
                raw, field=field, label=label,
                error_factory=self._response_error,
            )

        return TokenUsage(
            input_tokens=counter(value.get("prompt_tokens"), field="prompt_tokens"),
            output_tokens=counter(
                value.get("completion_tokens"),
                field="completion_tokens",
            ),
            total_tokens=counter(value.get("total_tokens"), field="total_tokens"),
            cached_input_tokens=(
                None
                if cached is None
                else _non_negative_int(
                    cached, field="prompt_cache_hit_tokens", label=label,
                    error_factory=self._response_error,
                )
            ),
            cache_miss_input_tokens=(
                None
                if cache_miss is None
                else _non_negative_int(
                    cache_miss,
                    field="prompt_cache_miss_tokens",
                    label=label,
                    error_factory=self._response_error,
                )
            ),
            reasoning_tokens=(
                None
                if reasoning is None
                else _non_negative_int(
                    reasoning, field="reasoning_tokens", label=label,
                    error_factory=self._response_error,
                )
            ),
        )
