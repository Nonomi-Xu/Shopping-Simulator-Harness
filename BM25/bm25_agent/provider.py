"""One-call, tool-constrained selectors for five frozen BM25 candidates.

Both the paid DeepSeek API and a local Ollama runtime speak the OpenAI Chat
Completions protocol, so the request/parse logic lives in
:class:`ChatCompletionsSelector` and the subclasses only change identity,
credentials, and backend-specific payload fields.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.request import Request

from .types import AgentDecision, Candidate, SelectorResult, TokenUsage


# Use the explicit OpenAI-compatible /v1 endpoint.  urllib intentionally does
# not replay POST bodies across HTTP 307 redirects, so the bare host can turn
# every request into a non-retryable redirect instead of a model response.
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-v4-flash"

# Local runtime defaults; Ollama serves an OpenAI-compatible API on localhost.
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:7b-instruct"


class SelectorError(RuntimeError):
    pass


@runtime_checkable
class CandidateSelector(Protocol):
    def select(self, query: str, candidates: Sequence[Candidate]) -> SelectorResult: ...


class ChatCompletionsSelector:
    """Give the model no tools except one terminal Top-5 selection call."""

    #: Reported in :class:`SelectorResult` and used in error messages.
    provider_name = "deepseek"
    label = "DeepSeek"
    #: Local servers do not authenticate.
    require_api_key = True
    #: Environment variables consulted for the credential, in order.
    api_key_env_names: tuple[str, ...] = ("DEEPSEEK_API_KEY", "DEEPSEEK_API")
    #: Backend-specific request fields sent alongside the tool call.
    extra_payload: Mapping[str, Any] = {"thinking": {"type": "disabled"}}
    #: Whether the backend honours a forced ``tool_choice``.
    supports_forced_tool_choice = True

    #: Frozen prompt text. Subclasses may extend it for weaker local models,
    #: but the DeepSeek wording stays byte-identical so historical runs remain
    #: comparable.
    tool_description = (
        "Select exactly one product from the supplied BM25 Top-5 and "
        "choose only available option values shown for that product."
    )
    system_prompt = (
        "你是购物推荐决策 Agent。BM25 检索已经结束。你只能比较用户消息中"
        "给出的候选，必须选择恰好一个候选 ASIN，并仅从该商品明确列出的可用"
        "规格值中选择。不得搜索、虚构商品或规格。仔细匹配用途、属性、品牌、"
        "型号、预算和规格；BM25 排名只是参考。最后只调用 select_product。"
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str,
        base_url: str,
        timeout: float = 60.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        opener: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        resolved_key = api_key or self._api_key_from_environment()
        if self.require_api_key and (not resolved_key or not resolved_key.strip()):
            names = " or ".join(self.api_key_env_names)
            raise SelectorError(f"set {names} before running Agent evaluation")
        if not model.strip() or not base_url.strip():
            raise ValueError("model and base_url must not be empty")
        if timeout <= 0 or max_retries < 0:
            raise ValueError("timeout must be positive and max_retries non-negative")
        self.api_key = resolved_key.strip() if resolved_key else "not-required"
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.temperature = float(temperature)
        self._opener = opener
        self._curl_path = shutil.which("curl") if opener is None else None
        if opener is None and self._curl_path is None:
            raise SelectorError("curl is required for verified HTTPS transport")
        self._sleeper = sleeper or time.sleep

    def _api_key_from_environment(self) -> str | None:
        for name in self.api_key_env_names:
            value = os.environ.get(name)
            if value:
                return value
        return None

    @property
    def url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def select(self, query: str, candidates: Sequence[Candidate]) -> SelectorResult:
        if not candidates:
            raise SelectorError("BM25 returned no candidates")
        if len(candidates) > 5:
            raise SelectorError("selector received more than five candidates")
        candidate_asins = [candidate.asin for candidate in candidates]
        tool = {
            "type": "function",
            "function": {
                "name": "select_product",
                "description": self.tool_description,
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "asin": {"type": "string", "enum": candidate_asins},
                        "selected_options": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                        "reason": {"type": "string", "minLength": 1},
                    },
                    "required": ["asin", "selected_options", "reason"],
                },
            },
        }
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_query": query,
                        "bm25_top5": [candidate.agent_payload() for candidate in candidates],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": [tool],
        }
        if self.supports_forced_tool_choice:
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": "select_product"},
            }
        # DeepSeek V4 defaults to thinking mode, which rejects forced function
        # selection.  This benchmark needs a single auditable terminal call, so
        # use non-thinking mode explicitly where the backend understands it.
        payload.update(self.extra_payload)
        payload["temperature"] = self.temperature
        payload["stream"] = False
        response = self._request(payload)
        decision = self._parse_decision(response)
        usage = self._parse_usage(response.get("usage"))
        return SelectorResult(
            decision=decision,
            provider=self.provider_name,
            model=self.model,
            usage=usage,
            response_id=str(response.get("id")) if response.get("id") else None,
        )

    def _request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._opener is None:
            return self._curl_request(payload)
        return self._urllib_request(payload)

    def _curl_request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Use the system trust store without exposing the API key in argv."""
        assert self._curl_path is not None
        with tempfile.TemporaryDirectory(prefix="bm25-agent-") as directory:
            temp_dir = Path(directory)
            headers_path = temp_dir / "headers.txt"
            payload_path = temp_dir / "payload.json"
            headers_path.write_text(
                "\n".join(
                    (
                        f"Authorization: Bearer {self.api_key}",
                        "Content-Type: application/json",
                        "Accept: application/json",
                        "User-Agent: shopping-bm25-agent/0.1",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(headers_path, 0o600)
            payload_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            os.chmod(payload_path, 0o600)
            command = [
                self._curl_path,
                "--silent",
                "--show-error",
                "--location",
                "--max-redirs",
                "5",
                "--fail-with-body",
                "--request",
                "POST",
                "--connect-timeout",
                str(min(self.timeout, 30.0)),
                "--max-time",
                str(self.timeout),
                "--retry",
                str(self.max_retries),
                "--retry-all-errors",
                "--retry-delay",
                "1",
                "--header",
                f"@{headers_path}",
                "--data-binary",
                f"@{payload_path}",
                self.url,
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    timeout=self.timeout * (self.max_retries + 1) + 15,
                )
            except subprocess.TimeoutExpired as exc:
                raise SelectorError("provider request timed out") from exc

        stdout = completed.stdout.decode("utf-8", errors="replace")
        if completed.returncode != 0:
            detail = stdout.strip() or completed.stderr.decode(
                "utf-8", errors="replace"
            ).strip()
            raise SelectorError(
                f"provider transport failed (curl {completed.returncode}): {detail[:500]}"
            )
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise SelectorError("provider returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise SelectorError("provider response must be a JSON object")
        return result

    def _urllib_request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Injectable transport retained for deterministic unit tests."""
        request = Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "shopping-bm25-agent/0.1",
            },
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                assert self._opener is not None
                with self._opener(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                if not isinstance(result, dict):
                    raise SelectorError("provider response must be a JSON object")
                return result
            except HTTPError as exc:
                retryable = exc.code in {408, 409, 429} or exc.code >= 500
                if retryable and attempt < self.max_retries:
                    self._sleeper(0.5 * (2**attempt))
                    continue
                location = exc.headers.get("Location")
                redirect = f" (Location: {location})" if location else ""
                raise SelectorError(
                    f"provider returned HTTP {exc.code}{redirect}"
                ) from exc
            except (URLError, TimeoutError, OSError) as exc:
                if attempt < self.max_retries:
                    self._sleeper(0.5 * (2**attempt))
                    continue
                raise SelectorError(f"provider request failed: {exc}") from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SelectorError("provider returned invalid JSON") from exc
        raise AssertionError("retry loop exhausted")

    @staticmethod
    def _parse_decision(response: Mapping[str, Any]) -> AgentDecision:
        try:
            message = response["choices"][0]["message"]
            calls = message["tool_calls"]
            if not isinstance(calls, list) or len(calls) != 1:
                raise SelectorError("Agent must emit exactly one tool call")
            function = calls[0]["function"]
            if function.get("name") != "select_product":
                raise SelectorError("Agent called an unsupported tool")
            arguments = function["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise SelectorError("tool arguments must be an object")
            if set(arguments) != {"asin", "selected_options", "reason"}:
                raise SelectorError("tool arguments contain missing or extra fields")
            selected_options = arguments["selected_options"]
            if not isinstance(selected_options, dict) or not all(
                isinstance(axis, str) and isinstance(value, str)
                for axis, value in selected_options.items()
            ):
                raise SelectorError("selected_options must map strings to strings")
            if not isinstance(arguments["asin"], str) or not isinstance(
                arguments["reason"], str
            ):
                raise SelectorError("asin and reason must be strings")
            return AgentDecision(
                asin=arguments["asin"].strip(),
                selected_options={
                    axis.strip(): value.strip() for axis, value in selected_options.items()
                },
                reason=arguments["reason"].strip(),
            )
        except SelectorError:
            raise
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise SelectorError("provider response has no valid select_product call") from exc

    @staticmethod
    def _parse_usage(value: object) -> TokenUsage | None:
        if not isinstance(value, Mapping):
            return None
        try:
            return TokenUsage(
                input_tokens=int(value.get("prompt_tokens", 0)),
                output_tokens=int(value.get("completion_tokens", 0)),
                total_tokens=int(value.get("total_tokens", 0)),
            )
        except (TypeError, ValueError):
            return None


class DeepSeekSelector(ChatCompletionsSelector):
    """Paid remote selector; the historical default behaviour."""

    provider_name = "deepseek"
    label = "DeepSeek"
    require_api_key = True
    api_key_env_names = ("DEEPSEEK_API_KEY", "DEEPSEEK_API")
    extra_payload = {"thinking": {"type": "disabled"}}
    supports_forced_tool_choice = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        opener: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
            opener=opener,
            sleeper=sleeper,
        )


class OllamaSelector(ChatCompletionsSelector):
    """Local selector backed by an Ollama server; no per-token cost."""

    provider_name = "ollama"
    label = "Ollama"
    require_api_key = False
    api_key_env_names = ("OLLAMA_API_KEY",)
    # Selecting one product from five fixed candidates needs format compliance,
    # not deliberation.  Measured on qwen3:8b: default thinking spent 662
    # completion tokens on a trivial prompt, while ``reasoning_effort: none``
    # spent 4.  Ollama ignores this field on non-reasoning models (verified on
    # qwen2.5), so it is a safe default for every local model.
    extra_payload: Mapping[str, Any] = {"reasoning_effort": "none"}
    # Ollama honours a forced function choice ("required" and named-function
    # forms both verified), so the single terminal call stays deterministic.
    supports_forced_tool_choice = True
    # A 7B local model mirrors the nested customization_options structure it is
    # shown, so spell out the flattened shape it must produce. DeepSeek does not
    # need this, and its prompt is deliberately left untouched.
    tool_description = (
        "Select exactly one product from the supplied BM25 Top-5 and "
        "choose only available option values shown for that product. "
        "selected_options must map every option name to exactly ONE "
        "string, copied from that option's available 'value' entries; "
        "never copy the surrounding object or list. Example: "
        '{"asin":"A001","selected_options":{"颜色":"白色"},'
        '"reason":"匹配需求"}'
    )
    system_prompt = (
        "你是购物推荐决策 Agent。BM25 检索已经结束。你只能比较用户消息中"
        "给出的候选，必须选择恰好一个候选 ASIN，并仅从该商品明确列出的可用"
        "规格值中选择。不得搜索、虚构商品或规格。仔细匹配用途、属性、品牌、"
        "型号、预算和规格；BM25 排名只是参考。"
        "候选的 customization_options 是一个对象：键是规格名（例如"
        "「规格描述」），值是该规格下的候选列表，列表元素形如"
        "{\"value\":\"海军蓝700毫升\",\"is_available\":true}。"
        "selected_options 的键必须用规格名本身，值必须用元素里的 value 字符串，"
        "例如 {\"规格描述\":\"海军蓝700毫升\"}；"
        "不要把 value/is_available/price/asin 这些内层字段当作规格名，"
        "也不要输出数组或对象。最后只调用 select_product。"
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout: float = 120.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        opener: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
            opener=opener,
            sleeper=sleeper,
        )
