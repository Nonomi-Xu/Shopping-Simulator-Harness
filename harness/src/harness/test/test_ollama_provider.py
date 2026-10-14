from __future__ import annotations

import json
import os
import unittest
from argparse import Namespace
from unittest.mock import patch

from src.harness.cli import _build_provider, _resolve_provider_defaults
from src.harness.providers import (
    ChatMessage,
    DeepSeekProvider,
    ModelProvider,
    OllamaAPIError,
    OllamaConfigurationError,
    OllamaProvider,
    build_provider,
    list_local_models,
    provider_class,
)


class _Response:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return self._body


def _tool_response() -> dict:
    """Shaped like Ollama's OpenAI-compatible reply: no ``id``, partial usage."""
    return {
        "model": "qwen2.5:7b-instruct",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_products",
                                "arguments": '{"query":"降噪耳机"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 31, "completion_tokens": 9, "total_tokens": 40},
    }


class OllamaProviderTests(unittest.TestCase):
    def test_runs_without_any_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            provider = OllamaProvider()

        self.assertIsInstance(provider, ModelProvider)
        self.assertEqual(provider.name, "ollama")
        self.assertEqual(provider.model, "qwen2.5:7b-instruct")
        self.assertEqual(provider.base_url, "http://localhost:11434/v1")
        self.assertEqual(
            provider.chat_completions_url,
            "http://localhost:11434/v1/chat/completions",
        )

    def test_sends_placeholder_authorization_to_the_local_server(self) -> None:
        captured = {}

        def opener(request, *, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            return _Response(_tool_response())

        provider = OllamaProvider(opener=opener)
        provider.complete([ChatMessage(role="user", content="推荐耳机")])

        self.assertEqual(
            captured["url"],
            "http://localhost:11434/v1/chat/completions",
        )
        # A local server ignores the token, but some clients require the header.
        self.assertEqual(captured["authorization"], "Bearer ollama")

    def test_deepseek_only_thinking_parameter_is_dropped(self) -> None:
        captured = {}

        def opener(request, *, timeout):
            captured.update(json.loads(request.data.decode("utf-8")))
            return _Response(_tool_response())

        provider = OllamaProvider(
            opener=opener,
            default_parameters={"temperature": 0, "thinking": {"type": "disabled"}},
        )
        provider.complete(
            [ChatMessage(role="user", content="推荐耳机")],
            parameters={"thinking": {"type": "enabled"}},
        )

        self.assertNotIn("thinking", captured)
        self.assertEqual(captured["temperature"], 0)
        self.assertFalse(captured["stream"])

    def test_thinking_controls_reach_the_server(self) -> None:
        """Qwen3's thinking mode is driven by these fields; do not strip them."""
        captured = {}

        def opener(request, *, timeout):
            captured.update(json.loads(request.data.decode("utf-8")))
            return _Response(_tool_response())

        provider = OllamaProvider(opener=opener)
        provider.complete(
            [ChatMessage(role="user", content="x")],
            parameters={"reasoning_effort": "none", "think": False},
        )

        self.assertEqual(captured["reasoning_effort"], "none")
        self.assertIs(captured["think"], False)

    def test_parses_tool_call_without_response_id_or_full_usage(self) -> None:
        provider = OllamaProvider(opener=lambda request, *, timeout: _Response(_tool_response()))
        turn = provider.complete([ChatMessage(role="user", content="推荐耳机")])

        self.assertEqual(turn.provider, "ollama")
        self.assertEqual(turn.finish_reason, "tool_calls")
        self.assertIsNone(turn.response_id)
        self.assertEqual(turn.tool_calls[0].name, "search_products")
        self.assertEqual(turn.tool_calls[0].arguments, {"query": "降噪耳机"})
        self.assertEqual(turn.usage.total_tokens, 40)

    def test_missing_usage_counters_are_zero_for_local_backends(self) -> None:
        payload = _tool_response()
        payload["usage"] = {}
        provider = OllamaProvider(opener=lambda request, *, timeout: _Response(payload))
        turn = provider.complete([ChatMessage(role="user", content="x")])

        self.assertEqual(turn.usage.total_tokens, 0)
        self.assertEqual(turn.usage.input_tokens, 0)

    def test_deepseek_still_rejects_missing_usage_counters(self) -> None:
        from src.harness.providers import DeepSeekResponseError

        payload = _tool_response()
        payload["usage"] = {}
        provider = DeepSeekProvider(
            api_key="test-key",
            opener=lambda request, *, timeout: _Response(payload),
        )
        # A remote API that omits counters is a contract violation, not a zero.
        with self.assertRaises(DeepSeekResponseError):
            provider.complete([ChatMessage(role="user", content="x")])

    def test_connection_failure_explains_how_to_start_the_server(self) -> None:
        def opener(request, *, timeout):
            raise OSError("Connection refused")

        provider = OllamaProvider(opener=opener, max_retries=0, sleeper=lambda _: None)
        with self.assertRaises(OllamaAPIError) as caught:
            provider.complete([ChatMessage(role="user", content="x")])

        self.assertIn("ollama serve", str(caught.exception))

    def test_defaults_are_slower_than_the_hosted_api(self) -> None:
        self.assertGreater(OllamaProvider().timeout, DeepSeekProvider(api_key="k").timeout)

    def test_list_local_models_reports_an_unreachable_server(self) -> None:
        def unreachable(request, *, timeout):
            raise OSError("Connection refused")

        with patch(
            "src.harness.providers.ollama.default_transport_opener",
            return_value=unreachable,
        ):
            with self.assertRaises(OllamaConfigurationError) as caught:
                list_local_models()

        self.assertIn("ollama serve", str(caught.exception))


class ProviderFactoryTests(unittest.TestCase):
    def test_provider_class_maps_both_backends(self) -> None:
        self.assertIs(provider_class("ollama"), OllamaProvider)
        self.assertIs(provider_class("deepseek"), DeepSeekProvider)
        with self.assertRaises(ValueError):
            provider_class("scripted")

    def test_build_provider_fills_backend_defaults(self) -> None:
        local = build_provider("ollama")
        self.assertEqual(local.base_url, "http://localhost:11434/v1")
        self.assertEqual(local.timeout, 120.0)

        remote = build_provider("deepseek", api_key="test-key")
        self.assertEqual(remote.base_url, "https://api.deepseek.com")
        self.assertEqual(remote.timeout, 60.0)

    def test_build_provider_honours_explicit_overrides(self) -> None:
        local = build_provider(
            "ollama",
            model="qwen3:4b",
            base_url="http://127.0.0.1:9999/v1",
            timeout=7.0,
        )
        self.assertEqual(local.model, "qwen3:4b")
        self.assertEqual(local.base_url, "http://127.0.0.1:9999/v1")
        self.assertEqual(local.timeout, 7.0)


class ProviderCliWiringTests(unittest.TestCase):
    def _args(self, provider: str) -> Namespace:
        return Namespace(provider=provider, model=None, base_url=None, timeout=None)

    def test_ollama_selection_resolves_local_defaults(self) -> None:
        args = self._args("ollama")
        _resolve_provider_defaults(args)

        self.assertEqual(args.model, "qwen2.5:7b-instruct")
        self.assertEqual(args.base_url, "http://localhost:11434/v1")
        self.assertEqual(args.timeout, 120.0)

    def test_deepseek_remains_the_default_wiring(self) -> None:
        args = self._args("deepseek")
        _resolve_provider_defaults(args)

        self.assertEqual(args.model, "deepseek-v4-flash")
        self.assertEqual(args.base_url, "https://api.deepseek.com")
        self.assertEqual(args.timeout, 60.0)

    def test_explicit_flags_survive_resolution(self) -> None:
        args = Namespace(
            provider="ollama",
            model="qwen3:4b",
            base_url="http://127.0.0.1:11434/v1",
            timeout=5.0,
        )
        _resolve_provider_defaults(args)

        self.assertEqual(args.model, "qwen3:4b")
        self.assertEqual(args.base_url, "http://127.0.0.1:11434/v1")
        self.assertEqual(args.timeout, 5.0)

    def test_checkpoint_restored_unknown_provider_does_not_crash(self) -> None:
        # A checkpoint may name a provider injected through the Python API.
        args = self._args("scripted")
        _resolve_provider_defaults(args)

        self.assertEqual(args.model, "deepseek-v4-flash")
        self.assertEqual(args.base_url, "https://api.deepseek.com")

    def test_build_provider_selects_the_ollama_class(self) -> None:
        args = Namespace(
            provider="ollama",
            model="qwen2.5:7b-instruct",
            base_url="http://localhost:11434/v1",
            timeout=120.0,
            max_retries=2,
            retry_backoff=0.5,
            provider_parameters=None,
            temperature=0.0,
        )
        provider = _build_provider(args)

        self.assertIsInstance(provider, OllamaProvider)
        self.assertNotIsInstance(provider, DeepSeekProvider)


if __name__ == "__main__":
    unittest.main()
