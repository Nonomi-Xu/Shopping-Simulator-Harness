from __future__ import annotations

import json
import os
import unittest
from urllib.error import URLError
from unittest.mock import patch

from src.harness.api.deepseek import (
    DeepSeekClient,
    DeepSeekConfigurationError,
    call_deepseek,
)
from src.harness.providers import (
    ChatMessage,
    DeepSeekProvider,
    DeepSeekResponseError,
    ToolDefinition,
)
from src.harness.schemas import SearchProductsAction


class _Response:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return self._body


def _text_response(content: str = "hello") -> dict:
    return {
        "id": "response-1",
        "model": "deepseek-v4-flash",
        "created": 123,
        "object": "chat.completion",
        "system_fingerprint": "fp-test",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15,
            "prompt_cache_hit_tokens": 5,
            "prompt_cache_miss_tokens": 7,
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }


class DeepSeekProviderTests(unittest.TestCase):
    def test_complete_sends_request_and_normalizes_text_usage(self) -> None:
        captured = {}

        def opener(request, *, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _Response(_text_response())

        provider = DeepSeekProvider(
            api_key="test-key",
            timeout=12,
            opener=opener,
        )
        turn = provider.complete(
            [ChatMessage(role="user", content="Hi")],
            parameters={"max_tokens": 32, "temperature": 0},
        )

        self.assertEqual(turn.content, "hello")
        self.assertEqual(turn.finish_reason, "stop")
        self.assertEqual(turn.response_id, "response-1")
        self.assertEqual(turn.usage.input_tokens, 12)
        self.assertEqual(turn.usage.output_tokens, 3)
        self.assertEqual(turn.usage.cached_input_tokens, 5)
        self.assertEqual(turn.usage.reasoning_tokens, 2)
        self.assertEqual(
            captured["url"],
            "https://api.deepseek.com/chat/completions",
        )
        self.assertEqual(captured["authorization"], "Bearer test-key")
        self.assertEqual(captured["payload"]["messages"][0]["content"], "Hi")
        self.assertEqual(captured["payload"]["max_tokens"], 32)
        self.assertEqual(captured["payload"]["temperature"], 0)
        self.assertFalse(captured["payload"]["stream"])
        self.assertEqual(captured["timeout"], 12.0)

    def test_complete_parses_tool_call_and_exposes_generated_schema(self) -> None:
        captured = {}

        def opener(request, *, timeout):
            captured.update(json.loads(request.data.decode("utf-8")))
            return _Response(
                {
                    "id": "response-tools",
                    "model": "deepseek-v4-flash",
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning_content": "Need catalog evidence.",
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "search_products",
                                            "arguments": (
                                                '{"query":"降噪耳机","limit":10}'
                                            ),
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 8,
                        "total_tokens": 28,
                    },
                }
            )

        tool = ToolDefinition.from_action_model(
            name="search_products",
            description="Search the public product catalog.",
            action_model=SearchProductsAction,
        )
        provider = DeepSeekProvider(api_key="test-key", opener=opener)
        turn = provider.complete(
            [ChatMessage(role="user", content="推荐降噪耳机")],
            tools=[tool],
            tool_choice="auto",
        )

        self.assertEqual(turn.finish_reason, "tool_calls")
        self.assertEqual(turn.reasoning_content, "Need catalog evidence.")
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0].id, "call-1")
        self.assertEqual(turn.tool_calls[0].name, "search_products")
        self.assertEqual(
            turn.tool_calls[0].arguments,
            {"query": "降噪耳机", "limit": 10},
        )
        self.assertEqual(captured["tool_choice"], "auto")
        function = captured["tools"][0]["function"]
        self.assertEqual(function["name"], "search_products")
        self.assertEqual(function["parameters"]["type"], "object")
        self.assertFalse(function["parameters"]["additionalProperties"])

        replay = turn.as_message().as_provider_payload()
        self.assertEqual(replay["tool_calls"][0]["id"], "call-1")
        self.assertEqual(
            replay["tool_calls"][0]["function"]["arguments"],
            '{"query":"降噪耳机","limit":10}',
        )

    def test_invalid_tool_argument_json_is_a_response_error(self) -> None:
        def opener(request, *, timeout):
            return _Response(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-bad",
                                        "type": "function",
                                        "function": {
                                            "name": "search_products",
                                            "arguments": "{not-json",
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                }
            )

        provider = DeepSeekProvider(api_key="test-key", opener=opener)
        with self.assertRaisesRegex(
            DeepSeekResponseError,
            "invalid JSON arguments",
        ):
            provider.complete([ChatMessage(role="user", content="search")])

    def test_transient_network_error_retries_with_bounded_backoff(self) -> None:
        attempts = []
        delays = []

        def opener(request, *, timeout):
            attempts.append(request.full_url)
            if len(attempts) == 1:
                raise URLError("temporary outage")
            return _Response(_text_response("recovered"))

        provider = DeepSeekProvider(
            api_key="test-key",
            opener=opener,
            sleeper=delays.append,
            max_retries=1,
            retry_backoff=0.25,
        )
        turn = provider.complete([ChatMessage(role="user", content="Hi")])

        self.assertEqual(turn.content, "recovered")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(delays, [0.25])

    def test_missing_api_key_has_clear_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                DeepSeekConfigurationError,
                "DEEPSEEK_API_KEY",
            ):
                DeepSeekProvider()

    def test_legacy_client_and_single_turn_helper_remain_available(self) -> None:
        self.assertIs(DeepSeekClient, DeepSeekProvider)
        with patch.object(
            DeepSeekProvider,
            "chat",
            return_value="ok",
        ) as chat:
            result = call_deepseek(
                "question",
                system_prompt="instructions",
                api_key="test-key",
            )

        self.assertEqual(result, "ok")
        messages = chat.call_args.args[0]
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "instructions"},
                {"role": "user", "content": "question"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
