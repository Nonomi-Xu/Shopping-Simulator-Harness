from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from bm25_agent.provider import (
    DEFAULT_BASE_URL,
    OLLAMA_BASE_URL,
    ChatCompletionsSelector,
    DeepSeekSelector,
    OllamaSelector,
    SelectorError,
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


def _selection_response() -> dict:
    return {
        "model": "qwen2.5:7b-instruct",
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "select_product",
                                "arguments": json.dumps(
                                    {
                                        "asin": "A001",
                                        "selected_options": {"颜色": "白色"},
                                        "reason": "匹配需求",
                                    },
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    ]
                }
            }
        ],
    }


def _candidate():
    from bm25_agent.types import Candidate

    return Candidate(
        rank=1,
        bm25_score=2.5,
        product={
            "asin": "A001",
            "title": "测试商品",
            "brand": "品牌",
            "category": "类目",
            "price_min": 10.0,
            "price_max": 10.0,
            "customization_options": {"颜色": ["白色"]},
        },
    )


class ProviderParsingTest(unittest.TestCase):
    def test_default_url_uses_non_redirecting_v1_endpoint(self) -> None:
        self.assertEqual(DEFAULT_BASE_URL, "https://api.deepseek.com/v1")


    def test_parses_one_terminal_selection_call(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "select_product",
                                    "arguments": json.dumps(
                                        {
                                            "asin": "A001",
                                            "selected_options": {"颜色": "白色"},
                                            "reason": "匹配需求",
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            }
                        ]
                    }
                }
            ]
        }
        decision = DeepSeekSelector._parse_decision(response)
        self.assertEqual(decision.asin, "A001")
        self.assertEqual(decision.selected_options, {"颜色": "白色"})

    def test_rejects_prose_without_tool_call(self) -> None:
        with self.assertRaises(SelectorError):
            DeepSeekSelector._parse_decision(
                {"choices": [{"message": {"content": "推荐 A001"}}]}
            )


class OllamaSelectorTest(unittest.TestCase):
    def test_runs_without_a_credential_and_targets_localhost(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            selector = OllamaSelector()

        self.assertIsInstance(selector, ChatCompletionsSelector)
        self.assertEqual(selector.base_url, OLLAMA_BASE_URL)
        self.assertEqual(selector.url, f"{OLLAMA_BASE_URL}/chat/completions")
        self.assertEqual(selector.provider_name, "ollama")

    def test_deepseek_prompt_is_frozen_for_historical_comparability(self) -> None:
        # The documented baseline was measured with this exact wording; any
        # change to it invalidates comparison against those results.
        self.assertEqual(
            DeepSeekSelector.system_prompt,
            "你是购物推荐决策 Agent。BM25 检索已经结束。你只能比较用户消息中"
            "给出的候选，必须选择恰好一个候选 ASIN，并仅从该商品明确列出的可用"
            "规格值中选择。不得搜索、虚构商品或规格。仔细匹配用途、属性、品牌、"
            "型号、预算和规格；BM25 排名只是参考。最后只调用 select_product。",
        )
        self.assertEqual(
            DeepSeekSelector.tool_description,
            "Select exactly one product from the supplied BM25 Top-5 and "
            "choose only available option values shown for that product.",
        )

    def test_local_selector_disables_thinking_by_default(self) -> None:
        # Measured on qwen3:8b: thinking burned 662 completion tokens on a
        # trivial prompt versus 4 with reasoning disabled.
        self.assertEqual(OllamaSelector.extra_payload["reasoning_effort"], "none")
        # DeepSeek keeps its own switch and must not receive the local one.
        self.assertNotIn("reasoning_effort", DeepSeekSelector.extra_payload)
        self.assertIn("thinking", DeepSeekSelector.extra_payload)

    def test_local_prompt_clarifies_the_nested_option_shape(self) -> None:
        # A 7B local model mirrors the nested customization_options it is shown,
        # so the local prompt must spell out the flattened contract.
        self.assertNotEqual(OllamaSelector.system_prompt, DeepSeekSelector.system_prompt)
        self.assertIn("规格描述", OllamaSelector.system_prompt)
        self.assertNotEqual(
            OllamaSelector.tool_description,
            DeepSeekSelector.tool_description,
        )

    def test_local_payload_omits_deepseek_only_fields(self) -> None:
        captured = {}

        def opener(request, *, timeout):
            captured.update(json.loads(request.data.decode("utf-8")))
            return _Response(_selection_response())

        selector = OllamaSelector(opener=opener)
        result = selector.select("白色耳机", [_candidate()])

        # Ollama has no thinking switch, but it does honour a forced choice.
        self.assertNotIn("thinking", captured)
        self.assertEqual(
            captured["tool_choice"],
            {"type": "function", "function": {"name": "select_product"}},
        )
        self.assertIn("tools", captured)
        self.assertEqual(result.provider, "ollama")
        self.assertEqual(result.decision.asin, "A001")

    def test_deepseek_payload_is_unchanged(self) -> None:
        captured = {}

        def opener(request, *, timeout):
            captured.update(json.loads(request.data.decode("utf-8")))
            return _Response(_selection_response())

        selector = DeepSeekSelector(api_key="test-key", opener=opener)
        result = selector.select("白色耳机", [_candidate()])

        self.assertEqual(captured["thinking"], {"type": "disabled"})
        self.assertEqual(
            captured["tool_choice"],
            {"type": "function", "function": {"name": "select_product"}},
        )
        self.assertEqual(result.provider, "deepseek")


if __name__ == "__main__":
    unittest.main()
