from __future__ import annotations

import json
import unittest

from bm25_agent.provider import (
    DEFAULT_BASE_URL,
    DeepSeekSelector,
    SelectorError,
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


if __name__ == "__main__":
    unittest.main()
