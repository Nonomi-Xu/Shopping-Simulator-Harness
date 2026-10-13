"""Contract and integration tests for the project Agent Runner."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import tempfile
import unittest
from typing import Any

from src.data.search_bm25 import build_index, sha256_file
from src.harness.agent import (
    AgentRunner,
    load_agent_skill,
    shopping_tool_definitions,
)
from src.harness.harness import ShoppingHarness
from src.harness.providers import (
    AssistantTurn,
    ChatMessage,
    ProviderAPIError,
    TokenUsage,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    canonical_json,
)


TARGET_ASIN = "000000000001"
ALTERNATIVE_ASIN = "000000000002"
PRODUCTS = [
    {
        "asin": TARGET_ASIN,
        "tag": "eval",
        "title": "白色静音目标冰箱",
        "shop_name": "目标旗舰店",
        "category": "家电›冰箱",
        "pricing": [799],
        "attribute": ["静音", "节能"],
        "customization_options": {
            "颜色": [
                {"value": "白色", "price": 799, "is_available": True}
            ]
        },
        "instructions": [
            {
                "asin": TARGET_ASIN,
                "instruction": "寻找一款白色静音目标冰箱",
                "instruction_options": ["白色"],
                "attributes": ["静音"],
            }
        ],
    },
    {
        "asin": ALTERNATIVE_ASIN,
        "tag": "train",
        "title": "黑色普通冰箱",
        "shop_name": "普通旗舰店",
        "category": "家电›冰箱",
        "pricing": [699],
        "attribute": ["小容量"],
        "customization_options": {
            "颜色": [
                {"value": "黑色", "price": 699, "is_available": True}
            ]
        },
        "instructions": [
            {
                "asin": ALTERNATIVE_ASIN,
                "instruction": "寻找一款黑色普通冰箱",
                "instruction_options": ["黑色"],
                "attributes": ["小容量"],
            }
        ],
    },
]


class ScriptedProvider:
    def __init__(self, outputs: Sequence[AssistantTurn | ProviderAPIError]) -> None:
        self.outputs = list(outputs)
        self.requests: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "scripted"

    @property
    def model(self) -> str:
        return "scripted-model"

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolDefinition] = (),
        tool_choice: ToolChoice = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> AssistantTurn:
        self.requests.append(
            {
                "messages": list(messages),
                "tools": list(tools),
                "tool_choice": tool_choice,
                "parameters": dict(parameters or {}),
            }
        )
        if not self.outputs:
            raise AssertionError("scripted provider exhausted")
        output = self.outputs.pop(0)
        if isinstance(output, ProviderAPIError):
            raise output
        if callable(output):
            return output(messages)
        return output


def tool_turn(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
    *,
    input_tokens: int = 10,
    output_tokens: int = 4,
) -> AssistantTurn:
    return AssistantTurn(
        provider="scripted",
        model="scripted-model",
        tool_calls=[
            ToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
                raw_arguments=canonical_json(arguments),
            )
        ],
        finish_reason="tool_calls",
        usage=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


def requirement_turn(color: str):
    return tool_turn("plan", "update_plan", {
        "constraints": [{
            "constraint_id": "color", "description": f"需要{color}",
            "priority": "hard", "source": "user_query", "source_quote": color,
        }],
        "queries": [f"{color}冰箱"], "change_summary": "提取公开颜色要求",
    })


def color_evidence_turn(asin: str, color: str):
    def respond(messages):
        observation = json.loads(messages[-1].content)
        return tool_turn("color-evidence", "record_evidence_batch", {"assessments": [{
            "asin": asin, "constraint_id": "color", "status": "satisfied",
            "observation_id": observation["observation_id"],
            "field_path": "/customization_options/颜色/0/value", "quote": color,
            "explanation": "详情中的可选颜色符合要求",
        }]})
    return respond


class AgentRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        directory = Path(self._temporary_directory.name)
        products_path = directory / "products.json"
        index_path = directory / "products.sqlite3"
        products_path.write_text(
            json.dumps(PRODUCTS, ensure_ascii=False),
            encoding="utf-8",
        )
        build_index(
            PRODUCTS,
            index_path,
            product_data_sha256=sha256_file(products_path),
        )
        self.harness = ShoppingHarness(
            products_path=products_path,
            index_path=index_path,
            default_max_steps=7,
        )
        self.addCleanup(self.harness.close)
        self.task_id = self.harness.list_task_ids("all")[0]

    def test_skill_loader_compiles_all_declared_contracts(self) -> None:
        skill = load_agent_skill()

        self.assertEqual(skill.name, "shopping-agent")
        self.assertEqual(len(skill.references), 3)
        self.assertIn("# Shopping Agent Definition", skill.system_prompt)
        self.assertIn("# Harness Tool Contract", skill.system_prompt)
        self.assertIn("# Shopping Strategy", skill.system_prompt)
        self.assertEqual(len(skill.source_sha256), 64)

    def test_tool_definitions_come_from_harness_action_models(self) -> None:
        tools = shopping_tool_definitions()

        self.assertEqual(
            [tool.name for tool in tools],
            ["search_products", "get_product", "submit_recommendations"],
        )
        search_schema = tools[0].parameters
        self.assertEqual(search_schema["properties"]["limit"]["maximum"], 50)
        self.assertFalse(search_schema["additionalProperties"])
        self.assertIn("recommendations", tools[2].parameters["required"])

    def test_complete_agent_loop_submits_and_aggregates_usage(self) -> None:
        provider = ScriptedProvider(
            [
                requirement_turn("白色"),
                tool_turn(
                    "call-search",
                    "search_products",
                    {"query": "白色静音冰箱", "limit": 2},
                ),
                tool_turn(
                    "call-detail",
                    "get_product",
                    {"asin": TARGET_ASIN},
                ),
                color_evidence_turn(TARGET_ASIN, "白色"),
                tool_turn(
                    "call-submit",
                    "submit_recommendations",
                    {
                        "recommendations": [
                            {
                                "asin": TARGET_ASIN,
                                "selected_options": {"颜色": "白色"},
                                "reason": "商品详情确认白色规格可用，并具备静音属性",
                            }
                        ]
                    },
                ),
            ]
        )
        runner = AgentRunner(
            provider=provider,
            harness=self.harness,
            provider_parameters={"temperature": 0},
        )

        result = runner.run(self.task_id)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.terminal_reason, "submitted")
        self.assertEqual(result.model_turn_count, 5)
        self.assertEqual(result.tool_call_count, 3)
        self.assertTrue(result.session.evaluation.strict_success)
        self.assertEqual(result.usage.total_tokens, 70)
        self.assertEqual(
            [message.role for message in result.messages],
            ["system", "user", *(["assistant", "tool"] * 5)],
        )
        self.assertEqual(len(provider.requests), 5)
        self.assertEqual(provider.requests[0]["tool_choice"], "required")
        self.assertEqual(
            provider.requests[0]["parameters"],
            {"temperature": 0},
        )
        self.assertNotIn(TARGET_ASIN, provider.requests[0]["messages"][1].content)

    def test_harness_step_limit_is_a_runtime_termination(self) -> None:
        provider = ScriptedProvider(
            [
                tool_turn(
                    "call-search",
                    "search_products",
                    {"query": "冰箱"},
                )
            ]
        )
        runner = AgentRunner(provider=provider, harness=self.harness)

        result = runner.run(self.task_id, max_steps=1)

        self.assertEqual(result.status, "terminated")
        self.assertEqual(result.terminal_reason, "max_steps")
        self.assertEqual(result.session.state.status, "terminated")

    def test_text_response_cannot_replace_terminal_submission(self) -> None:
        provider = ScriptedProvider(
            [
                AssistantTurn(
                    provider="scripted",
                    model="scripted-model",
                    content="我推荐目标商品。",
                    finish_reason="stop",
                )
            ]
        )
        result = AgentRunner(
            provider=provider,
            harness=self.harness,
        ).run(self.task_id)

        self.assertEqual(result.status, "failed")
        self.assertEqual(
            result.terminal_reason,
            "model_stopped_without_submission",
        )
        self.assertEqual(result.tool_call_count, 0)
        self.assertEqual(result.session.state.status, "active")

    def test_parallel_tool_calls_fail_without_partial_dispatch(self) -> None:
        first = tool_turn(
            "call-one",
            "search_products",
            {"query": "冰箱"},
        ).tool_calls[0]
        second = tool_turn(
            "call-two",
            "search_products",
            {"query": "静音冰箱"},
        ).tool_calls[0]
        provider = ScriptedProvider(
            [
                AssistantTurn(
                    provider="scripted",
                    model="scripted-model",
                    tool_calls=[first, second],
                    finish_reason="tool_calls",
                )
            ]
        )

        result = AgentRunner(
            provider=provider,
            harness=self.harness,
        ).run(self.task_id)

        self.assertEqual(result.terminal_reason, "multiple_tool_calls")
        self.assertEqual(result.tool_call_count, 0)
        self.assertEqual(result.session.history, [])

    def test_provider_failure_is_returned_as_an_audit_result(self) -> None:
        provider = ScriptedProvider(
            [ProviderAPIError("temporary outage", retryable=True)]
        )

        result = AgentRunner(
            provider=provider,
            harness=self.harness,
        ).run(self.task_id)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.terminal_reason, "provider_error")
        self.assertEqual(result.error_type, "ProviderAPIError")
        self.assertEqual(result.error_message, "temporary outage")

    def test_model_turn_limit_preserves_the_incomplete_session(self) -> None:
        provider = ScriptedProvider(
            [
                tool_turn(
                    "call-search",
                    "search_products",
                    {"query": "冰箱"},
                )
            ]
        )
        result = AgentRunner(
            provider=provider,
            harness=self.harness,
            max_model_turns=1,
        ).run(self.task_id)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.terminal_reason, "max_model_turns")
        self.assertEqual(result.tool_call_count, 1)
        self.assertEqual(result.session.state.status, "active")


if __name__ == "__main__":
    unittest.main()
