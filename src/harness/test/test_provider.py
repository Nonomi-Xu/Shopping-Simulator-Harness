from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.harness.providers import (
    ChatMessage,
    ModelProvider,
    ToolCall,
    ToolDefinition,
)
from src.harness.providers.deepseek import DeepSeekProvider
from src.harness.schemas import GetProductAction


class ProviderContractTests(unittest.TestCase):
    def test_tool_message_requires_call_id(self) -> None:
        with self.assertRaises(ValidationError):
            ChatMessage(role="tool", content='{"ok":true}')

    def test_assistant_tool_call_and_tool_result_serialize_for_replay(self) -> None:
        call = ToolCall(
            id="call-1",
            name="get_product",
            arguments={"asin": "A1"},
            raw_arguments='{"asin":"A1"}',
        )
        assistant = ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[call],
        )
        tool_result = ChatMessage(
            role="tool",
            content='{"asin":"A1"}',
            tool_call_id="call-1",
        )

        self.assertEqual(
            assistant.as_provider_payload()["tool_calls"][0]["function"]["name"],
            "get_product",
        )
        self.assertEqual(
            tool_result.as_provider_payload()["tool_call_id"],
            "call-1",
        )

    def test_tool_definition_uses_authoritative_action_schema(self) -> None:
        tool = ToolDefinition.from_action_model(
            name="get_product",
            description="Read one surfaced product.",
            action_model=GetProductAction,
        )

        self.assertEqual(tool.parameters["required"], ["asin"])
        self.assertEqual(tool.parameters["properties"]["asin"]["minLength"], 1)
        self.assertFalse(tool.parameters["additionalProperties"])

    def test_deepseek_implements_runtime_provider_protocol(self) -> None:
        provider = DeepSeekProvider(api_key="test-key", max_retries=0)
        self.assertIsInstance(provider, ModelProvider)


if __name__ == "__main__":
    unittest.main()
