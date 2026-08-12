from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from src.harness.api.deepseek import (
    DeepSeekClient,
    DeepSeekConfigurationError,
    call_deepseek,
)


class _Response:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return self._body


class DeepSeekClientTests(unittest.TestCase):
    def test_chat_sends_expected_request_and_returns_content(self) -> None:
        captured = {}

        def opener(request, *, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _Response(
                {"choices": [{"message": {"content": "hello"}}]}
            )

        client = DeepSeekClient(
            api_key="test-key",
            timeout=12,
            opener=opener,
        )
        result = client.chat(
            [{"role": "user", "content": "Hi"}],
            max_tokens=32,
        )

        self.assertEqual(result, "hello")
        self.assertEqual(
            captured["url"],
            "https://api.deepseek.com/chat/completions",
        )
        self.assertEqual(captured["authorization"], "Bearer test-key")
        self.assertEqual(captured["payload"]["messages"][0]["content"], "Hi")
        self.assertEqual(captured["payload"]["max_tokens"], 32)
        self.assertFalse(captured["payload"]["stream"])
        self.assertEqual(captured["timeout"], 12.0)

    def test_call_deepseek_builds_system_and_user_messages(self) -> None:
        captured = {}

        def opener(request, *, timeout):
            captured.update(json.loads(request.data.decode("utf-8")))
            return _Response(
                {"choices": [{"message": {"content": "ok"}}]}
            )

        with patch("src.harness.api.deepseek.urlopen", opener):
            result = call_deepseek(
                "question",
                system_prompt="instructions",
                api_key="test-key",
            )

        self.assertEqual(result, "ok")
        self.assertEqual(
            captured["messages"],
            [
                {"role": "system", "content": "instructions"},
                {"role": "user", "content": "question"},
            ],
        )

    def test_missing_api_key_has_clear_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                DeepSeekConfigurationError,
                "DEEPSEEK_API_KEY",
            ):
                DeepSeekClient()


if __name__ == "__main__":
    unittest.main()
