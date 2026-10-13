"""Tests for dependency-free, non-overwriting dotenv loading."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.harness.env import DotEnvError, load_dotenv


class DotEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.directory = Path(self._temporary_directory.name)

    def test_loads_quotes_comments_and_export_without_overwriting(self) -> None:
        dotenv = self.directory / ".env"
        dotenv.write_text(
            "DEEPSEEK_API='file-secret' # local key\n"
            "export SHOPPING_TEST_VALUE=hello\\ world\n",
            encoding="utf-8",
        )
        with patch.dict(
            os.environ,
            {"DEEPSEEK_API": "process-secret"},
            clear=True,
        ):
            loaded = load_dotenv(dotenv)

            self.assertEqual(loaded, dotenv.resolve())
            self.assertEqual(os.environ["DEEPSEEK_API"], "process-secret")
            self.assertEqual(os.environ["SHOPPING_TEST_VALUE"], "hello world")

    def test_disabled_loading_does_not_require_a_file(self) -> None:
        self.assertIsNone(load_dotenv(self.directory / "missing", disabled=True))

    def test_explicit_missing_or_invalid_file_fails_clearly(self) -> None:
        with self.assertRaises(DotEnvError):
            load_dotenv(self.directory / "missing")

        invalid = self.directory / ".env"
        invalid.write_text("not an assignment\n", encoding="utf-8")
        with self.assertRaisesRegex(DotEnvError, "invalid dotenv assignment"):
            load_dotenv(invalid)


if __name__ == "__main__":
    unittest.main()
