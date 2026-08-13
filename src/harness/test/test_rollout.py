"""Batch rollout aggregation, concurrency, persistence, and CLI tests."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from threading import Lock
import unittest
from unittest.mock import patch

from src.data.search_bm25 import build_index, sha256_file
from src.harness.agent import AgentRunner, RolloutError, RolloutRunner
from src.harness.cli import main
from src.harness.harness import ShoppingHarness
from src.harness.providers import ProviderAPIError
from src.harness.test.test_agent_runner import (
    ALTERNATIVE_ASIN,
    PRODUCTS,
    TARGET_ASIN,
    ScriptedProvider,
    tool_turn,
)


def successful_provider(asin: str, color: str) -> ScriptedProvider:
    return ScriptedProvider(
        [
            tool_turn(
                f"search-{asin}",
                "search_products",
                {"query": "冰箱", "limit": 2},
            ),
            tool_turn(
                f"detail-{asin}",
                "get_product",
                {"asin": asin},
            ),
            tool_turn(
                f"submit-{asin}",
                "submit_recommendations",
                {
                    "recommendations": [
                        {
                            "asin": asin,
                            "selected_options": {"颜色": color},
                            "reason": "详情确认商品颜色和属性符合公开需求",
                        }
                    ]
                },
            ),
        ]
    )


class RolloutRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.directory = Path(self._temporary_directory.name)
        self.products_path = self.directory / "products.json"
        self.index_path = self.directory / "products.sqlite3"
        self.products_path.write_text(
            json.dumps(PRODUCTS, ensure_ascii=False),
            encoding="utf-8",
        )
        build_index(
            PRODUCTS,
            self.index_path,
            product_data_sha256=sha256_file(self.products_path),
        )
        self.harness = ShoppingHarness(
            products_path=self.products_path,
            index_path=self.index_path,
            default_max_steps=7,
        )
        self.addCleanup(self.harness.close)
        self.eval_task_id = self.harness.list_task_ids("eval")[0]
        self.train_task_id = self.harness.list_task_ids("train")[0]

    def test_batch_continues_and_writes_full_records_and_summary(self) -> None:
        providers = [
            successful_provider(TARGET_ASIN, "白色"),
            ScriptedProvider(
                [ProviderAPIError("provider unavailable", retryable=False)]
            ),
        ]
        factory_lock = Lock()

        def runner_factory() -> AgentRunner:
            with factory_lock:
                provider = providers.pop(0)
            return AgentRunner(provider=provider, harness=self.harness)

        report = RolloutRunner(
            harness=self.harness,
            runner_factory=runner_factory,
        ).run_tasks(
            [self.eval_task_id, self.train_task_id],
            max_steps=7,
            output_root=self.directory / "rollouts",
            rollout_id="batch_contract",
            metadata={"provider": "scripted", "model": "scripted-model"},
        )

        self.assertEqual([record.index for record in report.records], [0, 1])
        self.assertEqual(report.summary.requested_count, 2)
        self.assertEqual(report.summary.result_count, 2)
        self.assertEqual(report.summary.completed_count, 1)
        self.assertEqual(report.summary.failed_count, 1)
        self.assertEqual(report.summary.strict_success_count, 1)
        self.assertEqual(report.summary.completion_rate, 0.5)
        self.assertEqual(report.summary.strict_success_rate, 0.5)
        self.assertEqual(report.summary.average_reward, 0.5)
        self.assertEqual(report.summary.average_valid_reward, 1.0)
        self.assertEqual(report.summary.usage.total_tokens, 42)
        self.assertEqual(report.records[0].consumption.model_turn_count, 3)
        self.assertEqual(report.records[0].consumption.tool_call_count, 3)
        self.assertEqual(report.records[0].consumption.usage.total_tokens, 42)
        self.assertGreaterEqual(
            report.records[0].consumption.duration_seconds,
            0.0,
        )
        self.assertEqual(report.records[1].consumption.model_turn_count, 0)
        self.assertIsNone(report.records[1].consumption.usage)
        self.assertEqual(
            report.summary.terminal_reason_counts,
            {"provider_error": 1, "submitted": 1},
        )
        self.assertEqual(self.harness.list_sessions(), [])
        self.assertEqual(report.metadata["provider"], "scripted")

        output_dir = self.directory / "rollouts" / "batch_contract"
        lines = (output_dir / "runs.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual(len(lines), 2)
        first_run = json.loads(lines[0])
        self.assertEqual(first_run["index"], 0)
        self.assertEqual(
            first_run["consumption"]["usage"]["total_tokens"],
            42,
        )
        consumption_lines = (output_dir / "consumption.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual(len(consumption_lines), 2)
        first_consumption = json.loads(consumption_lines[0])
        self.assertEqual(first_consumption["task_id"], self.eval_task_id)
        self.assertEqual(
            first_consumption["consumption"]["usage"]["total_tokens"],
            42,
        )
        summary = json.loads(
            (output_dir / "summary.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("records", summary)
        self.assertEqual(summary["summary"]["requested_count"], 2)
        self.assertEqual(summary["metadata"]["model"], "scripted-model")

    def test_parallel_results_remain_in_input_order(self) -> None:
        def runner_factory() -> AgentRunner:
            return AgentRunner(
                provider=successful_provider(TARGET_ASIN, "白色"),
                harness=self.harness,
            )

        report = RolloutRunner(
            harness=self.harness,
            runner_factory=runner_factory,
        ).run_tasks(
            [self.eval_task_id, self.eval_task_id],
            workers=2,
            rollout_id="parallel_contract",
        )

        self.assertEqual([record.index for record in report.records], [0, 1])
        self.assertTrue(
            all(record.result.status == "completed" for record in report.records)
        )
        self.assertEqual(report.summary.strict_success_count, 2)
        self.assertEqual(self.harness.list_sessions(), [])

    def test_unexpected_factory_error_is_recorded_not_raised(self) -> None:
        def broken_factory() -> AgentRunner:
            raise RuntimeError("factory exploded")

        report = RolloutRunner(
            harness=self.harness,
            runner_factory=broken_factory,
        ).run_tasks([self.eval_task_id], rollout_id="broken_contract")

        record = report.records[0]
        self.assertIsNone(record.result)
        self.assertEqual(record.error_type, "RuntimeError")
        self.assertEqual(record.consumption.model_turn_count, 0)
        self.assertEqual(record.consumption.tool_call_count, 0)
        self.assertIsNone(record.consumption.usage)
        self.assertEqual(report.summary.unexpected_error_count, 1)
        self.assertEqual(
            report.summary.terminal_reason_counts,
            {"unexpected_error": 1},
        )

    def test_existing_rollout_directory_is_never_overwritten(self) -> None:
        root = self.directory / "rollouts"
        (root / "duplicate").mkdir(parents=True)
        rollout = RolloutRunner(
            harness=self.harness,
            runner_factory=lambda: AgentRunner(
                provider=successful_provider(TARGET_ASIN, "白色"),
                harness=self.harness,
            ),
        )

        with self.assertRaises(RolloutError):
            rollout.run_tasks(
                [self.eval_task_id],
                output_root=root,
                rollout_id="duplicate",
            )

    def test_run_split_uses_harness_sampling_metadata(self) -> None:
        report = RolloutRunner(
            harness=self.harness,
            runner_factory=lambda: AgentRunner(
                provider=successful_provider(TARGET_ASIN, "白色"),
                harness=self.harness,
            ),
        ).run_split(
            "eval",
            1,
            seed="fixed-seed",
            rollout_id="sample_contract",
        )

        self.assertEqual(report.split, "eval")
        self.assertEqual(report.seed, "fixed-seed")
        self.assertEqual(report.task_ids, [self.eval_task_id])

    def test_cli_run_executes_without_network_using_provider_contract(self) -> None:
        output = io.StringIO()
        provider = successful_provider(TARGET_ASIN, "白色")
        with patch("src.harness.cli.DeepSeekProvider", return_value=provider):
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "run",
                        "--task-id",
                        self.eval_task_id,
                        "--products-path",
                        str(self.products_path),
                        "--index-path",
                        str(self.index_path),
                        "--max-steps",
                        "7",
                        "--no-env-file",
                    ]
                )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "completed")
        self.assertTrue(payload["session"]["evaluation"]["strict_success"])
        self.assertEqual(provider.requests[0]["tool_choice"], "auto")

    def test_cli_rollout_creates_artifacts_and_compact_stdout(self) -> None:
        output = io.StringIO()
        provider = successful_provider(TARGET_ASIN, "白色")
        with patch("src.harness.cli.DeepSeekProvider", return_value=provider):
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "rollout",
                        "--task-id",
                        self.eval_task_id,
                        "--products-path",
                        str(self.products_path),
                        "--index-path",
                        str(self.index_path),
                        "--output-root",
                        str(self.directory / "cli-rollouts"),
                        "--rollout-id",
                        "cli_contract",
                        "--max-steps",
                        "7",
                        "--no-env-file",
                    ]
                )

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["rollout_id"], "cli_contract")
        self.assertEqual(payload["summary"]["strict_success_count"], 1)
        self.assertTrue(
            (self.directory / "cli-rollouts" / "cli_contract" / "runs.jsonl")
            .is_file()
        )

    def test_cli_requires_count_for_split(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                main(["rollout", "--split", "eval"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--count is required", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
