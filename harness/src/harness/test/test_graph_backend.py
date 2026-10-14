"""Same tasks/turns through both schedulers: actions, budgets, results and scores."""

import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from unittest.mock import patch
import unittest

from src.harness.agent import AgentRunner, RuntimeConfig, write_run_artifacts
from src.harness.agent.backend_comparison import compare_results
from src.harness.agent.graph import build_graph
from src.harness.providers import AssistantTurn, ProviderAPIError
from src.harness.test import test_agent_runner as fixtures
from src.harness.test.test_agent_architecture import plan_turn, evidence_arguments, SESSION
from src.harness.test.test_rollout import successful_provider


def review(next_step, asins=None):
    return fixtures.tool_turn("review", "review_progress", {
        "next_step": next_step, "summary": "Review public evidence",
        "reviewed_asins": asins or [],
    })


def multi_script(asin=fixtures.TARGET_ASIN):
    return [
        plan_turn(),
        fixtures.tool_turn("search", "search_products", {"query": "冰箱", "method": "bm25"}),
        review("act"),
        fixtures.tool_turn("detail", "get_product", {"asin": asin}),
        fixtures.tool_turn("evidence", "record_evidence", evidence_arguments(
            asin=asin, observation_id=f"{SESSION}:action_0004:observation")),
        review("recommend", [asin]),
        successful_provider(asin, "白色").outputs[-1],
    ]


@unittest.skipUnless(importlib.util.find_spec("langgraph"), "install .[graph] for parity tests")
class GraphBackendTest(unittest.TestCase):
    setUp = fixtures.AgentRunnerTest.setUp

    def pair(self, turns, *, agent_mode="multi", max_model_turns=None, **config):
        results, requests = [], []
        with patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}):
            for backend in ("native", "langgraph"):
                provider = fixtures.ScriptedProvider(turns)
                result = AgentRunner(
                    harness=self.harness, provider=provider,
                    max_model_turns=max_model_turns or len(turns),
                    config=RuntimeConfig(agent_mode=agent_mode, orchestration_backend=backend, **config),
                ).run(self.task_id, session_id=SESSION)
                results.append(result)
                requests.append(provider.requests)
                self.harness.release_session(SESSION, force=True)
        comparison = compare_results(*results)
        self.assertTrue(comparison["equal"], comparison)
        self.assertEqual(requests[0], requests[1])
        self.assertNotEqual(results[1].terminal_reason, "runtime_error", results[1].error_message)
        return results

    def test_multi_role_submission_preserves_score_and_counts(self):
        native, graph = self.pair(multi_script())
        self.assertEqual(graph.status, "completed")
        self.assertTrue(graph.session.evaluation.strict_success)
        self.assertEqual((graph.action_count, graph.search_count, graph.tool_call_count), (7, 1, 3))
        self.assertEqual([e.payload["agent_role"] for e in graph.events if e.kind == "model_turn"],
                         ["planner", "executor", "observer", "executor", "observer", "observer", "executor"])

    def test_failed_score_is_preserved(self):
        turns = multi_script()
        turns[-1] = turns[-1].model_copy(update={"tool_calls": [
            turns[-1].tool_calls[0].model_copy(update={"arguments": {
                "recommendations": [{"asin": fixtures.TARGET_ASIN,
                                     "selected_options": {}, "reason": "静音"}]
            }})
        ]})
        _, result = self.pair(turns)
        self.assertEqual(result.status, "completed")
        self.assertFalse(result.session.evaluation.strict_success)

    def test_replan_and_rejected_role_action(self):
        turns = multi_script()
        turns = [turns[1], turns[0], turns[1], review("replan"), turns[0], *turns[3:]]
        turns[6] = fixtures.tool_turn("evidence", "record_evidence", evidence_arguments(
            observation_id=f"{SESSION}:action_0006:observation"))
        _, result = self.pair(turns)
        self.assertEqual(result.status, "completed")
        observations = [e.payload for e in result.events if e.kind == "observation"]
        self.assertEqual(observations[0]["status"], "rejected")

    def test_budget_and_model_failure_paths(self):
        for config, reason in [
            ({"max_actions": 3}, "max_actions"),
            ({"max_total_tokens": 15}, "token_budget_exhausted"),
            ({"max_model_turns": 2}, "max_model_turns"),
            ({"context_max_chars": 2000}, "context_budget_exhausted"),
        ]:
            with self.subTest(config=config):
                _, result = self.pair(multi_script(), **config)
                self.assertEqual(result.terminal_reason, reason)
        _, result = self.pair([ProviderAPIError("offline failure")])
        self.assertEqual(result.terminal_reason, "provider_error")
        _, result = self.pair([AssistantTurn(provider="scripted", model="scripted-model",
                                            content="done", finish_reason="stop")])
        self.assertEqual(result.terminal_reason, "model_stopped_without_submission")
        _, result = self.pair([plan_turn().model_copy(update={"usage": None})], max_total_tokens=100)
        self.assertEqual(result.terminal_reason, "usage_unavailable")

    def test_submit_guard_and_batch_rejection(self):
        turns = multi_script()
        _, result = self.pair([*turns[:4], *turns[5:]])
        self.assertEqual(result.terminal_reason, "max_model_turns")
        self.assertIsNone(result.session.evaluation)
        calls = turns[3].tool_calls
        turns[3] = turns[3].model_copy(update={"tool_calls": [
            *calls, calls[0].model_copy(update={"id": "second-detail"})]})
        _, result = self.pair(turns)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.action_count, 8)

    def test_single_agent_remains_supported(self):
        _, result = self.pair(successful_provider(fixtures.TARGET_ASIN, "白色").outputs,
                              agent_mode="single")
        self.assertEqual(result.status, "completed")

    def test_graph_exposes_roles_instead_of_one_wrapped_loop(self):
        graph = build_graph(lambda role: None, multi_agent=True)
        self.assertEqual(set(graph.get_graph().nodes),
                         {"__start__", "__end__", "planner", "executor", "observer"})

    def test_cli_selects_graph_and_preserves_runner_contract(self):
        from src.harness.cli import main

        root = Path(self._temporary_directory.name)
        output = io.StringIO()
        turns = multi_script()
        turns[4] = fixtures.tool_turn("evidence", "record_evidence", evidence_arguments(
            observation_id="session_00000001:action_0004:observation"))
        with patch("src.harness.cli._build_provider", return_value=fixtures.ScriptedProvider(turns)), \
                redirect_stdout(output):
            code = main([
                "run", "--task-id", self.task_id,
                "--agent-mode", "multi", "--orchestration-backend", "langgraph",
                "--no-env-file", "--products-path", str(root / "products.json"),
                "--index-path", str(root / "products.sqlite3"),
            ])
        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["manifest"]["config"]["orchestration_backend"], "langgraph")

    def test_replay_comparison_command_writes_auditable_report(self):
        from tools.compare_backends import compare_bundles

        native, _ = self.pair(multi_script())
        root = Path(self._temporary_directory.name)
        write_run_artifacts(native, root / "recording")
        report = compare_bundles([root / "recording"], root / "products.json",
                                 root / "products.sqlite3", root / "comparison")
        self.assertTrue(report["all_equal"])
        self.assertEqual(report["rows"][0]["recording_exhausted"],
                         {"native": False, "langgraph": False})
        self.assertEqual(json.loads((root / "comparison/report.json").read_text()), report)


if __name__ == "__main__":
    unittest.main()
