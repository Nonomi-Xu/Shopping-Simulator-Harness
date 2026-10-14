"""Tracing remains optional, public-only, nested, and behavior preserving."""

from contextlib import contextmanager
import importlib.util
import json
import unittest
from unittest.mock import Mock, patch

from src.harness.agent import AgentRunner, RuntimeConfig
from src.harness.agent.backend_comparison import compare_results
from src.harness.test import test_agent_runner as fixtures
from src.harness.test.test_rollout import successful_provider


@unittest.skipUnless(importlib.util.find_spec("langsmith"), "install .[tracing]")
class TracingTest(unittest.TestCase):
    setUp = fixtures.AgentRunnerTest.setUp

    def run_script(self, backend="native", **kwargs):
        provider = successful_provider(fixtures.TARGET_ASIN, "白色")
        provider.default_parameters = {"api_key": "PRIVATE_PROVIDER_KEY"}
        result = AgentRunner(harness=self.harness, provider=provider,
                             config=RuntimeConfig(orchestration_backend=backend),
                             **kwargs).run(self.task_id, session_id="trace-test")
        self.harness.release_session("trace-test", force=True)
        return result, provider

    def test_disabled_tracing_does_not_touch_sdk(self):
        with patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}), \
                patch("langsmith.trace", side_effect=AssertionError("must stay disabled")):
            result, _ = self.run_script()
        self.assertEqual(result.status, "completed")

    def test_real_sdk_spans_are_nested_and_exclude_private_evaluation(self):
        import langsmith
        real_trace = langsmith.trace
        spans = []

        @contextmanager
        def capture(*args, **kwargs):
            with real_trace(*args, **kwargs) as span:
                spans.append(span)
                yield span

        with patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}):
            baseline, _ = self.run_script()
        # Local SDK context creates real RunTrees without exporting anything.
        with patch.dict("os.environ", {"LANGSMITH_TRACING": "true"}), \
                langsmith.tracing_context(enabled="local"), \
                patch("langsmith.trace", side_effect=capture):
            result, _ = self.run_script()
        self.assertTrue(compare_results(baseline, result)["equal"])
        root = next(s for s in spans if s.name == "shopping.run")
        turns = [s for s in spans if s.name == "shopping.single"]
        models = [s for s in spans if s.run_type == "llm"]
        tools = [s for s in spans if s.run_type == "tool"]
        self.assertEqual((len(turns), len(models), len(tools)), (5, 5, 5))
        self.assertTrue(all(s.parent_run_id == root.id for s in turns))
        self.assertTrue(all(s.parent_run_id in {t.id for t in turns} for s in models + tools))
        self.assertEqual(models[0].outputs["usage_metadata"]["total_tokens"], 14)
        self.assertEqual(root.outputs["status"], "completed")
        payload = json.dumps([{"inputs": s.inputs, "outputs": s.outputs,
                               "metadata": s.extra} for s in spans], ensure_ascii=False)
        self.assertNotIn("PRIVATE_PROVIDER_KEY", payload)
        self.assertNotIn('"evaluation":', payload)
        self.assertNotIn('"strict_success":', payload)
        self.assertNotIn('"private_facts":', payload)
        self.assertTrue(result.session.evaluation.strict_success)

    def test_export_failure_does_not_retry_or_change_actions(self):
        with patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}):
            baseline, _ = self.run_script()
        with patch.dict("os.environ", {"LANGSMITH_TRACING": "true"}), \
                patch("langsmith.trace", side_effect=RuntimeError("export unavailable")), \
                self.assertLogs("src.harness.agent.tracing", level="WARNING"):
            result, provider = self.run_script()
        self.assertEqual(len(provider.requests), 5)
        self.assertTrue(compare_results(baseline, result)["equal"])

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "install .[graph]")
    def test_graph_runs_inside_trace_without_serializing_evaluator(self):
        import langsmith
        from langsmith.run_trees import RunTree

        captured = []
        client = Mock(spec=langsmith.Client)
        with patch.dict("os.environ", {"LANGSMITH_TRACING": "true"}), \
                langsmith.tracing_context(enabled="local", client=client), \
                patch("langchain_core.tracers.langchain.get_client", return_value=client), \
                patch.object(RunTree, "post", autospec=True), \
                patch.object(RunTree, "patch", autospec=True,
                             side_effect=lambda run, *a, **kw: captured.append(run)):
            result, _ = self.run_script("langgraph")
        self.assertEqual(result.status, "completed", result.error_message)
        self.assertTrue(captured)
        raw = json.dumps([{"inputs": s.inputs, "outputs": s.outputs} for s in captured], default=str)
        self.assertNotIn('"evaluation":', raw)
        self.assertNotIn('"strict_success":', raw)


if __name__ == "__main__":
    unittest.main()
