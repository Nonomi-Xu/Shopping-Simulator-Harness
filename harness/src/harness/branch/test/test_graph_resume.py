"""Graph resume keeps the existing durable Session/audit boundary intact."""

import importlib.util
from datetime import datetime, timezone
import unittest
from unittest.mock import patch

from src.harness.agent import AgentRunner, RuntimeConfig, AgentRunResult, UserReply
from src.harness.agent.backend_comparison import compare_results
from src.harness.agent.events import digest, replay_events
from src.harness.branch.test import test_stage_three as interactive_fixtures
from src.harness.branch.test.test_stage_three import plan, submission
from src.harness.test.test_agent_runner import ScriptedProvider, tool_turn
from src.harness.test.test_graph_backend import review


@unittest.skipUnless(importlib.util.find_spec("langgraph"), "install .[graph] for parity tests")
class GraphResumeTest(unittest.TestCase):
    setUp = interactive_fixtures.InteractiveTest.setUp
    new_harness = interactive_fixtures.InteractiveTest.new_harness

    def test_native_resume_accepts_checkpoint_without_new_backend_field(self):
        paused, _, _ = interactive_fixtures.InteractiveTest.pause(self)
        paused = paused.model_copy(deep=True)
        # Simulate the exact pre-migration manifest schema, then rehash its
        # audit chain. The signed task/session checkpoint payload is unchanged.
        paused.manifest["config"].pop("orchestration_backend")
        previous = "0" * 64
        events = []
        for event in paused.events:
            values = event.model_dump(mode="json", exclude={"event_hash"})
            values["previous_hash"] = previous
            if event.kind == "run_started":
                values["payload"]["manifest"]["config"].pop("orchestration_backend")
            previous = digest(values)
            events.append(type(event)(**values, event_hash=previous))
        paused.manifest["final_event_hash"] = previous
        paused = paused.model_copy(update={"events": events})
        runner = AgentRunner(
            provider=ScriptedProvider([plan(), interactive_fixtures.ask()]),
            harness=self.new_harness(), config=self.config,
        )
        result = runner.resume(paused, "仍然白色")
        self.assertEqual(result.status, "waiting_for_user", result.error_message)

    def test_multi_role_pause_and_new_runner_resume(self):
        paused_runs, completed_runs = [], []
        with patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}), \
                patch("src.harness.sources.datetime") as source_clock:
            source_clock.now.return_value = datetime(2026, 10, 14, tzinfo=timezone.utc)
            for backend in ("native", "langgraph"):
                config = RuntimeConfig(agent_mode="multi", interaction_mode="interactive",
                                       orchestration_backend=backend)
                before = ScriptedProvider([
                    plan(),
                    tool_turn("search", "search_products", {"query": "耳机", "method": "bm25"}),
                    review("replan"),
                    tool_turn("ask", "ask_user", {"question": "要换颜色吗？",
                              "information_gap": "颜色", "constraint_ids": ["color"]}),
                ])
                paused = AgentRunner(provider=before, harness=self.new_harness(), config=config).run_query(
                    "寻找白色耳机", session_id="graph-resume")
                self.assertEqual(paused.status, "waiting_for_user", paused.error_message)
                paused_runs.append(paused)
                restored = AgentRunResult.model_validate_json(paused.model_dump_json())
                after = ScriptedProvider([
                    plan("黑色", "user_0001"),
                    tool_turn("detail", "get_product", {"asin": "black"}),
                    tool_turn("evidence", "record_evidence", {
                        "asin": "black", "constraint_id": "color", "status": "satisfied",
                        "observation_id": "graph-resume:action_0006:observation",
                        "field_path": "/attributes/0", "quote": "黑色", "explanation": "公开颜色",
                    }),
                    review("recommend", ["black"]),
                    submission(),
                ])
                runner = AgentRunner(provider=after, harness=self.new_harness(), config=config)
                result = runner.resume(restored, UserReply(text="改成黑色", replace_constraint_ids=["color"]))
                self.assertEqual(result.status, "completed", result.error_message)
                self.assertEqual((result.action_count, result.tool_call_count, result.search_count), (9, 3, 1))
                self.assertIsNone(result.session.evaluation)
                self.assertEqual(replay_events(result.events)["outcome"]["status"], "completed")
                self.assertNotIn("PRIVATE_SOURCE_SENTINEL", result.model_dump_json())
                completed_runs.append(result)
                with self.assertRaisesRegex(ValueError, "consumed"):
                    runner.resume(restored, "重复回答")
        paused_comparison = compare_results(*paused_runs)
        completed_comparison = compare_results(*completed_runs)
        self.assertTrue(paused_comparison["equal"], paused_comparison)
        self.assertTrue(completed_comparison["equal"], completed_comparison)


if __name__ == "__main__":
    unittest.main()
