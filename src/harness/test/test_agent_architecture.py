"""Behavioral tests for planning, evidence, budgets, isolation and audit replay."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.harness.agent import (
    AgentRunner,
    RuntimeConfig,
    UserProfile,
    UserPreference,
    replay_events,
    replay_run,
    write_run_artifacts,
)
from src.harness.cli import main
from src.harness.test import test_agent_runner as fixtures
from src.harness.test.test_rollout import successful_provider


ASIN = fixtures.TARGET_ASIN
SESSION = "architecture-test"


def plan_arguments():
    return {
        "constraints": [
            {
                "constraint_id": "quiet",
                "description": "需要静音",
                "priority": "hard",
                "source": "user_query",
                "source_quote": "静音",
            }
        ],
        "queries": ["白色静音冰箱"],
        "verification_targets": ["静音属性和白色规格"],
        "change_summary": "从当前用户需求提取搜索计划",
        "evidence_observation_ids": [],
    }


def plan_turn(arguments=None):
    return fixtures.tool_turn("plan", "update_plan", arguments or plan_arguments())


def search_turn(query="冰箱"):
    return fixtures.tool_turn("search", "search_products", {"query": query})


def detail_turn(asin=ASIN):
    return fixtures.tool_turn("detail", "get_product", {"asin": asin})


def evidence_arguments(**overrides):
    return {
        "asin": ASIN,
        "constraint_id": "quiet",
        "status": "satisfied",
        "observation_id": f"{SESSION}:action_0003:observation",
        "field_path": "/attributes/0",
        "quote": "静音",
        "explanation": "公开属性明确注明静音",
        **overrides,
    }


class AgentArchitectureTest(unittest.TestCase):
    setUp = fixtures.AgentRunnerTest.setUp

    def run_script(self, outputs, **kwargs):
        provider = fixtures.ScriptedProvider(outputs)
        result = AgentRunner(
            provider=provider,
            harness=self.harness,
            max_model_turns=len(outputs),
            **kwargs,
        ).run(self.task_id, session_id=SESSION)
        return result, provider

    def test_plan_evidence_and_submission_are_one_auditable_loop(self):
        provider = successful_provider(ASIN, "白色")
        result, provider = self.run_script(
            [
                plan_turn(),
                search_turn(),
                detail_turn(),
                fixtures.tool_turn("evidence", "record_evidence", evidence_arguments()),
                provider.outputs[-1],
            ]
        )
        self.assertEqual(result.status, "completed", result.error_message)
        self.assertTrue(result.session.evaluation.strict_success)
        self.assertEqual(result.action_count, 5)
        self.assertEqual(result.tool_call_count, 3)
        self.assertEqual(result.search_count, 1)
        self.assertEqual(result.plan.phase, "completed")
        self.assertEqual(
            result.memory.candidates[ASIN].assessments["quiet"].quote, "静音"
        )
        final_context = json.loads(provider.requests[-1]["messages"][1].content)
        self.assertEqual(final_context["plan"]["phase"], "submit")
        self.assertEqual(
            final_context["memory"]["candidates"][0]["unknown_constraints"], []
        )
        replayed = replay_events(result.events)
        self.assertEqual(replayed["plan"], result.plan.model_dump(mode="json"))
        self.assertEqual(replayed["memory"], result.memory.model_dump(mode="json"))
        self.assertEqual(
            replayed["final_event_hash"], result.manifest["final_event_hash"]
        )

    def test_plan_cannot_drop_or_weaken_existing_hard_constraint(self):
        changed = plan_arguments()
        changed["constraints"] = []
        changed["no_constraints_reason"] = "试图删除之前的约束"
        result, _ = self.run_script([plan_turn(), plan_turn(changed)])
        self.assertEqual(len(result.task_profile.constraints), 1)
        observations = [e.payload for e in result.events if e.kind == "observation"]
        self.assertEqual(observations[-1]["status"], "rejected")
        self.assertIn("preserve existing hard", observations[-1]["message"])
        self.assertEqual(result.tool_call_count, 0)

    def test_fabricated_profile_constraint_is_rejected_without_memory_mutation(self):
        args = plan_arguments()
        args["constraints"][0]["source_quote"] = "用户没有说过的话"
        result, _ = self.run_script([plan_turn(args)])
        self.assertEqual(result.task_profile.constraints, [])
        self.assertEqual(result.memory.candidates, {})
        self.assertEqual(result.events[-1].kind, "run_finished")

    def test_background_profile_is_visible_but_cannot_create_hard_requirements(self):
        args = plan_arguments()
        args["constraints"] = [
            {
                "constraint_id": "color",
                "description": "偏好黑色",
                "priority": "hard",
                "source": "user_profile",
                "source_quote": "黑色",
            }
        ]
        result, provider = self.run_script(
            [plan_turn(args)],
            user_profile=UserProfile(
                profile_id="shopper",
                preferences=[UserPreference(name="color", value="黑色")],
            ),
        )
        context = json.loads(provider.requests[0]["messages"][1].content)
        self.assertEqual(context["user_profile"]["preferences"][0]["value"], "黑色")
        self.assertIn("白色", context["task_profile"]["task"]["query"])
        self.assertEqual(result.task_profile.constraints, [])

    def test_fabricated_or_cross_product_evidence_is_rejected(self):
        for changes in (
            {"quote": "防火"},
            {"field_path": "/description/invalid"},
            {"observation_id": "another-session:observation"},
            {"asin": fixtures.ALTERNATIVE_ASIN},
            {"field_path": "/attributes"},
        ):
            with self.subTest(changes=changes):
                result, _ = self.run_script(
                    [
                        plan_turn(),
                        search_turn(),
                        detail_turn(),
                        fixtures.tool_turn(
                            "evidence", "record_evidence", evidence_arguments(**changes)
                        ),
                    ]
                )
                self.assertEqual(result.memory.candidates[ASIN].assessments, {})
                obs = [e.payload for e in result.events if e.kind == "observation"]
                self.assertEqual(obs[-1]["status"], "rejected")
                self.harness.release_session(SESSION, force=True)

    def test_missing_evidence_stays_unknown(self):
        result, provider = self.run_script(
            [
                plan_turn(),
                search_turn(),
                detail_turn(),
                fixtures.tool_turn("bad", "unknown", {}),
            ]
        )
        context = json.loads(provider.requests[-1]["messages"][1].content)
        candidate = next(
            c for c in context["memory"]["candidates"] if c["asin"] == ASIN
        )
        self.assertEqual(candidate["unknown_constraints"], ["quiet"])
        self.assertEqual(context["plan"]["phase"], "verify")

    def test_search_budget_still_allows_inspection_and_submission(self):
        result, provider = self.run_script(
            [
                fixtures.requirement_turn("白色"),
                search_turn(),
                search_turn("静音"),
                detail_turn(),
                fixtures.color_evidence_turn(ASIN, "白色"),
                successful_provider(ASIN, "白色").outputs[-1],
            ],
            config=RuntimeConfig(max_searches=1),
        )
        self.assertEqual(result.status, "completed", result.error_message)
        self.assertEqual(result.search_count, 1)
        self.assertEqual(result.action_count, 6)
        self.assertEqual(result.tool_call_count, 3)
        tools = {t.name for t in provider.requests[2]["tools"]}
        self.assertNotIn("search_products", tools)
        self.assertIn("get_product", tools)

    def test_repeated_search_limit_ignores_case_width_whitespace_and_limit(self):
        result, _ = self.run_script(
            [
                search_turn("ＡＢＣ 冰箱"),
                fixtures.tool_turn(
                    "again", "search_products", {"query": "abc   冰箱", "limit": 50}
                ),
            ],
            config=RuntimeConfig(max_repeated_actions=1),
        )
        self.assertEqual(result.search_count, 1)
        self.assertEqual(result.action_count, 2)
        obs = [e.payload for e in result.events if e.kind == "observation"]
        self.assertEqual(obs[-1]["message"], "repeated_action")

    def test_runtime_actions_have_an_independent_budget(self):
        result, provider = self.run_script(
            [plan_turn(), search_turn()], config=RuntimeConfig(max_actions=1)
        )
        self.assertEqual(result.terminal_reason, "max_actions")
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(result.tool_call_count, 0)

    def test_reported_token_budget_stops_before_dispatch(self):
        result, _ = self.run_script(
            [search_turn()], config=RuntimeConfig(max_total_tokens=10)
        )
        self.assertEqual(result.terminal_reason, "token_budget_exhausted")
        self.assertEqual(result.usage.total_tokens, 14)
        self.assertEqual(result.action_count, 0)

    def test_missing_usage_is_not_treated_as_zero_under_a_token_budget(self):
        turn = search_turn().model_copy(update={"usage": None})
        result, _ = self.run_script([turn], config=RuntimeConfig(max_total_tokens=100))
        self.assertEqual(result.terminal_reason, "usage_unavailable")
        self.assertIsNone(result.usage)

    def test_time_limit_is_checked_before_model_request(self):
        with patch("src.harness.agent.orchestrator.time.monotonic", side_effect=[0, 2, 2]):
            result, provider = self.run_script(
                [search_turn()], config=RuntimeConfig(max_duration_seconds=1)
            )
        self.assertEqual(result.terminal_reason, "time_budget_exhausted")
        self.assertEqual(provider.requests, [])

    def test_mandatory_context_cannot_be_silently_truncated(self):
        result, provider = self.run_script(
            [search_turn()], config=RuntimeConfig(context_max_chars=2000)
        )
        self.assertEqual(result.terminal_reason, "context_budget_exhausted")
        self.assertEqual(provider.requests, [])

    def test_context_keeps_complete_tool_pairs_and_persistent_working_memory(self):
        outputs = [
            plan_turn(),
            search_turn(),
            detail_turn(),
            fixtures.tool_turn("evidence", "record_evidence", evidence_arguments()),
            successful_provider(ASIN, "白色").outputs[-1],
        ]
        result, provider = self.run_script(
            outputs, config=RuntimeConfig(recent_turns=1)
        )
        messages = provider.requests[-1]["messages"]
        self.assertEqual(
            [m.role for m in messages], ["system", "user", "assistant", "tool"]
        )
        self.assertEqual(messages[-1].tool_call_id, messages[-2].tool_calls[0].id)
        context = json.loads(messages[1].content)
        self.assertEqual(context["memory"]["queries"], ["冰箱"])
        self.assertTrue(result.session.evaluation.strict_success)
        self.assertEqual(len(result.messages), 12)

    def test_private_evaluation_never_enters_actor_observations_or_memory(self):
        result, provider = self.run_script(successful_provider(ASIN, "白色").outputs)
        forbidden = {
            "reward",
            "reward_valid",
            "strict_success",
            "target_asin",
            "source_row_number",
            "required_option_values",
        }

        def assert_public(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(value))
                for v in value.values():
                    assert_public(v)
            elif isinstance(value, list):
                for v in value:
                    assert_public(v)

        for request in provider.requests:
            assert_public(json.loads(request["messages"][1].content))
        for event in result.events:
            if event.kind in {"observation", "memory_updated"}:
                assert_public(event.payload)
        self.assertEqual(result.session.history[-1].observation, {"submitted": True})
        self.assertTrue(result.session.evaluation.strict_success)

    def test_same_runner_does_not_share_task_memory(self):
        provider = fixtures.ScriptedProvider([search_turn(), search_turn("普通冰箱")])
        runner = AgentRunner(provider=provider, harness=self.harness, max_model_turns=1)
        first = runner.run(self.task_id)
        second = runner.run(self.harness.list_task_ids("all")[1])
        self.assertNotEqual(first.memory.session_id, second.memory.session_id)
        context = json.loads(provider.requests[1]["messages"][1].content)
        self.assertEqual(context["memory"]["queries"], [])
        self.assertEqual(context["memory"]["candidates"], [])

    def test_event_tampering_and_truncation_are_detected(self):
        result, _ = self.run_script([search_turn()])
        events = list(result.events)
        events[1] = events[1].model_copy(update={"payload": {"characters": 0}})
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            replay_events(events)
        with self.assertRaisesRegex(ValueError, "complete"):
            replay_events(result.events[:-1])

    def test_bundle_roundtrip_and_cli_replay_need_no_provider(self):
        result, _ = self.run_script(successful_provider(ASIN, "白色").outputs)
        directory = Path(self._temporary_directory.name) / "bundle"
        write_run_artifacts(result, directory)
        replayed = replay_run(directory)
        self.assertEqual(replayed["memory"], result.memory.model_dump(mode="json"))
        with self.assertRaises(FileExistsError):
            write_run_artifacts(result, directory)
        stdout = io.StringIO()
        with (
            patch(
                "src.harness.cli._build_provider",
                side_effect=AssertionError("network forbidden"),
            ),
            redirect_stdout(stdout),
        ):
            self.assertEqual(main(["replay", "--run-dir", str(directory)]), 0)
        self.assertEqual(
            json.loads(stdout.getvalue())["outcome"]["status"], "completed"
        )

    def test_schema_command_and_invalid_config_are_offline(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(["schema"]), 0)
        self.assertIn("AgentObservation", json.loads(stdout.getvalue()))
        config = Path(self._temporary_directory.name) / "invalid.json"
        config.write_text('{"max_searches": 0}', encoding="utf-8")
        with (
            redirect_stderr(io.StringIO()),
            patch("src.harness.cli._build_provider") as provider,
        ):
            self.assertEqual(
                main(
                    ["run", "--task-id", self.task_id, "--runtime-config", str(config)]
                ),
                2,
            )
            provider.assert_not_called()

    def test_changed_soft_constraint_invalidates_previous_assessment(self):
        first = plan_arguments()
        first["constraints"][0]["priority"] = "soft"
        updated = plan_arguments()
        updated["constraints"][0]["priority"] = "soft"
        updated["constraints"][0]["description"] = "新的偏好解释"
        result, _ = self.run_script(
            [
                plan_turn(first),
                search_turn(),
                detail_turn(),
                fixtures.tool_turn("evidence", "record_evidence", evidence_arguments()),
                plan_turn(updated),
            ]
        )
        self.assertEqual(result.memory.candidates[ASIN].assessments, {})
        self.assertEqual(result.task_profile.constraints[0].description, "新的偏好解释")

    def test_explicit_unknown_assessment_remains_unknown_in_context(self):
        result, provider = self.run_script(
            [
                plan_turn(),
                search_turn(),
                detail_turn(),
                fixtures.tool_turn(
                    "unknown", "record_evidence", evidence_arguments(status="unknown")
                ),
                fixtures.tool_turn("bad", "unknown_tool", {}),
            ]
        )
        context = json.loads(provider.requests[-1]["messages"][1].content)
        candidate = next(
            c for c in context["memory"]["candidates"] if c["asin"] == ASIN
        )
        self.assertEqual(candidate["unknown_constraints"], ["quiet"])
        self.assertEqual(
            result.memory.candidates[ASIN].assessments["quiet"].status, "unknown"
        )

    def test_observation_projection_failure_is_audited_as_runtime_failure(self):
        with patch(
            "src.harness.agent.observations.ObservationProjector.project",
            side_effect=ValueError("malformed adapter response"),
        ):
            result, _ = self.run_script([search_turn()])
        self.assertEqual(result.terminal_reason, "runtime_error")
        self.assertEqual(result.tool_call_count, 1)
        self.assertEqual(result.action_count, 1)
        self.assertEqual(result.memory.candidates, {})
        self.assertEqual(replay_events(result.events)["outcome"]["status"], "failed")

    def test_saved_final_state_tampering_is_detected(self):
        result, _ = self.run_script([search_turn()])
        directory = Path(self._temporary_directory.name) / "bundle"
        write_run_artifacts(result, directory)
        (directory / "final_memory.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from replayed"):
            replay_run(directory)

    def test_cli_loads_runtime_profiles_and_writes_replayable_bundle(self):
        root = Path(self._temporary_directory.name)
        runtime_path = root / "runtime.json"
        user_path = root / "user.json"
        runtime_path.write_text(
            '{"max_actions": 8, "max_searches": 1}', encoding="utf-8"
        )
        user_path.write_text('{"profile_id": "cli-shopper"}', encoding="utf-8")
        output = root / "cli-bundle"
        stdout = io.StringIO()
        with (
            patch(
                "src.harness.cli._build_provider",
                return_value=successful_provider(ASIN, "白色"),
            ),
            redirect_stdout(stdout),
        ):
            code = main(
                [
                    "run",
                    "--task-id",
                    self.task_id,
                    "--no-env-file",
                    "--products-path",
                    str(root / "products.json"),
                    "--index-path",
                    str(root / "products.sqlite3"),
                    "--runtime-config",
                    str(runtime_path),
                    "--user-profile",
                    str(user_path),
                    "--artifacts-dir",
                    str(output),
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["manifest"]["config"]["max_actions"], 8)
        self.assertEqual(
            replay_run(output)["user_profile"]["profile_id"], "cli-shopper"
        )

    def test_manifest_captures_defaults_and_redacts_nested_credentials(self):
        provider = successful_provider(ASIN, "白色")
        provider.default_parameters = {
            "temperature": 0.2,
            "max_tokens": 99,
            "nested": {"access_token": "sensitive"},
        }
        result = AgentRunner(
            provider=provider,
            harness=self.harness,
            provider_parameters={"temperature": 0.0},
        ).run(self.task_id)
        parameters = result.manifest["provider_parameters"]
        self.assertEqual(parameters["temperature"], 0.0)
        self.assertEqual(parameters["max_tokens"], 99)
        self.assertEqual(parameters["nested"]["access_token"], "***")


if __name__ == "__main__":
    unittest.main()
