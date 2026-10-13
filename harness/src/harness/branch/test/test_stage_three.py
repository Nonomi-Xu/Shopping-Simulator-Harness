"""Real-user lifecycle, resume integrity, preference isolation and public sources."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from urllib.error import HTTPError

from src.harness.agent import (
    AgentRunner,
    RuntimeConfig,
    UserReply,
    UserProfile,
    AgentRunResult,
    replay_run,
    write_run_artifacts,
)
from src.harness.branch.preferences import PreferenceStore, PreferenceConflict
from src.harness.agent.orchestrator import validate_checkpoint
from src.harness.agent.profile import validate_constraints
from src.harness.agent.contracts import Constraint, TaskProfile, UserMessage
from src.harness.agent.memory import MemoryStore
from src.harness.agent.events import replay_events
from src.harness.catalog import CatalogDataError
from src.harness.branch.interactive import InteractiveHarness
from src.harness.harness import HarnessError
from src.harness.sources import JsonProductSource, HttpProductSource, _NoRedirect
from src.harness.schemas import SearchProductsAction, PublicTask
from src.harness.cli import main
from src.harness.test.test_agent_runner import ScriptedProvider, tool_turn


def product(asin, color):
    return {
        "asin": asin,
        "title": f"{color}耳机",
        "category": "数码耳机",
        "brand": "Acme",
        "price_min": 100,
        "price_max": 100,
        "pricing": [100],
        "attributes": [color],
        "customization_options": {
            "颜色": [{"value": color, "price": 100, "is_available": True}]
        },
        "variant_combinations": [],
        "instructions": "PRIVATE_SOURCE_SENTINEL",
    }


def plan(color="白色", message_id=None):
    constraint = {
        "constraint_id": "color",
        "description": f"需要{color}",
        "priority": "hard",
        "source": "user_message" if message_id else "user_query",
        "source_quote": color,
    }
    if message_id:
        constraint["source_message_id"] = message_id
    return tool_turn(
        "plan",
        "update_plan",
        {
            "constraints": [constraint],
            "queries": ["耳机"],
            "change_summary": "更新颜色要求",
        },
    )


def ask():
    return tool_turn(
        "ask",
        "ask_user",
        {
            "question": "颜色是否需要调整？",
            "information_gap": "确认最终颜色",
            "constraint_ids": ["color"],
        },
    )


def evidence(asin, color, action):
    return tool_turn(
        "evidence",
        "record_evidence",
        {
            "asin": asin,
            "constraint_id": "color",
            "status": "satisfied",
            "observation_id": f"chat-test:action_{action:04d}:observation",
            "field_path": "/attributes/0",
            "quote": color,
            "explanation": "详情注明颜色",
        },
    )


def submission(asin="black", color="黑色"):
    return tool_turn(
        "submit",
        "submit_recommendations",
        {
            "recommendations": [
                {
                    "asin": asin,
                    "selected_options": {"颜色": color},
                    "reason": f"根据最新要求推荐{color}耳机",
                }
            ]
        },
    )


class InteractiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "products.json"
        self.path.write_text(
            json.dumps(
                [product("white", "白色"), product("black", "黑色")], ensure_ascii=False
            )
        )
        self.harness = self.new_harness()
        self.config = RuntimeConfig(interaction_mode="interactive", recent_turns=1)

    def new_harness(self):
        harness = InteractiveHarness(JsonProductSource(self.path, currency="CNY"))
        self.addCleanup(harness.close)
        return harness

    def pause(self, outputs=None, **kwargs):
        outputs = outputs or [
            plan(),
            tool_turn("search", "search_products", {"query": "耳机"}),
            tool_turn("get", "get_product", {"asin": "white"}),
            evidence("white", "白色", 3),
            ask(),
        ]
        provider = ScriptedProvider(outputs)
        runner = AgentRunner(
            provider=provider, harness=self.harness, config=self.config, **kwargs
        )
        paused = runner.run_query("寻找白色耳机", session_id="chat-test")
        self.assertEqual(paused.status, "waiting_for_user", paused.error_message)
        return paused, provider, runner

    def test_resume_after_new_process_preserves_evidence_budget_and_revises_user_needs(
        self,
    ):
        paused, _, _ = self.pause()
        self.assertEqual(paused.session.state.status, "active")
        self.assertIsNone(paused.session.evaluation)
        directory = write_run_artifacts(paused, self.root / "waiting")
        replay_run(directory)
        restored = AgentRunResult.model_validate_json(
            (directory / "checkpoint.json").read_text()
        )
        provider = ScriptedProvider(
            [
                plan("黑色", "user_0001"),
                tool_turn("get-black", "get_product", {"asin": "black"}),
                evidence("black", "黑色", 7),
                submission(),
            ]
        )
        harness = self.new_harness()
        runner = AgentRunner(provider=provider, harness=harness, config=self.config)
        result = runner.resume(
            restored, UserReply(text="改成黑色", replace_constraint_ids=["color"])
        )
        self.assertEqual(result.status, "completed", result.error_message)
        self.assertIsNone(result.session.evaluation)
        self.assertEqual(result.action_count, 9)
        self.assertEqual(result.search_count, 1)
        self.assertEqual(result.tool_call_count, 4)
        self.assertEqual(result.model_turn_count, 9)
        self.assertEqual(result.task_profile.revision, 1)
        self.assertEqual(
            result.task_profile.constraints[0].source_message_id, "user_0001"
        )
        self.assertEqual(result.memory.candidates["white"].assessments, {})
        self.assertIn(
            "chat-test:action_0003:observation", result.memory.evidence_archive
        )
        payload = json.loads(provider.requests[0]["messages"][1].content)
        self.assertEqual(payload["task_profile"]["requirements_status"], "unparsed")
        self.assertEqual(payload["budget"]["actions_remaining"], 35)
        self.assertEqual(
            payload["task_profile"]["user_messages"][0]["text"], "改成黑色"
        )
        self.assertNotIn("PRIVATE_SOURCE_SENTINEL", result.model_dump_json())
        self.assertEqual(
            result.memory.evidence_sources["chat-test:action_0007:observation"][
                "currency"
            ],
            "CNY",
        )
        replayed = replay_run(write_run_artifacts(result, self.root / "completed"))
        self.assertEqual(
            replayed["task_profile"], result.task_profile.model_dump(mode="json")
        )
        with self.assertRaisesRegex(ValueError, "consumed"):
            runner.resume(restored, "重复回复")

    def test_no_automatic_relaxation_of_hard_constraints(self):
        paused, provider, runner = self.pause()
        provider.outputs = [plan("黑色", "user_0001"), ask()]
        result = runner.resume(paused, "我也喜欢黑色")
        self.assertEqual(result.status, "waiting_for_user", result.error_message)
        rejected = [
            e.payload
            for e in result.events
            if e.kind == "observation" and e.payload["status"] == "rejected"
        ]
        self.assertIn("preserve existing hard", rejected[-1]["message"])
        self.assertEqual(result.task_profile.constraints[0].source_quote, "白色")
        self.assertEqual(result.model_turn_count, 7)

    def test_resume_must_reparse_before_submission(self):
        paused, provider, runner = self.pause()
        provider.outputs = [submission("white", "白色"), ask()]
        result = runner.resume(paused, "仍然白色")
        self.assertEqual(result.status, "waiting_for_user")
        issues = [
            e.payload
            for e in result.events
            if e.kind == "observation"
            and e.payload.get("error_code") == "submission_not_ready"
        ]
        self.assertIn(
            "requirements_unparsed", [i["code"] for i in issues[0]["data"]["issues"]]
        )
        self.assertEqual(
            result.session.state.step_count, paused.session.state.step_count
        )

    def test_resume_does_not_reset_model_action_or_token_budgets(self):
        self.config = RuntimeConfig(interaction_mode="interactive", max_actions=5)
        paused, provider, runner = self.pause(max_model_turns=5)
        result = runner.resume(paused, "仍然白色")
        self.assertEqual(result.terminal_reason, "max_actions")
        self.assertEqual(len(provider.requests), 5)
        self.assertEqual(result.usage, paused.usage)

    def test_model_turn_budget_is_cumulative(self):
        paused, provider, runner = self.pause(max_model_turns=5)
        result = runner.resume(paused, "仍然白色")
        self.assertEqual(result.terminal_reason, "max_model_turns")
        self.assertEqual(len(provider.requests), 5)

    def test_wait_time_is_excluded_from_active_time(self):
        self.config = RuntimeConfig(
            interaction_mode="interactive", max_duration_seconds=10
        )
        with patch("src.harness.agent.orchestrator.time.monotonic", return_value=10):
            paused, provider, runner = self.pause()
        provider.outputs = [ask()]
        with patch("src.harness.agent.orchestrator.time.monotonic", return_value=10000):
            result = runner.resume(paused, "请再确认")
        self.assertEqual(result.status, "waiting_for_user", result.error_message)
        self.assertEqual(result.active_duration_seconds, 0)

    def test_invalid_and_tampered_checkpoint_rejected_before_any_model_call(self):
        paused, provider, runner = self.pause()
        for changes in (
            {"action_count": 0},
            {"search_count": 0},
            {"active_duration_seconds": 500},
            {"user_profile": UserProfile(profile_id="someone-else")},
            {"messages": []},
            {"model_turn_count": 0},
            {"session": paused.session.model_copy(update={"history": []})},
        ):
            with self.subTest(changes=list(changes)):
                with self.assertRaises(ValueError):
                    runner.resume(paused.model_copy(update=changes), "白色")
        with self.assertRaises(ValueError):
            runner.resume(
                paused, UserReply(text="修改", replace_constraint_ids=["missing"])
            )
        self.assertEqual(len(provider.requests), 5)
        # An invalid attempt must not consume the checkpoint.
        provider.outputs = [ask()]
        self.assertEqual(runner.resume(paused, "白色").status, "waiting_for_user")

    def test_changed_source_or_config_cannot_silently_resume(self):
        paused, provider, _ = self.pause()
        changed = AgentRunner(
            provider=provider,
            harness=self.harness,
            config=self.config.model_copy(update={"max_actions": 99}),
        )
        with self.assertRaisesRegex(ValueError, "configuration"):
            changed.resume(paused, "白色")
        self.path.write_text(json.dumps([product("new", "白色")]))
        changed_source = AgentRunner(
            provider=provider, harness=self.new_harness(), config=self.config
        )
        with self.assertRaisesRegex(ValueError, "source"):
            changed_source.resume(paused, "白色")

    def test_question_budget_hides_and_rejects_extra_questions(self):
        self.config = RuntimeConfig(
            interaction_mode="interactive", max_user_questions=1
        )
        paused, provider, runner = self.pause(max_model_turns=6)
        provider.outputs = [ask()]
        result = runner.resume(paused, "白色")
        self.assertEqual(result.terminal_reason, "max_model_turns")
        self.assertNotIn("ask_user", [t.name for t in provider.requests[-1]["tools"]])
        observation = [e.payload for e in result.events if e.kind == "observation"][-1]
        self.assertEqual(observation["status"], "rejected")

    def test_user_message_provenance_must_match_real_reply(self):
        task = TaskProfile(
            task=PublicTask(task_id="task_" + "a" * 24, query="耳机"),
            user_messages=[UserMessage(message_id="user_0001", text="黑色")],
        )
        constraint = Constraint(
            constraint_id="color",
            description="颜色",
            priority="hard",
            source="user_message",
            source_message_id="fake",
            source_quote="黑色",
        )
        with self.assertRaises(ValueError):
            validate_constraints([constraint], task, UserProfile())
        validate_constraints(
            [constraint.model_copy(update={"source_message_id": "user_0001"})],
            task,
            UserProfile(),
        )

    def test_source_missing_fields_remain_unobserved(self):
        self.path.write_text(
            json.dumps([{"asin": "minimal", "title": "耳机", "category": "耳机"}])
        )
        harness = self.new_harness()
        snapshot = harness.create_interactive_session("耳机")
        sid = snapshot.state.session_id
        harness.dispatch(sid, "search_products", {"query": "耳机"})
        step = harness.dispatch(sid, "get_product", {"asin": "minimal"})
        self.assertNotIn("customization_options", step.observation)
        self.assertNotIn("price_min", step.observation)

    def test_external_refresh_invalidates_old_assessments_and_changed_quotes(self):
        row = {
            "asin": "white",
            "title": "白色耳机",
            "category": "耳机",
            "rank": 1,
            "bm25_score": 1,
        }
        source = HttpSourceTest().source(
            [
                {"items": [row], "page": {"offset": 0}},
                product("white", "白色"),
                {**product("white", "白色"), "attributes": ["黑色"]},
            ]
        )
        self.harness = InteractiveHarness(source)
        self.addCleanup(self.harness.close)
        paused, provider, runner = self.pause()
        provider.outputs = [
            plan(),
            tool_turn("refresh", "get_product", {"asin": "white"}),
            evidence("white", "白色", 3),
            ask(),
        ]
        result = runner.resume(paused, "还是白色")
        self.assertEqual(result.status, "waiting_for_user", result.error_message)
        self.assertEqual(result.memory.candidates["white"].assessments, {})
        errors = [
            e.payload
            for e in result.events
            if e.kind == "observation" and e.payload["status"] == "rejected"
        ]
        self.assertIn("superseded", errors[-1]["message"])

    def test_offline_mode_has_no_ask_tool_or_interactive_entrypoint(self):
        from src.harness.agent import ToolRegistry

        self.assertNotIn("ask_user", [t.name for t in ToolRegistry().definitions])
        runner = AgentRunner(provider=ScriptedProvider([]), harness=self.harness)
        with self.assertRaisesRegex(ValueError, "interaction_mode"):
            runner.run_query("耳机")
        with self.assertRaisesRegex(HarnessError, "splits"):
            self.harness.list_task_ids("all")

    def test_token_budget_is_cumulative_across_reply(self):
        self.config = RuntimeConfig(interaction_mode="interactive", max_total_tokens=80)
        paused, provider, runner = self.pause()
        provider.outputs = [plan()]
        result = runner.resume(paused, "仍然白色")
        self.assertEqual(result.terminal_reason, "token_budget_exhausted")
        self.assertEqual(result.action_count, 5)
        self.assertEqual(result.model_turn_count, 6)
        self.assertEqual(result.usage.total_tokens, 84)

    def test_preferences_cli_and_interactive_cli_resume(self):
        db = self.root / "preferences.sqlite3"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                main(
                    [
                        "preferences",
                        "set",
                        "--db",
                        str(db),
                        "--profile-id",
                        "alice",
                        "--name",
                        "color",
                        "--value",
                        "白色",
                    ]
                ),
                0,
            )
        provider = ScriptedProvider(
            [
                plan(),
                tool_turn("s", "search_products", {"query": "耳机"}),
                tool_turn("g", "get_product", {"asin": "white"}),
                ask(),
            ]
        )
        common = ["--source-json", str(self.path), "--currency", "CNY", "--no-env-file"]
        with (
            patch("src.harness.cli._build_provider", return_value=provider),
            redirect_stdout(io.StringIO()),
        ):
            code = main(
                [
                    "chat",
                    "--query",
                    "白色耳机",
                    *common,
                    "--preferences-db",
                    str(db),
                    "--profile-id",
                    "alice",
                    "--artifacts-dir",
                    str(self.root / "cli-wait"),
                ]
            )
        self.assertEqual(code, 0)
        checkpoint = AgentRunResult.model_validate_json(
            (self.root / "cli-wait/checkpoint.json").read_text()
        )
        self.assertEqual(checkpoint.user_profile.profile_id, "alice")
        provider.outputs = [ask()]
        with (
            patch("src.harness.cli._build_provider", return_value=provider),
            redirect_stdout(io.StringIO()),
        ):
            code = main(
                [
                    "resume",
                    "--run-dir",
                    str(self.root / "cli-wait"),
                    "--reply",
                    "继续白色",
                    *common,
                    "--artifacts-dir",
                    str(self.root / "cli-again"),
                ]
            )
        self.assertEqual(code, 0)
        replay_run(self.root / "cli-again")


class PreferencesTest(unittest.TestCase):
    def test_explicit_updates_persist_are_isolated_and_detect_stale_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefs.sqlite3"
            first = PreferenceStore(path)
            self.assertEqual(first.set("alice", "颜色", "白色", expected_revision=0), 1)
            second = PreferenceStore(path)
            profile, revision = second.load("alice")
            self.assertEqual(profile.preferences[0].value, "白色")
            self.assertEqual(second.load("bob")[0].preferences, [])
            self.assertEqual(
                second.set("alice", "颜色", "黑色", expected_revision=revision), 2
            )
            with self.assertRaises(PreferenceConflict):
                first.set("alice", "颜色", "红色", expected_revision=1)
            second.remove("alice", "颜色", expected_revision=2)
            self.assertEqual(first.load("alice")[0].preferences, [])
            self.assertEqual(
                [r["operation"] for r in first.history("alice")],
                ["remove", "set", "set"],
            )
            self.assertTrue(
                all(r["source"] == "explicit_caller" for r in first.history("alice"))
            )


class HttpSourceTest(unittest.TestCase):
    def source(self, responses):
        source = HttpProductSource(
            "https://catalog.example", api_key="private-credential", currency="CNY"
        )
        opener = Mock()
        replies = []
        for raw in responses:
            response = Mock()
            response.read.return_value = json.dumps(raw).encode()
            context = Mock()
            context.__enter__ = Mock(return_value=response)
            context.__exit__ = Mock(return_value=False)
            replies.append(context)
        opener.open.side_effect = replies
        source._opener = opener
        return source

    def test_gateway_contract_projection_provenance_and_changed_observations(self):
        item = {
            "asin": "black",
            "title": "耳机",
            "category": "数码",
            "bm25_score": 1,
            "rank": 1,
            "target_asin": "PRIVATE_SOURCE_SENTINEL",
        }
        source = self.source(
            [
                {"items": [item], "page": {"offset": 0, "has_more": False}},
                product("black", "黑色"),
                {**product("black", "黑色"), "price_min": 200, "price_max": 200},
            ]
        )
        rows, page = source.search_page(SearchProductsAction(query="耳机"))
        request = source._opener.open.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(json.loads(request.data)["query"], "耳机")
        self.assertEqual(
            request.get_header("Authorization"), "Bearer private-credential"
        )
        source.get_product("black")
        old_hash = source.last_observation_source["content_sha256"]
        self.assertEqual(source.get_product("black").price_min, 200)
        self.assertNotEqual(old_hash, source.last_observation_source["content_sha256"])
        self.assertNotIn("private-credential", json.dumps(source.manifest))
        self.assertNotIn("PRIVATE_SOURCE_SENTINEL", rows[0].model_dump_json())

    def test_bad_gateway_responses_and_secrets_do_not_reach_actor(self):
        for raw in (
            {"items": [], "page": {"has_more": True, "next_offset": 0}},
            {"items": [], "page": {"offset": 99}},
            {"not_items": "bad"},
        ):
            with self.subTest(raw=raw), self.assertRaises(CatalogDataError):
                self.source([raw]).search_page(SearchProductsAction(query="耳机"))
        source = self.source([product("wrong-id", "黑色")])
        with self.assertRaises(CatalogDataError):
            source.get_product("black")
        source = self.source([])
        source._opener.open.side_effect = HTTPError(
            "https://catalog.example?secret", 401, "secret body", {}, None
        )
        with self.assertRaises(CatalogDataError) as raised:
            source.get_product("black")
        self.assertNotIn("secret", str(raised.exception))
        source = self.source([product("black", "黑色")])
        source.max_response_bytes = 10
        with self.assertRaises(CatalogDataError):
            source.get_product("black")

    def test_credentials_in_url_and_redirects_are_rejected(self):
        with self.assertRaises(ValueError):
            HttpProductSource("https://user:password@example.com")
        with self.assertRaises(HTTPError):
            _NoRedirect().redirect_request(
                Mock(full_url="https://catalog.example"),
                None,
                302,
                "redirect",
                {},
                "https://elsewhere.example",
            )

    def test_external_availability_cannot_be_inferred_from_schema_defaults(self):
        raw = product("black", "黑色")
        del raw["customization_options"]["颜色"][0]["is_available"]
        with self.assertRaises(CatalogDataError):
            self.source([raw]).get_product("black")


if __name__ == "__main__":
    unittest.main()
