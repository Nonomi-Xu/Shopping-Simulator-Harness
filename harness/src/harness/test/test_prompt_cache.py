"""Regression tests for the cache-shaped prompt layout.

Provider prefix caches (DeepSeek context caching) reuse only the longest leading
token run that is byte-identical to an earlier request.  The runtime keeps the
system policy invariant, serializes the per-turn state block stable-first, and
drops prompt content that carries no information the prompt does not already
have (concluded reasoning, runtime-tool echoes, title-echo snippets).  These tests
pin those properties.
"""

import json
import unittest

from src.harness.agent import (
    AgentRunner,
    AgentProfile,
    RuntimeConfig,
    SearchPlan,
    TaskProfile,
    UserProfile,
)
from src.harness.agent.context import (
    ECHO_OBSERVATION_TOOLS,
    STATE_FIELD_ORDER,
    ContextBuilder,
    RUNTIME_INSTRUCTIONS,
    prompt_message,
    state_json,
)
from src.harness.agent.contracts import BudgetState
from src.harness.agent.memory import MemoryStore, summary_view
from src.harness.providers import ChatMessage, canonical_json
from src.harness.schemas import PublicTask
from src.harness.test import test_agent_runner as fixtures
from src.harness.test.test_rollout import successful_provider


def shared_prefix(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def sample_payload():
    return {
        "profile": {"agent_id": "shopping-agent", "version": "shopping-profile-v1"},
        "user_profile": {"profile_id": "anonymous", "locale": "zh-CN"},
        "task_profile": {
            "task": {"task_id": "task_" + "0" * 24, "query": "寻找白色静音冰箱"},
            "constraints": [],
            "requirements_status": "parsed",
        },
        "plan": {"plan_id": "plan-1", "revision": 1, "phase": "search"},
        "memory": {
            "queries": ["冰箱"],
            "candidates": [{"asin": "000000000001", "inspected": True}],
        },
        "budget": {"actions_remaining": 10, "searches_remaining": 3},
        "allowed_tools": ["update_plan", "search_products"],
        "action_proposals": [{"tool_name": "search_products"}],
    }


class StateBlockOrderTest(unittest.TestCase):
    def test_fields_are_serialized_stable_first(self):
        payload = sample_payload()
        self.assertEqual(list(json.loads(state_json(payload))), list(STATE_FIELD_ORDER))
        # The most volatile field is last, the identity fields are first.
        self.assertEqual(list(json.loads(state_json(payload)))[0], "profile")
        self.assertEqual(list(json.loads(state_json(payload)))[-1], "action_proposals")

    def test_unknown_fields_follow_the_known_ones(self):
        payload = {**sample_payload(), "agent_state": {"role": "planner"}}
        keys = list(json.loads(state_json(payload)))
        self.assertEqual(keys[: len(STATE_FIELD_ORDER)], list(STATE_FIELD_ORDER))
        self.assertEqual(keys[-1], "agent_state")

    def test_stable_fields_keep_more_of_the_block_cacheable(self):
        """The whole point: a turn that only changes the volatile tail must leave
        the leading bytes of the state block untouched."""
        first = sample_payload()
        second = json.loads(json.dumps(first))
        second["budget"]["actions_remaining"] = 9
        second["action_proposals"] = [{"tool_name": "get_product"}]

        keepable = shared_prefix(state_json(first), state_json(second))
        alphabetical = shared_prefix(canonical_json(first), canonical_json(second))
        self.assertGreater(keepable, alphabetical)
        # The reusable prefix reaches the first volatile field and stops there.
        head = state_json(first)
        self.assertGreaterEqual(keepable, head.index('"budget":'))
        self.assertLess(keepable, head.index('"action_proposals":'))


class VariantSuffixTest(unittest.TestCase):
    """Mode- and role-specific text must trail the shared policy."""

    def build(self, variant_suffix: str):
        task = TaskProfile(
            task=PublicTask(task_id="task_" + "0" * 24, query="寻找白色静音冰箱")
        )
        history = [
            ChatMessage(role="system", content="skill"),
            ChatMessage(role="user", content='{"query":"寻找白色静音冰箱"}'),
        ]
        return ContextBuilder().build(
            system_prompt="SHARED-SKILL-POLICY",
            profile=AgentProfile(),
            user=UserProfile(),
            task=task,
            plan=SearchPlan(plan_id="plan-1", objective="找冰箱", queries=["冰箱"]),
            memory=MemoryStore("session"),
            budget=BudgetState(
                actions_remaining=10,
                searches_remaining=3,
                environment_steps_remaining=5,
                model_turns_remaining=4,
                total_tokens=0,
                usage_complete=True,
            ),
            allowed_tools=["update_plan"],
            config=RuntimeConfig(),
            history=history,
            variant_suffix=variant_suffix,
        )[0]

    def test_shared_policy_comes_first_and_the_variant_trails(self):
        core = "SHARED-SKILL-POLICY" + RUNTIME_INSTRUCTIONS
        single, coordinator = self.build(""), self.build("ROLE: planner")
        self.assertEqual(single[0].content, core)
        self.assertEqual(coordinator[0].content, core + "ROLE: planner")
        self.assertTrue(coordinator[0].content.startswith(core))
        # Only the system message changes; the state block and transcript do not.
        self.assertEqual(
            [m.as_provider_payload() for m in single[1:]],
            [m.as_provider_payload() for m in coordinator[1:]],
        )


class PromptDietTest(unittest.TestCase):
    """Content the prompt must not pay for twice."""

    def test_concluded_reasoning_is_not_replayed(self):
        message = fixtures.tool_turn(
            "call-1", "update_plan", {"queries": ["冰箱"], "change_summary": "计划"}
        ).as_message().model_copy(update={"reasoning_content": "先比较价格，再核验颜色。"})
        projected = prompt_message(message)
        self.assertIsNone(projected.reasoning_content)
        # The audit copy keeps the original reasoning.
        self.assertEqual(message.reasoning_content, "先比较价格，再核验颜色。")
        self.assertEqual(projected.tool_calls, message.tool_calls)
        # A user or tool message is never touched by the reasoning rule.
        user = ChatMessage(role="user", content="寻找冰箱")
        self.assertIs(prompt_message(user), user)

    def test_echo_observation_keeps_only_what_the_state_does_not_repeat(self):
        payload = {
            "observation_id": "s:action_0001:observation",
            "action_id": "s:action_0001",
            "tool_name": "update_plan",
            "status": "success",
            "data": {
                "plan": {"plan_id": "p1", "phase": "search"},
                "task_profile": {"constraints": [{"constraint_id": "quiet"}]},
            },
            "error_code": None,
            "message": "",
            "environment_step_id": None,
            "terminated": False,
            "terminal_reason": None,
            "allowed_next_tools": ["update_plan", "search_products"],
            "budget": {"actions_remaining": 39},
            "projection": {"phase": "search"},
        }
        projected = json.loads(
            prompt_message(
                ChatMessage(
                    role="tool",
                    name="update_plan",
                    tool_call_id="call-1",
                    content=json.dumps(payload),
                )
            ).content
        )
        self.assertEqual(projected["status"], "success")
        self.assertEqual(projected["observation_id"], payload["observation_id"])
        for dropped in ("data", "projection", "budget", "allowed_next_tools"):
            self.assertNotIn(dropped, projected)

    def test_evidence_echo_keeps_only_the_verdicts(self):
        payload = {
            "observation_id": "s:action_0003:observation",
            "tool_name": "record_evidence_batch",
            "status": "success",
            "data": {
                "assessments": [
                    {
                        "asin": "000000000001",
                        "constraint_id": "quiet",
                        "status": "satisfied",
                        "observation_id": "s:action_0002:observation",
                        "field_path": "/attributes/0",
                        "quote": "静音",
                        "explanation": "公开属性支持静音要求",
                    }
                ]
            },
        }
        projected = json.loads(
            prompt_message(
                ChatMessage(
                    role="tool",
                    name="record_evidence_batch",
                    tool_call_id="call-2",
                    content=json.dumps(payload),
                )
            ).content
        )
        self.assertEqual(
            projected["assessments"],
            [{"asin": "000000000001", "constraint_id": "quiet", "status": "satisfied"}],
        )
        self.assertNotIn("quote", projected["assessments"][0])

    def test_non_echo_observations_are_untouched(self):
        payload = {"tool_name": "get_product", "status": "success", "data": {"asin": "a"}}
        message = ChatMessage(
            role="tool", name="get_product", tool_call_id="call-3", content=json.dumps(payload)
        )
        self.assertIs(prompt_message(message), message)
        self.assertNotIn("get_product", ECHO_OBSERVATION_TOOLS)

    def test_summary_view_drops_title_and_category_echoes(self):
        summary = {
            "asin": "000000000001",
            "title": "白色静音冰箱",
            "brand": "示例",
            "category": "家电›冰箱",
            "price_min": 799.0,
            "price_max": 899.0,
            "rank": 1,
            "bm25_score": 26.9767384609796,
            "match_snippets": [
                {"field_path": "/title", "text": "白色静音冰箱"},
                {"field_path": "/description", "text": "冰箱具备静音与节能双模式"},
                {"field_path": "/category", "text": "家电›冰箱"},
            ],
        }
        view = summary_view(summary)
        self.assertEqual(
            view["match_snippets"],
            [{"field_path": "/description", "text": "冰箱具备静音与节能双模式"}],
        )
        self.assertEqual(view["bm25_score"], 26.977)
        self.assertEqual(view["asin"], "000000000001")
        self.assertEqual(view["price_min"], 799.0)

    def test_summary_view_keeps_an_empty_snippet_list_out(self):
        view = summary_view({"asin": "a", "title": "t", "match_snippets": []})
        self.assertNotIn("match_snippets", view)


class PromptPrefixCacheTest(unittest.TestCase):
    setUp = fixtures.AgentRunnerTest.setUp

    def run_script(self, outputs=None):
        if outputs is None:
            provider = successful_provider(fixtures.TARGET_ASIN, "白色")
            outputs = list(provider.outputs)
        provider = fixtures.ScriptedProvider(list(outputs))
        result = AgentRunner(
            provider=provider,
            harness=self.harness,
            max_model_turns=len(outputs),
        ).run(self.task_id, max_steps=20, session_id="prompt-cache")
        return result, provider

    def test_system_policy_and_state_prefix_survive_between_turns(self):
        result, provider = self.run_script()
        self.assertTrue(result.session.evaluation.strict_success)
        requests = [request["messages"] for request in provider.requests]
        self.assertGreater(len(requests), 2)
        system_prompt = requests[0][0].content
        for messages in requests:
            self.assertEqual(messages[0].content, system_prompt)
            self.assertEqual(messages[1].role, "user")
            self.assertEqual(list(json.loads(messages[1].content))[0], "profile")
        for previous, current in zip(requests, requests[1:]):
            # The identity and plan fields never change inside a run.
            self.assertGreater(
                shared_prefix(previous[1].content, current[1].content), 100
            )
        self.assertEqual(result.status, "completed")

    def test_reasoning_is_kept_in_the_audit_but_not_sent_again(self):
        turn = fixtures.requirement_turn("白色").model_copy(
            update={"reasoning_content": "先记录硬约束，再搜索白色冰箱。"}
        )
        outputs = [turn, *list(successful_provider(fixtures.TARGET_ASIN, "白色").outputs)[1:]]
        result, provider = self.run_script(outputs)
        # The audit trail keeps the model's own reasoning ...
        self.assertEqual(result.messages[2].reasoning_content, "先记录硬约束，再搜索白色冰箱。")
        # ... but no later request replays it.
        for request in provider.requests[1:]:
            for message in request["messages"]:
                self.assertIsNone(getattr(message, "reasoning_content", None))

    def test_runtime_tool_echoes_are_projected_in_sent_prompts(self):
        _, provider = self.run_script()
        seen = {}
        for request in provider.requests:
            for message in request["messages"]:
                if message.role != "tool":
                    continue
                payload = json.loads(message.content)
                seen.setdefault(payload["tool_name"], payload)
        self.assertIn("update_plan", seen)
        self.assertNotIn("projection", seen["update_plan"])
        self.assertNotIn("allowed_next_tools", seen["update_plan"])
        self.assertEqual(seen["update_plan"]["status"], "success")
        self.assertLess(len(canonical_json(seen["update_plan"])), 400)

    def test_context_events_record_the_layout(self):
        result, _ = self.run_script()
        context_events = [e for e in result.events if e.kind == "context_built"]
        self.assertTrue(context_events)
        for event in context_events:
            self.assertEqual(
                event.payload["prompt_layout"], "stable-first-state-fields"
            )
            messages = event.payload["messages"]
            self.assertEqual(messages[0]["role"], "system")
            self.assertIn("plan", messages[1]["content"])
            self.assertIn("allowed_tools", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
