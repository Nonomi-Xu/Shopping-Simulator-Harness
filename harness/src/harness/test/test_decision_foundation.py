"""Decision coverage, archived evidence and recoverable submission preflight."""

import json
from pathlib import Path
import unittest

from pydantic import ValidationError

from src.harness.agent import (
    AgentRunner,
    AgentSubmission,
    PlanUpdate,
    ReadEvidence,
    RuntimeConfig,
    TaskProfile,
    UserProfile,
    replay_run,
    write_run_artifacts,
)
from src.harness.agent.memory import MemoryStore
from src.harness.agent.planner import Planner
from src.harness.agent.submission import SubmissionNotReady, check_submission
from src.harness.schemas import PublicTask
from src.harness.test import test_agent_runner as f


SESSION = "decision-foundation"
ASIN = f.TARGET_ASIN
OTHER = f.ALTERNATIVE_ASIN


def search():
    return f.tool_turn("search", "search_products", {"query": "冰箱"})


def detail(asin=ASIN):
    return f.tool_turn(f"detail-{asin}", "get_product", {"asin": asin})


def recommendation(asin=ASIN, color="白色", reason="详情确认白色可选"):
    return {"asin": asin, "selected_options": {"颜色": color}, "reason": reason}


def submit(**overrides):
    return f.tool_turn(
        "submit",
        "submit_recommendations",
        {
            "recommendations": [recommendation()],
            **overrides,
        },
    )


def color_assessment(asin=ASIN, action=3, **overrides):
    return {
        "asin": asin,
        "constraint_id": "color",
        "status": "satisfied",
        "observation_id": f"{SESSION}:action_{action:04d}:observation",
        "field_path": "/customization_options/颜色/0/value",
        "quote": "白色",
        "explanation": "详情存在所需颜色",
        **overrides,
    }


def observations(result):
    return [e.payload for e in result.events if e.kind == "observation"]


class DecisionFoundationTest(unittest.TestCase):
    setUp = f.AgentRunnerTest.setUp

    def run_script(self, outputs, **kwargs):
        provider = f.ScriptedProvider(outputs)
        result = AgentRunner(
            provider=provider,
            harness=self.harness,
            max_model_turns=len(outputs),
            **kwargs,
        ).run(self.task_id, session_id=SESSION)
        return result, provider

    def test_unparsed_requirements_never_enable_submission_or_advance_to_submit(self):
        result, provider = self.run_script([search(), detail(), submit()])
        self.assertEqual(result.task_profile.requirements_status, "unparsed")
        context = json.loads(provider.requests[-1]["messages"][1].content)
        self.assertEqual(context["plan"]["phase"], "verify")
        self.assertNotIn("submit_recommendations", context["allowed_tools"])
        self.assertNotIn(
            "submit_recommendations", observations(result)[1]["allowed_next_tools"]
        )
        self.assertEqual(observations(result)[-1]["error_code"], "submission_not_ready")
        self.assertEqual(result.tool_call_count, 2)
        self.assertIsNone(result.session.evaluation)
        self.assertEqual(result.session.state.status, "active")

    def test_empty_constraint_analysis_requires_explicit_reason(self):
        with self.assertRaisesRegex(ValidationError, "no_constraints_reason"):
            PlanUpdate(queries=["商品"], change_summary="已分析")
        task = TaskProfile(
            task=PublicTask(task_id=self.task_id, query="随便推荐一件商品")
        )
        planner = Planner()
        _, parsed = planner.update(
            planner.start(task, SESSION),
            PlanUpdate(
                queries=["商品"],
                change_summary="已分析",
                no_constraints_reason="用户没有限定商品类型或属性",
            ),
            task,
            UserProfile(),
            set(),
        )
        self.assertEqual(parsed.requirements_status, "parsed")
        self.assertEqual(parsed.constraints, [])
        self.assertTrue(parsed.no_constraints_reason)

    def test_unknown_hard_constraint_rejects_then_recovers_without_environment_step(
        self,
    ):
        result, provider = self.run_script(
            [
                f.requirement_turn("白色"),
                search(),
                detail(),
                submit(),
                f.tool_turn("evidence", "record_evidence", color_assessment()),
                submit(),
            ]
        )
        blocked = observations(result)[3]
        self.assertEqual(blocked["data"]["issues"][0]["status"], "unknown")
        self.assertIsNone(blocked["environment_step_id"])
        self.assertEqual(blocked["budget"]["environment_steps_remaining"], 5)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.action_count, 6)
        self.assertEqual(result.tool_call_count, 3)
        self.assertTrue(result.session.evaluation.strict_success)
        before = json.loads(provider.requests[3]["messages"][1].content)["memory"][
            "candidates"
        ][0]
        after = json.loads(provider.requests[-1]["messages"][1].content)["memory"][
            "candidates"
        ][0]
        self.assertFalse(before["evidence_ready"])
        self.assertTrue(after["evidence_ready"])

    def test_parsed_request_without_constraints_can_submit(self):
        result, _ = self.run_script([search(), detail()])
        task = TaskProfile(
            task=PublicTask(task_id=self.task_id, query="随便推荐一件商品"),
            requirements_status="parsed",
            no_constraints_reason="用户未限定商品条件",
        )
        check_submission(
            AgentSubmission(recommendations=[recommendation()]), task, result.memory
        )
        plan = Planner().advance(Planner().start(task, SESSION), task, result.memory, 1)
        self.assertEqual(plan.phase, "submit")

    def test_soft_violation_does_not_force_another_search(self):
        arguments = f.requirement_turn("白色").tool_calls[0].arguments
        arguments = {
            **arguments,
            "constraints": [
                {**arguments["constraints"][0], "priority": "soft"},
                {
                    "constraint_id": "quiet",
                    "description": "需要静音",
                    "priority": "hard",
                    "source": "user_query",
                    "source_quote": "静音",
                },
            ],
        }
        result, provider = self.run_script(
            [
                f.tool_turn("plan", "update_plan", arguments),
                search(),
                detail(OTHER),
                f.tool_turn(
                    "soft-conflict",
                    "record_evidence",
                    color_assessment(
                        asin=OTHER,
                        status="violated",
                        quote="黑色",
                        explanation="颜色不符合偏好",
                    ),
                ),
                f.tool_turn("next", "unknown", {}),
            ]
        )
        context = json.loads(provider.requests[-1]["messages"][1].content)
        self.assertEqual(context["plan"]["phase"], "verify")
        self.assertEqual(context["memory"]["hard_conflict_count"], 0)

    def test_duplicate_pairs_in_batch_are_rejected_without_mutation(self):
        result, _ = self.run_script(
            [
                f.requirement_turn("白色"),
                search(),
                detail(),
                f.tool_turn(
                    "duplicate",
                    "record_evidence_batch",
                    {
                        "assessments": [
                            color_assessment(),
                            color_assessment(status="unknown"),
                        ]
                    },
                ),
            ]
        )
        self.assertEqual(observations(result)[-1]["status"], "rejected")
        self.assertEqual(result.memory.candidates[ASIN].assessments, {})

    def test_every_recommended_product_requires_coverage(self):
        result, _ = self.run_script(
            [
                f.requirement_turn("白色"),
                search(),
                detail(),
                f.color_evidence_turn(ASIN, "白色"),
                detail(OTHER),
                submit(
                    recommendations=[
                        recommendation(),
                        recommendation(OTHER, "黑色", "黑色可选"),
                    ]
                ),
            ]
        )
        issue = observations(result)[-1]["data"]["issues"][0]
        self.assertEqual(issue["asin"], OTHER)
        self.assertEqual(issue["status"], "unknown")
        self.assertIsNone(result.session.evaluation)

    def test_best_effort_must_disclose_exact_gaps_in_the_recommendation_reason(self):
        disclosure = "尚未核实白色要求"
        result, _ = self.run_script(
            [
                f.requirement_turn("白色"),
                search(),
                detail(),
                submit(mode="best_effort"),
                submit(
                    mode="best_effort",
                    gaps=[
                        {
                            "asin": ASIN,
                            "constraint_id": "color",
                            "status": "unknown",
                            "disclosure": disclosure,
                        }
                    ],
                ),
                submit(
                    mode="best_effort",
                    recommendations=[recommendation(reason=disclosure)],
                    gaps=[
                        {
                            "asin": ASIN,
                            "constraint_id": "color",
                            "status": "unknown",
                            "disclosure": disclosure,
                        }
                    ],
                ),
            ]
        )
        self.assertEqual(
            observations(result)[3]["data"]["issues"][0]["code"],
            "incomplete_or_inaccurate_gap_disclosure",
        )
        self.assertEqual(
            observations(result)[4]["data"]["issues"][0]["code"],
            "gap_missing_from_reason",
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tool_call_count, 3)
        self.assertEqual(set(result.session.history[-1].arguments), {"recommendations"})

    def test_known_hard_conflict_is_blocked_and_ranked_below_uninspected_candidate(
        self,
    ):
        result, provider = self.run_script(
            [
                f.requirement_turn("白色"),
                search(),
                detail(OTHER),
                f.tool_turn(
                    "conflict",
                    "record_evidence",
                    color_assessment(
                        asin=OTHER,
                        status="violated",
                        quote="黑色",
                        explanation="黑色不符合白色要求",
                    ),
                ),
                submit(recommendations=[recommendation(OTHER, "黑色", "商品只有黑色")]),
            ],
            config=RuntimeConfig(memory_candidate_limit=1),
        )
        context = json.loads(provider.requests[-1]["messages"][1].content)
        self.assertEqual(context["memory"]["candidates"][0]["asin"], ASIN)
        self.assertEqual(context["memory"]["hard_conflict_count"], 1)
        self.assertEqual(context["memory"]["omitted_candidates"], 1)
        self.assertEqual(
            observations(result)[-1]["data"]["issues"][0]["status"], "violated"
        )

    def test_batch_is_atomic_and_does_not_overwrite_previous_assessments_on_failure(
        self,
    ):
        plan = f.requirement_turn("白色")
        arguments = dict(plan.tool_calls[0].arguments)
        arguments["constraints"] = [
            *arguments["constraints"],
            {
                "constraint_id": "quiet",
                "description": "需要静音",
                "priority": "hard",
                "source": "user_query",
                "source_quote": "静音",
            },
        ]
        result, _ = self.run_script(
            [
                f.tool_turn("plan", "update_plan", arguments),
                search(),
                detail(),
                f.tool_turn("first", "record_evidence", color_assessment()),
                f.tool_turn(
                    "batch",
                    "record_evidence_batch",
                    {
                        "assessments": [
                            color_assessment(status="violated"),
                            color_assessment(
                                constraint_id="quiet",
                                field_path="/attributes/0",
                                quote="不存在的引文",
                            ),
                        ]
                    },
                ),
            ]
        )
        candidate = result.memory.candidates[ASIN]
        self.assertEqual(candidate.assessments["color"].status, "satisfied")
        self.assertNotIn("quiet", candidate.assessments)
        self.assertEqual(observations(result)[-1]["status"], "rejected")

    def test_batch_records_multiple_constraints_in_one_action(self):
        plan = f.requirement_turn("白色").tool_calls[0].arguments
        plan = {
            **plan,
            "constraints": [
                *plan["constraints"],
                {
                    "constraint_id": "quiet",
                    "description": "需要静音",
                    "priority": "hard",
                    "source": "user_query",
                    "source_quote": "静音",
                },
            ],
        }
        result, _ = self.run_script(
            [
                f.tool_turn("plan", "update_plan", plan),
                search(),
                detail(),
                f.tool_turn(
                    "batch",
                    "record_evidence_batch",
                    {
                        "assessments": [
                            color_assessment(),
                            color_assessment(
                                constraint_id="quiet",
                                field_path="/attributes/0",
                                quote="静音",
                            ),
                        ]
                    },
                ),
                submit(),
            ]
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.memory.candidates[ASIN].assessments), 2)
        self.assertEqual(result.action_count, 5)
        self.assertEqual(result.tool_call_count, 3)

    def test_read_archived_evidence_after_history_compaction_and_replay(self):
        def cite_retrieved(messages):
            data = json.loads(messages[-1].content)["data"]
            return f.tool_turn(
                "cite-read",
                "record_evidence",
                color_assessment(
                    observation_id=data["source_observation_id"],
                    field_path=data["items"][0]["field_path"],
                    quote=data["items"][0]["value"],
                ),
            )

        result, provider = self.run_script(
            [
                f.requirement_turn("白色"),
                search(),
                detail(),
                f.tool_turn("error", "unknown", {}),
                f.tool_turn(
                    "read",
                    "read_evidence",
                    {
                        "asin": ASIN,
                        "field_paths": ["/customization_options/颜色/0/value"],
                    },
                ),
                cite_retrieved,
                submit(),
            ],
            config=RuntimeConfig(recent_turns=1),
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tool_call_count, 3)
        before_read = json.loads(provider.requests[4]["messages"][1].content)
        self.assertNotIn("evidence_archive", before_read["memory"])
        self.assertEqual(
            before_read["memory"]["candidates"][0]["facts"]["attribute_previews"],
            ["静音", "节能"],
        )
        self.assertEqual(observations(result)[4]["data"]["items"][0]["value"], "白色")
        self.assertEqual(
            observations(result)[4]["budget"]["environment_steps_remaining"], 5
        )
        directory = Path(self._temporary_directory.name) / "archived-run"
        write_run_artifacts(result, directory)
        state = replay_run(directory)
        self.assertEqual(state["memory"], result.memory.model_dump(mode="json"))
        original_id = f"{SESSION}:action_0003:observation"
        self.assertEqual(state["memory"]["evidence_archive"][original_id]["asin"], ASIN)

    def test_evidence_read_cannot_access_unseen_or_other_session_product(self):
        result, _ = self.run_script(
            [
                search(),
                detail(),
                f.tool_turn("unseen", "read_evidence", {"asin": OTHER}),
                f.tool_turn(
                    "foreign",
                    "read_evidence",
                    {"asin": ASIN, "observation_id": "foreign-session:detail"},
                ),
                f.tool_turn(
                    "missing",
                    "read_evidence",
                    {"asin": ASIN, "field_paths": ["/private_label"]},
                ),
            ]
        )
        reads = observations(result)[2:]
        # A product this session never inspected, and an observation that belongs
        # to another session, are still hard rejections.
        self.assertEqual([o["status"] for o in reads[:2]], ["rejected", "rejected"])
        # An unobserved field no longer costs the whole call, but it is reported
        # rather than served, and never presented as an empty value.
        self.assertEqual(reads[2]["status"], "success")
        self.assertEqual(reads[2]["data"]["items"], [])
        self.assertEqual(
            [m["field_path"] for m in reads[2]["data"]["missing_field_paths"]],
            ["/private_label"],
        )
        self.assertEqual(
            reads[2]["data"]["missing_field_paths"][0]["reason"],
            "not observed in this session",
        )
        self.assertNotIn("/private_label", reads[2]["data"]["available_field_paths"])
        self.assertEqual(result.tool_call_count, 2)
        self.assertEqual(len(result.memory.evidence_archive), 1)

    def test_read_pagination_is_bounded_and_recovers_exact_long_text(self):
        result, _ = self.run_script([search(), detail()])
        store = MemoryStore(SESSION)
        store.state = result.memory.model_copy(deep=True)
        oid = store.state.candidates[ASIN].detail_observation_id
        original = "证据长文" * 8000
        store.state.evidence_archive[oid] = store.state.evidence_archive[
            oid
        ].model_copy(update={"description": original})
        offset, chunks = 0, []
        while True:
            page = store.read_evidence(
                ReadEvidence(
                    asin=ASIN, field_paths=["/description"], offset=offset, limit=20
                )
            )
            self.assertLessEqual(len(json.dumps(page, ensure_ascii=False)), 12000)
            self.assertEqual(page["source_observation_id"], oid)
            chunks.extend(item["value"] for item in page["items"])
            if page["next_offset"] is None:
                break
            self.assertGreater(page["next_offset"], offset)
            offset = page["next_offset"]
        self.assertEqual("".join(chunks), original)

    def test_public_option_errors_do_not_seal_the_session(self):
        result, _ = self.run_script(
            [
                f.requirement_turn("白色"),
                search(),
                detail(),
                f.color_evidence_turn(ASIN, "白色"),
                submit(recommendations=[recommendation(color="黑色")]),
                submit(),
            ]
        )
        self.assertEqual(
            observations(result)[4]["data"]["issues"][0]["code"],
            "invalid_or_unavailable_option",
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tool_call_count, 3)

    def test_unavailable_combination_cannot_pass_preflight_using_independent_options(
        self,
    ):
        result, _ = self.run_script(
            [
                f.requirement_turn("白色"),
                search(),
                detail(),
                f.color_evidence_turn(ASIN, "白色"),
            ]
        )
        memory = result.memory.model_copy(deep=True)
        oid = memory.candidates[ASIN].detail_observation_id
        raw = memory.evidence_archive[oid].model_dump(mode="json")
        raw["customization_options"]["大小"] = [{"value": "大", "is_available": True}]
        raw["variant_combinations"] = [
            {"options": {"颜色": "白色", "大小": "小"}, "is_available": True}
        ]
        memory.evidence_archive[oid] = type(
            memory.evidence_archive[oid]
        ).model_validate(raw)
        item = recommendation()
        item["selected_options"]["大小"] = "大"
        with self.assertRaises(SubmissionNotReady) as caught:
            check_submission(
                AgentSubmission(recommendations=[item]), result.task_profile, memory
            )
        self.assertEqual(
            caught.exception.issues[0]["code"], "unavailable_variant_combination"
        )


if __name__ == "__main__":
    unittest.main()
