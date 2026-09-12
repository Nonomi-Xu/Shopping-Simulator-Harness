"""Behavioral coverage for discovery, partial evidence, comparison and decisions."""

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from src.data.search_bm25 import build_index
from src.harness.catalog import ProductCatalog
from src.harness.schemas import (
    SearchProductsAction,
    GetProductAction,
    ProductDetail,
    PublicTask,
)
from src.harness.agent import (
    AgentRunner,
    RuntimeConfig,
    TaskProfile,
    CompareProducts,
    VerifySelection,
    replay_run,
    write_run_artifacts,
    AgentSubmission,
    ReadEvidence,
    ToolRegistry,
)
from src.harness.agent.comparison import compare_products, verify_selection
from src.harness.agent.contracts import (
    AgentObservation,
    CandidateMemory,
    WorkingMemory,
    Constraint,
)
from src.harness.agent.memory import MemoryStore, observed_product
from src.harness.agent.proposals import action_signature
from src.harness.agent.submission import SubmissionNotReady, check_submission
from src.harness.test import test_agent_runner as f


class DiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        rows = [
            {
                "asin": "expensive",
                "title": "耳机耳机耳机",
                "category": "数码›耳机",
                "shop_name": "ＡＣＭＥ",
                "pricing": [900],
            },
            {
                "asin": "cheap",
                "title": "蓝牙耳机",
                "category": "数码›耳机",
                "shop_name": "Acme",
                "pricing": [100],
                "attribute": ["通勤降噪"],
                "instructions": [{"instruction": "私有答案不可见"}],
            },
            {
                "asin": "unknown",
                "title": "耳机",
                "category": "数码›耳机",
                "shop_name": "Acme",
                "pricing": [],
            },
            {
                "asin": "other",
                "title": "耳机收纳包",
                "category": "配件",
                "shop_name": "Other",
                "pricing": [20],
            },
        ]
        path = Path(self.directory.name) / "index.sqlite3"
        build_index(
            rows,
            path,
            product_data_sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
        )
        self.catalog = ProductCatalog(path)
        self.addCleanup(self.catalog.close)

    def test_filters_apply_before_limit_and_unknown_prices_are_explicit(self):
        request = SearchProductsAction(
            query="耳机",
            limit=1,
            filters={
                "brand": "acme",
                "category": "耳机",
                "max_price": 150,
                "include_unknown_price": False,
            },
        )
        rows, page = self.catalog.search_page(request)
        self.assertEqual([r.asin for r in rows], ["cheap"])
        self.assertFalse(page.has_more)
        rows, _ = self.catalog.search_page(
            request.model_copy(
                update={
                    "limit": 10,
                    "filters": request.filters.model_copy(
                        update={"include_unknown_price": True}
                    ),
                }
            )
        )
        self.assertEqual({r.asin for r in rows}, {"cheap", "unknown"})
        rows = self.catalog.search(
            "耳机", filters=request.filters.model_copy(update={"max_price": 1000})
        )
        self.assertIn("expensive", {r.asin for r in rows})

    def test_paging_is_stable_and_exclusions_do_not_expose_skipped_products(self):
        all_rows = self.catalog.search("耳机")
        offset, collected = 0, []
        while True:
            rows, page = self.catalog.search_page(
                SearchProductsAction(query="耳机", limit=1, offset=offset)
            )
            collected.extend(r.asin for r in rows)
            self.assertEqual(rows[0].rank, offset + 1)
            self.assertIsNone(page.total_results)
            if page.next_offset is None:
                break
            offset = page.next_offset
        self.assertEqual(collected, [r.asin for r in all_rows])
        rows = self.catalog.search("耳机", exclude_asins=["cheap", "unknown"])
        self.assertEqual({r.asin for r in rows}, {"expensive", "other"})

    def test_snippets_only_quote_public_fields_and_sql_values_are_bound(self):
        rows = self.catalog.search("通勤降噪")
        snippets = rows[0].match_snippets
        self.assertTrue(
            any(s.field_path == "/attributes/0" and "通勤" in s.text for s in snippets)
        )
        self.assertNotIn(
            "私有答案", json.dumps([r.model_dump() for r in rows], ensure_ascii=False)
        )
        self.assertEqual(
            self.catalog.search("耳机", filters={"brand": "' OR 1=1 --"}), []
        )

    def test_invalid_filters_and_private_field_selection_are_rejected(self):
        with self.assertRaises(ValidationError):
            SearchProductsAction(
                query="耳机", filters={"min_price": 200, "max_price": 100}
            )
        with self.assertRaises(ValidationError):
            GetProductAction(asin="cheap", fields=["instructions"])


def selection_memory(variants=None):
    detail = ProductDetail(
        asin="p",
        title="耳机",
        category="耳机",
        price_min=100,
        price_max=500,
        customization_options={
            "颜色": [{"value": "黑色", "price": 100}],
            "容量": [{"value": "大", "price": 100}, {"value": "小"}],
        },
        variant_combinations=variants or [],
    )
    memory = WorkingMemory(
        session_id="s",
        candidates={
            "p": CandidateMemory(
                asin="p",
                summary={},
                inspected=True,
                observation_ids=["s:detail"],
                detail_observation_id="s:detail",
            )
        },
        evidence_archive={"s:detail": detail},
    )
    task = TaskProfile(
        task=PublicTask(task_id="task_" + "a" * 24, query="预算300以内"),
        requirements_status="parsed",
    )
    return task, memory


class SelectionTest(unittest.TestCase):
    def test_selected_variant_price_overrides_product_minimum(self):
        task, memory = selection_memory(
            [{"options": {"颜色": "黑色", "容量": "大"}, "price": 500}]
        )
        result = verify_selection(
            VerifySelection(
                asin="p", selected_options={"颜色": "黑色", "容量": "大"}, max_price=300
            ),
            task,
            memory,
        )
        self.assertEqual(result["selected_price"], 500)
        self.assertEqual(result["budget_status"], "violated")
        self.assertEqual(
            result["price_evidence"][0]["field_path"], "/variant_combinations/0/price"
        )

    def test_partial_selection_and_missing_combination_prices_stay_unknown(self):
        task, memory = selection_memory(
            [
                {"options": {"颜色": "黑色", "容量": "大"}, "price": 100},
                {"options": {"颜色": "黑色", "容量": "小"}, "price": 500},
            ]
        )
        result = verify_selection(
            VerifySelection(asin="p", selected_options={"颜色": "黑色"}, max_price=300),
            task,
            memory,
        )
        self.assertIsNone(result["selected_price"])
        self.assertEqual(result["budget_status"], "unknown")
        task, memory = selection_memory()
        result = verify_selection(
            VerifySelection(
                asin="p", selected_options={"颜色": "黑色", "容量": "大"}, max_price=300
            ),
            task,
            memory,
        )
        self.assertEqual(result["combination_status"], "unknown")
        self.assertIsNone(result["selected_price"])

    def test_individually_valid_options_do_not_prove_available_combination(self):
        task, memory = selection_memory(
            [
                {"options": {"颜色": "黑色", "容量": "大"}, "is_available": False},
                {"options": {"颜色": "黑色", "容量": "小"}, "price": 100},
            ]
        )
        result = verify_selection(
            VerifySelection(asin="p", selected_options={"颜色": "黑色", "容量": "大"}),
            task,
            memory,
        )
        self.assertEqual(result["selection_status"], "invalid")
        self.assertIn(
            "unavailable_variant_combination", [i["code"] for i in result["issues"]]
        )

    def test_price_conflicts_and_null_prices_are_not_resolved_by_guessing(self):
        for prices in ([100, 500], [None, 100]):
            task, memory = selection_memory(
                [
                    {"options": {"颜色": "黑色", "容量": "大"}, "price": p}
                    for p in prices
                ]
            )
            result = verify_selection(
                VerifySelection(
                    asin="p",
                    selected_options={"颜色": "黑色", "容量": "大"},
                    max_price=300,
                ),
                task,
                memory,
            )
            self.assertEqual(result["budget_status"], "unknown")


class StageTwoRuntimeTest(unittest.TestCase):
    setUp = f.AgentRunnerTest.setUp

    def run_script(self, outputs, **kwargs):
        provider = f.ScriptedProvider(outputs)
        result = AgentRunner(
            provider=provider,
            harness=self.harness,
            max_model_turns=len(outputs),
            **kwargs,
        ).run(self.task_id, max_steps=20, session_id="stage-two")
        return result, provider

    def test_partial_details_comparison_verification_and_submission_replay(self):
        asin, other = f.TARGET_ASIN, f.ALTERNATIVE_ASIN
        result, provider = self.run_script(
            [
                f.requirement_turn("白色"),
                f.tool_turn(
                    "search",
                    "search_products",
                    {"query": "冰箱", "filters": {"category": "冰箱"}},
                ),
                f.tool_turn(
                    "partial", "get_product", {"asin": asin, "fields": ["attributes"]}
                ),
                f.tool_turn("other", "get_product", {"asin": other}),
                f.tool_turn(
                    "compare",
                    "compare_products",
                    {
                        "asins": [asin, other],
                        "fields": ["attributes", "customization_options"],
                    },
                ),
                f.tool_turn(
                    "selection-fields",
                    "get_product",
                    {
                        "asin": asin,
                        "fields": ["customization_options", "variant_combinations"],
                    },
                ),
                f.tool_turn(
                    "verify",
                    "verify_selection",
                    {
                        "asin": asin,
                        "selected_options": {"颜色": "白色"},
                        "max_price": 800,
                    },
                ),
                f.tool_turn(
                    "evidence",
                    "record_evidence",
                    {
                        "asin": asin,
                        "constraint_id": "color",
                        "status": "satisfied",
                        "observation_id": "stage-two:action_0006:observation",
                        "field_path": "/customization_options/颜色/0/value",
                        "quote": "白色",
                        "explanation": "所选颜色在公开详情中可用",
                    },
                ),
                f.tool_turn(
                    "submit",
                    "submit_recommendations",
                    {
                        "recommendations": [
                            {
                                "asin": asin,
                                "selected_options": {"颜色": "白色"},
                                "reason": "白色可选，所选规格价格799",
                            }
                        ]
                    },
                ),
            ],
            config=RuntimeConfig(recent_turns=1),
        )
        self.assertEqual(result.status, "completed", result.error_message)
        self.assertTrue(result.session.evaluation.strict_success)
        self.assertEqual(result.tool_call_count, 5)
        observations = [e.payload for e in result.events if e.kind == "observation"]
        self.assertEqual(
            set(observations[2]["data"]), {"asin", "title", "category", "attributes"}
        )
        self.assertEqual(
            observations[4]["data"]["products"][0]["fields"]["customization_options"][
                "status"
            ],
            "unobserved",
        )
        self.assertEqual(observations[6]["data"]["selected_price"], 799)
        detail, sources = observed_product(result.memory, asin)
        self.assertEqual(detail.attributes, ["静音", "节能"])
        self.assertEqual(sources["attributes"], "stage-two:action_0003:observation")
        self.assertNotIn("price_min", sources)
        context = json.loads(provider.requests[4]["messages"][1].content)
        self.assertTrue(
            any(
                p["tool_name"] == "compare_products"
                for p in context["action_proposals"]
            )
        )
        self.assertLessEqual(len(context["action_proposals"]), 5)
        public_context = json.dumps(
            [json.loads(r["messages"][1].content) for r in provider.requests]
        )
        self.assertNotIn('"target_asin":', public_context)
        directory = Path(self._temporary_directory.name) / "stage-two-run"
        write_run_artifacts(result, directory)
        replayed = replay_run(directory)
        restored = WorkingMemory.model_validate(replayed["memory"])
        self.assertEqual(observed_product(restored, asin)[1], sources)
        store = MemoryStore("stage-two")
        store.state = restored
        with self.assertRaises(ValueError):
            store.read_evidence(
                ReadEvidence(
                    asin=asin,
                    observation_id="stage-two:action_0003:observation",
                    field_paths=["/price_min"],
                )
            )

    def test_comparison_and_verification_cannot_read_uninspected_or_foreign_products(
        self,
    ):
        result, _ = self.run_script(
            [
                f.tool_turn("search", "search_products", {"query": "冰箱"}),
                f.tool_turn("detail", "get_product", {"asin": f.TARGET_ASIN}),
            ]
        )
        for fn, request in (
            (
                compare_products,
                CompareProducts(asins=[f.TARGET_ASIN, f.ALTERNATIVE_ASIN]),
            ),
            (verify_selection, VerifySelection(asin=f.ALTERNATIVE_ASIN)),
        ):
            with self.assertRaises(ValueError):
                fn(request, result.task_profile, result.memory)

    def test_unobserved_selection_fields_cannot_pass_submission_preflight(self):
        result, _ = self.run_script(
            [
                f.requirement_turn("白色"),
                f.tool_turn("search", "search_products", {"query": "冰箱"}),
                f.tool_turn(
                    "detail",
                    "get_product",
                    {"asin": f.TARGET_ASIN, "fields": ["attributes"]},
                ),
            ]
        )
        with self.assertRaises(SubmissionNotReady) as caught:
            check_submission(
                AgentSubmission(
                    recommendations=[
                        {
                            "asin": f.TARGET_ASIN,
                            "selected_options": {},
                            "reason": "说明",
                        }
                    ]
                ),
                result.task_profile,
                result.memory,
            )
        self.assertEqual(
            caught.exception.issues[0]["code"], "selection_fields_unobserved"
        )

    def test_page_lookahead_does_not_grant_detail_access(self):
        started = self.harness.create_session(self.task_id)
        sid = started.state.session_id
        step = self.harness.dispatch(
            sid, "search_products", {"query": "冰箱", "limit": 1}
        )
        self.assertTrue(step.search_page.has_more)
        surfaced = step.observation[0]["asin"]
        other = f.ALTERNATIVE_ASIN if surfaced == f.TARGET_ASIN else f.TARGET_ASIN
        rejected = self.harness.dispatch(sid, "get_product", {"asin": other})
        self.assertEqual(rejected.error_code, "product_not_surfaced")

    def test_advisory_candidates_are_bounded_legal_and_can_be_disabled(self):
        outputs = [
            f.requirement_turn("白色"),
            f.tool_turn("search", "search_products", {"query": "冰箱"}),
            f.tool_turn("detail", "get_product", {"asin": f.TARGET_ASIN}),
            f.tool_turn(
                "verify",
                "verify_selection",
                {"asin": f.TARGET_ASIN, "selected_options": {"颜色": "白色"}},
            ),
        ]
        _, provider = self.run_script(outputs)
        registry = ToolRegistry()
        for request in provider.requests:
            context = json.loads(request["messages"][1].content)
            self.assertLessEqual(len(context["action_proposals"]), 5)
            for proposal in context["action_proposals"]:
                self.assertIn(proposal["tool_name"], context["allowed_tools"])
                if not proposal["requires_model_input"]:
                    registry.models[proposal["tool_name"]].model_validate(
                        proposal["arguments"]
                    )
        self.harness.release_session("stage-two", force=True)
        _, disabled = self.run_script(
            [f.requirement_turn("白色")], config=RuntimeConfig(action_proposal_limit=0)
        )
        self.assertEqual(
            json.loads(disabled.requests[0]["messages"][1].content)["action_proposals"],
            [],
        )

    def test_numeric_budget_conflict_is_not_hidden_by_model_assessment(self):
        plan = dict(f.requirement_turn("白色").tool_calls[0].arguments)
        plan["constraints"] = [
            {
                "constraint_id": "price",
                "description": "用于测试公开阈值冲突",
                "priority": "hard",
                "source": "user_query",
                "source_quote": "冰箱",
            }
        ]
        result, _ = self.run_script(
            [
                f.tool_turn("plan", "update_plan", plan),
                f.tool_turn("search", "search_products", {"query": "冰箱"}),
                f.tool_turn("detail", "get_product", {"asin": f.TARGET_ASIN}),
                f.tool_turn(
                    "assess",
                    "record_evidence",
                    {
                        "asin": f.TARGET_ASIN,
                        "constraint_id": "price",
                        "status": "satisfied",
                        "observation_id": "stage-two:action_0003:observation",
                        "field_path": "/price_min",
                        "quote": "799",
                        "explanation": "模型声称满足",
                    },
                ),
                f.tool_turn(
                    "verify",
                    "verify_selection",
                    {
                        "asin": f.TARGET_ASIN,
                        "selected_options": {"颜色": "白色"},
                        "max_price": 700,
                        "budget_constraint_id": "price",
                    },
                ),
                f.tool_turn(
                    "reread",
                    "get_product",
                    {"asin": f.TARGET_ASIN, "fields": ["attributes"]},
                ),
            ]
        )
        with self.assertRaises(SubmissionNotReady) as caught:
            check_submission(
                AgentSubmission(
                    recommendations=[
                        {
                            "asin": f.TARGET_ASIN,
                            "selected_options": {"颜色": "白色"},
                            "reason": "模型声称满足预算",
                        }
                    ]
                ),
                result.task_profile,
                result.memory,
            )
        self.assertTrue(
            any(issue.get("status") == "violated" for issue in caught.exception.issues)
        )

    def test_paging_and_filter_changes_are_not_repeated_query_errors(self):
        outputs = [
            f.tool_turn(str(i), "search_products", args)
            for i, args in enumerate(
                [
                    {"query": "冰箱", "limit": 1},
                    {"query": "冰箱", "limit": 1, "offset": 1},
                    {"query": "冰箱", "filters": {"max_price": 700}},
                ]
            )
        ]
        result, _ = self.run_script(
            outputs, config=RuntimeConfig(max_repeated_actions=1)
        )
        self.assertEqual(result.search_count, 3)
        self.assertTrue(
            all(
                e.payload["status"] == "success"
                for e in result.events
                if e.kind == "observation"
            )
        )
        self.assertEqual(
            action_signature("search_products", {"query": "ＡＢＣ 冰箱", "limit": 1}),
            action_signature("search_products", {"query": "abc   冰箱", "limit": 50}),
        )


if __name__ == "__main__":
    unittest.main()
