"""Contract tests for the bounded single-task Harness session."""

from __future__ import annotations

import json
import unittest

from src.harness.catalog import CatalogError, ProductNotFoundError
from src.harness.schemas import (
    GetProductAction,
    PrivateTaskFacts,
    ProductDetail,
    ProductOption,
    ProductSummary,
    PublicTask,
    RecommendationItem,
    SearchProductsAction,
    SubmitRecommendationsAction,
)
from src.harness.session import SessionTerminatedError, ShoppingSession


TASK_ID = "task_0123456789abcdef01234567"
TARGET_ASIN = "target-1"
ALTERNATIVE_ASIN = "alternative-1"
SEARCH_PRODUCTS = "search_products"
GET_PRODUCT = "get_product"
SUBMIT_RECOMMENDATIONS = "submit_recommendations"


def _product(asin: str) -> ProductDetail:
    return ProductDetail(
        asin=asin,
        title=f"商品 {asin}",
        category="测试分类",
        customization_options={
            "颜色": [
                ProductOption(value="白色", price=99),
                ProductOption(value="黑色", price=109),
            ]
        },
    )


class _TaskStore:
    def __init__(self) -> None:
        self.private_fact_reads = 0
        self.public_task = PublicTask(
            task_id=TASK_ID,
            query="寻找一款白色测试商品",
        )
        self.private_facts = PrivateTaskFacts(
            task_id=TASK_ID,
            split="eval",
            target_asin=TARGET_ASIN,
            target_title="秘密目标商品",
            target_category="秘密分类",
            required_options={"颜色": "白色"},
            required_option_values=["白色"],
            source_row_number=1,
            source_instruction_index=0,
        )

    def get_public_task(self, task_id: str) -> PublicTask:
        if task_id != TASK_ID:
            raise KeyError(task_id)
        return self.public_task.model_copy(deep=True)

    def get_private_facts(self, task_id: str) -> PrivateTaskFacts:
        if task_id != TASK_ID:
            raise KeyError(task_id)
        self.private_fact_reads += 1
        return self.private_facts.model_copy(deep=True)


class _Catalog:
    def __init__(self) -> None:
        self.products = {
            TARGET_ASIN: _product(TARGET_ASIN),
            ALTERNATIVE_ASIN: _product(ALTERNATIVE_ASIN),
        }

    def search(self, query: str, *, limit: int = 20) -> list[ProductSummary]:
        asins = (
            [TARGET_ASIN, ALTERNATIVE_ASIN]
            if "目标" in query
            else [ALTERNATIVE_ASIN]
        )
        return [
            ProductSummary(
                asin=asin,
                title=self.products[asin].title,
                category=self.products[asin].category,
                bm25_score=1.0 / rank,
                rank=rank,
            )
            for rank, asin in enumerate(asins[:limit], start=1)
        ]

    def get_product(self, asin: str) -> ProductDetail:
        try:
            return self.products[asin].model_copy(deep=True)
        except KeyError as exc:
            raise ProductNotFoundError(f"product not found: {asin}") from exc


def _session(*, max_steps: int = 20) -> tuple[ShoppingSession, _TaskStore]:
    task_store = _TaskStore()
    session = ShoppingSession(
        task_id=TASK_ID,
        task_store=task_store,
        catalog=_Catalog(),
        max_steps=max_steps,
    )
    return session, task_store


def _submission(asin: str = TARGET_ASIN) -> SubmitRecommendationsAction:
    return SubmitRecommendationsAction(
        recommendations=[
            RecommendationItem(
                asin=asin,
                selected_options={"颜色": "白色"},
                reason="满足颜色和品类需求",
            )
        ]
    )


class ShoppingSessionTest(unittest.TestCase):
    def test_initialization_exposes_only_public_task(self) -> None:
        session, task_store = _session()

        self.assertEqual(session.public_task.query, "寻找一款白色测试商品")
        self.assertEqual(task_store.private_fact_reads, 0)
        self.assertEqual(session.state.session_version, "shopping-session-v1")
        self.assertEqual(session.state.status, "active")
        self.assertEqual(session.state.step_count, 0)
        self.assertEqual(session.history, [])

        serialized = session.public_task.model_dump_json()
        self.assertNotIn("秘密目标商品", serialized)
        self.assertNotIn(TARGET_ASIN, serialized)

    def test_search_surfaces_results_and_records_public_observation(self) -> None:
        session, _ = _session()

        step = session.search_products(
            SearchProductsAction(query="普通候选", limit=5)
        )

        self.assertEqual(step.status, "success")
        self.assertEqual(step.step_id, "step_0001")
        self.assertEqual(session.state.surfaced_asins, [ALTERNATIVE_ASIN])
        self.assertEqual(step.observation[0]["asin"], ALTERNATIVE_ASIN)

    def test_get_product_rejects_an_asin_that_was_not_surfaced(self) -> None:
        session, _ = _session()

        step = session.get_product(GetProductAction(asin=TARGET_ASIN))

        self.assertEqual(step.status, "rejected")
        self.assertEqual(step.error_code, "product_not_surfaced")
        self.assertEqual(session.state.inspected_asins, [])
        self.assertEqual(session.state.step_count, 1)

    def test_get_product_records_an_inspected_public_detail(self) -> None:
        session, _ = _session()
        session.dispatch(SEARCH_PRODUCTS, {"query": "目标商品"})

        step = session.dispatch(GET_PRODUCT, {"asin": TARGET_ASIN})

        self.assertEqual(step.status, "success")
        self.assertEqual(step.observation["asin"], TARGET_ASIN)
        self.assertEqual(session.state.inspected_asins, [TARGET_ASIN])

    def test_submission_rejects_products_that_were_not_inspected(self) -> None:
        session, task_store = _session()
        session.dispatch(SEARCH_PRODUCTS, {"query": "目标商品"})

        step = session.submit_recommendations(_submission())

        self.assertEqual(step.status, "rejected")
        self.assertEqual(step.error_code, "product_not_inspected")
        self.assertEqual(session.state.status, "active")
        self.assertEqual(task_store.private_fact_reads, 0)
        self.assertIsNone(session.evaluation)

    def test_legal_submission_evaluates_once_and_seals_session(self) -> None:
        session, task_store = _session()
        session.dispatch(SEARCH_PRODUCTS, {"query": "目标商品"})
        session.dispatch(GET_PRODUCT, {"asin": TARGET_ASIN})

        step = session.dispatch(
            SUBMIT_RECOMMENDATIONS,
            _submission().model_dump(mode="json"),
        )

        self.assertEqual(step.status, "success")
        self.assertTrue(step.terminated)
        self.assertEqual(step.terminal_reason, "submitted")
        self.assertEqual(session.state.status, "terminated")
        self.assertEqual(session.state.terminal_reason, "submitted")
        self.assertEqual(task_store.private_fact_reads, 1)
        self.assertTrue(session.evaluation.strict_success)
        self.assertEqual(step.observation["reward"], 1.0)

        with self.assertRaises(SessionTerminatedError) as raised:
            session.dispatch(SEARCH_PRODUCTS, {"query": "再次搜索"})
        self.assertEqual(raised.exception.code, "session_terminated")
        self.assertEqual(len(session.history), 3)

    def test_unknown_and_malformed_actions_are_rejected_and_consume_steps(self) -> None:
        session, _ = _session(max_steps=3)

        unknown = session.dispatch("unknown_tool", {})
        malformed = session.dispatch(
            SEARCH_PRODUCTS,
            {"query": "冰箱", "private_override": True},
        )
        invalid_shape = session.dispatch(GET_PRODUCT, None)

        self.assertEqual(unknown.error_code, "unknown_action")
        self.assertEqual(malformed.error_code, "invalid_arguments")
        self.assertEqual(invalid_shape.error_code, "invalid_arguments")
        self.assertTrue(invalid_shape.terminated)
        self.assertEqual(session.state.terminal_reason, "max_steps")
        self.assertEqual(session.state.step_count, 3)

    def test_successful_action_at_limit_terminates_with_its_observation(self) -> None:
        session, _ = _session(max_steps=1)

        step = session.dispatch(SEARCH_PRODUCTS, {"query": "普通候选"})

        self.assertEqual(step.status, "success")
        self.assertTrue(step.observation)
        self.assertTrue(step.terminated)
        self.assertEqual(step.terminal_reason, "max_steps")
        self.assertIsNone(session.evaluation)

    def test_empty_submission_is_terminal_invalid_evaluation(self) -> None:
        session, task_store = _session()

        step = session.dispatch(
            SUBMIT_RECOMMENDATIONS,
            {"recommendations": []},
        )

        self.assertEqual(step.status, "success")
        self.assertEqual(step.observation["terminal_reason"], "invalid_submission")
        self.assertFalse(step.observation["reward_valid"])
        self.assertEqual(session.state.terminal_reason, "submitted")
        self.assertEqual(task_store.private_fact_reads, 1)

    def test_pre_submission_history_does_not_contain_private_facts(self) -> None:
        session, task_store = _session()
        session.dispatch(SEARCH_PRODUCTS, {"query": "普通候选"})
        session.dispatch(GET_PRODUCT, {"asin": ALTERNATIVE_ASIN})

        serialized = json.dumps(
            [step.model_dump(mode="json") for step in session.history],
            ensure_ascii=False,
            sort_keys=True,
        )

        self.assertEqual(task_store.private_fact_reads, 0)
        self.assertNotIn("秘密目标商品", serialized)
        self.assertNotIn("秘密分类", serialized)
        self.assertNotIn(TARGET_ASIN, serialized)

    def test_identical_action_sequences_produce_identical_trajectories(self) -> None:
        first, _ = _session()
        second, _ = _session()
        actions = [
            (SEARCH_PRODUCTS, {"query": "目标商品", "limit": 2}),
            (GET_PRODUCT, {"asin": TARGET_ASIN}),
            (
                SUBMIT_RECOMMENDATIONS,
                _submission().model_dump(mode="json"),
            ),
        ]

        for tool_name, arguments in actions:
            first.dispatch(tool_name, arguments)
            second.dispatch(tool_name, arguments)

        self.assertEqual(first.state, second.state)
        self.assertEqual(first.history, second.history)
        self.assertEqual(first.evaluation, second.evaluation)

    def test_catalog_failure_terminates_as_infrastructure_error(self) -> None:
        class _BrokenCatalog(_Catalog):
            def search(
                self,
                query: str,
                *,
                limit: int = 20,
            ) -> list[ProductSummary]:
                raise CatalogError("index unavailable")

        task_store = _TaskStore()
        session = ShoppingSession(
            task_id=TASK_ID,
            task_store=task_store,
            catalog=_BrokenCatalog(),
        )

        step = session.dispatch(SEARCH_PRODUCTS, {"query": "商品"})

        self.assertEqual(step.status, "error")
        self.assertEqual(step.error_code, "infrastructure_error")
        self.assertEqual(step.terminal_reason, "infrastructure_error")
        self.assertTrue(step.terminated)

if __name__ == "__main__":
    unittest.main()
