"""Contract tests for deterministic recommendation evaluation."""

from __future__ import annotations

import unittest

from src.harness.catalog import ProductNotFoundError
from src.harness.evaluator import EvaluatorDataError, evaluate_submission
from src.harness.schemas import (
    PrivateTaskFacts,
    ProductDetail,
    ProductOption,
    RecommendationItem,
    SubmitRecommendationsAction,
)


TASK_ID = "task_0123456789abcdef01234567"
TARGET_ASIN = "target-1"


def _product(
    asin: str,
    *,
    option_available: bool = True,
) -> ProductDetail:
    return ProductDetail(
        asin=asin,
        title="测试商品",
        category="测试分类",
        customization_options={
            "颜色": [
                ProductOption(
                    value="白色",
                    price=99,
                    is_available=option_available,
                ),
                ProductOption(value="黑色", price=109),
            ]
        },
    )


class _Catalog:
    def __init__(self, products: list[ProductDetail]) -> None:
        self.products = {product.asin: product for product in products}

    def get_product(self, asin: str) -> ProductDetail:
        try:
            return self.products[asin]
        except KeyError as exc:
            raise ProductNotFoundError(f"product not found: {asin}") from exc


def _facts(**updates) -> PrivateTaskFacts:
    values = {
        "task_id": TASK_ID,
        "split": "all",
        "target_asin": TARGET_ASIN,
        "target_title": "测试商品",
        "target_category": "测试分类",
        "required_options": {"颜色": "白色"},
        "required_option_values": ["白色"],
        "source_row_number": 1,
        "source_instruction_index": 0,
    }
    values.update(updates)
    return PrivateTaskFacts(**values)


def _submission(*items: RecommendationItem) -> SubmitRecommendationsAction:
    return SubmitRecommendationsAction(recommendations=list(items))


def _item(
    asin: str,
    *,
    options: dict[str, str] | None = None,
    reason: str = "满足需求",
) -> RecommendationItem:
    return RecommendationItem(
        asin=asin,
        selected_options=options or {},
        reason=reason,
    )


class EvaluatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = _Catalog(
            [_product(TARGET_ASIN), _product("alternative-1")]
        )

    def test_exact_target_and_required_options_receive_full_reward(self) -> None:
        result = evaluate_submission(
            facts=_facts(),
            submission=_submission(
                _item(TARGET_ASIN, options={"颜色": "白色"})
            ),
            catalog=self.catalog,  # type: ignore[arg-type]
        )

        self.assertTrue(result.strict_success)
        self.assertTrue(result.reward_valid)
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(result.terminal_reason, "strict_success")
        self.assertEqual(result.matched_rank, 1)

    def test_target_without_required_option_receives_partial_reward(self) -> None:
        result = evaluate_submission(
            facts=_facts(),
            submission=_submission(_item(TARGET_ASIN)),
            catalog=self.catalog,  # type: ignore[arg-type]
        )

        self.assertFalse(result.strict_success)
        self.assertTrue(result.reward_valid)
        self.assertEqual(result.reward, 0.25)
        self.assertEqual(result.terminal_reason, "partial_match")

    def test_valid_non_target_submission_is_a_replayable_failure(self) -> None:
        result = evaluate_submission(
            facts=_facts(),
            submission=_submission(
                _item("alternative-1", options={"颜色": "黑色"})
            ),
            catalog=self.catalog,  # type: ignore[arg-type]
        )

        self.assertTrue(result.reward_valid)
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.terminal_reason, "no_match")
        self.assertIsNone(result.matched_rank)

    def test_unknown_product_makes_submission_invalid(self) -> None:
        result = evaluate_submission(
            facts=_facts(),
            submission=_submission(_item("missing")),
            catalog=self.catalog,  # type: ignore[arg-type]
        )

        self.assertFalse(result.reward_valid)
        self.assertEqual(result.terminal_reason, "invalid_submission")
        self.assertIn("unknown_asin", result.recommendations[0].failures)

    def test_unavailable_option_makes_submission_invalid(self) -> None:
        catalog = _Catalog([_product(TARGET_ASIN, option_available=False)])
        result = evaluate_submission(
            facts=_facts(),
            submission=_submission(
                _item(TARGET_ASIN, options={"颜色": "白色"})
            ),
            catalog=catalog,  # type: ignore[arg-type]
        )

        self.assertFalse(result.reward_valid)
        self.assertIn(
            "unavailable_option:颜色=白色",
            result.recommendations[0].failures,
        )

    def test_duplicate_asin_makes_submission_invalid(self) -> None:
        result = evaluate_submission(
            facts=_facts(),
            submission=_submission(
                _item(TARGET_ASIN, options={"颜色": "白色"}),
                _item(TARGET_ASIN, options={"颜色": "白色"}),
            ),
            catalog=self.catalog,  # type: ignore[arg-type]
        )

        self.assertFalse(result.reward_valid)
        self.assertIn(
            f"duplicate_asin:{TARGET_ASIN}",
            result.submission_failures,
        )

    def test_unresolved_gold_options_are_not_scored_as_normal_failure(self) -> None:
        result = evaluate_submission(
            facts=_facts(unresolved_option_values=["白?色"]),
            submission=_submission(
                _item(TARGET_ASIN, options={"颜色": "白色"})
            ),
            catalog=self.catalog,  # type: ignore[arg-type]
        )

        self.assertFalse(result.reward_valid)
        self.assertFalse(result.recommendations[0].strict_match)
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.terminal_reason, "unverifiable_task")

    def test_missing_target_product_is_task_catalog_corruption(self) -> None:
        with self.assertRaisesRegex(EvaluatorDataError, "missing"):
            evaluate_submission(
                facts=_facts(),
                submission=_submission(_item("alternative-1")),
                catalog=_Catalog([_product("alternative-1")]),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
