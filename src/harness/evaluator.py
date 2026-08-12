"""Deterministic scoring for terminal recommendation submissions.

Evaluator v1 deliberately scores only the dataset's annotated target product
and required option values.  It does not call an LLM and does not infer valid
alternative products.  This keeps rewards replayable while the Harness is
still establishing its core runtime contract.
"""

from __future__ import annotations

from .catalog import ProductCatalog, ProductNotFoundError
from .schemas import (
    EvaluationResult,
    PrivateTaskFacts,
    ProductDetail,
    RecommendationEvaluation,
    RecommendationItem,
    SubmitRecommendationsAction,
)


EVALUATOR_VERSION = "shopping-recommendation-evaluator-v1"
MAX_RECOMMENDATIONS = 5
STRICT_SUCCESS_REWARD = 1.0
TARGET_WITH_OPTION_MISMATCH_REWARD = 0.25
FAILURE_REWARD = 0.0


class EvaluatorError(RuntimeError):
    """Base exception raised by deterministic recommendation evaluation."""


class EvaluatorDataError(EvaluatorError):
    """Private task facts and the public catalog are inconsistent."""


def _submission_failures(
    recommendations: list[RecommendationItem],
) -> tuple[list[str], set[int]]:
    """Return submission-level failures and ranks containing duplicate ASINs."""
    failures: list[str] = []
    duplicate_ranks: set[int] = set()
    count = len(recommendations)
    if count == 0:
        failures.append("empty_recommendations")
    if count > MAX_RECOMMENDATIONS:
        failures.append(
            f"too_many_recommendations:{count}>{MAX_RECOMMENDATIONS}"
        )

    ranks_by_asin: dict[str, list[int]] = {}
    for rank, item in enumerate(recommendations, start=1):
        asin = str(item.asin or "").strip()
        ranks_by_asin.setdefault(asin, []).append(rank)
    for asin, ranks in ranks_by_asin.items():
        if asin and len(ranks) > 1:
            failures.append(f"duplicate_asin:{asin}")
            duplicate_ranks.update(ranks)
    return failures, duplicate_ranks


def _option_failures(
    item: RecommendationItem,
    product: ProductDetail,
) -> list[str]:
    """Validate every submitted option against the public product record."""
    failures: list[str] = []
    for raw_axis, raw_value in item.selected_options.items():
        axis = str(raw_axis or "").strip()
        value = str(raw_value or "").strip()
        if not axis:
            failures.append("empty_option_axis")
            continue
        if not value:
            failures.append(f"empty_option_value:{axis}")
            continue

        catalog_options = product.customization_options.get(axis)
        if catalog_options is None:
            failures.append(f"unknown_option_axis:{axis}")
            continue
        matching = [option for option in catalog_options if option.value == value]
        if not matching:
            failures.append(f"unknown_option_value:{axis}={value}")
            continue
        if not any(option.is_available for option in matching):
            failures.append(f"unavailable_option:{axis}={value}")
    return failures


def _required_options_match(
    facts: PrivateTaskFacts,
    item: RecommendationItem,
) -> bool:
    if item.asin.strip() != facts.target_asin:
        return False
    return all(
        item.selected_options.get(axis) == expected_value
        for axis, expected_value in facts.required_options.items()
    )


def _evaluate_item(
    *,
    item: RecommendationItem,
    rank: int,
    facts: PrivateTaskFacts,
    catalog: ProductCatalog,
    target_product: ProductDetail,
    duplicate: bool,
) -> RecommendationEvaluation:
    asin = str(item.asin or "").strip()
    failures: list[str] = []
    if not asin:
        failures.append("empty_asin")
    if not str(item.reason or "").strip():
        failures.append("empty_reason")
    if duplicate:
        failures.append("duplicate_asin")

    product: ProductDetail | None = None
    if asin:
        if asin == facts.target_asin:
            product = target_product
        else:
            try:
                product = catalog.get_product(asin)
            except ProductNotFoundError:
                failures.append("unknown_asin")

    option_failures = _option_failures(item, product) if product else []
    failures.extend(option_failures)
    product_exists = product is not None
    asin_match = asin == facts.target_asin
    selected_options_valid = product_exists and not option_failures
    required_options_match = _required_options_match(facts, item)
    strict_match = (
        asin_match
        and selected_options_valid
        and required_options_match
        and not facts.unresolved_option_values
        and not failures
    )
    return RecommendationEvaluation(
        asin=asin,
        rank=rank,
        product_exists=product_exists,
        asin_match=asin_match,
        selected_options_valid=selected_options_valid,
        required_options_match=required_options_match,
        strict_match=strict_match,
        failures=failures,
    )


def evaluate_submission(
    *,
    facts: PrivateTaskFacts,
    submission: SubmitRecommendationsAction,
    catalog: ProductCatalog,
) -> EvaluationResult:
    """Evaluate one terminal recommendation submission deterministically.

    Invalid submissions and tasks with unresolved gold option annotations fail
    closed with ``reward_valid=False``.  A well-formed submission receives full
    reward for an exact target/option match, partial reward when the target ASIN
    is present but required options are missing or wrong, and zero otherwise.
    """
    if not isinstance(facts, PrivateTaskFacts):
        raise TypeError("facts must be PrivateTaskFacts")
    if not isinstance(submission, SubmitRecommendationsAction):
        raise TypeError("submission must be SubmitRecommendationsAction")

    try:
        target_product = catalog.get_product(facts.target_asin)
    except ProductNotFoundError as exc:
        raise EvaluatorDataError(
            "target product is missing from the public catalog: "
            f"{facts.target_asin}"
        ) from exc

    submission_failures, duplicate_ranks = _submission_failures(
        submission.recommendations
    )
    item_results = [
        _evaluate_item(
            item=item,
            rank=rank,
            facts=facts,
            catalog=catalog,
            target_product=target_product,
            duplicate=rank in duplicate_ranks,
        )
        for rank, item in enumerate(submission.recommendations, start=1)
    ]

    item_submission_invalid = any(result.failures for result in item_results)
    if submission_failures or item_submission_invalid:
        return EvaluationResult(
            task_id=facts.task_id,
            evaluator_version=EVALUATOR_VERSION,
            reward=FAILURE_REWARD,
            strict_success=False,
            reward_valid=False,
            terminal_reason="invalid_submission",
            submission_failures=submission_failures,
            recommendations=item_results,
        )

    matched_rank = next(
        (result.rank for result in item_results if result.asin_match),
        None,
    )
    if facts.unresolved_option_values:
        return EvaluationResult(
            task_id=facts.task_id,
            evaluator_version=EVALUATOR_VERSION,
            reward=FAILURE_REWARD,
            strict_success=False,
            reward_valid=False,
            terminal_reason="unverifiable_task",
            matched_rank=matched_rank,
            submission_failures=["unresolved_gold_options"],
            recommendations=item_results,
        )

    strict_rank = next(
        (result.rank for result in item_results if result.strict_match),
        None,
    )
    if strict_rank is not None:
        return EvaluationResult(
            task_id=facts.task_id,
            evaluator_version=EVALUATOR_VERSION,
            reward=STRICT_SUCCESS_REWARD,
            strict_success=True,
            reward_valid=True,
            terminal_reason="strict_success",
            matched_rank=strict_rank,
            recommendations=item_results,
        )
    if matched_rank is not None:
        return EvaluationResult(
            task_id=facts.task_id,
            evaluator_version=EVALUATOR_VERSION,
            reward=TARGET_WITH_OPTION_MISMATCH_REWARD,
            strict_success=False,
            reward_valid=True,
            terminal_reason="partial_match",
            matched_rank=matched_rank,
            recommendations=item_results,
        )
    return EvaluationResult(
        task_id=facts.task_id,
        evaluator_version=EVALUATOR_VERSION,
        reward=FAILURE_REWARD,
        strict_success=False,
        reward_valid=True,
        terminal_reason="no_match",
        recommendations=item_results,
    )
