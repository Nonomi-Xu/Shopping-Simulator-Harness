"""Deterministic evaluation compatible with the Harness reward semantics."""

from __future__ import annotations

from .types import AgentDecision, Candidate, EvaluationResult, PrivateTaskFacts


def validate_decision(
    decision: AgentDecision,
    candidates: list[Candidate],
) -> list[str]:
    """Reject any selection outside the frozen Top-5 candidate boundary."""
    failures: list[str] = []
    if not decision.asin.strip():
        failures.append("empty_asin")
    if not decision.reason.strip():
        failures.append("empty_reason")
    products = {candidate.asin: candidate.product for candidate in candidates}
    product = products.get(decision.asin)
    if product is None:
        failures.append("asin_outside_top5")
        return failures

    customization_options = product.get("customization_options")
    if not isinstance(customization_options, dict):
        customization_options = {}
    for raw_axis, raw_value in decision.selected_options.items():
        axis = str(raw_axis or "").strip()
        value = str(raw_value or "").strip()
        if not axis:
            failures.append("empty_option_axis")
            continue
        if not value:
            failures.append(f"empty_option_value:{axis}")
            continue
        choices = customization_options.get(axis)
        if not isinstance(choices, list):
            failures.append(f"unknown_option_axis:{axis}")
            continue
        matches = [
            option
            for option in choices
            if isinstance(option, dict) and str(option.get("value")) == value
        ]
        if not matches:
            failures.append(f"unknown_option_value:{axis}={value}")
        elif not any(bool(option.get("is_available", True)) for option in matches):
            failures.append(f"unavailable_option:{axis}={value}")
    return failures


def evaluate_decision(
    facts: PrivateTaskFacts,
    decision: AgentDecision,
    candidates: list[Candidate],
) -> EvaluationResult:
    """Score one selected product using Harness v1's 1.0 / 0.25 / 0 rule."""
    target_rank = next(
        (candidate.rank for candidate in candidates if candidate.asin == facts.target_asin),
        None,
    )
    failures = validate_decision(decision, candidates)
    asin_match = decision.asin == facts.target_asin
    required_options_match = asin_match and all(
        decision.selected_options.get(axis) == value
        for axis, value in facts.required_options.items()
    )
    options_valid = not any(
        failure.startswith(
            (
                "empty_option_",
                "unknown_option_",
                "unavailable_option:",
            )
        )
        for failure in failures
    )

    if failures:
        return EvaluationResult(
            reward=0.0,
            reward_valid=False,
            strict_success=False,
            terminal_reason="invalid_agent_decision",
            target_in_top5=target_rank is not None,
            target_bm25_rank=target_rank,
            asin_match=asin_match,
            required_options_match=required_options_match,
            selected_options_valid=options_valid,
            failures=failures,
        )
    if facts.unresolved_option_values:
        return EvaluationResult(
            reward=0.0,
            reward_valid=False,
            strict_success=False,
            terminal_reason="unverifiable_task",
            target_in_top5=target_rank is not None,
            target_bm25_rank=target_rank,
            asin_match=asin_match,
            required_options_match=required_options_match,
            selected_options_valid=True,
            failures=["unresolved_gold_options"],
        )
    if asin_match and required_options_match:
        return EvaluationResult(
            reward=1.0,
            reward_valid=True,
            strict_success=True,
            terminal_reason="strict_success",
            target_in_top5=target_rank is not None,
            target_bm25_rank=target_rank,
            asin_match=True,
            required_options_match=True,
            selected_options_valid=True,
        )
    if asin_match:
        return EvaluationResult(
            reward=0.25,
            reward_valid=True,
            strict_success=False,
            terminal_reason="partial_match",
            target_in_top5=target_rank is not None,
            target_bm25_rank=target_rank,
            asin_match=True,
            required_options_match=False,
            selected_options_valid=True,
        )
    return EvaluationResult(
        reward=0.0,
        reward_valid=True,
        strict_success=False,
        terminal_reason="no_match",
        target_in_top5=target_rank is not None,
        target_bm25_rank=target_rank,
        asin_match=False,
        required_options_match=False,
        selected_options_valid=True,
    )
