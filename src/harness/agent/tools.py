"""Authoritative model tool surface for the shopping Agent."""

from __future__ import annotations

from ..providers import ToolDefinition
from ..schemas import (
    GetProductAction,
    SearchProductsAction,
    SubmitRecommendationsAction,
)
from ..session import GET_PRODUCT, SEARCH_PRODUCTS, SUBMIT_RECOMMENDATIONS


def shopping_tool_definitions() -> tuple[ToolDefinition, ...]:
    """Generate tools from the same Pydantic actions enforced by Harness."""
    return (
        ToolDefinition.from_action_model(
            name=SEARCH_PRODUCTS,
            description=(
                "Search the public product catalog. Returned ASINs become "
                "eligible for get_product in this session."
            ),
            action_model=SearchProductsAction,
        ),
        ToolDefinition.from_action_model(
            name=GET_PRODUCT,
            description=(
                "Inspect the public detail of one ASIN previously returned "
                "by search_products."
            ),
            action_model=GetProductAction,
        ),
        ToolDefinition.from_action_model(
            name=SUBMIT_RECOMMENDATIONS,
            description=(
                "Submit the final ranked recommendations. Every ASIN must "
                "have been inspected; an accepted call terminates the session."
            ),
            action_model=SubmitRecommendationsAction,
        ),
    )
