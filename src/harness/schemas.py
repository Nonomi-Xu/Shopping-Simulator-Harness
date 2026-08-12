from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


TaskSplit = Literal["train", "eval"]
EvaluationTerminalReason = Literal[
    "strict_success",
    "partial_match",
    "no_match",
    "invalid_submission",
    "unverifiable_task",
]


class ProductOption(BaseModel):
    value: str
    price: float | None = None
    is_available: bool = True
    asin: str | None = None

class ProductVariant(BaseModel):
    options: dict[str, str]
    price: float | None = None
    asin: str | None = None
    is_available: bool = True


class ProductDetail(BaseModel):
    asin: str
    title: str
    brand: str | None = None
    category: str
    price_min: float | None = None
    price_max: float | None = None
    pricing: list[float] = Field(default_factory=list)
    description: str = ""
    attributes: list[str] = Field(default_factory=list)
    customization_options: dict[str, list[ProductOption]] = Field(default_factory=dict)
    variant_combinations: list[ProductVariant] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)

class ProductSummary(BaseModel):
    asin: str
    title: str
    category: str
    brand: str | None = None
    price_min: float | None = None
    price_max: float | None = None
    bm25_score: float
    rank: int

class PublicTask(BaseModel):
    """The complete task payload that may be shown to an actor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(pattern=r"^task_[0-9a-f]{24}$")
    query: str = Field(min_length=1)


class PrivateTaskFacts(BaseModel):
    """Gold task facts reserved for deterministic evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(pattern=r"^task_[0-9a-f]{24}$")
    split: TaskSplit
    target_asin: str = Field(min_length=1)
    target_title: str
    target_brand: str | None = None
    target_category: str
    required_options: dict[str, str] = Field(default_factory=dict)
    required_option_values: list[str] = Field(default_factory=list)
    unresolved_option_values: list[str] = Field(default_factory=list)
    required_attributes: list[str] = Field(default_factory=list)
    target_prices: list[float] = Field(default_factory=list)
    source_row_number: int = Field(ge=1)
    source_instruction_index: int = Field(ge=0)


class SearchProductsAction(BaseModel):
    query: str
    limit: int = Field(default=20, ge=1, le=50)


class GetProductAction(BaseModel):
    asin: str


class RecommendationItem(BaseModel):
    asin: str
    selected_options: dict[str, str]
    reason: str


class SubmitRecommendationsAction(BaseModel):
    recommendations: list[RecommendationItem]


class RecommendationEvaluation(BaseModel):
    """Deterministic evaluation facts for one submitted recommendation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asin: str
    rank: int = Field(ge=1)
    product_exists: bool
    asin_match: bool
    selected_options_valid: bool
    required_options_match: bool
    strict_match: bool
    failures: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Terminal result returned by the deterministic evaluator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(pattern=r"^task_[0-9a-f]{24}$")
    evaluator_version: str
    reward: float = Field(ge=0.0, le=1.0)
    strict_success: bool
    reward_valid: bool
    terminal_reason: EvaluationTerminalReason
    matched_rank: int | None = Field(default=None, ge=1)
    submission_failures: list[str] = Field(default_factory=list)
    recommendations: list[RecommendationEvaluation] = Field(
        default_factory=list
    )
