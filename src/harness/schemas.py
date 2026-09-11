from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


TaskSplit = Literal["all", "test"]
EvaluationTerminalReason = Literal[
    "strict_success",
    "partial_match",
    "no_match",
    "invalid_submission",
    "unverifiable_task",
]
SessionStatus = Literal["active", "terminated"]
SessionStepStatus = Literal["success", "rejected", "error"]
SessionTerminalReason = Literal[
    "submitted",
    "max_steps",
    "infrastructure_error",
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
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    query: str = Field(min_length=1)
    limit: int = Field(default=20, ge=1, le=50)


class GetProductAction(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    asin: str = Field(min_length=1)


class RecommendationItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    asin: str = Field(min_length=1)
    selected_options: dict[str, str]
    reason: str = Field(min_length=1)


class SubmitRecommendationsAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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


class SessionState(BaseModel):
    """Public lifecycle state for one isolated task session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_version: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    task_id: str = Field(pattern=r"^task_[0-9a-f]{24}$")
    status: SessionStatus
    step_count: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    surfaced_asins: list[str] = Field(default_factory=list)
    inspected_asins: list[str] = Field(default_factory=list)
    terminal_reason: SessionTerminalReason | None = None


class SessionStep(BaseModel):
    """One actor-visible tool attempt and its public observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str = Field(pattern=r"^step_[0-9]{4}$")
    step_number: int = Field(ge=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: SessionStepStatus
    observation: dict[str, Any] | list[dict[str, Any]] | None = None
    error_code: str | None = None
    message: str | None = None
    terminated: bool
    terminal_reason: SessionTerminalReason | None = None


class SessionSnapshot(BaseModel):
    """Serializable public view of one managed Harness session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    public_task: PublicTask
    state: SessionState
    history: list[SessionStep] = Field(default_factory=list)
    evaluation: EvaluationResult | None = None
