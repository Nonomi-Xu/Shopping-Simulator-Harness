from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


TaskSplit = Literal["all", "test"]


@dataclass(frozen=True)
class PublicTask:
    task_id: str
    query: str


@dataclass(frozen=True)
class PrivateTaskFacts:
    task_id: str
    split: TaskSplit
    target_asin: str
    required_options: dict[str, str]
    unresolved_option_values: list[str]


@dataclass(frozen=True)
class Candidate:
    rank: int
    bm25_score: float
    product: dict[str, Any]

    @property
    def asin(self) -> str:
        return str(self.product["asin"])

    def agent_payload(self) -> dict[str, Any]:
        """Return only decision-relevant public product fields."""
        return {
            "bm25_rank": self.rank,
            "bm25_score": self.bm25_score,
            "asin": self.asin,
            "title": self.product.get("title", ""),
            "brand": self.product.get("brand"),
            "category": self.product.get("category", ""),
            "price_min": self.product.get("price_min"),
            "price_max": self.product.get("price_max"),
            "pricing": self.product.get("pricing", []),
            "description": self.product.get("description", ""),
            "attributes": self.product.get("attributes", []),
            "customization_options": self.product.get(
                "customization_options", {}
            ),
            "variant_combinations": self.product.get(
                "variant_combinations", []
            ),
        }


@dataclass(frozen=True)
class AgentDecision:
    asin: str
    selected_options: dict[str, str]
    reason: str


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class SelectorResult:
    decision: AgentDecision
    provider: str
    model: str
    usage: TokenUsage | None = None
    response_id: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    reward: float
    reward_valid: bool
    strict_success: bool
    terminal_reason: str
    target_in_top5: bool
    target_bm25_rank: int | None
    asin_match: bool
    required_options_match: bool
    selected_options_valid: bool
    failures: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TaskResult:
    pipeline_version: str
    status: str
    task: PublicTask
    candidates: list[Candidate]
    selector: SelectorResult | None
    evaluation: EvaluationResult | None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
