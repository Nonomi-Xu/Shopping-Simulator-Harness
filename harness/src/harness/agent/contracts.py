"""Versioned, provider-independent contracts for the seven-part agent runtime."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..providers import AssistantTurn, ChatMessage, TokenUsage
from ..schemas import ProductDetail, PublicTask, RecommendationItem, SessionSnapshot
from ..constraints import Predicate


RUNTIME_VERSION = "shopping-agent-runtime-v5"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AgentProfile(Contract):
    agent_id: str = "shopping-agent"
    version: str = "shopping-profile-v1"
    role: str = "Catalog-grounded shopping assistant"
    objective: str = "Satisfy explicit requirements using observed product evidence."


class UserPreference(Contract):
    name: str = Field(min_length=1)
    value: str = Field(min_length=1)


class UserProfile(Contract):
    profile_id: str = "anonymous"
    locale: str = "zh-CN"
    preferences: list[UserPreference] = Field(default_factory=list)


class AskUser(Contract):
    question: str = Field(min_length=1, max_length=1000)
    information_gap: str = Field(min_length=1, max_length=1000)
    constraint_ids: list[str] = Field(default_factory=list, max_length=30)
    options: list[str] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def nonblank(self):
        if not self.question.strip() or not self.information_gap.strip():
            raise ValueError("question and information_gap must not be blank")
        if any(not option.strip() for option in self.options):
            raise ValueError("options must not be blank")
        return self


class UserReply(Contract):
    text: str = Field(min_length=1, max_length=4000)
    # The caller, never the model, authorizes replacement of existing constraints.
    replace_constraint_ids: list[str] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def nonblank(self):
        if not self.text.strip():
            raise ValueError("reply must not be blank")
        if len(set(self.replace_constraint_ids)) != len(self.replace_constraint_ids):
            raise ValueError("replacement constraint IDs must be unique")
        return self


class UserMessage(Contract):
    message_id: str
    text: str
    replace_constraint_ids: list[str] = Field(default_factory=list)


class PendingQuestion(AskUser):
    question_id: str


class Constraint(Contract):
    constraint_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    description: str = Field(min_length=1, max_length=500)
    priority: Literal["hard", "soft"]
    source: Literal["user_query", "user_profile", "user_message", "inferred"]
    source_quote: str = Field(min_length=1, max_length=500)
    source_message_id: str | None = None
    topic: str | None = None
    predicate: Predicate | None = None

    @model_validator(mode="after")
    def nonempty_quote(self):
        if not self.source_quote.strip():
            raise ValueError("source_quote must contain non-whitespace text")
        return self


class TaskProfile(Contract):
    task: PublicTask
    constraints: list[Constraint] = Field(default_factory=list)
    requirements_status: Literal["unparsed", "parsed"] = "unparsed"
    no_constraints_reason: str | None = None
    revision: int = Field(default=0, ge=0)
    user_messages: list[UserMessage] = Field(default_factory=list, max_length=20)
    uncertain_requirements: list[str] = Field(default_factory=list, max_length=30)


class PlanUpdate(Contract):
    constraints: list[Constraint] = Field(default_factory=list, max_length=30)
    uncertain_requirements: list[str] = Field(default_factory=list, max_length=30)
    queries: list[str] = Field(min_length=1, max_length=5)
    verification_targets: list[str] = Field(default_factory=list, max_length=30)
    change_summary: str = Field(min_length=1, max_length=1000)
    evidence_observation_ids: list[str] = Field(default_factory=list, max_length=20)
    no_constraints_reason: str | None = Field(
        default=None, min_length=1, max_length=500
    )

    @model_validator(mode="after")
    def validate_queries(self):
        if any(not q.strip() or len(q) > 1000 for q in self.queries):
            raise ValueError("queries must contain 1-1000 characters")
        ids = [item.constraint_id for item in self.constraints]
        if len(ids) != len(set(ids)):
            raise ValueError("constraint IDs must be unique")
        if not self.constraints and not (self.no_constraints_reason or "").strip():
            raise ValueError(
                "empty constraints require an explicit no_constraints_reason"
            )
        return self


class SearchPlan(Contract):
    version: str = "shopping-plan-v2"
    plan_id: str
    revision: int = Field(default=0, ge=0)
    objective: str
    queries: list[str]
    verification_targets: list[str] = Field(default_factory=list)
    phase: Literal[
        "search",
        "inspect",
        "verify",
        "submit",
        "waiting_for_user",
        "completed",
        "stopped",
    ] = "search"
    change_summary: str = "Bootstrap from the public query; refine with update_plan."
    stopping_conditions: list[str] = Field(
        default_factory=lambda: [
            "Inspect recommended products and verify hard constraints from public evidence.",
            "Use best_effort with explicit gaps in each reason when requirements remain unresolved.",
        ]
    )


class RuntimeConfig(Contract):
    orchestration_backend: Literal["native", "langgraph"] = "native"
    agent_mode: Literal["single", "multi"] = "single"
    interaction_mode: Literal["offline", "interactive"] = "offline"
    progressive_disclosure: bool = False
    enforce_constraint_filters: bool = False
    max_user_questions: int = Field(default=3, ge=1, le=20, strict=True)
    max_actions: int = Field(default=40, ge=1, strict=True)
    max_searches: int = Field(default=6, ge=1, strict=True)
    max_repeated_actions: int = Field(default=2, ge=1, strict=True)
    max_batch_calls: int = Field(default=8, ge=1, le=20, strict=True)
    max_total_tokens: int | None = Field(default=None, ge=1, strict=True)
    max_duration_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    context_max_chars: int = Field(default=80000, ge=2000, strict=True)
    recent_turns: int = Field(default=4, ge=1, strict=True)
    memory_candidate_limit: int = Field(default=10, ge=1, le=50, strict=True)
    action_proposal_limit: int = Field(default=5, ge=0, le=8, strict=True)


class BudgetState(Contract):
    actions_remaining: int = Field(ge=0)
    searches_remaining: int = Field(ge=0)
    environment_steps_remaining: int = Field(ge=0)
    model_turns_remaining: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    usage_complete: bool


class AgentAction(Contract):
    version: str = "shopping-action-v1"
    action_id: str
    session_id: str
    sequence: int = Field(ge=1)
    plan_id: str
    plan_revision: int = Field(ge=0)
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    task_revision: int = Field(default=0, ge=0)
    agent_role: Literal["single", "planner", "executor", "observer"] = "single"


class ReviewProgress(Contract):
    next_step: Literal["act", "replan", "recommend"]
    summary: str = Field(min_length=1, max_length=1500)
    reviewed_asins: list[str] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def valid_review(self):
        if not self.summary.strip():
            raise ValueError("review summary must not be blank")
        if len(set(self.reviewed_asins)) != len(self.reviewed_asins):
            raise ValueError("reviewed ASINs must be unique")
        if self.next_step == "recommend" and not self.reviewed_asins:
            raise ValueError("recommend requires reviewed_asins")
        if self.next_step != "recommend" and self.reviewed_asins:
            raise ValueError("reviewed_asins are only used to authorize recommendations")
        return self


class ProjectionMetadata(Contract):
    version: str = "shopping-observation-v4"
    returned_items: int | None = None
    # Unknown total is deliberately not replaced by the returned page length.
    total_results: int | None = None
    truncated: bool = False
    next_offset: int | None = None
    has_more: bool | None = None
    observed_fields: list[str] | None = None
    price_scope: str | None = None
    source: dict[str, Any] | None = None
    applied_filters: dict[str, Any] | None = None


class AgentObservation(Contract):
    observation_id: str
    action_id: str
    tool_name: str
    status: Literal["success", "rejected", "error"]
    data: dict[str, Any] | list[dict[str, Any]] | None = None
    error_code: str | None = None
    message: str | None = None
    environment_step_id: str | None = None
    terminated: bool = False
    terminal_reason: str | None = None
    allowed_next_tools: list[str] = Field(default_factory=list)
    budget: BudgetState
    projection: ProjectionMetadata = Field(default_factory=ProjectionMetadata)


class EvidenceAssessment(Contract):
    asin: str = Field(min_length=1)
    constraint_id: str = Field(min_length=1)
    status: Literal["satisfied", "violated", "unknown"]
    observation_id: str = Field(min_length=1)
    # RFC 6901 pointer into the observation's data, e.g. /attributes/0.
    field_path: str = Field(pattern=r"^/")
    quote: str = Field(min_length=1, max_length=2000)
    explanation: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def nonempty_quote(self):
        if not self.quote.strip():
            raise ValueError("evidence quote must contain non-whitespace text")
        return self


class EvidenceBatch(Contract):
    assessments: list[EvidenceAssessment] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def unique_targets(self):
        targets = [(a.asin, a.constraint_id) for a in self.assessments]
        if len(targets) != len(set(targets)):
            raise ValueError("a batch may assess each product/constraint only once")
        return self


class ReadEvidence(Contract):
    asin: str = Field(min_length=1)
    observation_id: str | None = None
    field_paths: list[str] = Field(default_factory=list, max_length=12)
    offset: int = Field(default=0, ge=0, strict=True)
    limit: int = Field(default=10, ge=1, le=20, strict=True)

    @model_validator(mode="after")
    def valid_paths(self):
        if any(not path.startswith("/") for path in self.field_paths):
            raise ValueError("field_paths must be JSON Pointers beginning with /")
        return self


class SubmissionGap(Contract):
    asin: str = Field(min_length=1)
    constraint_id: str = Field(min_length=1)
    status: Literal["unknown", "violated"]
    disclosure: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def nonempty_disclosure(self):
        if not self.disclosure.strip():
            raise ValueError("gap disclosure must contain non-whitespace text")
        return self


class AgentSubmission(Contract):
    """Runtime-only preflight metadata; only recommendations reach the environment."""

    recommendations: list[RecommendationItem] = Field(min_length=1, max_length=5)
    mode: Literal["verified", "best_effort"] = "verified"
    gaps: list[SubmissionGap] = Field(default_factory=list, max_length=150)


class CompareProducts(Contract):
    asins: list[str] = Field(min_length=2, max_length=5)
    constraint_ids: list[str] = Field(default_factory=list, max_length=10)
    fields: list[str] = Field(
        default_factory=lambda: ["brand", "price_min", "price_max"], max_length=8
    )

    @model_validator(mode="after")
    def unique_and_public(self):
        if len(set(self.asins)) != len(self.asins):
            raise ValueError("comparison requires distinct ASINs")
        if any(
            field not in ProductDetail.model_fields or field == "asin"
            for field in self.fields
        ):
            raise ValueError("comparison fields must be public product fields")
        return self


class VerifySelection(Contract):
    asin: str = Field(min_length=1)
    selected_options: dict[str, str] = Field(default_factory=dict)
    max_price: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    budget_constraint_id: str | None = None


class ActionProposal(Contract):
    tool_name: str
    arguments: dict[str, Any]
    information_gap: str
    target_asins: list[str] = Field(default_factory=list)
    constraint_ids: list[str] = Field(default_factory=list)
    evidence_observation_ids: list[str] = Field(default_factory=list)
    environment_steps: int = 0
    requires_model_input: list[str] = Field(default_factory=list)


class CandidateMemory(Contract):
    asin: str
    summary: dict[str, Any]
    inspected: bool = False
    observation_ids: list[str] = Field(default_factory=list)
    detail_observation_id: str | None = None
    assessments: dict[str, EvidenceAssessment] = Field(default_factory=dict)


class WorkingMemory(Contract):
    version: str = "shopping-memory-v3"
    session_id: str
    queries: list[str] = Field(default_factory=list)
    candidates: dict[str, CandidateMemory] = Field(default_factory=dict)
    evidence_archive: dict[str, ProductDetail] = Field(default_factory=dict)
    detail_fields: dict[str, list[str]] = Field(default_factory=dict)
    evidence_sources: dict[str, dict[str, Any]] = Field(default_factory=dict)
    selection_checks: dict[str, dict[str, Any]] = Field(default_factory=dict)
    search_requests: list[dict[str, Any]] = Field(default_factory=list)
    last_error: str | None = None


class AgentEvent(Contract):
    version: str = "shopping-event-v1"
    event_id: str
    session_id: str
    sequence: int = Field(ge=1)
    kind: Literal[
        "run_started",
        "context_built",
        "model_turn",
        "plan_updated",
        "action",
        "observation",
        "memory_updated",
        "run_finished",
        "run_paused",
        "run_resumed",
        "user_message",
        "agent_handoff",
    ]
    caused_by: str | None = None
    payload: dict[str, Any]
    previous_hash: str
    event_hash: str


RunStatus = Literal["completed", "terminated", "failed", "waiting_for_user"]
RunTerminalReason = Literal[
    "submitted",
    "max_steps",
    "infrastructure_error",
    "provider_error",
    "model_stopped_without_submission",
    "multiple_tool_calls",
    "max_model_turns",
    "max_actions",
    "token_budget_exhausted",
    "usage_unavailable",
    "time_budget_exhausted",
    "context_budget_exhausted",
    "runtime_error",
    "user_input_required",
    "simulator_error",
]


class AgentRunResult(Contract):
    runner_version: str = RUNTIME_VERSION
    status: RunStatus
    terminal_reason: RunTerminalReason
    task_id: str
    session_id: str
    provider: str
    model: str
    skill_name: str
    skill_sha256: str
    model_turn_count: int = Field(ge=0)
    # Environment calls only; action_count includes planning and memory actions.
    tool_call_count: int = Field(ge=0)
    action_count: int = Field(default=0, ge=0)
    search_count: int = Field(default=0, ge=0)
    usage: TokenUsage | None = None
    messages: list[ChatMessage]
    turns: list[AssistantTurn]
    session: SessionSnapshot
    manifest: dict[str, Any] = Field(default_factory=dict)
    profile: AgentProfile = Field(default_factory=AgentProfile)
    user_profile: UserProfile = Field(default_factory=UserProfile)
    task_profile: TaskProfile | None = None
    plan: SearchPlan | None = None
    memory: WorkingMemory | None = None
    events: list[AgentEvent] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None
    pending_question: PendingQuestion | None = None
    active_duration_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
