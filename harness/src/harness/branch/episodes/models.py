"""Separate private episode contracts from allowlisted public trajectories."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import Field, model_validator
from ...agent.contracts import Contract, Constraint, AgentAction, AgentObservation
from ...constraints import Predicate
from ...schemas import PublicTask, RecommendationItem


class GoalRequirement(Contract):
    requirement_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    topic: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=500)
    source_quote: str = Field(min_length=1, max_length=500)
    priority: Literal["hard", "soft"]
    predicate: Predicate | None = None
    critical: bool = False


class UserGoal(Contract):
    """Query-derived only: never enrich this object from a target product."""
    version: str = "shopping-user-goal-v1"
    requirements: list[GoalRequirement] = Field(min_length=1, max_length=30)
    status: Literal["ready", "needs_review"] = "ready"
    review_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [r.requirement_id for r in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate requirement IDs")
        if self.review_reasons and self.status == "ready":
            raise ValueError("review reasons require needs_review status")
        return self


class DisclosurePolicy(Contract):
    version: str = "shopping-disclosure-v1"
    initial_topics: list[str] = Field(default_factory=lambda: ["category", "use_case"], min_length=1)
    max_initial_requirements: int = Field(default=2, ge=1, le=5)
    max_requirements_per_answer: int = Field(default=1, ge=1, le=3)


class DisclosureState(Contract):
    """Private simulator state. Hidden field names must not be sent to the actor."""
    revealed_fields: set[str] = Field(default_factory=set)
    unrevealed_fields: set[str] = Field(default_factory=set)
    answered_topics: set[str] = Field(default_factory=set)


class ShoppingEpisode(Contract):
    """Private controller record; simulator receives only hidden_goal and policy."""
    version: str = "shopping-episode-v1"
    episode_id: str = Field(pattern=r"^episode_[0-9a-f]{24}$")
    task_id: str = Field(pattern=r"^task_[0-9a-f]{24}$")
    full_query: str = Field(min_length=1)
    hidden_goal: UserGoal
    initial_observation: str = Field(min_length=1)
    initial_requirement_ids: list[str] = Field(min_length=1)
    disclosure_policy: DisclosurePolicy = Field(default_factory=DisclosurePolicy)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parser_manifest: dict[str, Any]
    # Original TaskStore is the sole owner of target_asin and private answer facts.
    # task_id is the evaluator lookup key; duplicating gold facts invites drift.

    @model_validator(mode="after")
    def consistent(self):
        by_id = {r.requirement_id: r for r in self.hidden_goal.requirements}
        if any(r.source_quote not in self.full_query for r in by_id.values()):
            raise ValueError("goal requirements must quote the original query")
        ids = self.initial_requirement_ids
        if len(ids) != len(set(ids)) or not set(ids) <= by_id.keys():
            raise ValueError("invalid initial disclosure IDs")
        if self.initial_observation != "；".join(by_id[i].source_quote for i in ids):
            raise ValueError("initial observation must contain only selected disclosure quotes")
        if len(ids) >= len(by_id):
            raise ValueError("interactive episodes must leave a requirement undisclosed")
        expected = [r.requirement_id for r in self.hidden_goal.requirements
                    if r.topic in self.disclosure_policy.initial_topics][:self.disclosure_policy.max_initial_requirements]
        if ids != expected:
            raise ValueError("initial disclosure does not match policy")
        if any(r.source_quote in self.initial_observation for r in by_id.values() if r.requirement_id not in ids):
            raise ValueError("initial observation exposes an undisclosed requirement")
        return self

    def public_task(self, *, full_query: bool = False) -> PublicTask:
        return PublicTask(task_id=self.task_id, query=self.full_query if full_query else self.initial_observation)


class PreferenceState(Contract):
    """Actor-derived snapshot; TaskProfile.constraints remains the source of truth."""
    known_requirements: list[Constraint] = Field(default_factory=list)
    uncertain_requirements: list[str] = Field(default_factory=list)
    requirements_status: Literal["unparsed", "parsed"] = "unparsed"
    revision: int = 0


class CandidateSnapshot(Contract):
    surfaced_ids: list[str] = Field(default_factory=list)
    inspected_ids: list[str] = Field(default_factory=list)
    returned_ids: list[str] = Field(default_factory=list)
    search_total: int | None = None


class TrajectoryTurn(Contract):
    turn: int = Field(ge=1)
    action_type: Literal["ask", "search", "inspect", "compare", "recommend", "internal"]
    preferences_before: PreferenceState
    preferences_after: PreferenceState
    action: AgentAction
    observation: AgentObservation
    user_reply: str | None = None
    candidates: CandidateSnapshot


class Trajectory(Contract):
    """Public only. Scores, hidden IDs and evaluator metadata live in another file."""
    version: str = "shopping-trajectory-v1"
    episode_id: str
    task: PublicTask
    baseline: Literal["full_query", "long_horizon"]
    turns: list[TrajectoryTurn]
    final_recommendation: list[RecommendationItem] = Field(default_factory=list)
    terminal_reason: str
    model_turn_count: int
    tool_call_count: int
    environment_call_count: int


class EpisodeConfig(Contract):
    seed: int = 42
    max_turns: int = Field(default=60, ge=1, strict=True)
    max_tool_calls: int = Field(default=60, ge=1, strict=True)
    max_environment_calls: int = Field(default=20, ge=1, strict=True)
    max_searches: int = Field(default=8, ge=1, strict=True)
    max_user_questions: int = Field(default=10, ge=1, le=20, strict=True)
    currency: Literal["CNY"] = "CNY"


class DisclosureEvent(Contract):
    """Private audit of a simulator answer, including counterfactual pool counts."""
    question_id: str
    action_sequence: int
    topic: str | None
    revealed_ids: list[str]
    repeated: bool
    answer: str
    candidates_before: int | None = None
    candidates_after: int | None = None
    executable_constraints: int = 0
