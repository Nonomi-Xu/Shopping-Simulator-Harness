"""The seven-part runtime: profiles -> plan -> action -> observation -> memory."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
import time
from typing import Protocol, runtime_checkable

from ..providers import (
    AssistantTurn,
    ChatMessage,
    ModelProvider,
    ProviderError,
    TokenUsage,
    canonical_json,
)
from ..schemas import SessionSnapshot
from .context import ContextBudgetError, ContextBuilder, RUNTIME_INSTRUCTIONS
from .contracts import (
    AgentAction,
    AgentObservation,
    AgentProfile,
    AgentRunResult,
    BudgetState,
    EvidenceAssessment,
    EvidenceBatch,
    ReadEvidence,
    PlanUpdate,
    RuntimeConfig,
    TaskProfile,
    UserProfile,
    UserReply,
    UserMessage,
    PendingQuestion,
    ReviewProgress,
)
from .events import EventStore, digest, replay_events
from .memory import MemoryStore
from .comparison import compare_products, verify_selection
from .proposals import action_signature
from .observations import ObservationProjector
from .planner import Planner
from .policy import ModelPolicy
from .skill import AgentSkill, load_agent_skill
from .tools import ToolRegistry
from .submission import SubmissionNotReady, check_submission
from .multiagent import MultiAgentCoordinator, ROLE_TOOLS


@runtime_checkable
class HarnessRuntime(Protocol):
    def create_session(
        self,
        task_id: str,
        *,
        max_steps: int | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot: ...
    def dispatch(self, session_id: str, tool_name: str, arguments: Mapping): ...
    def get_session(self, session_id: str) -> SessionSnapshot: ...


class AgentOrchestrator:
    """Reusable configuration; every run owns fresh mutable state and evidence."""

    version = "shopping-orchestrator-v5"

    def __init__(
        self,
        *,
        provider: ModelProvider,
        harness: HarnessRuntime,
        skill: AgentSkill | None = None,
        skill_dir: str | Path | None = None,
        max_model_turns: int = 40,
        tool_choice="required",
        provider_parameters: Mapping | None = None,
        config: RuntimeConfig | None = None,
        profile: AgentProfile | None = None,
        user_profile: UserProfile | None = None,
    ):
        if skill is not None and skill_dir is not None:
            raise ValueError("pass skill or skill_dir, not both")
        if isinstance(max_model_turns, bool) or not isinstance(max_model_turns, int):
            raise TypeError("max_model_turns must be an integer")
        if max_model_turns < 1:
            raise ValueError("max_model_turns must be positive")
        if not isinstance(provider, ModelProvider):
            raise TypeError("provider must implement ModelProvider")
        if not isinstance(harness, HarnessRuntime):
            raise TypeError("harness must implement HarnessRuntime")
        self.provider = provider
        self.harness = harness
        self.skill = skill or load_agent_skill(skill_dir)
        self.max_model_turns = max_model_turns
        self.config = (config or RuntimeConfig()).model_copy(deep=True)
        self.profile = (profile or AgentProfile()).model_copy(deep=True)
        self.user_profile = (user_profile or UserProfile()).model_copy(deep=True)
        if (
            self.config.interaction_mode == "interactive"
            and getattr(harness, "mode", None) != "interactive"
        ):
            raise ValueError(
                "interactive mode requires InteractiveHarness; benchmark tasks cannot be revised"
            )
        self.registry = ToolRegistry(
            interactive=self.config.interaction_mode == "interactive",
            multi_agent=self.config.agent_mode == "multi",
        )
        self.planner = Planner()
        self.projector = ObservationProjector()
        self.context_builder = ContextBuilder()
        self.policy = ModelPolicy(provider, tool_choice, provider_parameters)

    def run(
        self,
        task_id: str,
        *,
        max_steps: int | None = None,
        session_id: str | None = None,
    ) -> AgentRunResult:
        if self.config.interaction_mode == "interactive":
            raise ValueError("use run_query for interactive tasks")
        started = self.harness.create_session(
            task_id, max_steps=max_steps, session_id=session_id
        )
        return _Run(self, started).execute()

    def run_query(
        self, query: str, *, max_steps=None, session_id=None
    ) -> AgentRunResult:
        """Start an unlabelled real-user task; never invokes the benchmark evaluator."""
        if self.config.interaction_mode != "interactive":
            raise ValueError("run_query requires interaction_mode=interactive")
        started = self.harness.create_interactive_session(
            query, max_steps=max_steps, session_id=session_id
        )
        return _Run(self, started).execute()

    def run_public_task(self, task, *, max_steps=None, session_id=None):
        """Start an explicitly identified, actor-safe task without gold access."""
        from ..schemas import PublicTask
        task = PublicTask.model_validate(task)
        started = self.harness.create_public_session(task, max_steps=max_steps, session_id=session_id)
        return _Run(self, started).execute()

    def resume(self, paused: AgentRunResult, reply: UserReply | str) -> AgentRunResult:
        """Resume a validated waiting checkpoint without repeating environment calls."""
        if self.config.interaction_mode != "interactive":
            raise ValueError("resume requires interaction_mode=interactive")
        reply = (
            UserReply(text=reply)
            if isinstance(reply, str)
            else UserReply.model_validate(reply)
        )
        validate_checkpoint(paused)
        if (
            paused.manifest["config"] != self.config.model_dump(mode="json")
            or paused.manifest["max_model_turns"] != self.max_model_turns
            or paused.skill_sha256 != self.skill.source_sha256
            or paused.profile != self.profile
            or paused.user_profile != self.user_profile
            or paused.provider != self.provider.name
            or paused.model != self.provider.model
            or paused.manifest.get("provider_endpoint_sha256")
            != digest(getattr(self.provider, "base_url", None))
            or paused.manifest["tool_choice"] != self.policy.tool_choice
            or paused.manifest["provider_parameters"]
            != _redact(
                {
                    **getattr(self.provider, "default_parameters", {}),
                    **self.policy.parameters,
                }
            )
            or paused.manifest["harness"] != self.harness.manifest
        ):
            raise ValueError(
                "checkpoint configuration, profile, skill or source has changed"
            )
        if not set(reply.replace_constraint_ids) <= {
            c.constraint_id for c in paused.task_profile.constraints
        }:
            raise ValueError("replacement references an unknown constraint")
        if len(paused.task_profile.user_messages) >= 20:
            raise ValueError("user-message limit reached")
        self.harness.restore_interactive_session(
            paused.session, checkpoint_hash=paused.manifest["final_event_hash"]
        )
        run = _Run.restore(self, paused)
        run.accept_reply(reply)
        return run.execute()

    def stop_waiting(self, paused: AgentRunResult, reason: str) -> AgentRunResult:
        """Seal a waiting run at a controller budget limit without disclosing a reply."""
        if reason not in {"max_actions", "max_model_turns", "simulator_error"}:
            raise ValueError("unsupported controller stop reason")
        validate_checkpoint(paused)
        run = _Run.restore(self, paused)
        run.events.append("run_resumed", {"question_id": paused.pending_question.question_id})
        run.pending_question = None
        return run.finish("failed" if reason == "simulator_error" else "terminated", reason)


class _Run:
    def __init__(self, owner: AgentOrchestrator, snapshot: SessionSnapshot):
        self.owner = owner
        self.snapshot = snapshot
        self.session_id = snapshot.state.session_id
        self.task = TaskProfile(task=snapshot.public_task)
        self.plan = owner.planner.start(self.task, self.session_id)
        self.memory = MemoryStore(self.session_id)
        self.events = EventStore(self.session_id)
        self.turns: list[AssistantTurn] = []
        self.messages = [
            ChatMessage(role="system", content=owner.skill.system_prompt),
            ChatMessage(
                role="user",
                content=canonical_json(snapshot.public_task.model_dump(mode="json")),
            ),
        ]
        self.actions = 0
        self.searches = 0
        self.repeated: Counter = Counter()
        self.clock_started = time.monotonic()
        self.elapsed_before = 0.0
        self.pending_question = None
        self.coordinator = (
            MultiAgentCoordinator() if owner.config.agent_mode == "multi" else None
        )
        self.manifest = {
            "runtime_version": owner.version,
            "components": {
                "profile": owner.profile.version,
                "planning": owner.planner.version,
                "tools": owner.registry.version,
                "memory": self.memory.version,
                "action": "shopping-action-v1",
                "observation": owner.projector.version,
                "orchestration": owner.version,
                "context": owner.context_builder.version,
                "policy": owner.policy.version,
                "events": self.events.version,
            },
            "config": owner.config.model_dump(mode="json"),
            "max_model_turns": owner.max_model_turns,
            "environment_max_steps": snapshot.state.max_steps,
            "skill_sha256": owner.skill.source_sha256,
            "runtime_instructions_sha256": digest(RUNTIME_INSTRUCTIONS),
            "profile_sha256": digest(owner.profile.model_dump(mode="json")),
            "user_profile_sha256": digest(owner.user_profile.model_dump(mode="json")),
            "provider": owner.provider.name,
            "model": owner.provider.model,
            "provider_endpoint_sha256": digest(
                getattr(owner.provider, "base_url", None)
            ),
            "tool_choice": owner.policy.tool_choice,
            "provider_parameters": _redact(
                {
                    **getattr(owner.provider, "default_parameters", {}),
                    **owner.policy.parameters,
                }
            ),
            "tool_specs": [s.model_dump(mode="json") for s in owner.registry.specs()],
            "harness": getattr(owner.harness, "manifest", {}),
        }
        self.events.append(
            "run_started",
            {
                "manifest": self.manifest,
                "profile": owner.profile.model_dump(mode="json"),
                "user_profile": owner.user_profile.model_dump(mode="json"),
                "task_profile": self.task.model_dump(mode="json"),
                "plan": self.plan.model_dump(mode="json"),
                "memory": self.memory.state.model_dump(mode="json"),
            },
        )

    @classmethod
    def restore(cls, owner, paused):
        run = cls.__new__(cls)
        run.owner = owner
        run.snapshot = paused.session.model_copy(deep=True)
        run.session_id = paused.session_id
        run.task = paused.task_profile.model_copy(deep=True)
        run.plan = paused.plan.model_copy(deep=True)
        run.memory = MemoryStore(run.session_id)
        run.memory.state = paused.memory.model_copy(deep=True)
        run.events = EventStore(run.session_id)
        run.events._events = [e.model_copy(deep=True) for e in paused.events]
        run.repeated = Counter()
        for event in paused.events:
            if event.kind == "observation":
                observation = AgentObservation.model_validate(event.payload)
                run.memory._observations[observation.observation_id] = observation
            elif event.kind == "action":
                payload = event.payload
                try:
                    model = owner.registry.models[payload["tool_name"]].model_validate(
                        payload["arguments"]
                    )
                except (ValueError, KeyError):
                    continue
                run.repeated[
                    action_signature(
                        payload["tool_name"], model.model_dump(mode="json")
                    )
                ] += 1
        run.turns = [t.model_copy(deep=True) for t in paused.turns]
        run.messages = [m.model_copy(deep=True) for m in paused.messages]
        run.actions, run.searches = paused.action_count, paused.search_count
        run.elapsed_before = paused.active_duration_seconds
        run.clock_started = time.monotonic()
        run.pending_question = paused.pending_question
        run.coordinator = (
            MultiAgentCoordinator.restore(paused.events)
            if owner.config.agent_mode == "multi" else None
        )
        run.manifest = {
            k: v for k, v in paused.manifest.items() if k != "final_event_hash"
        }
        return run

    def accept_reply(self, reply):
        question = self.pending_question
        self.events.append("run_resumed", {"question_id": question.question_id})
        message = UserMessage(
            message_id=f"user_{len(self.task.user_messages) + 1:04d}",
            text=reply.text,
            replace_constraint_ids=reply.replace_constraint_ids,
        )
        previous = self.task
        self.task = self.task.model_copy(
            update={
                "revision": self.task.revision + 1,
                "requirements_status": "unparsed",
                "no_constraints_reason": None,
                "constraints": [
                    c
                    for c in self.task.constraints
                    if c.constraint_id not in reply.replace_constraint_ids
                ],
                "user_messages": [*self.task.user_messages, message],
            },
            deep=True,
        )
        self.memory.invalidate_changed_constraints(previous, self.task)
        # Numeric checks were interpreted against the previous user request.
        self.memory.state = self.memory.state.model_copy(
            update={"selection_checks": {}}
        )
        self.plan = self.plan.model_copy(
            update={
                "phase": "verify",
                "revision": self.plan.revision + 1,
                "change_summary": "User replied; reparse requirements before recommending.",
            }
        )
        event = self.events.append(
            "user_message",
            {
                "message": message.model_dump(mode="json"),
                "question_id": question.question_id,
                "task_revision": self.task.revision,
            },
        )
        self.events.append(
            "plan_updated",
            {
                "plan": self.plan.model_dump(mode="json"),
                "task_profile": self.task.model_dump(mode="json"),
            },
            event.event_id,
        )
        self.events.append(
            "memory_updated", self.memory.state.model_dump(mode="json"), event.event_id
        )
        self.messages.append(
            ChatMessage(
                role="user",
                content=canonical_json(
                    {
                        "user_reply": message.model_dump(mode="json"),
                        "question_id": question.question_id,
                    }
                ),
            )
        )
        # A new user revision may justify rereading dynamic prices or repeating a
        # previously useful query. Global actions/searches/tokens/turns stay cumulative.
        self.repeated.clear()
        self.pending_question = None
        if self.coordinator:
            self.coordinator.reset_for_reply()
            self.events.append("agent_handoff", self.coordinator.snapshot(), event.event_id)

    def elapsed(self):
        return self.elapsed_before + time.monotonic() - self.clock_started

    def allowed(self, budget):
        names = self.owner.registry.allowed(self.snapshot.state, budget, self.task)
        if self.owner.config.enforce_constraint_filters and self.task.requirements_status != "parsed":
            names = [n for n in names if n not in {"search_products", "submit_recommendations"}]
        if (
            sum(e.kind == "run_paused" for e in self.events.events)
            >= self.owner.config.max_user_questions
        ):
            names = [n for n in names if n != "ask_user"]
        return self.coordinator.allowed(names, self.task) if self.coordinator else names

    def budget(self) -> BudgetState:
        usage = aggregate_usage(self.turns)
        return BudgetState(
            actions_remaining=max(0, self.owner.config.max_actions - self.actions),
            searches_remaining=max(0, self.owner.config.max_searches - self.searches),
            environment_steps_remaining=max(
                0, self.snapshot.state.max_steps - self.snapshot.state.step_count
            ),
            model_turns_remaining=max(0, self.owner.max_model_turns - len(self.turns)),
            total_tokens=usage.total_tokens if usage else 0,
            usage_complete=all(t.usage is not None for t in self.turns),
        )

    def limit_reason(self):
        config, budget = self.owner.config, self.budget()
        if (
            config.max_duration_seconds is not None
            and self.elapsed() >= config.max_duration_seconds
        ):
            return "time_budget_exhausted"
        if config.max_total_tokens is not None:
            if not budget.usage_complete:
                return "usage_unavailable"
            if budget.total_tokens >= config.max_total_tokens:
                return "token_budget_exhausted"
        if budget.actions_remaining == 0:
            return "max_actions"
        return None

    def execute(self) -> AgentRunResult:
        try:
            return self._loop()
        except Exception as exc:
            return self.finish("failed", "runtime_error", error=exc)

    def _loop(self) -> AgentRunResult:
        owner = self.owner
        for _ in range(max(0, owner.max_model_turns - len(self.turns))):
            if reason := self.limit_reason():
                return self.finish("terminated", reason)
            budget = self.budget()
            allowed = self.allowed(budget)
            try:
                messages, context_metadata = owner.context_builder.build(
                    system_prompt=owner.skill.system_prompt + (
                        self.coordinator.prompt(getattr(owner.harness, "manifest", {}))
                        if self.coordinator else ""
                    ),
                    profile=owner.profile,
                    user=owner.user_profile,
                    task=self.task,
                    plan=self.plan,
                    memory=self.memory,
                    budget=budget,
                    allowed_tools=allowed,
                    config=owner.config,
                    history=self.messages,
                    blocked_action_signatures=set(self.repeated),
                )
            except ContextBudgetError as exc:
                return self.finish("terminated", "context_budget_exhausted", error=exc)
            context_event = self.events.append(
                "context_built",
                {
                    **context_metadata,
                    "messages": [m.model_dump(mode="json") for m in messages],
                    "system_prompt_sha256": digest(messages[0].content),
                    "available_tools": allowed,
                    "agent_role": self.coordinator.role if self.coordinator else "single",
                },
            )
            try:
                turn = owner.policy.decide(
                    messages,
                    [d for d in owner.registry.definitions if d.name in allowed],
                )
            except ProviderError as exc:
                return self.finish("failed", "provider_error", error=exc)
            self.turns.append(turn)
            self.messages.append(turn.as_message())
            turn_event = self.events.append(
                "model_turn",
                {
                    "turn_number": len(self.turns),
                    "agent_role": self.coordinator.role if self.coordinator else "single",
                    "tool_calls": [c.model_dump(mode="json") for c in turn.tool_calls],
                    "usage": turn.usage.model_dump(mode="json") if turn.usage else None,
                },
                context_event.event_id,
            )
            if not turn.tool_calls:
                return self.finish("failed", "model_stopped_without_submission")
            if len(turn.tool_calls) != 1:
                return self.finish("failed", "multiple_tool_calls")
            if reason := self.limit_reason():
                return self.finish("terminated", reason)
            call = turn.tool_calls[0]
            action = AgentAction(
                action_id=f"{self.session_id}:action_{self.actions + 1:04d}",
                session_id=self.session_id,
                sequence=self.actions + 1,
                plan_id=self.plan.plan_id,
                plan_revision=self.plan.revision,
                tool_call_id=call.id,
                tool_name=call.name,
                arguments=call.arguments,
                task_revision=self.task.revision,
                agent_role=self.coordinator.role if self.coordinator else "single",
            )
            action_event = self.events.append(
                "action", action.model_dump(mode="json"), turn_event.event_id
            )
            budget_before = self.budget()
            self.actions += 1
            previous_plan = self.plan
            observation = self.perform(action, call, budget_before)
            if self.coordinator:
                self.coordinator.after_action(call.name, call.arguments, observation, self.task)
                self.events.append(
                    "agent_handoff", self.coordinator.snapshot(), action_event.event_id
                )
                observation = observation.model_copy(update={
                    "allowed_next_tools": [] if observation.terminated or self.pending_question
                    else self.allowed(self.budget())
                })
            obs_event = self.events.append(
                "observation",
                observation.model_dump(mode="json"),
                action_event.event_id,
            )
            self.memory.observe(observation)
            if not observation.terminated and self.pending_question is None:
                self.plan = owner.planner.advance(
                    self.plan,
                    self.task,
                    self.memory.state,
                    self.budget().searches_remaining,
                )
            if self.plan != previous_plan:
                self.events.append(
                    "plan_updated",
                    {
                        "plan": self.plan.model_dump(mode="json"),
                        "task_profile": self.task.model_dump(mode="json"),
                    },
                    obs_event.event_id,
                )
            self.events.append(
                "memory_updated",
                self.memory.state.model_dump(mode="json"),
                obs_event.event_id,
            )
            self.messages.append(
                ChatMessage(
                    role="tool",
                    tool_call_id=call.id,
                    name=call.name,
                    content=canonical_json(observation.model_dump(mode="json")),
                )
            )
            if self.pending_question is not None:
                return self.finish("waiting_for_user", "user_input_required")
            if observation.terminated:
                return self.finish(
                    (
                        "completed"
                        if observation.terminal_reason == "submitted"
                        else "terminated"
                    ),
                    observation.terminal_reason,
                )
        if reason := self.limit_reason():
            return self.finish("terminated", reason)
        return self.finish("failed", "max_model_turns")

    def perform(self, action, call, budget_before) -> AgentObservation:
        owner = self.owner
        try:
            if self.coordinator:
                if call.name not in ROLE_TOOLS[self.coordinator.role]:
                    raise ValueError("role_not_authorized")
                if call.name not in self.allowed(budget_before):
                    raise ValueError("role_tool_not_available")
                if call.name == "search_products" and "method" not in call.arguments:
                    raise ValueError("search_method_must_be_explicit")
            if call.name == "ask_user" and call.name not in self.allowed(budget_before):
                raise ValueError("user_question_limit_reached")
            model = owner.registry.validate(call, self.snapshot.state, budget_before)
            if owner.config.enforce_constraint_filters and call.name in {"search_products", "submit_recommendations"} and self.task.requirements_status != "parsed":
                raise ValueError("update_plan_required_after_user_reply")
            signature = action_signature(call.name, model.model_dump(mode="json"))
            self.repeated[signature] += 1
            if self.repeated[signature] > owner.config.max_repeated_actions:
                raise ValueError("repeated_action")
            if call.name == "ask_user":
                if not set(model.constraint_ids) <= {
                    c.constraint_id for c in self.task.constraints
                }:
                    raise ValueError("question references an unknown constraint")
                self.pending_question = PendingQuestion(
                    **model.model_dump(), question_id=f"{action.action_id}:question"
                )
                data = {
                    "pending_question": self.pending_question.model_dump(mode="json")
                }
            elif call.name == "update_plan":
                assert isinstance(model, PlanUpdate)
                if owner.config.enforce_constraint_filters:
                    from ..constraints import infer_literal_predicate
                    model = model.model_copy(update={"constraints": [
                        c.model_copy(update={"predicate": infer_literal_predicate(c.source_quote)})
                        if c.priority == "hard" and c.predicate is None else c
                        for c in model.constraints]})
                plan, task = owner.planner.update(
                    self.plan,
                    model,
                    self.task,
                    owner.user_profile,
                    self.memory.observation_ids,
                )
                self.memory.invalidate_changed_constraints(self.task, task)
                self.plan, self.task = plan, task
                data = {
                    "plan": plan.model_dump(mode="json"),
                    "task_profile": task.model_dump(mode="json"),
                }
            elif call.name == "record_evidence":
                assert isinstance(model, EvidenceAssessment)
                self.memory.assess(model, self.task)
                data = {
                    "assessment": model.model_dump(mode="json"),
                    "provenance_checked": True,
                    "semantic_verification": "model_assessment",
                }
            elif call.name == "record_evidence_batch":
                assert isinstance(model, EvidenceBatch)
                self.memory.assess_batch(model.assessments, self.task)
                data = {
                    "assessments": [
                        a.model_dump(mode="json") for a in model.assessments
                    ],
                    "provenance_checked": True,
                    "semantic_verification": "model_assessment",
                }
            elif call.name == "read_evidence":
                assert isinstance(model, ReadEvidence)
                data = self.memory.read_evidence(model)
            elif call.name == "compare_products":
                data = compare_products(model, self.task, self.memory.state)
            elif call.name == "verify_selection":
                data = verify_selection(model, self.task, self.memory.state)
            elif call.name == "review_progress":
                assert isinstance(model, ReviewProgress) and self.coordinator is not None
                self.coordinator.validate_review(model, self.task, self.memory.state)
                data = model.model_dump(mode="json")
            elif call.name == "submit_recommendations":
                if owner.config.progressive_disclosure and (not self.task.user_messages or self.task.uncertain_requirements):
                    raise ValueError("clarify_public_information_gaps_before_recommending")
                if self.coordinator:
                    self.coordinator.validate_submission(model, self.task)
                check_submission(model, self.task, self.memory.state, enforce_predicates=owner.config.enforce_constraint_filters)
        except SubmissionNotReady as exc:
            return self.internal_observation(
                action,
                status="rejected",
                error_code="submission_not_ready",
                message=str(exc),
                data={"issues": exc.issues},
            )
        except ValueError as exc:
            return self.internal_observation(
                action,
                status="rejected",
                error_code="action_rejected",
                message=str(exc),
            )
        if call.name in owner.registry.environment_names:
            if call.name == "search_products":
                self.searches += 1
            arguments = model.model_dump(mode="json")
            if call.name == "search_products" and owner.config.enforce_constraint_filters:
                from ..schemas import SearchProductsAction
                from ..constraints import predicate_key
                by_key = {predicate_key(p): p for p in model.filters.predicates}
                for constraint in self.task.constraints:
                    if constraint.priority == "hard" and constraint.predicate is not None:
                        by_key[predicate_key(constraint.predicate)] = constraint.predicate
                arguments["filters"]["predicates"] = [p.model_dump(mode="json") for p in by_key.values()]
                arguments = SearchProductsAction.model_validate(arguments).model_dump(mode="json")
            if call.name == "submit_recommendations":
                arguments = {"recommendations": arguments["recommendations"]}
            step = owner.harness.dispatch(self.session_id, call.name, arguments)
            self.snapshot = owner.harness.get_session(self.session_id)
            # Adapter/schema errors are runtime failures, not recoverable model mistakes.
            observation = owner.projector.project(
                action,
                step,
                self.snapshot.state,
                self.budget(),
                owner.registry,
                self.task,
            )
            if call.name == "search_products" and observation.status == "success":
                self.memory.record_query(
                    model.query, arguments, observation.projection.next_offset
                )
            return observation
        return self.internal_observation(action, data=data)

    def internal_observation(
        self, action, *, status="success", data=None, error_code=None, message=None
    ):
        budget = self.budget()
        return AgentObservation(
            observation_id=f"{action.action_id}:observation",
            action_id=action.action_id,
            tool_name=action.tool_name,
            status=status,
            data=data,
            error_code=error_code,
            message=message,
            budget=budget,
            allowed_next_tools=[] if self.pending_question else self.allowed(budget),
        )

    def finish(self, status, reason, *, error=None) -> AgentRunResult:
        self.snapshot = self.owner.harness.get_session(self.session_id)
        duration = self.elapsed()
        self.plan = self.plan.model_copy(
            update={
                "phase": (
                    "waiting_for_user"
                    if status == "waiting_for_user"
                    else "completed" if reason == "submitted" else "stopped"
                ),
                "revision": self.plan.revision + 1,
            }
        )
        self.events.append(
            "plan_updated",
            {
                "plan": self.plan.model_dump(mode="json"),
                "task_profile": self.task.model_dump(mode="json"),
            },
        )
        outcome = {
            "status": status,
            "terminal_reason": reason,
            "action_count": self.actions,
            "search_count": self.searches,
            "budget": self.budget().model_dump(mode="json"),
            "error_type": type(error).__name__ if error else None,
            "error_message": str(error) if error else None,
            "active_duration_seconds": duration,
            "pending_question": (
                self.pending_question.model_dump(mode="json")
                if self.pending_question
                else None
            ),
        }
        if status == "waiting_for_user":
            outcome["checkpoint_sha256"] = digest(
                {
                    "task": self.task.model_dump(mode="json"),
                    "plan": self.plan.model_dump(mode="json"),
                    "memory": self.memory.state.model_dump(mode="json"),
                    "session": self.snapshot.model_dump(mode="json"),
                    "messages": [m.model_dump(mode="json") for m in self.messages],
                    "turns": [t.model_dump(mode="json") for t in self.turns],
                }
            )
        last = self.events.append(
            "run_paused" if status == "waiting_for_user" else "run_finished",
            outcome,
        )
        manifest = {**self.manifest, "final_event_hash": last.event_hash}
        return AgentRunResult(
            status=status,
            terminal_reason=reason,
            task_id=self.task.task.task_id,
            session_id=self.session_id,
            provider=self.owner.provider.name,
            model=self.owner.provider.model,
            skill_name=self.owner.skill.name,
            skill_sha256=self.owner.skill.source_sha256,
            model_turn_count=len(self.turns),
            tool_call_count=len(self.snapshot.history),
            action_count=self.actions,
            search_count=self.searches,
            usage=aggregate_usage(self.turns),
            messages=self.messages,
            turns=self.turns,
            session=self.snapshot,
            manifest=manifest,
            profile=self.owner.profile,
            user_profile=self.owner.user_profile,
            task_profile=self.task,
            plan=self.plan,
            memory=self.memory.state,
            events=self.events.events,
            error_type=type(error).__name__ if error else None,
            error_message=str(error) if error else None,
            pending_question=self.pending_question,
            active_duration_seconds=duration,
        )


def validate_checkpoint(result: AgentRunResult) -> None:
    if (
        result.runner_version != "shopping-agent-runtime-v5"
        or result.status != "waiting_for_user"
        or result.pending_question is None
    ):
        raise ValueError("only a v5 waiting-for-user result can be resumed")
    state = replay_events(result.events)
    outcome = state["outcome"]
    if result.session.state.status != "active" or result.session.evaluation is not None:
        raise ValueError("checkpoint must contain an active unlabelled session")
    if (
        result.task_profile is None
        or result.plan is None
        or result.memory is None
        or result.session_id != result.session.state.session_id
        or result.task_id != result.session.public_task.task_id
        or result.task_profile.task != result.session.public_task
    ):
        raise ValueError("checkpoint task/session boundary mismatch")
    payload = {
        "task": result.task_profile.model_dump(mode="json"),
        "plan": result.plan.model_dump(mode="json"),
        "memory": result.memory.model_dump(mode="json"),
        "session": result.session.model_dump(mode="json"),
        "messages": [m.model_dump(mode="json") for m in result.messages],
        "turns": [t.model_dump(mode="json") for t in result.turns],
    }
    if (
        digest(payload) != outcome.get("checkpoint_sha256")
        or result.manifest
        != {**state["manifest"], "final_event_hash": state["final_event_hash"]}
        or result.profile.model_dump(mode="json") != state["profile"]
        or result.user_profile.model_dump(mode="json") != state["user_profile"]
        or result.action_count != outcome["action_count"]
        or result.search_count != outcome["search_count"]
        or result.model_turn_count != len(result.turns)
        or result.tool_call_count != len(result.session.history)
        or result.usage != aggregate_usage(result.turns)
        or result.active_duration_seconds != outcome["active_duration_seconds"]
        or result.pending_question.model_dump(mode="json")
        != outcome["pending_question"]
        or result.terminal_reason != outcome["terminal_reason"]
        or result.status != outcome["status"]
        or result.provider != result.manifest["provider"]
        or result.model != result.manifest["model"]
        or result.skill_sha256 != result.manifest["skill_sha256"]
        or state["plan"] != payload["plan"]
        or state["memory"] != payload["memory"]
        or state["task_profile"] != payload["task"]
    ):
        raise ValueError("checkpoint differs from its recorded event state")


def aggregate_usage(turns: list[AssistantTurn]) -> TokenUsage | None:
    usages = [turn.usage for turn in turns if turn.usage is not None]
    if not usages:
        return None

    def optional_sum(field):
        values = [getattr(u, field) for u in usages if getattr(u, field) is not None]
        return sum(values) if values else None

    return TokenUsage(
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        total_tokens=sum(u.total_tokens for u in usages),
        cached_input_tokens=optional_sum("cached_input_tokens"),
        cache_miss_input_tokens=optional_sum("cache_miss_input_tokens"),
        reasoning_tokens=optional_sum("reasoning_tokens"),
    )


def _redact(value):
    if isinstance(value, dict):
        return {
            k: (
                "***"
                if (
                    any(
                        s in k.lower()
                        for s in ("key", "password", "secret", "authorization")
                    )
                    or k.lower() == "token"
                    or k.lower().endswith("_token")
                )
                else _redact(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value
