"""Bounded Provider -> Harness tool loop for the project shopping Agent."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ..providers import (
    AssistantTurn,
    ChatMessage,
    ModelProvider,
    ProviderError,
    TokenUsage,
    ToolChoice,
    canonical_json,
)
from ..schemas import SessionSnapshot, SessionStep
from .skill import AgentSkill, load_agent_skill
from .tools import shopping_tool_definitions


AGENT_RUNNER_VERSION = "shopping-agent-runner-v1"
RunStatus = Literal["completed", "terminated", "failed"]
RunTerminalReason = Literal[
    "submitted",
    "max_steps",
    "infrastructure_error",
    "provider_error",
    "model_stopped_without_submission",
    "multiple_tool_calls",
    "max_model_turns",
]


@runtime_checkable
class HarnessRuntime(Protocol):
    """Public Harness operations required by AgentRunner."""

    def create_session(
        self,
        task_id: str,
        *,
        max_steps: int | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot: ...

    def dispatch(
        self,
        session_id: str,
        tool_name: str,
        arguments: Mapping[str, Any] | object,
    ) -> SessionStep: ...

    def get_session(self, session_id: str) -> SessionSnapshot: ...


class AgentRunResult(BaseModel):
    """Serializable audit artifact for one complete or failed Agent run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runner_version: str = AGENT_RUNNER_VERSION
    status: RunStatus
    terminal_reason: RunTerminalReason
    task_id: str
    session_id: str
    provider: str
    model: str
    skill_name: str
    skill_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_turn_count: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    usage: TokenUsage | None = None
    messages: list[ChatMessage]
    turns: list[AssistantTurn]
    session: SessionSnapshot
    error_type: str | None = None
    error_message: str | None = None


class AgentRunner:
    """Run one shopping task using a model Provider and managed Harness.

    The runner owns orchestration only. Harness remains authoritative for tool
    validation, public observations, step limits, and deterministic evaluation.
    """

    def __init__(
        self,
        *,
        provider: ModelProvider,
        harness: HarnessRuntime,
        skill: AgentSkill | None = None,
        skill_dir: str | Path | None = None,
        max_model_turns: int = 40,
        tool_choice: ToolChoice = "required",
        provider_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        if skill is not None and skill_dir is not None:
            raise ValueError("pass skill or skill_dir, not both")
        if isinstance(max_model_turns, bool) or not isinstance(
            max_model_turns,
            int,
        ):
            raise TypeError("max_model_turns must be an integer")
        if max_model_turns < 1:
            raise ValueError("max_model_turns must be positive")
        if not isinstance(provider, ModelProvider):
            raise TypeError("provider must implement ModelProvider")
        if not isinstance(harness, HarnessRuntime):
            raise TypeError("harness must implement HarnessRuntime")

        self._provider = provider
        self._harness = harness
        self._skill = skill or load_agent_skill(skill_dir)
        self._max_model_turns = max_model_turns
        self._tool_choice = tool_choice
        self._provider_parameters = dict(provider_parameters or {})
        self._tools = shopping_tool_definitions()

    @property
    def skill(self) -> AgentSkill:
        return self._skill

    def run(
        self,
        task_id: str,
        *,
        max_steps: int | None = None,
        session_id: str | None = None,
    ) -> AgentRunResult:
        """Create a session and drive it until submission or a bounded stop."""
        started = self._harness.create_session(
            task_id,
            max_steps=max_steps,
            session_id=session_id,
        )
        resolved_session_id = started.state.session_id
        messages = self._initial_messages(started)
        turns: list[AssistantTurn] = []

        for _ in range(self._max_model_turns):
            try:
                turn = self._provider.complete(
                    tuple(messages),
                    tools=self._tools,
                    tool_choice=self._tool_choice,
                    parameters=self._provider_parameters,
                )
            except ProviderError as exc:
                return self._result(
                    status="failed",
                    terminal_reason="provider_error",
                    session_id=resolved_session_id,
                    messages=messages,
                    turns=turns,
                    error=exc,
                )

            turns.append(turn)
            messages.append(turn.as_message())
            if not turn.tool_calls:
                return self._result(
                    status="failed",
                    terminal_reason="model_stopped_without_submission",
                    session_id=resolved_session_id,
                    messages=messages,
                    turns=turns,
                    error_message=(
                        "model returned prose without the required terminal "
                        "Harness tool call"
                    ),
                )
            if len(turn.tool_calls) != 1:
                return self._result(
                    status="failed",
                    terminal_reason="multiple_tool_calls",
                    session_id=resolved_session_id,
                    messages=messages,
                    turns=turns,
                    error_message=(
                        "model emitted multiple tool calls in one assistant turn"
                    ),
                )

            call = turn.tool_calls[0]
            step = self._harness.dispatch(
                resolved_session_id,
                call.name,
                call.arguments,
            )
            snapshot = self._harness.get_session(resolved_session_id)
            messages.append(
                ChatMessage(
                    role="tool",
                    tool_call_id=call.id,
                    name=call.name,
                    content=canonical_json(
                        self._tool_observation(step, snapshot)
                    ),
                )
            )
            if step.terminated:
                reason = step.terminal_reason
                if reason not in (
                    "submitted",
                    "max_steps",
                    "infrastructure_error",
                ):
                    raise RuntimeError(
                        f"unsupported Harness terminal reason: {reason}"
                    )
                status: RunStatus = (
                    "completed" if reason == "submitted" else "terminated"
                )
                return self._result(
                    status=status,
                    terminal_reason=reason,
                    session_id=resolved_session_id,
                    messages=messages,
                    turns=turns,
                )

        return self._result(
            status="failed",
            terminal_reason="max_model_turns",
            session_id=resolved_session_id,
            messages=messages,
            turns=turns,
            error_message="Agent Runner exhausted max_model_turns",
        )

    def _initial_messages(self, started: SessionSnapshot) -> list[ChatMessage]:
        task_payload = canonical_json(
            {
                "task_id": started.public_task.task_id,
                "query": started.public_task.query,
                "max_steps": started.state.max_steps,
            }
        )
        return [
            ChatMessage(role="system", content=self._skill.system_prompt),
            ChatMessage(
                role="user",
                content=(
                    "Solve this public Shopping Harness task. Use exactly one "
                    "tool call per turn and finish with submit_recommendations.\n"
                    f"{task_payload}"
                ),
            ),
        ]

    @staticmethod
    def _tool_observation(
        step: SessionStep,
        snapshot: SessionSnapshot,
    ) -> dict[str, Any]:
        remaining_steps = max(
            snapshot.state.max_steps - snapshot.state.step_count,
            0,
        )
        return {
            "step": step.model_dump(mode="json"),
            "session": {
                "status": snapshot.state.status,
                "step_count": snapshot.state.step_count,
                "max_steps": snapshot.state.max_steps,
                "remaining_steps": remaining_steps,
                "terminal_reason": snapshot.state.terminal_reason,
            },
        }

    def _result(
        self,
        *,
        status: RunStatus,
        terminal_reason: RunTerminalReason,
        session_id: str,
        messages: list[ChatMessage],
        turns: list[AssistantTurn],
        error: Exception | None = None,
        error_message: str | None = None,
    ) -> AgentRunResult:
        snapshot = self._harness.get_session(session_id)
        return AgentRunResult(
            status=status,
            terminal_reason=terminal_reason,
            task_id=snapshot.public_task.task_id,
            session_id=session_id,
            provider=self._provider.name,
            model=self._provider.model,
            skill_name=self._skill.name,
            skill_sha256=self._skill.source_sha256,
            model_turn_count=len(turns),
            tool_call_count=len(snapshot.history),
            usage=_aggregate_usage(turns),
            messages=list(messages),
            turns=list(turns),
            session=snapshot,
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else error_message,
        )


def _aggregate_usage(turns: list[AssistantTurn]) -> TokenUsage | None:
    usages = [turn.usage for turn in turns if turn.usage is not None]
    if not usages:
        return None

    def optional_sum(field: str) -> int | None:
        values = [getattr(usage, field) for usage in usages]
        present = [value for value in values if value is not None]
        return sum(present) if present else None

    return TokenUsage(
        input_tokens=sum(usage.input_tokens for usage in usages),
        output_tokens=sum(usage.output_tokens for usage in usages),
        total_tokens=sum(usage.total_tokens for usage in usages),
        cached_input_tokens=optional_sum("cached_input_tokens"),
        cache_miss_input_tokens=optional_sum("cache_miss_input_tokens"),
        reasoning_tokens=optional_sum("reasoning_tokens"),
    )
