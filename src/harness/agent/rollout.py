"""Batch rollout orchestration, aggregation, and durable result storage."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..paths import new_run_id
from ..providers import TokenUsage
from ..schemas import SessionSnapshot, TaskSplit
from .runner import AgentRunResult
from .artifacts import write_run_artifacts


ROLLOUT_RUNNER_VERSION = "shopping-rollout-runner-v3"
_ROLLOUT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_TASK_ID_PATTERN = re.compile(r"^task_[0-9a-f]{24}$")


class RolloutError(RuntimeError):
    """The batch rollout configuration or persistence step is invalid."""


@runtime_checkable
class RolloutHarness(Protocol):
    """Harness operations required above the single-task Agent loop."""

    def list_task_ids(self, split: TaskSplit) -> list[str]: ...

    def sample_task_ids(
        self,
        split: TaskSplit,
        count: int,
        *,
        seed: int | str,
    ) -> list[str]: ...

    def release_session(
        self,
        session_id: str,
        *,
        force: bool = False,
    ) -> SessionSnapshot: ...


@runtime_checkable
class TaskRunner(Protocol):
    """One isolated runner created for each rollout task."""

    def run(
        self,
        task_id: str,
        *,
        max_steps: int | None = None,
        session_id: str | None = None,
    ) -> AgentRunResult: ...


class RunConsumption(BaseModel):
    """Explicit resource accounting for one task run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    duration_seconds: float = Field(ge=0.0)
    model_turn_count: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    action_count: int = Field(default=0, ge=0)
    search_count: int = Field(default=0, ge=0)
    usage: TokenUsage | None = None


class RolloutTaskRecord(BaseModel):
    """One task outcome, including unexpected pre-result failures."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    task_id: str = Field(pattern=r"^task_[0-9a-f]{24}$")
    session_id: str = Field(min_length=1)
    duration_seconds: float = Field(ge=0.0)
    consumption: RunConsumption
    result: AgentRunResult | None = None
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def _require_result_or_error(self) -> RolloutTaskRecord:
        if self.result is None:
            if not self.error_type or not self.error_message:
                raise ValueError("failed task records require error details")
        elif self.error_type is not None or self.error_message is not None:
            raise ValueError("successful task records cannot include batch errors")
        if self.consumption.duration_seconds != self.duration_seconds:
            raise ValueError("consumption duration must match record duration")
        if self.result is None:
            if self.consumption.model_turn_count or self.consumption.tool_call_count:
                raise ValueError("pre-result failures cannot report Agent activity")
            if self.consumption.usage is not None:
                raise ValueError("pre-result failures cannot report token usage")
        else:
            if self.consumption.model_turn_count != self.result.model_turn_count:
                raise ValueError("consumption model turns must match result")
            if self.consumption.tool_call_count != self.result.tool_call_count:
                raise ValueError("consumption tool calls must match result")
            if self.consumption.action_count != self.result.action_count:
                raise ValueError("consumption actions must match result")
            if self.consumption.search_count != self.result.search_count:
                raise ValueError("consumption searches must match result")
            if self.consumption.usage != self.result.usage:
                raise ValueError("consumption token usage must match result")
        return self


class RolloutSummary(BaseModel):
    """Aggregate quality, runtime, and token metrics for one rollout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_count: int = Field(ge=1)
    result_count: int = Field(ge=0)
    unexpected_error_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    terminated_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    evaluated_count: int = Field(ge=0)
    reward_valid_count: int = Field(ge=0)
    strict_success_count: int = Field(ge=0)
    completion_rate: float = Field(ge=0.0, le=1.0)
    strict_success_rate: float = Field(ge=0.0, le=1.0)
    reward_coverage: float = Field(ge=0.0, le=1.0)
    average_reward: float = Field(ge=0.0, le=1.0)
    average_valid_reward: float | None = Field(default=None, ge=0.0, le=1.0)
    total_model_turns: int = Field(ge=0)
    total_tool_calls: int = Field(ge=0)
    total_actions: int = Field(default=0, ge=0)
    total_searches: int = Field(default=0, ge=0)
    usage: TokenUsage | None = None
    terminal_reason_counts: dict[str, int]
    total_duration_seconds: float = Field(ge=0.0)


class RolloutReport(BaseModel):
    """Complete in-memory representation of a batch rollout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rollout_version: str = ROLLOUT_RUNNER_VERSION
    rollout_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0.0)
    split: TaskSplit | None = None
    seed: int | str | None = None
    task_ids: list[str]
    workers: int = Field(ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    output_dir: Path | None = None
    summary: RolloutSummary
    records: list[RolloutTaskRecord]


class RolloutRunner:
    """Execute independent Agent runs and aggregate deterministic metrics."""

    def __init__(
        self,
        *,
        harness: RolloutHarness,
        runner_factory: Callable[[], TaskRunner],
    ) -> None:
        if not isinstance(harness, RolloutHarness):
            raise TypeError("harness must implement RolloutHarness")
        if not callable(runner_factory):
            raise TypeError("runner_factory must be callable")
        self._harness = harness
        self._runner_factory = runner_factory

    def run_split(
        self,
        split: TaskSplit,
        count: int | None = None,
        *,
        seed: int | str = 42,
        workers: int = 1,
        max_steps: int | None = None,
        output_root: str | Path | None = None,
        rollout_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RolloutReport:
        """Run the full selection, or a deterministic sample when count is given."""
        if count is None:
            task_ids = self._harness.list_task_ids(split)
        else:
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError("count must be an integer")
            if count < 1:
                raise ValueError("count must be positive")
            task_ids = self._harness.sample_task_ids(split, count, seed=seed)
        return self.run_tasks(
            task_ids,
            workers=workers,
            max_steps=max_steps,
            output_root=output_root,
            rollout_id=rollout_id,
            split=split,
            seed=seed if count is not None else None,
            metadata=metadata,
        )

    def run_tasks(
        self,
        task_ids: Sequence[str],
        *,
        workers: int = 1,
        max_steps: int | None = None,
        output_root: str | Path | None = None,
        rollout_id: str | None = None,
        split: TaskSplit | None = None,
        seed: int | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RolloutReport:
        """Run explicit task IDs, continuing after individual task failures."""
        normalized_task_ids = self._validate_task_ids(task_ids)
        self._validate_workers(workers)
        self._validate_max_steps(max_steps)
        normalized_metadata = self._validate_metadata(metadata)
        resolved_rollout_id = _resolve_rollout_id(rollout_id)
        output_dir = _prepare_output_dir(output_root, resolved_rollout_id)
        started_at = datetime.now(timezone.utc)
        started_clock = time.perf_counter()

        records: list[RolloutTaskRecord | None] = [None for _ in normalized_task_ids]
        if workers == 1:
            for index, task_id in enumerate(normalized_task_ids):
                records[index] = self._run_one(
                    index,
                    task_id,
                    resolved_rollout_id,
                    max_steps,
                )
        else:
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="shopping-rollout",
            ) as executor:
                futures = {
                    executor.submit(
                        self._run_one,
                        index,
                        task_id,
                        resolved_rollout_id,
                        max_steps,
                    ): index
                    for index, task_id in enumerate(normalized_task_ids)
                }
                for future in as_completed(futures):
                    records[futures[future]] = future.result()

        completed_records = [record for record in records if record is not None]
        if len(completed_records) != len(normalized_task_ids):
            raise AssertionError("rollout completed without recording every task")
        finished_at = datetime.now(timezone.utc)
        report = RolloutReport(
            rollout_id=resolved_rollout_id,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=time.perf_counter() - started_clock,
            split=split,
            seed=seed,
            task_ids=normalized_task_ids,
            workers=workers,
            max_steps=max_steps,
            metadata=normalized_metadata,
            output_dir=output_dir,
            summary=_summarize(completed_records),
            records=completed_records,
        )
        if output_dir is not None:
            _persist_report(report, output_dir)
        return report

    def _run_one(
        self,
        index: int,
        task_id: str,
        rollout_id: str,
        max_steps: int | None,
    ) -> RolloutTaskRecord:
        session_id = f"{rollout_id}_{index + 1:06d}"
        started = time.perf_counter()
        result: AgentRunResult | None = None
        try:
            runner = self._runner_factory()
            if not isinstance(runner, TaskRunner):
                raise TypeError("runner_factory must return a TaskRunner")
            result = runner.run(
                task_id,
                max_steps=max_steps,
                session_id=session_id,
            )
            if result.status == "waiting_for_user":
                raise RolloutError("batch rollout does not support interactive pauses; use AgentRunner.run_query")
            if result.task_id != task_id or result.session_id != session_id:
                raise RolloutError(
                    "TaskRunner returned a result for a different task or session"
                )
            duration_seconds = time.perf_counter() - started
            return RolloutTaskRecord(
                index=index,
                task_id=task_id,
                session_id=session_id,
                duration_seconds=duration_seconds,
                consumption=RunConsumption(
                    duration_seconds=duration_seconds,
                    model_turn_count=result.model_turn_count,
                    tool_call_count=result.tool_call_count,
                    action_count=result.action_count,
                    search_count=result.search_count,
                    usage=result.usage,
                ),
                result=result,
            )
        except Exception as exc:
            duration_seconds = time.perf_counter() - started
            return RolloutTaskRecord(
                index=index,
                task_id=task_id,
                session_id=session_id,
                duration_seconds=duration_seconds,
                consumption=RunConsumption(
                    duration_seconds=duration_seconds,
                    model_turn_count=0,
                    tool_call_count=0,
                ),
                error_type=type(exc).__name__,
                error_message=str(exc) or type(exc).__name__,
            )
        finally:
            if result is not None:
                try:
                    self._harness.release_session(
                        session_id,
                        force=result.session.state.status == "active",
                    )
                except Exception:
                    pass
            else:
                try:
                    self._harness.release_session(session_id, force=True)
                except Exception:
                    pass

    @staticmethod
    def _validate_task_ids(task_ids: Sequence[str]) -> list[str]:
        if not isinstance(task_ids, Sequence) or isinstance(
            task_ids,
            (str, bytes),
        ):
            raise TypeError("task_ids must be a sequence of strings")
        normalized = []
        for index, task_id in enumerate(task_ids):
            if not isinstance(task_id, str):
                raise TypeError(f"task_ids[{index}] must be a string")
            normalized_task_id = task_id.strip()
            if not _TASK_ID_PATTERN.fullmatch(normalized_task_id):
                raise ValueError(f"task_ids[{index}] is not a valid public task ID")
            normalized.append(normalized_task_id)
        if not normalized:
            raise ValueError("task_ids must not be empty")
        return normalized

    @staticmethod
    def _validate_workers(workers: int) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int):
            raise TypeError("workers must be an integer")
        if not 1 <= workers <= 64:
            raise ValueError("workers must be between 1 and 64")

    @staticmethod
    def _validate_max_steps(max_steps: int | None) -> None:
        if max_steps is None:
            return
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise TypeError("max_steps must be an integer or None")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")

    @staticmethod
    def _validate_metadata(
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        normalized = dict(metadata or {})
        try:
            json.dumps(normalized, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise RolloutError("metadata must be JSON serializable") from exc
        return normalized


def _resolve_rollout_id(rollout_id: str | None) -> str:
    if rollout_id is None:
        return new_run_id("rollout")
    if not isinstance(rollout_id, str) or not _ROLLOUT_ID_PATTERN.fullmatch(
        rollout_id,
    ):
        raise RolloutError(
            "rollout_id must contain 1-80 letters, digits, dots, underscores, "
            "or hyphens and must start with a letter or digit"
        )
    return rollout_id


def _prepare_output_dir(
    output_root: str | Path | None,
    rollout_id: str,
) -> Path | None:
    if output_root is None:
        return None
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    output_dir = root / rollout_id
    try:
        output_dir.mkdir()
    except FileExistsError as exc:
        raise RolloutError(f"rollout output already exists: {output_dir}") from exc
    return output_dir


def _summarize(records: list[RolloutTaskRecord]) -> RolloutSummary:
    results = [record.result for record in records if record.result is not None]
    evaluations = [
        result.session.evaluation
        for result in results
        if result.session.evaluation is not None
    ]
    valid_evaluations = [item for item in evaluations if item.reward_valid]
    requested_count = len(records)
    strict_success_count = sum(item.strict_success for item in evaluations)
    terminal_reasons = Counter(result.terminal_reason for result in results)
    unexpected_error_count = requested_count - len(results)
    if unexpected_error_count:
        terminal_reasons["unexpected_error"] = unexpected_error_count
    total_reward = sum(item.reward for item in valid_evaluations)
    return RolloutSummary(
        requested_count=requested_count,
        result_count=len(results),
        unexpected_error_count=unexpected_error_count,
        completed_count=sum(result.status == "completed" for result in results),
        terminated_count=sum(result.status == "terminated" for result in results),
        failed_count=sum(result.status == "failed" for result in results),
        evaluated_count=len(evaluations),
        reward_valid_count=len(valid_evaluations),
        strict_success_count=strict_success_count,
        completion_rate=(
            sum(result.status == "completed" for result in results) / requested_count
        ),
        strict_success_rate=strict_success_count / requested_count,
        reward_coverage=len(valid_evaluations) / requested_count,
        average_reward=total_reward / requested_count,
        average_valid_reward=(
            total_reward / len(valid_evaluations) if valid_evaluations else None
        ),
        total_model_turns=sum(result.model_turn_count for result in results),
        total_tool_calls=sum(result.tool_call_count for result in results),
        total_actions=sum(result.action_count for result in results),
        total_searches=sum(result.search_count for result in results),
        usage=_aggregate_usage(results),
        terminal_reason_counts=dict(sorted(terminal_reasons.items())),
        total_duration_seconds=sum(record.duration_seconds for record in records),
    )


def _aggregate_usage(results: list[AgentRunResult]) -> TokenUsage | None:
    usages = [result.usage for result in results if result.usage is not None]
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


def _persist_report(report: RolloutReport, output_dir: Path) -> None:
    for record in report.records:
        if record.result is not None:
            write_run_artifacts(
                record.result, output_dir / "tasks" / f"{record.index + 1:06d}"
            )
    records_text = "".join(f"{record.model_dump_json()}\n" for record in report.records)
    summary_payload = report.model_dump(
        mode="json",
        exclude={"records"},
    )
    consumption_text = "".join(
        json.dumps(
            {
                "index": record.index,
                "task_id": record.task_id,
                "session_id": record.session_id,
                "status": (
                    record.result.status
                    if record.result is not None
                    else "unexpected_error"
                ),
                "terminal_reason": (
                    record.result.terminal_reason
                    if record.result is not None
                    else "unexpected_error"
                ),
                "consumption": record.consumption.model_dump(mode="json"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for record in report.records
    )
    _atomic_write(output_dir / "runs.jsonl", records_text)
    _atomic_write(output_dir / "consumption.jsonl", consumption_text)
    _atomic_write(
        output_dir / "summary.json",
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
    )


def _atomic_write(path: Path, text: str) -> None:
    if path.exists():
        raise RolloutError(f"refusing to overwrite rollout artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
