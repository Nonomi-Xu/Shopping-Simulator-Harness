"""Fixed retrieval-then-decision pipeline with no iterative Agent loop."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .catalog import BM25Catalog
from .evaluator import evaluate_decision
from .provider import CandidateSelector
from .tasks import TaskStore
from .types import TaskResult


PIPELINE_VERSION = "bm25-top5-agent-v1"


class BM25AgentPipeline:
    """BM25 Top 5 -> one constrained Agent decision -> deterministic score."""

    def __init__(
        self,
        *,
        tasks: TaskStore,
        catalog: BM25Catalog,
        selector: CandidateSelector,
    ) -> None:
        if not isinstance(selector, CandidateSelector):
            raise TypeError("selector must implement CandidateSelector")
        self.tasks = tasks
        self.catalog = catalog
        self.selector = selector

    def run(self, task_id: str) -> TaskResult:
        task = self.tasks.get_public_task(task_id)
        candidates = self.catalog.search_top5(task.query)
        if not candidates:
            return TaskResult(
                pipeline_version=PIPELINE_VERSION,
                status="failed",
                task=task,
                candidates=[],
                selector=None,
                evaluation=None,
                error_type="NoCandidatesError",
                error_message="BM25 returned no candidates",
            )
        try:
            selector_result = self.selector.select(task.query, candidates)
            evaluation = evaluate_decision(
                self.tasks.get_private_facts(task_id),
                selector_result.decision,
                candidates,
            )
            return TaskResult(
                pipeline_version=PIPELINE_VERSION,
                status=(
                    "completed"
                    if evaluation.terminal_reason != "invalid_agent_decision"
                    else "failed"
                ),
                task=task,
                candidates=candidates,
                selector=selector_result,
                evaluation=evaluation,
                error_type=(
                    None
                    if evaluation.terminal_reason != "invalid_agent_decision"
                    else "InvalidAgentDecision"
                ),
                error_message=(
                    None
                    if evaluation.terminal_reason != "invalid_agent_decision"
                    else ", ".join(evaluation.failures)
                ),
            )
        except Exception as exc:
            return TaskResult(
                pipeline_version=PIPELINE_VERSION,
                status="failed",
                task=task,
                candidates=candidates,
                selector=None,
                evaluation=None,
                error_type=type(exc).__name__,
                error_message=str(exc) or type(exc).__name__,
            )

    def run_many(self, task_ids: Sequence[str], *, workers: int = 1) -> list[TaskResult]:
        results: list[TaskResult | None] = [None] * len(task_ids)
        for index, result in self.run_many_iter(task_ids, workers=workers):
            results[index] = result
        return [result for result in results if result is not None]

    def run_many_iter(
        self,
        task_ids: Sequence[str],
        *,
        workers: int = 1,
    ) -> Iterable[tuple[int, TaskResult]]:
        """Yield results as they finish so long runs can checkpoint each task."""
        if not task_ids:
            raise ValueError("task_ids must not be empty")
        if not 1 <= workers <= 64:
            raise ValueError("workers must be between 1 and 64")
        if workers == 1:
            for index, task_id in enumerate(task_ids):
                yield index, self.run(task_id)
            return
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bm25-agent") as pool:
            futures = {
                pool.submit(self.run, task_id): index
                for index, task_id in enumerate(task_ids)
            }
            for future in as_completed(futures):
                yield futures[future], future.result()


def retrieval_summary(
    tasks: TaskStore,
    catalog: BM25Catalog,
    task_ids: Sequence[str],
) -> dict[str, Any]:
    """Measure the hard Top-5 recall ceiling without making any model calls."""
    rank_counts: Counter[str] = Counter()
    hits = 0
    for task_id in task_ids:
        task = tasks.get_public_task(task_id)
        target = tasks.get_private_facts(task_id).target_asin
        rank = next(
            (
                candidate.rank
                for candidate in catalog.search_top5(task.query)
                if candidate.asin == target
            ),
            None,
        )
        if rank is None:
            rank_counts["miss"] += 1
        else:
            hits += 1
            rank_counts[str(rank)] += 1
    count = len(task_ids)
    return {
        "task_count": count,
        "top5_hit_count": hits,
        "top5_recall": hits / count if count else 0.0,
        "target_rank_counts": dict(sorted(rank_counts.items())),
    }


def evaluation_summary(results: Sequence[TaskResult]) -> dict[str, Any]:
    return evaluation_summary_payloads(result.to_dict() for result in results)


def evaluation_summary_payloads(
    payloads: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate JSONL records without retaining full candidate data in memory."""
    task_count = 0
    completed = 0
    evaluated = 0
    reward_valid = 0
    valid_reward_sum = 0.0
    top5_hits = 0
    asin_matches = 0
    strict_successes = 0
    reasons: Counter[str] = Counter()
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    usage_seen = False
    for payload in payloads:
        task_count += 1
        completed += payload.get("status") == "completed"
        evaluation = payload.get("evaluation")
        if isinstance(evaluation, Mapping):
            evaluated += 1
            is_valid = bool(evaluation.get("reward_valid"))
            reward_valid += is_valid
            if is_valid:
                valid_reward_sum += float(evaluation.get("reward", 0.0))
            top5_hits += bool(evaluation.get("target_in_top5"))
            asin_matches += bool(evaluation.get("asin_match"))
            strict_successes += bool(evaluation.get("strict_success"))
            reasons[str(evaluation.get("terminal_reason") or "unknown_error")] += 1
        else:
            reasons[str(payload.get("error_type") or "unknown_error")] += 1

        selector = payload.get("selector")
        usage = selector.get("usage") if isinstance(selector, Mapping) else None
        if isinstance(usage, Mapping):
            usage_seen = True
            for field in usage_totals:
                usage_totals[field] += int(usage.get(field, 0))

    return {
        "pipeline_version": PIPELINE_VERSION,
        "task_count": task_count,
        "completed_count": completed,
        "failed_count": task_count - completed,
        "completion_rate": completed / task_count if task_count else 0.0,
        "evaluated_count": evaluated,
        "reward_valid_count": reward_valid,
        "reward_coverage": reward_valid / task_count if task_count else 0.0,
        "strict_success_count": strict_successes,
        "strict_success_rate": (
            strict_successes / task_count if task_count else 0.0
        ),
        "top5_hit_count": top5_hits,
        "top5_recall": top5_hits / task_count if task_count else 0.0,
        "asin_match_count": asin_matches,
        "asin_selection_rate_given_top5": (
            asin_matches / top5_hits if top5_hits else 0.0
        ),
        "strict_success_rate_given_top5": (
            strict_successes / top5_hits if top5_hits else 0.0
        ),
        "strict_success_rate_given_asin_match": (
            strict_successes / asin_matches if asin_matches else 0.0
        ),
        "average_reward": (
            valid_reward_sum / task_count if task_count else 0.0
        ),
        "average_valid_reward": (
            valid_reward_sum / reward_valid if reward_valid else None
        ),
        "terminal_reason_counts": dict(sorted(reasons.items())),
        "usage": usage_totals if usage_seen else None,
    }


def compact_result(result: TaskResult) -> dict[str, Any]:
    payload = result.to_dict()
    payload["candidates"] = [
        {
            "rank": candidate.rank,
            "bm25_score": candidate.bm25_score,
            "asin": candidate.asin,
            "title": candidate.product.get("title", ""),
        }
        for candidate in result.candidates
    ]
    return payload
