"""Deterministic, versioned episode metrics. Unknown evidence is reported explicitly."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from ....data.product_facts import predicate_statuses
from ...constraints import predicate_key


@dataclass(frozen=True)
class MetricContext:
    episode: object
    trajectory: object
    evaluation: object
    disclosures: list
    revealed_ids: set[str]
    catalog: object


class EpisodeMetric(Protocol):
    name: str
    version: str
    def compute(self, context: MetricContext) -> dict: ...


def ratio(a, b):
    return a / b if b else None


def tracked(requirement, constraints):
    return any(c.priority == requirement.priority and (
        c.source_quote == requirement.source_quote or
        (c.predicate is not None and requirement.predicate is not None and predicate_key(c.predicate) == predicate_key(requirement.predicate))) for c in constraints)


class ShoppingMetrics:
    name = "shopping"
    version = "long-horizon-metrics-v1"

    def compute(self, context):
        t, goal = context.trajectory, context.episode.hidden_goal
        evaluation = context.evaluation
        matched_rank = next((r.rank for r in evaluation.recommendations if r.asin_match), None) if evaluation else None
        valid_submission = bool(evaluation and evaluation.terminal_reason != "invalid_submission")
        counts = {name: sum(turn.action_type == name for turn in t.turns) for name in ["ask", "search", "inspect", "compare", "recommend", "internal"]}
        hard = [r for r in goal.requirements if r.priority == "hard" and r.requirement_id in context.revealed_ids]
        statuses = []
        details = []
        for recommendation in t.final_recommendation:
            product = context.catalog.get_product(recommendation.asin).model_dump(mode="json")
            for requirement in hard:
                status = predicate_statuses(product, [requirement.predicate], selected=recommendation.selected_options)[0] if requirement.predicate else "unknown"
                statuses.append(status)
                details.append({"asin": recommendation.asin, "requirement_id": requirement.requirement_id, "status": status})
        satisfied, violated = statuses.count("satisfied"), statuses.count("violated")
        final_state = t.turns[-1].preferences_after.known_requirements if t.turns else []
        remembered = {r.requirement_id for r in hard if tracked(r, final_state)}
        critical = {r.requirement_id for r in goal.requirements if r.critical}
        premature = bool(t.final_recommendation) and bool(critical-context.revealed_ids or
            any(r.requirement_id in critical and not tracked(r, final_state) for r in goal.requirements))
        forgotten = opportunities = 0
        acquired = {}
        for turn in t.turns:
            if turn.action_type in {"search", "inspect", "compare", "recommend"}:
                opportunities += len(acquired)
                forgotten += sum(not tracked(r, turn.preferences_before.known_requirements) for r in acquired.values())
            for r in hard:
                if tracked(r, turn.preferences_after.known_requirements): acquired[r.requirement_id] = r
        gains = [ratio(e.candidates_before-e.candidates_after, e.candidates_before) for e in context.disclosures
                 if e.candidates_before is not None and e.candidates_after is not None]
        gains = [g for g in gains if g is not None]
        return {"metric_version": self.version,
            "success_at_1": bool(valid_submission and matched_rank == 1),
            "success_at_k": {str(k): bool(valid_submission and matched_rank is not None and matched_rank <= k) for k in range(1,6)},
            "strict_success": bool(evaluation and evaluation.strict_success),
            "reward_valid": bool(evaluation and evaluation.reward_valid),
            "turns_to_success": t.model_turn_count if valid_submission and matched_rank is not None else None,
            "model_turns": t.model_turn_count, "tool_calls": t.tool_call_count,
            "environment_calls": t.environment_call_count, "action_counts": counts,
            "constraint_satisfaction_rate": ratio(satisfied, len(statuses)),
            "constraint_satisfaction_when_verifiable": ratio(satisfied, satisfied+violated),
            "constraint_verification_coverage": ratio(satisfied+violated, len(statuses)),
            "constraint_status_counts": {s: statuses.count(s) for s in ("satisfied", "violated", "unknown")},
            "constraint_checks": details,
            "constraint_forgetting_rate": ratio(forgotten, opportunities),
            "constraint_forgetting_opportunities": opportunities,
            "constraint_omission_rate": ratio(len(hard)-len(remembered), len(hard)),
            "post_disclosure_recommendation_violation_rate": ratio(violated, len(statuses)),
            "redundant_question_rate": ratio(sum(e.repeated for e in context.disclosures), len(context.disclosures)),
            "answered_questions": len(context.disclosures), "premature_recommendation": premature,
            "rejected_recommendation_attempts": sum(turn.action_type == "recommend" and turn.observation.status == "rejected" for turn in t.turns),
            "mean_information_gain": ratio(sum(gains), len(gains)),
            "candidate_reductions": [{"action_sequence": e.action_sequence, "before": e.candidates_before, "after": e.candidates_after,
                "executable_constraints": e.executable_constraints} for e in context.disclosures],
            "information_gain_basis": "fixed_catalog_executable_revealed_hard_predicates_only"}


def compute_metrics(context, metrics=None):
    result = {}
    for metric in metrics or [ShoppingMetrics()]:
        if metric.name in result: raise ValueError("duplicate metric namespace")
        result[metric.name] = metric.compute(context)
    return result


def summarize(results):
    values = [r.metrics["shopping"] for r in results]
    successes = sum(v["success_at_k"]["5"] for v in values)
    turns = [v["turns_to_success"] for v in values if v["turns_to_success"] is not None]
    output = {"episode_count": len(values), "success_at_1": ratio(sum(v["success_at_1"] for v in values), len(values)),
        "success_at_5": ratio(successes, len(values)), "strict_success_rate": ratio(sum(v["strict_success"] for v in values), len(values)),
        "average_turns_to_success": ratio(sum(turns), len(turns)),
        "average_tool_calls": ratio(sum(v["tool_calls"] for v in values), len(values)),
        "average_search_calls": ratio(sum(v["action_counts"]["search"] for v in values), len(values)),
        "search_calls_per_success": ratio(sum(v["action_counts"]["search"] for v in values), successes),
        "premature_recommendation_rate": ratio(sum(v["premature_recommendation"] for v in values), len(values))}
    for name in ["constraint_satisfaction_rate", "constraint_verification_coverage", "constraint_forgetting_rate", "constraint_omission_rate", "redundant_question_rate", "mean_information_gain"]:
        known = [v[name] for v in values if v[name] is not None]
        output[name] = ratio(sum(known), len(known))
        output[name+"_episode_coverage"] = ratio(len(known), len(values))
    return output
