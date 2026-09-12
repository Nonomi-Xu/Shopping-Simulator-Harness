"""Project existing outcomes, resource use and episode metrics into a common report."""


METRICS = {
    "strict_success": "higher", "success_at_1": "higher", "success_at_5": "higher",
    "completed": "higher", "reward_valid": "higher", "reward": "higher",
    "model_turns": "lower", "actions": "lower", "environment_calls": "lower",
    "searches": "lower", "input_tokens": "lower", "output_tokens": "lower",
    "total_tokens": "lower", "wall_seconds": "lower", "active_seconds": "lower",
    "rejected_action_rate": "lower",
    "constraint_satisfaction_rate": "higher", "constraint_verification_coverage": "higher",
    "constraint_forgetting_rate": "lower", "constraint_omission_rate": "lower",
    "post_disclosure_recommendation_violation_rate": "lower",
    "redundant_question_rate": "lower", "premature_recommendation": "lower",
    "mean_information_gain": "higher", "answered_questions": "context",
}


def empty_metrics(wall_seconds):
    values = dict.fromkeys(METRICS)
    # Failures remain in success denominators. Unavailable measurements are null.
    values.update(strict_success=0, success_at_1=0, success_at_5=0, completed=0,
                  reward_valid=0, wall_seconds=wall_seconds)
    return values


def project_metrics(run, evaluation, wall_seconds, episode_metrics=None):
    values = empty_metrics(wall_seconds)
    rank = evaluation.matched_rank if evaluation else None
    valid_submission = evaluation and evaluation.terminal_reason != "invalid_submission"
    values.update(
        strict_success=int(bool(evaluation and evaluation.strict_success)),
        success_at_1=int(bool(valid_submission and rank == 1)),
        success_at_5=int(bool(valid_submission and rank is not None and rank <= 5)),
        completed=int(run.terminal_reason == "submitted"),
        reward_valid=int(bool(evaluation and evaluation.reward_valid)),
        reward=evaluation.reward if evaluation and evaluation.reward_valid else None,
        model_turns=run.model_turn_count, actions=run.action_count,
        environment_calls=run.tool_call_count, searches=run.search_count,
        active_seconds=run.active_duration_seconds,
    )
    if run.usage is not None and all(t.usage is not None for t in run.turns):
        values.update(input_tokens=run.usage.input_tokens, output_tokens=run.usage.output_tokens,
                      total_tokens=run.usage.total_tokens)
    observations = [e.payload for e in run.events if e.kind == "observation"]
    values["rejected_action_rate"] = (
        sum(o["status"] == "rejected" for o in observations)/len(observations)
        if observations else None)
    for name, value in (episode_metrics or {}).items():
        if name in METRICS and name not in {"strict_success", "success_at_1", "reward_valid", "model_turns"}:
            values[name] = int(value) if isinstance(value, bool) else value
    return values


def failure_category(run, evaluation):
    reason = run.terminal_reason
    if run.status == "failed":
        return reason
    if reason != "submitted":
        return "budget_exhausted" if reason in {
            "max_actions", "max_steps", "max_model_turns", "token_budget_exhausted",
            "time_budget_exhausted", "context_budget_exhausted"} else reason
    if evaluation is None:
        return "evaluation_missing"
    return "success" if evaluation.strict_success else evaluation.terminal_reason
