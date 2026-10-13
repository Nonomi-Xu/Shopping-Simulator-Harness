"""Profile provenance and task-over-preference rules."""

from .contracts import Constraint, TaskProfile, UserProfile


def validate_constraints(
    constraints: list[Constraint], task: TaskProfile, user: UserProfile
) -> None:
    preferences = [p.value for p in user.preferences]
    for constraint in constraints:
        if constraint.predicate is not None:
            # Structural grounding protects thresholds from arbitrary model changes.
            from ..constraints import validate_predicate_quote
            validate_predicate_quote(constraint.predicate, constraint.source_quote)
        if constraint.source == "user_query":
            if constraint.source_quote not in task.task.query:
                raise ValueError("query constraints must quote the public query")
        elif constraint.source == "user_message":
            message = next(
                (
                    m
                    for m in task.user_messages
                    if m.message_id == constraint.source_message_id
                ),
                None,
            )
            if message is None or constraint.source_quote not in message.text:
                raise ValueError(
                    "user-message constraints must quote an actual caller reply"
                )
        elif constraint.source == "user_profile":
            if not any(constraint.source_quote in p for p in preferences):
                raise ValueError(
                    "profile constraints must quote an explicit preference"
                )
            if constraint.priority != "soft":
                raise ValueError("background preferences may only be soft constraints")
        elif constraint.priority != "soft":
            raise ValueError("inferred requirements may only be soft constraints")
        if (
            constraint.source != "user_message"
            and constraint.source_message_id is not None
        ):
            raise ValueError(
                "source_message_id is only valid for user_message constraints"
            )
    by_id = {c.constraint_id: c for c in constraints}
    # An autonomous replan cannot erase or weaken a user's hard requirement.
    for original in task.constraints:
        if (
            original.priority == "hard"
            and by_id.get(original.constraint_id) != original
        ):
            raise ValueError("replanning must preserve existing hard constraints")
