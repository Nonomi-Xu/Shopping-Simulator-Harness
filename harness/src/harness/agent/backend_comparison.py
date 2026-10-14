"""Behavioral parity checks; deliberately separate from model quality evaluation."""

from .events import digest, replay_events


def behavior(result):
    replay_events(result.events)
    return {
        "actions": [e.payload for e in result.events if e.kind == "action"],
        "observations": [e.payload for e in result.events if e.kind == "observation"],
        "handoffs": [e.payload for e in result.events if e.kind == "agent_handoff"],
        "budgets": [
            e.payload["budget"] for e in result.events
            if e.kind in {"run_finished", "run_paused"}
        ],
        "counts": {
            key: getattr(result, key) for key in (
                "action_count", "search_count", "model_turn_count", "tool_call_count"
            )
        },
        "usage": result.usage.model_dump(mode="json") if result.usage else None,
        "session": result.session.model_dump(mode="json"),
        "evaluation": (
            result.session.evaluation.model_dump(mode="json")
            if result.session.evaluation else None
        ),
        "context": [e.payload for e in result.events if e.kind == "context_built"],
        "plan": result.plan.model_dump(mode="json"),
        "memory": result.memory.model_dump(mode="json"),
        "task": result.task_profile.model_dump(mode="json"),
        "outcome": {
            key: getattr(result, key)
            for key in ("status", "terminal_reason", "error_type", "error_message")
        },
    }


def compare_results(native, graph):
    """Compare same-session executions; time, backend manifest and hashes may differ."""
    if native.session_id != graph.session_id or native.task_id != graph.task_id:
        raise ValueError("comparison requires the same task and session IDs")
    left, right = behavior(native), behavior(graph)
    checks = {key: left[key] == right[key] for key in left}
    return {
        "equal": all(checks.values()),
        "checks": checks,
        "native_sha256": digest(left),
        "langgraph_sha256": digest(right),
        "outcome": left["outcome"],
        "counts": left["counts"],
        "evaluation": left["evaluation"],
    }
