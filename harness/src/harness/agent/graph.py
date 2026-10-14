"""LangGraph scheduling over the existing per-run business state.

Graph state is a public inspection projection, not a second mutable memory.
The runtime owns Session, evidence and budgets. Waiting runs use the existing
validated audit checkpoint and AgentRunner.resume, not graph-node replay.
"""

from typing import Callable, TypedDict

from .tracing import public_outcome


class ShoppingGraphState(TypedDict):
    session_id: str
    role: str
    task: dict
    plan: dict
    memory: dict
    budget: dict
    review: dict | None
    pending_question: dict | None
    outcome: dict | None
    done: bool


def require_langgraph():
    try:
        from langgraph.graph import StateGraph  # noqa: F401
    except ImportError as exc:
        raise ImportError("LangGraph backend requires pip install '.[graph]'") from exc


def build_graph(step: Callable, *, multi_agent: bool):
    """Build explicit role nodes; each node executes exactly one guarded turn."""
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(ShoppingGraphState)
    roles = ("planner", "executor", "observer") if multi_agent else ("single",)

    def route(state):
        return "end" if state["done"] else state["role"]

    def node(role):
        def execute(state):
            if state["role"] != role:
                raise ValueError("graph role differs from runtime handoff")
            return step(role)
        return execute

    routes = {role: role for role in roles}
    routes["end"] = END
    builder.add_conditional_edges(START, route, routes)
    for role in roles:
        builder.add_node(role, node(role))
        builder.add_conditional_edges(role, route, routes)
    # No retry/cache/checkpointer: a Session mutation must never be replayed by
    # the graph independently of the runtime's validated resume protocol.
    return builder.compile(name="shopping-agent")


def execute_graph(run):
    result = None

    def view():
        return ShoppingGraphState(
            session_id=run.session_id,
            role=run.coordinator.role if run.coordinator else "single",
            task=run.task.model_dump(mode="json"),
            plan=run.plan.model_dump(mode="json"),
            memory=run.memory.state.model_dump(mode="json"),
            budget=run.budget().model_dump(mode="json"),
            review=run.coordinator.snapshot() if run.coordinator else None,
            pending_question=(
                run.pending_question.model_dump(mode="json")
                if run.pending_question else None
            ),
            outcome=public_outcome(result) if result is not None else None,
            done=result is not None,
        )

    def step(role):
        nonlocal result
        current = run.coordinator.role if run.coordinator else "single"
        if current != role:
            raise ValueError("graph and runtime role are out of sync")
        result = run._step()
        return view()

    graph = build_graph(step, multi_agent=run.coordinator is not None)
    graph.invoke(
        view(),
        config={
            # A final step may be needed to report exhaustion without a model
            # call. Graph supersteps are not shopping actions or model turns.
            "recursion_limit": max(0, run.owner.max_model_turns - len(run.turns)) + 3,
            "metadata": {"session_id": run.session_id, "thread_id": run.session_id},
        },
    )
    if result is None:
        raise RuntimeError("graph ended without a runtime outcome")
    return result
