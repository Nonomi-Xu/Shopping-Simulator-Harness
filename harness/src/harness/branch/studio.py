"""Local Agent Server graph with serializable, per-thread shopping runtime state.

Only the unlabelled interactive environment is exposed. Every node restores its
own Session from checkpoint data and returns the new state. There is no global
mutable run registry. Re-execution may repeat model/read calls, but cannot place
orders: the environment's submit operation only records a recommendation.
"""
from collections import Counter
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from ..agent import AgentRunner, RuntimeConfig
from ..agent.contracts import (
    AgentEvent, AgentRunResult, PendingQuestion, SearchPlan, TaskProfile, UserReply, WorkingMemory,
)
from ..agent.events import digest
from ..agent.orchestrator import _Run, validate_checkpoint
from ..agent.tracing import public_outcome
from ..catalog import ProductCatalog
from ..paths import HARNESS_DIR
from ..providers import AssistantTurn, ChatMessage
from ..providers.factory import build_provider
from ..sources import JsonProductSource
from ..schemas import SessionSnapshot
from .interactive import InteractiveHarness
from .studio_demo import StudioDemoProvider


class StudioInput(BaseModel):
    query: str = Field(min_length=1, max_length=16000, description="Demo: 寻找白色耳机；or 寻找耳机 to test clarification.")


class RuntimeSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    session: SessionSnapshot
    task_profile: TaskProfile
    plan: SearchPlan
    memory: WorkingMemory
    events: list[AgentEvent]
    turns: list[AssistantTurn]
    messages: list[ChatMessage]
    action_count: int
    search_count: int
    active_duration_seconds: float
    pending_question: PendingQuestion | None
    manifest: dict
    repeated: dict[str, int]


class StudioState(TypedDict, total=False):
    query: str
    runtime: dict
    role: str
    status: str
    plan: dict
    memory: dict
    budget: dict
    pending_question: dict | None
    outcome: dict | None
    recommendations: list[dict]
    result: dict | None


@contextmanager
def environment():
    provider_name = os.environ.get("SHOPPING_STUDIO_PROVIDER", "demo")
    if provider_name == "demo":
        provider = StudioDemoProvider()
        source = JsonProductSource(HARNESS_DIR / "examples/public-products.json", currency="CNY")
    else:
        provider = build_provider(
            provider_name, model=os.environ.get("SHOPPING_STUDIO_MODEL"),
            base_url=os.environ.get("SHOPPING_STUDIO_BASE_URL"),
        )
        source_json = os.environ.get("SHOPPING_STUDIO_SOURCE_JSON")
        source = (JsonProductSource(Path(source_json).expanduser(), currency="CNY") if source_json
                  else ProductCatalog(Path(os.environ.get("SHOPPING_STUDIO_INDEX", str(HARNESS_DIR / "data/product_search.sqlite3")))))
    with InteractiveHarness(source) as harness:
        runner = AgentRunner(provider=provider, harness=harness,
                             config=RuntimeConfig(agent_mode="multi", interaction_mode="interactive",
                                                  orchestration_backend="langgraph"))
        yield runner


def snapshot(run):
    return RuntimeSnapshot(
        session_id=run.session_id, session=run.owner.harness.get_session(run.session_id).model_dump(mode="json"),
        task_profile=run.task, plan=run.plan, memory=run.memory.state, events=run.events.events,
        turns=run.turns, messages=run.messages, action_count=run.actions, search_count=run.searches,
        active_duration_seconds=run.elapsed(), pending_question=run.pending_question,
        manifest=run.manifest, repeated=dict(run.repeated),
    ).model_dump(mode="json")


def restore(runner, value):
    saved = RuntimeSnapshot.model_validate(value)
    expected = saved.manifest
    if (expected["harness"] != runner.harness.manifest
            or expected["provider"] != runner.provider.name
            or expected["model"] != runner.provider.model
            or expected["provider_endpoint_sha256"] != digest(getattr(runner.provider, "base_url", None))
            or expected["skill_sha256"] != runner.skill.source_sha256
            or RuntimeConfig.model_validate(expected["config"]) != runner.config):
        raise ValueError("Studio checkpoint configuration or source changed; start a new thread")
    # _Run.restore consumes the typed runtime fields used by AgentRunResult;
    # active Studio checkpoints are not falsely labelled as paused audit runs.
    if saved.session.evaluation is not None or saved.session.state.status != "active":
        raise ValueError("Studio restore requires an active unlabelled session")
    runner.harness.restore_interactive_session(saved.session, checkpoint_hash=digest(value))
    run = _Run.restore(runner, saved)
    # Unlike audit reconstruction, preserve exact counters, including rejected
    # calls and resets after user replies, at every graph node boundary.
    run.repeated = Counter(saved.repeated)
    return run


def project(run, result=None):
    recommendations = []
    if result:
        for step in result.session.history:
            if step.tool_name == "submit_recommendations" and step.status == "success":
                recommendations = step.arguments.get("recommendations", [])
    return {
        "runtime": snapshot(run), "role": run.coordinator.role,
        "status": result.status if result else "running",
        "plan": run.plan.model_dump(mode="json"), "memory": run.memory.state.model_dump(mode="json"),
        "budget": run.budget().model_dump(mode="json"),
        "pending_question": run.pending_question.model_dump(mode="json") if run.pending_question else None,
        "outcome": public_outcome(result) if result else None,
        "recommendations": recommendations,
        "result": result.model_dump(mode="json") if result else None,
    }


def initialize(state):
    query = StudioInput(query=state["query"]).query
    with environment() as runner:
        if runner.provider.name == "studio-demo" and "耳机" not in query:
            raise ValueError("离线演示支持耳机颜色流程，请输入：寻找白色耳机，或配置真实模型")
        session = runner.harness.create_interactive_session(query, session_id=f"studio_{uuid4().hex}")
        return project(_Run(runner, session))


def role_node(role):
    def execute(state):
        with environment() as runner:
            run = restore(runner, state["runtime"])
            if run.coordinator.role != role:
                raise ValueError("Studio role differs from checkpoint")
            try:
                result = run._step()
            except Exception as exc:
                result = run.finish("failed", "runtime_error", error=exc)
            return project(run, result)
    return execute


def await_user(state):
    # Interrupt before opening a source or invoking a model: resuming this node
    # cannot repeat the action that asked the question.
    reply = interrupt(state["pending_question"])
    reply = UserReply(text=reply) if isinstance(reply, str) else UserReply.model_validate(reply)
    paused = AgentRunResult.model_validate(state["result"])
    validate_checkpoint(paused)
    if not set(reply.replace_constraint_ids) <= {c.constraint_id for c in paused.task_profile.constraints}:
        raise ValueError("replacement references an unknown constraint")
    if len(paused.task_profile.user_messages) >= 20:
        raise ValueError("user-message limit reached")
    with environment() as runner:
        run = restore(runner, state["runtime"])
        run.accept_reply(reply)
        return project(run)


def route(state) -> Literal["planner", "executor", "observer", "await_user", "__end__"]:
    if state["status"] == "waiting_for_user":
        return "await_user"
    if state["status"] != "running":
        return END
    return state["role"]


def create_graph(checkpointer=None):
    builder = StateGraph(StudioState, input_schema=StudioInput)
    builder.add_node("initialize", initialize)
    for role in ("planner", "executor", "observer"):
        builder.add_node(role, role_node(role))
    builder.add_node("await_user", await_user)
    builder.add_edge(START, "initialize")
    for name in ("initialize", "planner", "executor", "observer", "await_user"):
        builder.add_conditional_edges(name, route)
    return builder.compile(checkpointer=checkpointer, name="Shopping Simulator").with_config(
        {"recursion_limit": 100}
    )


graph = create_graph()
