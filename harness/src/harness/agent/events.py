"""Append-only hash-linked events and offline state reconstruction."""

import hashlib
import json
from typing import Any

from .contracts import (
    AgentEvent,
    AgentObservation,
    SearchPlan,
    TaskProfile,
    WorkingMemory,
)


def digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class EventStore:
    version = "shopping-event-v1"

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._events: list[AgentEvent] = []

    @property
    def events(self) -> list[AgentEvent]:
        return [event.model_copy(deep=True) for event in self._events]

    def append(
        self, kind: str, payload: dict, caused_by: str | None = None
    ) -> AgentEvent:
        sequence = len(self._events) + 1
        if caused_by is not None and caused_by not in {
            e.event_id for e in self._events
        }:
            raise ValueError("event cause must reference an earlier event")
        values = {
            "version": self.version,
            "event_id": f"event_{sequence:06d}",
            "session_id": self.session_id,
            "sequence": sequence,
            "kind": kind,
            "caused_by": caused_by,
            "payload": payload,
            "previous_hash": self._events[-1].event_hash if self._events else "0" * 64,
        }
        event = AgentEvent(**values, event_hash=digest(values))
        self._events.append(event.model_copy(deep=True))
        return event


def replay_events(events: list[AgentEvent]) -> dict:
    """Validate a complete event chain and reconstruct its recorded public state.

    This is audit replay, not re-execution of a remote model or a purchase.
    Hashes detect changes relative to a trusted final hash, not malicious rewriting.
    """
    if (
        not events
        or events[0].kind != "run_started"
        or events[-1].kind not in {"run_finished", "run_paused"}
    ):
        raise ValueError(
            "replay requires a complete run_started ... run_finished/run_paused chain"
        )
    previous = "0" * 64
    seen: set[str] = set()
    actions: set[str] = set()
    paused = False
    state: dict = {"session_id": events[0].session_id}
    for sequence, event in enumerate(events, 1):
        if (
            event.sequence != sequence
            or event.event_id != f"event_{sequence:06d}"
            or event.session_id != state["session_id"]
            or event.previous_hash != previous
        ):
            raise ValueError("broken event sequence or session boundary")
        if event.caused_by is not None and event.caused_by not in seen:
            raise ValueError("invalid event cause")
        if event.event_hash != digest(
            event.model_dump(mode="json", exclude={"event_hash"})
        ):
            raise ValueError("event hash mismatch")
        payload = event.payload
        if paused and event.kind != "run_resumed":
            raise ValueError("a paused run must resume before further events")
        if event.kind == "run_started":
            if sequence != 1:
                raise ValueError("duplicate run_started")
            state.update(payload)
        elif event.kind == "action":
            if payload["action_id"] in actions:
                raise ValueError("duplicate action ID")
            actions.add(payload["action_id"])
        elif event.kind == "observation":
            observation = AgentObservation.model_validate(payload)
            if observation.action_id not in actions:
                raise ValueError("observation references unknown action")
            state["latest_observation"] = payload
        elif event.kind == "plan_updated":
            SearchPlan.model_validate(payload["plan"])
            TaskProfile.model_validate(payload["task_profile"])
            # Validate without injecting new defaults into historical audit payloads.
            state["plan"] = payload["plan"]
            state["task_profile"] = payload["task_profile"]
        elif event.kind == "memory_updated":
            memory = WorkingMemory.model_validate(payload)
            if memory.session_id != state["session_id"]:
                raise ValueError("memory belongs to another session")
            state["memory"] = payload
        elif event.kind == "run_finished":
            if sequence != len(events):
                raise ValueError("events after terminal event")
            state["outcome"] = payload
        elif event.kind == "run_paused":
            paused = True
            state["outcome"] = payload
        elif event.kind == "run_resumed":
            if not paused:
                raise ValueError("run_resumed requires a paused run")
            if (
                payload["question_id"]
                != state["outcome"]["pending_question"]["question_id"]
            ):
                raise ValueError("resume question mismatch")
            paused = False
        previous = event.event_hash
        seen.add(event.event_id)
    state["final_event_hash"] = previous
    return state
