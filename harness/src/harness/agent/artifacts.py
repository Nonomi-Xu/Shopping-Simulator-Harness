"""Immutable per-run audit bundles and verified offline replay."""

import json
from itertools import zip_longest
from pathlib import Path

from .contracts import AgentEvent, AgentRunResult
from .events import replay_events


def write_run_artifacts(result: AgentRunResult, directory: str | Path, *, streamed: bool = False) -> Path:
    if result.status == "waiting_for_user":
        from .orchestrator import validate_checkpoint

        validate_checkpoint(result)
    path = Path(directory).expanduser().resolve()
    if streamed:
        # Reuse only a matching live journal, never a completed or unrelated bundle.
        if (path / "run_manifest.json").exists():
            raise FileExistsError("audit bundle is already complete")
        with (path / "events.jsonl").open(encoding="utf-8") as handle:
            for line, event in zip_longest(handle, result.events):
                if line is None or event is None or AgentEvent.model_validate_json(line) != event:
                    raise ValueError("live event journal differs from completed run")
    else:
        path.mkdir(parents=True, exist_ok=False)
    payloads = {
        "profile.json": {
            "agent": result.profile.model_dump(mode="json"),
            "user": result.user_profile.model_dump(mode="json"),
            "task": (
                result.task_profile.model_dump(mode="json")
                if result.task_profile
                else None
            ),
        },
        "plan.json": result.plan.model_dump(mode="json") if result.plan else None,
        "final_memory.json": (
            result.memory.model_dump(mode="json") if result.memory else None
        ),
        # This file belongs to the evaluator, never to actor context or memory.
        "evaluation.json": (
            result.session.evaluation.model_dump(mode="json")
            if result.session.evaluation
            else None
        ),
        "summary.json": {
            "status": result.status,
            "terminal_reason": result.terminal_reason,
            "task_id": result.task_id,
            "session_id": result.session_id,
            "action_count": result.action_count,
            "search_count": result.search_count,
            "model_turn_count": result.model_turn_count,
            "tool_call_count": result.tool_call_count,
            "usage": result.usage.model_dump(mode="json") if result.usage else None,
            "error_type": result.error_type,
            "error_message": result.error_message,
        },
    }
    if result.status == "waiting_for_user":
        payloads["checkpoint.json"] = result.model_dump(mode="json")
    if not streamed:
        with (path / "events.jsonl").open("x", encoding="utf-8") as handle:
            for event in result.events:
                handle.write(event.model_dump_json() + "\n")
    for name, payload in payloads.items():
        with (path / name).open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    # Written last: an incomplete bundle has no completion manifest.
    with (path / "run_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(result.manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def replay_run(directory: str | Path) -> dict:
    path = Path(directory).expanduser().resolve()
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    events = [
        AgentEvent.model_validate_json(line)
        for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    state = replay_events(events)
    if state["final_event_hash"] != manifest.get("final_event_hash"):
        raise ValueError("manifest and event chain disagree")
    expected_manifest = {
        **state["manifest"],
        "final_event_hash": state["final_event_hash"],
    }
    if manifest != expected_manifest:
        raise ValueError("manifest differs from the recorded run configuration")
    for filename, key in (("plan.json", "plan"), ("final_memory.json", "memory")):
        saved = json.loads((path / filename).read_text(encoding="utf-8"))
        if saved != state[key]:
            raise ValueError(f"{filename} differs from replayed state")
    saved_profile = json.loads((path / "profile.json").read_text(encoding="utf-8"))
    if saved_profile != {
        "agent": state["profile"],
        "user": state["user_profile"],
        "task": state["task_profile"],
    }:
        raise ValueError("profile.json differs from replayed state")
    if state["manifest"].get("harness", {}).get("mode") == "interactive":
        if (
            json.loads((path / "evaluation.json").read_text(encoding="utf-8"))
            is not None
        ):
            raise ValueError("interactive tasks must not contain benchmark evaluation")
    if state["outcome"]["status"] == "waiting_for_user":
        from .orchestrator import validate_checkpoint

        checkpoint = AgentRunResult.model_validate_json(
            (path / "checkpoint.json").read_text(encoding="utf-8")
        )
        validate_checkpoint(checkpoint)
        if checkpoint.manifest != manifest:
            raise ValueError("checkpoint differs from bundle manifest")
    return state
