"""Estimate the prompt-size and prefix-cache effect of a layout change.

Provider prefix caches (DeepSeek context caching) reuse the leading tokens that are
byte-identical to an earlier request, so the layout question is: how much of the
prompt survives from one turn to the next?  This tool rebuilds the prompt each
recorded turn would have sent under three layouts and compares per-turn cacheable
prefix and total size:

* ``recorded``  - the bytes in ``context_built`` (alphabetical state fields, full
  history);
* ``layout``    - the state block re-serialized by ``state_json`` (stable-first
  fields);
* ``current``   - ``layout`` plus the prompt diet: ``summary_view`` on candidate
  rows and ``prompt_message`` on the transcript (no replayed reasoning, projected
  runtime-tool echoes).

It never calls a provider, so a prompt change can be evaluated before spending
tokens.  The window is the recorded one, so the reported effect is conservative:
under the smaller prompt the runtime would keep at least as many history groups.

    python tools/prefix_cache_sim.py --run-dir ../outputs/rollouts/<run> [--limit N]
"""

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.harness.agent.context import prompt_message, state_json  # noqa: E402
from src.harness.agent.memory import summary_view  # noqa: E402
from src.harness.providers import ChatMessage  # noqa: E402


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def layout_prompt(recorded, mode):
    """Rebuild one recorded prompt under the requested layout."""
    state_content = recorded[1]["content"]
    if mode != "recorded":
        state = json.loads(state_content)
        if mode == "current":
            for candidate in state.get("memory", {}).get("candidates", []):
                if isinstance(candidate.get("summary"), dict):
                    candidate["summary"] = summary_view(candidate["summary"])
        state_content = state_json(state)
    messages = [ChatMessage.model_validate(recorded[0]), ChatMessage(role="user", content=state_content)]
    for raw in recorded[2:]:
        message = ChatMessage.model_validate(raw)
        messages.append(prompt_message(message) if mode == "current" else message)
    return "\n".join(canonical(m.as_provider_payload()) for m in messages)


def shared_prefix(left, right):
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def task_directories(root: pathlib.Path, limit: int | None) -> list[pathlib.Path]:
    if (root / "events.jsonl").exists():
        return [root]
    directories = sorted(p for p in root.glob("tasks/*") if (p / "events.jsonl").exists())
    return directories[:limit] if limit else directories


def measure(directory: pathlib.Path) -> dict | None:
    events = [
        json.loads(line)
        for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    contexts = [e["payload"]["messages"] for e in events if e["kind"] == "context_built"]
    usages = [e["payload"].get("usage") or {} for e in events if e["kind"] == "model_turn"]
    if len(contexts) < 2:
        return None
    misses = {"recorded": 0, "layout": 0, "current": 0}
    totals = {"recorded": 0, "layout": 0, "current": 0}
    previous = {"recorded": None, "layout": None, "current": None}
    for messages in contexts:
        for mode in misses:
            prompt = layout_prompt(messages, mode)
            if previous[mode] is not None:
                misses[mode] += len(prompt) - shared_prefix(previous[mode], prompt)
                totals[mode] += len(prompt)
            previous[mode] = prompt
    return {
        "turns": len(contexts),
        "chars": totals,
        "miss_chars": misses,
        "input_tokens": sum(u.get("input_tokens") or 0 for u in usages),
        "cached_input_tokens": sum(u.get("cached_input_tokens") or 0 for u in usages),
        "cache_miss_input_tokens": sum(u.get("cache_miss_input_tokens") or 0 for u in usages),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    measured = [
        result
        for result in (measure(d) for d in task_directories(args.run_dir, args.limit))
        if result
    ]
    if not measured:
        raise SystemExit("no multi-turn runs found under this run directory")

    def total(field):
        return sum(item[field] for item in measured)

    def summed(key, mode):
        return sum(item[key][mode] for item in measured)

    report = {
        "run_dir": str(args.run_dir.resolve()),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": len(measured),
        "turns_after_first": total("turns") - len(measured),
        "observed": {
            "input_tokens": total("input_tokens"),
            "cached_input_tokens": total("cached_input_tokens"),
            "cache_miss_input_tokens": total("cache_miss_input_tokens"),
        },
    }
    for mode in ("recorded", "layout", "current"):
        miss = summed("miss_chars", mode)
        size = summed("chars", mode)
        report[mode] = {
            "prompt_chars": size,
            "miss_chars": miss,
            "cacheable_share": (1 - miss / size) if size else None,
            "miss_reduction_vs_recorded": (
                1 - miss / summed("miss_chars", "recorded")
            )
            if summed("miss_chars", "recorded")
            else None,
        }
    observed_input = report["observed"]["input_tokens"]
    if observed_input:
        report["observed"]["cache_hit_rate"] = (
            report["observed"]["cached_input_tokens"] / observed_input
        )
        reduction = report["current"]["miss_reduction_vs_recorded"]
        projected_miss = report["observed"]["cache_miss_input_tokens"] * (1 - reduction)
        report["projected"] = {
            "cache_miss_input_tokens": round(projected_miss),
            "cache_hit_rate": (observed_input - projected_miss) / observed_input,
        }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
