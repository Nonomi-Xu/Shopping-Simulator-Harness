"""Replay identical recorded model turns through native and LangGraph schedulers.

No model API calls. Run from harness with --run-dir repeated for each audit
bundle, --products-path, --index-path and a new --output-dir. Exits nonzero on
behavior differences or invalid input. This checks scheduler parity, not live
model accuracy or equality to a historical runtime version.
"""

import argparse
import json
from pathlib import Path

from src.harness.agent import AgentRunner, RuntimeConfig, write_run_artifacts, replay_run
from src.harness.agent.backend_comparison import compare_results
from src.harness.agent.contracts import AgentProfile, UserProfile
from src.harness.harness import ShoppingHarness
from src.harness.providers import AssistantTurn, ProviderAPIError


class RecordedProvider:
    def __init__(self, manifest, turns):
        self.name = manifest["provider"]
        self.model = manifest["model"]
        self.turns = iter(turns)
        self.exhausted = False

    def complete(self, messages, *, tools=(), tool_choice=None, parameters=None):
        try:
            return next(self.turns).model_copy(deep=True)
        except StopIteration:
            self.exhausted = True
            raise ProviderAPIError("recorded model turns exhausted") from None


def compare_bundles(run_dirs, products_path, index_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    with ShoppingHarness(products_path=products_path, index_path=index_path) as harness:
        for index, directory in enumerate(run_dirs, 1):
            directory = Path(directory)
            state = replay_run(directory)
            manifest = state["manifest"]
            config = RuntimeConfig.model_validate(manifest["config"])
            if config.interaction_mode != "offline":
                raise ValueError("interactive recordings require the resume parity tests")
            if config.max_duration_seconds is not None:
                raise ValueError("wall-clock-limited recordings are not deterministic")
            recorded_environment = {k: v for k, v in manifest["harness"].items()
                                    if k != "default_max_steps"}
            replay_environment = {k: v for k, v in harness.manifest.items()
                                  if k != "default_max_steps"}
            if recorded_environment != replay_environment:
                raise ValueError("recording source/index manifest differs from replay environment")
            events = [json.loads(line) for line in
                      (directory / "events.jsonl").read_text().splitlines() if line.strip()]
            turns = [AssistantTurn.model_validate(e["payload"]["response"])
                     for e in events if e["kind"] == "model_turn"]
            if not turns:
                raise ValueError("recording contains no model turns")
            task_id = state["task_profile"]["task"]["task_id"]
            session_id = state["session_id"]
            results, exhausted = {}, {}
            for backend in ("native", "langgraph"):
                provider = RecordedProvider(manifest, turns)
                runner = AgentRunner(
                    harness=harness, provider=provider,
                    config=config.model_copy(update={"orchestration_backend": backend}),
                    profile=AgentProfile.model_validate(state["profile"]),
                    user_profile=UserProfile.model_validate(state["user_profile"]),
                    max_model_turns=manifest["max_model_turns"],
                    tool_choice=manifest["tool_choice"],
                )
                result = runner.run(task_id, session_id=session_id,
                                    max_steps=manifest["environment_max_steps"])
                results[backend] = result
                exhausted[backend] = provider.exhausted
                write_run_artifacts(result, output_dir / f"{index:04d}" / backend)
                harness.release_session(session_id, force=True)
            comparison = compare_results(results["native"], results["langgraph"])
            rows.append({
                "source": str(directory.resolve()), "task_id": task_id,
                "agent_mode": config.agent_mode,
                "recording_exhausted": exhausted,
                "historical_skill_matches": manifest["skill_sha256"] == results["native"].skill_sha256,
                **comparison,
            })
    report = {
        "version": "shopping-backend-parity-v1",
        "method": "identical recorded model turns; real environment and evaluator",
        "task_count": len(rows),
        "equal_count": sum(r["equal"] for r in rows),
        "all_equal": bool(rows) and all(r["equal"] for r in rows),
        "rows": rows,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--products-path", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = compare_bundles(args.run_dir, args.products_path, args.index_path, args.output_dir)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return 0 if report["all_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
