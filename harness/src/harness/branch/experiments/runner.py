"""Fresh providers per run, frozen cases, balanced order and incremental audit output."""
from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

from ....data.search_bm25 import sha256_file
from ...agent import AgentRunner
from ...agent.artifacts import write_run_artifacts
from ...agent.events import digest
from ...agent.orchestrator import _redact
from ...agent.skill import load_agent_skill
from ...catalog import ProductCatalog
from ..episodes.artifacts import write_episode_artifacts
from ..episodes.runner import EpisodeRunner
from ...harness import ShoppingHarness
from ...tasks import TaskStore
from .metrics import empty_metrics, failure_category, project_metrics
from .models import ExperimentConfig
from .report import build_report, write_report


@dataclass(frozen=True)
class Case:
    task_id: str
    episode: object = None


def effective_runtime(config, variant):
    budget = config.budget
    runtime = variant.runtime.model_copy(update={
        "max_actions": budget.max_tool_calls,
        "max_searches": budget.max_searches,
        "max_user_questions": budget.max_user_questions,
    })
    if config.mode == "episode":
        interactive = variant.baseline == "long_horizon"
        runtime = runtime.model_copy(update={
            "interaction_mode": "interactive" if interactive else "offline",
            "progressive_disclosure": interactive, "enforce_constraint_filters": True,
        })
    return runtime


def run_experiment(config, cases, *, products_path, index_path, output_dir, provider_factory, progress=None):
    """provider_factory(variant, repeat_seed) must return a fresh ModelProvider.

    Model execution failures are observations; persistence/data/config failures abort.
    Nothing in this controller's private cases or report is sent to the provider.
    """
    config = ExperimentConfig.model_validate(config)
    cases = list(cases)
    if not cases or len({c.task_id for c in cases}) != len(cases):
        raise ValueError("select one frozen case per task; cases must be nonempty and unique")
    root = Path(output_dir).resolve()
    if root.exists():
        raise ValueError("output directory already exists")
    tasks = TaskStore(products_path)
    for case in cases:
        public = tasks.get_public_task(case.task_id)
        if config.mode == "episode":
            if case.episode is None or case.episode.task_id != case.task_id:
                raise ValueError("episode mode requires a matching cached episode per case")
            if case.episode.dataset_sha256 != tasks.product_data_sha256 or case.episode.full_query != public.query:
                raise ValueError("cached episode does not match the dataset")
            if case.episode.hidden_goal.status != "ready":
                raise ValueError("episode goal is not ready")
        elif case.episode is not None:
            raise ValueError("offline cases must not contain episodes")
    with ProductCatalog(index_path, expected_product_sha256=tasks.product_data_sha256):
        pass
    skills = {v.name: load_agent_skill(v.skill_dir) for v in config.variants}
    ordered_cases = sorted(cases, key=lambda c: digest([config.selection_seed, c.task_id]))
    schedule = []
    for case in ordered_cases:
        for repeat_seed in config.repeat_seeds:
            variants = list(config.variants)
            random.Random(digest([config.selection_seed, repeat_seed, case.task_id])).shuffle(variants)
            schedule.extend((case, repeat_seed, v) for v in variants)
    root.mkdir(parents=True, exist_ok=False)
    private = root / "private"
    private.mkdir(mode=0o700)
    safe_config = _redact(config.model_dump(mode="json"))
    for variant, saved in zip(config.variants, safe_config["variants"]):
        # Environment variable names are provenance, never secret values.
        saved["api_key_env"] = variant.api_key_env
    manifest = {
        "version": config.version,
        "config": safe_config,
        "dataset_sha256": tasks.product_data_sha256,
        "index_sha256": sha256_file(index_path),
        "cases": [{"task_id": c.task_id, "episode_id": c.episode.episode_id if c.episode else None,
                   "episode_sha256": digest(c.episode.model_dump(mode="json")) if c.episode else None}
                  for c in ordered_cases],
        "skill_hashes": {name: skill.source_sha256 for name, skill in skills.items()},
        "effective_runtime": {v.name: effective_runtime(config, v).model_dump(mode="json") for v in config.variants},
        "seed_semantics": "selection and run order are seeded; model seed only sent when explicitly configured",
        "schedule": [{"task_id": c.task_id, "repeat_seed": seed, "variant": v.name} for c, seed, v in schedule],
    }
    with (private / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
    records = []
    with ExitStack() as stack:
        harness = stack.enter_context(ShoppingHarness(products_path=products_path, index_path=index_path)) if config.mode == "offline" else None
        journal = stack.enter_context((private / "records.jsonl").open("x", encoding="utf-8"))
        for index, (case, repeat_seed, variant) in enumerate(schedule, 1):
            row = {"variant": variant.name, "task_id": case.task_id, "repeat_seed": repeat_seed,
                   "episode_id": case.episode.episode_id if case.episode else None,
                   "parameters_sha256": digest(variant.parameters(repeat_seed)), "artifact_dir": None}
            runtime = effective_runtime(config, variant)
            session_id = f"experiment_{index:06d}"
            run = result = None
            started = time.monotonic()
            try:
                provider = provider_factory(variant, repeat_seed)
                if config.mode == "offline":
                    run = AgentRunner(provider=provider, harness=harness, config=runtime,
                        skill=skills[variant.name], max_model_turns=config.budget.max_turns,
                        tool_choice="auto", provider_parameters=variant.parameters(repeat_seed)).run(
                            case.task_id, session_id=session_id, max_steps=config.budget.max_environment_calls)
                    evaluation = run.session.evaluation
                else:
                    # EpisodeRunner resolves interaction flags itself. The remaining runtime
                    # configuration and shared budget match the effective manifest above.
                    runner = EpisodeRunner(products_path=products_path, index_path=index_path,
                        task_store=tasks, provider=provider, config=config.budget,
                        runtime_config=runtime, provider_parameters=variant.parameters(repeat_seed),
                        skill_dir=variant.skill_dir)
                    result = runner.run_episode(case.episode, baseline=variant.baseline)
                    run, evaluation = result.run, result.evaluation
                duration = time.monotonic() - started
                row.update(status=run.status, terminal_reason=run.terminal_reason,
                           failure_category=failure_category(run, evaluation), error_type=run.error_type,
                           metrics=project_metrics(run, evaluation, duration,
                               result.metrics["shopping"] if result else None))
            except Exception as exc:
                row.update(status="failed", terminal_reason="execution_error",
                           failure_category="execution_error", error_type=type(exc).__name__,
                           metrics=empty_metrics(time.monotonic() - started))
            # Artifact failures are not model failures: stop rather than claim a complete report.
            try:
                if run is not None:
                    artifact = private / "runs" / f"{index:06d}"
                    if result is not None:
                        write_episode_artifacts(result, artifact)
                    else:
                        write_run_artifacts(run, artifact)
                    row["artifact_dir"] = str(artifact.relative_to(root))
                    row["provider"] = run.provider
                    row["model"] = run.model
                    row["runtime_manifest_sha256"] = digest(run.manifest)
            finally:
                if harness and any(s.session_id == session_id for s in harness.list_sessions()):
                    harness.release_session(session_id, force=True)
            journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            journal.flush()
            records.append(row)
            if progress:
                progress(index, len(schedule), row)
    report = build_report(records, config)
    write_report(report, records, private)
    completion = {"record_count": len(records), "manifest_sha256": sha256_file(private / "manifest.json"),
                  "records_sha256": sha256_file(private / "records.jsonl")}
    with (private / "completed.json").open("x", encoding="utf-8") as handle:
        json.dump(completion, handle, indent=2)
    return report


def load_completed(directory):
    """Verify input journal before recomputing statistics without any model calls."""
    path = Path(directory) / "private"
    completion = json.loads((path / "completed.json").read_text())
    for name in ("manifest", "records"):
        filename = path / (name + (".jsonl" if name == "records" else ".json"))
        if sha256_file(filename) != completion[name + "_sha256"]:
            raise ValueError("experiment journal/manifest hash mismatch")
    manifest = json.loads((path / "manifest.json").read_text())
    records = [json.loads(line) for line in (path / "records.jsonl").read_text().splitlines()]
    if len(records) != completion["record_count"]:
        raise ValueError("incomplete experiment journal")
    actual = [{k: r[k] for k in ("task_id", "repeat_seed", "variant")} for r in records]
    if actual != manifest["schedule"]:
        raise ValueError("journal does not match planned matrix")
    return ExperimentConfig.model_validate(manifest["config"]), records
