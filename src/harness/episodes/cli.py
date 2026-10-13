"""Episode CLI; expensive extraction is always an explicit preprocessing command."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from ..agent.contracts import RuntimeConfig
from ..agent.events import digest
from ..paths import PROJECT_DIR, result_path, default_run_path
from ..env import load_dotenv
from ..providers import DeepSeekProvider, DEFAULT_MODEL, DEFAULT_BASE_URL
from ..tasks import TaskStore
from .models import EpisodeConfig, DisclosurePolicy
from .parser import RuleGoalParser, StructuredGoalParser
from .preprocess import prepare_episodes
from .store import EpisodeStore
from .runner import EpisodeRunner
from .artifacts import write_episode_artifacts, replay_episode
from .baselines import run_comparison


def register(parent):
    command = parent.add_parser("episodes", help="Prepare and run private-goal interactive benchmarks")
    sub = command.add_subparsers(dest="episode_command", required=True)
    for name in ("prepare", "run", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--products-path", type=Path, default=PROJECT_DIR / "data/fine_items_eval_train_all.json")
        p.add_argument("--index-path", type=Path, default=PROJECT_DIR / "data/product_search.sqlite3")
        p.add_argument("--cache", type=Path, required=True)
        p.add_argument("--count", type=int, default=10, help="maximum selected records; 0 explicitly selects all")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--episode-id", action="append")
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--base-url", default=DEFAULT_BASE_URL)
        p.add_argument("--provider-parameters", default='{"temperature":0}')
        p.add_argument("--env-file", type=Path)
        p.add_argument("--no-env-file", action="store_true")
        if name == "prepare":
            p.add_argument("--parser", choices=("structured", "rule"), default="structured")
            p.add_argument("--split", choices=("all", "test"), default="test")
            p.add_argument("--disclosure-policy", type=Path)
        else:
            p.add_argument("--output-dir", type=result_path, help="Exact new directory, relative to Harness project; default: unique outputs directory.")
            p.add_argument("--config", type=Path)
            p.add_argument("--runtime-config", type=Path)
            p.add_argument("--skill-dir", type=Path)
        if name == "run": p.add_argument("--baseline", choices=("full_query", "long_horizon"), default="long_horizon")
    replay = sub.add_parser("replay")
    replay.add_argument("--run-dir", type=result_path, required=True)


def handle(args):
    if args.episode_command == "replay":
        result = replay_episode(args.run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.count < 0: raise ValueError("count must be nonnegative")
    parameters = json.loads(args.provider_parameters)
    if not isinstance(parameters, dict): raise ValueError("provider parameters must be an object")
    def provider():
        load_dotenv(args.env_file, disabled=args.no_env_file)
        return DeepSeekProvider(model=args.model, base_url=args.base_url, default_parameters=parameters)
    if args.episode_command == "prepare":
        tasks = TaskStore(args.products_path)
        ids = tasks.list_task_ids(args.split)
        if args.count:
            ids = tasks.sample_task_ids(args.split, min(args.count, len(ids)), seed=args.seed)
        parser = RuleGoalParser() if args.parser == "rule" else StructuredGoalParser(provider(), parameters=parameters)
        policy = DisclosurePolicy.model_validate_json(args.disclosure_policy.read_text()) if args.disclosure_policy else DisclosurePolicy()
        with EpisodeStore(args.cache) as store:
            summary = prepare_episodes((tasks.get_public_task(i) for i in ids), parser, store,
                                       dataset_sha256=tasks.product_data_sha256, policy=policy, seed=args.seed)
        report = {"summary": summary, "selected_task_ids": ids, "dataset": tasks.manifest,
                  "parser": parser.manifest, "policy": policy.model_dump(mode="json"), "seed": args.seed}
        args.cache.with_suffix(".prepare.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if summary["failed"] else 0
    if args.output_dir is None:
        args.output_dir = default_run_path("episodes", args.episode_command)
    if args.output_dir.exists(): raise ValueError("output directory already exists")
    with EpisodeStore(args.cache, readonly=True) as store:
        episodes = list(store.episodes())
    if args.episode_id:
        wanted = set(args.episode_id)
        episodes = [e for e in episodes if e.episode_id in wanted]
        if {e.episode_id for e in episodes} != wanted: raise ValueError("episode ID missing from cache")
    config = EpisodeConfig.model_validate_json(args.config.read_text()) if args.config else EpisodeConfig(seed=args.seed)
    episodes.sort(key=lambda e: digest([config.seed, e.episode_id]))
    if args.count: episodes = episodes[:args.count]
    if not episodes: raise ValueError("no prepared episodes selected; run episodes prepare first")
    if args.episode_command == "run" and len(episodes) != 1:
        raise ValueError("run requires --episode-id ID or --count 1")
    runtime = RuntimeConfig.model_validate_json(args.runtime_config.read_text()) if args.runtime_config else RuntimeConfig()
    tasks = TaskStore(args.products_path)
    def factory():
        return EpisodeRunner(products_path=args.products_path, index_path=args.index_path,
            provider=provider(), config=config, runtime_config=runtime, skill_dir=args.skill_dir, task_store=tasks)
    if args.episode_command == "compare":
        result = run_comparison(episodes, factory, args.output_dir)
    else:
        run = factory().run_episode(episodes[0], baseline=args.baseline)
        write_episode_artifacts(run, args.output_dir)
        result = {"episode_id": run.trajectory.episode_id, "terminal_reason": run.trajectory.terminal_reason, "metrics": run.metrics}
    print(json.dumps({**result, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))
    return 0


def main():
    p = argparse.ArgumentParser()
    register(p.add_subparsers(dest="command", required=True))
    raise SystemExit(handle(p.parse_args()))


if __name__ == "__main__": main()
