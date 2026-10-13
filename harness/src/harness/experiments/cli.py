"""CLI for explicit experiment execution and offline statistical reanalysis."""
import json
import os
from pathlib import Path
import sys

from ..agent.events import digest
from ..paths import HARNESS_DIR, result_path, default_run_path
from ..env import load_dotenv
from ..episodes.store import EpisodeStore
from ..providers import DeepSeekProvider
from ..tasks import TaskStore
from .models import ExperimentConfig
from .report import build_report, write_report
from .runner import Case, load_completed, run_experiment


def register(parent):
    parser = parent.add_parser("experiments", help="Paired model/configuration matrix and statistical reports")
    sub = parser.add_subparsers(dest="experiment_command", required=True)
    run = sub.add_parser("run", help="Execute the matrix; calls the configured model providers")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--products-path", type=Path, default=HARNESS_DIR / "data/fine_items_eval_train_all.json")
    run.add_argument("--index-path", type=Path, default=HARNESS_DIR / "data/product_search.sqlite3")
    run.add_argument("--cache", type=Path)
    run.add_argument("--split", choices=("Original", "Test1", "all", "test"), default="test")
    run.add_argument("--count", type=int, default=10, help="fixed case count; 0 selects all")
    selected = run.add_mutually_exclusive_group()
    selected.add_argument("--task-id", action="append")
    selected.add_argument("--episode-id", action="append")
    run.add_argument("--output-dir", type=result_path, help="Exact new directory, relative to Shopping-Simulator repository; default: unique outputs directory.")
    run.add_argument("--env-file", type=Path)
    run.add_argument("--no-env-file", action="store_true")
    report = sub.add_parser("report", help="Verify completed journal and recompute statistics offline")
    report.add_argument("--run-dir", type=result_path, required=True)
    report.add_argument("--output-dir", type=result_path, help="Exact new directory, relative to Shopping-Simulator repository; default: unique outputs directory.")


def select_cases(args, config):
    if args.count < 0:
        raise ValueError("count must be nonnegative")
    if config.mode == "offline":
        if args.cache or args.episode_id:
            raise ValueError("cache/episode-id require episode mode")
        tasks = TaskStore(args.products_path)
        ids = args.task_id if args.task_id else tasks.list_task_ids(args.split)
        if args.task_id:
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate task ID")
            for task_id in ids:
                tasks.get_public_task(task_id)
        cases = [Case(i) for i in ids]
    else:
        if not args.cache or args.task_id:
            raise ValueError("episode mode requires --cache and optional --episode-id")
        with EpisodeStore(args.cache, readonly=True) as store:
            episodes = list(store.episodes())
        if args.episode_id:
            wanted = set(args.episode_id)
            if len(wanted) != len(args.episode_id):
                raise ValueError("duplicate episode ID")
            episodes = [e for e in episodes if e.episode_id in wanted]
            if {e.episode_id for e in episodes} != wanted:
                raise ValueError("episode ID missing from cache")
        # Reject ambiguous cache versions before sampling can conceal the ambiguity.
        if len({e.task_id for e in episodes}) != len(episodes):
            raise ValueError("cache contains multiple versions per task; select explicit episode IDs")
        tasks = TaskStore(args.products_path)
        if not args.episode_id:
            split_ids = set(tasks.list_task_ids(args.split))
            episodes = [e for e in episodes if e.task_id in split_ids]
        cases = [Case(e.task_id, e) for e in episodes]
    cases.sort(key=lambda c: digest([config.selection_seed, c.task_id]))
    # Explicit IDs always mean exactly the supplied set, irrespective of default count.
    if args.count and not (args.task_id or args.episode_id):
        cases = cases[:args.count]
    if not cases:
        raise ValueError("no cases selected")
    return cases


def handle(args):
    if args.output_dir is None:
        category = "reports" if args.experiment_command == "report" else "experiments"
        args.output_dir = default_run_path(category, args.experiment_command)
    if args.output_dir.exists():
        raise ValueError("output directory already exists")
    if args.experiment_command == "report":
        config, records = load_completed(args.run_dir)
        report = build_report(records, config)
        args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        write_report(report, records, args.output_dir)
    else:
        config = ExperimentConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        cases = select_cases(args, config)
        load_dotenv(args.env_file, disabled=args.no_env_file)
        for variant in config.variants:
            if not os.environ.get(variant.api_key_env):
                raise ValueError(f"missing credential environment variable: {variant.api_key_env}")

        def factory(variant, seed):
            return DeepSeekProvider(model=variant.model, base_url=variant.base_url,
                api_key=os.environ[variant.api_key_env], default_parameters=variant.parameters(seed))

        def progress(done, total, row):
            print(f"[{done}/{total}] {row['variant']} {row['task_id']} "
                  f"repeat={row['repeat_seed']} {row['terminal_reason']}", file=sys.stderr)

        report = run_experiment(config, cases, products_path=args.products_path, index_path=args.index_path,
            output_dir=args.output_dir, provider_factory=factory, progress=progress)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "reference": report["reference"],
                      "variants": {k: {"runs": v["runs"], "tasks": v["tasks"],
                                         "strict_success": v["metrics"]["strict_success"]}
                                   for k, v in report["variants"].items()}}, ensure_ascii=False, indent=2))
    # A finished experiment can legitimately contain model failures and zero successes.
    return 0
