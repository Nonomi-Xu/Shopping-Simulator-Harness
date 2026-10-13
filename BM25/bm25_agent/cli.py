from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .paths import PROJECT_DIR, OUTPUTS_DIR, result_path, new_run_id
from .catalog import BM25Catalog
from .env import load_dotenv
from .pipeline import (
    BM25AgentPipeline,
    compact_result,
    evaluation_summary_payloads,
    retrieval_summary,
)
from .provider import DEFAULT_BASE_URL, DEFAULT_MODEL, DeepSeekSelector
from .tasks import TaskStore


DEFAULT_PRODUCTS = PROJECT_DIR / "harness/data/fine_items_eval_train_all.json"
DEFAULT_INDEX = PROJECT_DIR / "harness/data/product_search.sqlite3"


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bm25-agent",
        description="Evaluate BM25 Top-5 retrieval plus one constrained Agent decision.",
    )
    parser.add_argument("--products-path", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX)
    subparsers = parser.add_subparsers(dest="command", required=True)

    recall = subparsers.add_parser(
        "recall", help="Evaluate the BM25 Top-5 retrieval ceiling without an API call."
    )
    _add_selection(recall)

    run = subparsers.add_parser("run", help="Run one task through BM25 and the Agent.")
    run.add_argument("--task-id", required=True)
    _add_provider(run)

    evaluate = subparsers.add_parser("evaluate", help="Run and persist a batch evaluation.")
    _add_selection(evaluate)
    _add_provider(evaluate)
    evaluate.add_argument("--workers", type=_positive_int, default=1)
    evaluate.add_argument(
        "--output-root", "--output-dir", dest="output_dir", type=result_path,
        default=OUTPUTS_DIR / "evaluations",
        help="Parent of unique run directories (absolute path required; --output-dir is an alias).",
    )
    evaluate.add_argument(
        "--resume-dir",
        type=result_path,
        help="Resume an interrupted evaluation directory using its run_config.json.",
    )
    return parser


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", choices=("Original", "Test1", "all", "test"), default="all")
    parser.add_argument("--count", type=_positive_int,
                        help="Optional sample size; omit to run the entire selection.")
    parser.add_argument("--seed", type=_seed, default=42)


def _seed(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _add_provider(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--env-file", type=Path)


def _selected_ids(args: argparse.Namespace, tasks: TaskStore) -> list[str]:
    if args.count is None:
        return tasks.list_task_ids(args.split)
    return tasks.sample_task_ids(args.split, args.count, args.seed)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _iter_run_payloads(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid checkpoint JSON at {path}:{line_number}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"invalid checkpoint record at {path}:{line_number}")
            yield payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tasks = TaskStore(args.products_path)
        with BM25Catalog(args.index_path) as catalog:
            if args.command == "recall":
                summary = retrieval_summary(tasks, catalog, _selected_ids(args, tasks))
                print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
                return 0

            load_dotenv(args.env_file)
            selector = DeepSeekSelector(
                model=args.model,
                base_url=args.base_url,
                timeout=args.timeout,
                max_retries=args.max_retries,
                temperature=args.temperature,
            )
            pipeline = BM25AgentPipeline(tasks=tasks, catalog=catalog, selector=selector)
            if args.command == "run":
                result = pipeline.run(args.task_id)
                print(
                    json.dumps(
                        compact_result(result),
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0 if result.status == "completed" else 1

            task_ids = _selected_ids(args, tasks)
            configuration = {
                "split": args.split,
                "task_ids": task_ids,
                "model": args.model,
                "base_url": args.base_url,
                "temperature": args.temperature,
                "products_path": str(args.products_path.expanduser().resolve()),
                "index_path": str(args.index_path.expanduser().resolve()),
            }
            if args.resume_dir is None:
                output_dir = (
                    args.output_dir.expanduser().resolve()
                    / new_run_id(args.split)
                )
                output_dir.mkdir(parents=True, exist_ok=False)
                _write_json(output_dir / "run_config.json", configuration)
            else:
                output_dir = args.resume_dir.expanduser().resolve()
                saved_configuration = json.loads(
                    (output_dir / "run_config.json").read_text(encoding="utf-8")
                )
                if saved_configuration != configuration:
                    raise ValueError("resume configuration does not match run_config.json")

            print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False),
                  file=sys.stderr, flush=True)
            runs_path = output_dir / "runs.jsonl"
            completed_ids = [
                payload.get("task", {}).get("task_id")
                for payload in _iter_run_payloads(runs_path)
            ]
            if len(completed_ids) != len(set(completed_ids)):
                raise ValueError("checkpoint contains duplicate task IDs")
            unexpected = set(completed_ids) - set(task_ids)
            if unexpected:
                raise ValueError("checkpoint contains tasks outside this selection")
            pending_ids = [task_id for task_id in task_ids if task_id not in set(completed_ids)]
            completed_count = len(completed_ids)
            if pending_ids:
                with runs_path.open("a", encoding="utf-8") as handle:
                    for _, result in pipeline.run_many_iter(
                        pending_ids,
                        workers=args.workers,
                    ):
                        handle.write(
                            json.dumps(
                                result.to_dict(),
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        handle.flush()
                        completed_count += 1
                        if completed_count % 100 == 0 or completed_count == len(task_ids):
                            print(
                                json.dumps(
                                    {
                                        "checkpoint": str(runs_path),
                                        "completed": completed_count,
                                        "total": len(task_ids),
                                    },
                                    ensure_ascii=False,
                                ),
                                file=sys.stderr,
                                flush=True,
                            )

            summary = evaluation_summary_payloads(_iter_run_payloads(runs_path))
            if summary["task_count"] != len(task_ids):
                raise ValueError(
                    f"checkpoint has {summary['task_count']} records; "
                    f"expected {len(task_ids)}"
                )
            summary["dataset_selection"] = {
                "split": args.split,
                "available_count": len(tasks.list_task_ids(args.split)),
                "selected_count": len(task_ids),
                "seed": args.seed if args.count is not None else None,
                "task_ids": task_ids,
            }
            _write_json(output_dir / "summary.json", summary)
            print(
                json.dumps(
                    {"output_dir": str(output_dir), "summary": summary},
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0 if summary["failed_count"] == 0 else 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
