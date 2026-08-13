"""Command-line entry points for single and batch shopping Agent runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

from .agent import (
    AgentRunner,
    AgentSkill,
    AgentSkillError,
    RolloutError,
    RolloutRunner,
    load_agent_skill,
)
from .catalog import CatalogError
from .env import DotEnvError, load_dotenv
from .harness import HarnessError, ShoppingHarness
from .providers import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DeepSeekProvider,
    ProviderError,
)
from .tasks import TaskStoreError


DEFAULT_PRODUCTS_PATH = Path("data/fine_items_eval_train_all.json")
DEFAULT_INDEX_PATH = Path("data/product_search.sqlite3")
DEFAULT_OUTPUT_ROOT = Path("outputs/rollouts")


class CLIError(RuntimeError):
    """A user-facing command configuration or output error."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shopping-agent",
        description="Run Shopping Harness agents and deterministic rollouts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    _add_common_arguments(common)

    run_parser = subparsers.add_parser(
        "run",
        parents=[common],
        help="Run one explicit task and print its complete audit result.",
    )
    run_parser.add_argument("--task-id", required=True)
    run_parser.add_argument(
        "--output",
        type=Path,
        help="Write the result JSON to a new file instead of stdout.",
    )

    rollout_parser = subparsers.add_parser(
        "rollout",
        parents=[common],
        help="Run a deterministic task sample or explicit task list.",
    )
    selection = rollout_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--split",
        choices=("train", "eval"),
        help="Dataset split from which to sample tasks.",
    )
    selection.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="Explicit task ID; repeat the option to run more than one.",
    )
    rollout_parser.add_argument(
        "--count",
        type=_positive_int,
        help="Required with --split; number of tasks to sample.",
    )
    rollout_parser.add_argument(
        "--seed",
        type=_seed,
        default=42,
        help="Deterministic sampling seed (default: 42).",
    )
    rollout_parser.add_argument(
        "--workers",
        type=_worker_count,
        default=1,
        help="Independent task workers, 1-64 (default: 1).",
    )
    rollout_parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory for rollout_<id>/ artifacts.",
    )
    rollout_parser.add_argument(
        "--rollout-id",
        help="Optional stable artifact directory name; must not already exist.",
    )
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    environment = parser.add_mutually_exclusive_group()
    environment.add_argument(
        "--env-file",
        type=Path,
        help="Load missing variables from this dotenv file.",
    )
    environment.add_argument(
        "--no-env-file",
        action="store_true",
        help="Disable automatic .env discovery.",
    )
    parser.add_argument(
        "--products-path",
        type=Path,
        default=DEFAULT_PRODUCTS_PATH,
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
    )
    parser.add_argument("--skill-dir", type=Path)
    parser.add_argument("--provider", choices=("deepseek",), default="deepseek")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--tool-choice",
        choices=("auto", "required"),
        default="auto",
        help="Provider tool policy; auto supports DeepSeek thinking mode.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=_positive_float, default=60.0)
    parser.add_argument("--max-retries", type=_non_negative_int, default=2)
    parser.add_argument("--retry-backoff", type=_non_negative_float, default=0.5)
    parser.add_argument(
        "--temperature",
        type=_finite_float,
        default=0.0,
        help="Model temperature (default: 0).",
    )
    parser.add_argument(
        "--provider-parameters",
        type=_json_object,
        default=None,
        metavar="JSON",
        help="Additional non-reserved provider parameters as a JSON object.",
    )
    parser.add_argument("--max-steps", type=_positive_int, default=20)
    parser.add_argument("--max-model-turns", type=_positive_int, default=40)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.loaded_env_file = load_dotenv(
            args.env_file,
            disabled=args.no_env_file,
        )
        if args.command == "run":
            return _run_command(args)
        if args.command == "rollout":
            return _rollout_command(args, parser)
        raise AssertionError(f"unhandled command: {args.command}")
    except (
        AgentSkillError,
        CatalogError,
        CLIError,
        DotEnvError,
        HarnessError,
        ProviderError,
        RolloutError,
        TaskStoreError,
        ValueError,
        OSError,
    ) as exc:
        print(f"shopping-agent: error: {exc}", file=sys.stderr)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())


def _run_command(args: argparse.Namespace) -> int:
    skill = load_agent_skill(args.skill_dir)
    with ShoppingHarness(
        products_path=args.products_path,
        index_path=args.index_path,
        default_max_steps=args.max_steps,
    ) as harness:
        result = AgentRunner(
            provider=_build_provider(args),
            harness=harness,
            skill=skill,
            max_model_turns=args.max_model_turns,
            tool_choice=args.tool_choice,
        ).run(args.task_id, max_steps=args.max_steps)
    payload = result.model_dump_json(indent=2)
    if args.output is None:
        print(payload)
    else:
        _write_new_file(args.output, payload + "\n")
        print(args.output.expanduser().resolve())
    return 0


def _rollout_command(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    if args.split is not None and args.count is None:
        parser.error("--count is required when using --split")
    if args.task_ids is not None and args.count is not None:
        parser.error("--count can only be used with --split")

    skill = load_agent_skill(args.skill_dir)
    with ShoppingHarness(
        products_path=args.products_path,
        index_path=args.index_path,
        default_max_steps=args.max_steps,
    ) as harness:
        def runner_factory() -> AgentRunner:
            return AgentRunner(
                provider=_build_provider(args),
                harness=harness,
                skill=skill,
                max_model_turns=args.max_model_turns,
                tool_choice=args.tool_choice,
            )

        rollout = RolloutRunner(
            harness=harness,
            runner_factory=runner_factory,
        )
        metadata = _rollout_metadata(args, harness.manifest, skill)
        if args.split is not None:
            report = rollout.run_split(
                args.split,
                args.count,
                seed=args.seed,
                workers=args.workers,
                max_steps=args.max_steps,
                output_root=args.output_root,
                rollout_id=args.rollout_id,
                metadata=metadata,
            )
        else:
            report = rollout.run_tasks(
                args.task_ids,
                workers=args.workers,
                max_steps=args.max_steps,
                output_root=args.output_root,
                rollout_id=args.rollout_id,
                metadata=metadata,
            )

    output = {
        "rollout_id": report.rollout_id,
        "output_dir": str(report.output_dir),
        "summary": report.summary.model_dump(mode="json"),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def _build_provider(args: argparse.Namespace) -> DeepSeekProvider:
    parameters = dict(args.provider_parameters or {})
    parameters.setdefault("temperature", args.temperature)
    return DeepSeekProvider(
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        default_parameters=parameters,
    )


def _rollout_metadata(
    args: argparse.Namespace,
    harness_manifest: dict[str, Any],
    skill: AgentSkill,
) -> dict[str, Any]:
    parameters = dict(args.provider_parameters or {})
    parameters.setdefault("temperature", args.temperature)
    return {
        "provider": args.provider,
        "model": args.model,
        "base_url": args.base_url,
        "provider_parameters": _redact_parameters(parameters),
        "max_model_turns": args.max_model_turns,
        "tool_choice": args.tool_choice,
        "skill_name": skill.name,
        "skill_sha256": skill.source_sha256,
        "harness_manifest": harness_manifest,
        "env_file": (
            str(args.loaded_env_file) if args.loaded_env_file is not None else None
        ),
    }


def _write_new_file(path: Path, text: str) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        with resolved.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise CLIError(f"refusing to overwrite output file: {resolved}") from exc


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _worker_count(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 64:
        raise argparse.ArgumentTypeError("must be between 1 and 64")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _seed(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return parsed


def _redact_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    sensitive_fragments = ("authorization", "key", "password", "secret", "token")
    return {
        key: (
            "***"
            if any(fragment in key.lower() for fragment in sensitive_fragments)
            else value
        )
        for key, value in parameters.items()
    }


if __name__ == "__main__":
    entrypoint()
