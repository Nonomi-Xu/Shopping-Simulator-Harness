"""Command-line entry points for single and batch shopping Agent runs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Sequence

from .agent import (
    AgentRunner,
    AgentSkill,
    AgentSkillError,
    RolloutError,
    RolloutRunner,
    load_agent_skill,
    AgentProfile,
    UserProfile,
    RuntimeConfig,
    AgentRunResult,
    UserReply,
)
from .agent.artifacts import replay_run, write_run_artifacts
from .agent.audit import RunAudit
from .paths import HARNESS_DIR, OUTPUTS_DIR, result_path
from .catalog import CatalogError
from .env import DotEnvError, load_dotenv
from .harness import HarnessError, ShoppingHarness
from .providers import (
    DEFAULT_BASE_URLS,
    DEFAULT_MODELS,
    DEFAULT_TIMEOUTS,
    PROVIDER_NAMES,
    DeepSeekProvider,
    OllamaProvider,
    ProviderError,
)
from .tasks import TaskStoreError
from .sources import JsonProductSource, HttpProductSource


DEFAULT_PRODUCTS_PATH = HARNESS_DIR / "data/fine_items_eval_train_all.json"
DEFAULT_INDEX_PATH = HARNESS_DIR / "data/product_search.sqlite3"
DEFAULT_OUTPUT_ROOT = OUTPUTS_DIR / "rollouts"


class CLIError(RuntimeError):
    """A user-facing command configuration or output error."""


def load_branch():
    """Return the optional interaction branch, or ``None`` when it is absent.

    The branch (``interactive`` / ``episodes`` / ``experiments`` / long-term
    preferences) is deliberately not part of the offline Test1 mainline.
    Importing it here is the only mainline -> branch edge, and it is optional:
    deleting ``src/harness/branch/`` leaves every offline command working and
    simply removes the branch subcommands from the CLI.
    """
    try:
        from .branch.episodes import cli as episodes_cli
        from .branch.experiments import cli as experiments_cli
        from .branch.interactive import InteractiveHarness
        from .branch.preferences import PreferenceStore
    except ImportError:
        return None
    return SimpleNamespace(
        episodes_cli=episodes_cli,
        experiments_cli=experiments_cli,
        InteractiveHarness=InteractiveHarness,
        PreferenceStore=PreferenceStore,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shopping-agent",
        description="Run Shopping Harness agents and deterministic rollouts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    branch = load_branch()
    if branch is not None:
        branch.episodes_cli.register(subparsers)
        branch.experiments_cli.register(subparsers)
    common = argparse.ArgumentParser(add_help=False)
    _add_common_arguments(common)

    run_parser = subparsers.add_parser(
        "run",
        parents=[common],
        help="Run one explicit task and print its complete audit result.",
    )
    run_parser.add_argument("--task-id", required=True)
    run_parser.add_argument(
        "--artifacts-dir",
        type=result_path,
        help="New directory for the versioned audit bundle.",
    )
    run_parser.add_argument(
        "--output",
        type=result_path,
        help="Write the result JSON to a new file instead of stdout.",
    )

    if branch is not None:
        chat_parser = subparsers.add_parser(
            "chat",
            parents=[common],
            help="Start an unlabelled interactive shopping task.",
        )
        chat_parser.add_argument("--query", required=True)
        _add_interactive_arguments(chat_parser)
        chat_parser.add_argument("--preferences-db", type=Path)
        chat_parser.add_argument("--profile-id")
        resume_parser = subparsers.add_parser(
            "resume",
            parents=[common],
            help="Resume a waiting interactive audit bundle with a real user reply.",
        )
        resume_parser.add_argument("--run-dir", type=result_path, required=True)
        resume_parser.add_argument("--reply", required=True)
        resume_parser.add_argument(
            "--replace-constraint-id",
            action="append",
            default=[],
            dest="replace_constraint_ids",
        )
        _add_interactive_arguments(resume_parser)

        prefs = subparsers.add_parser(
            "preferences",
            help="Explicitly view, set or remove long-term user preferences.",
        )
        prefs.add_argument("operation", choices=("show", "set", "remove", "history"))
        prefs.add_argument("--db", type=Path, required=True)
        prefs.add_argument("--profile-id", required=True)
        prefs.add_argument("--name")
        prefs.add_argument("--value")
        prefs.add_argument("--expected-revision", type=_non_negative_int)

    rollout_parser = subparsers.add_parser(
        "rollout",
        parents=[common],
        help="Run a deterministic task sample or explicit task list.",
    )
    selection = rollout_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--split",
        choices=("Original", "Test1", "all", "test"),
        help="all: every task; test: fixed smoke-test subset.",
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
        help="Optional sample size; omit to run the entire selected dataset.",
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
        type=result_path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent of unique rollout directories (relative to Shopping-Simulator repository).",
    )
    rollout_parser.add_argument(
        "--rollout-id",
        help="Optional stable artifact directory name; must not already exist.",
    )
    replay_parser = subparsers.add_parser(
        "replay", help="Validate an audit bundle and reconstruct public state offline."
    )
    replay_parser.add_argument("--run-dir", type=result_path, required=True)
    subparsers.add_parser(
        "schema", help="Print versioned runtime JSON schemas without starting a model."
    )
    vector_parser = subparsers.add_parser(
        "vector-index", help="Build a new semantic index from public product fields."
    )
    vector_source = vector_parser.add_mutually_exclusive_group(required=True)
    vector_source.add_argument("--source-json", type=Path)
    vector_source.add_argument("--index-path", type=Path)
    vector_parser.add_argument("--output", type=Path, required=True)
    vector_parser.add_argument("--batch-size", type=_positive_int, default=64)
    vector_parser.add_argument("--env-file", type=Path)
    vector_parser.add_argument("--no-env-file", action="store_true")
    _add_embedding_arguments(vector_parser)
    return parser


def _add_interactive_arguments(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--source-json", type=Path, help="Public ProductDetail JSON array."
    )
    source.add_argument(
        "--source-url", help="Configured public product gateway base URL."
    )
    parser.add_argument("--source-id", default="public-products")
    parser.add_argument("--currency", help="Explicit source currency, e.g. CNY.")
    parser.add_argument(
        "--source-key-env",
        help="Environment variable holding the gateway bearer credential.",
    )
    parser.add_argument("--source-timeout", type=_positive_float, default=15.0)
    parser.add_argument(
        "--artifacts-dir",
        type=result_path,
        required=True,
        help="New audit directory; waiting runs include checkpoint.json.",
    )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--agent-mode", choices=("single", "multi"), default=None,
        help="multi uses planner/executor/observer; new chat tasks default to multi.",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        help="RuntimeConfig JSON: action/search/token/time/context budgets.",
    )
    parser.add_argument(
        "--agent-profile", type=Path, help="AgentProfile JSON: identity and objective."
    )
    parser.add_argument(
        "--user-profile",
        type=Path,
        help="UserProfile JSON: explicit background preferences.",
    )
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
    parser.add_argument(
        "--provider",
        choices=("deepseek", "ollama"),
        default="deepseek",
        help=(
            "Model backend: 'deepseek' calls the paid remote API, 'ollama' calls "
            "a local OpenAI-compatible server at no per-token cost (default: deepseek)."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name; defaults to the selected provider's default.",
    )
    parser.add_argument(
        "--tool-choice",
        choices=("auto", "required"),
        default="auto",
        help="Provider tool policy; auto supports DeepSeek thinking mode.",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--timeout", type=_positive_float, default=None)
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


def _add_embedding_arguments(parser):
    parser.add_argument("--embedding-model", help="Embedding model, or EMBEDDING_MODEL.")
    parser.add_argument(
        "--embedding-base-url",
        help="Embedding API base URL ending in /v1, or EMBEDDING_BASE_URL.",
    )
    parser.add_argument("--embedding-key-env", default="EMBEDDING_API_KEY")
    parser.add_argument("--embedding-timeout", type=_positive_float, default=30.0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "provider"):
        _resolve_provider_defaults(args)
    try:
        if args.command == "schema":
            from .agent import contracts

            names = (
                "AgentProfile",
                "UserProfile",
                "TaskProfile",
                "SearchPlan",
                "PlanUpdate",
                "EvidenceAssessment",
                "EvidenceBatch",
                "ReadEvidence",
                "CompareProducts",
                "VerifySelection",
                "ActionProposal",
                "AgentSubmission",
                "WorkingMemory",
                "AgentAction",
                "AgentObservation",
                "RuntimeConfig",
                "AgentEvent",
                "AgentRunResult",
                "AskUser",
                "UserReply",
                "UserMessage",
                "PendingQuestion",
                "ReviewProgress",
            )
            print(
                json.dumps(
                    {
                        name: getattr(contracts, name).model_json_schema()
                        for name in names
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command in {"episodes", "experiments", "preferences", "chat", "resume"}:
            # Only reachable when the optional branch registered these subcommands.
            branch = load_branch()
            if branch is None:
                raise CLIError(
                    f"command {args.command!r} requires src/harness/branch/, which is absent"
                )
            if args.command == "episodes":
                return branch.episodes_cli.handle(args)
            if args.command == "experiments":
                return branch.experiments_cli.handle(args)
            if args.command == "preferences":
                return _preferences_command(args, branch)
            load_dotenv(args.env_file, disabled=args.no_env_file)
            return _interactive_command(args, branch)
        if args.command == "replay":
            print(json.dumps(replay_run(args.run_dir), ensure_ascii=False, indent=2))
            return 0
        if args.command == "vector-index":
            load_dotenv(args.env_file, disabled=args.no_env_file)
            return _vector_index_command(args)
        args.runtime_config = _load_contract(args.runtime_config, RuntimeConfig)
        if args.agent_mode is not None:
            args.runtime_config = args.runtime_config.model_copy(
                update={"agent_mode": args.agent_mode}
            )
        args.agent_profile = _load_contract(args.agent_profile, AgentProfile)
        args.user_profile = _load_contract(args.user_profile, UserProfile)
        if args.runtime_config.interaction_mode != "offline":
            raise CLIError(
                "run/rollout use offline evaluation; use chat for interactive tasks"
            )
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


def _preferences_command(args, branch):
    if args.operation in {"set", "remove"} and not args.name:
        raise CLIError("--name is required for set/remove")
    if args.operation == "set" and args.value is None:
        raise CLIError("--value is required for set")
    store = branch.PreferenceStore(args.db)
    if args.operation == "set":
        store.set(
            args.profile_id,
            args.name,
            args.value,
            expected_revision=args.expected_revision,
        )
    elif args.operation == "remove":
        store.remove(
            args.profile_id, args.name, expected_revision=args.expected_revision
        )
    if args.operation == "history":
        payload = store.history(args.profile_id)
    else:
        profile, revision = store.load(args.profile_id)
        payload = {"profile": profile.model_dump(mode="json"), "revision": revision}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _interactive_command(args, branch):
    if args.artifacts_dir.expanduser().exists():
        raise CLIError("refusing to overwrite existing audit directory")
    paused = None
    if args.command == "resume":
        replay_run(args.run_dir)
        paused = AgentRunResult.model_validate_json(
            (args.run_dir / "checkpoint.json").read_text(encoding="utf-8")
        )
        config = (
            _load_contract(args.runtime_config, RuntimeConfig)
            if args.runtime_config
            else RuntimeConfig.model_validate(paused.manifest["config"])
        )
        profile = (
            _load_contract(args.agent_profile, AgentProfile)
            if args.agent_profile
            else paused.profile
        )
        user = (
            _load_contract(args.user_profile, UserProfile)
            if args.user_profile
            else paused.user_profile
        )
        max_steps = paused.session.state.max_steps
        args.max_model_turns = paused.manifest["max_model_turns"]
        # Restore the backend that produced the checkpoint so a resume keeps
        # using the same model, endpoint, and provider identity.
        args.provider = paused.manifest.get("provider", args.provider)
        args.model = paused.model
        args.base_url = paused.manifest.get("base_url", args.base_url)
        _resolve_provider_defaults(args)
        args.tool_choice = paused.manifest["tool_choice"]
        if args.provider_parameters is None:
            if _has_redacted_value(paused.manifest["provider_parameters"]):
                raise CLIError(
                    "checkpoint contains redacted provider parameters; supply original --provider-parameters"
                )
            args.provider_parameters = paused.manifest["provider_parameters"]
    else:
        config = _load_contract(args.runtime_config, RuntimeConfig).model_copy(
            update={"interaction_mode": "interactive"}
        )
        profile = _load_contract(args.agent_profile, AgentProfile)
        if args.preferences_db:
            if not args.profile_id or args.user_profile:
                raise CLIError(
                    "--preferences-db requires --profile-id and cannot be combined with --user-profile"
                )
            user, _ = branch.PreferenceStore(args.preferences_db).load(args.profile_id)
        else:
            if args.profile_id:
                raise CLIError("--profile-id requires --preferences-db")
            user = _load_contract(args.user_profile, UserProfile)
        max_steps = args.max_steps
    if args.agent_mode is not None:
        config = config.model_copy(update={"agent_mode": args.agent_mode})
    if args.source_key_env and not os.environ.get(args.source_key_env):
        raise CLIError("configured source credential environment variable is empty")
    # The public search contract exposes BM25 only: HttpProductSource validates a
    # finite bm25_score on every returned item and has no vector transport.
    source = (
        JsonProductSource(
            args.source_json, source_id=args.source_id, currency=args.currency,
        )
        if args.source_json
        else HttpProductSource(
            args.source_url,
            source_id=args.source_id,
            currency=args.currency,
            api_key=(
                os.environ.get(args.source_key_env) if args.source_key_env else None
            ),
            timeout=args.source_timeout,
        )
    )
    with branch.InteractiveHarness(source, default_max_steps=max_steps) as harness:
        runner = AgentRunner(
            provider=_build_provider(args),
            harness=harness,
            skill_dir=args.skill_dir,
            config=config,
            profile=profile,
            user_profile=user,
            max_model_turns=args.max_model_turns,
            tool_choice=args.tool_choice,
        )
        result = (
            runner.resume(
                paused,
                UserReply(
                    text=args.reply, replace_constraint_ids=args.replace_constraint_ids
                ),
            )
            if paused
            else runner.run_query(args.query, max_steps=max_steps)
        )
        directory = write_run_artifacts(result, args.artifacts_dir)
    recommendations = next(
        (
            step.arguments["recommendations"]
            for step in reversed(result.session.history)
            if step.tool_name == "submit_recommendations" and step.status == "success"
        ),
        [],
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "terminal_reason": result.terminal_reason,
                "artifacts_dir": str(directory),
                "session_id": result.session_id,
                "pending_question": (
                    result.pending_question.model_dump(mode="json")
                    if result.pending_question
                    else None
                ),
                "recommendations": recommendations,
                "evaluation": None,
                "error": result.error_message,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.status in {"completed", "waiting_for_user"} else 1


def _run_command(args: argparse.Namespace) -> int:
    for output in (args.output, args.artifacts_dir):
        if output is not None and output.expanduser().exists():
            raise CLIError(f"refusing to overwrite existing output: {output}")
    skill = load_agent_skill(args.skill_dir)
    audit = RunAudit(args.artifacts_dir) if args.artifacts_dir is not None else None
    provider = _build_provider(args)
    if audit is not None:
        provider.request_observer = audit.transport
    with ShoppingHarness(
        products_path=args.products_path,
        index_path=args.index_path,
        default_max_steps=args.max_steps,
    ) as harness:
        result = AgentRunner(
            provider=provider,
            harness=harness,
            skill=skill,
            max_model_turns=args.max_model_turns,
            tool_choice=args.tool_choice,
            config=args.runtime_config,
            profile=args.agent_profile,
            user_profile=args.user_profile,
            event_observer=audit,
        ).run(args.task_id, max_steps=args.max_steps)
    if audit is not None:
        audit.finish(result)
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
                config=args.runtime_config,
                profile=args.agent_profile,
                user_profile=args.user_profile,
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


def _resolve_provider_defaults(args: argparse.Namespace) -> None:
    """Fill provider-dependent defaults once ``--provider`` is known.

    ``--model``, ``--base-url`` and ``--timeout`` default to ``None`` so the
    selected backend supplies its own values.  Anything already set on ``args``
    (an explicit flag, or a value restored from a checkpoint) is preserved.

    A checkpoint may name a provider injected through the Python API rather
    than one of the CLI backends, so unknown names fall back to the historical
    DeepSeek defaults instead of failing; ``_provider_class`` enforces the
    whitelist when a client is actually built.
    """
    fallbacks = {
        "model": DEFAULT_MODELS.get(args.provider, DEFAULT_MODELS["deepseek"]),
        "base_url": DEFAULT_BASE_URLS.get(args.provider, DEFAULT_BASE_URLS["deepseek"]),
        "timeout": DEFAULT_TIMEOUTS.get(args.provider, DEFAULT_TIMEOUTS["deepseek"]),
    }
    for name, value in fallbacks.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)


def _provider_class(name: str):
    """Resolve a backend class; module-level names keep patching in tests working."""
    if name == "ollama":
        return OllamaProvider
    if name == "deepseek":
        return DeepSeekProvider
    raise CLIError(
        f"unknown provider {name!r}; expected one of {', '.join(PROVIDER_NAMES)}"
    )


def _build_provider(args: argparse.Namespace):
    parameters = dict(args.provider_parameters or {})
    parameters.setdefault("temperature", args.temperature)
    return _provider_class(args.provider)(
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        default_parameters=parameters,
    )


def _build_embedding_provider(args):
    from ..data.search_vector import OpenAIEmbeddingProvider

    model = args.embedding_model or os.environ.get("EMBEDDING_MODEL")
    base_url = args.embedding_base_url or os.environ.get("EMBEDDING_BASE_URL")
    if not model or not base_url:
        raise CLIError(
            "vector search requires --embedding-model and --embedding-base-url "
            "(or EMBEDDING_MODEL and EMBEDDING_BASE_URL); use the index's original model"
        )
    return OpenAIEmbeddingProvider(
        model=model, base_url=base_url,
        api_key=os.environ.get(args.embedding_key_env), timeout=args.embedding_timeout,
    )


def _vector_index_command(args):
    from ..data.search_vector import build_vector_index, build_json_vector_index

    if args.output.expanduser().exists():
        raise CLIError("refusing to overwrite an existing vector index")
    provider = _build_embedding_provider(args)
    builder = build_json_vector_index if args.source_json else build_vector_index
    manifest = builder(
        args.source_json or args.index_path, args.output, provider,
        batch_size=args.batch_size,
    )
    print(json.dumps({"output": str(args.output.resolve()), "manifest": manifest},
                     ensure_ascii=False, indent=2))
    return 0


def _has_redacted_value(value):
    if isinstance(value, dict):
        return any(_has_redacted_value(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_redacted_value(v) for v in value)
    return value == "***"


def _rollout_metadata(
    args: argparse.Namespace,
    harness_manifest: dict[str, Any],
    skill: AgentSkill,
) -> dict[str, Any]:
    parameters = dict(args.provider_parameters or {})
    parameters.setdefault("temperature", args.temperature)
    return {
        "runtime_config": args.runtime_config.model_dump(mode="json"),
        "agent_profile": args.agent_profile.model_dump(mode="json"),
        "user_profile": args.user_profile.model_dump(mode="json"),
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


def _load_contract(path, model):
    if path is None:
        return model()
    return model.model_validate_json(path.expanduser().read_text(encoding="utf-8"))


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
