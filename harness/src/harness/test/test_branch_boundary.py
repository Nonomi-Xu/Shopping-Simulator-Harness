"""Enforce the one-way boundary between the Test1 mainline and the branch.

Rule: ``src.harness.branch`` may import ``src.harness``; ``src.harness`` must
never require ``src.harness.branch``. The only sanctioned mainline -> branch
edge is :func:`src.harness.cli.load_branch`, which is optional and tolerates
the branch being absent.

The dynamic check runs in a fresh interpreter with ``src.harness.branch``
blocked at the import machinery level, which is equivalent to deleting the
directory: every offline module must still import and every offline CLI
subcommand must still be registered.
"""

import re
import subprocess
import sys
import unittest
from pathlib import Path

MAINLINE_ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = MAINLINE_ROOT.parents[1]
BRANCH_ROOT = MAINLINE_ROOT / "branch"
BRANCH_SUBCOMMANDS = {"episodes", "experiments", "chat", "resume", "preferences"}
OFFLINE_SUBCOMMANDS = {"run", "rollout", "replay", "schema", "vector-index"}

# Match import syntax only, so prose in comments and docstrings is ignored.
BRANCH_IMPORT_PATTERNS = (
    re.compile(r"^\s*from\s+[\w.]*\bbranch\b[\w.]*\s+import\b", re.MULTILINE),
    re.compile(r"^\s*import\s+[\w.]*\bbranch\b", re.MULTILINE),
    re.compile(r"import_module\(\s*[\"'][\w.]*\bbranch\b"),
)

# Scan the mainline by naming its packages, so a renamed or relocated branch
# directory can never be mistaken for mainline code.
MAINLINE_PACKAGES = ("agent", "api", "benchmark", "providers", "test")


def _mainline_sources():
    for path in sorted(MAINLINE_ROOT.glob("*.py")):
        yield path
    for package in MAINLINE_PACKAGES:
        root = MAINLINE_ROOT / package
        if root.is_dir():
            for path in sorted(root.rglob("*.py")):
                if "__pycache__" not in path.parts:
                    yield path

_BLOCKED_IMPORT_PROBE = r'''
import importlib
import sys

if __name__ == "__main__":
    for name in [m for m in list(sys.modules) if m.startswith("src.harness")]:
        del sys.modules[name]
    sys.path.insert(0, ".")

    class BlockBranch:
        """Simulate an absent src/harness/branch/ directory."""

        def find_spec(self, name, path=None, target=None):
            if name == "src.harness.branch" or name.startswith("src.harness.branch."):
                raise ImportError("blocked for boundary test: %s" % name)
            return None

    sys.meta_path.insert(0, BlockBranch())

    MAINLINE_MODULES = [
        "src.harness.catalog",
        "src.harness.cli",
        "src.harness.constraints",
        "src.harness.env",
        "src.harness.evaluator",
        "src.harness.harness",
        "src.harness.paths",
        "src.harness.schemas",
        "src.harness.session",
        "src.harness.sources",
        "src.harness.tasks",
        "src.harness.agent",
        "src.harness.agent.orchestrator",
        "src.harness.agent.rollout",
        "src.harness.agent.artifacts",
        "src.harness.agent.audit",
        "src.harness.benchmark.runner",
        "src.harness.benchmark.report",
        "src.harness.benchmark.datasets",
        "src.harness.benchmark.retrieval",
        "src.harness.benchmark.annotations",
        "src.harness.providers",
        "src.harness.providers.deepseek",
    ]
    for module in MAINLINE_MODULES:
        importlib.import_module(module)

    from src.harness.cli import build_parser, load_branch

    if load_branch() is not None:
        raise AssertionError("load_branch() must report the branch as absent")

    parser = build_parser()
    choices = set(parser._subparsers._group_actions[0].choices)  # noqa: SLF001

    missing = OFFLINE_SUBCOMMANDS - choices
    if missing:
        raise AssertionError("offline subcommands missing: %s" % sorted(missing))
    leaked = choices & BRANCH_SUBCOMMANDS
    if leaked:
        raise AssertionError("branch subcommands leaked: %s" % sorted(leaked))

    print("MAINLINE_CLEAN")
'''.replace("OFFLINE_SUBCOMMANDS", repr(OFFLINE_SUBCOMMANDS)).replace(
    "BRANCH_SUBCOMMANDS", repr(BRANCH_SUBCOMMANDS)
)


class BranchBoundaryTest(unittest.TestCase):
    def test_mainline_imports_without_branch(self) -> None:
        """Offline modules import and offline CLI commands exist with no branch."""
        completed = subprocess.run(
            [sys.executable, "-c", _BLOCKED_IMPORT_PROBE],
            cwd=str(HARNESS_DIR),
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn("MAINLINE_CLEAN", completed.stdout)

    def test_only_cli_references_the_branch(self) -> None:
        """No mainline module except cli.load_branch() imports the branch package."""
        offenders = []
        for path in _mainline_sources():
            if path.name == "cli.py" or path == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8")
            if any(pattern.search(text) for pattern in BRANCH_IMPORT_PATTERNS):
                offenders.append(str(path.relative_to(MAINLINE_ROOT)))
        self.assertEqual(offenders, [], "mainline modules importing the branch")

    @unittest.skipUnless(BRANCH_ROOT.is_dir(), "interaction branch is not installed")
    def test_branch_is_present_and_registered(self) -> None:
        """With the branch installed, its CLI commands are registered again."""
        from src.harness.cli import build_parser, load_branch

        self.assertIsNotNone(load_branch())
        parser = build_parser()
        choices = set(parser._subparsers._group_actions[0].choices)  # noqa: SLF001
        self.assertTrue(OFFLINE_SUBCOMMANDS <= choices)
        self.assertTrue(BRANCH_SUBCOMMANDS <= choices)


if __name__ == "__main__":
    unittest.main()
