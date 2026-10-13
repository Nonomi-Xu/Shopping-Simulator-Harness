"""Regression guard: the CLI must not advertise unimplemented retrieval options.

`--vector-index` and `--source-search-methods` used to be accepted by
``run`` / ``rollout`` / ``chat`` and then forwarded as keyword arguments to
``ShoppingHarness``, ``ProductCatalog``, ``JsonProductSource`` and
``HttpProductSource``. None of those constructors accepts them, so the flags
raised an unhandled ``TypeError`` instead of doing anything. The public search
contract is BM25-only, so the options were removed rather than left broken.
"""

import contextlib
import inspect
import io
import unittest
from pathlib import Path

from src.harness.catalog import ProductCatalog
from src.harness.cli import build_parser
from src.harness.harness import ShoppingHarness
from src.harness.sources import HttpProductSource, JsonProductSource

BRANCH_ROOT = Path(__file__).resolve().parents[1] / "branch"
REMOVED_OFFLINE_OPTIONS = ("--vector-index",)
REMOVED_INTERACTIVE_OPTIONS = ("--source-search-methods",)


def _rejects(parser, argv):
    """Return True when argparse refuses the argument list."""
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            parser.parse_args(argv)
        except SystemExit:
            return True
    return False


class CliRetrievalOptionsTest(unittest.TestCase):
    def test_offline_commands_reject_vector_index(self):
        parser = build_parser()
        for argv in (["run", "--task-id", "x"], ["rollout", "--split", "test"]):
            for option in REMOVED_OFFLINE_OPTIONS:
                with self.subTest(command=argv[0], option=option):
                    self.assertTrue(
                        _rejects(parser, [*argv, option, "vectors.bin"]),
                        f"{argv[0]} still accepts {option}",
                    )

    @unittest.skipUnless(BRANCH_ROOT.is_dir(), "interaction branch is not installed")
    def test_interactive_commands_reject_search_methods(self):
        parser = build_parser()
        base = ["chat", "--query", "q", "--source-url", "https://example.test",
                "--artifacts-dir", "outputs/never-created"]
        for option in REMOVED_INTERACTIVE_OPTIONS:
            with self.subTest(option=option):
                self.assertTrue(_rejects(parser, [*base, option, "vector"]))

    def test_constructors_accept_no_vector_keywords(self):
        """Documents why the CLI options could never have worked as written."""
        for constructor in (
            ShoppingHarness.__init__,
            ProductCatalog.__init__,
            JsonProductSource.__init__,
            HttpProductSource.__init__,
        ):
            parameters = set(inspect.signature(constructor).parameters)
            with self.subTest(constructor=constructor.__qualname__):
                self.assertNotIn("vector_index_path", parameters)
                self.assertNotIn("embedding_provider", parameters)
                self.assertNotIn("search_methods", parameters)

    def test_offline_namespace_has_no_vector_attributes(self):
        parser = build_parser()
        for argv in (["run", "--task-id", "x"], ["rollout", "--split", "test"]):
            namespace = parser.parse_args(argv)
            with self.subTest(command=argv[0]):
                self.assertFalse(hasattr(namespace, "vector_index"))
                self.assertFalse(
                    any(name.startswith("embedding_") for name in vars(namespace))
                )


if __name__ == "__main__":
    unittest.main()
