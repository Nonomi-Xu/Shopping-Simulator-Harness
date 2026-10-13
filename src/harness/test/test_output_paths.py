"""CLI result paths stay stable across working directories and repeat invocations."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.harness.cli import build_parser
from src.harness import paths
from src.harness.agent.rollout import _prepare_output_dir, _resolve_rollout_id, RolloutError


class OutputPathsTest(unittest.TestCase):
    def test_output_and_replay_paths_are_project_relative(self):
        cases = [
            (['run', '--task-id', 'x', '--output', 'outputs/result.json'], 'output', 'outputs/result.json'),
            (['run', '--task-id', 'x', '--artifacts-dir', 'outputs/audit'], 'artifacts_dir', 'outputs/audit'),
            (['rollout', '--split', 'test', '--output-root', 'outputs/custom'], 'output_root', 'outputs/custom'),
            (['replay', '--run-dir', 'outputs/old-run'], 'run_dir', 'outputs/old-run'),
            (['episodes', 'replay', '--run-dir', 'outputs/episode'], 'run_dir', 'outputs/episode'),
            (['episodes', 'run', '--cache', 'cache.db', '--output-dir', 'outputs/episode'], 'output_dir', 'outputs/episode'),
            (['experiments', 'report', '--run-dir', 'outputs/experiment', '--output-dir', 'outputs/report'], 'output_dir', 'outputs/report'),
        ]
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                parser = build_parser()
                for command, field, relative in cases:
                    with self.subTest(command=command):
                        self.assertEqual(getattr(parser.parse_args(command), field), paths.PROJECT_DIR / relative)
                defaults = parser.parse_args(['rollout', '--split', 'test'])
                self.assertEqual(defaults.output_root, paths.OUTPUTS_DIR / 'rollouts')
                self.assertTrue(defaults.products_path.is_absolute())
                self.assertEqual(paths.result_path('/tmp/explicit'), Path('/tmp/explicit').resolve())
                self.assertIsNone(parser.parse_args(['experiments', 'run', '--config', 'x']).output_dir)
                self.assertIsNone(parser.parse_args(['episodes', 'compare', '--cache', 'x']).output_dir)
            finally:
                os.chdir(original)

    def test_same_clock_runs_are_distinct_and_existing_directory_is_preserved(self):
        with patch.object(paths, 'datetime') as clock, tempfile.TemporaryDirectory() as directory:
            clock.now.return_value.strftime.return_value = '20261013T000000000000Z'
            first, second = _resolve_rollout_id(None), _resolve_rollout_id(None)
            self.assertNotEqual(first, second)
            self.assertLessEqual(len(first), 80)
            for run_id in (first, second):
                _prepare_output_dir(directory, run_id)
            marker = Path(directory) / first / 'summary.json'
            marker.write_text('preserve')
            with self.assertRaises(RolloutError):
                _prepare_output_dir(directory, first)
            self.assertEqual(marker.read_text(), 'preserve')
            self.assertNotEqual(paths.default_run_path('episodes', 'run'), paths.default_run_path('episodes', 'run'))
