"""Regression coverage for output location, collisions, and resume compatibility."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from bm25_agent import cli, paths


class OutputPathsTest(unittest.TestCase):
    def test_cli_paths_do_not_depend_on_working_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                defaults = cli.build_parser().parse_args(['evaluate'])
                args = cli.build_parser().parse_args([
                    'evaluate', '--output-dir', 'outputs/custom',
                    '--resume-dir', 'outputs/old-run',
                ])
            finally:
                os.chdir(original)
        self.assertEqual(defaults.output_dir, paths.OUTPUTS_DIR / 'evaluations')
        self.assertEqual(args.output_dir, paths.PROJECT_DIR / 'outputs/custom')
        self.assertEqual(args.resume_dir, paths.PROJECT_DIR / 'outputs/old-run')
        self.assertEqual(paths.result_path('/tmp/custom'), Path('/tmp/custom').resolve())

    def test_same_clock_batch_runs_are_distinct_and_resume_reuses_directory(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(cli, 'TaskStore') as tasks, \
                patch.object(cli, 'BM25Catalog'), \
                patch.object(cli, 'load_dotenv'), \
                patch.object(cli, 'DeepSeekSelector'), \
                patch.object(cli, 'BM25AgentPipeline') as pipeline, \
                patch.object(paths, 'datetime') as clock:
            tasks.return_value.list_task_ids.return_value = ['task_one']
            result = Mock()
            result.to_dict.return_value = {'task': {'task_id': 'task_one'}}
            pipeline.return_value.run_many_iter.return_value = [(0, result)]
            clock.now.return_value.strftime.return_value = '20261013T000000000000Z'
            with patch.object(cli, 'evaluation_summary_payloads', return_value={'task_count': 1, 'failed_count': 0}):
                created = []
                for _ in range(2):
                    out, err = io.StringIO(), io.StringIO()
                    with redirect_stdout(out), redirect_stderr(err):
                        self.assertEqual(cli.main(['evaluate', '--output-root', directory]), 0)
                    path = Path(json.loads(out.getvalue())['output_dir'])
                    self.assertIn(str(path), err.getvalue())
                    self.assertTrue((path / 'runs.jsonl').is_file())
                    created.append(path)
                self.assertNotEqual(*created)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(cli.main(['evaluate', '--resume-dir', str(created[0])]), 0)
                self.assertEqual(pipeline.return_value.run_many_iter.call_count, 2)
                self.assertEqual(len(list(Path(directory).iterdir())), 2)
