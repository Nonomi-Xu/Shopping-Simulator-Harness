import io
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from src.harness.test.test_episode_runner import EpisodeFixture
from src.harness.cli import main
from examples.long_horizon_demo import DemoProvider


class EpisodeCLITest(EpisodeFixture):
    def test_offline_prepare_cached_run_and_replay(self):
        cache = Path(self.tmp.name)/"cache.sqlite3"
        common = ["--cache", str(cache), "--products-path", str(self.path), "--index-path", str(self.index), "--no-env-file"]
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["episodes", "prepare", *common, "--parser", "rule", "--count", "0"]), 0)
        self.assertIn('"prepared": 2', output.getvalue())
        run = Path(self.tmp.name)/"cli-run"
        with patch("src.harness.episodes.cli.DeepSeekProvider", side_effect=lambda **kw: DemoProvider()), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["episodes", "run", *common, "--episode-id", self.episode.episode_id, "--output-dir", str(run)]), 0)
            self.assertEqual(main(["episodes", "replay", "--run-dir", str(run)]), 0)
        self.assertTrue((run/"public/trajectory.jsonl").is_file())
