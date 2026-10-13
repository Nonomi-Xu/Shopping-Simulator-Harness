from pathlib import Path
from src.harness.test.test_episode_runner import EpisodeFixture
from src.harness.episodes.runner import EpisodeRunner
from src.harness.episodes.baselines import run_comparison
from examples.long_horizon_demo import DemoProvider


class EpisodeBaselinesTest(EpisodeFixture):
    def test_paired_success_same_database_and_model(self):
        summary = run_comparison([self.episode], lambda: EpisodeRunner(products_path=self.path, index_path=self.index, provider=DemoProvider()), Path(self.tmp.name)/"comparison")
        a, b = summary["baselines"]["full_query"], summary["baselines"]["long_horizon"]
        self.assertEqual(a["strict_success_rate"], 1)
        self.assertEqual(b["strict_success_rate"], 1)
        self.assertGreater(b["average_tool_calls"], a["average_tool_calls"])
        self.assertEqual(b["constraint_satisfaction_rate"], 1)
        self.assertEqual(b["premature_recommendation_rate"], 0)
