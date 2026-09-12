import json
from pathlib import Path
from src.harness.test.test_episode_runner import EpisodeFixture
from src.harness.test.test_agent_runner import tool_turn
from src.harness.episodes.models import EpisodeConfig
from src.harness.episodes.artifacts import write_episode_artifacts, replay_episode
from src.harness.episodes.metrics import summarize, compute_metrics, MetricContext


class LongHorizonMetricsTest(EpisodeFixture):
    def test_ranked_success_violation_and_unknown_coverage(self):
        from examples.long_horizon_demo import DemoProvider
        from src.harness.episodes.runner import EpisodeRunner
        from src.harness.catalog import ProductCatalog
        from src.harness.schemas import RecommendationItem, SubmitRecommendationsAction
        from src.harness.evaluator import evaluate_submission
        runner = EpisodeRunner(products_path=self.path, index_path=self.index, provider=DemoProvider())
        result = runner.run_episode(self.episode)
        recommendations = [RecommendationItem(asin=asin, selected_options={}, reason="public recommendation") for asin in ["expensive", "cheap"]]
        trajectory = result.trajectory.model_copy(update={"final_recommendation": recommendations})
        with ProductCatalog(self.index) as catalog:
            evaluation = evaluate_submission(facts=runner.tasks.get_private_facts(self.episode.task_id),
                submission=SubmitRecommendationsAction(recommendations=recommendations), catalog=catalog)
            context = MetricContext(self.episode, trajectory, evaluation, result.disclosures, set(result.revealed_requirement_ids), catalog)
            metrics = compute_metrics(context)["shopping"]
            self.assertFalse(metrics["success_at_1"])
            self.assertTrue(metrics["success_at_k"]["2"])
            self.assertEqual(metrics["constraint_satisfaction_rate"], 0.75)
            self.assertEqual(metrics["constraint_status_counts"]["violated"], 1)
            requirements = [r.model_copy(update={"predicate": None}) if r.topic == "budget" else r for r in self.episode.hidden_goal.requirements]
            episode = self.episode.model_copy(update={"hidden_goal": self.episode.hidden_goal.model_copy(update={"requirements": requirements})})
            unknown = compute_metrics(MetricContext(episode, trajectory, evaluation, result.disclosures, set(result.revealed_requirement_ids), catalog))["shopping"]
            self.assertEqual(unknown["constraint_verification_coverage"], 0.5)
            self.assertEqual(unknown["constraint_status_counts"]["unknown"], 2)

    def test_information_gain_redundancy_and_serialization(self):
        ask = lambda i: tool_turn(i, "ask_user", {"question": "预算是多少？", "information_gap": "预算"})
        runner, _ = self.runner([ask("a"), ask("b"), ask("c")], EpisodeConfig(max_turns=3))
        result = runner.run_episode(self.episode)
        self.assertEqual(result.disclosures[0].candidates_before, 2)
        self.assertEqual(result.disclosures[0].candidates_after, 1)
        m = result.metrics["shopping"]
        self.assertEqual(m["redundant_question_rate"], 0.5)
        self.assertEqual(m["mean_information_gain"], 0.25)
        self.assertFalse(m["success_at_1"])
        self.assertIsNone(summarize([result])["average_turns_to_success"])
        path = write_episode_artifacts(result, Path(self.tmp.name)/"run")
        self.assertTrue(replay_episode(path)["verified"])
        raw = (path/"public/trajectory.jsonl").read_text()
        for forbidden in ['"hidden_goal"', '"evaluation"', '"target_asin"', '"unrevealed_fields"']:
            self.assertNotIn(forbidden, raw)
        (path/"public/trajectory.jsonl").write_text(raw.replace('"model_turn_count": 3', '"model_turn_count": 4'))
        with self.assertRaises(ValueError): replay_episode(path)
