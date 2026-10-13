import json
from pathlib import Path
from src.harness.branch.test.test_episode_runner import EpisodeFixture
from src.harness.branch.episodes.runner import EpisodeRunner
from src.harness.branch.episodes.artifacts import write_episode_artifacts
from examples.long_horizon_demo import DemoProvider


class EpisodeLeakageTest(EpisodeFixture):
    def test_every_provider_input_and_public_artifact_is_projected(self):
        provider = DemoProvider()
        runner = EpisodeRunner(products_path=self.path, index_path=self.index, provider=provider)
        # A private-only canary must not survive into requests, tools, plans or events.
        original = runner.tasks._private_facts[self.episode.task_id]
        runner.tasks._private_facts[self.episode.task_id] = original.model_copy(update={"target_title": "PRIVATE_ANSWER_CANARY_731", "required_attributes": ["HIDDEN_ANSWER_CANARY_938"]})
        result = runner.run_episode(self.episode)
        self.assertTrue(result.evaluation.strict_success)
        before_budget = True
        for messages in provider.requests:
            payload = json.loads(messages[1]["content"])
            text = json.dumps(messages, ensure_ascii=False)
            for forbidden in ['"target_product_id":', '"target_asin":', '"hidden_goal":', '"private_answer":', '"unrevealed_fields":', "PRIVATE_ANSWER_CANARY_731", "HIDDEN_ANSWER_CANARY_938"]:
                self.assertNotIn(forbidden, text)
            if not payload["task_profile"]["user_messages"]:
                self.assertNotIn("预算100元以内", text)
            self.assertNotIn(self.episode.full_query, text)
        root = write_episode_artifacts(result, Path(self.tmp.name)/"audit")
        for path in (root/"public").rglob("*"):
            if path.is_file():
                text = path.read_text()
                for secret in ["PRIVATE_ANSWER_CANARY_731", "HIDDEN_ANSWER_CANARY_938", self.episode.full_query, '"hidden_goal":']:
                    self.assertNotIn(secret, text)
        # A normally retrieved target ID is public product identity, not answer leakage.
        self.assertIn(original.target_asin, json.dumps(provider.requests, ensure_ascii=False))

    def test_simulator_has_no_product_or_task_store_capability(self):
        from src.harness.branch.episodes.simulator import DeterministicSimulator
        simulator = DeterministicSimulator(self.episode.hidden_goal, self.episode.disclosure_policy, self.episode.initial_requirement_ids)
        self.assertEqual(set(vars(simulator)), {"goal", "policy", "state"})
