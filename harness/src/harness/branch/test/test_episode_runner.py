import json
import tempfile
import unittest
from pathlib import Path
from src.data.search_bm25 import build_index, sha256_file
from src.harness.tasks import TaskStore
from src.harness.branch.episodes.parser import RuleGoalParser
from src.harness.branch.episodes.preprocess import build_episode
from src.harness.branch.episodes.models import EpisodeConfig, Trajectory
from src.harness.branch.episodes.runner import EpisodeRunner
from src.harness.test.test_agent_runner import ScriptedProvider, tool_turn


class EpisodeFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/"products.json"
        self.index = Path(self.tmp.name)/"index.sqlite3"
        products = []
        for asin, price in [("cheap", 80), ("expensive", 200)]:
            products.append(dict(asin=asin, title="耳机", category="耳机", shop_name="商店", pricing=[price], attribute=["耳机"], customization_options={},
                instructions=[dict(asin=asin, instruction="我想买耳机，预算100元以内" if asin == "cheap" else "我想买耳机，预算300元以内", instruction_options=[], attributes=[])]))
        self.path.write_text(json.dumps(products, ensure_ascii=False))
        build_index(products, self.index, product_data_sha256=sha256_file(self.path))
        tasks = TaskStore(self.path)
        task = next(tasks.iter_public_tasks("all"))
        parser = RuleGoalParser()
        self.episode = build_episode(task, parser.parse(task.query), tasks.product_data_sha256, parser.manifest)

    def runner(self, outputs, config=None):
        provider = ScriptedProvider(outputs)
        return EpisodeRunner(products_path=self.path, index_path=self.index, provider=provider, config=config), provider


class EpisodeRunnerTest(EpisodeFixture):
    def test_environment_limit_and_simulator_failure(self):
        from examples.long_horizon_demo import DemoProvider
        runner = EpisodeRunner(products_path=self.path, index_path=self.index, provider=DemoProvider(), config=EpisodeConfig(max_environment_calls=1))
        result = runner.run_episode(self.episode)
        self.assertEqual(result.run.terminal_reason, "max_steps")
        self.assertEqual(result.run.tool_call_count, 1)
        from src.harness.branch.episodes.simulator import DeterministicSimulator
        class BrokenSimulator(DeterministicSimulator):
            def answer(self, *args, **kwargs):
                super().answer(*args, **kwargs)
                raise ValueError("PRIVATE_INTERNAL_FAILURE")
        runner, _ = self.runner([tool_turn("q", "ask_user", {"question": "预算？", "information_gap": "预算"})])
        runner.simulator_factory = BrokenSimulator
        result = runner.run_episode(self.episode)
        self.assertEqual(result.run.terminal_reason, "simulator_error")
        self.assertEqual(result.revealed_requirement_ids, self.episode.initial_requirement_ids)
        self.assertNotIn("PRIVATE_INTERNAL_FAILURE", result.trajectory.model_dump_json())

    def test_applied_filters_preserve_known_hard_budget(self):
        from examples.long_horizon_demo import DemoProvider
        class OmitTypedBudget(DemoProvider):
            def decide(self, task, payload):
                name, args = super().decide(task, payload)
                if name == "update_plan":
                    for c in args["constraints"]:
                        if c["constraint_id"] == "budget": c.pop("predicate")
                return name, args
        runner = EpisodeRunner(products_path=self.path, index_path=self.index, provider=OmitTypedBudget())
        result = runner.run_episode(self.episode)
        searches = [t for t in result.trajectory.turns if t.action_type == "search"]
        self.assertEqual(len(searches), 2)
        # The policy deliberately omitted filters; the runtime enforces the known budget.
        self.assertNotIn("filters", searches[-1].action.arguments)
        self.assertTrue(any(p["field"] == "price" and p["value"] == 100 for p in searches[-1].observation.projection.applied_filters["predicates"]))
        self.assertEqual(searches[-1].candidates.returned_ids, ["cheap"])

    def test_max_turns_ends_without_extra_disclosure(self):
        runner, provider = self.runner([tool_turn("q", "ask_user", {"question": "预算？", "information_gap": "预算未知"})], EpisodeConfig(max_turns=1))
        result = runner.run_episode(self.episode)
        self.assertEqual(result.trajectory.terminal_reason, "max_model_turns")
        self.assertEqual(result.disclosures, [])
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(result.run.session.public_task.task_id, self.episode.task_id)
        self.assertIsNone(result.run.session.evaluation)
        self.assertEqual(Trajectory.model_validate_json(result.trajectory.model_dump_json()), result.trajectory)

    def test_budget_cumulative_across_answer_and_state_update(self):
        runner, provider = self.runner([
            tool_turn("q", "ask_user", {"question": "预算？", "information_gap": "预算未知"}),
            tool_turn("p", "update_plan", {"constraints": [{"constraint_id": "budget", "description": "100元以内", "priority": "hard", "source": "user_message", "source_message_id": "user_0001", "source_quote": "预算100元以内", "predicate": {"field": "price", "op": "le", "value": 100, "unit": "CNY"}}], "queries": ["耳机"], "change_summary": "记录预算"})
        ], EpisodeConfig(max_tool_calls=2))
        result = runner.run_episode(self.episode)
        self.assertEqual(result.trajectory.terminal_reason, "max_actions")
        self.assertEqual(len(result.disclosures), 1)
        self.assertEqual(result.run.task_profile.constraints[0].predicate.value, 100)
        self.assertEqual(result.run.action_count, 2)
        first = str(provider.requests[0])
        self.assertNotIn("预算100元以内", first)
        self.assertNotIn('"target_asin":', first)
