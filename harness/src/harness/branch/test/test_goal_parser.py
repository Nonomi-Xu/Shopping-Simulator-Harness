import tempfile
import unittest
from pathlib import Path
from src.harness.branch.episodes.parser import RuleGoalParser, StructuredGoalParser, validate_goal
from src.harness.branch.episodes.preprocess import prepare_episodes
from src.harness.branch.episodes.store import EpisodeStore
from src.harness.schemas import PublicTask
from src.harness.test.test_agent_runner import ScriptedProvider, tool_turn


class GoalParserTest(unittest.TestCase):
    def test_rule_and_cache_no_reparse(self):
        task = PublicTask(task_id="task_" + "a"*24, query="我想买耳机，预算100元以内")
        parser = RuleGoalParser()
        goal = parser.parse(task.query)
        self.assertEqual(goal.status, "ready")
        self.assertEqual(goal.requirements[1].predicate.value, 100)
        with tempfile.TemporaryDirectory() as directory, EpisodeStore(Path(directory)/"cache.sqlite3") as store:
            self.assertEqual(prepare_episodes([task], parser, store, dataset_sha256="0"*64)["prepared"], 1)
            parser.parse = lambda _: self.fail("cache hit called parser")
            self.assertEqual(prepare_episodes([task], parser, store, dataset_sha256="0"*64)["cached"], 1)
            episode = next(store.episodes())
            self.assertEqual(episode.initial_observation, "耳机")
            self.assertEqual(episode.full_query, task.query)

    def test_structured_output_and_provenance(self):
        query = "我想买耳机，预算100元以内"
        goal = RuleGoalParser().parse(query)
        provider = ScriptedProvider([tool_turn("p", "parse_goal", goal.model_dump(mode="json"))])
        self.assertEqual(StructuredGoalParser(provider).parse(query), goal)
        with self.assertRaises(ValueError): validate_goal("耳机", goal)
        changed = goal.model_copy(update={"requirements": [goal.requirements[1].model_copy(update={"predicate": goal.requirements[1].predicate.model_copy(update={"value": 999})})]})
        with self.assertRaises(ValueError): validate_goal(query, changed)

    def test_approximate_and_unsupported_queries_require_review(self):
        self.assertEqual(RuleGoalParser().parse("我想买耳机，40元左右，火星材质").status, "needs_review")
