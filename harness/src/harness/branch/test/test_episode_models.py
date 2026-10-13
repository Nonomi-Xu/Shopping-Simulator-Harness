import unittest
from pydantic import ValidationError
from src.harness.constraints import Predicate
from src.harness.branch.episodes.models import UserGoal, GoalRequirement, DisclosureState
from src.harness.schemas import SearchProductsAction
from src.harness.agent.tools import ToolRegistry


class EpisodeModelsTest(unittest.TestCase):
    def test_schema_and_private_state(self):
        r = GoalRequirement(requirement_id="budget", topic="budget", description="预算", source_quote="100元以内", priority="hard", predicate=Predicate(field="price", op="le", value=100, unit="CNY"))
        goal = UserGoal(requirements=[r])
        self.assertEqual(UserGoal.model_validate_json(goal.model_dump_json()), goal)
        with self.assertRaises(ValidationError):
            UserGoal(requirements=[r, r])
        state = DisclosureState(revealed_fields={"budget"})
        self.assertEqual(DisclosureState.model_validate_json(state.model_dump_json()), state)

    def test_numeric_contract(self):
        for value in [float("nan"), -1, "100"]:
            with self.assertRaises(ValidationError):
                Predicate(field="price", op="le", value=value, unit="CNY")

    def test_explicit_bm25_is_valid_in_multi_schema(self):
        request = SearchProductsAction(query="耳机", method="bm25")
        registry = ToolRegistry(multi_agent=True)
        schema = next(d.parameters for d in registry.definitions if d.name == "search_products")
        self.assertIn("method", schema["properties"])
        self.assertEqual(request.method, "bm25")
        with self.assertRaises(ValidationError):
            SearchProductsAction(query="耳机", method="vector")
