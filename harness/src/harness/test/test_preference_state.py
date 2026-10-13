import unittest
from src.harness.agent.contracts import Constraint, TaskProfile, UserProfile, PlanUpdate, UserMessage
from src.harness.agent.planner import Planner
from src.harness.constraints import Predicate
from src.harness.schemas import PublicTask


class PreferenceStateTest(unittest.TestCase):
    def test_explicit_reply_state_and_hard_preservation(self):
        task = TaskProfile(task=PublicTask(task_id="task_"+"a"*24, query="耳机"), user_messages=[UserMessage(message_id="user_0001", text="预算100元以内")])
        planner = Planner()
        budget = Constraint(constraint_id="budget", description="预算100元以内", priority="hard", source="user_message", source_message_id="user_0001", source_quote="预算100元以内", topic="budget", predicate=Predicate(field="price", op="le", value=100, unit="CNY"))
        plan, task = planner.update(planner.start(task,"s"), PlanUpdate(constraints=[budget], queries=["耳机"], change_summary="预算已知", uncertain_requirements=["颜色是否有要求"]), task, UserProfile(), set())
        self.assertEqual(task.constraints[0].predicate.value, 100)
        self.assertEqual(task.uncertain_requirements, ["颜色是否有要求"])
        with self.assertRaises(ValueError):
            planner.update(plan, PlanUpdate(constraints=[], queries=["耳机"], change_summary="忽略", no_constraints_reason="无"), task, UserProfile(), set())
        forged = budget.model_copy(update={"predicate": Predicate(field="price", op="le", value=999, unit="CNY")})
        with self.assertRaises(ValueError):
            planner.update(plan, PlanUpdate(constraints=[forged], queries=["耳机"], change_summary="修改"), task, UserProfile(), set())
