"""Studio checkpoints survive node/graph recreation and preserve runtime semantics."""
import importlib.util
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from unittest.mock import patch

from src.harness.agent.backend_comparison import compare_results
from src.harness.agent.contracts import AgentRunResult
from src.harness.agent.events import replay_events
from src.harness.agent.orchestrator import validate_checkpoint


@unittest.skipUnless(importlib.util.find_spec('langgraph'), 'install .[studio]')
class StudioTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict('os.environ', {'SHOPPING_STUDIO_PROVIDER': 'demo', 'LANGSMITH_TRACING': 'false'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_explicit_roles_complete_and_match_agent_runner(self):
        from src.harness.branch.studio import graph, environment
        from types import SimpleNamespace

        with patch('src.harness.sources.datetime') as clock, \
                patch('src.harness.branch.studio.uuid4', return_value=SimpleNamespace(hex='parity')):
            clock.now.return_value = datetime(2026, 10, 15, tzinfo=timezone.utc)
            actual = graph.invoke({'query': '寻找白色耳机'})
            with environment() as runner:
                expected = runner.run_query('寻找白色耳机', session_id='studio_parity')
        self.assertEqual(actual['status'], 'completed')
        self.assertEqual(actual['recommendations'][0]['asin'], 'white-headphones')
        result = AgentRunResult.model_validate(actual['result'])
        comparison = compare_results(expected, result)
        self.assertTrue(comparison['equal'], comparison)
        self.assertIsNone(result.session.evaluation)
        self.assertEqual(set(graph.get_graph().nodes),
                         {'__start__','__end__','initialize','planner','executor','observer','await_user'})
        self.assertEqual(set(graph.get_input_jsonschema()['properties']), {'query'})

    def test_clarification_resumes_with_new_graph_and_no_duplicate_search(self):
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import Command
        from src.harness.branch.studio import create_graph
        saver = InMemorySaver()
        config = {'configurable': {'thread_id': 'clarification'}}
        first = create_graph(saver).invoke({'query': '寻找耳机'}, config)
        self.assertTrue(first['__interrupt__'])
        validate_checkpoint(AgentRunResult.model_validate(first['result']))
        result = create_graph(saver).invoke(Command(resume='黑色'), config)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['recommendations'][0]['asin'], 'black-headphones')
        self.assertEqual(result['outcome']['action_count'], 8)
        self.assertEqual(result['outcome']['search_count'], 1)
        self.assertEqual(result['outcome']['tool_call_count'], 3)
        parsed = AgentRunResult.model_validate(result['result'])
        self.assertEqual(replay_events(parsed.events)['outcome']['status'], 'completed')

    def test_sessions_are_isolated_under_concurrent_runs(self):
        from src.harness.branch.studio import graph
        with ThreadPoolExecutor(max_workers=2) as pool:
            white, black = list(pool.map(lambda q: graph.invoke({'query': q}),
                                        ['寻找白色耳机', '寻找黑色耳机']))
        self.assertNotEqual(white['runtime']['session_id'], black['runtime']['session_id'])
        self.assertEqual(white['recommendations'][0]['asin'], 'white-headphones')
        self.assertEqual(black['recommendations'][0]['asin'], 'black-headphones')

    def test_configuration_changes_are_rejected_on_resume(self):
        from src.harness.branch.studio import environment, initialize, restore
        state = initialize({'query': '寻找白色耳机'})
        state['runtime']['manifest']['model'] = 'different-model'
        with environment() as runner, self.assertRaisesRegex(ValueError, 'configuration or source changed'):
            restore(runner, state['runtime'])

    def test_demo_rejects_unrelated_request(self):
        from src.harness.branch.studio import graph
        with self.assertRaisesRegex(ValueError, '离线演示'):
            graph.invoke({'query':'帮我买电视'})
