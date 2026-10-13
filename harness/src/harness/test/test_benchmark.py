import json
import unittest

from src.harness.benchmark.datasets import freeze, load
from src.harness.benchmark.runner import score_harness
from src.harness.agent import AgentRunner
from src.harness.agent.audit import RunAudit
from src.harness.agent.artifacts import replay_run
from src.harness.test import test_rollout as fixtures


class AuditTest(unittest.TestCase):
    setUp = fixtures.RolloutRunnerTest.setUp

    def test_streamed_audit_records_full_chain_and_replays(self):
        audit = RunAudit(self.directory/'audit')
        result = AgentRunner(provider=fixtures.successful_provider(fixtures.TARGET_ASIN, '白色'),
            harness=self.harness, event_observer=audit).run(self.target_task_id, max_steps=7)
        events = [json.loads(line) for line in (audit.directory/'events.jsonl').read_text().splitlines()]
        self.assertEqual(len(events), len(result.events))
        self.assertIn('response', next(e['payload'] for e in events if e['kind']=='model_turn'))
        self.assertTrue(any(e['kind']=='context_built' for e in events))
        self.assertFalse((audit.directory/'run_manifest.json').exists())
        audit.finish(result)
        self.assertEqual(replay_run(audit.directory)['outcome']['status'], 'completed')
        scores = score_harness(result, fixtures.TARGET_ASIN)
        self.assertTrue(scores['strict_at_1'])
        self.assertTrue(scores['retrieval_hit'])
        self.assertTrue(scores['inspected_hit'])
        with self.assertRaises(FileExistsError):
            RunAudit(audit.directory)

    def test_freeze_is_immutable_and_validates_source(self):
        store = self.harness._task_store
        path = self.directory/'Test1.json'
        frozen = freeze(store, path, 'Test1', count=2)
        self.assertEqual(load(path, store), frozen)
        with self.assertRaises(ValueError):
            freeze(store, path, 'Test1', count=1)
        broken = {**frozen, 'source_sha256': '0'*64}
        path.write_text(json.dumps(broken))
        with self.assertRaises(ValueError):
            load(path, store)

    def test_model_trace_redacts_credentials_without_removing_usage(self):
        audit = RunAudit(self.directory/'redaction')
        audit.transport('request', {'api_key': 'never-log', 'nested': {'authorization': 'bearer'}, 'max_tokens': 50})
        raw = (audit.directory/'model_io.jsonl').read_text()
        self.assertNotIn('never-log', raw)
        self.assertNotIn('bearer', raw)
        self.assertEqual(json.loads(raw)['payload']['max_tokens'], 50)

class RedirectTest(unittest.TestCase):
    def test_same_origin_post_redirect_preserves_body_and_records_it(self):
        from urllib.error import HTTPError
        from src.harness.providers.deepseek import DeepSeekProvider
        from src.harness.providers import ChatMessage
        from src.harness.test.test_deepseek import _Response, _text_response
        requests, observed = [], []
        def opener(request, timeout):
            requests.append(request)
            if len(requests) == 1:
                raise HTTPError(request.full_url, 307, 'redirect', {'Location':'/v1/chat/completions'}, None)
            return _Response(_text_response())
        provider=DeepSeekProvider(api_key='test-secret', opener=opener, request_observer=lambda k,p: observed.append((k,p)))
        provider.complete([ChatMessage(role='user', content='hi')])
        self.assertEqual(requests[1].data,requests[0].data)
        self.assertEqual(requests[1].get_method(),'POST')
        self.assertEqual(requests[1].full_url,'https://api.deepseek.com/v1/chat/completions')
        self.assertEqual([k for k,p in observed],['request','redirect','response'])
        self.assertNotIn('test-secret',json.dumps(observed))

    def test_cross_origin_and_redirect_loops_are_rejected(self):
        from urllib.error import HTTPError
        from src.harness.providers.deepseek import DeepSeekProvider, DeepSeekAPIError
        from src.harness.providers import ChatMessage
        for destination in ['https://other.example/completions','http://api.deepseek.com/chat/completions','/chat/completions']:
            calls=[]
            def opener(request, timeout):
                calls.append(request)
                raise HTTPError(request.full_url,307,'redirect',{'Location':destination},None)
            provider=DeepSeekProvider(api_key='test-secret',opener=opener)
            with self.assertRaises(DeepSeekAPIError):provider.complete([ChatMessage(role='user',content='hi')])
            self.assertLessEqual(len(calls),4)

class PairedReportTest(unittest.TestCase):
    def test_rerun_reuses_verified_baseline_and_flags_infrastructure_failures(self):
        import tempfile
        from pathlib import Path
        from src.harness.agent.audit import write_json
        from src.harness.agent.events import digest
        from src.harness.benchmark.datasets import sha256_file
        from src.harness.benchmark.report import generate
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old=root/'old';new=root/'rerun';old.mkdir();new.mkdir()
            dataset={'name':'Test1','task_ids':['task_one'],'source_sha256':'source','original_count':1}
            config={'task_ids':['task_one'],'dataset_sha256':digest(dataset),'source_sha256':'source',
                    'smoke_limit':None,'model':'test','harness_thinking':'enabled'}
            for directory,method in [(old,'bm25'),(new,'harness')]:
                write_json(directory/'config.json',config);write_json(directory/'dataset.json',dataset)
                (directory/method).mkdir()
                artifact=directory/method/'task';artifact.mkdir()
                row={'task_id':'task_one','index':1,'method':method,'config_sha256':digest(config),
                     'artifact_dir':f'{method}/task','status':'completed','terminal_reason':'submitted',
                     'strict_at_1':True,'asin_at_1':True,'strict_any':True,'reward_valid':True,
                     'retrieval_hit':True,'first_search_top5_hit':True,'inspected_hit':True,
                     'duration_seconds':1,'model_turns':1,'searches':1,'actions':1,'environment_calls':1,
                     'reward':1,'recommendation_count':1,'usage':{'input_tokens':1,'output_tokens':1,'total_tokens':2},'usage_complete':True}
                if method=='harness':
                    row.update(status='failed',terminal_reason='provider_error',strict_at_1=False,strict_any=False,
                               error_message='DeepSeek API returned HTTP 402: Insufficient Balance')
                journal=directory/method/'records.jsonl';journal.write_text(json.dumps(row)+'\n')
                write_json(directory/method/'completed.json',{'config_sha256':digest(config),'records_sha256':sha256_file(journal)})
            summary=generate(new,old)
            self.assertEqual(summary['source_runs']['bm25'],str(old.resolve()))
            self.assertFalse(summary['paired']['valid_for_algorithm_claims'])
            self.assertEqual(summary['execution_health']['provider_errors']['harness']['insufficient_balance_402'],1)
            text=(new/'analysis/report.md').read_text()
            self.assertIn('BM25 本次没有重复调用',text)
            self.assertIn('基础设施故障',text)
            row.update(terminal_reason='runtime_error',error_type='IncompleteRead',
                       error_message='IncompleteRead(0 bytes read)')
            journal=new/'harness/records.jsonl'
            journal.write_text(json.dumps(row)+'\n')
            write_json(new/'harness/completed.json',{'config_sha256':digest(config),'records_sha256':sha256_file(journal)})
            interrupted=generate(new,old)
            self.assertEqual(interrupted['execution_health']['infrastructure_task_counts']['harness'],1)
            self.assertEqual(interrupted['execution_health']['strict_at_1_sensitivity']['harness'],
                             {'observed':0.0,'if_all_infrastructure_failures_succeeded':1.0})
            changed=dict(config,source_sha256='changed')
            write_json(old/'config.json',changed)
            with self.assertRaises(ValueError):generate(new,old)
