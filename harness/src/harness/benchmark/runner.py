"""Run both methods on one frozen dataset, journaling each task immediately.

Run from harness: python -m src.harness.benchmark.runner --dataset Test1
Resume the exact configuration using --run-dir /absolute/path/to/the/run.
"""
import argparse
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from threading import Event

from ..paths import PROJECT_DIR, OUTPUTS_DIR, new_run_id
from ..tasks import TaskStore
from ..harness import ShoppingHarness
from ..agent import AgentRunner, RuntimeConfig
from ..agent.audit import RunAudit, write_json
from ..agent.events import digest
from ..agent.skill import load_agent_skill
from ..env import load_dotenv
from ..providers import DeepSeekProvider
from .datasets import load, sha256_file

sys.path.insert(0, str(PROJECT_DIR / 'BM25'))
from bm25_agent.tasks import TaskStore as BM25Tasks
from bm25_agent.catalog import BM25Catalog
from bm25_agent.pipeline import BM25AgentPipeline
from bm25_agent.provider import DeepSeekSelector


class AuditedSelector(DeepSeekSelector):
    def __init__(self, audit, **kwargs):
        super().__init__(**kwargs)
        self.audit = audit

    def _request(self, payload):
        self.audit.transport('request', {'body': payload})
        try:
            result = super()._request(payload)
        except Exception as error:
            self.audit.transport('error', {'type': type(error).__name__, 'message': str(error)})
            raise
        self.audit.transport('response', {'body': result})
        return result


def score_harness(result, target):
    e = result.session.evaluation
    evaluated = e.model_dump(mode='json') if e else None
    recommendations = e.recommendations if e else []
    first = next((r for r in recommendations if r.rank == 1), None)
    searches, inspected = [], []
    for step in result.session.history:
        if step.status != 'success':
            continue
        observation = step.model_dump(mode='json')['observation']
        if step.tool_name == 'search_products':
            if isinstance(observation, dict):
                observation = observation.get('products', observation.get('results', []))
            searches.append([row['asin'] for row in observation if isinstance(row, dict) and 'asin' in row])
        elif step.tool_name == 'get_product' and isinstance(observation, dict):
            if observation.get('asin'):
                inspected.append(observation['asin'])
    retrieved = list(dict.fromkeys(asin for group in searches for asin in group))
    usage_complete = bool(result.turns) and all(turn.usage is not None for turn in result.turns)
    return {
        'status': result.status, 'terminal_reason': result.terminal_reason,
        'evaluation_reason': e.terminal_reason if e else None,
        'strict_at_1': bool(e and e.reward_valid and e.terminal_reason != 'invalid_submission' and first and first.strict_match),
        'asin_at_1': bool(e and e.terminal_reason != 'invalid_submission' and first and first.asin_match),
        'strict_any': bool(e and e.strict_success),
        'reward_valid': bool(e and e.reward_valid),
        'reward': e.reward if e and e.reward_valid else None,
        'retrieval_hit': target in retrieved,
        'first_search_top5_hit': bool(searches and target in searches[0][:5]),
        'inspected_hit': target in inspected,
        'retrieved_asins': retrieved, 'inspected_asins': list(dict.fromkeys(inspected)),
        'searches': len(searches), 'model_turns': result.model_turn_count,
        'actions': result.action_count, 'environment_calls': result.tool_call_count,
        'recommendation_count': len(recommendations),
        'usage': result.usage.model_dump(mode='json') if result.usage else None,
        'usage_complete': usage_complete, 'evaluation': evaluated,
    }


def run_method(method, root, ids, tasks, bm25_tasks, catalog, harness, config, workers):
    directory = root / method
    directory.mkdir(exist_ok=True)
    (directory / 'tasks').mkdir(exist_ok=True)
    finished = {}
    journal = directory / 'records.jsonl'
    if journal.exists():
        with journal.open() as handle:
            for line in handle:
                if not line.endswith('\n'):
                    raise ValueError('truncated journal: recover record.json files before resuming')
                row = json.loads(line)
                if row['task_id'] in finished or row['task_id'] not in ids or row['config_sha256'] != digest(config):
                    raise ValueError('journal task/config mismatch')
                finished[row['task_id']] = row
    # Recover completed task artifacts if interrupted between per-task and global writes.
    with journal.open('a', encoding='utf-8') as handle:
        for index, tid in enumerate(ids):
            record_path = directory/'tasks'/f'{index+1:06d}'/'record.json'
            if tid not in finished and record_path.exists():
                row = json.loads(record_path.read_text())
                if row['task_id'] != tid or row['config_sha256'] != digest(config):
                    raise ValueError('recovery record mismatch')
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
                handle.flush()
                finished[tid] = row
    pending = [(i, tid) for i, tid in enumerate(ids) if tid not in finished]
    print(json.dumps({'method': method, 'completed': len(finished), 'pending': len(pending), 'output_dir': str(directory)}), flush=True)

    stop = Event()

    def one(index, tid):
        parent = directory/'tasks'/f'{index+1:06d}'
        parent.mkdir(exist_ok=True)
        attempt = 1
        while (parent/f'attempt_{attempt:02d}').exists():
            attempt += 1
        audit = RunAudit(parent/f'attempt_{attempt:02d}')
        started = time.perf_counter()
        session_id = f'{method}_{index+1:06d}_{attempt:02d}'
        facts = tasks.get_private_facts(tid)
        try:
            if method == 'bm25':
                selector = AuditedSelector(audit, model=config['model'], base_url=config['base_url'],
                    timeout=config['timeout'], max_retries=config['max_retries'], temperature=0)
                result = BM25AgentPipeline(tasks=bm25_tasks, catalog=catalog, selector=selector).run(tid)
                write_json(audit.directory/'result.json', result.to_dict())
                e, selection = result.evaluation, result.selector
                candidates = [candidate.asin for candidate in result.candidates]
                row = {'status': result.status, 'terminal_reason': e.terminal_reason if e else result.error_type,
                    'evaluation_reason': e.terminal_reason if e else None,
                    'strict_at_1': bool(e and e.strict_success), 'strict_any': bool(e and e.strict_success),
                    'asin_at_1': bool(e and e.asin_match and e.selected_options_valid),
                    'reward_valid': bool(e and e.reward_valid), 'reward': e.reward if e and e.reward_valid else None,
                    'retrieval_hit': facts.target_asin in candidates, 'first_search_top5_hit': facts.target_asin in candidates,
                    'retrieved_asins': candidates, 'inspected_asins': candidates, 'inspected_hit': facts.target_asin in candidates,
                    'model_turns': 1 if candidates else 0, 'searches': 1, 'actions': 1, 'environment_calls': 1,
                    'recommendation_count': 1 if selection else 0,
                    'usage': asdict(selection.usage) if selection and selection.usage else None,
                    'usage_complete': bool(selection and selection.usage), 'evaluation': asdict(e) if e else None}
            else:
                provider = DeepSeekProvider(model=config['model'], base_url=config['base_url'],
                    timeout=config['timeout'], max_retries=config['max_retries'],
                    default_parameters={'temperature': 0, 'thinking': {'type': config['harness_thinking']}},
                    request_observer=audit.transport)
                original_complete = provider.complete
                def guarded_complete(*args, **kwargs):
                    from ..providers import ProviderAPIError
                    if stop.is_set():
                        raise ProviderAPIError("batch stopped after fatal provider configuration error")
                    try:
                        return original_complete(*args, **kwargs)
                    except ProviderAPIError as error:
                        if error.status_code in (401, 402, 403):
                            stop.set()
                        raise
                provider.complete = guarded_complete
                result = AgentRunner(provider=provider, harness=harness,
                    config=RuntimeConfig.model_validate(config['runtime']), max_model_turns=config['max_model_turns'],
                    tool_choice='auto', event_observer=audit).run(tid, max_steps=config['max_steps'], session_id=session_id)
                audit.finish(result)
                row = score_harness(result, facts.target_asin)
                row['error_message'] = result.error_message
                row['error_type'] = result.error_type
        except Exception as error:
            row = {'status': 'error', 'terminal_reason': 'runtime_error', 'error_type': type(error).__name__,
                   'error_message': str(error), 'strict_at_1': False, 'asin_at_1': False, 'strict_any': False,
                   'reward_valid': False, 'reward': None, 'usage': None, 'usage_complete': False,
                   'retrieval_hit': None, 'inspected_hit': None}
            write_json(audit.directory/'error.json', row)
        finally:
            if method == 'harness':
                try:
                    harness.release_session(session_id, force=True)
                except LookupError:
                    pass
                except Exception:
                    # A session may not have been created; do not hide the original error.
                    pass
        row.update(task_id=tid, index=index+1, method=method, config_sha256=digest(config),
                   duration_seconds=time.perf_counter()-started,
                   artifact_dir=str(audit.directory.relative_to(root)))
        write_json(parent/'record.json', row)
        return row

    # Keep at most workers tasks in flight; fatal account errors halt new work.
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=method) as pool, journal.open('a', encoding='utf-8') as handle:
        remaining = iter(pending)
        futures = {}
        def schedule():
            while not stop.is_set() and len(futures) < workers:
                item = next(remaining, None)
                if item is None:
                    break
                index, tid = item
                futures[pool.submit(one, index, tid)] = tid
        schedule()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                row = future.result()
                handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
                handle.flush()
                os.fsync(handle.fileno())
                finished[row['task_id']] = row
                print(json.dumps({'method': method, 'completed': len(finished), 'total': len(ids),
                                  'task_id': row['task_id'], 'status': row['terminal_reason'],
                                  'strict_at_1': row['strict_at_1'], 'seconds': round(row['duration_seconds'], 2)}), flush=True)
            schedule()
    if stop.is_set():
        write_json(directory/'blocked.json', {'reason': 'fatal_provider_account_error',
                    'completed_count': len(finished), 'requested_count': len(ids)})
        raise RuntimeError('Provider account failure: batch stopped; pending tasks were not sent')
    write_json(directory/'completed.json', {'task_count': len(finished), 'records_sha256': sha256_file(journal),
                                          'config_sha256': digest(config)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='Test1')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--method', choices=['both', 'bm25', 'harness'], default='both')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--model', default='deepseek-v4-flash')
    parser.add_argument('--base-url', default='https://api.deepseek.com/v1')
    parser.add_argument('--harness-thinking', choices=['enabled', 'disabled'], default='enabled')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--max-retries', type=int, default=2)
    parser.add_argument('--limit', type=int, help='Only for an explicitly separate smoke run, not the final benchmark')
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error('workers must be 1..32')
    load_dotenv(PROJECT_DIR/'harness/src/.env')
    products, index = PROJECT_DIR/'harness/data/fine_items_eval_train_all.json', PROJECT_DIR/'harness/data/product_search.sqlite3'
    tasks, bm25_tasks = TaskStore(products), BM25Tasks(products)
    frozen = load(PROJECT_DIR/f'datasets/{args.dataset}.json', tasks)
    ids = frozen['task_ids']
    if args.limit is not None:
        if args.limit < 1: parser.error('limit must be positive')
        ids = ids[:args.limit]
    if tasks.list_task_ids('Original') != bm25_tasks.list_task_ids('Original'):
        raise ValueError('method task IDs/order differ')
    runtime = RuntimeConfig().model_dump(mode='json')
    config = {'version': 'shopping-paired-benchmark-v1', 'transport_version': 'same-origin-redirect-v1', 'dataset': args.dataset,
        'dataset_sha256': digest(frozen), 'source_sha256': tasks.product_data_sha256,
        'index_sha256': sha256_file(index), 'task_ids': ids, 'smoke_limit': args.limit,
        'model': args.model, 'base_url': args.base_url, 'temperature': 0,
        'bm25_thinking': 'disabled', 'harness_thinking': args.harness_thinking, 'timeout': args.timeout, 'max_retries': args.max_retries,
        'runtime': runtime, 'max_steps': 20, 'max_model_turns': 40,
        'harness_tool_choice': 'auto', 'bm25_tool_choice': 'select_product',
        'harness_skill_sha256': load_agent_skill().source_sha256,
        'recommendation_policy': 'existing methods; report Strict@1 and Strict-any separately'}
    root = (args.run_dir or OUTPUTS_DIR/'benchmarks'/new_run_id(args.dataset)).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/'config.json').exists():
        if json.loads((root/'config.json').read_text()) != config:
            raise ValueError('resume configuration mismatch')
    else:
        write_json(root/'config.json', config)
        write_json(root/'dataset.json', frozen)
        code_files = list((PROJECT_DIR/'harness/src').rglob('*.py')) + list((PROJECT_DIR/'BM25/bm25_agent').glob('*.py'))
        write_json(root/'source_hashes.json', {str(p.relative_to(PROJECT_DIR)): sha256_file(p) for p in code_files})
        write_json(root/'provenance.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
                   'git_head': subprocess.check_output(['git', '-C', str(PROJECT_DIR), 'rev-parse', 'HEAD'], text=True).strip(),
                   'workers': args.workers, 'python': sys.version})
    print(json.dumps({'run_dir': str(root), 'task_count': len(ids), 'method': args.method}), flush=True)
    with BM25Catalog(index) as catalog, ShoppingHarness(products_path=products, index_path=index) as harness:
        for method in (['bm25', 'harness'] if args.method == 'both' else [args.method]):
            run_method(method, root, ids, tasks, bm25_tasks, catalog, harness, config, args.workers)


if __name__ == '__main__':
    main()
