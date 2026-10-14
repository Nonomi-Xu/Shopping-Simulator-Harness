"""Replay historical failure traces against the current runtime.

The 20261013 Test1 batch failed 105 tasks without a submission: 67 assistant
turns carried more than one tool call, 10 responses were truncated
(``IncompleteRead``), and several evidence reads could not see a field that an
earlier turn had already observed.  Those old runs are the only recordings of
the exact model output that used to break the runtime, so this tool feeds them
back through today's code.

Two arms per task:

``replay``
    The recorded assistant turns are served verbatim through the real
    ``OpenAICompatibleProvider`` transport machinery (a stub opener replaces the
    network).  No model calls, no cost.  When the trace runs out the run ends
    through the normal contract: one explicit terminal tool call.

``transport_fault``
    The same trace, but the opener fails the first two HTTP attempts on every
    turn: first a truncated body (``IncompleteRead``), then an undecodable
    envelope.  This is the regression for the connection-retry fix, and it
    cannot be produced by the live API on demand.

Every arm is scored for the runtime behaviours under test, including whether a
multi-call turn was executed instead of aborting the run.

Run from harness:

    python -m tools.replay_regression \
        --run /absolute/path/to/Test1_harness_rerun_20261013 \
        --subset ../outputs/experiments/exp-20261014-qwen3/subsets/regression-v1.json \
        --output-dir ../outputs/experiments/exp-20261014-qwen3/replay-regression
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from http.client import IncompleteRead
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.harness.agent import AgentRunner, RuntimeConfig
from src.harness.agent.audit import RunAudit, write_json
from src.harness.agent.skill import load_agent_skill
from src.harness.benchmark.datasets import load, load_subset, sha256_file
from src.harness.harness import ShoppingHarness
from src.harness.paths import PROJECT_DIR
from src.harness.providers import DeepSeekProvider, ToolCall
from src.harness.tasks import TaskStore

#: A replay cannot invent a model turn.  When the recorded trace runs out the
#: run ends through the normal contract: one explicit terminal tool call.
TERMINAL_TOOL = 'submit_recommendations'


def _recorded_responses(run_dir: Path, index: int) -> list[dict]:
    """Return the raw recorded response envelopes for one historical task."""
    path = run_dir / 'harness' / 'tasks' / f'{index:06d}' / 'attempt_01' / 'model_io.jsonl'
    bodies = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record['kind'] == 'response':
            bodies.append(record['payload']['body'])
    # A task whose first request was truncated has no recorded response at all
    # (task 000141 in the 20261013 batch).  Replaying it as an empty trace still
    # exercises the retry path through the injected faults.
    return bodies


class StreamOpener:
    """Stub HTTP opener serving recorded JSON bodies.

    ``inject_faults`` fails the first two attempts of *every* provider request:
    a truncated body first, then a syntactically broken envelope.  Both are
    transport faults that historically ended a task without an evaluation.
    Each attempt returns a fresh stream object, because a retry counts attempts
    by how often the opener is called, not by how often one object is entered.
    """

    #: Attempts to fail per request before serving the recorded body.  The
    #: provider retries on the next attempt, so exactly two injected faults prove
    #: the retry result is used rather than merely logged.
    FAULTS_PER_REQUEST = 2

    def __init__(self, bodies: list[dict], *, inject_faults: bool = False):
        self._bodies = bodies
        self._inject = inject_faults
        self._index = 0
        self._faults_by_type: dict[str, int] = {}
        self.requests = 0

    def __call__(self, request, *, timeout):
        self.requests += 1
        attempt = (self.requests - 1) % (self.FAULTS_PER_REQUEST + 1) if self._inject else self.FAULTS_PER_REQUEST
        if attempt < self.FAULTS_PER_REQUEST:
            if attempt == 0:
                self._faults_by_type['IncompleteRead'] = self._faults_by_type.get('IncompleteRead', 0) + 1
                return _FaultyStream(lambda: IncompleteRead(b'{"choices": [', 4096))
            self._faults_by_type['JSONDecodeError'] = self._faults_by_type.get('JSONDecodeError', 0) + 1
            return _FaultyStream(lambda: json.JSONDecodeError('truncated envelope', '{"choices"', 9))
        if self._index < len(self._bodies):
            body = self._bodies[self._index]
            self._index += 1
        else:
            body = _terminal_envelope(self._index)
            self._index += 1
        return _BodyStream(body)

    @property
    def faults_by_type(self) -> dict[str, int]:
        return dict(self._faults_by_type)


class _FaultyStream:
    """Context manager that raises ``error`` on entry, like a broken socket."""

    def __init__(self, error_factory):
        self._error_factory = error_factory

    def __enter__(self):
        raise self._error_factory()

    def __exit__(self, *exc):
        return False


class _BodyStream:
    """Context manager returning one recorded JSON body."""

    def __init__(self, body: dict):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return json.dumps(self._body, ensure_ascii=False).encode('utf-8')


def _terminal_envelope(sequence: int) -> dict:
    call = ToolCall(id=f'replay_end_{sequence}', name=TERMINAL_TOOL,
                    arguments={'recommendations': []}, raw_arguments='{"recommendations": []}')
    return {'id': f'replay_end_{sequence}', 'model': 'recorded-deepseek-v4-flash',
            'choices': [{'finish_reason': 'tool_calls', 'message': {
                'role': 'assistant', 'content': None,
                'tool_calls': [{'index': 0, 'id': call.id, 'type': 'function',
                                'function': {'name': call.name, 'arguments': call.raw_arguments}}]}}]}


def build_provider(bodies: list[dict], audit: RunAudit, *, inject_faults: bool):
    opener = StreamOpener(bodies, inject_faults=inject_faults)
    provider = DeepSeekProvider(api_key='replay-not-a-real-key', model='recorded-deepseek-v4-flash',
                                base_url='https://replay.invalid/v1', timeout=5, max_retries=2,
                                retry_backoff=0, opener=opener, sleeper=lambda _: None,
                                request_observer=audit.transport)
    return provider, opener


def analyse(directory: Path, record: dict, historical_row: dict, history_root: Path) -> dict:
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    model_io_path = directory / 'model_io.jsonl'
    model_io = ([json.loads(line) for line in model_io_path.read_text(encoding='utf-8').splitlines() if line.strip()]
                if model_io_path.exists() else [])
    turns = [e['payload'] for e in events if e['kind'] == 'model_turn']
    observations = [e['payload'] for e in events if e['kind'] == 'observation']
    batch_turns = [t for t in turns if len(t.get('tool_calls') or []) > 1]
    history_path = history_root / historical_row['artifact_dir'] / 'events.jsonl'
    historical = [json.loads(line) for line in history_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    historical_turn_sizes = [len(e['payload'].get('tool_calls') or []) for e in historical if e['kind'] == 'model_turn']
    read_evidence = [o for o in observations if o.get('tool_name') == 'read_evidence']
    errors = [r['payload'] for r in model_io if r['kind'] == 'error']
    return {
        'task_id': record['task_id'], 'index': record['index'],
        'historical_terminal_reason': historical_row['terminal_reason'],
        'historical_batch_turns': sum(1 for n in historical_turn_sizes if n > 1),
        'historical_max_calls_in_turn': max(historical_turn_sizes, default=0),
        'replay_status': record.get('replay_status'),
        'replay_terminal_reason': record.get('replay_terminal_reason'),
        'replay_batch_turns_executed': len(batch_turns),
        'replay_batch_calls_executed': sum(len(t.get('tool_calls') or []) for t in batch_turns),
        'batch_not_supported_refusals': sum(1 for o in observations if o.get('error_code') == 'batch_not_supported'),
        'batch_limit_exceeded_refusals': sum(1 for o in observations if o.get('error_code') == 'batch_limit_exceeded'),
        'read_evidence_calls': len(read_evidence),
        'read_evidence_failures': sum(1 for o in read_evidence if o.get('status') != 'success'),
        'transport_error_events': len(errors),
        'incomplete_read_events': sum(1 for e in errors if e.get('type') == 'IncompleteRead'),
        'json_error_events': sum(1 for e in errors if e.get('type') == 'JSONDecodeError'),
        'max_error_attempt': max([e.get('attempt') or 1 for e in errors], default=0),
        'usage_complete': record.get('usage_complete'),
    }


def replay_task(harness: ShoppingHarness, task_id: str, index: int, bodies: list[dict], directory: Path,
                *, inject_faults: bool, max_steps: int, max_model_turns: int, session_id: str) -> dict:
    # RunAudit refuses to overwrite an existing arm, so a re-run keeps the previous
    # attempt beside the new one instead of silently replacing evidence.
    if directory.exists():
        attempt = 2
        while (attempt_dir := directory.with_name(f'{directory.name}_attempt_{attempt:02d}')).exists():
            attempt += 1
        directory = attempt_dir
    audit = RunAudit(directory)
    provider, opener = build_provider(bodies, audit, inject_faults=inject_faults)
    try:
        result = AgentRunner(
            provider=provider, harness=harness,
            config=RuntimeConfig(), max_model_turns=max_model_turns,
            tool_choice='auto', event_observer=audit,
        ).run(task_id, max_steps=max_steps, session_id=session_id)
        audit.finish(result)
        return {'task_id': task_id, 'index': index, 'artifact_dir': str(audit.directory),
                'replay_status': result.status,
                'replay_terminal_reason': result.terminal_reason,
                'replay_error_type': result.error_type,
                'opener_requests': opener.requests,
                'usage_complete': bool(result.usage) and all(t.usage is not None for t in result.turns)}
    except Exception as error:  # an escaped runtime exception is itself a finding
        write_json(directory / 'error.json', {'type': type(error).__name__, 'message': str(error)})
        return {'task_id': task_id, 'index': index, 'artifact_dir': str(audit.directory),
                'replay_status': 'error', 'replay_terminal_reason': 'runtime_error',
                'replay_error_type': type(error).__name__, 'replay_error_message': str(error),
                'opener_requests': opener.requests, 'usage_complete': False}
    finally:
        try:
            harness.release_session(session_id, force=True)
        except LookupError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Historical Test1 run root')
    parser.add_argument('--subset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset', default='Test1')
    parser.add_argument('--max-steps', type=int, default=20)
    parser.add_argument('--max-model-turns', type=int, default=40)
    parser.add_argument('--limit', type=int, help='Debug only: replay the first N subset tasks')
    args = parser.parse_args()

    products = PROJECT_DIR / 'harness/data/fine_items_eval_train_all.json'
    index = PROJECT_DIR / 'harness/data/product_search.sqlite3'
    store = TaskStore(products)
    frozen = load(PROJECT_DIR / f'datasets/{args.dataset}.json', store)
    subset = load_subset(args.subset, frozen)
    ids = list(subset['task_ids'])
    if args.limit:
        ids = ids[:args.limit]
    rows = {json.loads(line)['task_id']: json.loads(line)
            for line in (args.run / 'harness/records.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / 'replay_manifest.json', {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'historical_run': str(args.run.resolve()), 'subset': subset['name'],
        'subset_sha256': sha256_file(args.subset), 'task_ids': ids,
        'runtime_config': RuntimeConfig().model_dump(mode='json'),
        'skill_sha256': load_agent_skill().source_sha256,
        'arms': {'replay': 'recorded responses served through the real provider transport',
                 'transport_fault': 'same trace with two injected transport faults per request'},
        'note': 'recorded model output replayed; no model calls and no token cost',
    })

    results = []
    with ShoppingHarness(products_path=products, index_path=index) as harness:
        for position, task_id in enumerate(ids, start=1):
            row = rows[task_id]
            history_index = row['index']
            bodies = _recorded_responses(args.run, history_index)
            arms = {}
            for arm, inject in [('replay', False), ('transport_fault', True)]:
                directory = args.output_dir / 'tasks' / f'{history_index:06d}' / arm
                record = replay_task(harness, task_id, history_index, bodies, directory, inject_faults=inject,
                                     max_steps=args.max_steps, max_model_turns=args.max_model_turns,
                                     session_id=f'replay_{history_index:06d}_{arm}')
                arms[arm] = analyse(Path(record['artifact_dir']), record, row, args.run)
            results.append({'task_id': task_id, 'index': history_index,
                            'historical_terminal_reason': row['terminal_reason'], 'arms': arms})
            print(json.dumps({'replayed': position, 'total': len(ids), 'task_id': task_id,
                              'replay': arms['replay']['replay_terminal_reason'],
                              'fault_arm': arms['transport_fault']['replay_terminal_reason'],
                              'batch_turns': arms['replay']['replay_batch_turns_executed'],
                              'batch_refusals': arms['replay']['batch_not_supported_refusals'],
                              'retries': arms['transport_fault']['transport_error_events'],
                              'evidence_failures': arms['replay']['read_evidence_failures']},
                             ensure_ascii=False), flush=True)

    write_json(args.output_dir / 'replay_results.json', results)
    summary = summarize(results)
    write_json(args.output_dir / 'replay_summary.json', summary)
    (args.output_dir / 'report.md').write_text(render(summary, subset, args), encoding='utf-8')
    print(json.dumps({'output_dir': str(args.output_dir.resolve()), 'verdict': summary['verdict']}, ensure_ascii=False))


def summarize(results: list[dict]) -> dict:
    replay = [r['arms']['replay'] for r in results]
    fault = [r['arms']['transport_fault'] for r in results]
    old_batch = [r for r in results if r['historical_terminal_reason'] == 'multiple_tool_calls']
    old_conn = [r for r in results if r['historical_terminal_reason'] == 'runtime_error']
    return {
        'task_count': len(results),
        'historical_classes': dict(Counter(r['historical_terminal_reason'] for r in results)),
        'verdict': {
            'old_batch_aborts_replayed': len(old_batch),
            'old_batch_aborts_still_ending_in_multiple_tool_calls':
                sum(1 for r in old_batch if r['arms']['replay']['replay_terminal_reason'] == 'multiple_tool_calls'),
            'old_batch_aborts_executing_a_multi_call_turn':
                sum(1 for r in old_batch if r['arms']['replay']['replay_batch_turns_executed'] > 0),
            'old_batch_aborts_executing_the_historical_batch_size':
                sum(1 for r in old_batch if r['arms']['replay']['replay_batch_calls_executed']
                    >= r['arms']['replay']['historical_max_calls_in_turn']),
            'batch_refusals_any_kind': sum(r['arms']['replay']['batch_not_supported_refusals']
                                           + r['arms']['replay']['batch_limit_exceeded_refusals'] for r in results),
            'read_evidence_calls': sum(r['arms']['replay']['read_evidence_calls'] for r in results),
            'read_evidence_failures': sum(r['arms']['replay']['read_evidence_failures'] for r in results),
            'old_connection_failures_replayed': len(old_conn),
            'fault_arm_tasks_with_a_retry_event': sum(1 for r in fault if r['transport_error_events'] > 0),
            'fault_arm_tasks_without_a_retry_event': sum(1 for r in fault if r['transport_error_events'] == 0),
            'fault_arm_max_attempt_reached': max([r['max_error_attempt'] for r in fault], default=0),
            'fault_arm_incomplete_read_events': sum(r['incomplete_read_events'] for r in fault),
            'fault_arm_json_envelope_events': sum(r['json_error_events'] for r in fault),
            'fault_arm_tasks_reaching_a_normal_terminal':
                sum(1 for r in fault if r['replay_terminal_reason'] not in ('provider_error', 'runtime_error')),
            'replay_terminal_reasons': dict(Counter(r['replay_terminal_reason'] for r in replay)),
            'fault_arm_terminal_reasons': dict(Counter(r['replay_terminal_reason'] for r in fault)),
            'runtime_exceptions': sum(1 for r in replay if r['replay_status'] == 'error'),
            'fault_arm_runtime_exceptions': sum(1 for r in fault if r['replay_status'] == 'error'),
        },
    }


def render(summary: dict, subset: dict, args) -> str:
    v = summary['verdict']
    lines = [
        '# 历史失败轨迹回放回归（无模型调用、无 Token 费用）', '',
        f"- 历史批次：`{Path(args.run).resolve()}`",
        f"- 子集：`{subset['name']}`（{summary['task_count']} 题）",
        f"- 选择规则：{subset['selection']}",
        '- 回放的是历史真实模型输出，经当前 Provider 传输层（替换 opener）执行；回放耗尽后以一次显式终局工具调用结束，不伪造模型轮次。',
        '- `transport_fault` 组在每次模型请求的前两次尝试注入截断响应与损坏信封，用于验证连接重试；该故障无法在真实 API 上按需复现。', '',
        '## 结论', '',
        f"- 旧 `multiple_tool_calls` 终止的 {v['old_batch_aborts_replayed']} 题，回放后仍以该原因终止：**{v['old_batch_aborts_still_ending_in_multiple_tool_calls']}** 题。",
        f"- 其中真正执行了多调用轮次的：**{v['old_batch_aborts_executing_a_multi_call_turn']}** 题；执行的调用数达到历史批大小的：**{v['old_batch_aborts_executing_the_historical_batch_size']}** 题。",
        f"- 全部回放中的批量拒绝（`batch_not_supported` / `batch_limit_exceeded`）：**{v['batch_refusals_any_kind']}** 次。",
        f"- 旧连接中断（`runtime_error`）题 {v['old_connection_failures_replayed']} 题。", '',
        f"- 注入故障组：记录了重试事件的 {v['fault_arm_tasks_with_a_retry_event']} 题，缺失 {v['fault_arm_tasks_without_a_retry_event']} 题；",
        f"  最大尝试次数 {v['fault_arm_max_attempt_reached']}，其中 `IncompleteRead` {v['fault_arm_incomplete_read_events']} 次、损坏信封 {v['fault_arm_json_envelope_events']} 次；",
        f"  走到正常终局（非 `provider_error`/`runtime_error`）{v['fault_arm_tasks_reaching_a_normal_terminal']} 题。", '',
        f"- `read_evidence` 调用 {v['read_evidence_calls']} 次，非成功 {v['read_evidence_failures']} 次。", '',
        f"- 回放组终局分布：`{json.dumps(v['replay_terminal_reasons'], ensure_ascii=False)}`。",
        f"- 注入组终局分布：`{json.dumps(v['fault_arm_terminal_reasons'], ensure_ascii=False)}`。",
        f"- 逃逸的运行期异常：回放组 {v['runtime_exceptions']} 题，注入组 {v['fault_arm_runtime_exceptions']} 题。", '',
        '## 历史类别分布', '', '| 历史终止原因 | 题数 |', '|---|---:|',
    ]
    for reason, count in sorted(summary['historical_classes'].items(), key=lambda kv: -kv[1]):
        lines.append(f'| {reason} | {count} |')
    lines += ['', '本报告只验证运行时行为（批量执行、证据读取、传输重试），不代表模型质量或总体成绩。', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
