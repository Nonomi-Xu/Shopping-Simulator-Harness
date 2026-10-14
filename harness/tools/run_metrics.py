"""Fine-grained run metrics that the summary report does not carry.

The paired report answers "did it succeed".  These metrics answer "where did the
run spend its turns": first-argument pass rate per tool, rejection and recovery
rates, evidence reads, repeated actions without new information, retries, and
the token/latency distribution.  They are read from the per-task audit trail, so
a partial or blocked batch is measurable too.

Run from harness:

    python -m tools.run_metrics --run-dir /absolute/path/to/a/benchmark/run
    python -m tools.run_metrics --run-dir /tmp/replay_probe --label replay-probe
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.harness.agent.audit import write_json

#: Tools whose whole purpose is to read back already observed fields.
EVIDENCE_TOOLS = {'read_evidence', 'record_evidence', 'record_evidence_batch',
                  'compare_products', 'check_selection', 'submit_recommendations'}


def quantile(values, p):
    if not values:
        return None
    values = sorted(values)
    point = (len(values) - 1) * p
    lower = int(point)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (point - lower)


def arm_directories(root: Path, include: str | None) -> list[Path]:
    """Return one audit directory per (task, arm), preferring the newest attempt."""
    found: dict[tuple[str, str], Path] = {}
    if (root / 'events.jsonl').exists():
        return [root]
    tasks_root = root / 'tasks'
    if not tasks_root.is_dir():
        raise ValueError(f'{root} has no tasks/ or events.jsonl')
    for task_dir in sorted(tasks_root.iterdir()):
        if not task_dir.is_dir():
            continue
        for arm_dir in sorted(task_dir.iterdir(), reverse=True):
            if not arm_dir.is_dir() or not (arm_dir / 'events.jsonl').exists():
                continue
            name = arm_dir.name
            key = (task_dir.name, name.split('_attempt_')[0])
            if include and include not in name:
                continue
            if key not in found:
                found[key] = arm_dir
    return [found[key] for key in sorted(found)]


def read_events(directory: Path) -> list[dict]:
    return [json.loads(line) for line in (directory / 'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def analyse_arm(directory: Path) -> dict:
    events = read_events(directory)
    model_io = read_jsonl(directory / 'model_io.jsonl')
    result = (json.loads((directory / 'result.json').read_text(encoding='utf-8'))
              if (directory / 'result.json').exists() else {})
    turns = [e for e in events if e['kind'] == 'model_turn']
    actions = [e for e in events if e['kind'] == 'action']
    observations = [e for e in events if e['kind'] == 'observation']
    obs_by_call = {o['payload'].get('action_id'): o['payload'] for o in observations}
    by_tool: dict[str, Counter] = {}
    rejections = 0
    rejected_then_succeeded: set[str] = set()
    succeeded: set[str] = set()
    rejected_before: set[str] = set()
    for action in actions:
        payload = action['payload']
        name = payload.get('tool_name') or 'unknown'
        counters = by_tool.setdefault(name, Counter())
        counters['calls'] += 1
        observation = obs_by_call.get(payload.get('action_id')) or {}
        if observation.get('status') == 'success':
            counters['success'] += 1
            succeeded.add(name)
            if name in rejected_before:
                rejected_then_succeeded.add(name)
        else:
            counters['rejected'] += 1
            rejections += 1
            rejected_before.add(name)
    # A rejection that is later followed by a success of the same tool counts as
    # recovered.  Counting per tool avoids inflating the rate with unrelated
    # successful calls.
    recoveries = sum(by_tool[name]['rejected'] for name in rejected_then_succeeded)
    seen_tools: set[str] = set()
    for action in actions:
        name = action['payload'].get('tool_name') or 'unknown'
        if name in seen_tools:
            continue
        seen_tools.add(name)
        observation = obs_by_call.get(action['payload'].get('action_id')) or {}
        if observation.get('status') != 'success':
            by_tool[name]['first_call_rejected'] = 1
    batch_turns = [t['payload'] for t in turns if len(t.get('tool_calls') or []) > 1]
    refusals = Counter(o['payload'].get('error_code') for o in observations
                       if o['payload'].get('status') != 'success')
    # Repeated identical actions without new information: same tool and arguments
    # submitted again while an identical observation is already in the trace.
    signatures: Counter = Counter()
    repeats = 0
    for action in actions:
        payload = action['payload']
        signature = (payload.get('tool_name'), json.dumps(payload.get('arguments'), sort_keys=True, ensure_ascii=False))
        if signatures[signature]:
            repeats += 1
        signatures[signature] += 1
    errors = [r['payload'] for r in model_io if r['kind'] == 'error']
    usage_present = 0
    tokens = 0
    for turn in turns:
        usage = turn.get('usage')
        if usage:
            usage_present += 1
            tokens += usage.get('total_tokens') or 0
    return {
        'task': directory.parent.name, 'arm': directory.name,
        'terminal_reason': result.get('terminal_reason'),
        'status': result.get('status'),
        'model_turns': len(turns),
        'actions': len(actions),
        'batch_turns': len(batch_turns),
        'batch_calls': sum(len(t.get('tool_calls') or []) for t in batch_turns),
        'batch_refusals': sum(1 for o in observations
                              if o['payload'].get('error_code') in ('batch_not_supported', 'batch_limit_exceeded')),
        'refusal_reasons': dict(refusals),
        'by_tool': {k: dict(v) for k, v in sorted(by_tool.items())},
        'rejections': rejections,
        'recovered_rejections': recoveries,
        'repeated_actions_without_new_information': repeats,
        'evidence_read_calls': sum(1 for a in actions if (a['payload'].get('tool_name') or '') == 'read_evidence'),
        'evidence_read_rejections': sum(1 for a in actions
                                        if (a['payload'].get('tool_name') or '') == 'read_evidence'
                                        and (obs_by_call.get(a['payload'].get('action_id')) or {}).get('status') != 'success'),
        'transport_errors': len(errors),
        'transport_error_types': dict(Counter(e.get('type') for e in errors)),
        'max_transport_attempt': max([e.get('attempt') or 1 for e in errors], default=0),
        'turns_with_usage': usage_present,
        'total_tokens_in_turns': tokens,
        'usage_complete': usage_present == len(turns) and len(turns) > 0,
    }


def aggregate(arms: list[dict], label: str, root: Path) -> dict:
    n = len(arms)
    def rate(key):
        return sum(1 for a in arms if a.get(key)) / n
    by_tool: dict[str, Counter] = {}
    for arm in arms:
        for tool, counters in arm['by_tool'].items():
            by_tool.setdefault(tool, Counter()).update(counters)
            by_tool[tool]['task_arms_with_tool'] += 1
    tokens = [a['total_tokens_in_turns'] for a in arms if a['total_tokens_in_turns']]
    return {
        'label': label, 'run_dir': str(root.resolve()),
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'task_arms': n,
        'terminal_reasons': dict(Counter(a['terminal_reason'] for a in arms)),
        'statuses': dict(Counter(a['status'] for a in arms)),
        'mean_model_turns': statistics.mean([a['model_turns'] for a in arms]) if n else None,
        'mean_actions': statistics.mean([a['actions'] for a in arms]) if n else None,
        'batch_turn_arms': sum(1 for a in arms if a['batch_turns'] > 0),
        'batch_refusals': sum(a['batch_refusals'] for a in arms),
        'rejections_total': sum(a['rejections'] for a in arms),
        'recovered_rejections': sum(a['recovered_rejections'] for a in arms),
        'recovery_rate_given_rejection': (sum(a['recovered_rejections'] for a in arms)
                                          / sum(a['rejections'] for a in arms)
                                          if sum(a['rejections'] for a in arms) else None),
        'repeated_actions_without_new_information': sum(a['repeated_actions_without_new_information'] for a in arms),
        'evidence_read_calls': sum(a['evidence_read_calls'] for a in arms),
        'evidence_read_rejections': sum(a['evidence_read_rejections'] for a in arms),
        'history_arms_with_a_transport_error': sum(1 for a in arms if a['transport_errors'] > 0),
        'max_transport_attempt': max([a['max_transport_attempt'] for a in arms], default=0),
        'transport_error_types': dict(Counter(t for a in arms for t in a['transport_error_types'])),
        'usage_complete_arms': sum(1 for a in arms if a['usage_complete']),
        'usage_coverage': sum(1 for a in arms if a['usage_complete']) / n if n else None,
        'total_tokens': {'sum_complete': sum(tokens),
                         'mean_complete': statistics.mean(tokens) if tokens else None,
                         'p50': quantile(tokens, .5), 'p95': quantile(tokens, .95)},
        'tools': {tool: dict(counters) for tool, counters in sorted(by_tool.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--label', default=None)
    parser.add_argument('--include', default=None,
                        help='Only arms whose directory name contains this text, e.g. transport_fault')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    directories = arm_directories(args.run_dir, args.include)
    if not directories:
        raise SystemExit(f'no audit directories under {args.run_dir}')
    arms = [analyse_arm(directory) for directory in directories]
    label = args.label or args.run_dir.name
    summary = aggregate(arms, label, args.run_dir)
    summary['arms'] = arms
    summary['arm_directories'] = [str(d) for d in directories]
    output = args.output or (args.run_dir / 'run_metrics.json')
    write_json(output, summary)
    print(json.dumps({'metrics': str(output), 'task_arms': summary['task_arms'],
                      'terminal_reasons': summary['terminal_reasons'],
                      'recovery_rate_given_rejection': summary['recovery_rate_given_rejection'],
                      'usage_coverage': summary['usage_coverage']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
