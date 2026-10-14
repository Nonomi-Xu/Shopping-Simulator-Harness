"""Freeze regression and development task subsets from historical journals.

The subsets are selected by *observable failure class* and by a fixed-stride
stratified draw, never by how a future run turns out.  Every manifest records
the parent dataset hash, the selection rule, and the exact task IDs, so a later
batch can prove it ran the same tasks.

Run from harness:

    python -m tools.build_subsets \
        --harness-run ../outputs/benchmarks/Test1_harness_rerun_20261013 \
        --bm25-run ../outputs/benchmarks/Test1_20261013 \
        --output-dir ../outputs/experiments/exp-20261014-qwen3/subsets

Both historical directories are read-only inputs; nothing here calls a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.harness.agent.events import digest
from src.harness.benchmark.datasets import load, sha256_file
from src.harness.paths import PROJECT_DIR
from src.harness.tasks import TaskStore

SUBSET_VERSION = 'shopping-task-subset-v1'


def read_journal(run_dir: Path, method: str) -> list[dict]:
    path = run_dir / method / 'records.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    completed = json.loads((run_dir / method / 'completed.json').read_text())
    if completed['records_sha256'] != sha256_file(path):
        raise ValueError(f'{path} does not match its completed.json hash')
    return rows


def error_text(row: dict) -> str:
    return f"{row.get('error_type') or ''} {row.get('error_message') or ''}"


def select_rows(harness_rows: list[dict], bm25_rows: list[dict]) -> dict[str, list[dict]]:
    """Return the named selections, each already in frozen-dataset order."""
    bm25 = {r['task_id']: r for r in bm25_rows}
    by_index = sorted(harness_rows, key=lambda r: r['index'])
    return {
        # The old run aborted as soon as one assistant turn carried more than one
        # tool call.  These are exactly those turns, split by whether the BM25
        # baseline had already retrieved the gold product: the first group is
        # where the protocol fix has a measurable ceiling.
        'batch_regression': [r for r in by_index
                             if r['terminal_reason'] == 'multiple_tool_calls'
                             and bm25[r['task_id']]['retrieval_hit']],
        'batch_regression_no_recall': [r for r in by_index
                                       if r['terminal_reason'] == 'multiple_tool_calls'
                                       and not bm25[r['task_id']]['retrieval_hit']],
        'context_regression': [r for r in by_index if r['terminal_reason'] == 'context_budget_exhausted'],
        'invalid_tool_json_regression': [r for r in by_index
                                         if r['terminal_reason'] == 'provider_error'
                                         and 'invalid JSON arguments' in error_text(r)],
        'connection_regression': [r for r in by_index
                                  if r['terminal_reason'] == 'runtime_error'
                                  and 'IncompleteRead' in error_text(r)],
        'success_control': [r for r in by_index if r['strict_at_1']],
    }


def take(rows: list[dict], count: int, *, offset: int = 0) -> list[dict]:
    """Take a deterministic, evenly spaced sample so the pick is reproducible."""
    if count >= len(rows):
        return list(rows)
    stride = len(rows) / count
    picked, seen = [], set()
    for slot in range(count):
        row = rows[min(int(offset + slot * stride), len(rows) - 1)]
        if row['task_id'] not in seen:
            seen.add(row['task_id'])
            picked.append(row)
    for row in rows:
        if len(picked) >= count:
            break
        if row['task_id'] not in seen:
            seen.add(row['task_id'])
            picked.append(row)
    return picked


def ordering_key(frozen: dict) -> dict[str, int]:
    return {tid: i for i, tid in enumerate(frozen['task_ids'])}


def manifest(frozen: dict, name: str, selection: str, ids: list[str]) -> dict:
    ordered = sorted(set(ids), key=ordering_key(frozen).__getitem__)
    return {'version': SUBSET_VERSION, 'name': name,
            'parent_dataset': frozen['name'], 'parent_sha256': digest(frozen),
            'source_sha256': frozen['source_sha256'],
            'selection': selection, 'task_count': len(ordered), 'task_ids': ordered}


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != payload:
            raise ValueError(f'frozen subset cannot be overwritten: {path}')
        return
    with path.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='Test1')
    parser.add_argument('--harness-run', type=Path, required=True)
    parser.add_argument('--bm25-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    store = TaskStore(PROJECT_DIR / 'harness/data/fine_items_eval_train_all.json')
    frozen = load(PROJECT_DIR / f'datasets/{args.dataset}.json', store)
    harness_rows = read_journal(args.harness_run, 'harness')
    bm25_rows = read_journal(args.bm25_run, 'bm25')
    if {r['task_id'] for r in harness_rows} != {r['task_id'] for r in bm25_rows}:
        raise ValueError('historical runs cover different tasks')
    classes = select_rows(harness_rows, bm25_rows)

    plans = {
        'regression-v1': (
            'engineering regression: 12 old multi-tool-call aborts where BM25 had recalled the gold product, '
            '6 context-budget aborts, 2 invalid tool-argument provider errors, 2 connection IncompleteRead '
            'failures, and 8 old strict successes as the no-regression control',
            [r['task_id'] for r in take(classes['batch_regression'], 12)]
            + [r['task_id'] for r in take(classes['context_regression'], 6)]
            + [r['task_id'] for r in take(classes['invalid_tool_json_regression'], 2)]
            + [r['task_id'] for r in take(classes['connection_regression'], 2)]
            + [r['task_id'] for r in take(classes['success_control'], 8, offset=3)],
        ),
        'replay-failures-v1': (
            'complete historical failure population for offline replay: every multiple_tool_calls abort, '
            'every context-budget abort, every runtime_error and every provider_error, plus 20 fixed-stride '
            'old successes as the no-regression control',
            [r['task_id'] for r in classes['batch_regression']]
            + [r['task_id'] for r in classes['batch_regression_no_recall']]
            + [r['task_id'] for r in classes['context_regression']]
            + [r['task_id'] for r in classes['invalid_tool_json_regression']]
            + [r['task_id'] for r in classes['connection_regression']]
            + [r['task_id'] for r in take(classes['success_control'], 20)],
        ),
        'dev-v1': (
            'fixed development subset, stratified by historical outcome and independent of this run: '
            'a fixed-stride draw of 12 old strict successes and 28 old failures, with the failure half '
            'ordered so a prefix still has failures if the batch must stop early',
            [r['task_id'] for r in take(classes['success_control'], 12)]
            + [r['task_id'] for r in take([r for r in harness_rows if not r['strict_at_1']], 28)],
        ),
    }
    counts = {}
    for name, (selection, ids) in plans.items():
        payload = manifest(frozen, name, selection, ids)
        write(args.output_dir / f'{name}.json', payload)
        counts[name] = subset_counters(payload['task_ids'], harness_rows, bm25_rows)
    (args.output_dir / 'selection_audit.json').parent.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'selection_audit.json').write_text(json.dumps({
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': frozen['name'], 'dataset_sha256': digest(frozen),
        'harness_run': str(args.harness_run.resolve()), 'bm25_run': str(args.bm25_run.resolve()),
        'histories': {
            'harness_terminal_reasons': dict(Counter(r['terminal_reason'] for r in harness_rows)),
            'harness_strict_at_1': sum(r['strict_at_1'] for r in harness_rows),
            'candidate_sizes': {k: len(v) for k, v in classes.items()},
        },
        'subsets': counts,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output_dir': str(args.output_dir.resolve()), 'subsets': {k: v['task_count'] for k, v in counts.items()}}, ensure_ascii=False))


def subset_counters(ids: list[str], harness_rows: list[dict], bm25_rows: list[dict]) -> dict:
    harness = {r['task_id']: r for r in harness_rows}
    bm25 = {r['task_id']: r for r in bm25_rows}
    return {
        'task_count': len(ids),
        'old_harness_terminal_reasons': dict(Counter(harness[i]['terminal_reason'] for i in ids)),
        'old_harness_strict_at_1': sum(harness[i]['strict_at_1'] for i in ids),
        'old_bm25_terminal_reasons': dict(Counter(bm25[i]['terminal_reason'] for i in ids)),
        'old_bm25_retrieval_hit': sum(bm25[i]['retrieval_hit'] for i in ids),
    }


if __name__ == '__main__':
    main()
