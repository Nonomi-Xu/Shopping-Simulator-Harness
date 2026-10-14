"""Immutable source-order task manifests. No private gold facts enter the manifest."""
import argparse
import hashlib
import json
from pathlib import Path

from ..agent.events import digest
from ..paths import PROJECT_DIR
from ..tasks import TaskStore, TASK_ID_VERSION


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(store, path, name, start=1, count=None):
    ids = store.list_task_ids('Original')
    if start < 1 or (count is not None and count < 1):
        raise ValueError('start and count must be positive')
    end = len(ids) if count is None else start - 1 + count
    if end > len(ids) or start > len(ids):
        raise ValueError('selection exceeds Original; refusing to freeze a smaller dataset')
    selected = ids[start - 1:end]
    payload = {
        'version': 'shopping-frozen-dataset-v1', 'name': name,
        'source_sha256': store.product_data_sha256,
        'task_id_version': TASK_ID_VERSION, 'original_count': len(ids),
        'selection': {'method': 'source_order', 'start_1based': start, 'end_1based': end},
        'task_count': len(selected), 'task_ids': selected,
    }
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f'frozen dataset cannot be overwritten: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
    return payload


def load(path, store):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload['version'] != 'shopping-frozen-dataset-v1':
        raise ValueError('unsupported dataset version')
    if payload['source_sha256'] != store.product_data_sha256:
        raise ValueError('frozen source hash mismatch')
    original = store.list_task_ids('Original')
    selection = payload['selection']
    ids = payload['task_ids']
    if (payload['task_id_version'] != TASK_ID_VERSION or
        payload['original_count'] != len(original) or
        payload['task_count'] != len(ids) or len(ids) != len(set(ids)) or not ids or
        selection['method'] != 'source_order' or selection['start_1based'] < 1 or
        selection['end_1based'] > len(original) or
        ids != original[selection['start_1based'] - 1:selection['end_1based']]):
        raise ValueError('frozen dataset membership/order mismatch')
    return payload


def load_subset(path, frozen):
    """Load an explicit task subset drawn from a frozen dataset.

    ``--limit`` can only take a prefix, which cannot express a stratified or
    failure-selected development set.  A subset manifest is a frozen, hashable
    selection instead: it records the parent dataset, the selection rule, and the
    exact ordered task IDs, and it is validated against the parent dataset so a
    subset can never introduce a task the parent does not contain.
    """
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('version') != 'shopping-task-subset-v1':
        raise ValueError('unsupported task subset version')
    ids = payload.get('task_ids')
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) and i for i in ids):
        raise ValueError('task subset needs a non-empty list of task IDs')
    if len(ids) != len(set(ids)):
        raise ValueError('task subset contains duplicate task IDs')
    if payload.get('parent_dataset') != frozen['name']:
        raise ValueError('task subset parent dataset mismatch')
    if payload.get('parent_sha256') != digest(frozen):
        raise ValueError('task subset parent hash mismatch')
    if not set(ids) <= set(frozen['task_ids']):
        raise ValueError('task subset is not contained in the frozen dataset')
    if not isinstance(payload.get('name'), str) or not payload['name']:
        raise ValueError('task subset needs a name')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--products-path', type=Path, default=PROJECT_DIR/'harness/data/fine_items_eval_train_all.json')
    parser.add_argument('--name', default='Test1')
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--count', type=int, default=500)
    args = parser.parse_args()
    if not args.name.isalnum():
        parser.error('dataset name must be alphanumeric')
    store = TaskStore(args.products_path)
    original = freeze(store, PROJECT_DIR/'datasets/Original.json', 'Original')
    subset = freeze(store, PROJECT_DIR/f'datasets/{args.name}.json', args.name, args.start, args.count)
    print(json.dumps({'Original': original['task_count'], args.name: subset['task_count'], 'source_sha256': original['source_sha256']}))


if __name__ == '__main__':
    main()
