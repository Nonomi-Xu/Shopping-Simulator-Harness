"""Offline recall curve using exactly the baseline tokenizer, weights and tie order."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

from ..paths import PROJECT_DIR
from ..tasks import TaskStore
from ..agent.audit import write_json
from .datasets import load
sys.path.insert(0, str(PROJECT_DIR/'BM25'))
from bm25_agent.catalog import BM25Catalog, search_tokens, _fts_query


def analyze(run_dir):
    root=Path(run_dir)
    config=json.loads((root/'config.json').read_text())
    store=TaskStore(PROJECT_DIR/'harness/data/fine_items_eval_train_all.json')
    dataset=load(root/'dataset.json',store)
    ids=config['task_ids'];rows=[];categories=Counter()
    with BM25Catalog(PROJECT_DIR/'harness/data/product_search.sqlite3') as catalog:
        placeholders=', '.join('?' for _ in catalog.weights)
        for tid in ids:
            task=store.get_public_task(tid);facts=store.get_private_facts(tid)
            tokens=search_tokens(task.query)
            found=catalog._connection.execute(
                f'SELECT asin, bm25(products, {placeholders}) AS score FROM products WHERE products MATCH ? ORDER BY score ASC, asin ASC LIMIT 50',
                (*catalog.weights,_fts_query(tokens))).fetchall() if tokens else []
            candidates=[str(r['asin']) for r in found]
            # Detect accidental drift from the live BM25 implementation.
            if candidates[:5] != [c.asin for c in catalog.search_top5(task.query)]:
                raise ValueError('offline retrieval differs from the BM25 baseline')
            rank=candidates.index(facts.target_asin)+1 if facts.target_asin in candidates else None
            category=facts.target_category
            categories[category]+=1
            rows.append({'task_id':tid,'target_rank_top50':rank,'category':category,
                         'unresolved_gold':bool(facts.unresolved_option_values),
                         'required_option_axes':len(facts.required_options)})
    original_categories=Counter(store.get_private_facts(tid).target_category for tid in store.list_task_ids('Original'))
    out={'dataset':dataset['name'],'task_count':len(ids),
         'recall_at_k':{str(k):{'hits':sum(r['target_rank_top50'] is not None and r['target_rank_top50']<=k for r in rows),
                               'rate':sum(r['target_rank_top50'] is not None and r['target_rank_top50']<=k for r in rows)/len(rows)} for k in [1,5,10,20,50]},
         'category_counts':dict(categories),'original_category_counts':dict(original_categories),'tasks':rows}
    output=root/'analysis';output.mkdir(exist_ok=True)
    write_json(output/'retrieval_curve.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='tasks'},ensure_ascii=False))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    analyze(p.parse_args().run_dir)


if __name__=='__main__':main()
