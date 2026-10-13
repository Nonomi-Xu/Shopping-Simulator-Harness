"""Evidence-based, offline annotations of the completed paired benchmark.

Gold is loaded only after evaluation and never sent to a model. Labels describe
observable failure stages, not independently verified causal explanations.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from ..agent.audit import write_json
from ..paths import PROJECT_DIR
from ..tasks import TaskStore
from .datasets import load
from .report import failure_bucket, generate, quantile


LABELS = {
    'strict_at_1_success': '首件严格正确',
    'unverifiable_or_no_evaluation': '无有效评分',
    'target_not_retrieved': '未召回目标商品',
    'retrieved_but_not_inspected': '已召回但未查看目标详情',
    'target_selected_wrong_options': '首件商品正确但规格错误',
    'correct_only_at_later_rank': '正确答案未排在首位',
    'retrieved_but_wrong_selection': '已查看目标但选择错误',
    'runtime:multiple_tool_calls': '单轮多工具调用导致终止',
    'runtime:context_budget_exhausted': '上下文预算耗尽',
    'runtime:provider_error': '模型接口或响应解析失败',
    'runtime:runtime_error': '运行或传输异常',
    'runtime:max_actions': '动作预算耗尽',
    'runtime:max_model_turns': '模型轮次耗尽',
}


def transport_summary(path):
    counts, models, errors = Counter(), Counter(), Counter()
    usage = Counter()
    if path.exists():
        with path.open() as handle:
            for line in handle:
                event = json.loads(line)
                counts[event['kind']] += 1
                payload = event.get('payload', {})
                if event['kind'] == 'response':
                    body = payload.get('body', {})
                    if body.get('model'):
                        models[body['model']] += 1
                    for key, value in (body.get('usage') or {}).items():
                        if isinstance(value, (int, float)):
                            usage[key] += value
                elif event['kind'] == 'error':
                    errors[str(payload.get('status_code') or payload.get('type') or 'unknown')] += 1
    return {'event_counts': dict(counts), 'response_models': dict(models),
            'error_types': dict(errors), 'returned_usage': dict(usage)}


def annotate_method(row, root, target):
    directory = root / row['artifact_dir']
    bucket = failure_bucket(row)
    annotation = {
        'label': bucket, 'label_zh': LABELS.get(bucket, bucket),
        'strict_at_1': row['strict_at_1'], 'strict_any': row['strict_any'],
        'asin_at_1': row['asin_at_1'], 'retrieval_hit': row.get('retrieval_hit'),
        'inspected_hit': row.get('inspected_hit'), 'terminal_reason': row['terminal_reason'],
        'evaluation_reason': row.get('evaluation_reason'),
        'error_type': row.get('error_type'), 'error_message': row.get('error_message'),
        'evaluation': row.get('evaluation'), 'artifact_dir': str(directory),
        'searches': [], 'recommendations': [],
        'transport': transport_summary(directory / 'model_io.jsonl'),
    }
    if row['method'] == 'harness' and (directory / 'session.json').exists():
        session = json.loads((directory / 'session.json').read_text())
        for step in session['history']:
            if step['tool_name'] == 'search_products' and step['status'] == 'success':
                products = step['observation']
                if isinstance(products, dict):
                    products = products.get('products', products.get('results', []))
                asins = [p['asin'] for p in products if 'asin' in p]
                annotation['searches'].append({
                    'step': step['step_number'], 'arguments': step['arguments'],
                    'candidate_count': len(asins),
                    'target_rank_in_page': asins.index(target) + 1 if target in asins else None,
                })
            elif step['tool_name'] == 'submit_recommendations' and step['status'] == 'success':
                annotation['recommendations'] = step['arguments'].get('recommendations', [])
        if not annotation['recommendations']:
            annotation['recommendations'] = (session.get('state') or {}).get('recommendations', [])
    elif row['method'] == 'bm25' and (directory / 'result.json').exists():
        result = json.loads((directory / 'result.json').read_text())
        selector = result.get('selector')
        if selector:
            annotation['recommendations'] = [selector['decision']]
    annotation['missing_evaluation'] = not row.get('reward_valid')
    annotation['target_recalled_but_first_incorrect'] = bool(row.get('retrieval_hit') and not row['strict_at_1'])
    annotation['strict_correct_at_later_rank'] = bool(row.get('strict_any') and not row['strict_at_1'])
    return annotation


def grouped_comparison(annotations):
    groups = defaultdict(list)
    for item in annotations:
        b, h = (item['methods'][m]['strict_at_1'] for m in ['bm25', 'harness'])
        groups[item['target_asin']].append(int(h) - int(b))
    totals = [(sum(v), len(v)) for v in groups.values()]
    rng = random.Random(20261013)
    samples = []
    for _ in range(5000):
        draw = rng.choices(totals, k=len(totals))
        samples.append(sum(v for v, _ in draw) / sum(n for _, n in draw))
    return {'asin_groups': len(groups), 'bootstrap_unit': 'target_asin',
            'bootstrap_samples': 5000, 'seed': 20261013,
            'strict_at_1_difference95': [quantile(samples, .025), quantile(samples, .975)]}


def annotate(root, bm25_run_dir=None):
    root = Path(root).resolve()
    summary = generate(root, bm25_run_dir)
    store = TaskStore(PROJECT_DIR / 'harness/data/fine_items_eval_train_all.json')
    load(root / 'dataset.json', store)
    ids = summary['config']['task_ids']
    roots = {m: Path(p) for m, p in summary['source_runs'].items()}
    records = {}
    for method, directory in roots.items():
        with (directory / method / 'records.jsonl').open() as handle:
            records[method] = {row['task_id']: row for row in map(json.loads, handle)}
    annotations, categories = [], defaultdict(lambda: Counter())
    labels = {m: Counter() for m in roots}
    models = {m: Counter() for m in roots}
    transport = {m: Counter() for m in roots}
    returned_usage = {m: Counter() for m in roots}
    for index, task_id in enumerate(ids, 1):
        facts = store.get_private_facts(task_id)
        item = {'index': index, 'task_id': task_id, 'query': store.get_public_task(task_id).query,
                'category': facts.target_category, 'target_asin': facts.target_asin,
                'required_options': facts.required_options,
                'unresolved_gold_options': facts.unresolved_option_values,
                'annotation_source': 'deterministic_offline_evidence_v1', 'methods': {}}
        categories[facts.target_category]['tasks'] += 1
        for method in roots:
            row = records[method][task_id]
            annotated = annotate_method(row, roots[method], facts.target_asin)
            item['methods'][method] = annotated
            labels[method][annotated['label']] += 1
            models[method].update(annotated['transport']['response_models'])
            transport[method].update(annotated['transport']['event_counts'])
            returned_usage[method].update(annotated['transport']['returned_usage'])
            categories[facts.target_category][method + '_strict_at_1'] += row['strict_at_1']
            categories[facts.target_category][method + '_recall'] += bool(row.get('retrieval_hit'))
        b, h = (item['methods'][m]['strict_at_1'] for m in ['bm25', 'harness'])
        item['paired_label'] = 'both_success' if b and h else 'bm25_only' if b else 'harness_only' if h else 'both_fail'
        annotations.append(item)
    output = root / 'analysis'
    with (output / 'annotations.jsonl').open('w', encoding='utf-8') as handle:
        for item in annotations:
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    diagnostics = {'annotation_source': 'deterministic_offline_evidence_v1',
                   'task_count': len(ids), 'labels': labels, 'response_models': models,
                   'transport_events': transport, 'returned_usage': returned_usage,
                   'categories': categories, 'grouped_comparison': grouped_comparison(annotations)}
    first_recalled = sum(bool(a['methods']['harness']['searches'] and
                         a['methods']['harness']['searches'][0]['target_rank_in_page']) for a in annotations)
    total_recalled = sum(bool(a['methods']['harness']['retrieval_hit']) for a in annotations)
    diagnostics['harness_retrieval_stages'] = {
        'first_search_full_page_recalled': first_recalled,
        'later_search_additional_recalled': total_recalled - first_recalled,
        'all_searches_recalled': total_recalled,
    }
    write_json(output / 'annotation_summary.json', diagnostics)
    lines = ['## 逐题标注与证据', '',
             '标注由本地规则读取真实搜索、详情查看、提交和 Evaluator 结果生成，没有额外调用模型。'
             '它表示可观察的失败阶段，不等同于人工审核的因果结论；未对未标为 Gold 的商品判定实际可接受性。', '',
             '每题在 `annotations.jsonl` 中保留原需求、Gold 商品/规格、两种方法的标签、'
             'Harness 每次查询与目标排名、提交结果及绝对审计路径。Gold 仅用于离线报告。', '',
             '| 标注 | BM25 | Harness |', '|---|---:|---:|']
    for label in sorted(set(labels['bm25']) | set(labels['harness']), key=lambda x: -labels['harness'][x]):
        lines.append(f"| {LABELS.get(label, label)} | {labels['bm25'][label]} | {labels['harness'][label]} |")
    lines += ['', '### 代表案例（按源序号选取，不挑选主观好坏）', '']
    seen = Counter()
    for item in annotations:
        h = item['methods']['harness']; label = h['label']
        if label == 'strict_at_1_success' or seen[label] >= 2:
            continue
        seen[label] += 1
        ranks = ', '.join(str(s['target_rank_in_page'] or '未命中') for s in h['searches']) or '无成功搜索'
        evidence = Path(h['artifact_dir']) / 'evaluation.json'
        if not evidence.exists():
            evidence = Path(h['artifact_dir']) / 'error.json'
        if not evidence.exists():
            evidence = Path(h['artifact_dir']) / 'events.jsonl'
        lines += [f"- **第 {item['index']} 题 · {h['label_zh']}**：{item['query']}",
                  f"  Gold `{item['target_asin']}`；每次搜索目标页内排名：{ranks}；"
                  f"BM25：{item['methods']['bm25']['label_zh']}。"
                  f"[Harness 评分或错误记录]({evidence})"]
    lines += ['', '### 模型及接口核对', '',
              f"配置请求模型为 `{summary['config']['model']}`。下表记录接口原始响应的模型名称与次数，不能把别名当作独立的版本保证。", '',
              '| 方法 | 返回模型 | 响应次数 |', '|---|---|---:|']
    for method in models:
        for model, count in models[method].items():
            lines.append(f'| {method} | {model} | {count} |')
    lines += ['', f"接口事件计数：`{json.dumps(transport, ensure_ascii=False)}`。错误事件含已恢复的重试，不能直接当作失败任务数。", '',
              '### 按品类核对', '', '| 品类 | 题数 | BM25 Strict@1 | Harness Strict@1 |', '|---|---:|---:|---:|']
    for category, counts in sorted(categories.items(), key=lambda x: -x[1]['tasks']):
        n = counts['tasks']
        lines.append(f"| {category} | {n} | {counts['bm25_strict_at_1'] / n:.1%} | {counts['harness_strict_at_1'] / n:.1%} |")
    lines += ['', '同一商品可能对应多道需求，任务间不完全独立；当前任务级区间可能偏窄。'
              '后续留出集应按商品 ASIN 分组划分，并按品类分层；仅按题号 501–1000 划分仍需检查商品泄漏。', '']
    grouped = diagnostics['grouped_comparison']
    interval = grouped['strict_at_1_difference95']
    lines += [f"本轮 {len(ids)} 题对应 {grouped['asin_groups']} 个 Gold 商品。按商品分组重采样 5,000 次，"
              f"Strict@1 差值的 95% 区间为 [{interval[0]*100:+.2f}, {interval[1]*100:+.2f}] 个百分点。"
              '这仍不消除源顺序选样的品类偏差。', '',
              '## 基于本轮标注的优化顺序', '',
              f"Harness 首次搜索整页召回 {first_recalled}/{len(ids)}，后续搜索新增召回 {total_recalled-first_recalled} 题。"
              '首次整页与 Top-5 指标分开，避免把扩大返回数量误认为多轮推理的收益。', '']
    lh = labels['harness']
    priorities = [
        ('检索', lh['target_not_retrieved'], '保留原始需求查询作为召回通道，与 Agent 改写查询合并去重；离线比较 Top-5/10/20，随后在相同选择预算下测 Strict@1。'),
        ('详情查看', lh['retrieved_but_not_inspected'], '改进候选排序与详情查看分配，记录目标已召回到被查看的转化率；不能依赖 Gold 决定线上查看对象。'),
        ('选择与排序', lh['retrieved_but_wrong_selection'] + lh['correct_only_at_later_rank'], '比较候选约束证据，明确首位决策；分别记录“已看目标→选对”与 Strict-any→Strict@1 的差距。'),
        ('规格', lh['target_selected_wrong_options'], '提交前使用商品真实可选值做合法性校验，并对每条用户规格建立证据；区分合法但不匹配与非法选项。'),
        ('协议', lh['runtime:multiple_tool_calls'], '对多个工具调用做有界修正；独立只读调用才考虑受控批量执行，比较恢复提交数与额外 Token。'),
        ('上下文', lh['runtime:context_budget_exhausted'], '减少重复商品详情和历史上下文，保留候选证据摘要；用固定 Token 上限验证准确率与终止率。'),
    ]
    for rank, (name, count, action) in enumerate(sorted(priorities, key=lambda x: -x[1]), 1):
        lines.append(f'{rank}. **{name}：{count} 题对应失败。** {action}')
    bstats, hstats = (summary['methods'][m] for m in ['bm25', 'harness'])
    btoken, htoken = bstats['total_tokens']['mean_complete'], hstats['total_tokens']['mean_complete']
    if btoken and htoken:
        lines += ['', f"Harness 平均记录 Token 为 BM25 的 {htoken/btoken:.1f} 倍。"
                  '资源优化应与准确率一起评估：先去除重复上下文和无收益轮次，再尝试证据充分时提前提交。'
                  'Token 比例不是费用比例，缓存和输入输出单价尚未换算。']
    lines += ['', '以上数量用于确定诊断顺序，不代表修复后必然新增的成功题数。'
              'Test1 用于开发后，最终效果需在按商品隔离的 Test2 上验证；另做同 Thinking、同搜索与 Token 预算消融。', '']
    base = (output / 'report.md').read_text()
    (output / 'annotated_report.md').write_text(base + '\n' + '\n'.join(lines), encoding='utf-8')
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--bm25-run-dir', type=Path)
    args = parser.parse_args()
    result = annotate(args.run_dir, args.bm25_run_dir)
    print(json.dumps({'task_count': result['task_count'], 'labels': result['labels']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
