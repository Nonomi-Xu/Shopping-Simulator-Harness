"""Offline paired report from completed, hash-verified benchmark journals."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics

from ..agent.audit import write_json
from ..agent.events import digest
from .datasets import sha256_file


def quantile(values, p):
    if not values: return None
    values = sorted(values)
    point = (len(values)-1)*p
    lower = int(point)
    return values[lower] + (values[min(lower+1, len(values)-1)]-values[lower])*(point-lower)


def wilson(hits, n):
    if not n: return None
    z = 1.959963984540054
    p, z2 = hits/n, z*z
    mid = (p + z2/(2*n))/(1+z2/n)
    half = z*math.sqrt(p*(1-p)/n + z2/(4*n*n))/(1+z2/n)
    return [mid-half, mid+half]


def rate(rows, field):
    known = [r[field] for r in rows if r.get(field) is not None]
    hits = sum(bool(x) for x in known)
    return {'hits': hits, 'total': len(rows), 'rate': hits/len(rows),
            'known': len(known), 'coverage': len(known)/len(rows),
            'wilson95': wilson(hits, len(rows))}


def failure_bucket(row):
    if row['strict_at_1']: return 'strict_at_1_success'
    if row.get('status') in ('failed', 'terminated', 'error'):
        return 'runtime:' + str(row['terminal_reason'])
    if not row.get('reward_valid'): return 'unverifiable_or_no_evaluation'
    if not row.get('retrieval_hit'): return 'target_not_retrieved'
    if not row.get('inspected_hit'): return 'retrieved_but_not_inspected'
    if row.get('asin_at_1'): return 'target_selected_wrong_options'
    if row.get('strict_any'): return 'correct_only_at_later_rank'
    return 'retrieved_but_wrong_selection'


def summarize(rows):
    out = {field: rate(rows, field) for field in [
        'strict_at_1','asin_at_1','strict_any','retrieval_hit','first_search_top5_hit','inspected_hit','reward_valid']}
    out['task_count'] = len(rows)
    out['status_counts'] = dict(Counter(r['status'] for r in rows))
    out['terminal_reasons'] = dict(Counter(r['terminal_reason'] for r in rows))
    out['failure_buckets'] = dict(Counter(failure_bucket(r) for r in rows))
    out['recommendation_counts'] = dict(Counter(str(r.get('recommendation_count')) for r in rows))
    out['reward_all_mean'] = sum(r.get('reward') or 0 for r in rows)/len(rows)
    valid = [r['reward'] for r in rows if r.get('reward_valid') and r.get('reward') is not None]
    out['reward_valid_mean'] = statistics.mean(valid) if valid else None
    for field in ['duration_seconds','model_turns','actions','searches','environment_calls']:
        values = [r[field] for r in rows if r.get(field) is not None]
        out[field] = {'mean': statistics.mean(values) if values else None,
                      'p50': quantile(values,.5), 'p95': quantile(values,.95), 'coverage': len(values)/len(rows)}
    usage = [r['usage'] for r in rows if r.get('usage_complete') and r.get('usage')]
    out['token_coverage'] = len(usage)/len(rows)
    for key in ['input_tokens','output_tokens','total_tokens']:
        values = [u[key] for u in usage]
        out[key] = {'sum_complete': sum(values), 'mean_complete': statistics.mean(values) if values else None,
                    'p50': quantile(values,.5), 'p95': quantile(values,.95)}
    recalled = [r for r in rows if r.get('retrieval_hit')]
    out['strict_given_retrieved'] = sum(r['strict_at_1'] for r in recalled)/len(recalled) if recalled else None
    return out


def paired_statistics(bm25, harness):
    differences = [int(h['strict_at_1'])-int(b['strict_at_1']) for b,h in zip(bm25,harness)]
    n=len(differences)
    wins=sum(x==1 for x in differences); losses=sum(x==-1 for x in differences)
    rng=random.Random(20261013)
    means=[sum(rng.choices(differences,k=n))/n for _ in range(5000)]
    discordant=wins+losses
    pvalue = min(1.,2*sum(math.comb(discordant,k) for k in range(min(wins,losses)+1))/(2**discordant)) if discordant else 1.
    return {'strict_at_1_difference':sum(differences)/n, 'bootstrap95':[quantile(means,.025),quantile(means,.975)],
        'bootstrap_samples':5000,'bootstrap_seed':20261013,'mcnemar_exact_p':pvalue,
        'both_success':sum(b['strict_at_1'] and h['strict_at_1'] for b,h in zip(bm25,harness)),
        'harness_only':wins,'bm25_only':losses,
        'both_fail':sum(not b['strict_at_1'] and not h['strict_at_1'] for b,h in zip(bm25,harness)),
        'oracle_union_rate':sum(b['strict_at_1'] or h['strict_at_1'] for b,h in zip(bm25,harness))/n,
        'harness_rescues_bm25_miss':sum(not b.get('retrieval_hit') and h['strict_at_1'] for b,h in zip(bm25,harness))}


def generate(root, bm25_run_dir=None):
    root=Path(root).resolve()
    config=json.loads((root/'config.json').read_text())
    dataset=json.loads((root/'dataset.json').read_text())
    if digest(dataset)!=config['dataset_sha256']:
        raise ValueError('dataset hash mismatch')
    expected_ids = dataset['task_ids'][:config['smoke_limit']] if config['smoke_limit'] else dataset['task_ids']
    if config.get('subset'):
        expected_ids = dataset['task_ids']
        if config['task_ids'] != [i for i in expected_ids if i in set(config['task_ids'])]:
            raise ValueError('subset task order is not the frozen dataset order')
    if config['task_ids'] != expected_ids or config['source_sha256'] != dataset['source_sha256']:
        raise ValueError('configuration is not the frozen selection')
    methods={}
    source_roots={"bm25": Path(bm25_run_dir).resolve() if bm25_run_dir else root, "harness": root}
    source_configs={}
    for method in ['bm25','harness']:
        source_root=source_roots[method]
        source_config=json.loads((source_root/'config.json').read_text())
        source_configs[method]=source_config
        # Infrastructure transport version may change; task/data/model settings may not.
        common_keys=['dataset_sha256','source_sha256','index_sha256','task_ids','smoke_limit','model',
                     'base_url','temperature','bm25_thinking']
        if any(source_config.get(k)!=config.get(k) for k in common_keys):
            raise ValueError('baseline and rerun differ in frozen tasks/data/model settings')
        directory=source_root/method
        completed=json.loads((directory/'completed.json').read_text())
        if completed['records_sha256']!=sha256_file(directory/'records.jsonl') or completed['config_sha256']!=digest(source_config):
            raise ValueError('incomplete or modified journal')
        with (directory/'records.jsonl').open() as f: rows=[json.loads(line) for line in f]
        if (len(rows)!=len(config['task_ids']) or len({r['task_id'] for r in rows})!=len(rows) or
            any(r['config_sha256']!=digest(source_config) or r['method']!=method for r in rows) or
            {r['task_id'] for r in rows}!=set(config['task_ids'])):
            raise ValueError('methods do not cover the same frozen tasks')
        mapping={r['task_id']:r for r in rows}
        methods[method]=[mapping[i] for i in config['task_ids']]
        for row in methods[method]:
            row['artifact_dir']=str(source_root/row['artifact_dir'])
            # Earlier model turns can have usage even when the failing final
            # response was not parsed; do not call that task's usage complete.
            if row['terminal_reason'] in ('provider_error', 'runtime_error'):
                row['usage_complete'] = False
    b,h=methods['bm25'],methods['harness']
    paired=paired_statistics(b,h)
    provider_errors={}
    for method, rows in methods.items():
        errors=Counter()
        for row in rows:
            if row['terminal_reason'] not in ('provider_error', 'runtime_error'):
                continue
            result_path=Path(row['artifact_dir'])/'result.json'
            message=row.get('error_message') or (json.loads(result_path.read_text()).get('error_message','') if result_path.exists() else '')
            category=('connection_incomplete_read' if row.get('error_type') == 'IncompleteRead' or 'IncompleteRead(' in message else
                      'insufficient_balance_402' if 'HTTP 402' in message else
                      'redirect_307' if 'HTTP 307' in message else
                      'invalid_tool_json' if 'invalid JSON arguments' in message else
                      'other_provider_error' if row['terminal_reason']=='provider_error' else 'other_runtime_error')
            errors[category]+=1
        provider_errors[method]=dict(errors)
    infrastructure_categories = {'insufficient_balance_402','redirect_307',
                                  'connection_incomplete_read','other_provider_error','other_runtime_error'}
    infrastructure_counts = {m: sum(count for k, count in errors.items() if k in infrastructure_categories)
                             for m, errors in provider_errors.items()}
    infrastructure_failure = any(infrastructure_counts.values())
    health={'status':'infrastructure_compromised' if infrastructure_failure else 'completed',
            'provider_errors':provider_errors, 'valid_for_algorithm_claims':not infrastructure_failure,
            'note':'Only supports this Test1 system comparison; thinking, recommendation and search budgets differ.'}
    health['infrastructure_task_counts'] = infrastructure_counts
    health['strict_at_1_sensitivity'] = {
        m: {'observed':sum(r['strict_at_1'] for r in rows)/len(rows),
            'if_all_infrastructure_failures_succeeded':
                (sum(r['strict_at_1'] for r in rows)+infrastructure_counts[m])/len(rows)}
        for m, rows in methods.items()}
    paired['valid_for_algorithm_claims']=not infrastructure_failure
    summary={'dataset':dataset['name'],'task_count':len(b),'smoke_limit':config['smoke_limit'],
        'generated_utc':datetime.now(timezone.utc).isoformat(),'config':config,
        'methods':{k:summarize(v) for k,v in methods.items()},'paired':paired,
        'source_runs':{k:str(v) for k,v in source_roots.items()},'execution_health':health}
    report_dir=root/'analysis';report_dir.mkdir(exist_ok=True)
    write_json(report_dir/'summary.json',summary)
    write_json(report_dir/'execution_health.json',health)
    write_json(report_dir/'comparison_sources.json',{'runs':summary['source_runs'],
        'config_sha256':{k:digest(v) for k,v in source_configs.items()},
        'records_sha256':{k:sha256_file(source_roots[k]/k/'records.jsonl') for k in methods}})
    fields=['index','task_id','method','strict_at_1','asin_at_1','strict_any','retrieval_hit','inspected_hit',
            'terminal_reason','reward_valid','reward','duration_seconds','model_turns','searches','actions','usage_complete','artifact_dir']
    with (report_dir/'tasks.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for method in methods: writer.writerows(methods[method])
    failures=[{'task_id':r['task_id'],'method':m,'bucket':failure_bucket(r),'artifact_dir':r['artifact_dir']}
              for m,rows in methods.items() for r in rows if not r['strict_at_1']]
    write_json(report_dir/'failures.json',failures)
    sb,sh=summary['methods']['bm25'],summary['methods']['harness']
    def pct(v):return f'{v*100:.2f}%'
    def fmt(v):return 'NA' if v is None else f'{v:,.2f}'
    def cell(s,key):
        m=s[key];return f"{m['hits']}/{m['total']} ({pct(m['rate'])})"
    subset_line = (f"本批次是冻结的显式子集 `{config['subset']['name']}`（SHA-256 `{config['subset']['sha256']}`），"
                   "由失败类型与对照题固定选出，不能当作 Test1 总体成绩。") if config.get('subset') else ''
    lines=[f"# {dataset['name']}：BM25 与 Harness 新评测报告" + ("（仅预检，不是500题结果）" if config["smoke_limit"] else "") + (f"（冻结子集：{config['subset']['name']}，不是500题结果）" if config.get('subset') else ''),'',
        f"本报告对照 {len(b)} 个共同任务 × 2 种方法。Original 共 {dataset['original_count']} 题；Test1 为源顺序展开后的第 1–500 题。"] + ([subset_line] if subset_line else []) + [
        ('本报告使用全量重跑的 Harness 500 题，与已完成的同一 Test1 BM25 500 题配对；BM25 本次没有重复调用。' if bm25_run_dir else '本报告使用该批次的逐题结果。'), '',
        f"- Harness 批次：`{source_roots['harness']}`。",
        f"- BM25 批次：`{source_roots['bm25']}`。",
        '上一轮因余额不足/重定向导致的 Harness 失败结果未混入本次重跑统计。' if bm25_run_dir else '', '',
        '## 评测条件与边界','',
        f"- 数据源 SHA-256：`{dataset['source_sha256']}`。",
        f"- 模型：`{config['model']}`；temperature=0；BM25 Thinking=disabled，Harness Thinking={config['harness_thinking']}。",
        '- 保留现有算法：BM25 固定 Top-5、一次选择；Harness 多轮工具调用、最多 5 件推荐。',
        '- Strict@1 必须第一件商品 ASIN、规格均正确且评分有效；Strict-any 允许命中推荐列表的任意位置，单独报告。',
        '- BM25 召回是固定 Top-5；Harness 召回是实际多轮搜索结果的并集，两者检索预算不同，不能解释为同预算检索器比较。',
        '- 未提交、运行失败和不可评分任务保留在 500 题分母中；召回记录缺失另报覆盖率。',
        '- Token 仅统计 usage 完整的任务并报告覆盖率，不把未知消耗当作零费用。',
        '- Test1 为前 500 题，不是随机样本；区间描述此任务样本的不确定性，不消除顺序/品类偏差；每题仅运行一次，未评估随机重复稳定性。','',
        '## 新结果','', '| 指标 | BM25 | Harness |','|---|---:|---:|']
    for label,key in [('首件严格准确率','strict_at_1'),('首件 ASIN 命中率','asin_at_1'),('列表任意位置严格正确率','strict_any'),
        ('目标商品召回率（预算不同）','retrieval_hit'),('首次搜索 Top-5 召回率','first_search_top5_hit'),('目标详情已观察率','inspected_hit'),('有效评分覆盖率','reward_valid')]:
        lines.append(f'| {label} | {cell(sb,key)} | {cell(sh,key)} |')
    lines += [f"| 召回记录覆盖率 | {pct(sb['retrieval_hit']['coverage'])} | {pct(sh['retrieval_hit']['coverage'])} |",
        f"| 平均 Token（完整记录任务） | {fmt(sb['total_tokens']['mean_complete'])} | {fmt(sh['total_tokens']['mean_complete'])} |",
        f"| Token 完整记录覆盖率 | {pct(sb['token_coverage'])} | {pct(sh['token_coverage'])} |",
        f"| 平均单题耗时（秒） | {fmt(sb['duration_seconds']['mean'])} | {fmt(sh['duration_seconds']['mean'])} |",
        f"| P95 单题耗时（秒） | {fmt(sb['duration_seconds']['p95'])} | {fmt(sh['duration_seconds']['p95'])} |",
        f"| 平均模型轮次 | {fmt(sb['model_turns']['mean'])} | {fmt(sh['model_turns']['mean'])} |",
        f"| 平均搜索次数 | {fmt(sb['searches']['mean'])} | {fmt(sh['searches']['mean'])} |",'',
        '## 配对比较','',
        f"Harness−BM25 Strict@1 差值：{paired['strict_at_1_difference']*100:+.2f} 个百分点；任务级配对 bootstrap 95% 区间：[{paired['bootstrap95'][0]*100:+.2f}, {paired['bootstrap95'][1]*100:+.2f}] 个百分点。",
        f"McNemar 双侧精确检验 p={paired['mcnemar_exact_p']:.6g}，只针对预先指定的 Strict@1 主指标。",
        f"两者都成功 {paired['both_success']} 题；仅 Harness 成功 {paired['harness_only']} 题；仅 BM25 成功 {paired['bm25_only']} 题；两者都失败 {paired['both_fail']} 题。",
        f"成功集合并集为 {pct(paired['oracle_union_rate'])}，仅是事后选择上界，不是实际路由准确率。Harness 在 BM25 未召回的任务中严格成功 {paired['harness_rescues_bm25_miss']} 题。",'',
        '## 失败分解','', '| 方法 | 失败/成功类别 | 数量 |','|---|---|---:|']
    for m in ['bm25','harness']:
        for reason,count in sorted(summary['methods'][m]['failure_buckets'].items(),key=lambda kv:-kv[1]):
            lines.append(f'| {m} | {reason} | {count} |')
    lines += ['', '## 下一步实验建议','',
        '按本次失败分解的数量优先排期，保留本轮基线，逐项消融后再组合：',
        '1. 协议可靠性：若 multiple_tool_calls 占主要失败，先做有界协议修正或受控只读批量调用；不得把一组相互依赖的调用无条件执行。报告修复后提交率、Strict@1 和额外 Token。',
        '2. 检索缺口：对 target_not_retrieved 任务先离线扫描 Recall@5/10/20/50，再比较需求拆分、多查询去重与候选重排。Gold 只用于离线诊断。',
        '3. 选择与规格：对已召回但未选中、规格错误任务分别实验候选比较和合法规格约束；保留 ASIN@1 与 Strict@1 两种指标。',
        '4. 成本与停止：按实际 model_io/timeline 定位重复上下文与无收益动作，扫描搜索/动作/Token 预算；比较相同预算下的准确率及 P95 延迟。',
        '5. 自适应执行：以公开证据缺口决定 BM25 结果是否升级到 Harness，结合仅单方成功任务学习规则；路由阈值只能在开发数据上调优。',
        '6. 后续评测：Test1 从本轮起作为诊断开发集；另冻结不重叠 Test2（例如源序号 501–1000）作留出集，并增加品类分层随机留出集来检查顺序偏差。最终候选在独立集上重复 3 次并报告配对区间。',
        '7. 消融推理模式：本轮是现有系统对比，Thinking 配置不同；后续增加相同 Thinking/Token 预算的受控组，避免将模型推理量误认为框架收益。','',
        '## 文件索引','',
        '- `summary.json`：全部指标、配置、置信区间和配对统计。',
        f'- `tasks.csv`：{len(b)*2} 条方法×任务记录，便于筛选。',
        '- `failures.json`：失败类别与对应审计目录。',
        '- `../harness/tasks/<序号>/attempt_01/`：逐事件、HTTP 请求/响应、时间线、会话、计划、记忆和最终评分。',
        '- BM25 审计记录：见 comparison_sources.json 中的原批次目录；tasks.csv 使用绝对审计路径。','']
    if infrastructure_failure:
        lines[2:2]=['> **基础设施故障影响本轮结果。** 具体接口/传输/运行异常及数量见 execution_health.json；这些任务保留在分母并计为失败。全任务成功率反映系统执行结果，不应单凭配对 p 值声称算法优劣。','']
    lines += ['## 执行异常核对','', '| 方法 | 异常类型 | 题数 |','|---|---|---:|']
    for method, errors in provider_errors.items():
        for category, count in errors.items():
            lines.append(f'| {method} | {category} | {count} |')
    if not any(provider_errors.values()):
        lines += ['| 两种方法 | 无接口/运行异常 | 0 |']
    lines += ['', '模型输出非法工具 JSON 属于接口解析失败，与账号余额和网络连接问题分别统计。'
              '没有有效终局的任务可能仅有实时 events/model_io 和 error.json；召回未知会降低覆盖率。', '']
    if infrastructure_failure:
        lines += ['把全部基础设施失败假设为成功，仅用作极端敏感性上界，不能替换实测结果：', '']
        for method, bounds in health['strict_at_1_sensitivity'].items():
            lines.append(f"- {method}：实测 {pct(bounds['observed'])}；极端上界 {pct(bounds['if_all_infrastructure_failures_succeeded'])}。")
        lines += ['']
    curve_path=source_roots['bm25']/'analysis/retrieval_curve.json'
    if curve_path.exists():
        curve=json.loads(curve_path.read_text())
        lines += ['## BM25 离线召回曲线','', '| K | 召回数量 | 召回率 |','|---|---:|---:|']
        for k,item in curve['recall_at_k'].items():
            lines.append(f"| {k} | {item['hits']} | {pct(item['rate'])} |")
        lines += ['', '该曲线使用原始需求、同一 BM25 分词/权重/排序，不额外调用模型；扩大候选池的收益仍需在相同选择模型预算下验证。','']
    (report_dir/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--bm25-run-dir',type=Path,help='Reuse a completed BM25 batch with identical frozen tasks/data/model')
    args=parser.parse_args();summary=generate(args.run_dir,args.bm25_run_dir)
    print(json.dumps({'report':str(args.run_dir.resolve()/'analysis/report.md'),
                      'tasks':summary['task_count'],'paired':summary['paired']},ensure_ascii=False))


if __name__=='__main__':main()
