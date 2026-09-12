"""Rebuildable aggregate reports from compact, private experiment records."""
from collections import Counter
import csv
import json
from pathlib import Path

from .metrics import METRICS
from .statistics import holm_adjust, paired_metric, summarize_metric


def build_report(records, config):
    groups = {v.name: [r for r in records if r["variant"] == v.name] for v in config.variants}
    keys = [(r["variant"], r["task_id"], r["repeat_seed"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate experiment records")
    if set(r["variant"] for r in records) != set(groups):
        raise ValueError("records must cover exactly the configured variants")
    settings = dict(samples=config.bootstrap_samples, confidence=config.confidence, seed=config.statistics_seed)
    variants = {}
    for name, rows in groups.items():
        variants[name] = {
            "runs": len(rows), "tasks": len({r["task_id"] for r in rows}),
            "terminal_reasons": dict(Counter(r["terminal_reason"] for r in rows)),
            "failure_categories": dict(Counter(r["failure_category"] for r in rows)),
            "metrics": {m: summarize_metric(rows, m, **settings) for m in METRICS},
        }
        for m in ("strict_success", "success_at_5"):
            by_task = {}
            for row in rows:
                by_task.setdefault(row["task_id"], []).append(row["metrics"][m])
            repeated = [v for v in by_task.values() if len(v) > 1]
            variants[name][m + "_unstable_task_rate"] = (
                sum(min(v) != max(v) for v in repeated)/len(repeated) if repeated else None)
    comparisons = {
        name: {m: paired_metric(groups[config.reference], rows, m, **settings) for m in METRICS}
        for name, rows in groups.items() if name != config.reference
    }
    holm_adjust(comparisons)
    # Keep differences in their natural units; sign indicates candidate - reference.
    return {
        "version": config.version, "reference": config.reference,
        "confidence": config.confidence, "metric_directions": METRICS,
        "statistics": {
            "unit": "task; repeated runs averaged within task",
            "interval": "percentile task-cluster bootstrap",
            "paired_difference": "candidate minus reference; same task and repeat seed",
            "test": "two-sided task-level paired sign randomization",
            "multiple_testing": "Holm across all available candidate/metric comparisons",
            "bootstrap_samples": config.bootstrap_samples, "seed": config.statistics_seed,
            "limitations": [
                "Intervals and tests require at least two tasks; small samples can yield degenerate intervals.",
                "Missing metrics use available pairs only; coverage must be considered when comparing.",
                "Sign randomization assumes exchangeable paired labels; p-values do not establish causality.",
                "Repeated runs do not increase the number of independent tasks.",
                "Token counts are not monetary cost; wall time includes provider retries and controller work.",
            ],
        },
        "variants": variants, "comparisons": comparisons,
    }


def markdown_report(report):
    def fmt(value):
        return "—" if value is None else f"{value:.4g}"
    lines = ["# Agent 配对评测", "", f"参考组：`{report['reference']}`。差值为候选组减参考组。",
             "", "每项均值先在任务内平均重复运行，再对任务取平均。缺失指标显示为 —。",
             "", "| 配置 | 任务 / 运行 | 严格成功率 | Success@5 | Token 均值 | P95 耗时（秒） |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, data in report["variants"].items():
        m = data["metrics"]
        lines.append(f"| {name} | {data['tasks']} / {data['runs']} | {fmt(m['strict_success']['mean'])} | "
                     f"{fmt(m['success_at_5']['mean'])} | {fmt(m['total_tokens']['mean'])} | {fmt(m['wall_seconds']['p95'])} |")
    lines += ["", "## 配对差值", "", f"{report['confidence']:.0%} 置信区间按任务重采样；Holm 校正覆盖全部候选组和指标。",
              "", "| 配置 | 指标 | 方向 | 差值 | 置信区间 | 有效配对 / 全部 | 校正 p 值 |",
              "|---|---|---|---:|---|---:|---:|"]
    for name, metrics in report["comparisons"].items():
        for metric, data in metrics.items():
            interval = data["ci"]
            ci = "—" if interval is None else f"[{fmt(interval[0])}, {fmt(interval[1])}]"
            lines.append(f"| {name} | {metric} | {METRICS[metric]} | {fmt(data['mean'])} | {ci} | "
                         f"{data['paired_runs']} / {data['total_pairs']} | {fmt(data['p_value_holm'])} |")
    lines += ["", "## 失败分类", ""]
    for name, data in report["variants"].items():
        lines.append(f"- {name}: `{json.dumps(data['failure_categories'], ensure_ascii=False)}`")
    lines += ["", "详细均值区间、指标覆盖率、P50/P95 和重复运行波动见 summary.json。",
              "当前实验不改变原有评分标准。冻结任务的重复运行种子不代表服务端模型支持确定性采样。",
              "少量任务的统计区间可能退化，不能据此宣称模型普遍更优。", ""]
    return "\n".join(lines)


def write_report(report, records, directory):
    path = Path(directory)
    # Caller creates a fresh directory; never overwrite previous report files.
    with (path / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (path / "report.md").open("x", encoding="utf-8") as handle:
        handle.write(markdown_report(report))
    with (path / "metrics.csv").open("x", encoding="utf-8", newline="") as handle:
        columns = ["variant", "task_id", "repeat_seed", "terminal_reason", "failure_category", *METRICS]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({**{k: row[k] for k in columns if k not in METRICS}, **row["metrics"]})
