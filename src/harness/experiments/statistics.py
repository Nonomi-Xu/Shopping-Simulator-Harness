"""Task-cluster bootstrap and paired sign randomization using only stdlib.

Repeats of one task are averaged before resampling. Missing observations remain
missing; comparisons use identical task/repeat keys for each metric.
"""
from collections import defaultdict
import math
import random
from statistics import fmean


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def estimate(values, *, samples, confidence, seed):
    if not values:
        return {"mean": None, "ci": None, "task_count": 0}
    point = fmean(values)
    if len(values) < 2:
        return {"mean": point, "ci": None, "task_count": len(values)}
    rng = random.Random(seed)
    draws = [fmean(rng.choices(values, k=len(values))) for _ in range(samples)]
    tail = (1 - confidence) / 2
    return {"mean": point, "ci": [quantile(draws, tail), quantile(draws, 1-tail)],
            "task_count": len(values)}


def randomization_p(deltas, *, samples, seed):
    """Two-sided task-level sign-flip test under exchangeable paired labels."""
    if len(deltas) < 2:
        return None
    target = abs(sum(deltas))
    if len(deltas) <= 12:
        totals = [abs(sum(d if mask & (1 << i) else -d for i, d in enumerate(deltas)))
                  for mask in range(2 ** len(deltas))]
        return sum(t >= target - 1e-12 for t in totals) / len(totals)
    rng = random.Random(seed)
    extreme = sum(abs(sum(d * rng.choice((-1, 1)) for d in deltas)) >= target - 1e-12
                  for _ in range(samples))
    return (extreme + 1) / (samples + 1)


def holm_adjust(comparisons):
    tests = [metric for comparison in comparisons.values() for metric in comparison.values()
             if metric.get("p_value") is not None]
    previous = 0
    for rank, metric in enumerate(sorted(tests, key=lambda m: m["p_value"])):
        previous = max(previous, min(1, (len(tests) - rank) * metric["p_value"]))
        metric["p_value_holm"] = previous


def summarize_metric(rows, metric, **settings):
    clusters = defaultdict(list)
    values = []
    for row in rows:
        value = row["metrics"].get(metric)
        if value is not None:
            clusters[row["task_id"]].append(value)
            values.append(value)
    result = estimate([fmean(v) for v in clusters.values()], **settings)
    result.update(observed_runs=len(values), total_runs=len(rows),
                  coverage=len(values)/len(rows) if rows else None,
                  p50=quantile(values, .5) if values else None,
                  p95=quantile(values, .95) if values else None,
                  mean_within_task_stddev=fmean([
                      math.sqrt(sum((x-fmean(v))**2 for x in v)/(len(v)-1))
                      for v in clusters.values() if len(v) > 1
                  ]) if any(len(v) > 1 for v in clusters.values()) else None)
    return result


def paired_metric(reference, candidate, metric, **settings):
    def keyed(rows):
        return {(r["task_id"], r["repeat_seed"]): r["metrics"].get(metric) for r in rows}
    left, right = keyed(reference), keyed(candidate)
    if left.keys() != right.keys():
        raise ValueError("paired comparison requires identical task/repeat keys")
    clusters = defaultdict(list)
    pairs = 0
    for key in sorted(left):
        if left[key] is not None and right[key] is not None:
            clusters[key[0]].append(right[key] - left[key])
            pairs += 1
    deltas = [fmean(values) for values in clusters.values()]
    result = estimate(deltas, **settings)
    result.update(paired_runs=pairs, total_pairs=len(left),
                  coverage=pairs/len(left) if left else None,
                  positive_task_deltas=sum(d > 0 for d in deltas), ties=sum(d == 0 for d in deltas),
                  negative_task_deltas=sum(d < 0 for d in deltas),
                  p_value=randomization_p(deltas, samples=settings["samples"], seed=settings["seed"]),
                  p_value_holm=None)
    return result
