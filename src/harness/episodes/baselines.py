"""Paired full-query and interactive runs over the same tasks and resource config."""
import json
from pathlib import Path
from .artifacts import write_episode_artifacts
from .metrics import summarize


def run_comparison(episodes, runner_factory, output_dir):
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    audit = root / "private"
    audit.mkdir(mode=0o700)
    compact = {"full_query": [], "long_horizon": []}
    pairs = []
    seen = set()
    for episode in episodes:
        if episode.task_id in seen:
            raise ValueError("comparison requires one frozen episode per task; select a cache version")
        seen.add(episode.task_id)
        pair = {"task_id": episode.task_id, "episode_id": episode.episode_id}
        signatures = []
        for baseline in compact:
            runner = runner_factory()
            from ..agent.skill import load_agent_skill
            from ...data.search_bm25 import sha256_file
            signatures.append((runner.provider.name, runner.provider.model, runner.provider_parameters,
                getattr(runner.provider, "default_parameters", {}), getattr(runner.provider, "base_url", None),
                runner.config, runner.runtime, load_agent_skill(runner.skill_dir).source_sha256,
                runner.tasks.product_data_sha256, sha256_file(runner.index_path)))
            if len(signatures) == 2 and signatures[0] != signatures[1]:
                raise ValueError("paired baselines require identical model and runtime configuration")
            result = runner.run_episode(episode, baseline=baseline)
            directory = write_episode_artifacts(result, root/episode.episode_id/baseline)
            pair[baseline] = {"metrics": result.metrics, "directory": str(directory)}
            # Retain metrics only, not thousands of full model contexts.
            from types import SimpleNamespace
            compact[baseline].append(SimpleNamespace(metrics=result.metrics))
        pairs.append(pair)
        with (audit/"pairs.jsonl").open("a") as handle:
            handle.write(json.dumps(pair, ensure_ascii=False)+"\n")
    if not pairs: raise ValueError("no prepared episodes selected")
    summaries = {baseline: summarize(results) for baseline, results in compact.items()}
    summary = {"comparison_version": "paired-episodes-v1", "paired_episode_count": len(pairs), "baselines": summaries,
        "note": "full_query has complete information; interaction need not improve its success rate"}
    (audit/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    return summary
