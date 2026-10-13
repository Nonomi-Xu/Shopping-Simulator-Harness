"""Offline experiment/report demo with synthetic products and scripted providers."""
import argparse
import json
from pathlib import Path
from src.harness.paths import result_path

from examples.long_horizon_demo import DemoProvider
from src.data.search_bm25 import build_index, sha256_file
from src.harness.agent import RuntimeConfig
from src.harness.branch.episodes.parser import RuleGoalParser
from src.harness.branch.episodes.preprocess import build_episode
from src.harness.branch.experiments.models import ExperimentConfig, Variant
from src.harness.branch.experiments.report import build_report
from src.harness.branch.experiments.runner import Case, load_completed, run_experiment
from src.harness.tasks import TaskStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=result_path, required=True)
    args = parser.parse_args(argv)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    data = root / "private-data"
    data.mkdir(mode=0o700)
    products = [dict(asin=asin, title="耳机", category="耳机", shop_name="演示商店",
        pricing=[price], attribute=["耳机"], customization_options={},
        instructions=[dict(asin=asin, instruction=f"我想买耳机，预算{budget}元以内",
                           instruction_options=[], attributes=[])])
        for asin, price, budget in [("earphone-a", 80, 100), ("earphone-b", 90, 150), ("earphone-c", 200, 300)]]
    products_path, index_path = data / "products.json", data / "index.sqlite3"
    products_path.write_text(json.dumps(products, ensure_ascii=False), encoding="utf-8")
    build_index(products, index_path, product_data_sha256=sha256_file(products_path))
    tasks, parser = TaskStore(products_path), RuleGoalParser()
    cases = [Case(task.task_id, build_episode(task, parser.parse(task.query), tasks.product_data_sha256, parser.manifest))
             for task in tasks.iter_public_tasks("all")]
    config = ExperimentConfig(mode="episode", reference="current", repeat_seeds=[11, 22, 33],
        bootstrap_samples=500, variants=[
            Variant(name="current", model=DemoProvider.model),
            Variant(name="short-context", model=DemoProvider.model,
                    runtime=RuntimeConfig(recent_turns=2, memory_candidate_limit=5)),
        ])
    report = run_experiment(config, cases, products_path=products_path, index_path=index_path,
                           output_dir=root / "experiment", provider_factory=lambda variant, seed: DemoProvider())
    saved_config, records = load_completed(root / "experiment")
    assert build_report(records, saved_config) == report
    print(root / "experiment/private/report.md")
    print("Synthetic scripted run; validates engineering and statistics, not real model quality or token cost.")


if __name__ == "__main__":
    main()
