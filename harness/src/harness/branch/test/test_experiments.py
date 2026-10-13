"""Statistical invariants and real offline/episode matrix integration."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import patch

from examples.long_horizon_demo import DemoProvider
from src.harness.cli import main
from src.harness.branch.episodes.runner import EpisodeRunner
from src.harness.branch.experiments.metrics import project_metrics
from src.harness.branch.experiments.models import ExperimentConfig, Variant
from src.harness.branch.experiments.report import build_report
from src.harness.branch.experiments.runner import Case, load_completed, run_experiment
from src.harness.branch.experiments.statistics import estimate, holm_adjust, paired_metric, summarize_metric
from src.harness.providers import ProviderAPIError
from src.harness.tasks import TaskStore
from src.harness.test.test_agent_runner import ScriptedProvider
from src.harness.branch.test.test_episode_runner import EpisodeFixture


class StatisticsTest(unittest.TestCase):
    settings = dict(samples=100, confidence=.95, seed=42)

    def row(self, task, repeat, value):
        return {"task_id": task, "repeat_seed": repeat, "metrics": {"score": value}}

    def test_task_clusters_prevent_pseudoreplication(self):
        original = [self.row("a", 1, 0), self.row("b", 1, 1), self.row("c", 1, 1)]
        repeated = [self.row(row["task_id"], seed, row["metrics"]["score"])
                    for seed in range(20) for row in original]
        a = summarize_metric(original, "score", **self.settings)
        b = summarize_metric(repeated, "score", **self.settings)
        self.assertEqual(a["mean"], b["mean"])
        self.assertEqual(a["ci"], b["ci"])
        self.assertEqual(b["task_count"], 3)
        self.assertEqual(b["observed_runs"], 60)

    def test_pair_by_key_not_position_and_handle_missing(self):
        a = [self.row("a", 1, 0), self.row("b", 1, 1), self.row("b", 2, None)]
        b = [self.row("b", 2, 0), self.row("a", 1, 1), self.row("b", 1, 1)]
        result = paired_metric(a, b, "score", **self.settings)
        self.assertEqual(result["mean"], .5)
        self.assertEqual(result["paired_runs"], 2)
        self.assertAlmostEqual(result["coverage"], 2/3)
        self.assertEqual(result["p_value"], 1)
        with self.assertRaises(ValueError):
            paired_metric(a, b[:-1], "score", **self.settings)

    def test_null_single_task_and_holm(self):
        self.assertEqual(estimate([], **self.settings)["mean"], None)
        self.assertIsNone(estimate([1], **self.settings)["ci"])
        data = {"a": {"one": {"p_value": .01}, "two": {"p_value": .03}, "three": {"p_value": None}}}
        holm_adjust(data)
        self.assertEqual(data["a"]["one"]["p_value_holm"], .02)
        self.assertEqual(data["a"]["two"]["p_value_holm"], .03)


class ExperimentTest(EpisodeFixture):
    def config(self, mode="episode"):
        return ExperimentConfig(mode=mode, reference="current", repeat_seeds=[11, 22], bootstrap_samples=100,
            variants=[Variant(name="current", model="test"), Variant(name="other", model="test")])

    def test_episode_matrix_failure_denominator_artifacts_and_reanalysis(self):
        output = Path(self.tmp.name) / "experiment"
        calls = []

        def factory(variant, seed):
            calls.append((variant.name, seed))
            return DemoProvider() if variant.name == "current" else ScriptedProvider([
                ProviderAPIError("synthetic failure")])

        report = run_experiment(self.config(), [Case(self.episode.task_id, self.episode)],
            products_path=self.path, index_path=self.index, output_dir=output, provider_factory=factory)
        self.assertEqual(len(calls), 4)
        self.assertEqual(report["variants"]["current"]["metrics"]["strict_success"]["mean"], 1)
        self.assertEqual(report["variants"]["other"]["metrics"]["strict_success"]["mean"], 0)
        self.assertEqual(report["comparisons"]["other"]["strict_success"]["mean"], -1)
        self.assertIsNone(report["comparisons"]["other"]["strict_success"]["ci"])
        self.assertEqual(report["variants"]["other"]["metrics"]["total_tokens"]["coverage"], 0)
        self.assertEqual(report["variants"]["other"]["failure_categories"], {"provider_error": 2})
        config, records = load_completed(output)
        self.assertEqual(build_report(records, config), report)
        for row in records:
            audit = output / row["artifact_dir"]
            self.assertTrue((audit / "public/runtime/events.jsonl").exists())
            self.assertIsNone(json.loads((audit / "public/runtime/evaluation.json").read_text()))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["experiments", "report", "--run-dir", str(output),
                                   "--output-dir", str(output / "reanalysis")]), 0)
        self.assertEqual(json.loads((output / "reanalysis/summary.json").read_text()), report)
        with patch("src.harness.paths.OUTPUTS_DIR", Path(self.tmp.name) / "automatic"), \
             redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(main(["experiments", "report", "--run-dir", str(output)]), 0)
        generated = Path(json.loads(printed.getvalue())["output_dir"])
        self.assertEqual(generated.parent, (Path(self.tmp.name) / "automatic/reports").resolve())
        self.assertEqual(json.loads((generated / "summary.json").read_text()), report)
        with (output / "private/records.jsonl").open("a") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_completed(output)

    def test_offline_cli_matrix_and_no_overwrite(self):
        config = self.config(mode="offline")
        file = Path(self.tmp.name) / "config.json"
        file.write_text(config.model_dump_json())
        output = Path(self.tmp.name) / "offline"
        args = ["experiments", "run", "--config", str(file), "--products-path", str(self.path),
                "--index-path", str(self.index), "--task-id", self.episode.task_id,
                "--output-dir", str(output), "--no-env-file"]
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "synthetic-key"}), \
             patch("src.harness.branch.experiments.cli.DeepSeekProvider", side_effect=lambda **kw: DemoProvider()), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        manifest = json.loads((output / "private/manifest.json").read_text())
        self.assertEqual(len(manifest["schedule"]), 4)
        summary = json.loads((output / "private/summary.json").read_text())
        self.assertEqual(summary["variants"]["current"]["metrics"]["strict_success"]["mean"], 1)
        self.assertIsNone(summary["variants"]["current"]["metrics"]["constraint_satisfaction_rate"]["mean"])
        self.assertNotIn("synthetic-key", (output / "private/manifest.json").read_text())
        with self.assertRaisesRegex(ValueError, "already exists"):
            run_experiment(config, [Case(self.episode.task_id)], products_path=self.path,
                           index_path=self.index, output_dir=output, provider_factory=lambda *a: None)

    def test_unexpected_failure_has_null_cost_and_does_not_drop_task(self):
        tasks = TaskStore(self.path).list_task_ids("all")
        def broken(*args):
            raise RuntimeError("not a model result")
        report = run_experiment(self.config(mode="offline"), [Case(t) for t in tasks],
            products_path=self.path, index_path=self.index,
            output_dir=Path(self.tmp.name) / "errors", provider_factory=broken)
        current = report["variants"]["current"]
        self.assertEqual(current["runs"], 4)
        self.assertEqual(current["metrics"]["strict_success"]["mean"], 0)
        self.assertEqual(current["metrics"]["actions"]["coverage"], 0)
        self.assertEqual(current["failure_categories"], {"execution_error": 4})

    def test_reject_duplicate_cases_before_writing_and_explicit_seed(self):
        output = Path(self.tmp.name) / "invalid"
        with self.assertRaisesRegex(ValueError, "unique"):
            run_experiment(self.config(), [Case(self.episode.task_id, self.episode)] * 2,
                products_path=self.path, index_path=self.index, output_dir=output, provider_factory=lambda *a: None)
        self.assertFalse(output.exists())
        v = Variant(name="v", model="test", model_seed_parameter="seed")
        self.assertEqual(v.parameters(17)["seed"], 17)
        self.assertNotIn("seed", Variant(name="v", model="test").parameters(17))

    def test_partial_usage_is_unknown(self):
        result = EpisodeRunner(products_path=self.path, index_path=self.index, provider=DemoProvider()).run_episode(self.episode)
        run = result.run
        turns = [run.turns[0].model_copy(update={"usage": None}), *run.turns[1:]]
        partial = run.model_copy(update={"turns": turns})
        values = project_metrics(partial, result.evaluation, 1)
        self.assertIsNone(values["total_tokens"])
        self.assertEqual(values["strict_success"], 1)
