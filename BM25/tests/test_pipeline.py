from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from bm25_agent.catalog import BM25Catalog, FIELDS, search_tokens
from bm25_agent.pipeline import BM25AgentPipeline, evaluation_summary, retrieval_summary
from bm25_agent.tasks import TaskStore
from bm25_agent.types import AgentDecision, SelectorResult


def _product(index: int) -> dict:
    asin = f"A{index:03d}"
    target = index == 1
    title = "白色 静音 冰箱 节能 家用" if target else f"白色 冰箱 普通 候选 {index}"
    query = "寻找白色静音节能家用冰箱" if target else f"寻找普通候选商品 {index}"
    color = "白色" if target else "灰色"
    return {
        "asin": asin,
        "tag": "eval",
        "title": title,
        "shop_name": "测试店",
        "category": "家电›冰箱",
        "pricing": [700 + index],
        "attribute": ["静音", "节能"] if target else ["普通"],
        "customization_options": {
            "颜色": [{"value": color, "price": 700 + index, "is_available": True}]
        },
        "instructions": [
            {
                "asin": asin,
                "instruction": query,
                "instruction_options": [color],
                "attributes": [],
            }
        ],
    }


def _field_text(product: dict, field: str) -> str:
    values = {
        "title": product["title"],
        "brand": product["shop_name"],
        "category": product["category"],
        "model": product.get("_model", ""),
        "attributes": " ".join(product["attribute"]),
        "options": "",
        "bullets": "",
    }
    if field == "options":
        parts = []
        for axis, options in product["customization_options"].items():
            parts.append(axis)
            parts.extend(str(option["value"]) for option in options)
        return " ".join(parts)
    return values[field]


def _build_index(path: Path, products: list[dict]) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE VIRTUAL TABLE products USING fts5("
        "asin UNINDEXED, title, brand, category, model, attributes, options, bullets,"
        " tokenize='unicode61 remove_diacritics 2')"
    )
    connection.execute(
        "CREATE TABLE product_catalog ("
        "asin TEXT PRIMARY KEY, title TEXT NOT NULL, brand TEXT, category TEXT NOT NULL,"
        "price_min REAL, price_max REAL, pricing TEXT NOT NULL, description TEXT NOT NULL,"
        "attributes TEXT NOT NULL, customization_options TEXT NOT NULL,"
        "variant_combinations TEXT NOT NULL, images TEXT NOT NULL)"
    )
    for product in products:
        tokenized = [
            " ".join(search_tokens(_field_text(product, field))) for field in FIELDS
        ]
        connection.execute(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (product["asin"], *tokenized),
        )
        connection.execute(
            "INSERT INTO product_catalog VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                product["asin"],
                product["title"],
                product["shop_name"],
                product["category"],
                min(product["pricing"]),
                max(product["pricing"]),
                json.dumps(product["pricing"]),
                "测试商品描述",
                json.dumps(product["attribute"], ensure_ascii=False),
                json.dumps(product["customization_options"], ensure_ascii=False),
                "[]",
                json.dumps(["private-looking-image-url"]),
            ),
        )
    manifest = {
        "search_version": "shopsimulator-multifield-bm25-v3",
        "field_weights": {
            "title": 3.0,
            "brand": 2.0,
            "category": 2.0,
            "model": 2.5,
            "attributes": 1.5,
            "options": 1.2,
            "bullets": 0.8,
        },
    }
    connection.execute("CREATE TABLE manifest(payload TEXT NOT NULL)")
    connection.execute("INSERT INTO manifest VALUES (?)", (json.dumps(manifest),))
    connection.commit()
    connection.close()


class FixedSelector:
    def __init__(self, decision: AgentDecision) -> None:
        self.decision = decision
        self.seen_candidates = []

    def select(self, query, candidates):
        self.seen_candidates = list(candidates)
        return SelectorResult(
            decision=self.decision,
            provider="fake",
            model="fake-model",
        )


class PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        directory = Path(self.temporary.name)
        self.products = [_product(index) for index in range(1, 7)]
        self.products_path = directory / "products.json"
        self.products_path.write_text(
            json.dumps(self.products, ensure_ascii=False), encoding="utf-8"
        )
        self.index_path = directory / "products.sqlite3"
        _build_index(self.index_path, self.products)
        self.tasks = TaskStore(self.products_path)
        self.catalog = BM25Catalog(self.index_path)
        self.addCleanup(self.catalog.close)
        self.target_task_id = self.tasks.list_task_ids("all")[0]

    def test_cli_runs_full_test_subset_without_count(self) -> None:
        from contextlib import redirect_stdout
        import io
        from unittest.mock import patch
        from bm25_agent.cli import main

        class FirstCandidateSelector:
            def select(self, query, candidates):
                return SelectorResult(
                    decision=AgentDecision(
                        asin=candidates[0].asin, selected_options={}, reason="测试",
                    ),
                    provider="fake", model="fake-model",
                )

        selector = FirstCandidateSelector()
        output = io.StringIO()
        with patch("bm25_agent.cli.DeepSeekSelector", return_value=selector), redirect_stdout(output):
            code = main([
                "--products-path", str(self.products_path),
                "--index-path", str(self.index_path),
                "evaluate", "--split", "test",
                "--output-dir", str(self.products_path.parent / "outputs"),
            ])
        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        selection = payload["summary"]["dataset_selection"]
        self.assertEqual(selection["task_ids"], self.tasks.list_task_ids("test"))
        self.assertEqual(selection["selected_count"], 6)
        self.assertIsNone(selection["seed"])
        self.assertTrue(Path(payload["output_dir"]).name.startswith("test_"))
        self.assertEqual(len((Path(payload["output_dir"]) / "runs.jsonl").read_text().splitlines()), 6)

    def test_cli_defaults_to_all_and_optional_sampling(self) -> None:
        from bm25_agent.cli import build_parser, _selected_ids

        args = build_parser().parse_args(["recall"])
        self.assertEqual(_selected_ids(args, self.tasks), self.tasks.list_task_ids("all"))
        args = build_parser().parse_args(["recall", "--split", "test", "--count", "1", "--seed", "42"])
        self.assertEqual(_selected_ids(args, self.tasks), self.tasks.sample_task_ids("test", 1, 42))

    def test_pipeline_retrieves_only_top5_and_scores_agent_choice(self) -> None:
        selector = FixedSelector(
            AgentDecision(
                asin="A001",
                selected_options={"颜色": "白色"},
                reason="符合白色、静音和节能需求",
            )
        )
        result = BM25AgentPipeline(
            tasks=self.tasks,
            catalog=self.catalog,
            selector=selector,
        ).run(self.target_task_id)

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.candidates), 5)
        self.assertEqual(len(selector.seen_candidates), 5)
        self.assertTrue(result.evaluation.strict_success)
        self.assertEqual(result.evaluation.reward, 1.0)

    def test_outside_top5_selection_is_rejected(self) -> None:
        selector = FixedSelector(
            AgentDecision(asin="NOT-IN-TOP5", selected_options={}, reason="越界")
        )
        result = BM25AgentPipeline(
            tasks=self.tasks,
            catalog=self.catalog,
            selector=selector,
        ).run(self.target_task_id)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.evaluation.terminal_reason, "invalid_agent_decision")
        self.assertIn("asin_outside_top5", result.evaluation.failures)

    def test_candidate_payload_excludes_images_and_private_instructions(self) -> None:
        candidate = self.catalog.search_top5("白色静音冰箱")[0]
        payload = candidate.agent_payload()

        self.assertNotIn("images", payload)
        self.assertNotIn("instructions", payload)
        self.assertIn("customization_options", payload)
        self.assertIn("variant_combinations", payload)

    def test_bm25_weights_include_unindexed_asin_position(self) -> None:
        """Model 2.5 must beat category 2.0 for symmetric documents.

        With the old seven-argument call the weights shifted: category received
        2.5 and model received 1.5, so this assertion failed.
        """
        directory = Path(self.temporary.name)
        index_path = directory / "weight_alignment.sqlite3"
        category_match = _product(1)
        category_match.update(
            title="first product",
            category="weightprobe",
            _model="noise",
        )
        model_match = _product(2)
        model_match.update(
            title="second product",
            category="noise",
            _model="weightprobe",
        )
        _build_index(index_path, [category_match, model_match])

        with BM25Catalog(index_path) as catalog:
            candidates = catalog.search_top5("weightprobe")
            self.assertEqual(
                catalog.weights,
                (0.0, 3.0, 2.0, 2.0, 2.5, 1.5, 1.2, 0.8),
            )

        self.assertEqual([candidate.asin for candidate in candidates], ["A002", "A001"])

    def test_retrieval_and_agent_summaries_are_separate(self) -> None:
        recall = retrieval_summary(
            self.tasks, self.catalog, [self.target_task_id]
        )
        self.assertEqual(recall["top5_recall"], 1.0)

        selector = FixedSelector(
            AgentDecision(
                asin="A001",
                selected_options={"颜色": "白色"},
                reason="匹配",
            )
        )
        result = BM25AgentPipeline(
            tasks=self.tasks,
            catalog=self.catalog,
            selector=selector,
        ).run(self.target_task_id)
        summary = evaluation_summary([result])
        self.assertEqual(summary["strict_success_rate"], 1.0)
        self.assertEqual(summary["top5_recall"], 1.0)
        self.assertEqual(summary["top5_hit_count"], 1)
        self.assertEqual(summary["asin_selection_rate_given_top5"], 1.0)
        self.assertEqual(summary["strict_success_rate_given_top5"], 1.0)


if __name__ == "__main__":
    unittest.main()
