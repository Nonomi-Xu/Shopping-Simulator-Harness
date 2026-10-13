"""Contract tests for the read-only public product catalog."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from src.data.search_bm25 import (
    MultiFieldBM25Searcher,
    SearchIndexError,
    build_index,
)
from src.harness.catalog import (
    CatalogDataError,
    CatalogError,
    ProductCatalog,
    ProductNotFoundError,
)
from src.harness.schemas import ProductDetail, ProductSummary


PRODUCTS = [
    {
        "asin": "000000000001",
        "tag": "train",
        "title": "宿舍静音小冰箱",
        "shop_name": "小熊旗舰店",
        "category": "家电›冰箱",
        "pricing": [799, "829"],
        "full_description": "适合宿舍使用的节能静音小冰箱。",
        "attribute": ["节能", "静音"],
        "customization_options": {
            "颜色": [
                {
                    "value": "白色",
                    "price": 799,
                    "is_available": True,
                    "asin": "variant-white",
                },
                {
                    "value": "灰色",
                    "price": 829,
                    "is_available": False,
                    "asin": "variant-gray",
                },
            ]
        },
        "variant_combinations": [
            {
                "options": {"颜色": "白色"},
                "price": 799,
                "asin": "variant-white",
                "is_available": True,
            }
        ],
        "images": ["https://example.test/fridge.jpg"],
        "instructions": [
            {
                "instruction": "private gold instruction",
                "worker_id": "private-worker",
            }
        ],
        "user_persona": {"private": True},
        "reward_detail": {"gold": True},
    },
    {
        "asin": "000000000002",
        "tag": "eval",
        "title": "开放式蓝牙耳机",
        "shop_name": "声动旗舰店",
        "category": "数码›耳机",
        "pricing": [299],
        "attribute": ["低夹耳压力"],
        "instructions": [{"instruction": "private earphone task"}],
    },
    {
        "asin": "000000000003",
        "tag": "train",
        "title": "迷你单门冰箱",
        "shop_name": "海尔旗舰店",
        "category": "家电›冰箱",
        "pricing": [999],
        "attribute": ["宿舍", "小容量"],
        "instructions": [{"instruction": "private fridge task"}],
    },
]


class ProductCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.index_path = (
            Path(self._temporary_directory.name) / "products.sqlite3"
        )
        self.product_sha256 = "fixture-product-sha256"
        build_index(
            PRODUCTS,
            self.index_path,
            product_data_sha256=self.product_sha256,
        )

    def test_search_returns_summaries_in_bm25_rank_order(self) -> None:
        raw_searcher = MultiFieldBM25Searcher(self.index_path)
        self.addCleanup(raw_searcher.close)
        expected_hits = raw_searcher.search("宿舍 静音 冰箱", k=3)

        with self._catalog() as catalog:
            results = catalog.search("宿舍 静音 冰箱", limit=3)

        self.assertTrue(results)
        self.assertTrue(
            all(isinstance(result, ProductSummary) for result in results)
        )
        self.assertEqual(
            [result.asin for result in results],
            [hit.asin for hit in expected_hits],
        )
        self.assertEqual(
            [result.rank for result in results],
            [hit.rank for hit in expected_hits],
        )
        self.assertEqual(
            [result.bm25_score for result in results],
            [hit.score for hit in expected_hits],
        )

    def test_bm25_field_weights_account_for_unindexed_asin_column(self) -> None:
        products = [
            {
                "asin": "000000000002",
                "title": "needle",
                "shop_name": "other",
                "category": "test",
            },
            {
                "asin": "000000000001",
                "title": "other",
                "shop_name": "needle",
                "category": "test",
            },
        ]
        weights = {
            "title": 100.0,
            "brand": 1.0,
            "category": 1.0,
            "model": 1.0,
            "attributes": 1.0,
            "options": 1.0,
            "bullets": 1.0,
        }
        build_index(
            products,
            self.index_path,
            product_data_sha256=self.product_sha256,
            field_weights=weights,
        )

        searcher = MultiFieldBM25Searcher(self.index_path)
        self.addCleanup(searcher.close)
        hits = searcher.search("needle", k=2)

        self.assertEqual(
            [hit.asin for hit in hits],
            ["000000000002", "000000000001"],
        )
        self.assertGreater(hits[0].score, hits[1].score)

    def test_search_honors_limit_and_validates_arguments(self) -> None:
        with self._catalog() as catalog:
            self.assertEqual(len(catalog.search("冰箱", limit=1)), 1)
            self.assertEqual(catalog.search(""), [])

            with self.assertRaisesRegex(ValueError, "between 1 and 50"):
                catalog.search("冰箱", limit=0)
            with self.assertRaisesRegex(ValueError, "between 1 and 50"):
                catalog.search("冰箱", limit=51)
            with self.assertRaisesRegex(TypeError, "limit must be an integer"):
                catalog.search("冰箱", limit=True)
            with self.assertRaisesRegex(TypeError, "query must be a string"):
                catalog.search(123)  # type: ignore[arg-type]

    def test_get_product_decodes_complete_public_detail(self) -> None:
        with self._catalog() as catalog:
            detail = catalog.get_product("000000000001")

        self.assertIsInstance(detail, ProductDetail)
        self.assertEqual(detail.brand, "小熊旗舰店")
        self.assertEqual(detail.price_min, 799)
        self.assertEqual(detail.price_max, 829)
        self.assertEqual(detail.pricing, [799, 829])
        self.assertEqual(detail.attributes, ["节能", "静音"])
        self.assertEqual(
            detail.customization_options["颜色"][0].value,
            "白色",
        )
        self.assertEqual(
            detail.variant_combinations[0].options,
            {"颜色": "白色"},
        )
        self.assertEqual(
            detail.images,
            ["https://example.test/fridge.jpg"],
        )

    def test_product_detail_does_not_leak_private_task_fields(self) -> None:
        with self._catalog() as catalog:
            detail = catalog.get_product("000000000001")

        serialized = json.dumps(
            detail.model_dump(),
            ensure_ascii=False,
            sort_keys=True,
        )
        for private_value in (
            "instructions",
            "private gold instruction",
            "private-worker",
            "user_persona",
            "reward_detail",
        ):
            self.assertNotIn(private_value, serialized)

    def test_unknown_product_raises_explicit_error(self) -> None:
        with self._catalog() as catalog:
            with self.assertRaisesRegex(
                ProductNotFoundError,
                "product not found: missing",
            ):
                catalog.get_product("missing")

            with self.assertRaisesRegex(ValueError, "must not be empty"):
                catalog.get_product("  ")

    def test_contains_checks_the_public_catalog(self) -> None:
        with self._catalog() as catalog:
            self.assertTrue(catalog.contains("000000000001"))
            self.assertFalse(catalog.contains("missing"))
            self.assertFalse(catalog.contains(""))
            self.assertFalse(catalog.contains(None))

    def test_manifest_is_copied_and_product_hash_is_checked(self) -> None:
        with self._catalog() as catalog:
            manifest = catalog.manifest
            manifest["product_count"] = -1
            self.assertEqual(catalog.manifest["product_count"], len(PRODUCTS))

        with self.assertRaisesRegex(SearchIndexError, "SHA-256"):
            ProductCatalog(
                self.index_path,
                expected_product_sha256="wrong-sha256",
            )

    def test_incompatible_catalog_manifest_fails_closed(self) -> None:
        with sqlite3.connect(self.index_path) as connection:
            row = connection.execute(
                "SELECT payload FROM manifest"
            ).fetchone()
            manifest = json.loads(row[0])
            manifest["catalog_schema_version"] = 999
            connection.execute(
                "UPDATE manifest SET payload = ?",
                (json.dumps(manifest, sort_keys=True),),
            )

        with self.assertRaisesRegex(
            CatalogError,
            "catalog schema version mismatch",
        ):
            ProductCatalog(self.index_path)

    def test_missing_catalog_row_referenced_by_search_fails_closed(self) -> None:
        with sqlite3.connect(self.index_path) as connection:
            connection.execute(
                "DELETE FROM product_catalog WHERE asin = ?",
                ("000000000001",),
            )

        with self._catalog() as catalog:
            with self.assertRaisesRegex(
                CatalogDataError,
                "missing from product_catalog",
            ):
                catalog.search("宿舍 静音 小冰箱", limit=3)

    def test_corrupted_detail_json_fails_closed(self) -> None:
        with sqlite3.connect(self.index_path) as connection:
            connection.execute(
                "UPDATE product_catalog SET attributes = ? WHERE asin = ?",
                ("not-json", "000000000001"),
            )

        with self._catalog() as catalog:
            with self.assertRaisesRegex(
                CatalogDataError,
                "invalid public product detail",
            ):
                catalog.get_product("000000000001")

    def test_context_manager_closes_catalog(self) -> None:
        catalog = self._catalog()
        with catalog as active_catalog:
            self.assertIs(active_catalog, catalog)
            self.assertTrue(catalog.contains("000000000001"))

        catalog.close()
        with self.assertRaisesRegex(CatalogError, "catalog is closed"):
            catalog.search("冰箱")
        with self.assertRaisesRegex(CatalogError, "catalog is closed"):
            catalog.get_product("000000000001")
        with self.assertRaisesRegex(CatalogError, "catalog is closed"):
            _ = catalog.manifest

    def _catalog(self) -> ProductCatalog:
        return ProductCatalog(
            self.index_path,
            expected_product_sha256=self.product_sha256,
        )


if __name__ == "__main__":
    unittest.main()
