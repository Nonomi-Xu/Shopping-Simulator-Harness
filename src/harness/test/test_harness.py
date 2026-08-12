"""Integration tests for the unified multi-session Harness manager."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.data.search_bm25 import SearchIndexError, build_index, sha256_file
from src.harness.harness import (
    ActiveSessionReleaseError,
    DuplicateSessionError,
    HarnessClosedError,
    SessionNotFoundError,
    ShoppingHarness,
)


TARGET_ASIN = "000000000001"
ALTERNATIVE_ASIN = "000000000002"

PRODUCTS = [
    {
        "asin": TARGET_ASIN,
        "tag": "eval",
        "title": "白色静音目标冰箱",
        "shop_name": "目标旗舰店",
        "category": "家电›冰箱",
        "pricing": [799],
        "attribute": ["静音", "节能"],
        "customization_options": {
            "颜色": [
                {
                    "value": "白色",
                    "price": 799,
                    "is_available": True,
                }
            ]
        },
        "instructions": [
            {
                "asin": TARGET_ASIN,
                "instruction": "寻找一款白色静音目标冰箱",
                "instruction_options": ["白色"],
                "attributes": ["静音"],
            }
        ],
    },
    {
        "asin": ALTERNATIVE_ASIN,
        "tag": "train",
        "title": "黑色普通冰箱",
        "shop_name": "普通旗舰店",
        "category": "家电›冰箱",
        "pricing": [699],
        "attribute": ["小容量"],
        "customization_options": {
            "颜色": [
                {
                    "value": "黑色",
                    "price": 699,
                    "is_available": True,
                }
            ]
        },
        "instructions": [
            {
                "asin": ALTERNATIVE_ASIN,
                "instruction": "寻找一款黑色普通冰箱",
                "instruction_options": ["黑色"],
                "attributes": ["小容量"],
            }
        ],
    },
]


class ShoppingHarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        directory = Path(self._temporary_directory.name)
        self.products_path = directory / "products.json"
        self.index_path = directory / "products.sqlite3"
        self.products_path.write_text(
            json.dumps(PRODUCTS, ensure_ascii=False),
            encoding="utf-8",
        )
        build_index(
            PRODUCTS,
            self.index_path,
            product_data_sha256=sha256_file(self.products_path),
        )
        self.harness = ShoppingHarness(
            products_path=self.products_path,
            index_path=self.index_path,
            default_max_steps=7,
        )
        self.addCleanup(self.harness.close)
        self.eval_task_id = self.harness.list_task_ids("eval")[0]

    def test_manifest_and_public_task_boundary_are_auditable(self) -> None:
        manifest = self.harness.manifest
        task = self.harness.get_public_task(self.eval_task_id)

        self.assertEqual(manifest["harness_version"], "shopping-harness-v1")
        self.assertEqual(manifest["default_max_steps"], 7)
        self.assertEqual(
            manifest["product_data_sha256"],
            manifest["catalog"]["product_data_sha256"],
        )
        self.assertEqual(task.query, "寻找一款白色静音目标冰箱")
        serialized = task.model_dump_json()
        self.assertNotIn(TARGET_ASIN, serialized)
        self.assertNotIn("source_row_number", serialized)

        manifest["catalog"]["field_weights"]["title"] = -1
        self.assertGreater(
            self.harness.manifest["catalog"]["field_weights"]["title"],
            0,
        )

    def test_task_sampling_remains_deterministic(self) -> None:
        first = self.harness.sample_task_ids("eval", 1, seed=42)
        second = self.harness.sample_task_ids("eval", 1, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(first, [self.eval_task_id])

    def test_create_sessions_assigns_unique_stable_ids(self) -> None:
        first = self.harness.create_session(self.eval_task_id)
        second = self.harness.create_session(self.eval_task_id, max_steps=3)

        self.assertEqual(first.state.session_id, "session_00000001")
        self.assertEqual(second.state.session_id, "session_00000002")
        self.assertEqual(first.state.max_steps, 7)
        self.assertEqual(second.state.max_steps, 3)
        self.assertEqual(len(self.harness.list_sessions(status="active")), 2)

    def test_explicit_duplicate_session_id_is_rejected(self) -> None:
        self.harness.create_session(
            self.eval_task_id,
            session_id="custom-session",
        )

        with self.assertRaises(DuplicateSessionError) as raised:
            self.harness.create_session(
                self.eval_task_id,
                session_id="custom-session",
            )

        self.assertEqual(raised.exception.code, "duplicate_session")

    def test_dispatch_runs_complete_session_without_exposing_components(self) -> None:
        started = self.harness.create_session(self.eval_task_id)
        session_id = started.state.session_id

        search = self.harness.dispatch(
            session_id,
            "search_products",
            {"query": "白色静音目标冰箱", "limit": 2},
        )
        self.assertEqual(search.status, "success")
        self.assertIn(
            TARGET_ASIN,
            [item["asin"] for item in search.observation],
        )

        detail = self.harness.dispatch(
            session_id,
            "get_product",
            {"asin": TARGET_ASIN},
        )
        self.assertEqual(detail.status, "success")

        submitted = self.harness.dispatch(
            session_id,
            "submit_recommendations",
            {
                "recommendations": [
                    {
                        "asin": TARGET_ASIN,
                        "selected_options": {"颜色": "白色"},
                        "reason": "满足静音和颜色需求",
                    }
                ]
            },
        )

        snapshot = self.harness.get_session(session_id)
        self.assertTrue(submitted.terminated)
        self.assertEqual(snapshot.state.status, "terminated")
        self.assertEqual(len(snapshot.history), 3)
        self.assertTrue(snapshot.evaluation.strict_success)
        self.assertTrue(self.harness.get_evaluation(session_id).strict_success)

    def test_session_snapshots_are_independent_copies(self) -> None:
        started = self.harness.create_session(self.eval_task_id)
        session_id = started.state.session_id
        snapshot = self.harness.get_session(session_id)

        snapshot.state.surfaced_asins.append("tampered")
        snapshot.history.append(None)  # type: ignore[arg-type]

        fresh = self.harness.get_session(session_id)
        self.assertEqual(fresh.state.surfaced_asins, [])
        self.assertEqual(fresh.history, [])

    def test_release_protects_active_session_and_returns_terminal_snapshot(self) -> None:
        started = self.harness.create_session(self.eval_task_id)
        session_id = started.state.session_id

        with self.assertRaises(ActiveSessionReleaseError) as raised:
            self.harness.release_session(session_id)
        self.assertEqual(raised.exception.code, "active_session_release")

        released = self.harness.release_session(session_id, force=True)
        self.assertEqual(released.state.status, "active")
        with self.assertRaises(SessionNotFoundError):
            self.harness.get_session(session_id)

    def test_terminated_session_can_be_released_without_force(self) -> None:
        started = self.harness.create_session(
            self.eval_task_id,
            max_steps=1,
        )
        session_id = started.state.session_id
        self.harness.dispatch(
            session_id,
            "search_products",
            {"query": "冰箱"},
        )

        released = self.harness.release_session(session_id)

        self.assertEqual(released.state.status, "terminated")
        self.assertEqual(released.state.terminal_reason, "max_steps")
        self.assertEqual(self.harness.list_sessions(), [])

    def test_unknown_session_uses_stable_error(self) -> None:
        with self.assertRaises(SessionNotFoundError) as raised:
            self.harness.dispatch("missing", "search_products", {})

        self.assertEqual(raised.exception.code, "session_not_found")

    def test_close_is_idempotent_and_blocks_runtime_operations(self) -> None:
        started = self.harness.create_session(self.eval_task_id)
        self.harness.close()
        self.harness.close()

        with self.assertRaises(HarnessClosedError) as raised:
            self.harness.create_session(self.eval_task_id)
        self.assertEqual(raised.exception.code, "harness_closed")

        with self.assertRaises(HarnessClosedError):
            self.harness.dispatch(
                started.state.session_id,
                "search_products",
                {"query": "冰箱"},
            )

        snapshot = self.harness.get_session(started.state.session_id)
        self.assertEqual(snapshot.state.status, "active")

    def test_product_hash_mismatch_fails_during_initialization(self) -> None:
        bad_index = Path(self._temporary_directory.name) / "bad.sqlite3"
        build_index(
            PRODUCTS,
            bad_index,
            product_data_sha256="wrong-hash",
        )

        with self.assertRaisesRegex(SearchIndexError, "SHA-256"):
            ShoppingHarness(
                products_path=self.products_path,
                index_path=bad_index,
            )


if __name__ == "__main__":
    unittest.main()
