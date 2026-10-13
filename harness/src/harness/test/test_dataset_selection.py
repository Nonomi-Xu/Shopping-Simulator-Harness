import json
from pathlib import Path
import tempfile
import unittest

from src.harness.tasks import TaskStore

class DatasetSelectionTest(unittest.TestCase):
    def test_all_and_fixed_test_subset_ignore_source_tags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "products.json"
            products = []
            for index in range(505):
                asin = f"A{index:03d}"
                product = {
                    "asin": asin,
                    "instructions": [{"asin": asin, "instruction": f"query {index}",
                                      "instruction_options": [], "attributes": []}],
                }
                if index % 3:
                    product["tag"] = "train" if index % 2 else "eval"
                products.append(product)
            path.write_text(json.dumps(products), encoding="utf-8")
            store = TaskStore(path)
            all_ids = store.list_task_ids("all")
            test_ids = store.list_task_ids("test")
            self.assertEqual(len(all_ids), 505)
            self.assertEqual(len(set(all_ids)), 505)
            self.assertEqual(len(test_ids), 500)
            self.assertTrue(set(test_ids) < set(all_ids))
            self.assertEqual(test_ids, TaskStore(path).list_task_ids("test"))
            self.assertEqual(test_ids, all_ids[:500])
            self.assertEqual(store.list_task_ids("Original"), all_ids)
            self.assertEqual(store.list_task_ids("Test1"), test_ids)
            for product in products:
                product["tag"] = "unused-source-label"
            path.write_text(json.dumps(products), encoding="utf-8")
            reloaded = TaskStore(path)
            self.assertEqual(all_ids, reloaded.list_task_ids("all"))
            self.assertEqual(test_ids, reloaded.list_task_ids("test"))
            for legacy in ("train", "eval"):
                with self.assertRaises(ValueError):
                    store.list_task_ids(legacy)
            with self.assertRaises(ValueError):
                store.sample_task_ids("test", 501, seed=42)
            path.write_text(json.dumps(products[:2]), encoding="utf-8")
            small = TaskStore(path)
            self.assertEqual(small.list_task_ids("test"), small.list_task_ids("all"))
