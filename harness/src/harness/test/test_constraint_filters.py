import json
import tempfile
import unittest
from pathlib import Path
from src.data.search_bm25 import build_index
from src.data.product_facts import matches_predicates, predicate_statuses, numeric_fact
from src.harness.constraints import Predicate
from src.harness.schemas import SearchFilters, SearchProductsAction
from src.harness.catalog import ProductCatalog


class ConstraintFiltersTest(unittest.TestCase):
    def test_sql_filters_before_topk_and_pagination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"i.sqlite3"
            products = [dict(asin=str(i), title="笔记本", category="笔记本", shop_name=brand,
                pricing=[price], attribute=[f"内存 {ram}GB"], customization_options={})
                for i, price, ram, brand in [(1, 6000, 32, "A"), (2, 3000, 8, "A"), (3, 4500, 16, "B"), (4, 4000, 32, "A")]]
            build_index(products, path, product_data_sha256="a"*64)
            predicates = [Predicate(field="price", op="le", value=5000, unit="CNY"),
                Predicate(field="attribute", key="ram", op="ge", value=16, unit="GB"),
                Predicate(field="brand", op="ne", value="B")]
            with ProductCatalog(path) as catalog:
                items, page = catalog.search_page(SearchProductsAction(query="笔记本", limit=1, filters=SearchFilters(predicates=predicates)))
                self.assertEqual([i.asin for i in items], ["4"])
                self.assertEqual(page.total_results, 1)
                self.assertEqual(catalog.eligible_count(predicates), 1)
                self.assertEqual([r.asin for r in catalog.search("笔记本")], ["1","2","3","4"])

    def test_variant_constraints_share_one_selection(self):
        product = dict(category="笔记本", customization_options={"内存": [
            dict(value="8GB", price=3000, is_available=True), dict(value="16GB", price=6000, is_available=True)]}, attributes=[])
        predicates = [Predicate(field="price", op="le", value=5000, unit="CNY"), Predicate(field="attribute", key="ram", op="ge", value=16, unit="GB")]
        self.assertFalse(matches_predicates(product, predicates))
        self.assertEqual(predicate_statuses(product, predicates, selected={"内存": "16GB"}), ["violated", "satisfied"])
        product["customization_options"]["内存"][1]["price"] = 4500
        self.assertTrue(matches_predicates(product, predicates))

    def test_unknown_and_units(self):
        self.assertEqual(numeric_fact("重量 1.5kg", "weight", "g"), 1500)
        self.assertIsNone(numeric_fact("硬盘 512GB", "ram", "GB"))
        self.assertIsNone(numeric_fact("最高内存 16GB", "ram", "GB"))
        self.assertFalse(matches_predicates({"attributes": ["不支持蓝牙"]}, [Predicate(field="attribute", key="function", op="contains", value="蓝牙")]))
        self.assertFalse(matches_predicates({}, [Predicate(field="price", op="le", value=5000, unit="CNY")]))
        self.assertFalse(matches_predicates({"customization_options": {"颜色": [], "尺寸": []}}, [Predicate(field="option", key="颜色", op="eq", value="黑色")]))
