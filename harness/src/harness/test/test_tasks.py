"""Tests for deterministic task option normalization and resolution."""

import unittest

from src.harness.tasks import TaskStore, _normalize_option_key


class TaskOptionNormalizationTest(unittest.TestCase):
    def test_shared_nfkc_rules_normalize_width_and_whitespace(self) -> None:
        self.assertEqual(
            _normalize_option_key("deep BBM\u00a0 1公斤"),
            _normalize_option_key("DEEP BBM  1公斤"),
        )
        self.assertEqual(
            _normalize_option_key("１Ｗ　３０００Ｋ 暖白光"),
            _normalize_option_key("1W 3000K 暖白光"),
        )

    def test_normalized_symbol_match_returns_catalog_value(self) -> None:
        product = {
            "customization_options": {
                "颜色分类": [
                    {"value": "加速器➕线➕控制器一套"},
                ],
            },
        }
        lookup = TaskStore._build_option_lookup(product)

        resolved, unresolved = TaskStore._resolve_options(
            ["加速器?线?控制器一套"],
            lookup,
        )

        self.assertEqual(
            resolved,
            {"颜色分类": "加速器➕线➕控制器一套"},
        )
        self.assertEqual(unresolved, [])

    def test_exact_match_wins_over_normalized_collision(self) -> None:
        product = {
            "customization_options": {
                "颜色分类": [
                    {"value": "#110 黄调一白"},
                    {"value": "110 黄调一白"},
                ],
            },
        }
        lookup = TaskStore._build_option_lookup(product)

        resolved, unresolved = TaskStore._resolve_options(
            ["#110 黄调一白"],
            lookup,
        )

        self.assertEqual(resolved, {"颜色分类": "#110 黄调一白"})
        self.assertEqual(unresolved, [])

    def test_cross_axis_ambiguity_remains_unresolved(self) -> None:
        product = {
            "customization_options": {
                "颜色分类": [{"value": "标准款"}],
                "套餐类型": [{"value": "标准款"}],
            },
        }
        lookup = TaskStore._build_option_lookup(product)

        resolved, unresolved = TaskStore._resolve_options(
            ["标准款"],
            lookup,
        )

        self.assertEqual(resolved, {})
        self.assertEqual(unresolved, ["标准款"])


if __name__ == "__main__":
    unittest.main()
