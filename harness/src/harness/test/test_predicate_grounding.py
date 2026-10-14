"""Grounding rules for typed predicates.

Test1 recorded 190 constraints rejected for "not grounded in its quote" across
166 tasks. The dominant cause was a contract contradiction: the schema demanded a
literal numeric bound while `infer_literal_predicate` declared "左右" bounds
un-inferable, so a shopper saying "40元左右" left the agent with no legal
encoding. These tests pin the resolved contract against the exact quotes that
failed.
"""

from __future__ import annotations

import unittest

from src.harness.constraints import (
    APPROXIMATE_TOLERANCE,
    Predicate,
    infer_literal_predicate,
    is_approximate_amount,
    parse_cjk_number,
    stated_amounts,
    validate_predicate_quote,
)


def _price(op: str, value: float, quote: str) -> None:
    validate_predicate_quote(
        Predicate(field="price", op=op, value=value, unit="CNY"), quote
    )


class ChineseNumeralTests(unittest.TestCase):
    def test_colloquial_trailing_digit_takes_the_next_lower_unit(self) -> None:
        # 两千五 is 2500 and 一百八 is 180, not 2005 and 108.
        self.assertEqual(parse_cjk_number("两千五"), 2500.0)
        self.assertEqual(parse_cjk_number("一百八"), 180.0)
        self.assertEqual(parse_cjk_number("一万二"), 12000.0)

    def test_plain_numerals(self) -> None:
        self.assertEqual(parse_cjk_number("二十五"), 25.0)
        self.assertEqual(parse_cjk_number("十"), 10.0)
        self.assertEqual(parse_cjk_number("百"), 100.0)
        self.assertEqual(parse_cjk_number("两千八百"), 2800.0)
        self.assertEqual(parse_cjk_number("三十"), 30.0)
        self.assertIsNone(parse_cjk_number("几"))

    def test_stated_amounts_scale_units(self) -> None:
        self.assertIn(300.0, stated_amounts("预算在3百元以内"))
        self.assertIn(2000.0, stated_amounts("预算在两千元以内"))
        self.assertIn(11000.0, stated_amounts("价格在1万到1.1万之间的"))
        self.assertIn(40000.0, stated_amounts("预算4万"))

    def test_approximate_markers_do_not_fire_on_ordinary_words(self) -> None:
        self.assertFalse(is_approximate_amount("余额充足的话预算500元以内"))
        self.assertFalse(is_approximate_amount("多功能工具预算300元以内"))
        self.assertTrue(is_approximate_amount("预算三十多元"))
        self.assertTrue(is_approximate_amount("预算在两千八百左右"))


class PriceGroundingTests(unittest.TestCase):
    def test_stated_amount_is_always_accepted(self) -> None:
        _price("le", 40, "价格在40元左右")
        _price("le", 660, "整体价格不超过660元")
        _price("le", 2000, "预算在两千元以内")

    def test_frozen_tolerance_is_accepted(self) -> None:
        self.assertEqual(APPROXIMATE_TOLERANCE, 1.1)
        _price("le", 44, "价格在40元左右")

    def test_model_ceiling_within_the_documented_slack_is_accepted(self) -> None:
        # Every ratio Test1 actually produced: 1.33, 1.5, 1.2, 1.25.
        _price("le", 60, "价格在40元左右")
        _price("le", 40, "差不多30元左右")
        _price("le", 200, "价格在150元左右")
        _price("le", 30, "价格在20元左右")

    def test_small_amounts_get_an_absolute_allowance(self) -> None:
        # Catalog price granularity means "around 10 yuan" can legitimately reach 20.
        _price("le", 20, "价格在10元左右")
        _price("le", 10, "价格5块钱左右")
        _price("le", 15, "价格5块钱左右")

    def test_ceiling_beyond_the_slack_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "the quote says about 10"):
            _price("le", 21, "价格在10元左右")
        with self.assertRaisesRegex(ValueError, "the quote says about 40"):
            _price("le", 61, "价格在40元左右")

    def test_ceiling_below_a_strict_maximum_is_rejected(self) -> None:
        # A ceiling under the stated maximum would filter out products the
        # shopper accepts, so it is a real extraction error.
        with self.assertRaisesRegex(ValueError, "the quote states 660"):
            _price("le", 220, "整体价格不超过660元")

    def test_lower_bound_below_the_stated_amount_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "the quote states 1400"):
            _price("ge", 900, "售价1400元左右")

    def test_open_ended_lower_bound_is_accepted(self) -> None:
        # "4k+" is an open-ended lower bound, not a maximum.
        _price("ge", 4000, "预算4k+")

    def test_range_high_end_is_accepted(self) -> None:
        _price("le", 11000, "价格在1万到1.1万之间的")

    def test_chinese_numeral_bounds_are_accepted(self) -> None:
        _price("le", 2800, "预算在两千八百左右")
        _price("le", 1800, "预算不超过一千八")
        _price("le", 300, "预算在3百元以内")
        _price("le", 130, "大概一百元左右")

    def test_an_ungrounded_amount_is_still_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "states no amount"):
            _price("le", 9.99, "价格特别便宜，个位数价格的")
        with self.assertRaisesRegex(ValueError, "states no amount"):
            _price("le", 20, "十几块钱左右")

    def test_the_error_names_the_amounts_that_would_be_accepted(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _price("le", 61, "价格在40元左右")
        message = str(caught.exception)
        self.assertIn("40", message)
        self.assertIn("44", message)
        self.assertIn("60", message)

    def test_unit_conversion_regression(self) -> None:
        validate_predicate_quote(
            Predicate(field="attribute", key="weight", op="ge", value=1500, unit="g"),
            "重量 1.5kg",
        )


class TextGroundingTests(unittest.TestCase):
    def test_brand_values_must_be_literal(self) -> None:
        with self.assertRaisesRegex(ValueError, "is not grounded"):
            validate_predicate_quote(
                Predicate(field="brand", op="eq", value="耐克"),
                "鞋底带有胶钉的棒球鞋",
            )
        validate_predicate_quote(
            Predicate(field="brand", op="eq", value="Nike"), "要求是Nike的正品"
        )

    def test_option_values_must_be_literal(self) -> None:
        validate_predicate_quote(
            Predicate(field="option", key="颜色", op="eq", value="白色"), "需要白色"
        )
        with self.assertRaisesRegex(ValueError, "is not grounded"):
            validate_predicate_quote(
                Predicate(field="option", key="颜色", op="eq", value="白色"), "需要浅色系"
            )

    def test_category_and_attribute_values_may_be_catalog_mappings(self) -> None:
        # "小工具" -> 模具 and "很快就能干" -> 速干 are the translation the task
        # asks for; requiring a literal substring forbade it and cost 26 rejections.
        validate_predicate_quote(
            Predicate(field="category", op="contains", value="模具"),
            "手工制作中式点心的小工具",
        )
        validate_predicate_quote(
            Predicate(field="attribute", key="速干", op="contains", value="速干"),
            "很快就能干的衣裤",
        )

    def test_the_rejection_tells_the_model_what_to_do_instead(self) -> None:
        with self.assertRaises(ValueError) as caught:
            validate_predicate_quote(
                Predicate(field="brand", op="eq", value="耐克"), "棒球鞋"
            )
        self.assertIn("category or attribute", str(caught.exception))


class InferredBoundTests(unittest.TestCase):
    def test_strict_bounds_are_inferred_including_chinese_numerals(self) -> None:
        inferred = infer_literal_predicate("预算在两千元以内")
        self.assertEqual((inferred.op, inferred.value, inferred.unit), ("le", 2000.0, "CNY"))
        self.assertEqual(infer_literal_predicate("不超过80元").value, 80.0)
        self.assertEqual(infer_literal_predicate("价格在10-20之间").value, 20.0)
        ram = infer_literal_predicate("至少16GB内存")
        self.assertEqual((ram.field, ram.key, ram.op, ram.value), ("attribute", "ram", "ge", 16.0))

    def test_approximate_bounds_are_still_never_inferred(self) -> None:
        # The runtime must not invent a hard bound the shopper never gave, even
        # though it accepts one the model declares explicitly.
        self.assertIsNone(infer_literal_predicate("价格在150元左右"))
        self.assertIsNone(infer_literal_predicate("预算控制在两千五百元左右"))

    def test_open_ended_lower_bound_is_not_a_ceiling(self) -> None:
        self.assertIsNone(infer_literal_predicate("预算4k+"))


if __name__ == "__main__":
    unittest.main()
