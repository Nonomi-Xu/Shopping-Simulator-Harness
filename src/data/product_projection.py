from __future__ import annotations

import math
from typing import Any

def _valid_price(value: object) -> float | None:
    """把单个值转换成合法的非负有限价格。"""
    if isinstance(value, bool):
        return None

    if not isinstance(value, (str, int, float)):
        return None

    try:
        price = float(value)
    except ValueError:
        return None

    if not math.isfinite(price) or price < 0:
        return None

    return price


def _numeric_prices(values: object) -> list[float]:
    """从列表中提取合法价格。"""
    if not isinstance(values, list):
        return []

    prices = []

    for value in values:
        price = _valid_price(value)
        if price is not None:
            prices.append(price)

    return prices


def _project_variant_combinations(
    value: object,
) -> list[dict]:
    if not isinstance(value, list):
        return []

    variants = []

    for raw_variant in value:
        if not isinstance(raw_variant, dict):
            continue

        raw_options = raw_variant.get("options")
        if not isinstance(raw_options, dict):
            continue

        options = {
            str(axis): str(option_value)
            for axis, option_value in raw_options.items()
            if axis is not None and option_value is not None
        }

        if not options:
            continue

        price = _valid_price(raw_variant.get("price"))
        variant_asin = raw_variant.get("asin")

        variants.append(
            {
                "options": options,
                "price": price,
                "asin": (
                    str(variant_asin)
                    if variant_asin is not None
                    else None
                ),
                "is_available": bool(
                    raw_variant.get("is_available", True)
                ),
            }
        )

    return variants


def project_public_product(product: dict[str, Any]) -> dict[str, Any]:
    """将原始商品转换为模型可见的严格白名单对象。"""
    asin = str(product.get("asin") or "").strip()
    if not asin:
        raise ValueError("商品缺少 asin")

    pricing = _numeric_prices(product.get("pricing"))
    summary_prices = list(pricing)

    raw_options = product.get("customization_options")
    if not isinstance(raw_options, dict):
        raw_options = {}

    customization_options = {}

    for raw_axis, raw_values in raw_options.items():
        if not isinstance(raw_values, list):
            continue

        axis = str(raw_axis).strip()
        if not axis:
            continue

        options = []

        for raw_option in raw_values:
            if not isinstance(raw_option, dict):
                continue

            value = str(raw_option.get("value") or "").strip()
            if not value:
                continue

            price = _valid_price(raw_option.get("price"))
            is_available = bool(
                raw_option.get("is_available", True)
            )

            if price is not None and is_available:
                summary_prices.append(price)

            variant_asin = raw_option.get("asin")

            options.append(
                {
                    "value": value,
                    "price": price,
                    "is_available": is_available,
                    "asin": (
                        str(variant_asin)
                        if variant_asin is not None
                        else None
                    ),
                }
            )

        if options:
            customization_options[axis] = options

    variant_combinations = _project_variant_combinations(
        product.get("variant_combinations")
    )

    for variant in variant_combinations:
        if (
            variant["is_available"]
            and variant["price"] is not None
        ):
            summary_prices.append(variant["price"])

    attributes = product.get("attribute")
    if not isinstance(attributes, list):
        attributes = []

    images = product.get("images")
    if not isinstance(images, list):
        images = []

    return {
        "asin": asin,
        "title": str(product.get("title") or ""),
        "brand": str(product.get("shop_name") or "") or None,
        "category": str(product.get("category") or ""),
        "price_min": min(summary_prices) if summary_prices else None,
        "price_max": max(summary_prices) if summary_prices else None,
        "pricing": pricing,
        "description": str(
            product.get("full_description")
            or product.get("sub_title")
            or ""
        ),
        "attributes": [
            str(value)
            for value in attributes
            if value is not None
        ],
        "customization_options": customization_options,
        "variant_combinations": variant_combinations,
        "images": [
            str(value)
            for value in images
            if value is not None
        ],
    }