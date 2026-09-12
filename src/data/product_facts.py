"""Conservative facts from public product fields; shared SQL filter/metric semantics.

Numeric attributes require a nearby explicit label and unit. Unsupported semantics,
missing prices and unlisted multi-axis combinations remain unknown. No gold fields.
"""
from __future__ import annotations
import json
import math
import re
from functools import lru_cache
from .NFKC import normalize_query
from ..harness.constraints import Predicate

FACTS_VERSION = "public-predicates-v1"
NUMERIC_KEYS = {"ram": ("ram", "内存", "运行内存"), "weight": ("weight", "重量", "净重"),
                "power": ("power", "功率"), "battery": ("battery", "续航"),
                "length": ("length", "长度")}
UNIT_VALUES = {"gb": ("GB", 1), "g": ("g", 1), "kg": ("g", 1000), "克": ("g", 1),
               "千克": ("g", 1000), "公斤": ("g", 1000), "w": ("W", 1), "瓦": ("W", 1),
               "h": ("h", 1), "小时": ("h", 1), "cm": ("cm", 1), "厘米": ("cm", 1)}


def numeric_fact(text, key, unit, *, axis=None):
    labels = NUMERIC_KEYS.get(key, (key,))
    normalized = normalize_query(text)
    if any(word in normalized for word in ("最高", "可选", "至多", "至少", "约", "up to")):
        return None
    units = "|".join(sorted(UNIT_VALUES, key=len, reverse=True))
    pattern = rf"(\d+(?:\.\d+)?)\s*({units})(?![a-z])"
    values = []
    for match in re.finditer(pattern, normalized):
        nearby = normalized[max(0, match.start()-12):match.end()+12]
        if not any(normalize_query(label) in (normalize_query(axis or "") or nearby) for label in labels):
            continue
        base, factor = UNIT_VALUES[match[2]]
        # RAM option axes commonly abbreviate GB to G.
        if key == "ram" and match[2] == "g": base = "GB"
        if base == unit: values.append(float(match[1])*factor)
    return values[0] if values and len(set(values)) == 1 else None


def compare(actual, predicate):
    if actual is None: return "unknown"
    expected = predicate.value
    if isinstance(expected, str):
        actual, expected = normalize_query(actual), normalize_query(expected)
    op = predicate.op
    ok = {"eq": lambda: actual == expected, "ne": lambda: actual != expected,
          "contains": lambda: expected in actual, "le": lambda: actual <= expected,
          "ge": lambda: actual >= expected}[op]()
    return "satisfied" if ok else "violated"


def selections(product, selected=None):
    """Return verifiable available option/price scenarios; no Cartesian guessing."""
    options = product.get("customization_options", {})
    variants = product.get("variant_combinations", [])
    axes = set(options)
    for v in variants: axes.update(v["options"])
    if selected is not None and not axes <= selected.keys(): return []
    if selected is not None and any(not any(o["value"] == value and o.get("is_available", True)
        for o in options.get(axis, [])) for axis, value in selected.items()): return []
    if variants:
        return [(v["options"], v.get("price")) for v in variants if v.get("is_available", True)
                and (selected is None or all(v["options"].get(k) == value for k, value in selected.items()))]
    if len(options) == 1:
        axis = next(iter(options))
        return [({axis: o["value"]}, o.get("price")) for o in options[axis] if o.get("is_available", True)
                and (selected is None or selected.get(axis) == o["value"])]
    if axes: return []
    price = product.get("price_min")
    if price != product.get("price_max"): price = None
    return [({}, price)]


def status_for(product, predicate, selection, price):
    field = predicate.field
    if field in {"category", "brand"}:
        return compare(product.get(field) or None, predicate)
    if field == "price":
        if price is not None and (not math.isfinite(price) or price < 0): price = None
        return compare(price, predicate)
    if field == "option":
        return compare(selection.get(predicate.key), predicate)
    if predicate.op in {"le", "ge"}:
        axes = [axis for axis in product.get("customization_options", {})
                if any(label in normalize_query(axis) for label in NUMERIC_KEYS.get(predicate.key, (predicate.key,))) ]
        if axes:
            values = [numeric_fact(selection.get(axis, ""), predicate.key, predicate.unit, axis=axis) for axis in axes]
        else:
            # The description can describe several variants. Only labeled attributes
            # and the title are eligible for conservative numeric extraction.
            values = [numeric_fact(str(value), predicate.key, predicate.unit)
                      for value in [*product.get("attributes", []), product.get("title", "")]]
        known = [v for v in values if v is not None]
        actual = known[0] if known and len(set(known)) == 1 else None
        return compare(actual, predicate)
    attrs = [normalize_query(a) for a in product.get("attributes", [])]
    value = normalize_query(predicate.value)
    # Absence in marketing text cannot prove a feature is absent or excluded.
    negative = any(re.search(r"(?:不支持|不|无|非|without\s+|not\s+|no\s+)" + re.escape(value), a) for a in attrs)
    if negative:
        return "satisfied" if predicate.op == "ne" else "violated"
    present = any(a == value if predicate.op == "eq" else value in a for a in attrs)
    if not present: return "unknown"
    return "violated" if predicate.op == "ne" else "satisfied"


def predicate_statuses(product, predicates, *, selected=None):
    scenarios = selections(product, selected)
    if not scenarios:
        # Product-wide facts can still be checked when variant compatibility is unknown.
        return [status_for(product, p, {}, None) if p.field in {"brand", "category", "attribute"} else "unknown" for p in predicates]
    outcomes = [[status_for(product, p, options, price) for p in predicates] for options, price in scenarios]
    # One common scenario must satisfy all predicates; do not splice across variants.
    for row in outcomes:
        if all(s == "satisfied" for s in row): return row
    return min(outcomes, key=lambda row: (row.count("violated"), row.count("unknown")))


def matches_predicates(product, predicates):
    if not predicates: return True
    if len(product.get("customization_options", {})) > 1 and not product.get("variant_combinations"):
        if any(p.field in {"option", "price"} for p in predicates): return False
    return all(s == "satisfied" for s in predicate_statuses(product, predicates))


@lru_cache(maxsize=256)
def parse_predicates(payload):
    return [Predicate.model_validate(p) for p in json.loads(payload)]


def sqlite_predicates(title, brand, category, low, high, attributes, options, variants, payload):
    product = dict(title=title, brand=brand, category=category, price_min=low, price_max=high,
                   attributes=json.loads(attributes), customization_options=json.loads(options),
                   variant_combinations=json.loads(variants))
    return int(matches_predicates(product, parse_predicates(payload)))


SQL_PREDICATES = "public_predicates(c.title,c.brand,c.category,c.price_min,c.price_max,c.attributes,c.customization_options,c.variant_combinations,?) = 1"
