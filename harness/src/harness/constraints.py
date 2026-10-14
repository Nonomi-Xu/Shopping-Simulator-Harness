"""Typed public predicates shared by retrieval, preference tracking and metrics.

No evaluator annotations are consulted. Unknown facts never count as satisfied.
"""
from __future__ import annotations

from typing import Literal
import math
import re
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator


class Predicate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    field: Literal["category", "brand", "price", "attribute", "option"]
    key: str | None = Field(default=None, max_length=100)
    op: Literal["eq", "ne", "contains", "le", "ge"]
    value: str | float
    unit: Literal["CNY", "GB", "g", "W", "h", "cm"] | None = None

    @field_validator("value", mode="before")
    @classmethod
    def reject_boolean(cls, value):
        if isinstance(value, bool): raise ValueError("predicate value cannot be boolean")
        return value

    @model_validator(mode="after")
    def consistent(self):
        if self.field in {"option", "attribute"} and not self.key:
            raise ValueError("attribute and option predicates require a key")
        if self.op in {"le", "ge"} or self.field == "price":
            if isinstance(self.value, str) or not math.isfinite(self.value) or self.value < 0:
                raise ValueError("numeric predicates require finite nonnegative numbers")
        elif not isinstance(self.value, str) or not self.value.strip():
            raise ValueError("text predicates require nonblank strings")
        if self.field == "price" and (self.op not in {"le", "ge"} or self.unit != "CNY"):
            raise ValueError("price predicates require le/ge and explicit CNY")
        if self.field in {"category", "brand", "option"} and self.op not in {"eq", "ne", "contains"}:
            raise ValueError("text field does not support numeric comparison")
        if self.field == "attribute" and self.op in {"le", "ge"} and self.unit is None:
            raise ValueError("numeric attributes require a unit")
        return self


def predicate_key(predicate: Predicate) -> str:
    return predicate.model_dump_json()


#: Tolerance Environment v2 freezes for "左右"-style budgets. It is the encoding
#: the contract recommends so the agent's hard bound and the reward agree.
APPROXIMATE_TOLERANCE = 1.1
#: Largest multiplicative slack the runtime accepts in a model-declared
#: approximate ceiling. 1.5 keeps the bound recognisable as "around the amount".
APPROXIMATE_MAX_SLACK = 1.5
#: Absolute slack floor, in price units, for small amounts where catalog price
#: granularity means "around 10 yuan" can legitimately reach 20.
APPROXIMATE_ABSOLUTE_ALLOWANCE = 10.0
#: Fields whose value must appear literally in the quote. Category and attribute
#: values are the model's catalog mapping of a quoted requirement -- "小工具"
#: becomes 模具, "很快就能干" becomes 速干 -- so a literal substring is the wrong
#: test for them. Brand and option values stay strict: inventing a brand or an
#: option string is what actually breaks correctness.
GROUNDED_TEXT_FIELDS = frozenset({"brand", "option"})

_APPROXIMATE_AMOUNT = re.compile(
    r"左右|大概|大约|约|差不多|上下|附近|出头|来块"
    r"|(?<=[零一二两三四五六七八九十百千万\d])多"
    r"|(?<=[零一二两三四五六七八九十百千万\d])余"
)
_STRICT_CEILING = re.compile(
    r"以内|以下|之内|不超过|不高于|别超过|别超|最高|最多|至多|上限"
)
_PRICE_RANGE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?\s*(?:-|~|～|至|到|—)\s*"
    r"(\d+(?:\.\d+)?)\s*(万|千|[kK])?"
)
_OPEN_LOWER_BOUND = re.compile(r"[kK千]\s*\+|以上|起步|至少|不低于")
_CJK_DIGITS = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CJK_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}
_CJK_NUMERAL = r"[零一二两三四五六七八九十百千万]+"


def parse_cjk_number(text: str) -> float | None:
    """Parse a Chinese numeral such as 两千八百, 三十 or 一百八.

    A trailing bare digit after a unit takes the next lower unit, which is how
    these are spoken: 两千五 is 2500 and 一百八 is 180, not 2005 and 108.
    """
    total = section = number = 0.0
    last_unit: int | None = None
    seen = False
    for character in text:
        if character in _CJK_DIGITS:
            number = float(_CJK_DIGITS[character])
            seen = True
        elif character in _CJK_UNITS:
            unit = _CJK_UNITS[character]
            if unit == 10000:
                section = (section + number) * unit
                total += section
                section = 0.0
            else:
                section += (number or 1.0) * unit
            number = 0.0
            last_unit = unit
            seen = True
        else:
            return None
    if not seen:
        return None
    if number and last_unit and last_unit > 10 and text and text[-1] in _CJK_DIGITS:
        return total + section + number * (last_unit // 10)
    return total + section + number


def stated_amounts(quote: str) -> list[float]:
    """Return every amount a quote states, in price units.

    Handles Arabic digits with 万/千/k/百 scaling and Chinese numerals that sit
    next to a currency word or follow a budget keyword.
    """
    amounts: list[float] = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(万|千|[kK]|百)?", quote):
        value = float(match.group(1))
        unit = (match.group(2) or "").casefold()
        if unit == "万":
            value *= 10000
        elif unit in ("千", "k"):
            value *= 1000
        elif unit == "百":
            value *= 100
        amounts.append(value)
    for match in re.finditer(_CJK_NUMERAL, quote):
        value = parse_cjk_number(match.group(0))
        if value is None:
            continue
        tail = quote[match.end(): match.end() + 2]
        head = quote[: match.start()]
        if re.match(r"\s*(?:元|块|钱)", tail) or re.search(
            r"(?:预算|价格|售价|单价|价位)[^，。；]{0,4}$", head
        ):
            amounts.append(value)
    return amounts


def is_approximate_amount(quote: str) -> bool:
    return bool(_APPROXIMATE_AMOUNT.search(quote))


def _numeric_grounding_error(predicate: Predicate, quote: str, amounts: list[float]) -> str:
    """Explain exactly which encodings the runtime would accept."""
    if not amounts:
        return (
            f"numeric predicate is not grounded in its quote {quote!r}: the quote "
            "states no amount. Quote the sentence that gives the number, or drop "
            "the predicate and keep the requirement as a soft constraint."
        )
    base = max(amounts)
    if predicate.op == "le" and is_approximate_amount(quote):
        limit = max(base * APPROXIMATE_MAX_SLACK, base + APPROXIMATE_ABSOLUTE_ALLOWANCE)
        return (
            f"numeric predicate {predicate.value} is not grounded in its quote "
            f"{quote!r}: the quote says about {base:g}. Use {base:g} "
            f"(the stated amount), {base * APPROXIMATE_TOLERANCE:g} (Environment v2's "
            f"frozen tolerance), or any ceiling up to {limit:g}."
        )
    return (
        f"numeric predicate {predicate.value} is not grounded in its quote {quote!r}: "
        f"the quote states {base:g}. Use exactly {base:g}, or quote the sentence that "
        "gives the bound you intend."
    )


def validate_predicate_quote(predicate: Predicate, quote: str) -> None:
    """Check that a predicate is grounded in the quote it cites.

    Literal provenance is required for brand and option values, and for every
    numeric bound: the value must appear in the quote, or be a documented ceiling
    for an amount the quote states as approximate. Semantic extraction remains
    independently auditable.
    """
    from ..data.NFKC import normalize_query
    if isinstance(predicate.value, str):
        if predicate.field not in GROUNDED_TEXT_FIELDS:
            return
        if normalize_query(predicate.value) not in normalize_query(quote):
            raise ValueError(
                f"text predicate {predicate.value!r} is not grounded in its quote "
                f"{quote!r}: quote the words you are copying, or use a category or "
                "attribute constraint for a catalog term of your own choosing"
            )
        return
    numbers = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", quote)]
    possible = set(numbers)
    possible.update(stated_amounts(quote))
    if predicate.unit == "g" and re.search(r"kg|千克|公斤", quote, re.I):
        possible.update(n * 1000 for n in numbers)
    if any(abs(n - predicate.value) < 1e-8 for n in possible):
        return
    amounts = sorted(possible)
    if predicate.op == "le" and is_approximate_amount(quote) and amounts:
        base = max(amounts)
        limit = max(base * APPROXIMATE_MAX_SLACK, base + APPROXIMATE_ABSOLUTE_ALLOWANCE)
        if base <= predicate.value <= limit:
            return
    if predicate.op == "ge" and _OPEN_LOWER_BOUND.search(quote) and amounts:
        if predicate.value <= max(amounts):
            return
    raise ValueError(_numeric_grounding_error(predicate, quote, amounts))


def infer_literal_predicate(quote: str) -> Predicate | None:
    """Compile explicit public hard bounds when an actor omits its typed predicate.

    Does not classify hardness or read private goals. Approximate wording and
    unsupported semantic requirements still return None: the runtime does not
    invent a hard bound the user never gave, even though the validator accepts an
    explicitly declared approximate ceiling. See APPROXIMATE_TOLERANCE.
    """
    if is_approximate_amount(quote): return None
    amounts = stated_amounts(quote)
    if not amounts: return None
    if _PRICE_RANGE.search(quote) or _STRICT_CEILING.search(quote):
        # A range states an explicit upper bound too; Environment v2 takes the
        # high end, and Chinese numerals are amounts like any other.
        return Predicate(field="price", op="le", value=max(amounts), unit="CNY")
    ram = re.search(
        r"(?:至少|不低于)\s*(\d+(?:\.\d+)?)\s*(?:GB|G)\s*内存", quote, re.I
    )
    if ram:
        return Predicate(field="attribute", key="ram", op="ge", value=float(ram[1]), unit="GB")
    return None
