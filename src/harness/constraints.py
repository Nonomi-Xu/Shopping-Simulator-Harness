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


def validate_predicate_quote(predicate: Predicate, quote: str) -> None:
    """Check literal grounding; semantic extraction remains independently auditable."""
    from ..data.NFKC import normalize_query
    if isinstance(predicate.value, str):
        if normalize_query(predicate.value) not in normalize_query(quote):
            raise ValueError("text predicate is not grounded in its quote")
        return
    numbers = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", quote)]
    possible = set(numbers)
    if predicate.unit == "g" and re.search(r"kg|千克|公斤", quote, re.I):
        possible.update(n * 1000 for n in numbers)
    if not any(abs(n-predicate.value) < 1e-8 for n in possible):
        raise ValueError("numeric predicate is not grounded in its quote")


def infer_literal_predicate(quote: str) -> Predicate | None:
    """Compile explicit public hard bounds when an actor omits its typed predicate.

    Does not classify hardness or read private goals. Approximate and unsupported
    semantic requirements return None rather than silently creating hard bounds.
    """
    if re.search(r"左右|大概|大约|约\d|差不多", quote): return None
    ceiling = re.search(r"(\d+(?:\.\d+)?)\s*元(?:以内|以下|之内)", quote)
    if ceiling:
        return Predicate(field="price", op="le", value=float(ceiling[1]), unit="CNY")
    ceiling = re.search(r"(?:不超过|最多|至多|上限(?:为|是)?)\s*(\d+(?:\.\d+)?)\s*元", quote)
    if ceiling:
        return Predicate(field="price", op="le", value=float(ceiling[1]), unit="CNY")
    ram = re.search(r"(?:至少|不低于)\s*(\d+(?:\.\d+)?)\s*(?:GB|G)\s*内存", quote, re.I)
    if ram:
        return Predicate(field="attribute", key="ram", op="ge", value=float(ram[1]), unit="GB")
    return None
