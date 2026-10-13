"""Read-only access to the Harness-compatible SQLite FTS5 BM25 index."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3
from threading import RLock
import unicodedata
from typing import Any

from .types import Candidate


TOP_K = 5
SEARCH_VERSION = "shopsimulator-multifield-bm25-v3"
FIELDS = (
    "title",
    "brand",
    "category",
    "model",
    "attributes",
    "options",
    "bullets",
)
# FTS5 auxiliary-function weights are positional and include UNINDEXED
# columns. Keep an explicit ASIN slot so the seven manifest weights align
# with title through bullets instead of shifting one column to the left.
UNINDEXED_ASIN_WEIGHT = 0.0
_HAN_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_LATIN_OR_NUMBER = re.compile(r"[a-z0-9]+(?:[._+\-/][a-z0-9]+)*")
_DETAIL_JSON_FIELDS = (
    "pricing",
    "attributes",
    "customization_options",
    "variant_combinations",
    "images",
)


class CatalogError(RuntimeError):
    pass


def normalize_query(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("人民币", "元").replace("块钱", "元")
    text = re.sub(r"(?<=\d)\s*(?:rmb|cny|yuan)\b", "元", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[，,。；;：:！!？?、|]+", " ", text)
    return " ".join(text.split())


def search_tokens(value: object) -> tuple[str, ...]:
    text = normalize_query(value)
    tokens: list[str] = []
    for match in _HAN_RUN.finditer(text):
        run = match.group(0)
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    tokens.extend(_LATIN_OR_NUMBER.findall(text))
    return tuple(dict.fromkeys(token for token in tokens if token))


def _fts_query(tokens: tuple[str, ...]) -> str:
    return " OR ".join(f'"{token}"' for token in tokens)


class BM25Catalog:
    """Expose exactly one retrieval operation: deterministic BM25 Top 5."""

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path).expanduser().resolve()
        if not self.index_path.is_file():
            raise FileNotFoundError(f"BM25 index not found: {self.index_path}")
        self._lock = RLock()
        self._closed = False
        uri = f"file:{self.index_path}?mode=ro"
        self._connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        try:
            row = self._connection.execute("SELECT payload FROM manifest").fetchone()
            if row is None:
                raise CatalogError("index manifest is missing")
            self.manifest = json.loads(row["payload"])
            self._validate_manifest()
            self._connection.execute("SELECT asin FROM product_catalog LIMIT 0")
        except Exception:
            self._connection.close()
            raise

    def _validate_manifest(self) -> None:
        if self.manifest.get("search_version") != SEARCH_VERSION:
            raise CatalogError(
                "unsupported search version: "
                f"{self.manifest.get('search_version')!r}"
            )
        weights = self.manifest.get("field_weights")
        if not isinstance(weights, dict) or set(weights) != set(FIELDS):
            raise CatalogError("index field weights do not match the Harness schema")
        self.weights = (
            UNINDEXED_ASIN_WEIGHT,
            *(float(weights[field]) for field in FIELDS),
        )

    def search_top5(self, query: str) -> list[Candidate]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        tokens = search_tokens(query)
        if not tokens:
            return []
        placeholders = ", ".join("?" for _ in self.weights)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT asin, bm25(products, {placeholders}) AS score "
                "FROM products WHERE products MATCH ? "
                "ORDER BY score ASC, asin ASC LIMIT ?",
                (*self.weights, _fts_query(tokens), TOP_K),
            ).fetchall()
            products = self._products_by_asin([str(row["asin"]) for row in rows])

        candidates: list[Candidate] = []
        for rank, row in enumerate(rows, start=1):
            asin = str(row["asin"])
            product = products.get(asin)
            if product is None:
                raise CatalogError(f"index result missing from product_catalog: {asin}")
            candidates.append(
                Candidate(
                    rank=rank,
                    bm25_score=-float(row["score"]),
                    product=product,
                )
            )
        return candidates

    def _products_by_asin(self, asins: list[str]) -> dict[str, dict[str, Any]]:
        if not asins:
            return {}
        placeholders = ", ".join("?" for _ in asins)
        rows = self._connection.execute(
            f"SELECT * FROM product_catalog WHERE asin IN ({placeholders})",
            asins,
        ).fetchall()
        products: dict[str, dict[str, Any]] = {}
        for row in rows:
            product = dict(row)
            try:
                for field in _DETAIL_JSON_FIELDS:
                    product[field] = json.loads(product[field])
            except (json.JSONDecodeError, TypeError) as exc:
                raise CatalogError(
                    f"invalid catalog JSON for ASIN {product.get('asin')}"
                ) from exc
            products[str(product["asin"])] = product
        return products

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    def __enter__(self) -> "BM25Catalog":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
