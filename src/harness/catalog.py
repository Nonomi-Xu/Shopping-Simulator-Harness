"""Read-only runtime access to the public product catalog.

The build pipeline stores both the BM25 index and a strictly projected public
product record in the same SQLite database.  This module is the runtime boundary
around that database: callers can search for summaries or fetch a product
detail, but they never receive the private task fields from the source dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any, Iterable

from pydantic import ValidationError

from ..data.search_bm25 import (
    CATALOG_SCHEMA_VERSION,
    MultiFieldBM25Searcher,
)
from .schemas import ProductDetail, ProductSummary


PUBLIC_PROJECTION_VERSION = "product-public-v1"
MAX_SEARCH_LIMIT = 50

_DETAIL_JSON_FIELDS = (
    "pricing",
    "attributes",
    "customization_options",
    "variant_combinations",
    "images",
)


class CatalogError(RuntimeError):
    """The public catalog is unavailable, incompatible, or corrupted."""


class ProductNotFoundError(CatalogError, LookupError):
    """The requested ASIN does not exist in the public catalog."""


class CatalogDataError(CatalogError):
    """A stored public product cannot be decoded or validated."""


class ProductCatalog:
    """Search and retrieve public products from a prepared Harness index."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        expected_product_sha256: str | None = None,
    ) -> None:
        self.index_path = Path(index_path).resolve()
        self._lock = RLock()
        self._closed = False
        self._searcher = MultiFieldBM25Searcher(
            self.index_path,
            expected_product_sha256=expected_product_sha256,
        )

        try:
            self._validate_manifest(self._searcher.manifest)
            uri = f"file:{self.index_path}?mode=ro"
            self._connection = sqlite3.connect(
                uri,
                uri=True,
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute(
                "SELECT asin FROM product_catalog LIMIT 0"
            )
        except Exception:
            self._searcher.close()
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            self._closed = True
            raise

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a copy of the immutable index manifest."""
        self._ensure_open()
        return dict(self._searcher.manifest)

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> list[ProductSummary]:
        """Return public product summaries in deterministic BM25 rank order."""
        self._ensure_open()
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if not 1 <= limit <= MAX_SEARCH_LIMIT:
            raise ValueError(
                f"limit must be between 1 and {MAX_SEARCH_LIMIT}"
            )

        with self._lock:
            hits = self._searcher.search(query, k=limit)
            if not hits:
                return []
            rows = self._rows_by_asin(hit.asin for hit in hits)

        summaries = []
        for hit in hits:
            row = rows.get(hit.asin)
            if row is None:
                raise CatalogDataError(
                    "search index references an ASIN missing from "
                    f"product_catalog: {hit.asin}"
                )
            try:
                summaries.append(
                    ProductSummary(
                        asin=row["asin"],
                        title=row["title"],
                        brand=row["brand"],
                        category=row["category"],
                        price_min=row["price_min"],
                        price_max=row["price_max"],
                        bm25_score=hit.score,
                        rank=hit.rank,
                    )
                )
            except ValidationError as exc:
                raise CatalogDataError(
                    f"invalid public product summary for ASIN {hit.asin}: {exc}"
                ) from exc
        return summaries

    def get_product(self, asin: str) -> ProductDetail:
        """Return one public product detail or raise ProductNotFoundError."""
        self._ensure_open()
        normalized_asin = self._validated_asin(asin)
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM product_catalog WHERE asin = ? LIMIT 1",
                (normalized_asin,),
            ).fetchone()

        if row is None:
            raise ProductNotFoundError(
                f"product not found: {normalized_asin}"
            )
        return self._detail_from_row(row)

    def contains(self, asin: object) -> bool:
        """Return whether an ASIN has a readable public catalog record."""
        self._ensure_open()
        normalized_asin = str(asin or "").strip()
        if not normalized_asin:
            return False
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM product_catalog WHERE asin = ? LIMIT 1",
                (normalized_asin,),
            ).fetchone()
        return row is not None

    def close(self) -> None:
        """Close the catalog. Calling close more than once is safe."""
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._searcher.close()
            self._closed = True

    def __enter__(self) -> ProductCatalog:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _rows_by_asin(
        self,
        asins: Iterable[str],
    ) -> dict[str, sqlite3.Row]:
        asin_list = list(asins)
        placeholders = ", ".join("?" for _ in asin_list)
        rows = self._connection.execute(
            "SELECT asin, title, brand, category, price_min, price_max "
            f"FROM product_catalog WHERE asin IN ({placeholders})",
            asin_list,
        ).fetchall()
        return {row["asin"]: row for row in rows}

    @staticmethod
    def _validated_asin(asin: str) -> str:
        if not isinstance(asin, str):
            raise TypeError("asin must be a string")
        normalized_asin = asin.strip()
        if not normalized_asin:
            raise ValueError("asin must not be empty")
        return normalized_asin

    @staticmethod
    def _validate_manifest(manifest: dict[str, Any]) -> None:
        catalog_version = manifest.get("catalog_schema_version")
        if catalog_version != CATALOG_SCHEMA_VERSION:
            raise CatalogError(
                "catalog schema version mismatch: "
                f"expected {CATALOG_SCHEMA_VERSION}, got {catalog_version!r}"
            )
        projection_version = manifest.get("public_projection_version")
        if projection_version != PUBLIC_PROJECTION_VERSION:
            raise CatalogError(
                "public projection version mismatch: "
                f"expected {PUBLIC_PROJECTION_VERSION!r}, "
                f"got {projection_version!r}"
            )

    @staticmethod
    def _detail_from_row(row: sqlite3.Row) -> ProductDetail:
        payload = dict(row)
        try:
            for field in _DETAIL_JSON_FIELDS:
                payload[field] = json.loads(payload[field])
            return ProductDetail(**payload)
        except (json.JSONDecodeError, TypeError, ValidationError) as exc:
            raise CatalogDataError(
                f"invalid public product detail for ASIN {row['asin']}: {exc}"
            ) from exc

    def _ensure_open(self) -> None:
        if self._closed:
            raise CatalogError("catalog is closed")
