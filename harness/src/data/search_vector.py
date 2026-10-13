"""Persisted cosine retrieval using real, explicitly configured embeddings.

Only public product fields are embedded. Building an index calls the embedding
provider; opening it never does. Credentials are never written into the index.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import math
from pathlib import Path
import sqlite3
import tempfile
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .NFKC import normalize_query
from .search_bm25 import SearchHit, sha256_file


VECTOR_VERSION = "public-product-cosine-v1"


class VectorSearchError(RuntimeError):
    """An embedding provider or vector index is unavailable or incompatible."""


class EmbeddingProvider(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "redirects disabled", headers, fp)


class OpenAIEmbeddingProvider:
    """OpenAI-compatible POST /embeddings; no implicit endpoint or model."""

    def __init__(self, model: str, base_url: str, api_key: str | None = None,
                 timeout: float = 30.0):
        parsed = urlsplit(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.netloc
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("embedding base URL must be HTTP(S) without credentials/query")
        if not model.strip() or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("embedding model and positive finite timeout are required")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.timeout = timeout
        self._opener = build_opener(_NoRedirect())

    def embed(self, texts):
        if not texts:
            return []
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = Request(
            self.base_url + "/embeddings",
            data=json.dumps({"model": self.model, "input": texts,
                             "encoding_format": "float"}).encode(), headers=headers,
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read(32_000_001)
            if len(raw) > 32_000_000:
                raise ValueError("oversized embedding response")
            payload = json.loads(raw)
            rows = payload["data"]
            if not isinstance(rows, list) or len(rows) != len(texts):
                raise ValueError("wrong embedding count")
            indexed = {}
            for row in rows:
                position = row["index"]
                if type(position) is not int or position in indexed:
                    raise ValueError("invalid embedding index")
                indexed[position] = row["embedding"]
            if set(indexed) != set(range(len(texts))):
                raise ValueError("missing embedding index")
            return [indexed[i] for i in range(len(texts))]
        except (HTTPError, URLError, OSError, TimeoutError, ValueError, KeyError, TypeError) as exc:
            raise VectorSearchError("embedding request failed or returned invalid data") from exc


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise VectorSearchError("vector search requires the 'vector' NumPy extra") from exc
    return np


def _normalized(vectors, count, dimension=None):
    np = _numpy()
    try:
        matrix = np.asarray(vectors, dtype=np.float32)
        if (matrix.ndim != 2 or matrix.shape[0] != count or matrix.shape[1] < 1
                or (dimension is not None and matrix.shape[1] != dimension)
                or not np.isfinite(matrix).all()):
            raise ValueError("invalid shape or values")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("zero or overflowing vector")
        return matrix / norms
    except (ValueError, TypeError, OverflowError) as exc:
        raise VectorSearchError("embeddings must be finite nonzero vectors with consistent dimensions") from exc


def _public_text(product):
    """Allowlist public meaning-bearing text; never serialize entire source rows."""
    fields = [str(product.get(k) or "") for k in ("title", "brand", "category", "description")]
    fields.extend(str(v) for v in product.get("attributes", []))
    for name, options in product.get("customization_options", {}).items():
        fields.append(str(name))
        fields.extend(str(option.get("value", "")) for option in options)
    return "\n".join(fields)


def _build(products, source_path, output_path, provider, batch_size):
    np = _numpy()
    source_path, output_path = Path(source_path).resolve(), Path(output_path).resolve()
    if output_path == source_path:
        raise ValueError("vector output must differ from source")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    products = sorted(products, key=lambda p: p["asin"])
    asins = [p["asin"] for p in products]
    if not asins or any(not isinstance(a, str) or not a.strip() for a in asins) or len(set(asins)) != len(asins):
        raise VectorSearchError("vector source must have nonempty unique product IDs")
    source_hash = sha256_file(source_path)
    batches, dimension = [], None
    for start in range(0, len(products), batch_size):
        texts = [_public_text(p) for p in products[start:start + batch_size]]
        matrix = _normalized(provider.embed(texts), len(texts), dimension)
        dimension = matrix.shape[1]
        batches.append(matrix)
    if sha256_file(source_path) != source_hash:
        raise VectorSearchError("source changed during embedding build; rebuild index")
    manifest = {
        "search_version": VECTOR_VERSION, "source_sha256": source_hash,
        "model": provider.model, "dimension": dimension, "product_count": len(asins),
        "similarity": "cosine", "tie_breaker": "asin-ascending",
        "text_projection": "public-title-brand-category-description-attributes-options-v1",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".building", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, manifest=json.dumps(manifest, sort_keys=True),
                                asins=np.asarray(asins), vectors=np.concatenate(batches))
        temporary.replace(output_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return manifest


def build_vector_index(catalog_path, output_path, provider, batch_size=64):
    """Embed public SQLite catalog records into an atomic NumPy index."""
    path = Path(catalog_path).resolve()
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        products = [dict(row) for row in connection.execute("SELECT * FROM product_catalog ORDER BY asin")]
    for product in products:
        for field in ("attributes", "customization_options"):
            product[field] = json.loads(product[field])
    return _build(products, path, output_path, provider, batch_size)


def build_json_vector_index(source_path, output_path, provider, batch_size=64):
    """Embed public ProductDetail JSON, sharing the source adapter's validation."""
    from ..harness.sources import _public_product

    payload = json.loads(Path(source_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("public product JSON must be an array")
    products = [_public_product(row).model_dump(mode="json") for row in payload]
    return _build(products, source_path, output_path, provider, batch_size)


class VectorSearcher:
    """Read-only normalized matrix, source/model binding, stable exact cosine rank."""

    def __init__(self, index_path, catalog_path, provider):
        np = _numpy()
        self.provider = provider
        self._queries = OrderedDict()
        try:
            with np.load(index_path, allow_pickle=False) as archive:
                self.manifest = json.loads(str(archive["manifest"].item()))
                self.asins = archive["asins"].tolist()
                self._vectors = _normalized(archive["vectors"], len(self.asins),
                                            self.manifest["dimension"])
            if (self.manifest["search_version"] != VECTOR_VERSION
                    or self.manifest["source_sha256"] != sha256_file(catalog_path)
                    or self.manifest["model"] != provider.model
                    or self.manifest["product_count"] != len(self.asins)
                    or not self.asins or self.asins != sorted(set(self.asins))
                    or any(not isinstance(a, str) or not a.strip() for a in self.asins)):
                raise ValueError("vector source/model/manifest mismatch")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise VectorSearchError("vector index is invalid or does not match source/model; rebuild it") from exc

    def search(self, query, k=20, *, allowed_asins=None, offset=0):
        np = _numpy()
        if not isinstance(query, str) or not query.strip() or k <= 0:
            return []
        # Reuse an identical query embedding when paging; no repeated provider cost.
        if query not in self._queries:
            self._queries[query] = _normalized(self.provider.embed([query]), 1,
                                              self.manifest["dimension"])[0]
            if len(self._queries) > 64:
                self._queries.popitem(last=False)
        self._queries.move_to_end(query)
        scores = self._vectors @ self._queries[query]
        indices = np.asarray([i for i, asin in enumerate(self.asins)
                              if allowed_asins is None or asin in allowed_asins], dtype=np.int64)
        # ASINs are already sorted; stable sorting preserves deterministic ties.
        indices = indices[np.argsort(-scores[indices], kind="stable")][offset:offset + k]
        return [SearchHit(self.asins[i], float(scores[i]), offset + rank)
                for rank, i in enumerate(indices, 1)]


def matches_filters(product, filters, exclude_asins=()):
    """Same range-overlap semantics as SQLite BM25; this is not SKU validation."""
    if product["asin"] in exclude_asins:
        return False
    if filters.get("category") and normalize_query(filters["category"]) not in normalize_query(product.get("category") or ""):
        return False
    if filters.get("brand") and normalize_query(filters["brand"]) != normalize_query(product.get("brand") or ""):
        return False
    for key, field, minimum in (("min_price", "price_max", True), ("max_price", "price_min", False)):
        bound = filters.get(key)
        if bound is None:
            continue
        price = product.get(field)
        if price is None:
            if not filters.get("include_unknown_price", True):
                return False
        elif (minimum and price < bound) or (not minimum and price > bound):
            return False
    return True
