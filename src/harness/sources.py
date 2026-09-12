"""Public product sources for unlabelled interactive shopping tasks.

HTTP adapters implement our small read-only gateway contract, not an assumed
retailer API. Credentials and raw responses never enter source manifests.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, HTTPRedirectHandler, build_opener

from .catalog import CatalogDataError, ProductNotFoundError
from .schemas import ProductDetail, ProductSummary, SearchPage, SearchProductsAction
from ..data.search_bm25 import search_tokens
from ..data.NFKC import normalize_query


@runtime_checkable
class ProductSource(Protocol):
    @property
    def manifest(self) -> dict: ...
    def search_page(
        self, request: SearchProductsAction
    ) -> tuple[list[ProductSummary], SearchPage]: ...
    def get_product(self, asin: str) -> ProductDetail: ...
    def close(self) -> None: ...


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _public_product(value):
    product = ProductDetail.model_validate(value)
    if (
        not product.asin.strip()
        or not product.title.strip()
        or not product.category.strip()
    ):
        raise ValueError("product identity fields must not be empty")
    if product.asin in {".", ".."}:
        raise ValueError("invalid product ID")
    # Availability defaults in the benchmark schema cannot establish external stock.
    for options in product.customization_options.values():
        if any("is_available" not in option.model_fields_set for option in options):
            raise ValueError("external option availability must be explicit")
    if any(
        "is_available" not in variant.model_fields_set
        for variant in product.variant_combinations
    ):
        raise ValueError("external variant availability must be explicit")
    prices = [product.price_min, product.price_max, *product.pricing]
    prices += [
        o.price for options in product.customization_options.values() for o in options
    ]
    prices += [v.price for v in product.variant_combinations]
    if any(p is not None and (not math.isfinite(p) or p < 0) for p in prices):
        raise ValueError("product prices must be finite and non-negative")
    if (
        product.price_min is not None
        and product.price_max is not None
        and product.price_min > product.price_max
    ):
        raise ValueError("invalid product price range")
    return product


class _Source:
    def __init__(self, source_id, currency=None):
        if (
            not isinstance(source_id, str)
            or not source_id.strip()
            or len(source_id) > 100
        ):
            raise ValueError("source_id must contain 1-100 characters")
        if currency is not None and (len(currency) != 3 or not currency.isalpha()):
            raise ValueError("currency must be an explicit three-letter code or null")
        self.source_id = source_id
        self.currency = currency.upper() if currency else None
        self.last_observation_source = None
        self._closed = False

    def _record(self, public_payload):
        self.last_observation_source = {
            "source_id": self.source_id,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "content_sha256": _digest(public_payload),
            "currency": self.currency,
            "price_guarantee": "observed_snapshot_only",
        }

    def _ensure_open(self):
        if self._closed:
            raise CatalogDataError("product source is closed")

    def close(self):
        self._closed = True


class JsonProductSource(_Source):
    """Load a public ProductDetail array; snapshot and deterministic token-overlap search."""

    def __init__(self, path: str | Path, *, source_id="public-json", currency=None):
        super().__init__(source_id, currency)
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("public product JSON must be an array")
        products = [_public_product(row) for row in payload]
        if len({p.asin for p in products}) != len(products):
            raise ValueError("duplicate product IDs in public source")
        self._products = {p.asin: p for p in products}
        self._content_hash = _digest(
            [p.model_dump(mode="json", exclude_unset=True) for p in products]
        )

    @property
    def manifest(self):
        return {
            "type": "public-json-v1",
            "source_id": self.source_id,
            "currency": self.currency,
            "content_sha256": self._content_hash,
            "ranking": "token_overlap_then_asin",
        }

    def get_product(self, asin):
        self._ensure_open()
        if asin not in self._products:
            raise ProductNotFoundError("product not found in public source")
        product = self._products[asin].model_copy(deep=True)
        self._record(product.model_dump(mode="json", exclude_unset=True))
        return product

    def search_page(self, request):
        if request.filters.predicates:
            raise ValueError("typed predicate filtering requires the SQLite catalog")
        self._ensure_open()
        query = set(search_tokens(request.query))
        filters = request.filters
        ranked = []
        for product in self._products.values():
            if product.asin in request.exclude_asins:
                continue
            if filters.category and normalize_query(
                filters.category
            ) not in normalize_query(product.category):
                continue
            if filters.brand and normalize_query(filters.brand) != normalize_query(
                product.brand or ""
            ):
                continue
            if filters.min_price is not None or filters.max_price is not None:
                if product.price_min is None or product.price_max is None:
                    if not filters.include_unknown_price:
                        continue
                elif (
                    filters.min_price is not None
                    and product.price_max < filters.min_price
                ) or (
                    filters.max_price is not None
                    and product.price_min > filters.max_price
                ):
                    continue
            text = " ".join(
                [
                    product.title,
                    product.category,
                    product.brand or "",
                    product.description,
                    *product.attributes,
                    *(
                        o.value
                        for options in product.customization_options.values()
                        for o in options
                    ),
                ]
            )
            score = len(query & set(search_tokens(text)))
            if score:
                ranked.append((score, product))
        ranked.sort(key=lambda pair: (-pair[0], pair[1].asin))
        selected = ranked[request.offset : request.offset + request.limit]
        items = [
            ProductSummary(
                asin=p.asin,
                title=p.title,
                category=p.category,
                brand=p.brand,
                price_min=p.price_min,
                price_max=p.price_max,
                # Compatibility field only; this source does not claim to run BM25.
                bm25_score=0,
                rank=request.offset + rank,
            )
            for rank, (_, p) in enumerate(selected, 1)
        ]
        more = request.offset + len(items) < len(ranked)
        page = SearchPage(
            offset=request.offset,
            total_results=len(ranked),
            has_more=more,
            next_offset=request.offset + len(items) if more else None,
        )
        self._record(
            {
                "items": [p.model_dump(mode="json") for p in items],
                "page": page.model_dump(),
            }
        )
        return items, page


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "redirects disabled", headers, fp)


class HttpProductSource(_Source):
    """POST /search -> {items, page}; GET /products/{id} -> ProductDetail.

    Only an explicitly configured base URL is called. The gateway must return
    search ranks, apply all filters/paging, and omit unavailable detail fields.
    """

    def __init__(
        self,
        base_url,
        *,
        source_id="public-http",
        currency=None,
        api_key=None,
        timeout=15.0,
        max_response_bytes=2_000_000,
    ):
        super().__init__(source_id, currency)
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "source base URL must be HTTP(S) without credentials, query or fragment"
            )
        if not math.isfinite(timeout) or timeout <= 0 or max_response_bytes < 1:
            raise ValueError("invalid source timeout/response limit")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self._api_key = api_key
        self._opener = build_opener(_NoRedirect())

    @property
    def manifest(self):
        return {
            "type": "public-http-v1",
            "source_id": self.source_id,
            "base_url": self.base_url,
            "currency": self.currency,
            "timeout": self.timeout,
            "max_response_bytes": self.max_response_bytes,
            "ranking": "gateway_rank",
            "data_freshness": "per_request_snapshot",
        }

    def _request(self, path, payload=None):
        self._ensure_open()
        headers = {"Accept": "application/json"}
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            headers["Content-Type"] = "application/json"
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = Request(self.base_url + path, data=data, headers=headers)
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read(self.max_response_bytes + 1)
            if len(raw) > self.max_response_bytes:
                raise CatalogDataError(
                    "public source response exceeds configured byte limit"
                )
            return json.loads(raw)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            # Do not include raw body, credentials or request URLs in actor errors.
            raise CatalogDataError(
                "public source request failed or returned invalid JSON"
            ) from exc

    def get_product(self, asin):
        if not isinstance(asin, str) or not asin.strip() or asin in {".", ".."}:
            raise CatalogDataError("invalid public product ID")
        raw = self._request("/products/" + quote(asin, safe=""))
        try:
            product = _public_product(raw)
            if product.asin != asin:
                raise ValueError("mismatched product ID")
        except (ValueError, TypeError) as exc:
            raise CatalogDataError("public source returned an invalid product") from exc
        self._record(product.model_dump(mode="json", exclude_unset=True))
        return product

    def search_page(self, request):
        raw = self._request("/search", request.model_dump(mode="json"))
        try:
            items = [ProductSummary.model_validate(row) for row in raw["items"]]
            page = SearchPage.model_validate(raw["page"])
            if len(items) > request.limit or len({p.asin for p in items}) != len(items):
                raise ValueError("invalid result count or duplicates")
            if page.offset != request.offset or (
                page.has_more
                and (not items or page.next_offset != request.offset + len(items))
            ):
                raise ValueError("invalid search pagination")
            if not page.has_more and page.next_offset is not None:
                raise ValueError("unexpected next page")
            for rank, product in enumerate(items, request.offset + 1):
                if product.asin in request.exclude_asins or product.rank != rank:
                    raise ValueError("invalid search result identity or rank")
                if not math.isfinite(product.bm25_score):
                    raise ValueError("invalid search score")
                if len(product.match_snippets) > 4 or any(
                    s.field_path.split("/")[1:2]
                    not in [[field] for field in ProductDetail.model_fields]
                    or len(s.text) > 500
                    for s in product.match_snippets
                ):
                    raise ValueError(
                        "search snippets must be bounded public field references"
                    )
                _public_product(
                    product.model_dump(exclude={"bm25_score", "rank", "match_snippets"})
                )
        except (ValueError, TypeError, KeyError) as exc:
            raise CatalogDataError(
                "public source returned an invalid search page"
            ) from exc
        self._record(
            {
                "items": [p.model_dump(mode="json") for p in items],
                "page": page.model_dump(),
            }
        )
        return items, page
