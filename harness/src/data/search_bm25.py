from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sqlite3

from .NFKC import normalize_query
from .product_projection import project_public_product

SEARCH_VERSION = "shopsimulator-multifield-bm25-v3"
INDEX_SCHEMA_VERSION = 1
CATALOG_SCHEMA_VERSION = 1
_HAN_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_LATIN_OR_NUMBER = re.compile(r"[a-z0-9]+(?:[._+\-/][a-z0-9]+)*")
_MODEL = re.compile(
    r"(?<![a-z0-9])(?=[a-z0-9._+\-/]*[a-z])(?=[a-z0-9._+\-/]*\d)"
    r"[a-z0-9]+(?:[._+\-/][a-z0-9]+)*(?![a-z0-9])"
)
DEFAULT_FIELD_WEIGHTS = {
    "title": 3.0,
    "brand": 2.0,
    "category": 2.0,
    "model": 2.5,
    "attributes": 1.5,
    "options": 1.2,
    "bullets": 0.8,
}
FIELDS = tuple(DEFAULT_FIELD_WEIGHTS)


class SearchIndexError(RuntimeError):
    """The Environment v2 index is absent, incompatible, or corrupted."""


@dataclass(frozen=True)
class SearchHit:
    asin: str
    score: float
    rank: int


def search_tokens(value: object) -> tuple[str, ...]:
    """返回稳定、有序的搜索词元，包括中文二元组。"""
    text = normalize_query(value)
    tokens = []

    # 提取连续中文，并拆成相邻二字组合
    for match in _HAN_RUN.finditer(text):
        run = match.group(0)

        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))

    # 提取英文、数字和商品型号
    tokens.extend(_LATIN_OR_NUMBER.findall(text))

    # 删除空字符串和重复项，同时保持顺序
    return tuple(dict.fromkeys(token for token in tokens if token))


def _flatten(value: object) -> list[str]:
    """
    把嵌套的数据结构递归展开成一个一维字符串列表。
    """
    if value is None:
        return []
    if isinstance(value, dict):
        flattened = []
        for key in sorted(value, key=str):
            flattened.append(str(key))
            flattened.extend(_flatten(value[key]))
        return flattened
    if isinstance(value, (list, tuple, set)):
        flattened = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    return [str(value)]


def product_fields(product: dict) -> dict[str, str]:
    """Extract the frozen v2 search document without task or reward fields."""
    title = str(product.get("title") or product.get("Title") or "")
    brand = str(product.get("brand") or product.get("shop_name") or "")
    category = str(product.get("category") or product.get("product_category") or "")
    attributes = " ".join(
        _flatten(product.get("attribute") or product.get("Attributes") or [])
    )
    options = " ".join(
        _flatten(product.get("customization_options") or product.get("options") or {})
    )
    bullets = " ".join(
        _flatten(
            product.get("small_description")
            or product.get("sub_title")
            or product.get("BulletPoints")
            or []
        )
    )
    model = " ".join(dict.fromkeys(_MODEL.findall(normalize_query(title))))
    return {
        "title": title,
        "brand": brand,
        "category": category,
        "model": model,
        "attributes": attributes,
        "options": options,
        "bullets": bullets,
    }


def _tokenized(value: object) -> str:
    return " ".join(search_tokens(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_weights(weights: dict[str, float]) -> dict[str, float]:
    if set(weights) != set(FIELDS):
        raise ValueError(f"field weights must contain exactly: {', '.join(FIELDS)}")
    result = {}
    for field in FIELDS:
        value = float(weights[field])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"weight for {field} must be finite and positive")
        result[field] = value
    return result


def build_index(
    products,
    output_path: str | Path,
    *,
    product_data_sha256: str,
    field_weights: dict[str, float] | None = None,
) -> dict:
    """Build an Environment v2 index atomically from an iterable of products."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    if temporary.exists():
        temporary.unlink()
    weights = _validated_weights(field_weights or DEFAULT_FIELD_WEIGHTS)
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "asin UNINDEXED, title, brand, category, model, attributes, options, bullets,"
            " tokenize='unicode61 remove_diacritics 2')"
        )
        connection.execute(
            """
            CREATE TABLE product_catalog (
                asin TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                brand TEXT,
                category TEXT NOT NULL,
                price_min REAL,
                price_max REAL,
                pricing TEXT NOT NULL,
                description TEXT NOT NULL,
                attributes TEXT NOT NULL,
                customization_options TEXT NOT NULL,
                variant_combinations TEXT NOT NULL,
                images TEXT NOT NULL
            )
            """
        )
        seen = set()
        count = 0
        for product in products:
            asin = str(product.get("asin", "")).strip()
            if not asin or asin == "nan" or asin in seen:
                continue
            fields = product_fields(product)
            public_product = project_public_product(product)

            connection.execute(
                """
                INSERT INTO products(
                    asin,
                    title,
                    brand,
                    category,
                    model,
                    attributes,
                    options,
                    bullets
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asin,
                    *(_tokenized(fields[field]) for field in FIELDS),
                ),
            )

            connection.execute(
                """
                INSERT INTO product_catalog(
                    asin,
                    title,
                    brand,
                    category,
                    price_min,
                    price_max,
                    pricing,
                    description,
                    attributes,
                    customization_options,
                    variant_combinations,
                    images
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    public_product["asin"],
                    public_product["title"],
                    public_product["brand"],
                    public_product["category"],
                    public_product["price_min"],
                    public_product["price_max"],
                    json.dumps(
                        public_product["pricing"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    public_product["description"],
                    json.dumps(
                        public_product["attributes"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        public_product["customization_options"],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        public_product["variant_combinations"],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        public_product["images"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ),
            )

            seen.add(asin)
            count += 1
        manifest = {
            "search_version": SEARCH_VERSION,
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "product_count": count,
            "product_data_sha256": str(product_data_sha256),
            "field_weights": weights,
            "query_normalizer": "nfkc-casefold-basic-punctuation-v1",
            "tokenizer": "han-bigram-latin-number-v1",
            "tie_breaker": "asin-ascending",
            "python_version": platform.python_version(),
            "sqlite_version": sqlite3.sqlite_version,
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "public_projection_version": "product-public-v1",
        }
        connection.execute("CREATE TABLE manifest(payload TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO manifest(payload) VALUES (?)",
            (json.dumps(manifest, ensure_ascii=False, sort_keys=True),),
        )
        connection.commit()
    except Exception:
        connection.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        connection.close()
    temporary.replace(output)
    manifest["index_sha256"] = sha256_file(output)
    return manifest


def _fts_query(tokens: tuple[str, ...]) -> str:
    # Every token is generated by this module and contains no quote, but quoting
    # still keeps FTS syntax separate from user-provided text.
    return " OR ".join(f'"{token}"' for token in tokens)


class MultiFieldBM25Searcher:
    """Read-only, deterministic search interface used by SimServer."""

    def __init__(
        self, index_path: str | Path, *, expected_product_sha256: str | None = None
    ):
        self.index_path = Path(index_path).resolve()
        if not self.index_path.is_file():
            raise SearchIndexError(f"Environment index not found: {self.index_path}. ")
        uri = f"file:{self.index_path}?mode=ro"
        self._connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        from .product_facts import sqlite_predicates
        self._connection.create_function("public_predicates", 9, sqlite_predicates, deterministic=True)
        self._connection.create_function(
            "normalize_public",
            1,
            lambda value: normalize_query(value or ""),
            deterministic=True,
        )
        try:
            row = self._connection.execute("SELECT payload FROM manifest").fetchone()
            self.manifest = json.loads(row["payload"])
        except Exception as exc:
            self._connection.close()
            raise SearchIndexError(
                f"invalid Environment v2 index manifest: {exc}"
            ) from exc
        if self.manifest.get("search_version") != SEARCH_VERSION:
            raise SearchIndexError(
                f"search version mismatch: {self.manifest.get('search_version')!r}"
            )
        if self.manifest.get("index_schema_version") != INDEX_SCHEMA_VERSION:
            raise SearchIndexError("index schema version mismatch")
        if (
            expected_product_sha256 is not None
            and self.manifest.get("product_data_sha256") != expected_product_sha256
        ):
            raise SearchIndexError(
                "product data SHA-256 does not match the search index"
            )
        self.weights = _validated_weights(self.manifest["field_weights"])

    def search(
        self,
        query: object,
        k: int = 150,
        *,
        filters: dict | None = None,
        offset: int = 0,
        exclude_asins: list[str] | None = None,
    ) -> list[SearchHit]:
        tokens = search_tokens(query)
        if not tokens or int(k) <= 0:
            return []
        # bm25() weights are positional and include every FTS5 column.  The
        # first column is the unindexed ASIN, so reserve its position before
        # appending the weights for title through bullets.
        weight_values = [0.0, *(self.weights[field] for field in FIELDS)]
        weight_placeholders = ", ".join("?" for _ in weight_values)
        where, parameters = _search_where(tokens, filters, exclude_asins)
        rows = self._connection.execute(
            f"SELECT products.asin, bm25(products, {weight_placeholders}) AS score FROM products "
            "LEFT JOIN product_catalog c ON c.asin = products.asin WHERE "
            + where
            + " ORDER BY score ASC, products.asin ASC LIMIT ? OFFSET ?",
            (*weight_values, *parameters, int(k), offset),
        ).fetchall()
        return [
            SearchHit(asin=row["asin"], score=-float(row["score"]), rank=index + offset)
            for index, row in enumerate(rows, start=1)
        ]

    def count(self, query, *, filters=None, exclude_asins=None) -> int:
        """Count the identical filtered match set without sorting/materializing hits."""
        tokens = search_tokens(query)
        if not tokens:
            return 0
        where, parameters = _search_where(tokens, filters, exclude_asins)
        row = self._connection.execute(
            "SELECT count(*) FROM products LEFT JOIN product_catalog c ON c.asin=products.asin WHERE " + where,
            parameters).fetchone()
        return int(row[0])

    def contains_asin(self, asin: object) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM products WHERE asin = ? LIMIT 1",
            (str(asin),),
        ).fetchone()
        return row is not None

    def close(self):
        self._connection.close()


def _search_where(tokens, filters, exclude_asins):
    clauses = ["products MATCH ?"]
    parameters = [_fts_query(tokens)]
    filters = filters or {}
    if filters.get("predicates"):
        from .product_facts import SQL_PREDICATES
        clauses.append(SQL_PREDICATES)
        parameters.append(json.dumps(filters["predicates"], ensure_ascii=False, sort_keys=True))
    if filters.get("category"):
        clauses.append("instr(normalize_public(c.category), ?) > 0")
        parameters.append(normalize_query(filters["category"]))
    if filters.get("brand"):
        clauses.append("normalize_public(c.brand) = ?")
        parameters.append(normalize_query(filters["brand"]))
    for key, column, operator in (
        ("min_price", "price_max", ">="),
        ("max_price", "price_min", "<="),
    ):
        if filters.get(key) is not None:
            condition = f"c.{column} {operator} ?"
            if filters.get("include_unknown_price", True):
                condition = f"({condition} OR c.{column} IS NULL)"
            clauses.append(condition)
            parameters.append(filters[key])
    if exclude_asins:
        clauses.append(f"c.asin NOT IN ({','.join('?' for _ in exclude_asins)})")
        parameters.extend(exclude_asins)
    # Keep orphaned FTS hits visible to ProductCatalog's integrity check;
    # an inner join would silently hide a corrupted public catalog.
    clauses = [
        clauses[0],
        *(f"(c.asin IS NULL OR {clause})" for clause in clauses[1:]),
    ]
    return " AND ".join(clauses), parameters
