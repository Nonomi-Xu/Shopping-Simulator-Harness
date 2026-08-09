from __future__ import annotations

from pathlib import Path
import json
from .search_bm25 import sha256_file, INDEX_SCHEMA_VERSION, CATALOG_SCHEMA_VERSION
from .build_index import iter_json_array

HARNESS_ENV = Path(__file__).resolve().parents[2]

INDEX_PATH = HARNESS_ENV / "data" / "product_search.sqlite3"
MANIFEST_PATH = HARNESS_ENV / "data" / "product_search.manifest.json"



def validate_products(path: Path) -> dict:
    """校验商品数据，并返回可审计的数据统计。"""
    seen_asins: set[str] = set()
    split_counts = {"train": 0, "eval": 0}

    for row_number, product in enumerate(iter_json_array(path), start=1):
        asin = str(product.get("asin") or "").strip()
        if not asin:
            raise ValueError(f"第 {row_number} 个商品缺少 asin")

        if asin in seen_asins:
            raise ValueError(f"发现重复 asin：{asin}")
        seen_asins.add(asin)

        tag = product.get("tag")
        if tag not in split_counts:
            raise ValueError(
                f"商品 {asin} 的 tag 非法：{tag!r}"
            )
        split_counts[tag] += 1

        instructions = product.get("instructions")
        if not isinstance(instructions, list) or not instructions:
            raise ValueError(f"商品 {asin} 缺少 instructions")

    return {
        "product_count": len(seen_asins),
        "split_counts": split_counts,
    }

def index_is_current(product_sha256: str) -> bool:
    """检查现有索引及其 manifest 是否与当前商品数据一致。"""
    if not INDEX_PATH.is_file() or not MANIFEST_PATH.is_file():
        return False

    try:
        manifest = json.loads(
            MANIFEST_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False

    if manifest.get("product_data_sha256") != product_sha256:
        return False

    expected_index_sha = manifest.get("index_sha256")
    if not expected_index_sha:
        return False

    if manifest.get("index_schema_version") != INDEX_SCHEMA_VERSION:
        return False

    if manifest.get("catalog_schema_version") != CATALOG_SCHEMA_VERSION:
        return False

    return sha256_file(INDEX_PATH) == expected_index_sha