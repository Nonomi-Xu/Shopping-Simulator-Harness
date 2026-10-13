from __future__ import annotations

from pathlib import Path

import argparse
import json
from pathlib import Path

from .build_index import iter_json_array
from .decompress import decompress_gzip
from .search_bm25 import build_index, sha256_file
from .validate import validate_products, index_is_current

HARNESS_ENV = Path(__file__).resolve().parents[2]

GZIP_PATH = HARNESS_ENV / "data" /"fine_items_eval_train_all.json.gz"
INDEX_PATH = HARNESS_ENV / "data" / "product_search.sqlite3"
PRODUCTS_PATH = HARNESS_ENV / "data" / "fine_items_eval_train_all.json"
MANIFEST_PATH = HARNESS_ENV / "data" / "product_search.manifest.json"

def prepare_data(*, force: bool = False) -> dict:
    """生成 Harness 运行所需的数据和检索索引。"""
    if not GZIP_PATH.is_file():
        raise FileNotFoundError(f"找不到数据压缩包：{GZIP_PATH}")

    if force or not PRODUCTS_PATH.is_file():
        decompress_gzip(
            input_path=str(GZIP_PATH),
            output_path=str(PRODUCTS_PATH),
            overwrite=force,
        )

    validation = validate_products(PRODUCTS_PATH)
    product_sha256 = sha256_file(PRODUCTS_PATH)

    if force or not index_is_current(product_sha256):
        manifest = build_index(
            iter_json_array(PRODUCTS_PATH),
            INDEX_PATH,
            product_data_sha256=product_sha256,
        )
    else:
        manifest = json.loads(
            MANIFEST_PATH.read_text(encoding="utf-8")
        )

    # Refresh selection metadata even when the product index is already current.
    from ..harness.tasks import TaskStore

    manifest["dataset"] = validation
    manifest["tasks"] = TaskStore(PRODUCTS_PATH).manifest
    manifest["source_gzip_sha256"] = sha256_file(GZIP_PATH)
    temporary_manifest = MANIFEST_PATH.with_suffix(MANIFEST_PATH.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(MANIFEST_PATH)

    return manifest

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Shopping Simulator Harness data"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="重新解压数据并重建搜索索引",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    manifest = prepare_data(force=args.force)
    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()