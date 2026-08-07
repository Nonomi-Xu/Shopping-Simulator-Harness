from __future__ import annotations

from decompress import decompress_gzip

from pathlib import Path

from search_bm25 import (
    MultiFieldBM25Searcher,
    build_index,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parent

GZIP_PATH = PROJECT_ROOT / "data" /"fine_items_eval_train_all.json.gz"
INDEX_PATH = PROJECT_ROOT / "data" / "product_search.sqlite"


def main():
    output_path = decompress_gzip(
        input_path = GZIP_PATH,
    )
    print(f"输出文件：{output_path}")


if __name__ == "__main__":
    main()