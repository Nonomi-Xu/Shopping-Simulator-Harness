import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from decompress import decompress_gzip


def json_to_duckdb(
    json_path: str,
    database_path: str | Path = "preview.duckdb",
    table_name: str = "items",
) -> Path:
    '''
    数据预览
    将 JSON 文件导入 DuckDB 数据库。
    '''
    json_file = Path(json_path).resolve()
    database_file = Path(database_path).resolve()

    if not json_file.exists():
        raise FileNotFoundError(f"JSON 文件不存在：{json_file}")

    safe_json_path = str(json_file).replace("'", "''")
    safe_table_name = table_name.replace('"', '""')

    with duckdb.connect(str(database_file)) as connection:
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE "{safe_table_name}" AS
            SELECT *
            FROM read_json_auto(
                '{safe_json_path}',
                format = 'auto'
            )
            """
        )

        row_count = connection.execute(
            f'SELECT COUNT(*) FROM "{safe_table_name}"'
        ).fetchone()[0]

        print(f"DuckDB 文件：{database_file}")
        print(f"数据表：{table_name}")
        print(f"总行数：{row_count}")

        print("\n字段结构：")
        print(
            connection.execute(
                f'DESCRIBE "{safe_table_name}"'
            ).fetchdf().to_string(index=False)
        )

def main() -> None:
    gzip_path = PROJECT_ROOT / "fine_items_eval_train_all.json.gz"
    database_path = PROJECT_ROOT / "test" / "preview.duckdb"

    json_path = decompress_gzip(
        input_path=str(gzip_path),
        overwrite=True,
    )

    json_to_duckdb(
        json_path=json_path,
        database_path=database_path,
        table_name="items",
    )


if __name__ == "__main__":
    main()