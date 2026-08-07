from __future__ import annotations

import gzip
import shutil
from pathlib import Path


def decompress_gzip(input_path: str, output_path: str | None = None, overwrite: bool = True) -> Path:
    """
    解压 .gz 文件

    Args:
        input_path: .gz 文件路径。
        output_path: 输出文件路径。不传时自动移除 .gz 后缀。
        overwrite: 是否覆盖已存在的输出文件。
    Returns:
        解压后文件的路径。
    """
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"文件不存在：{input_file}")

    if input_file.suffix != ".gz":
        raise ValueError(f"输入文件必须是 .gz 文件：{input_file}")

    output_file = (
        Path(output_path)
        if output_path
        else input_file.with_suffix("")
    )

    if output_file.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在：{output_file}")

    with gzip.open(input_file, "rb") as compressed_file:
        with output_file.open("wb") as decompressed_file:
            shutil.copyfileobj(compressed_file, decompressed_file)

    print(f"解压完成：{output_file}")
    return output_file


if __name__ == "__main__":
    decompress_gzip("fine_items_eval_train_all.json.gz", overwrite=True)