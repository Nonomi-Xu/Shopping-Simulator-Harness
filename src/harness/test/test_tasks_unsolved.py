#!/usr/bin/env python3
"""Export tasks whose annotated options cannot be resolved uniquely.

This is a data-diagnostic utility rather than a unit test. Run it directly:

    python src/harness/test/test_tasks_unsolved.py

The default output is ``data/test/tasks_unresolved.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


HARNESS_ROOT = Path(__file__).resolve().parents[3]
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from src.data.build_index import iter_json_array  # noqa: E402
from src.harness.tasks import TaskStore  # noqa: E402


DEFAULT_PRODUCTS_PATH = (
    HARNESS_ROOT / "data" / "fine_items_eval_train_all.json"
)
DEFAULT_OUTPUT_PATH = (
    HARNESS_ROOT / "data" / "test" / "tasks_unresolved.json"
)


def _catalog_options(product: dict[str, Any]) -> dict[str, list[str]]:
    """Return canonical option values grouped by their catalog axis."""
    raw_options = product.get("customization_options")
    if not isinstance(raw_options, dict):
        return {}

    result: dict[str, list[str]] = {}
    for raw_axis, raw_values in raw_options.items():
        axis = str(raw_axis or "").strip()
        if not axis or not isinstance(raw_values, list):
            continue

        values: list[str] = []
        for raw_option in raw_values:
            if not isinstance(raw_option, dict):
                continue
            value = str(raw_option.get("value") or "").strip()
            if value and value not in values:
                values.append(value)
        if values:
            result[axis] = values
    return result


def collect_unresolved_tasks(products_path: str | Path) -> dict[str, Any]:
    """Collect unresolved tasks with their source catalog option values."""
    source_path = Path(products_path).resolve()
    store = TaskStore(source_path)

    unresolved = []
    needed_rows: set[int] = set()
    for split in ("train", "eval"):
        for task_id in store.list_task_ids(split):
            facts = store.get_private_facts(task_id)
            if not facts.unresolved_option_values:
                continue
            public_task = store.get_public_task(task_id)
            needed_rows.add(facts.source_row_number)
            unresolved.append(
                {
                    "task_id": task_id,
                    "split": facts.split,
                    "query": public_task.query,
                    "target_asin": facts.target_asin,
                    "target_title": facts.target_title,
                    "target_category": facts.target_category,
                    "required_option_values": facts.required_option_values,
                    "resolved_options": facts.required_options,
                    "unresolved_option_values": (
                        facts.unresolved_option_values
                    ),
                    "source_row_number": facts.source_row_number,
                    "source_instruction_index": (
                        facts.source_instruction_index
                    ),
                }
            )

    products_by_row: dict[int, dict[str, Any]] = {}
    for row_number, product in enumerate(
        iter_json_array(source_path),
        start=1,
    ):
        if row_number in needed_rows:
            products_by_row[row_number] = product
        if len(products_by_row) == len(needed_rows):
            break

    for task in unresolved:
        source_row_number = task["source_row_number"]
        product = products_by_row.get(source_row_number)
        if product is None:
            raise RuntimeError(
                f"source product row disappeared: {source_row_number}"
            )
        task["catalog_options"] = _catalog_options(product)

    unresolved.sort(
        key=lambda task: (
            task["split"],
            task["source_row_number"],
            task["source_instruction_index"],
            task["task_id"],
        )
    )
    unresolved_value_count = sum(
        len(task["unresolved_option_values"])
        for task in unresolved
    )

    return {
        "format_version": "tasks-unresolved-v1",
        "source": {
            "path": str(source_path),
            "product_data_sha256": store.product_data_sha256,
            "task_store_version": store.manifest["task_store_version"],
            "option_match_version": store.manifest[
                "option_match_version"
            ],
        },
        "summary": {
            "task_count": len(unresolved),
            "unresolved_option_value_count": unresolved_value_count,
            "split_counts": {
                split: sum(
                    task["split"] == split for task in unresolved
                )
                for split in ("train", "eval")
            },
        },
        "tasks": unresolved,
    }


def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write the report atomically and return its resolved path."""
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export tasks with unresolved annotated options",
    )
    parser.add_argument(
        "--products",
        type=Path,
        default=DEFAULT_PRODUCTS_PATH,
        help="decompressed source product JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSON report output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = collect_unresolved_tasks(args.products)
    output = write_report(report, args.output)
    summary = report["summary"]
    print(f"output: {output}")
    print(f"unresolved tasks: {summary['task_count']}")
    print(
        "unresolved option values: "
        f"{summary['unresolved_option_value_count']}"
    )
    print(f"split counts: {summary['split_counts']}")


if __name__ == "__main__":
    main()
