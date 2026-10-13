"""Public shopping queries and evaluator-only target facts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from .catalog import normalize_query
from .types import PrivateTaskFacts, PublicTask, TaskSplit


TASK_ID_VERSION = "opaque-row-query-v1"
VALID_SPLITS = ("all", "test")
TEST_TASK_COUNT = 10
TEST_TASK_SEED = 42


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    with path.open(encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            eof = not chunk
            buffer += chunk
            position = 0
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError("product file must contain a JSON array")
                    started = True
                    position += 1
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    return
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                if not isinstance(value, dict):
                    raise ValueError("every product entry must be an object")
                yield value
                position = end
            buffer = buffer[position:]
            if eof:
                raise ValueError("product JSON array has no closing bracket")


def _option_key(value: str) -> str:
    return "".join(character for character in normalize_query(value) if character.isalnum())


def _task_id(row_number: int, instruction_index: int, query: str) -> str:
    source = (
        f"{TASK_ID_VERSION}\0{row_number}\0{instruction_index}\0{query}"
    ).encode("utf-8")
    return f"task_{hashlib.sha256(source).hexdigest()[:24]}"


def _resolve_options(
    product: dict[str, Any], values: list[str]
) -> tuple[dict[str, str], list[str]]:
    exact: dict[str, set[tuple[str, str]]] = {}
    normalized: dict[str, set[tuple[str, str]]] = {}
    raw_options = product.get("customization_options")
    if not isinstance(raw_options, dict):
        raw_options = {}
    for raw_axis, raw_values in raw_options.items():
        axis = str(raw_axis or "").strip()
        if not axis or not isinstance(raw_values, list):
            continue
        for raw_option in raw_values:
            if not isinstance(raw_option, dict):
                continue
            value = str(raw_option.get("value") or "").strip()
            if not value:
                continue
            candidate = (axis, value)
            exact.setdefault(value, set()).add(candidate)
            key = _option_key(value)
            if key:
                normalized.setdefault(key, set()).add(candidate)

    resolved: dict[str, str] = {}
    unresolved: list[str] = []
    for value in values:
        matches = exact.get(value, set()) or normalized.get(_option_key(value), set())
        if len(matches) != 1:
            unresolved.append(value)
            continue
        axis, canonical_value = next(iter(matches))
        if axis in resolved and resolved[axis] != canonical_value:
            unresolved.append(value)
            continue
        resolved[axis] = canonical_value
    return resolved, unresolved


class TaskStore:
    """Load task queries while keeping gold answers out of the selector payload."""

    def __init__(self, products_path: str | Path) -> None:
        self.products_path = Path(products_path).expanduser().resolve()
        if not self.products_path.is_file():
            raise FileNotFoundError(f"product dataset not found: {self.products_path}")
        self._public: dict[str, PublicTask] = {}
        self._private: dict[str, PrivateTaskFacts] = {}
        self._ids: dict[str, list[str]] = {split: [] for split in VALID_SPLITS}
        self._load()
        # Test is a reproducible smoke-test subset, also included in all.
        test_ids = set(self.sample_task_ids(
            "all", min(TEST_TASK_COUNT, len(self._ids["all"])), seed=TEST_TASK_SEED,
        ))
        self._ids["test"] = [
            task_id for task_id in self._ids["all"] if task_id in test_ids
        ]

    def _load(self) -> None:
        seen_asins: set[str] = set()
        for row_number, product in enumerate(iter_json_array(self.products_path), start=1):
            asin = str(product.get("asin") or "").strip()
            instructions = product.get("instructions")
            if not asin or not isinstance(instructions, list) or not instructions:
                raise ValueError(f"invalid source product at row {row_number}")
            if asin in seen_asins:
                raise ValueError(f"duplicate product ASIN at row {row_number}: {asin}")
            seen_asins.add(asin)
            for instruction_index, instruction in enumerate(instructions):
                if not isinstance(instruction, dict):
                    raise ValueError(f"invalid instruction at row {row_number}")
                query = str(instruction.get("instruction") or "").strip()
                instruction_asin = str(instruction.get("asin") or "").strip()
                if not query or instruction_asin != asin:
                    raise ValueError(f"invalid instruction target at row {row_number}")
                raw_values = instruction.get("instruction_options")
                if not isinstance(raw_values, list):
                    raise ValueError(f"invalid instruction options at row {row_number}")
                values = [str(value or "").strip() for value in raw_values]
                if any(not value for value in values):
                    raise ValueError(f"empty instruction option at row {row_number}")
                required, unresolved = _resolve_options(product, values)
                task_id = _task_id(row_number, instruction_index, query)
                task = PublicTask(task_id=task_id, query=query)
                facts = PrivateTaskFacts(
                    task_id=task_id,
                    split="all",
                    target_asin=asin,
                    required_options=required,
                    unresolved_option_values=unresolved,
                )
                if task_id in self._public:
                    raise ValueError(f"duplicate generated task ID: {task_id}")
                self._public[task_id] = task
                self._private[task_id] = facts
                self._ids["all"].append(task_id)

    def get_public_task(self, task_id: str) -> PublicTask:
        try:
            return self._public[task_id]
        except KeyError as exc:
            raise KeyError(f"task not found: {task_id}") from exc

    def get_private_facts(self, task_id: str) -> PrivateTaskFacts:
        try:
            return self._private[task_id]
        except KeyError as exc:
            raise KeyError(f"task not found: {task_id}") from exc

    def list_task_ids(self, split: TaskSplit = "all") -> list[str]:
        if split not in VALID_SPLITS:
            raise ValueError("split must be all or test")
        return list(self._ids[split])

    def sample_task_ids(self, split: TaskSplit, count: int, seed: int | str) -> list[str]:
        available = self.list_task_ids(split)
        if not 0 <= count <= len(available):
            raise ValueError(f"count must be between 0 and {len(available)}")
        seed_text = f"{type(seed).__name__}:{seed}"
        ranked = sorted(
            available,
            key=lambda task_id: (
                hashlib.sha256(f"{seed_text}\0{task_id}".encode()).digest(),
                task_id,
            ),
        )
        return ranked[:count]
