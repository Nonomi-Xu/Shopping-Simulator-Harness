"""Load public shopping tasks and keep their private answer facts isolated."""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
import math
from pathlib import Path
from typing import Any, Literal

from ..data.build_index import iter_json_array
from ..data.NFKC import normalize_query
from ..data.search_bm25 import sha256_file
from .schemas import (
    PrivateTaskFacts,
    PublicTask,
)

TaskSplit = Literal["train", "eval"]
TASK_STORE_VERSION = "shopping-tasks-v1"
TASK_ID_VERSION = "opaque-row-query-v1"
OPTION_MATCH_VERSION = "nfkc-casefold-alnum-v1"
VALID_SPLITS = ("train", "eval")
OptionCandidate = tuple[str, str]
OptionLookup = dict[tuple[str, str], set[OptionCandidate]]


def _normalize_option_key(value: str) -> str:
    """Return a comparison-only option key based on the shared NFKC rules."""
    normalized = normalize_query(value)
    return "".join(character for character in normalized if character.isalnum())


class TaskStoreError(RuntimeError):
    """The task dataset is absent, invalid, or internally inconsistent."""


class TaskDataError(TaskStoreError):
    """A source product or instruction cannot form a valid task."""


class TaskNotFoundError(TaskStoreError, LookupError):
    """The requested opaque task ID does not exist."""


class TaskStore:
    """In-memory task registry with a strict public/private boundary.

    The source JSON is streamed once during initialization.  Public actor tasks
    and private evaluator facts are stored separately and can only be retrieved
    through different methods.  Task IDs are deterministic opaque hashes that
    do not contain the gold product ASIN.
    """

    def __init__(self, products_path: str | Path) -> None:
        self.products_path = Path(products_path).resolve()
        if not self.products_path.is_file():
            raise FileNotFoundError(
                f"task product dataset not found: {self.products_path}"
            )

        self._public_tasks: dict[str, PublicTask] = {}
        self._private_facts: dict[str, PrivateTaskFacts] = {}
        self._task_ids_by_split: dict[str, list[str]] = {
            split: [] for split in VALID_SPLITS
        }
        # Count of empty attributes ignored during task loading
        self._ignored_empty_attribute_count = 0
        # Count of unresolved option values across all tasks
        self._unresolved_option_value_count = 0
        # Count of tasks with at least one unresolved option value
        self._tasks_with_unresolved_options = 0
        self._load()
        self.product_data_sha256 = sha256_file(self.products_path)

    def __len__(self) -> int:
        return len(self._public_tasks)

    @property
    def manifest(self) -> dict[str, Any]:
        """Return auditable task-store metadata without private task facts."""
        return {
            "task_store_version": TASK_STORE_VERSION,
            "task_id_version": TASK_ID_VERSION,
            "option_match_version": OPTION_MATCH_VERSION,
            "product_data_sha256": self.product_data_sha256,
            "task_count": len(self),
            "split_counts": {
                split: len(self._task_ids_by_split[split])
                for split in VALID_SPLITS
            },
            "data_quality": {
                "ignored_empty_attribute_count": (
                    self._ignored_empty_attribute_count
                ),
                "unresolved_option_value_count": (
                    self._unresolved_option_value_count
                ),
                "tasks_with_unresolved_options": (
                    self._tasks_with_unresolved_options
                ),
            },
        }

    def list_task_ids(self, split: TaskSplit) -> list[str]:
        """Return task IDs for a split in frozen source order."""
        normalized_split = self._validated_split(split)
        return list(self._task_ids_by_split[normalized_split])

    def iter_public_tasks(self, split: TaskSplit) -> Iterator[PublicTask]:
        """Iterate over safe actor tasks in frozen source order."""
        for task_id in self.list_task_ids(split):
            yield self.get_public_task(task_id)

    def sample_task_ids(
        self,
        split: TaskSplit,
        count: int,
        *,
        seed: int | str,
    ) -> list[str]:
        """Select a deterministic sample without mutating task order."""
        normalized_split = self._validated_split(split)
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("count must be an integer")
        if isinstance(seed, bool) or not isinstance(seed, (int, str)):
            raise TypeError("seed must be an integer or string")
        available = self._task_ids_by_split[normalized_split]
        if not 0 <= count <= len(available):
            raise ValueError(
                f"count must be between 0 and {len(available)}"
            )
        seed_text = f"{type(seed).__name__}:{seed}"
        ranked = sorted(
            available,
            key=lambda task_id: (
                hashlib.sha256(
                    f"{seed_text}\0{task_id}".encode("utf-8")
                ).digest(),
                task_id,
            ),
        )
        return ranked[:count]

    def get_public_task(self, task_id: str) -> PublicTask:
        """Return only the query payload that is safe for an actor."""
        normalized_task_id = self._validated_task_id(task_id)
        try:
            task = self._public_tasks[normalized_task_id]
        except KeyError as exc:
            raise TaskNotFoundError(
                f"task not found: {normalized_task_id}"
            ) from exc
        return task.model_copy(deep=True)

    def get_private_facts(self, task_id: str) -> PrivateTaskFacts:
        """Return gold facts for the evaluator; never expose this to an actor."""
        normalized_task_id = self._validated_task_id(task_id)
        try:
            facts = self._private_facts[normalized_task_id]
        except KeyError as exc:
            raise TaskNotFoundError(
                f"task not found: {normalized_task_id}"
            ) from exc
        return facts.model_copy(deep=True)

    def _load(self) -> None:
        seen_asins: set[str] = set()
        for row_number, product in enumerate(
            iter_json_array(self.products_path),
            start=1,
        ):
            context = f"source product row {row_number}"
            target_asin = self._required_text(product.get("asin"), "asin", context)
            if target_asin in seen_asins:
                raise TaskDataError(f"duplicate product ASIN: {target_asin}")
            seen_asins.add(target_asin)

            split = self._validated_source_split(product.get("tag"), context)
            instructions = product.get("instructions")
            if not isinstance(instructions, list) or not instructions:
                raise TaskDataError(f"{context} has no instructions")

            option_lookup = self._build_option_lookup(product)
            for instruction_index, instruction in enumerate(instructions):
                self._add_task(
                    product=product,
                    instruction=instruction,
                    row_number=row_number,
                    instruction_index=instruction_index,
                    split=split,
                    target_asin=target_asin,
                    option_lookup=option_lookup,
                )

        if not self._public_tasks:
            raise TaskDataError("task dataset contains no tasks")

    def _add_task(
        self,
        *,
        product: dict[str, Any],
        instruction: object,
        row_number: int,
        instruction_index: int,
        split: TaskSplit,
        target_asin: str,
        option_lookup: OptionLookup,
    ) -> None:
        context = (
            f"instruction {instruction_index} for product {target_asin}"
        )
        if not isinstance(instruction, dict):
            raise TaskDataError(f"{context} must be an object")

        query = self._required_text(
            instruction.get("instruction"),
            "instruction",
            context,
        )
        instruction_asin = self._required_text(
            instruction.get("asin"),
            "instruction.asin",
            context,
        )
        if instruction_asin != target_asin:
            raise TaskDataError(
                f"{context} targets {instruction_asin}, expected {target_asin}"
            )

        option_values = self._string_list(
            instruction.get("instruction_options"),
            "instruction_options",
            context,
        )
        attributes = self._string_list(
            instruction.get("attributes"),
            "attributes",
            context,
            ignore_empty=True,
        )
        raw_attributes = instruction.get("attributes")
        self._ignored_empty_attribute_count += sum(
            not str(value or "").strip()
            for value in raw_attributes
        )
        required_options, unresolved_options = self._resolve_options(
            option_values,
            option_lookup,
        )
        self._unresolved_option_value_count += len(unresolved_options)
        if unresolved_options:
            self._tasks_with_unresolved_options += 1
        task_id = self._stable_task_id(
            row_number=row_number,
            instruction_index=instruction_index,
            query=query,
        )
        if task_id in self._public_tasks:
            raise TaskDataError(f"duplicate generated task ID: {task_id}")

        public_task = PublicTask(task_id=task_id, query=query)
        private_facts = PrivateTaskFacts(
            task_id=task_id,
            split=split,
            target_asin=target_asin,
            target_title=str(product.get("title") or ""),
            target_brand=str(product.get("shop_name") or "") or None,
            target_category=str(product.get("category") or ""),
            required_options=required_options,
            required_option_values=option_values,
            unresolved_option_values=unresolved_options,
            required_attributes=attributes,
            target_prices=self._prices(product.get("pricing")),
            source_row_number=row_number,
            source_instruction_index=instruction_index,
        )
        self._public_tasks[task_id] = public_task
        self._private_facts[task_id] = private_facts
        self._task_ids_by_split[split].append(task_id)

    @staticmethod
    def _stable_task_id(
        *,
        row_number: int,
        instruction_index: int,
        query: str,
    ) -> str:
        source = (
            f"{TASK_ID_VERSION}\0{row_number}\0{instruction_index}\0{query}"
        ).encode("utf-8")
        return f"task_{hashlib.sha256(source).hexdigest()[:24]}"

    @staticmethod
    def _build_option_lookup(product: dict[str, Any]) -> OptionLookup:
        """Index exact and normalized values to canonical catalog options."""
        raw_options = product.get("customization_options")
        if raw_options is None:
            return {}
        if not isinstance(raw_options, dict):
            raise TaskDataError("customization_options must be an object")

        lookup: OptionLookup = {}
        for raw_axis, raw_values in raw_options.items():
            axis = str(raw_axis or "").strip()
            if not axis or not isinstance(raw_values, list):
                continue
            for raw_option in raw_values:
                if not isinstance(raw_option, dict):
                    continue
                canonical_value = str(
                    raw_option.get("value") or ""
                ).strip()
                if not canonical_value:
                    continue

                candidate = (axis, canonical_value)
                lookup.setdefault(
                    ("exact", canonical_value),
                    set(),
                ).add(candidate)

                normalized_key = _normalize_option_key(canonical_value)
                if normalized_key:
                    lookup.setdefault(
                        ("normalized", normalized_key),
                        set(),
                    ).add(candidate)
        return lookup

    @staticmethod
    def _resolve_options(
        option_values: list[str],
        option_lookup: OptionLookup,
    ) -> tuple[dict[str, str], list[str]]:
        """Resolve annotations to unique canonical catalog axis/value pairs."""
        resolved: dict[str, str] = {}
        unresolved: list[str] = []
        for annotated_value in option_values:
            candidates = option_lookup.get(
                ("exact", annotated_value),
                set(),
            )
            if not candidates:
                normalized_key = _normalize_option_key(annotated_value)
                if normalized_key:
                    candidates = option_lookup.get(
                        ("normalized", normalized_key),
                        set(),
                    )

            if len(candidates) != 1:
                unresolved.append(annotated_value)
                continue

            axis, canonical_value = next(iter(candidates))
            existing = resolved.get(axis)
            if existing is not None and existing != canonical_value:
                unresolved.append(annotated_value)
                continue
            resolved[axis] = canonical_value
        return resolved, unresolved

    @staticmethod
    def _prices(raw_prices: object) -> list[float]:
        if not isinstance(raw_prices, list):
            return []
        prices = []
        for raw_price in raw_prices:
            if isinstance(raw_price, bool):
                continue
            try:
                price = float(raw_price)
            except (TypeError, ValueError):
                continue
            if math.isfinite(price) and price >= 0:
                prices.append(price)
        return prices

    @staticmethod
    def _string_list(
        value: object,
        field_name: str,
        context: str,
        *,
        ignore_empty: bool = False,
    ) -> list[str]:
        if not isinstance(value, list):
            raise TaskDataError(f"{context}.{field_name} must be a list")
        result = []
        for index, item in enumerate(value):
            text = str(item or "").strip()
            if not text:
                if ignore_empty:
                    continue
                raise TaskDataError(
                    f"{context}.{field_name}[{index}] must not be empty"
                )
            if text not in result:
                result.append(text)
        return result

    @staticmethod
    def _required_text(
        value: object,
        field_name: str,
        context: str,
    ) -> str:
        text = str(value or "").strip()
        if not text:
            raise TaskDataError(f"{context}.{field_name} must not be empty")
        return text

    @staticmethod
    def _validated_source_split(value: object, context: str) -> TaskSplit:
        if value not in VALID_SPLITS:
            raise TaskDataError(f"{context} has invalid split: {value!r}")
        return value  # type: ignore[return-value]

    @staticmethod
    def _validated_split(value: object) -> TaskSplit:
        if value not in VALID_SPLITS:
            raise ValueError(
                f"split must be one of: {', '.join(VALID_SPLITS)}"
            )
        return value  # type: ignore[return-value]

    @staticmethod
    def _validated_task_id(task_id: str) -> str:
        if not isinstance(task_id, str):
            raise TypeError("task_id must be a string")
        normalized_task_id = task_id.strip()
        if not normalized_task_id:
            raise ValueError("task_id must not be empty")
        return normalized_task_id
