"""Unified lifecycle manager for Shopping Simulator Harness sessions.

``ShoppingHarness`` owns the immutable task store and public catalog, creates
isolated single-task sessions, and exposes only actor-safe task and trajectory
views.  Model-specific rollout loops and provider clients live above this
layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any

from .catalog import ProductCatalog
from .schemas import (
    EvaluationResult,
    PublicTask,
    SessionSnapshot,
    SessionState,
    SessionStep,
    SessionStatus,
    TaskSplit,
)
from .session import DEFAULT_MAX_STEPS, ShoppingSession
from .tasks import TaskStore


HARNESS_VERSION = "shopping-harness-v2"


class HarnessError(RuntimeError):
    """Base exception with a stable machine-readable error code."""

    code = "harness_error"


class HarnessClosedError(HarnessError):
    """Raised when a caller tries to use a closed Harness runtime."""

    code = "harness_closed"


class SessionNotFoundError(HarnessError, LookupError):
    """Raised when a managed session ID does not exist."""

    code = "session_not_found"


class DuplicateSessionError(HarnessError):
    """Raised when an explicit session ID is already registered."""

    code = "duplicate_session"


class ActiveSessionReleaseError(HarnessError):
    """Raised when an active session would be discarded without force."""

    code = "active_session_release"


class ShoppingHarness:
    """Own shared data resources and manage isolated shopping sessions.

    The manager is the recommended application-facing API.  Private task facts
    and the raw catalog are deliberately not exposed as public properties.
    Every session shares one read-only catalog while retaining independent
    surfaced products, inspected products, history, and evaluation state.
    """

    def __init__(
        self,
        *,
        products_path: str | Path,
        index_path: str | Path,
        default_max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        self._validate_max_steps(default_max_steps)
        task_store = TaskStore(products_path)
        catalog = ProductCatalog(
            index_path,
            expected_product_sha256=task_store.product_data_sha256,
        )

        self._task_store = task_store
        self._catalog = catalog
        self._default_max_steps = default_max_steps
        self._sessions: dict[str, ShoppingSession] = {}
        self._next_session_number = 1
        self._closed = False
        self._lock = RLock()

    @property
    def manifest(self) -> dict[str, Any]:
        """Return auditable runtime metadata without private task facts."""
        with self._lock:
            self._ensure_open()
            return {
                "harness_version": HARNESS_VERSION,
                "default_max_steps": self._default_max_steps,
                "product_data_sha256": self._task_store.product_data_sha256,
                "task_store": deepcopy(self._task_store.manifest),
                "catalog": deepcopy(self._catalog.manifest),
            }

    def list_task_ids(self, split: TaskSplit = "all") -> list[str]:
        """Return frozen task IDs for a public dataset split."""
        with self._lock:
            self._ensure_open()
            return self._task_store.list_task_ids(split)

    def sample_task_ids(
        self,
        split: TaskSplit,
        count: int,
        *,
        seed: int | str,
    ) -> list[str]:
        """Return a deterministic task sample through the public boundary."""
        with self._lock:
            self._ensure_open()
            return self._task_store.sample_task_ids(
                split,
                count,
                seed=seed,
            )

    def get_public_task(self, task_id: str) -> PublicTask:
        """Return only the query payload safe for an actor."""
        with self._lock:
            self._ensure_open()
            return self._task_store.get_public_task(task_id)

    def create_session(
        self,
        task_id: str,
        *,
        max_steps: int | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot:
        """Create an isolated session and return its initial public snapshot."""
        with self._lock:
            self._ensure_open()
            resolved_max_steps = (
                self._default_max_steps
                if max_steps is None
                else max_steps
            )
            self._validate_max_steps(resolved_max_steps)
            resolved_session_id = self._resolve_session_id(session_id)
            if resolved_session_id in self._sessions:
                raise DuplicateSessionError(
                    f"session already exists: {resolved_session_id}"
                )

            session = ShoppingSession(
                task_id=task_id,
                task_store=self._task_store,
                catalog=self._catalog,
                max_steps=resolved_max_steps,
                session_id=resolved_session_id,
            )
            self._sessions[resolved_session_id] = session
            return self._snapshot(session)

    def dispatch(
        self,
        session_id: str,
        tool_name: str,
        arguments: Mapping[str, Any] | object,
    ) -> SessionStep:
        """Dispatch one tool call to a registered active session."""
        with self._lock:
            self._ensure_open()
            session = self._get_session(session_id)
            return session.dispatch(tool_name, arguments)

    def get_session(self, session_id: str) -> SessionSnapshot:
        """Return a deep, serializable snapshot without exposing internals."""
        with self._lock:
            session = self._get_session(session_id)
            return self._snapshot(session)

    def list_sessions(
        self,
        *,
        status: SessionStatus | None = None,
    ) -> list[SessionState]:
        """List managed session states in creation order."""
        with self._lock:
            if status not in (None, "active", "terminated"):
                raise ValueError("status must be active, terminated, or None")
            states = [session.state for session in self._sessions.values()]
            if status is not None:
                states = [state for state in states if state.status == status]
            return [state.model_copy(deep=True) for state in states]

    def get_evaluation(self, session_id: str) -> EvaluationResult | None:
        """Return a terminal evaluation, if the session submitted one."""
        with self._lock:
            session = self._get_session(session_id)
            return session.evaluation

    def release_session(
        self,
        session_id: str,
        *,
        force: bool = False,
    ) -> SessionSnapshot:
        """Remove a session, protecting active trajectories by default."""
        if not isinstance(force, bool):
            raise TypeError("force must be a boolean")
        with self._lock:
            session = self._get_session(session_id)
            snapshot = self._snapshot(session)
            if snapshot.state.status == "active" and not force:
                raise ActiveSessionReleaseError(
                    "cannot release an active session without force=True: "
                    f"{snapshot.state.session_id}"
                )
            del self._sessions[snapshot.state.session_id]
            return snapshot

    def close(self) -> None:
        """Close shared catalog resources; calling close twice is safe."""
        with self._lock:
            if self._closed:
                return
            self._catalog.close()
            self._closed = True

    def __enter__(self) -> ShoppingHarness:
        with self._lock:
            self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _get_session(self, session_id: str) -> ShoppingSession:
        normalized_session_id = self._validated_session_id(session_id)
        try:
            return self._sessions[normalized_session_id]
        except KeyError as exc:
            raise SessionNotFoundError(
                f"session not found: {normalized_session_id}"
            ) from exc

    def _resolve_session_id(self, session_id: str | None) -> str:
        if session_id is not None:
            return self._validated_session_id(session_id)

        while True:
            candidate = f"session_{self._next_session_number:08d}"
            self._next_session_number += 1
            if candidate not in self._sessions:
                return candidate

    @staticmethod
    def _snapshot(session: ShoppingSession) -> SessionSnapshot:
        return SessionSnapshot(
            public_task=session.public_task,
            state=session.state,
            history=session.history,
            evaluation=session.evaluation,
        )

    @staticmethod
    def _validated_session_id(session_id: str) -> str:
        if not isinstance(session_id, str):
            raise TypeError("session_id must be a string")
        normalized_session_id = session_id.strip()
        if not normalized_session_id:
            raise ValueError("session_id must not be empty")
        return normalized_session_id

    @staticmethod
    def _validate_max_steps(max_steps: int) -> None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise TypeError("max_steps must be an integer")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")

    def _ensure_open(self) -> None:
        if self._closed:
            raise HarnessClosedError("harness is closed")
