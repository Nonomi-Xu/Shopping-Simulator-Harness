"""Unlabelled shopping environment with restorable public sessions."""

from __future__ import annotations

import hashlib
from threading import RLock
from uuid import uuid4

from .harness import ShoppingHarness, DuplicateSessionError, HarnessError
from .schemas import PublicTask, SessionSnapshot
from .session import ShoppingSession
from .sources import ProductSource


class _PublicTaskStore:
    def __init__(self, task):
        self.task = task

    def get_public_task(self, task_id):
        if task_id != self.task.task_id:
            raise ValueError("unknown interactive task")
        return self.task

    def get_private_facts(self, task_id):
        raise AssertionError("interactive tasks have no private evaluation facts")


class InteractiveHarness(ShoppingHarness):
    """Reuse Session execution/lifecycle with a public source and no TaskStore.

    Owns the supplied source. Checkpoints restore observed state without issuing
    source requests. Consumed checkpoints cannot be reused within this instance.
    """

    mode = "interactive"

    def __init__(self, source: ProductSource, *, default_max_steps=20):
        self._validate_max_steps(default_max_steps)
        if not isinstance(source, ProductSource):
            raise TypeError("source must implement ProductSource")
        self._catalog = source
        self._default_max_steps = default_max_steps
        self._sessions = {}
        self._next_session_number = 1
        self._closed = False
        self._lock = RLock()
        self._consumed_checkpoints = set()

    @property
    def manifest(self):
        return {
            "harness_version": "shopping-interactive-v1",
            "mode": self.mode,
            "default_max_steps": self._default_max_steps,
            "source": dict(self._catalog.manifest),
            "evaluation": "unlabelled",
        }

    def create_session(self, *args, **kwargs):
        raise HarnessError(
            "interactive tasks start with create_interactive_session(query)"
        )

    def list_task_ids(self, *args, **kwargs):
        raise HarnessError("interactive sources have no benchmark splits")

    sample_task_ids = list_task_ids

    def get_public_task(self, task_id):
        with self._lock:
            for session in self._sessions.values():
                if session.public_task.task_id == task_id:
                    return session.public_task
        raise HarnessError("unknown interactive task")

    def create_interactive_session(self, query, *, max_steps=None, session_id=None):
        if not isinstance(query, str) or not query.strip() or len(query) > 16000:
            raise ValueError("query must contain 1-16000 characters")
        task = PublicTask(task_id="task_" + hashlib.sha256(query.encode()).hexdigest()[:24], query=query)
        return self.create_public_session(task, max_steps=max_steps, session_id=session_id)

    def create_public_session(self, task: PublicTask, *, max_steps=None, session_id=None):
        """Preserve benchmark IDs while receiving only an allowlisted public task."""
        task = PublicTask.model_validate(task)
        with self._lock:
            self._ensure_open()
            sid = self._validated_session_id(session_id or f"chat_{uuid4().hex}")
            steps = self._default_max_steps if max_steps is None else max_steps
            self._validate_max_steps(steps)
            if sid in self._sessions:
                raise DuplicateSessionError("interactive session ID already exists")
            session = ShoppingSession(
                task_id=task.task_id,
                task_store=_PublicTaskStore(task),
                catalog=self._catalog,
                max_steps=steps,
                session_id=sid,
                evaluate=False,
            )
            self._sessions[sid] = session
            return self._snapshot(session)

    def restore_interactive_session(
        self, snapshot: SessionSnapshot, *, checkpoint_hash: str
    ):
        """Called only after the runtime has validated the checkpoint event chain."""
        with self._lock:
            self._ensure_open()
            state = snapshot.state
            if state.status != "active" or snapshot.evaluation is not None:
                raise ValueError("only active unlabelled snapshots can be restored")
            if checkpoint_hash in self._consumed_checkpoints:
                raise ValueError("checkpoint has already been consumed")
            if state.session_id in self._sessions:
                if self._snapshot(self._sessions[state.session_id]) != snapshot:
                    raise ValueError("stale checkpoint: session has changed")
            else:
                self.create_public_session(
                    snapshot.public_task,
                    max_steps=state.max_steps,
                    session_id=state.session_id,
                )
                session = self._sessions[state.session_id]
                session._step_count = state.step_count
                session._surfaced_asins = list(state.surfaced_asins)
                session._surfaced_asin_set = set(state.surfaced_asins)
                session._inspected_asins = list(state.inspected_asins)
                session._inspected_asin_set = set(state.inspected_asins)
                session._history = [
                    step.model_copy(deep=True) for step in snapshot.history
                ]
            self._consumed_checkpoints.add(checkpoint_hash)
