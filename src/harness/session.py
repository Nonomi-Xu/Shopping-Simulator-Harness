"""Single-task runtime state machine for the Shopping Simulator Harness.

The session is the actor-facing boundary that joins public tasks, the public
catalog, and terminal deterministic evaluation.  It never exposes private task
facts before submission and records only observations that were returned to the
actor.
"""

from __future__ import annotations

from collections.abc import Mapping
from threading import RLock
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from .catalog import CatalogError, ProductCatalog
from .evaluator import EvaluatorError, evaluate_submission
from .schemas import (
    EvaluationResult,
    GetProductAction,
    PrivateTaskFacts,
    ProductDetail,
    ProductSummary,
    PublicTask,
    SearchProductsAction,
    SearchPage,
    SearchFilters,
    SessionState,
    SessionStatus,
    SessionStep,
    SessionStepStatus,
    SessionTerminalReason,
    SubmitRecommendationsAction,
)
from .tasks import TaskStore, TaskStoreError


SESSION_VERSION = "shopping-session-v2"
DEFAULT_MAX_STEPS = 20

SEARCH_PRODUCTS = "search_products"
GET_PRODUCT = "get_product"
SUBMIT_RECOMMENDATIONS = "submit_recommendations"

_ACTION_MODELS: dict[str, type[BaseModel]] = {
    SEARCH_PRODUCTS: SearchProductsAction,
    GET_PRODUCT: GetProductAction,
    SUBMIT_RECOMMENDATIONS: SubmitRecommendationsAction,
}


class _TaskStore(Protocol):
    def get_public_task(self, task_id: str) -> PublicTask: ...

    def get_private_facts(self, task_id: str) -> PrivateTaskFacts: ...


class _Catalog(Protocol):
    def search(self, query: str, *, limit: int = 20) -> list[ProductSummary]: ...

    def get_product(self, asin: str) -> ProductDetail: ...


class SessionError(RuntimeError):
    """Base runtime exception with a stable machine-readable error code."""

    code = "session_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)


class SessionTerminatedError(SessionError):
    """Raised when a caller tries to mutate a sealed session."""

    code = "session_terminated"


class ShoppingSession:
    """Run one shopping recommendation task through a bounded tool session.

    Recoverable actor mistakes are returned as rejected ``SessionStep`` values
    and consume one step.  Catalog, evaluator, or task-store failures are
    treated as infrastructure errors and terminate the session.  Once a session
    is terminated, its state and trajectory are sealed.
    """

    def __init__(
        self,
        *,
        task_id: str,
        task_store: TaskStore | _TaskStore,
        catalog: ProductCatalog | _Catalog,
        max_steps: int = DEFAULT_MAX_STEPS,
        session_id: str | None = None,
        evaluate: bool = True,
    ) -> None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise TypeError("max_steps must be an integer")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")

        public_task = task_store.get_public_task(task_id)
        if not isinstance(public_task, PublicTask):
            public_task = PublicTask.model_validate(public_task)

        resolved_session_id = (
            session_id
            if session_id is not None
            else f"session_{public_task.task_id[5:]}"
        )
        if not isinstance(resolved_session_id, str):
            raise TypeError("session_id must be a string")
        resolved_session_id = resolved_session_id.strip()
        if not resolved_session_id:
            raise ValueError("session_id must not be empty")

        self._task_store = task_store
        self._evaluate = evaluate
        self._catalog = catalog
        self._public_task = public_task.model_copy(deep=True)
        self._session_id = resolved_session_id
        self._max_steps = max_steps
        self._step_count = 0
        self._status: SessionStatus = "active"
        self._terminal_reason: SessionTerminalReason | None = None
        self._surfaced_asins: list[str] = []
        self._surfaced_asin_set: set[str] = set()
        self._inspected_asins: list[str] = []
        self._inspected_asin_set: set[str] = set()
        self._history: list[SessionStep] = []
        self._evaluation: EvaluationResult | None = None
        self._lock = RLock()

    @property
    def public_task(self) -> PublicTask:
        """Return an actor-safe copy of the task query."""
        return self._public_task.model_copy(deep=True)

    @property
    def state(self) -> SessionState:
        """Return an immutable snapshot of public lifecycle state."""
        with self._lock:
            return SessionState(
                session_version=SESSION_VERSION,
                session_id=self._session_id,
                task_id=self._public_task.task_id,
                status=self._status,
                step_count=self._step_count,
                max_steps=self._max_steps,
                surfaced_asins=list(self._surfaced_asins),
                inspected_asins=list(self._inspected_asins),
                terminal_reason=self._terminal_reason,
            )

    @property
    def history(self) -> list[SessionStep]:
        """Return deep copies of all actor-visible tool attempts."""
        with self._lock:
            return [step.model_copy(deep=True) for step in self._history]

    @property
    def evaluation(self) -> EvaluationResult | None:
        """Return the terminal evaluation after submission, if available."""
        with self._lock:
            if self._evaluation is None:
                return None
            return self._evaluation.model_copy(deep=True)

    def dispatch(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | object,
    ) -> SessionStep:
        """Validate and execute one standard Harness tool call."""
        with self._lock:
            self._ensure_active()

            if not isinstance(tool_name, str) or not tool_name.strip():
                return self._record(
                    tool_name="invalid_tool",
                    arguments={},
                    status="rejected",
                    error_code="invalid_tool_name",
                    message="tool_name must be a non-empty string",
                )
            normalized_name = tool_name.strip()
            if not isinstance(arguments, Mapping):
                return self._record(
                    tool_name=normalized_name,
                    arguments={},
                    status="rejected",
                    error_code="invalid_arguments",
                    message="tool arguments must be an object",
                )

            public_arguments = dict(arguments)
            action_model = _ACTION_MODELS.get(normalized_name)
            if action_model is None:
                return self._record(
                    tool_name=normalized_name,
                    arguments=public_arguments,
                    status="rejected",
                    error_code="unknown_action",
                    message=f"unknown tool: {normalized_name}",
                )

            try:
                action = action_model.model_validate(public_arguments)
            except ValidationError:
                return self._record(
                    tool_name=normalized_name,
                    arguments=public_arguments,
                    status="rejected",
                    error_code="invalid_arguments",
                    message="tool arguments failed schema validation",
                )
            return self._execute(normalized_name, action)

    def search_products(self, action: SearchProductsAction) -> SessionStep:
        """Execute a validated search action and record its summaries."""
        if not isinstance(action, SearchProductsAction):
            raise TypeError("action must be SearchProductsAction")
        with self._lock:
            self._ensure_active()
            return self._execute(SEARCH_PRODUCTS, action)

    def get_product(self, action: GetProductAction) -> SessionStep:
        """Fetch a surfaced product detail and record the public observation."""
        if not isinstance(action, GetProductAction):
            raise TypeError("action must be GetProductAction")
        with self._lock:
            self._ensure_active()
            return self._execute(GET_PRODUCT, action)

    def submit_recommendations(
        self,
        action: SubmitRecommendationsAction,
    ) -> SessionStep:
        """Evaluate a terminal submission after all products were inspected."""
        if not isinstance(action, SubmitRecommendationsAction):
            raise TypeError("action must be SubmitRecommendationsAction")
        with self._lock:
            self._ensure_active()
            return self._execute(SUBMIT_RECOMMENDATIONS, action)

    def _execute(self, tool_name: str, action: BaseModel) -> SessionStep:
        arguments = action.model_dump(mode="json")
        try:
            if tool_name == SEARCH_PRODUCTS:
                return self._execute_search(action, arguments)
            if tool_name == GET_PRODUCT:
                return self._execute_get_product(action, arguments)
            if tool_name == SUBMIT_RECOMMENDATIONS:
                return self._execute_submission(action, arguments)
            raise AssertionError(f"unmapped action: {tool_name}")
        except (CatalogError, EvaluatorError, TaskStoreError) as exc:
            return self._record(
                tool_name=tool_name,
                arguments=arguments,
                status="error",
                error_code="infrastructure_error",
                message="catalog, task store or evaluator infrastructure failure",
                terminal_reason="infrastructure_error",
            )

    def _execute_search(
        self,
        action: BaseModel,
        arguments: dict[str, Any],
    ) -> SessionStep:
        search_action = SearchProductsAction.model_validate(action)
        page = None
        if callable(getattr(self._catalog, "search_page", None)):
            results, page = self._catalog.search_page(search_action)
        else:
            if (
                search_action.offset
                or search_action.exclude_asins
                or search_action.filters != SearchFilters()
            ):
                return self._record(
                    tool_name=SEARCH_PRODUCTS,
                    arguments=arguments,
                    status="rejected",
                    error_code="unsupported_search_options",
                    message="catalog adapter does not support filtered/paged search",
                )
            results = self._catalog.search(
                search_action.query, limit=search_action.limit
            )
        for result in results:
            if result.asin not in self._surfaced_asin_set:
                self._surfaced_asin_set.add(result.asin)
                self._surfaced_asins.append(result.asin)
        observation = [result.model_dump(mode="json") for result in results]
        return self._record(
            tool_name=SEARCH_PRODUCTS,
            arguments=arguments,
            status="success",
            observation=observation,
            search_page=page,
            source=self._source_metadata(),
        )

    def _execute_get_product(
        self,
        action: BaseModel,
        arguments: dict[str, Any],
    ) -> SessionStep:
        product_action = GetProductAction.model_validate(action)
        if product_action.asin not in self._surfaced_asin_set:
            return self._record(
                tool_name=GET_PRODUCT,
                arguments=arguments,
                status="rejected",
                error_code="product_not_surfaced",
                message="product must appear in a previous search result",
            )

        product = self._catalog.get_product(product_action.asin)
        if product.asin not in self._inspected_asin_set:
            self._inspected_asin_set.add(product.asin)
            self._inspected_asins.append(product.asin)
        data = product.model_dump(mode="json", exclude_unset=not self._evaluate)
        if product_action.fields is not None:
            selected = {"asin", "title", "category", *product_action.fields}
            data = {key: value for key, value in data.items() if key in selected}
        return self._record(
            tool_name=GET_PRODUCT,
            arguments=arguments,
            status="success",
            observation=data,
            source=self._source_metadata(),
        )

    def _execute_submission(
        self,
        action: BaseModel,
        arguments: dict[str, Any],
    ) -> SessionStep:
        submission = SubmitRecommendationsAction.model_validate(action)
        uninspected = list(
            dict.fromkeys(
                item.asin
                for item in submission.recommendations
                if item.asin not in self._inspected_asin_set
            )
        )
        if uninspected:
            return self._record(
                tool_name=SUBMIT_RECOMMENDATIONS,
                arguments=arguments,
                status="rejected",
                error_code="product_not_inspected",
                message=(
                    "every recommended product must be inspected first: "
                    + ", ".join(uninspected)
                ),
            )

        if self._evaluate:
            facts = self._task_store.get_private_facts(self._public_task.task_id)
            self._evaluation = evaluate_submission(
                facts=facts,
                submission=submission,
                catalog=self._catalog,  # type: ignore[arg-type]
            )
        return self._record(
            tool_name=SUBMIT_RECOMMENDATIONS,
            arguments=arguments,
            status="success",
            observation={"submitted": True},
            terminal_reason="submitted",
        )

    def _record(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        status: SessionStepStatus,
        observation: dict[str, Any] | list[dict[str, Any]] | None = None,
        error_code: str | None = None,
        message: str | None = None,
        terminal_reason: SessionTerminalReason | None = None,
        search_page: SearchPage | None = None,
        source: dict[str, Any] | None = None,
    ) -> SessionStep:
        self._step_count += 1
        if terminal_reason is not None:
            self._status = "terminated"
            self._terminal_reason = terminal_reason
        elif self._step_count >= self._max_steps:
            self._status = "terminated"
            self._terminal_reason = "max_steps"

        step = SessionStep(
            step_id=f"step_{self._step_count:04d}",
            step_number=self._step_count,
            tool_name=tool_name,
            arguments=arguments,
            status=status,
            observation=observation,
            search_page=search_page,
            source=source,
            error_code=error_code,
            message=message,
            terminated=self._status == "terminated",
            terminal_reason=self._terminal_reason,
        )
        self._history.append(step)
        return step.model_copy(deep=True)

    def _source_metadata(self):
        metadata = getattr(self._catalog, "last_observation_source", None)
        return dict(metadata) if metadata else None

    def _ensure_active(self) -> None:
        if self._status == "terminated":
            reason = self._terminal_reason or "unknown"
            raise SessionTerminatedError(f"session is terminated: {reason}")
