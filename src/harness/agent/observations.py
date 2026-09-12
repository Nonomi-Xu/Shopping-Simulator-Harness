"""Project environment responses into the public agent observation contract."""

from ..schemas import ProductDetail, ProductSummary, SessionState, SessionStep
from .contracts import (
    AgentAction,
    AgentObservation,
    BudgetState,
    ProjectionMetadata,
    TaskProfile,
)
from .tools import ToolRegistry


class ObservationProjector:
    version = "shopping-observation-v4"

    def project(
        self,
        action: AgentAction,
        step: SessionStep,
        state: SessionState,
        budget: BudgetState,
        registry: ToolRegistry,
        task: TaskProfile | None = None,
    ) -> AgentObservation:
        data = None
        returned = None
        if step.status == "success":
            if step.tool_name == "search_products":
                data = [
                    ProductSummary.model_validate(item).model_dump(mode="json")
                    for item in step.observation
                ]
                returned = len(data)
            elif step.tool_name == "get_product":
                data = ProductDetail.model_validate(step.observation).model_dump(
                    mode="json", exclude_unset=True
                )
            elif step.tool_name == "submit_recommendations":
                # No scoring field, even after termination, enters actor memory.
                data = {"submitted": True}
        return AgentObservation(
            observation_id=f"{action.action_id}:observation",
            action_id=action.action_id,
            tool_name=action.tool_name,
            status=step.status,
            data=data,
            error_code=step.error_code,
            message=step.message,
            environment_step_id=step.step_id,
            terminated=step.terminated,
            terminal_reason=step.terminal_reason,
            budget=budget,
            allowed_next_tools=registry.allowed(state, budget, task),
            projection=ProjectionMetadata(
                applied_filters=step.arguments.get("filters") if step.tool_name == "search_products" else None,
                source=step.source,
                total_results=(
                    step.search_page.total_results if step.search_page else None
                ),
                returned_items=returned,
                observed_fields=(
                    sorted(data) if step.tool_name == "get_product" and data else None
                ),
                next_offset=step.search_page.next_offset if step.search_page else None,
                has_more=step.search_page.has_more if step.search_page else None,
                price_scope=step.search_page.price_scope if step.search_page else None,
            ),
        )
