"""Authoritative model tool surface for the shopping Agent."""

from __future__ import annotations

from ..providers import ToolDefinition
from ..constraints import APPROXIMATE_MAX_SLACK, APPROXIMATE_TOLERANCE
from ..schemas import (
    GetProductAction,
    SearchProductsAction,
    SubmitRecommendationsAction,
)
from ..session import GET_PRODUCT, SEARCH_PRODUCTS, SUBMIT_RECOMMENDATIONS
from ..schemas import SessionState
from ..providers import ToolCall
from pydantic import BaseModel, ValidationError
from .contracts import (
    AgentSubmission,
    BudgetState,
    Contract,
    EvidenceAssessment,
    EvidenceBatch,
    PlanUpdate,
    ReadEvidence,
    TaskProfile,
    CompareProducts,
    VerifySelection,
    AskUser,
    ReviewProgress,
)


class ToolSpec(Contract):
    name: str
    version: str = "shopping-tools-v5"
    scope: str
    side_effect: str
    input_schema: dict
    output_schema: dict


class ToolRegistry:
    """One schema registry for domain tools and runtime planning/memory tools."""

    version = "shopping-tools-v5"

    def __init__(self, *, interactive: bool = False, multi_agent: bool = False):
        self.interactive = interactive
        self.multi_agent = multi_agent
        self.models: dict[str, type[BaseModel]] = {
            SEARCH_PRODUCTS: SearchProductsAction,
            GET_PRODUCT: GetProductAction,
            SUBMIT_RECOMMENDATIONS: AgentSubmission,
            "update_plan": PlanUpdate,
            "record_evidence": EvidenceAssessment,
            "record_evidence_batch": EvidenceBatch,
            "read_evidence": ReadEvidence,
            "compare_products": CompareProducts,
            "verify_selection": VerifySelection,
        }
        self.definitions = (
            *shopping_tool_definitions()[:2],
            ToolDefinition.from_action_model(
                name="compare_products",
                action_model=CompareProducts,
                description="Compare 2-5 inspected products by public fields and existing constraint assessments. Returns differences, unknown fields and original citations, not an authoritative ranking. Large fields are previews; use read_evidence for exact values.",
            ),
            ToolDefinition.from_action_model(
                name="verify_selection",
                action_model=VerifySelection,
                description="Check an inspected product's exact selected_options, availability and price using observed fields only. Optional max_price is in catalog price units; budget_constraint_id identifies the public requirement you interpreted. Unknown selected prices never pass the numeric check. This does not automatically assess semantic constraints.",
            ),
            ToolDefinition.from_action_model(
                name=SUBMIT_RECOMMENDATIONS,
                action_model=AgentSubmission,
                description=(
                    "Submit 1-5 inspected products after update_plan. Verified mode requires "
                    "every hard constraint satisfied by cited evidence. For best_effort, list "
                    "every unknown/violated hard constraint in gaps and copy each disclosure "
                    "into that product's reason. Public option validity is always checked."
                ),
            ),
            ToolDefinition.from_action_model(
                name="update_plan",
                action_model=PlanUpdate,
                description=(
                    "Create or revise a concise search plan. Quote the query for explicit "
                    "constraints; keep existing hard constraints unchanged. Background "
                    "preferences and inferences must be soft. Empty constraints require "
                    "no_constraints_reason. A valid update marks requirements parsed. "
                    "A numeric predicate must be grounded in its source_quote: use the "
                    "amount the quote states, or, for an approximate amount (左右/大概/"
                    f"约), a ceiling up to {APPROXIMATE_MAX_SLACK:g}x it (Environment v2 "
                    f"freezes {APPROXIMATE_TOLERANCE:g}x). Brand and option values must "
                    "appear literally; category and attribute values are your catalog "
                    "mapping of the quoted requirement."
                ),
            ),
            ToolDefinition.from_action_model(
                name="record_evidence_batch",
                action_model=EvidenceBatch,
                description=(
                    "Record 1-30 product/constraint assessments in one action using the same "
                    "citation rules as record_evidence. Each pair must be unique. The whole "
                    "batch is rejected without updates if any citation is invalid."
                ),
            ),
            ToolDefinition.from_action_model(
                name="read_evidence",
                action_model=ReadEvidence,
                description=(
                    "Retrieve already observed product details without another environment step. "
                    "Provide an inspected ASIN, optional original observation_id and field_paths "
                    "(JSON Pointers). Returns bounded scalar fragments and next_offset for paging. "
                    "Use source_observation_id and each field_path for evidence citations."
                ),
            ),
            ToolDefinition.from_action_model(
                name="record_evidence",
                action_model=EvidenceAssessment,
                description=(
                    "Record your assessment of one constraint for an inspected product. "
                    "Cite a detail observation ID, a JSON Pointer into its data and an "
                    "exact scalar-field quote. The assessment is model judgment, not "
                    "deterministic proof. This consumes an action."
                ),
            ),
        )
        if multi_agent:
            self.models["review_progress"] = ReviewProgress
            self.definitions += (
                ToolDefinition.from_action_model(
                    name="review_progress",
                    action_model=ReviewProgress,
                    description=(
                        "Observer handoff: summarize the observed result and remaining gaps. "
                        "Choose act for another executor action, replan for revised planning, "
                        "or recommend to approve inspected reviewed_asins for recommendation. "
                        "Approval does not bypass constraint evidence or selection checks."
                    ),
                ),
            )
            # Backward-compatible low-level action defaults remain available to
            # callers, while the multi-agent model must actually choose a method.
            definitions = []
            for definition in self.definitions:
                if definition.name == SEARCH_PRODUCTS:
                    definition = definition.model_copy(deep=True)
                    definition.parameters["required"] = list(dict.fromkeys(
                        [*definition.parameters.get("required", []), "method"]
                    ))
                definitions.append(definition)
            self.definitions = tuple(definitions)
        if interactive:
            self.models["ask_user"] = AskUser
            self.definitions += (
                ToolDefinition.from_action_model(
                    name="ask_user",
                    action_model=AskUser,
                    description=(
                        "Pause for the real user when an ambiguity could change the recommendation. "
                        "Explain the information_gap, ask one concise question, and optionally list "
                        "affected constraint_ids and suggested answers. Never invent a user reply. "
                        "The caller must explicitly authorize replacing an existing hard constraint."
                    ),
                ),
            )

    def specs(self) -> list[ToolSpec]:
        from .contracts import AgentObservation

        return [
            ToolSpec(
                name=d.name,
                scope="environment" if d.name in self.environment_names else "runtime",
                side_effect=(
                    "wait_for_user"
                    if d.name == "ask_user"
                    else (
                        "terminal"
                        if d.name == SUBMIT_RECOMMENDATIONS
                        else (
                            "session_memory"
                            if d.name
                            in {
                                "update_plan",
                                "record_evidence",
                                "record_evidence_batch",
                                "review_progress",
                            }
                            else "read"
                        )
                    )
                ),
                input_schema=d.parameters,
                output_schema=AgentObservation.model_json_schema(),
            )
            for d in self.definitions
        ]

    @property
    def environment_names(self) -> set[str]:
        return {SEARCH_PRODUCTS, GET_PRODUCT, SUBMIT_RECOMMENDATIONS}

    def allowed(
        self, state: SessionState, budget: BudgetState, task: TaskProfile | None = None
    ) -> list[str]:
        if state.status != "active" or budget.actions_remaining == 0:
            return []
        names = ["update_plan"]
        if self.multi_agent:
            names.append("review_progress")
        if self.interactive:
            names.append("ask_user")
        if budget.searches_remaining and budget.environment_steps_remaining:
            names.append(SEARCH_PRODUCTS)
        if state.surfaced_asins and budget.environment_steps_remaining:
            names.append(GET_PRODUCT)
        if state.inspected_asins:
            names.extend(
                [
                    "record_evidence",
                    "record_evidence_batch",
                    "read_evidence",
                    "verify_selection",
                ]
            )
            if len(state.inspected_asins) >= 2:
                names.append("compare_products")
            if budget.environment_steps_remaining and (
                task is None or task.requirements_status == "parsed"
            ):
                names.append(SUBMIT_RECOMMENDATIONS)
        return names

    def validate(
        self, call: ToolCall, state: SessionState, budget: BudgetState
    ) -> BaseModel:
        if call.name not in self.models:
            raise ValueError("unknown_action")
        if call.name not in self.allowed(state, budget):
            raise ValueError("tool_not_available")
        if self.multi_agent and call.name == SEARCH_PRODUCTS and "method" not in call.arguments:
            raise ValueError("search_method_required")
        try:
            return self.models[call.name].model_validate(call.arguments)
        except ValidationError as exc:
            raise ValueError("invalid_arguments") from exc


def shopping_tool_definitions() -> tuple[ToolDefinition, ...]:
    """Generate tools from the same Pydantic actions enforced by Harness."""
    return (
        ToolDefinition.from_action_model(
            name=SEARCH_PRODUCTS,
            description=(
                "Search the public product catalog with method=bm25 for lexical "
                "brand/model/specification matches. Only BM25 is supported by this contract. "
                "SQLite typed predicates are executed before Top-K; unknown facts do not pass. "
                "Returned ASINs become "
                "eligible for get_product in this session. Supports normalized category substring, "
                "exact brand, product price-range overlap filters, offset pagination and exclude_asins. "
                "Unknown prices are included by default; ranges never prove a selected variant is in budget. "
                "Keep filters and exclusions fixed while paging; use next_offset in the observation. "
                "Match snippets are discovery clues; inspect details before citing them."
            ),
            action_model=SearchProductsAction,
        ),
        ToolDefinition.from_action_model(
            name=GET_PRODUCT,
            description=(
                "Inspect the public detail of one ASIN previously returned "
                "by search_products. Optional fields selects public top-level fields; asin/title/category "
                "are always returned. Omit fields for full detail. Unrequested fields remain unobserved, "
                "not absent. Selection checks need customization_options and variant_combinations."
            ),
            action_model=GetProductAction,
        ),
        ToolDefinition.from_action_model(
            name=SUBMIT_RECOMMENDATIONS,
            description=(
                "Submit the final ranked recommendations. Every ASIN must "
                "have been inspected; an accepted call terminates the session."
            ),
            action_model=SubmitRecommendationsAction,
        ),
    )
