"""Cheap advisory actions driven by public information gaps, never hidden labels."""

from ..schemas import GetProductAction, SearchProductsAction
from .contracts import (
    ActionProposal,
    SearchPlan,
    TaskProfile,
    CompareProducts,
    VerifySelection,
    ReadEvidence,
)
from .events import digest
from .memory import MemoryStore, normalized_query, observed_product


def action_signature(name: str, arguments: dict) -> str:
    args = dict(arguments)
    if name == "search_products":
        args = SearchProductsAction.model_validate(args).model_dump(
            mode="json", exclude={"limit"}
        )
        args["query"] = normalized_query(args["query"])
        for key in ("brand", "category"):
            if args["filters"][key] is not None:
                args["filters"][key] = normalized_query(args["filters"][key])
        args["exclude_asins"] = sorted(set(args["exclude_asins"]))
    elif name == "get_product":
        args = GetProductAction.model_validate(args).model_dump(mode="json")
        if args["fields"] is not None:
            args["fields"] = sorted(set(args["fields"]))
    elif name in {"compare_products", "verify_selection", "read_evidence"}:
        schema = {
            "compare_products": CompareProducts,
            "verify_selection": VerifySelection,
            "read_evidence": ReadEvidence,
        }[name]
        args = schema.model_validate(args).model_dump(mode="json")
    return digest({"tool": name, "arguments": args})


def propose_actions(
    task: TaskProfile,
    plan: SearchPlan,
    memory: MemoryStore,
    allowed: list[str],
    limit: int,
    blocked: set[str] | None = None,
) -> list[dict]:
    if not limit:
        return []
    blocked = blocked or set()
    proposals = []

    def add(name, args, gap, asins=(), ids=(), refs=(), needs=()):
        if name not in allowed or len(proposals) >= limit:
            return
        if not needs and action_signature(name, args) in blocked:
            return
        proposals.append(
            ActionProposal(
                tool_name=name,
                arguments=args,
                information_gap=gap,
                target_asins=list(asins),
                constraint_ids=list(ids),
                evidence_observation_ids=list(refs),
                environment_steps=int(
                    name in {"search_products", "get_product", "submit_recommendations"}
                ),
                requires_model_input=list(needs),
            ).model_dump(mode="json")
        )

    if task.requirements_status != "parsed":
        add(
            "update_plan",
            {"queries": plan.queries},
            "Extract requirements from the original query before submission.",
            needs=("constraints or no_constraints_reason", "change_summary"),
        )
    view = memory.compact(task, 5)
    candidates = view["candidates"]
    hard = {c.constraint_id for c in task.constraints if c.priority == "hard"}
    for candidate in candidates:
        if candidate["evidence_ready"]:
            checks = candidate.get("facts", {}).get("selection_checks", [])
            valid = [
                check
                for check in checks
                if not check["stale"]
                and check["selection_status"] == "valid"
                and check["combination_status"] == "available"
                and check["budget_status"] in {"satisfied", "not_requested"}
            ]
            if valid:
                check = valid[-1]
                add(
                    "submit_recommendations",
                    {
                        "recommendations": [
                            {
                                "asin": candidate["asin"],
                                "selected_options": check["selected_options"],
                            }
                        ]
                    },
                    "Evidence supports a candidate; write grounded reasons and submit when further comparison is unnecessary.",
                    (candidate["asin"],),
                    refs=sorted(set(check["field_sources"].values())),
                    needs=("recommendations[].reason",),
                )
                break
    inspected = []
    for candidate in candidates:
        asin = candidate["asin"]
        if candidate["hard_conflicts"]:
            continue
        if not candidate["inspected"]:
            add(
                "get_product",
                {
                    "asin": asin,
                    "fields": [
                        "attributes",
                        "customization_options",
                        "variant_combinations",
                        "price_min",
                        "price_max",
                    ],
                },
                "Inspect a promising candidate before assessing constraints.",
                (asin,),
            )
            continue
        inspected.append(asin)
        detail, sources = observed_product(memory.state, asin)
        missing = [
            f
            for f in ("customization_options", "variant_combinations")
            if f not in sources
        ]
        if missing:
            add(
                "get_product",
                {"asin": asin, "fields": missing},
                "Selection fields have not been observed.",
                (asin,),
            )
        else:
            values = {
                axis: [o.value for o in options if o.is_available]
                for axis, options in detail.customization_options.items()
            }
            if all(len(options) == 1 for options in values.values()):
                selected = {axis: options[0] for axis, options in values.items()}
                checked = any(
                    c["asin"] == asin
                    and c["selected_options"] == selected
                    and c.get("field_sources") == sources
                    for c in memory.state.selection_checks.values()
                )
                if not checked:
                    add(
                        "verify_selection",
                        {"asin": asin, "selected_options": selected},
                        "Check the unambiguous selection's availability and price.",
                        (asin,),
                        refs=sorted(set(sources.values())),
                    )
            else:
                add(
                    "verify_selection",
                    {"asin": asin},
                    "Choose options from observed evidence, then check the selection.",
                    (asin,),
                    refs=sorted(set(sources.values())),
                    needs=("selected_options",),
                )
        unknown = sorted(hard & set(candidate["unknown_constraints"]))
        if unknown:
            field = "attributes" if "attributes" in sources else None
            descriptions = [
                c.description + c.source_quote
                for c in task.constraints
                if c.constraint_id in unknown
            ]
            if "customization_options" in sources and any(
                axis in text or any(o.value in text for o in options)
                for axis, options in detail.customization_options.items()
                for text in descriptions
            ):
                field = "customization_options"
            if field:
                add(
                    "read_evidence",
                    {
                        "asin": asin,
                        "observation_id": sources[field],
                        "field_paths": [f"/{field}"],
                    },
                    "Review observed fields for unresolved requirements; choose other fields when appropriate.",
                    (asin,),
                    unknown,
                    (sources[field],),
                )
            else:
                add(
                    "get_product",
                    {"asin": asin, "fields": ["attributes", "description"]},
                    "Collect missing attribute evidence for unresolved requirements.",
                    (asin,),
                    unknown,
                )
    if len(inspected) >= 2:
        # Comparison competes with repeated inspection instead of being appended after every candidate.
        comparison = ActionProposal(
            tool_name="compare_products",
            arguments={"asins": inspected[:3]},
            information_gap="Compare observed alternatives and unresolved constraints.",
            target_asins=inspected[:3],
        ).model_dump(mode="json")
        if (
            "compare_products" in allowed
            and action_signature("compare_products", comparison["arguments"])
            not in blocked
        ):
            proposals.insert(1 if proposals else 0, comparison)
            proposals = proposals[:limit]
    used = {normalized_query(q) for q in memory.state.queries}
    for query in plan.queries:
        if normalized_query(query) not in used:
            add(
                "search_products",
                {"query": query},
                "Explore an unused planned query for additional candidates.",
                needs=("method",),
            )
            break
    if memory.state.search_requests:
        previous = memory.state.search_requests[-1]
        if previous["next_offset"] is not None:
            add(
                "search_products",
                {**previous["arguments"], "offset": previous["next_offset"]},
                "Continue the same filtered result set without repeating its first page.",
            )
    return proposals[:limit]
