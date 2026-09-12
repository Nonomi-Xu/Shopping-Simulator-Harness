"""Read-only comparisons and selection checks over observed public evidence."""

import json
import math

from .contracts import CompareProducts, TaskProfile, VerifySelection, WorkingMemory
from .memory import observed_product


def compare_products(
    request: CompareProducts, task: TaskProfile, memory: WorkingMemory
) -> dict:
    constraints = {c.constraint_id: c for c in task.constraints}
    ids = request.constraint_ids or list(constraints)[:10]
    if set(ids) - constraints.keys():
        raise ValueError("comparison references an unknown constraint")
    rows = []
    observed = []
    for asin in request.asins:
        detail, sources = observed_product(memory, asin)
        raw = detail.model_dump(mode="json")
        observed.append((raw, sources))
        fields = {}
        for field in request.fields:
            if field not in sources:
                fields[field] = {"status": "unobserved", "next_tool": "get_product"}
                continue
            value = raw[field]
            serialized = json.dumps(value, ensure_ascii=False)
            fields[field] = {
                "status": "observed",
                "source_observation_id": sources[field],
                "field_path": f"/{field}",
                "truncated": len(serialized) > 200,
            }
            if len(serialized) > 200:
                fields[field].update(
                    preview=serialized[:200], next_tool="read_evidence"
                )
            else:
                fields[field]["value"] = value
        assessments = memory.candidates[asin].assessments
        rows.append(
            {
                "asin": asin,
                "fields": fields,
                "constraints": {
                    cid: (
                        {
                            "status": assessments[cid].status,
                            "source_observation_id": assessments[cid].observation_id,
                            "field_path": assessments[cid].field_path,
                        }
                        if cid in assessments
                        else {"status": "unknown"}
                    )
                    for cid in ids
                },
            }
        )
    differences = []
    for field in request.fields:
        values = []
        for raw, sources in observed:
            values.append(raw[field] if field in sources else None)
        # Missing fields cannot establish equality, even when every value is missing.
        if any(row["fields"][field]["status"] == "unobserved" for row in rows):
            status = "unknown"
        else:
            status = (
                "equal" if all(value == values[0] for value in values) else "different"
            )
        differences.append({"field": field, "status": status})
    result = {
        "products": rows,
        "differences": differences,
        "constraints": [
            {
                "constraint_id": cid,
                "description": constraints[cid].description[:160],
                "description_truncated": len(constraints[cid].description) > 160,
                "priority": constraints[cid].priority,
            }
            for cid in ids
        ],
        "omitted_constraint_ids": [cid for cid in constraints if cid not in ids],
        "assessment_basis": "existing_model_assessments_not_independent_semantic_proof",
        "ranking": None,
    }
    if len(json.dumps(result, ensure_ascii=False)) > 24000:
        raise ValueError(
            "comparison exceeds response budget; request fewer products, fields or constraints"
        )
    return result


def verify_selection(
    request: VerifySelection, task: TaskProfile, memory: WorkingMemory
) -> dict:
    if (
        request.budget_constraint_id is not None
        and request.budget_constraint_id
        not in {c.constraint_id for c in task.constraints}
    ):
        raise ValueError("unknown budget constraint")
    detail, sources = observed_product(memory, request.asin)
    selected = request.selected_options
    issues = []
    missing = [
        field
        for field in ("customization_options", "variant_combinations")
        if field not in sources
    ]
    if missing:
        issues.append(
            {
                "code": "selection_fields_unobserved",
                "fields": missing,
                "next_tool": "get_product",
            }
        )
    if "customization_options" in sources:
        for axis, value in selected.items():
            if not any(
                o.value == value and o.is_available
                for o in detail.customization_options.get(axis, [])
            ):
                issues.append(
                    {
                        "code": "invalid_or_unavailable_option",
                        "axis": axis,
                        "value": value,
                    }
                )
    matches = [
        (index, variant)
        for index, variant in enumerate(detail.variant_combinations)
        if variant.is_available
        and all(variant.options.get(k) == v for k, v in selected.items())
    ]
    if selected and detail.variant_combinations and not matches:
        issues.append({"code": "unavailable_variant_combination"})
    invalid = any(i["code"] != "selection_fields_unobserved" for i in issues)
    price, basis, price_refs = None, "unknown", []
    axes = set(detail.customization_options)
    for variant in detail.variant_combinations:
        axes.update(variant.options)
    complete = not missing and axes <= selected.keys() and not invalid
    if complete and detail.variant_combinations and matches:
        prices = [v.price for _, v in matches]
        if (
            all(p is not None and math.isfinite(p) and p >= 0 for p in prices)
            and len(set(prices)) == 1
        ):
            price, basis = prices[0], "selected_variant"
            price_refs = [
                {
                    "source_observation_id": sources["variant_combinations"],
                    "field_path": f"/variant_combinations/{index}/price",
                }
                for index, _ in matches
            ]
    elif complete and len(axes) == 1:
        axis = next(iter(axes))
        options = [
            (i, o)
            for i, o in enumerate(detail.customization_options[axis])
            if o.value == selected.get(axis) and o.is_available
        ]
        prices = [o.price for _, o in options]
        if (
            prices
            and all(p is not None and math.isfinite(p) and p >= 0 for p in prices)
            and len(set(prices)) == 1
        ):
            price, basis = prices[0], "single_axis_option"
            escaped = axis.replace("~", "~0").replace("/", "~1")
            price_refs = [
                {
                    "source_observation_id": sources["customization_options"],
                    "field_path": f"/customization_options/{escaped}/{i}/price",
                }
                for i, _ in options
            ]
    elif complete and not axes and {"price_min", "price_max"} <= sources.keys():
        if (
            detail.price_min is not None
            and detail.price_min == detail.price_max
            and math.isfinite(detail.price_min)
            and detail.price_min >= 0
        ):
            price, basis = detail.price_min, "single_price_product_without_options"
            price_refs = [
                {"source_observation_id": sources[f], "field_path": f"/{f}"}
                for f in ("price_min", "price_max")
            ]
    budget = (
        "not_requested"
        if request.max_price is None
        else (
            "unknown"
            if price is None
            else "satisfied" if price <= request.max_price else "violated"
        )
    )
    return {
        "asin": request.asin,
        "selected_options": selected,
        "selection_status": (
            "invalid" if invalid else "unknown" if missing or not complete else "valid"
        ),
        "combination_status": (
            "invalid"
            if invalid
            else (
                "unknown"
                if missing
                or not complete
                or (len(axes) > 1 and not detail.variant_combinations)
                else "available"
            )
        ),
        "issues": issues,
        "unselected_axes": sorted(axes - selected.keys()),
        "selected_price": price,
        "price_basis": basis,
        "price_evidence": price_refs[:5],
        "omitted_price_evidence": max(0, len(price_refs) - 5),
        "max_price": request.max_price,
        "budget_status": budget,
        "budget_constraint_id": request.budget_constraint_id,
        "budget_basis": "model_supplied_threshold_in_catalog_price_units",
        "field_sources": sources,
        "semantic_verification": "not_performed",
    }
