"""Preflight recommendations using only the actor's task and observed products."""

from .contracts import AgentSubmission, TaskProfile, WorkingMemory, VerifySelection
from .comparison import verify_selection


class SubmissionNotReady(ValueError):
    def __init__(self, issues: list[dict]):
        self.issues = issues
        super().__init__(
            "Resolve the listed issues, or disclose every hard-constraint gap in best_effort mode."
        )


def check_submission(
    submission: AgentSubmission, task: TaskProfile, memory: WorkingMemory, *, enforce_predicates: bool = False
) -> None:
    """Check coverage and public validity, never consult private evaluation facts.

    Constraint statuses are the model's cited assessments, not semantic proofs.
    The caller strips mode/gaps before dispatching the unchanged environment contract.
    """
    issues = []
    if task.requirements_status != "parsed":
        issues.append({"code": "requirements_unparsed", "next_tool": "update_plan"})
    hard = {c.constraint_id for c in task.constraints if c.priority == "hard"}
    expected_gaps = {}
    reasons = {}
    seen = set()
    for item in submission.recommendations:
        asin = item.asin
        reasons[asin] = item.reason
        if asin in seen:
            issues.append({"code": "duplicate_asin", "asin": asin})
        seen.add(asin)
        candidate = memory.candidates.get(asin)
        if candidate is None or not candidate.inspected:
            issues.append({"code": "product_not_inspected", "asin": asin})
            continue
        detail = memory.evidence_archive.get(candidate.detail_observation_id)
        if detail is None or detail.asin != asin:
            issues.append({"code": "missing_detail_evidence", "asin": asin})
            continue
        selection = verify_selection(
            VerifySelection(asin=asin, selected_options=item.selected_options),
            task,
            memory,
        )
        issues.extend({**issue, "asin": asin} for issue in selection["issues"])
        for cid in sorted(hard):
            assessment = candidate.assessments.get(cid)
            status = assessment.status if assessment else "unknown"
            if status != "satisfied":
                expected_gaps[(asin, cid)] = status
        if enforce_predicates:
            from ...data.product_facts import predicate_statuses
            from .memory import observed_product
            observed, sources = observed_product(memory, asin)
            public = {k: v for k, v in observed.model_dump(mode="json").items() if k in sources}
            typed = [c for c in task.constraints if c.priority == "hard" and c.predicate is not None]
            statuses = predicate_statuses(public, [c.predicate for c in typed], selected=item.selected_options)
            for constraint, status in zip(typed, statuses):
                if status != "satisfied": expected_gaps[(asin, constraint.constraint_id)] = status
        # A recorded numeric contradiction must not be hidden by a self-reported assessment.
        for check in memory.selection_checks.values():
            cid = check.get("budget_constraint_id")
            if (
                check.get("asin") == asin
                and check.get("selected_options") == item.selected_options
                and cid in hard
                and check.get("max_price") is not None
            ):
                current = verify_selection(
                    VerifySelection(
                        asin=asin,
                        selected_options=item.selected_options,
                        max_price=check["max_price"],
                        budget_constraint_id=cid,
                    ),
                    task,
                    memory,
                )
                if current["budget_status"] in {"unknown", "violated"}:
                    expected_gaps[(asin, cid)] = current["budget_status"]

    actual_gaps = {}
    for gap in submission.gaps:
        key = (gap.asin, gap.constraint_id)
        if key in actual_gaps:
            issues.append(
                {
                    "code": "duplicate_gap",
                    "asin": gap.asin,
                    "constraint_id": gap.constraint_id,
                }
            )
        actual_gaps[key] = gap.status
        if gap.disclosure not in reasons.get(gap.asin, ""):
            issues.append(
                {
                    "code": "gap_missing_from_reason",
                    "asin": gap.asin,
                    "constraint_id": gap.constraint_id,
                }
            )
    if submission.mode == "verified":
        for (asin, cid), status in expected_gaps.items():
            issues.append(
                {
                    "code": "hard_constraint_gap",
                    "asin": asin,
                    "constraint_id": cid,
                    "status": status,
                }
            )
        if submission.gaps:
            issues.append({"code": "verified_submission_cannot_have_gaps"})
    elif actual_gaps != expected_gaps:
        issues.append(
            {
                "code": "incomplete_or_inaccurate_gap_disclosure",
                "expected_gaps": [
                    {"asin": asin, "constraint_id": cid, "status": status}
                    for (asin, cid), status in expected_gaps.items()
                ],
            }
        )
    if issues:
        raise SubmissionNotReady(issues)
