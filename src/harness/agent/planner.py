"""Explicit plans updated by the model and advanced by public runtime state."""

from .contracts import PlanUpdate, SearchPlan, TaskProfile, UserProfile, WorkingMemory
from .profile import validate_constraints


class Planner:
    version = "shopping-planner-v2"

    def start(self, profile: TaskProfile, session_id: str) -> SearchPlan:
        return SearchPlan(
            plan_id=f"{session_id}:plan",
            objective=profile.task.query,
            queries=[profile.task.query],
        )

    def update(
        self,
        plan: SearchPlan,
        update: PlanUpdate,
        task: TaskProfile,
        user: UserProfile,
        observation_ids: set[str],
    ) -> tuple[SearchPlan, TaskProfile]:
        validate_constraints(update.constraints, task, user)
        if not set(update.evidence_observation_ids) <= observation_ids:
            raise ValueError("replanning references an unknown observation")
        # Replacing a constraint invalidates its old model assessment in Memory.
        return plan.model_copy(
            update={
                "revision": plan.revision + 1,
                "queries": update.queries,
                "verification_targets": update.verification_targets,
                "change_summary": update.change_summary,
            }
        ), task.model_copy(
            update={
                "constraints": update.constraints,
                "uncertain_requirements": update.uncertain_requirements,
                "requirements_status": "parsed",
                "no_constraints_reason": update.no_constraints_reason,
            },
            deep=True,
        )

    def advance(
        self,
        plan: SearchPlan,
        task: TaskProfile,
        memory: WorkingMemory,
        searches_remaining: int,
    ) -> SearchPlan:
        inspected = [c for c in memory.candidates.values() if c.inspected]
        hard = [c.constraint_id for c in task.constraints if c.priority == "hard"]
        if task.requirements_status != "parsed":
            phase = "search" if not inspected else "verify"
        elif not memory.candidates:
            phase = "search"
        elif not inspected:
            phase = "inspect"
        elif any(
            all(
                cid in c.assessments and c.assessments[cid].status == "satisfied"
                for cid in hard
            )
            for c in inspected
        ):
            phase = "submit"
        elif searches_remaining > 0 and all(
            any(
                cid in c.assessments and c.assessments[cid].status == "violated"
                for cid in hard
            )
            for c in inspected
        ):
            phase = "search"
        else:
            phase = "verify"
        if phase == plan.phase:
            return plan
        return plan.model_copy(update={"phase": phase, "revision": plan.revision + 1})
