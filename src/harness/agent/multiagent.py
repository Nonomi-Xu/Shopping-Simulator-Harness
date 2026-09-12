"""Three model roles share public memory and a single global execution budget."""

from .contracts import ReviewProgress
from ..providers import canonical_json


ROLE_TOOLS = {
    "planner": {"update_plan", "ask_user"},
    "executor": {
        "search_products", "get_product", "read_evidence", "compare_products",
        "verify_selection", "submit_recommendations",
    },
    "observer": {
        "read_evidence", "record_evidence", "record_evidence_batch",
        "compare_products", "verify_selection", "review_progress",
    },
}

ROLE_PROMPTS = {
    "planner": (
        "You are the planning agent. Parse authentic user requirements, distinguish hard "
        "constraints from preferences, and produce a search and verification plan via "
        "update_plan. This hands control to the executor. In interactive mode ask one "
        "high-value question when necessary. Explain intended retrieval strategies in "
        "change_summary: BM25 for exact identifiers/keywords, vector for semantic needs. "
        "Choose only configured methods. On replan use the observer's review and keep "
        "requirements protected. Do not execute searches or assess product evidence."
    ),
    "executor": (
        "You are the action agent. Execute one useful action from the shared plan. "
        "For every search explicitly choose method=bm25 or method=vector based on the "
        "query and available methods; no automatic routing is performed for you. "
        "Inspect suitable returned products including exact variants. Your successful "
        "nonterminal action hands control to the observer. Only submit ASINs approved "
        "by the observer, with cited matches, tradeoffs and explicit unknowns. You cannot "
        "alter requirements or certify your own evidence."
    ),
    "observer": (
        "You are the observation agent, independently reviewing the executor's latest "
        "result and shared public evidence. Record cited constraint assessments, compare "
        "alternatives, and verify selections as needed. Then call review_progress: act "
        "to request another execution action, replan for changed strategy or user "
        "clarification, recommend with reviewed_asins only after inspecting their "
        "evidence. Explain gaps and next actions in summary. Approval means reviewed, "
        "not guaranteed correct; submission still enforces evidence and gap disclosure. "
        "After a search, normally request act to inspect useful candidates. Never search, "
        "invent facts, change requirements or submit a recommendation yourself."
    ),
}


class MultiAgentCoordinator:
    version = "shopping-multiagent-v1"

    def __init__(self):
        self.role = "planner"
        self.summary = "Parse the user's shopping needs."
        self.approved_asins = []
        self.approved_task_revision = None

    def snapshot(self):
        return {
            "agent_role": self.role,
            "summary": self.summary,
            "approved_asins": list(self.approved_asins),
            "approved_task_revision": self.approved_task_revision,
        }

    @classmethod
    def restore(cls, events):
        coordinator = cls()
        for event in events:
            if event.kind == "agent_handoff":
                state = event.payload
                coordinator.role = state["agent_role"]
                coordinator.summary = state["summary"]
                coordinator.approved_asins = list(state["approved_asins"])
                coordinator.approved_task_revision = state["approved_task_revision"]
        return coordinator

    def reset_for_reply(self):
        self.role = "planner"
        self.summary = "The user replied; update requirements and plan before acting."
        self.approved_asins = []
        self.approved_task_revision = None

    def prompt(self, harness_manifest):
        return (
            "\nMULTI-AGENT ROLE CONTRACT (the tool restrictions below take precedence "
            "over generic instructions describing other agents' work):\n"
            + ROLE_PROMPTS[self.role]
            + "\nShared handoff state: " + canonical_json(self.snapshot())
            + "\nConfigured public catalog capabilities: " + canonical_json(harness_manifest)
        )

    def allowed(self, names, task):
        allowed = [name for name in names if name in ROLE_TOOLS[self.role]]
        if self.approved_task_revision != task.revision or not self.approved_asins:
            allowed = [name for name in allowed if name != "submit_recommendations"]
        return allowed

    def validate_review(self, review, task, memory):
        if review.next_step == "recommend":
            if task.requirements_status != "parsed":
                raise ValueError("review_requires_parsed_requirements")
            for asin in review.reviewed_asins:
                candidate = memory.candidates.get(asin)
                if candidate is None or not candidate.inspected:
                    raise ValueError("review_requires_inspected_product")

    def validate_submission(self, submission, task):
        if (
            self.approved_task_revision != task.revision
            or not {item.asin for item in submission.recommendations}
            <= set(self.approved_asins)
        ):
            raise ValueError("observer_review_required")

    def after_action(self, name, arguments, observation, task):
        if observation.status != "success" or observation.terminated:
            return
        if self.role == "planner" and name == "update_plan":
            self.role = "executor"
            self.summary = "Execute the updated plan; choose the search method explicitly."
            self.approved_asins = []
            self.approved_task_revision = None
        elif self.role == "executor":
            self.role = "observer"
            self.summary = f"Review {name} observation {observation.observation_id}."
            self.approved_asins = []
            self.approved_task_revision = None
        elif self.role == "observer" and name == "review_progress":
            review = ReviewProgress.model_validate(arguments)
            self.role = "planner" if review.next_step == "replan" else "executor"
            self.summary = review.summary
            self.approved_asins = list(review.reviewed_asins)
            self.approved_task_revision = (
                task.revision if review.next_step == "recommend" else None
            )
