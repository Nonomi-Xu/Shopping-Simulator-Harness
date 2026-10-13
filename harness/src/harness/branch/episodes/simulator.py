"""Reproducible user replies. Simulator objects have no target-product capability."""
from typing import Literal
from ...agent.contracts import Contract
from ...agent.events import digest
from ...providers import ChatMessage, ToolDefinition
from .models import UserGoal, DisclosurePolicy, DisclosureState, DisclosureEvent
from .disclosure import TOPICS, question_topics


class DeterministicSimulator:
    version = "deterministic-user-v1"

    def __init__(self, goal: UserGoal, policy: DisclosurePolicy, initially_revealed: list[str]):
        ids = {r.requirement_id for r in goal.requirements}
        if not set(initially_revealed) <= ids: raise ValueError("invalid initial disclosure")
        self.goal = goal.model_copy(deep=True)
        self.policy = policy
        self.state = DisclosureState(revealed_fields=set(initially_revealed), unrevealed_fields=ids-set(initially_revealed))

    @property
    def manifest(self):
        return {"simulator_version": self.version, "topics_hash": digest(TOPICS), "policy": self.policy.model_dump(mode="json")}

    def answer(self, question: str, *, question_id: str, action_sequence: int) -> DisclosureEvent:
        topics = question_topics(question)
        relevant = [r for r in self.goal.requirements if r.topic in topics]
        fresh = [r for r in relevant if r.requirement_id not in self.state.revealed_fields]
        selected = (fresh or relevant)[:self.policy.max_requirements_per_answer]
        new_ids = [r.requirement_id for r in selected if r.requirement_id in self.state.unrevealed_fields]
        answer = "；".join(r.source_quote for r in selected)
        if not answer:
            answer = "这方面我还没有确定的要求。" if topics else "请具体问一下预算、用途或某项规格。"
        repeated = bool(topics) and not new_ids and (
            bool(relevant) or set(topics) <= self.state.answered_topics)
        # Quotes must be atomic: do not reveal any other atom as part of a reply.
        selected_ids = {r.requirement_id for r in selected}
        if any(r.source_quote in answer for r in self.goal.requirements
               if r.requirement_id in self.state.unrevealed_fields and r.requirement_id not in selected_ids):
            raise ValueError("overlapping goal quotes violate disclosure policy")
        self.state = DisclosureState(revealed_fields=self.state.revealed_fields | set(new_ids),
            unrevealed_fields=self.state.unrevealed_fields - set(new_ids),
            answered_topics=self.state.answered_topics | set(topics))
        return DisclosureEvent(question_id=question_id, action_sequence=action_sequence,
            topic=",".join(topics) or None, revealed_ids=new_ids, repeated=repeated, answer=answer)


class Wording(Contract):
    style: Literal["direct", "polite", "conversational"]


class ConstrainedLLMSimulator(DeterministicSimulator):
    """Optional model chooses wording style; it cannot invent or select hidden facts.

    Fact routing stays deterministic. The provider sees only this reply's allowed
    text, not the complete goal, original query, product or evaluation record.
    """
    version = "constrained-llm-user-v1"

    def __init__(self, goal, policy, initially_revealed, *, provider):
        super().__init__(goal, policy, initially_revealed)
        self.provider = provider

    @property
    def manifest(self):
        return {**super().manifest, "model": self.provider.model, "provider": self.provider.name}

    def answer(self, *args, **kwargs):
        event = super().answer(*args, **kwargs)
        turn = self.provider.complete(
            [ChatMessage(role="system", content="Choose a reply style with choose_style. All facts are fixed."),
             ChatMessage(role="user", content=event.answer)],
            tools=[ToolDefinition.from_action_model(name="choose_style", action_model=Wording, description="Choose only a wording style.")],
            tool_choice={"type": "function", "function": {"name": "choose_style"}}, parameters={"temperature": 0})
        if len(turn.tool_calls) != 1 or turn.tool_calls[0].name != "choose_style":
            raise ValueError("invalid simulator wording output")
        style = Wording.model_validate(turn.tool_calls[0].arguments).style
        prefix = {"direct": "", "polite": "我的想法是：", "conversational": "嗯，"}[style]
        return event.model_copy(update={"answer": prefix + event.answer})
