"""Offline structured extraction, independent from the shopping decision policy."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol
from ..providers import ChatMessage, ModelProvider, ToolDefinition
from ..agent.events import digest
from ..constraints import Predicate, validate_predicate_quote
from .models import GoalRequirement, UserGoal


class GoalParser(Protocol):
    @property
    def manifest(self) -> dict: ...
    def parse(self, query: str) -> UserGoal: ...


def validate_goal(query: str, goal: UserGoal) -> UserGoal:
    """Validate provenance without using product labels as extraction hints."""
    for requirement in goal.requirements:
        quote = requirement.source_quote
        if not quote.strip() or quote not in query:
            raise ValueError("goal quote is not present in the query")
        p = requirement.predicate
        if p: validate_predicate_quote(p, quote)
    return goal


class StructuredGoalParser:
    def __init__(self, provider: ModelProvider, *, parameters: dict | None = None):
        self.provider = provider
        self.parameters = parameters or {"temperature": 0}
        self.prompt = Path(__file__).with_name("prompts").joinpath("goal_parser.md").read_text()

    @property
    def manifest(self):
        return {"parser_version": "structured-goal-v1", "provider": self.provider.name,
                "model": self.provider.model, "parameters_hash": digest({**getattr(self.provider, "default_parameters", {}), **self.parameters}),
                "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest()}

    def parse(self, query):
        result = self.provider.complete(
            [ChatMessage(role="system", content=self.prompt), ChatMessage(role="user", content=query)],
            tools=[ToolDefinition.from_action_model(name="parse_goal", action_model=UserGoal, description="Extract only query-grounded user requirements.")],
            tool_choice={"type": "function", "function": {"name": "parse_goal"}}, parameters=self.parameters)
        if len(result.tool_calls) != 1 or result.tool_calls[0].name != "parse_goal":
            raise ValueError("parser requires exactly one structured parse_goal result")
        return validate_goal(query, UserGoal.model_validate(result.tool_calls[0].arguments))


class RuleGoalParser:
    """Small reproducible fixture parser, not a general natural-language extractor.

    Unhandled clauses are preserved as review requirements and cannot enter a benchmark
    silently. Use StructuredGoalParser or curated cache records for the full dataset.
    """
    CATEGORIES = ("笔记本电脑", "笔记本", "乳胶枕头", "乳胶枕", "耳机", "冰箱", "投光灯", "过滤网")

    @property
    def manifest(self):
        return {"parser_version": "rule-goal-v1", "model": None, "vocabulary_hash": digest(self.CATEGORIES)}

    def parse(self, query):
        requirements = []
        covered = []
        def add(topic, match, *, priority="hard", predicate=None):
            quote = match.group() if hasattr(match, "group") else match
            requirements.append(GoalRequirement(requirement_id=f"r{len(requirements)+1}", topic=topic,
                description=quote, source_quote=quote, priority=priority, predicate=predicate,
                critical=priority == "hard" and topic != "category"))
            covered.append(quote)
        category = next((c for c in self.CATEGORIES if c in query), None)
        if category:
            add("category", category, predicate=Predicate(field="category", op="contains", value=category))
        for m in re.finditer(r"(?:预算(?:在|为)?|不超过|最多)?\s*(\d+(?:\.\d+)?)元(?:以内|以下|之内)", query):
            add("budget", m, predicate=Predicate(field="price", op="le", value=float(m[1]), unit="CNY"))
        for m in re.finditer(r"(?:至少|不低于)(\d+(?:\.\d+)?)\s*(?:GB|G)内存", query, re.I):
            add("ram", m, predicate=Predicate(field="attribute", key="ram", op="ge", value=float(m[1]), unit="GB"))
        for word in ("编程", "数据分析", "办公", "游戏", "旅行"):
            if word in query: add("use_case", word, priority="soft")
        for word, topic in (("轻一点", "weight"), ("续航越长越好", "battery")):
            if word in query: add(topic, word, priority="soft")
        for m in re.finditer(r"(?:不要|不考虑)(?:Apple|苹果|Mac)", query, re.I):
            add("brand", m)
        rest = query
        for quote in covered: rest = rest.replace(quote, "")
        rest = re.sub(r"我想买|我想要|我需要|帮我推荐|一台|一副|一款|适合|主要用于|主要|希望|用于|和|的|，|。|；|、|\s", "", rest)
        reasons = []
        if rest:
            reasons.append("rule parser has uncovered text")
            # Preserve unsupported clauses instead of silently guessing constraints.
            for clause in re.split(r"[，。；]", query):
                if clause.strip() and not any(clause.strip() == r.source_quote for r in requirements):
                    add("other", clause.strip(), priority="soft")
        if category is None: reasons.append("category not recognized")
        if not requirements: add("other", query, priority="soft")
        return validate_goal(query, UserGoal(requirements=requirements,
            status="needs_review" if reasons else "ready", review_reasons=reasons))
