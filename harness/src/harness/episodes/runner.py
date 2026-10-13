"""Private episode controller around the existing AgentRunner; no second agent loop."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import Field
from ..agent import AgentRunner, RuntimeConfig
from ..agent.contracts import Contract, AgentRunResult, UserReply
from ..agent.events import digest
from ..catalog import ProductCatalog
from ..evaluator import evaluate_submission
from ..schemas import EvaluationResult, SubmitRecommendationsAction
from ..tasks import TaskStore
from .environment import EpisodeEnvironment
from .models import EpisodeConfig, ShoppingEpisode, Trajectory, DisclosureEvent
from .simulator import DeterministicSimulator
from .trajectory import project_trajectory
from .metrics import compute_metrics, MetricContext


class EpisodeResult(Contract):
    """Controller-only result. Export trajectory separately from this audit record."""
    trajectory: Trajectory
    evaluation: EvaluationResult | None
    disclosures: list[DisclosureEvent]
    revealed_requirement_ids: list[str]
    metrics: dict[str, Any] = Field(default_factory=dict)
    manifest: dict[str, Any]
    run: AgentRunResult


class EpisodeRunner:
    def __init__(self, *, products_path, index_path, provider, config=None, runtime_config=None,
                 provider_parameters=None, skill_dir=None, simulator_factory=DeterministicSimulator, task_store=None):
        self.tasks = task_store if task_store is not None else TaskStore(products_path)
        self.index_path = index_path
        self.provider = provider
        self.config = config or EpisodeConfig()
        self.runtime = runtime_config or RuntimeConfig()
        self.provider_parameters = provider_parameters
        self.skill_dir = skill_dir
        self.simulator_factory = simulator_factory

    def run_episode(self, episode: ShoppingEpisode, *, baseline: Literal["full_query", "long_horizon"] = "long_horizon") -> EpisodeResult:
        episode = ShoppingEpisode.model_validate(episode)
        if baseline not in {"full_query", "long_horizon"}: raise ValueError("unknown baseline")
        if episode.dataset_sha256 != self.tasks.product_data_sha256:
            raise ValueError("episode dataset hash does not match TaskStore")
        task = self.tasks.get_public_task(episode.task_id)
        if task.query != episode.full_query or episode.hidden_goal.status != "ready":
            raise ValueError("episode query or quality status mismatch")
        full = baseline == "full_query"
        initial = [r.requirement_id for r in episode.hidden_goal.requirements] if full else episode.initial_requirement_ids
        simulator = self.simulator_factory(episode.hidden_goal, episode.disclosure_policy, initial)
        config = self.runtime.model_copy(update={
            "interaction_mode": "offline" if full else "interactive", "progressive_disclosure": not full,
            "enforce_constraint_filters": True, "max_actions": self.config.max_tool_calls,
            "max_searches": self.config.max_searches, "max_user_questions": self.config.max_user_questions})
        disclosures = []
        with EpisodeEnvironment(ProductCatalog(self.index_path, expected_product_sha256=self.tasks.product_data_sha256),
                                default_max_steps=self.config.max_environment_calls) as env:
            agent = AgentRunner(provider=self.provider, harness=env, config=config, skill_dir=self.skill_dir,
                max_model_turns=self.config.max_turns, provider_parameters=self.provider_parameters)
            run = agent.run_public_task(episode.public_task(full_query=full), session_id=episode.episode_id+"_"+baseline)
            while run.status == "waiting_for_user":
                if run.model_turn_count >= self.config.max_turns or run.action_count >= self.config.max_tool_calls:
                    reason = "max_model_turns" if run.model_turn_count >= self.config.max_turns else "max_actions"
                    run = agent.stop_waiting(run, reason)
                    break
                question = run.pending_question
                def known_predicates():
                    return [r.predicate for r in episode.hidden_goal.requirements if r.priority == "hard"
                            and r.predicate is not None and r.requirement_id in simulator.state.revealed_fields]
                before = env._catalog.eligible_count(known_predicates())
                disclosure_before = simulator.state.model_copy(deep=True)
                try:
                    event = simulator.answer(question.question, question_id=question.question_id, action_sequence=run.action_count)
                except Exception:
                    simulator.state = disclosure_before
                    run = agent.stop_waiting(run, "simulator_error")
                    break
                after = env._catalog.eligible_count(known_predicates())
                event = event.model_copy(update={"candidates_before": before, "candidates_after": after,
                                                "executable_constraints": len(known_predicates())})
                disclosures.append(event)
                run = agent.resume(run, UserReply(text=event.answer))
            trajectory = project_trajectory(run, episode.episode_id, baseline)
            evaluation = None
            evaluation_error = None
            if run.terminal_reason == "submitted":
                try:
                    evaluation = evaluate_submission(facts=self.tasks.get_private_facts(episode.task_id),
                        submission=SubmitRecommendationsAction(recommendations=trajectory.final_recommendation), catalog=env._catalog)
                except Exception as exc:
                    evaluation_error = type(exc).__name__
            manifest = {"episode_runner_version": "episode-runner-v1", "episode_id": episode.episode_id,
                "episode_hash": digest(episode.model_dump(mode="json")), "dataset_sha256": episode.dataset_sha256,
                "config": self.config.model_dump(mode="json"), "baseline": baseline,
                "simulator": simulator.manifest, "parser": episode.parser_manifest, "runtime": run.manifest}
            manifest["evaluation_error"] = evaluation_error
            from ...data.product_facts import FACTS_VERSION
            from ...data.search_bm25 import sha256_file
            manifest.update(public_predicates_version=FACTS_VERSION, index_sha256=sha256_file(self.index_path))
            metrics = compute_metrics(MetricContext(episode, trajectory, evaluation, disclosures,
                simulator.state.revealed_fields, env._catalog))
        return EpisodeResult(trajectory=trajectory, evaluation=evaluation, disclosures=disclosures,
            revealed_requirement_ids=sorted(simulator.state.revealed_fields), manifest=manifest, run=run, metrics=metrics)
