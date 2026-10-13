"""Project existing runtime events into typed public per-action trajectories."""
from ...agent.contracts import TaskProfile, AgentAction, AgentObservation, WorkingMemory
from ...schemas import RecommendationItem
from .models import PreferenceState, CandidateSnapshot, TrajectoryTurn, Trajectory


ACTION_TYPES = {"ask_user": "ask", "search_products": "search", "get_product": "inspect",
                "compare_products": "compare", "submit_recommendations": "recommend"}


def preferences(task):
    return PreferenceState(known_requirements=task.constraints, uncertain_requirements=task.uncertain_requirements,
                           requirements_status=task.requirements_status, revision=task.revision)


def project_trajectory(run, episode_id, baseline):
    task = TaskProfile.model_validate(run.events[0].payload["task_profile"])
    memory = WorkingMemory.model_validate(run.events[0].payload["memory"])
    turns = []
    pending = None
    final = []
    for event in run.events:
        if event.kind == "action":
            action = AgentAction.model_validate(event.payload)
            pending = {"turn": action.sequence, "action_type": ACTION_TYPES.get(action.tool_name, "internal"),
                       "preferences_before": preferences(task), "action": action}
        elif event.kind == "observation":
            observation = AgentObservation.model_validate(event.payload)
            returned = [p["asin"] for p in observation.data] if observation.tool_name == "search_products" and observation.status == "success" else []
            snapshot = CandidateSnapshot(surfaced_ids=list(memory.candidates),
                inspected_ids=[p.asin for p in memory.candidates.values() if p.inspected], returned_ids=returned,
                search_total=observation.projection.total_results)
            turns.append(TrajectoryTurn(**pending, observation=observation, preferences_after=preferences(task), candidates=snapshot))
            if observation.tool_name == "submit_recommendations" and observation.status == "success":
                final = [RecommendationItem.model_validate(r) for r in pending["action"].arguments["recommendations"]]
        elif event.kind == "plan_updated":
            task = TaskProfile.model_validate(event.payload["task_profile"])
            if turns:
                turns[-1] = turns[-1].model_copy(update={"preferences_after": preferences(task)})
        elif event.kind == "memory_updated":
            memory = WorkingMemory.model_validate(event.payload)
            if turns:
                turns[-1] = turns[-1].model_copy(update={"candidates": turns[-1].candidates.model_copy(update={
                    "surfaced_ids": list(memory.candidates), "inspected_ids": [p.asin for p in memory.candidates.values() if p.inspected]})})
        elif event.kind == "user_message":
            if turns:
                turns[-1] = turns[-1].model_copy(update={"user_reply": event.payload["message"]["text"]})
    return Trajectory(episode_id=episode_id, task=run.session.public_task, baseline=baseline, turns=turns,
        final_recommendation=final, terminal_reason=run.terminal_reason, model_turn_count=run.model_turn_count,
        tool_call_count=run.action_count, environment_call_count=run.tool_call_count)
