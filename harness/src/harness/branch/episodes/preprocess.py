"""Batch query preprocessing. Evaluation never calls this pipeline implicitly."""
from ...agent.events import digest
from .models import ShoppingEpisode, DisclosurePolicy
from .parser import validate_goal


def build_episode(task, goal, dataset_sha256, parser_manifest, policy=None, seed=42):
    policy = policy or DisclosurePolicy()
    validate_goal(task.query, goal)
    if goal.status != "ready": raise ValueError("goal requires review")
    selected = [r for r in goal.requirements if r.topic in policy.initial_topics][:policy.max_initial_requirements]
    if not selected: raise ValueError("no initial topic can be disclosed")
    initial = "；".join(r.source_quote for r in selected)
    selected_ids = [r.requirement_id for r in selected]
    if any(r.source_quote in initial for r in goal.requirements if r.requirement_id not in selected_ids):
        raise ValueError("initial quote overlaps an undisclosed requirement")
    payload = dict(task_id=task.task_id, full_query=task.query, hidden_goal=goal,
        initial_observation=initial, initial_requirement_ids=selected_ids,
        disclosure_policy=policy, dataset_sha256=dataset_sha256,
        parser_manifest={**parser_manifest, "seed": seed})
    identity = digest({"task": task.model_dump(mode="json"), "goal": goal.model_dump(mode="json"),
                       "dataset": dataset_sha256, "policy": policy.model_dump(mode="json"), "parser": parser_manifest, "seed": seed})
    return ShoppingEpisode(episode_id="episode_" + identity[:24], **payload)


def prepare_episodes(tasks, parser, store, *, dataset_sha256, policy=None, seed=42):
    """Stream records, reuse validated cache, and return explicit failure coverage."""
    policy = policy or DisclosurePolicy()
    summary = {"prepared": 0, "cached": 0, "failed": 0, "failures": []}
    for task in tasks:
        key = digest({"task": task.model_dump(mode="json"), "dataset": dataset_sha256,
                      "parser": parser.manifest, "policy": policy.model_dump(mode="json"), "seed": seed})
        if store.get(key) is not None:
            summary["cached"] += 1
            continue
        try:
            goal = parser.parse(task.query)
            episode = build_episode(task, goal, dataset_sha256, parser.manifest, policy, seed)
        except Exception as exc:
            summary["failed"] += 1
            summary["failures"].append({"task_id": task.task_id, "error_type": type(exc).__name__})
            continue
        store.put(key, episode)
        summary["prepared"] += 1
    return summary
