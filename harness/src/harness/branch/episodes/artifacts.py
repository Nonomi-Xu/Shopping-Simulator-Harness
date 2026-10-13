"""Write public trajectories and private evaluation sidecars to separate directories."""
import json
from pathlib import Path
from ...agent.artifacts import write_run_artifacts, replay_run
from ...agent.events import digest
from .models import Trajectory


def write_episode_artifacts(result, directory):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=False)
    public = root / "public"
    private = root / "private"
    public.mkdir()
    private.mkdir(mode=0o700)
    trajectory = result.trajectory.model_dump(mode="json")
    (public/"trajectory.jsonl").write_text(json.dumps(trajectory, ensure_ascii=False)+"\n")
    # Runtime contains only the unlabelled public session. Never insert evaluation here.
    if result.run.session.evaluation is not None:
        raise ValueError("episode runtime artifacts must be unlabelled")
    write_run_artifacts(result.run, public/"runtime")
    private_payload = {"evaluation": result.evaluation.model_dump(mode="json") if result.evaluation else None,
        "metrics": result.metrics, "disclosures": [e.model_dump(mode="json") for e in result.disclosures],
        "revealed_requirement_ids": result.revealed_requirement_ids}
    (private/"evaluation.json").write_text(json.dumps(private_payload, ensure_ascii=False, indent=2)+"\n")
    manifest = {**result.manifest, "trajectory_sha256": digest(trajectory), "evaluation_sha256": digest(private_payload)}
    (private/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return root


def replay_episode(directory):
    root = Path(directory)
    manifest = json.loads((root/"private/manifest.json").read_text())
    lines = (root/"public/trajectory.jsonl").read_text().splitlines()
    if len(lines) != 1: raise ValueError("expected one episode trajectory")
    trajectory = Trajectory.model_validate_json(lines[0])
    evaluation = json.loads((root/"private/evaluation.json").read_text())
    if digest(trajectory.model_dump(mode="json")) != manifest["trajectory_sha256"] or digest(evaluation) != manifest["evaluation_sha256"]:
        raise ValueError("episode artifact hash mismatch")
    replay_run(root/"public/runtime")
    return {"verified": True, "episode_id": trajectory.episode_id, "metrics": evaluation["metrics"]}
