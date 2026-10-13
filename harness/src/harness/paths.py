"""Project-anchored CLI result paths, independent of the process working directory."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

HARNESS_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = HARNESS_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"


def result_path(value: str | Path) -> Path:
    """Resolve explicit relative result paths against this project, including readers."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


def new_run_id(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{timestamp}_{uuid4().hex}"


def default_run_path(category: str, prefix: str) -> Path:
    return (OUTPUTS_DIR / category / new_run_id(prefix)).resolve()
