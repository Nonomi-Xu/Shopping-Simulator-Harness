"""Project-anchored CLI result paths, independent of the process working directory."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_DIR / "outputs" / "bm25"


def result_path(value: str | Path) -> Path:
    """Require absolute result/checkpoint paths; never depend on the working directory."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("result paths must be absolute (or start with ~)")
    return path.resolve()


def new_run_id(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{timestamp}_{uuid4().hex}"
