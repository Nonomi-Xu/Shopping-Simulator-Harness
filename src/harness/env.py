"""Small dependency-free dotenv loader for local CLI configuration."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shlex


_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DotEnvError(RuntimeError):
    """A requested dotenv file cannot be read or parsed safely."""


def load_dotenv(
    path: str | Path | None = None,
    *,
    disabled: bool = False,
) -> Path | None:
    """Load missing variables from an explicit or project-local .env file.

    Existing process variables always win. Values are never returned, logged,
    interpolated, or otherwise exposed by this function.
    """
    if disabled:
        return None
    resolved = _resolve_dotenv(path)
    if resolved is None:
        return None

    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DotEnvError(f"cannot read dotenv file: {resolved}") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        name, separator, raw_value = stripped.partition("=")
        name = name.strip()
        if not separator or not _ENVIRONMENT_NAME.fullmatch(name):
            raise DotEnvError(
                f"invalid dotenv assignment at {resolved}:{line_number}"
            )
        value = _parse_value(raw_value, resolved, line_number)
        os.environ.setdefault(name, value)
    return resolved


def _resolve_dotenv(path: str | Path | None) -> Path | None:
    if path is not None:
        explicit = Path(path).expanduser().resolve()
        if not explicit.is_file():
            raise DotEnvError(f"dotenv file not found: {explicit}")
        return explicit

    repository_root = Path(__file__).resolve().parents[2]
    candidates = (
        Path.cwd() / ".env",
        Path.cwd() / "src" / ".env",
        repository_root / ".env",
        repository_root / "src" / ".env",
    )
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def _parse_value(value: str, path: Path, line_number: int) -> str:
    lexer = shlex.shlex(value, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    try:
        tokens = list(lexer)
    except ValueError as exc:
        raise DotEnvError(
            f"invalid quoted value at {path}:{line_number}"
        ) from exc
    return " ".join(tokens)
