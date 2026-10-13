from __future__ import annotations

import os
from pathlib import Path
import re
import shlex


_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_dotenv(path: str | Path | None = None) -> Path | None:
    candidates = [Path(path)] if path is not None else [Path.cwd() / ".env"]
    selected = next((item.expanduser().resolve() for item in candidates if item.is_file()), None)
    if selected is None:
        if path is not None:
            raise FileNotFoundError(f"dotenv file not found: {Path(path).expanduser()}")
        return None
    for line_number, line in enumerate(selected.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        name, separator, raw_value = stripped.partition("=")
        name = name.strip()
        if not separator or not _NAME.fullmatch(name):
            raise ValueError(f"invalid dotenv assignment at {selected}:{line_number}")
        lexer = shlex.shlex(raw_value, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = "#"
        os.environ.setdefault(name, " ".join(lexer))
    return selected
