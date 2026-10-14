"""Load the repository-owned shopping Agent Skill into a model prompt."""

from __future__ import annotations

import hashlib
import re
import sysconfig
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


_REFERENCE_LINK = re.compile(r"\[[^\]]+\]\((references/[^)]+\.md)\)")


class AgentSkillError(RuntimeError):
    """The Agent Skill is missing, unsafe, or structurally invalid."""


class AgentSkill(BaseModel):
    """Immutable, fully loaded Agent definition used for one run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    entrypoint: Path
    references: list[Path]
    system_prompt: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def default_shopping_skill_dir() -> Path:
    """Locate the project-local shopping Skill without consulting GRPO."""
    repository_candidate = (
        Path(__file__).resolve().parents[3] / "skills" / "shopping-agent"
    )
    working_directory_candidate = Path.cwd() / "skills" / "shopping-agent"
    installed_candidate = (
        Path(sysconfig.get_path("data"))
        / "share"
        / "shopping-simulator-harness"
        / "skills"
        / "shopping-agent"
    )
    for candidate in (
        repository_candidate,
        working_directory_candidate,
        installed_candidate,
    ):
        if (candidate / "SKILL.md").is_file():
            return candidate.resolve()
    raise AgentSkillError(
        "shopping-agent Skill not found; pass skill_dir explicitly or run "
        "from the Harness repository"
    )


def load_agent_skill(skill_dir: str | Path | None = None) -> AgentSkill:
    """Read SKILL.md and every directly declared Markdown reference.

    References are rendered in declaration order, and that order is part of the
    provider prefix a run sends on every turn.  Declaring the most stable
    reference first keeps the cacheable prefix as long as possible when a later
    reference is revised.
    """
    directory = (
        default_shopping_skill_dir()
        if skill_dir is None
        else Path(skill_dir).expanduser().resolve()
    )
    entrypoint = directory / "SKILL.md"
    if not entrypoint.is_file():
        raise AgentSkillError(f"Agent Skill entrypoint not found: {entrypoint}")

    entry_text = _read_text(entrypoint)
    metadata, body = _parse_frontmatter(entry_text, entrypoint)
    reference_paths = _resolve_references(directory, body)
    reference_texts = [(path, _read_text(path)) for path in reference_paths]

    prompt_parts = [
        "You are governed by the following project-owned Agent Skill. "
        "Follow it as the normative runtime policy.",
        f"# Skill: {metadata['name']}\n\n{body.strip()}",
    ]
    for path, text in reference_texts:
        relative_name = path.relative_to(directory).as_posix()
        prompt_parts.append(f"# Loaded reference: {relative_name}\n\n{text.strip()}")

    digest = hashlib.sha256()
    for path, text in [(entrypoint, entry_text), *reference_texts]:
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")

    return AgentSkill(
        name=metadata["name"],
        description=metadata["description"],
        entrypoint=entrypoint,
        references=reference_paths,
        system_prompt="\n\n".join(prompt_parts),
        source_sha256=digest.hexdigest(),
    )


def _read_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AgentSkillError(f"cannot read Agent Skill file: {path}") from exc
    if not text.strip():
        raise AgentSkillError(f"Agent Skill file is empty: {path}")
    return text


def _parse_frontmatter(text: str, path: Path) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise AgentSkillError(f"missing YAML frontmatter in {path}")
    try:
        end = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise AgentSkillError(f"unterminated YAML frontmatter in {path}") from exc

    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        if not separator or not key.strip() or not value.strip():
            raise AgentSkillError(f"unsupported frontmatter line in {path}: {line}")
        metadata[key.strip()] = value.strip()

    for required in ("name", "description"):
        if not metadata.get(required):
            raise AgentSkillError(f"frontmatter is missing {required!r} in {path}")
    body = "\n".join(lines[end + 1 :]).strip()
    if not body:
        raise AgentSkillError(f"Agent Skill body is empty: {path}")
    return metadata, body


def _resolve_references(directory: Path, body: str) -> list[Path]:
    declared = list(dict.fromkeys(_REFERENCE_LINK.findall(body)))
    if not declared:
        raise AgentSkillError("Agent Skill does not declare any Markdown references")

    directory = directory.resolve()
    references: list[Path] = []
    for relative_name in declared:
        candidate = (directory / relative_name).resolve()
        try:
            candidate.relative_to(directory)
        except ValueError as exc:
            raise AgentSkillError(
                f"Agent Skill reference escapes its directory: {relative_name}"
            ) from exc
        if not candidate.is_file():
            raise AgentSkillError(f"Agent Skill reference not found: {relative_name}")
        references.append(candidate)
    return references
