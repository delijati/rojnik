"""Explicit loading and prompt integration for Agent Skills ``SKILL.md`` files."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from rojnik.tools.base import tool

DEFAULT_MAX_SKILL_BYTES = 64 * 1024
DEFAULT_MAX_TOTAL_SKILL_BYTES = 256 * 1024
DEFAULT_MAX_SKILLS = 32

_SKILL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_FRONT_MATTER_PATTERN = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?(.*)\Z", re.DOTALL)


class SkillError(ValueError):
    """Raised when a skill file or catalog is invalid."""


@dataclass(frozen=True)
class Skill:
    """Validated instructions loaded from one ``SKILL.md`` file."""

    name: str
    description: str
    instructions: str
    path: Path

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        max_bytes: int = DEFAULT_MAX_SKILL_BYTES,
    ) -> Skill:
        skill_path = _resolve_skill_path(path)
        try:
            size = skill_path.stat().st_size
        except OSError as exc:
            raise SkillError(f"Could not inspect skill file {skill_path}: {exc}") from exc
        if size > max_bytes:
            raise SkillError(
                f"Skill file {skill_path} is {size} bytes; maximum is {max_bytes} bytes."
            )

        try:
            text = skill_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise SkillError(f"Skill file {skill_path} is not valid UTF-8.") from exc
        except OSError as exc:
            raise SkillError(f"Could not read skill file {skill_path}: {exc}") from exc

        match = _FRONT_MATTER_PATTERN.fullmatch(text)
        if match is None:
            raise SkillError(f"Skill file {skill_path} must start with YAML front matter.")

        try:
            metadata = yaml.safe_load(match.group(1))
        except yaml.YAMLError as exc:
            raise SkillError(f"Invalid YAML front matter in {skill_path}: {exc}") from exc
        if not isinstance(metadata, dict):
            raise SkillError(f"YAML front matter in {skill_path} must be a mapping.")

        name = metadata.get("name")
        description = metadata.get("description")
        instructions = match.group(2).strip()
        if not isinstance(name, str) or not name.strip():
            raise SkillError(f"Skill file {skill_path} requires a non-empty string 'name'.")
        name = name.strip()
        if not _SKILL_NAME_PATTERN.fullmatch(name):
            raise SkillError(
                f"Skill name {name!r} may contain only letters, numbers, underscores, and hyphens."
            )
        if not isinstance(description, str) or not description.strip():
            raise SkillError(f"Skill file {skill_path} requires a non-empty string 'description'.")
        if not instructions:
            raise SkillError(f"Skill file {skill_path} requires Markdown instructions after the front matter.")

        return cls(
            name=name,
            description=description.strip(),
            instructions=instructions,
            path=skill_path,
        )


SkillSource = str | Path | Skill


def _resolve_skill_path(path: str | Path) -> Path:
    skill_path = Path(path).expanduser().resolve()
    if skill_path.is_dir():
        skill_path = skill_path / "SKILL.md"
    elif skill_path.name != "SKILL.md":
        raise SkillError(f"Skill file must be named SKILL.md: {skill_path}")
    if not skill_path.exists():
        raise SkillError(f"Skill file not found: {skill_path}")
    if not skill_path.is_file():
        raise SkillError(f"Skill path is not a regular file: {skill_path}")
    return skill_path


def load_skills(
    sources: Sequence[SkillSource],
    *,
    max_skill_bytes: int = DEFAULT_MAX_SKILL_BYTES,
    max_total_skill_bytes: int = DEFAULT_MAX_TOTAL_SKILL_BYTES,
    max_skills: int = DEFAULT_MAX_SKILLS,
) -> tuple[Skill, ...]:
    """Load and validate an explicitly ordered skill catalog."""
    if len(sources) > max_skills:
        raise SkillError(f"Configured {len(sources)} skills; maximum is {max_skills}.")

    skills: list[Skill] = []
    names: set[str] = set()
    total_bytes = 0
    for source in sources:
        skill = source if isinstance(source, Skill) else Skill.from_path(source, max_bytes=max_skill_bytes)
        if not _SKILL_NAME_PATTERN.fullmatch(skill.name):
            raise SkillError(
                f"Skill name {skill.name!r} may contain only letters, numbers, underscores, and hyphens."
            )
        if not skill.description.strip():
            raise SkillError(f"Skill {skill.name!r} requires a non-empty description.")
        if not skill.instructions.strip():
            raise SkillError(f"Skill {skill.name!r} requires non-empty instructions.")
        if skill.name in names:
            raise SkillError(f"Duplicate skill name: {skill.name!r}")
        skill_bytes = len(skill.instructions.encode("utf-8"))
        if skill_bytes > max_skill_bytes:
            raise SkillError(
                f"Skill {skill.name!r} instructions are {skill_bytes} bytes; "
                f"maximum is {max_skill_bytes} bytes."
            )
        total_bytes += skill_bytes
        if total_bytes > max_total_skill_bytes:
            raise SkillError(
                f"Combined skill instructions exceed {max_total_skill_bytes} bytes."
            )
        names.add(skill.name)
        skills.append(skill)
    return tuple(skills)


def compose_skill_prompt(base_prompt: str, skills: Sequence[Skill], *, eager: bool) -> str:
    """Append a deterministic skill catalog or full skill instructions."""
    if not skills:
        return base_prompt

    if eager:
        entries = [
            f"## {skill.name}\n\nDescription: {skill.description}\n\n{skill.instructions}"
            for skill in skills
        ]
        guidance = (
            "The following trusted skills are part of your system instructions. "
            "Apply them when relevant."
        )
    else:
        entries = [f"- {skill.name}: {skill.description}" for skill in skills]
        guidance = (
            "Call load_skill with a listed name before applying that skill. "
            "Do not guess instructions that have not been loaded."
        )
    return f"{base_prompt.rstrip()}\n\n# Available Skills\n\n{guidance}\n\n" + "\n\n".join(entries)


def make_load_skill_tool(skills: Sequence[Skill]):
    """Create a constrained tool that loads instructions by catalog name."""
    catalog = {skill.name: skill for skill in skills}

    @tool(
        description=(
            "Load the complete instructions for one configured skill by name. "
            f"Available skills: {', '.join(catalog) or '(none)'}."
        )
    )
    async def load_skill(name: str) -> str:
        skill = catalog.get(name)
        if skill is None:
            return f"ERROR: Unknown skill {name!r}. Available skills: {list(catalog)}"
        return f"# Skill: {skill.name}\n\n{skill.instructions}"

    return load_skill
