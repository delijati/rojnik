"""Tests for SKILL.md loading and constrained on-demand access."""

from pathlib import Path

import pytest

from rojnik.skills import Skill, SkillError, compose_skill_prompt, load_skills, make_load_skill_tool


def _write_skill(
    directory: Path,
    *,
    name: str = "testing",
    description: str = "Run focused tests.",
    instructions: str = "# Testing\n\nRun the smallest relevant test suite.",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "SKILL.md"
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{instructions}\n",
        encoding="utf-8",
    )
    return path


class TestSkillLoading:
    def test_loads_file(self, tmp_path):
        path = _write_skill(tmp_path / "testing")

        skill = Skill.from_path(path)

        assert skill.name == "testing"
        assert skill.description == "Run focused tests."
        assert skill.instructions.startswith("# Testing")
        assert skill.path == path.resolve()

    def test_directory_resolves_skill_md(self, tmp_path):
        directory = tmp_path / "testing"
        _write_skill(directory)

        assert Skill.from_path(directory).name == "testing"

    def test_requires_skill_filename(self, tmp_path):
        path = tmp_path / "instructions.md"
        path.write_text("text", encoding="utf-8")

        with pytest.raises(SkillError, match="named SKILL.md"):
            Skill.from_path(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(SkillError, match="not found"):
            Skill.from_path(tmp_path / "missing" / "SKILL.md")

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("# no front matter", "YAML front matter"),
            ("---\nname: test\n---\nbody", "description"),
            ("---\ndescription: test\n---\nbody", "name"),
            ("---\nname: test\ndescription: test\n---\n", "Markdown instructions"),
            ("---\nname: bad name\ndescription: test\n---\nbody", "Skill name"),
        ],
    )
    def test_rejects_invalid_format(self, tmp_path, text, message):
        path = tmp_path / "SKILL.md"
        path.write_text(text, encoding="utf-8")

        with pytest.raises(SkillError, match=message):
            Skill.from_path(path)

    def test_rejects_malformed_yaml(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_text("---\nname: [\ndescription: test\n---\nbody", encoding="utf-8")

        with pytest.raises(SkillError, match="Invalid YAML"):
            Skill.from_path(path)

    def test_rejects_invalid_utf8(self, tmp_path):
        path = tmp_path / "SKILL.md"
        path.write_bytes(b"\xff\xfe")

        with pytest.raises(SkillError, match="UTF-8"):
            Skill.from_path(path)

    def test_enforces_file_size(self, tmp_path):
        path = _write_skill(tmp_path / "large", instructions="x" * 100)

        with pytest.raises(SkillError, match="maximum"):
            Skill.from_path(path, max_bytes=20)


class TestSkillCatalog:
    def test_preserves_order(self, tmp_path):
        first = _write_skill(tmp_path / "first", name="first")
        second = _write_skill(tmp_path / "second", name="second")

        skills = load_skills([first, second])

        assert [skill.name for skill in skills] == ["first", "second"]

    def test_rejects_duplicate_names(self, tmp_path):
        first = _write_skill(tmp_path / "first", name="same")
        second = _write_skill(tmp_path / "second", name="same")

        with pytest.raises(SkillError, match="Duplicate"):
            load_skills([first, second])

    def test_enforces_skill_count(self, tmp_path):
        paths = [_write_skill(tmp_path / str(index), name=f"skill-{index}") for index in range(2)]

        with pytest.raises(SkillError, match="maximum is 1"):
            load_skills(paths, max_skills=1)

    def test_enforces_total_size_for_preloaded_skills(self, tmp_path):
        skills = [
            Skill("first", "First", "a" * 10, tmp_path / "first"),
            Skill("second", "Second", "b" * 10, tmp_path / "second"),
        ]

        with pytest.raises(SkillError, match="Combined"):
            load_skills(skills, max_total_skill_bytes=15)

    def test_eager_prompt_contains_instructions(self, tmp_path):
        skill = Skill.from_path(_write_skill(tmp_path / "testing"))

        prompt = compose_skill_prompt("Base prompt", [skill], eager=True)

        assert prompt.startswith("Base prompt")
        assert "## testing" in prompt
        assert "Run the smallest relevant test suite" in prompt

    def test_on_demand_prompt_contains_only_catalog(self, tmp_path):
        skill = Skill.from_path(_write_skill(tmp_path / "testing"))

        prompt = compose_skill_prompt("Base prompt", [skill], eager=False)

        assert "testing: Run focused tests." in prompt
        assert "Run the smallest relevant test suite" not in prompt


class TestLoadSkillTool:
    async def test_loads_known_skill(self, tmp_path):
        skill = Skill.from_path(_write_skill(tmp_path / "testing"))
        load_skill = make_load_skill_tool([skill])

        result = await load_skill(name="testing")

        assert result.startswith("# Skill: testing")
        assert "smallest relevant test suite" in result

    async def test_unknown_name_does_not_accept_paths(self, tmp_path):
        skill = Skill.from_path(_write_skill(tmp_path / "testing"))
        load_skill = make_load_skill_tool([skill])

        result = await load_skill(name="../testing/SKILL.md")

        assert result.startswith("ERROR: Unknown skill")
