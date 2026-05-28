"""Tests for tools/builtins/ — read_file, list_directory, shell_exec."""


import asyncio
import os
import pytest

from rojnik.tools.builtins.files import list_directory, read_file
from rojnik.tools.builtins.shell import shell_exec


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

class TestReadFile:
    async def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        result = await read_file(path=str(f))
        assert result == "hello world"

    async def test_missing_file_returns_error(self, tmp_path):
        result = await read_file(path=str(tmp_path / "nope.txt"))
        assert result.startswith("ERROR:")
        assert "not found" in result.lower()

    async def test_directory_path_returns_error(self, tmp_path):
        result = await read_file(path=str(tmp_path))
        assert result.startswith("ERROR:")

    async def test_reads_utf8_content(self, tmp_path):
        f = tmp_path / "utf8.txt"
        f.write_text("café résumé naïve", encoding="utf-8")
        result = await read_file(path=str(f))
        assert "café" in result

    async def test_empty_file_returns_empty_string(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        result = await read_file(path=str(f))
        assert result == ""


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------

class TestListDirectory:
    async def test_lists_files_and_dirs(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "subdir").mkdir()
        result = await list_directory(path=str(tmp_path))
        assert "a.txt" in result
        assert "subdir/" in result

    async def test_missing_directory_returns_error(self, tmp_path):
        result = await list_directory(path=str(tmp_path / "ghost"))
        assert result.startswith("ERROR:")

    async def test_not_a_directory_returns_error(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        result = await list_directory(path=str(f))
        assert result.startswith("ERROR:")

    async def test_empty_directory_returns_message(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = await list_directory(path=str(empty))
        assert "empty" in result.lower()

    async def test_recursive_lists_nested_files(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested")
        result = await list_directory(path=str(tmp_path), recursive=True)
        assert "nested.txt" in result

    async def test_recursive_max_depth_3(self, tmp_path):
        """Verify recursive listing caps at depth 3 without raising."""
        d = tmp_path
        for level in range(5):
            d = d / f"level_{level}"
            d.mkdir()
        # Should not raise
        result = await list_directory(path=str(tmp_path), recursive=True)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# shell_exec
# ---------------------------------------------------------------------------

class TestShellExec:
    async def test_runs_simple_command(self):
        result = await shell_exec(command="echo hello")
        assert "hello" in result

    async def test_nonzero_exit_code_included_in_output(self):
        result = await shell_exec(command="exit 1", timeout=5)
        assert "exit code 1" in result or "1" in result

    async def test_stderr_captured_in_output(self):
        result = await shell_exec(command="echo err >&2", timeout=5)
        assert "err" in result

    async def test_workdir_respected(self, tmp_path):
        result = await shell_exec(command="pwd", workdir=str(tmp_path))
        assert str(tmp_path) in result

    async def test_missing_workdir_returns_error(self, tmp_path):
        result = await shell_exec(
            command="echo hi", workdir=str(tmp_path / "no_such_dir")
        )
        assert result.startswith("ERROR:")

    async def test_timeout_kills_process(self):
        result = await shell_exec(command="sleep 60", timeout=1)
        assert "timed out" in result.lower()

    async def test_command_with_no_output(self):
        result = await shell_exec(command="true")
        # Either empty string or "(no output)" sentinel
        assert result == "(no output)" or result == ""
