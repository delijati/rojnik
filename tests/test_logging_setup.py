"""Logging never writes files unless a file sink is explicitly configured."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from loguru import logger

import rojnik
from rojnik.config import Settings
from rojnik.logging_setup import setup_logging

_SRC = Path(rojnik.__file__).resolve().parents[1]


def _import_rojnik(
    cwd: Path, code: str = "import rojnik", **env: str
) -> subprocess.CompletedProcess[str]:
    clean = {key: value for key, value in os.environ.items() if not key.startswith("AGENT_")}
    clean.update({"AGENT_PROVIDER": "local", "PYTHONPATH": str(_SRC), **env})
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=clean,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.fixture
def restore_logging():
    yield
    setup_logging()


def test_settings_have_no_log_file_by_default(monkeypatch):
    monkeypatch.delenv("AGENT_LOG_FILE", raising=False)
    monkeypatch.setenv("AGENT_PROVIDER", "local")

    assert Settings().log_file == ""


def test_import_in_read_only_directory_writes_no_file(tmp_path):
    workdir = tmp_path / "app"
    workdir.mkdir()
    workdir.chmod(0o555)
    try:
        result = _import_rojnik(
            workdir, "import rojnik; from loguru import logger; logger.info('bereit')"
        )
    finally:
        workdir.chmod(0o755)

    assert result.returncode == 0, result.stderr
    assert "bereit" in result.stderr
    assert list(workdir.iterdir()) == []


def test_configured_log_file_is_not_opened_at_import(tmp_path):
    log_file = tmp_path / "logs" / "agent.log"
    log_file.parent.mkdir()

    result = _import_rojnik(tmp_path, AGENT_LOG_FILE=str(log_file))

    assert result.returncode == 0, result.stderr
    assert not log_file.exists()


def test_empty_log_file_configures_stderr_only(tmp_path, monkeypatch, restore_logging):
    monkeypatch.chdir(tmp_path)

    setup_logging(log_file="")
    logger.info("nur stderr")

    assert list(tmp_path.iterdir()) == []


def test_file_sink_opens_lazily_and_writes_json(tmp_path, restore_logging):
    log_file = tmp_path / "agent.log"

    setup_logging(log_file=str(log_file))
    assert not log_file.exists()

    logger.info("trace.written", marker="ok")
    setup_logging()

    records = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert [record["record"]["message"] for record in records] == ["trace.written"]
    assert records[0]["record"]["extra"] == {"marker": "ok"}
