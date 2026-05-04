"""Headless integration tests for examples/viz.py."""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

import pytest
import anyio  # noqa: F401  — registers the anyio pytest plugin

# Make examples/ importable without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import viz  # noqa: E402  (import after sys.path manipulation)
from viz import ContentPane, VizApp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(path: str) -> None:
    """Create a minimal agent.db fixture with one root session."""
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE sessions (
            id                TEXT PRIMARY KEY,
            agent_name        TEXT,
            parent_session_id TEXT,
            status            TEXT,
            task_preview      TEXT,
            result_preview    TEXT,
            created_at        TEXT,
            finished_at       TEXT,
            total_tokens      INTEGER
        );
        CREATE TABLE messages (
            id             TEXT PRIMARY KEY,
            session_id     TEXT,
            seq            INTEGER,
            role           TEXT,
            content        TEXT,
            tool_calls_json TEXT,
            tool_call_id   TEXT,
            token_count    INTEGER,
            created_at     TEXT
        );
        CREATE TABLE tool_results (
            id           TEXT PRIMARY KEY,
            session_id   TEXT,
            tool_call_id TEXT,
            tool_name    TEXT,
            input_json   TEXT,
            output       TEXT,
            error        TEXT,
            duration_ms  INTEGER,
            created_at   TEXT
        );
        INSERT INTO sessions VALUES (
            's1', 'orchestrator', NULL, 'completed',
            'do the thing with a very long task description that should not be truncated',
            'done with a very long result that should also not be truncated',
            '2024-01-01 10:00:00.000000',
            '2024-01-01 10:00:42.000000',
            2348
        );
        INSERT INTO messages VALUES (
            'm1', 's1', 1, 'user',
            'Please do the thing. This is a long message that would previously have been truncated at 72 characters.',
            NULL, NULL, 5, '2024-01-01 10:00:01.000000'
        );
        INSERT INTO messages VALUES (
            'm2', 's1', 2, 'assistant',
            'I will do it now.',
            '[{"id":"tc1","name":"run_shell","arguments":"{\"cmd\":\"echo hello world\"}"}]',
            NULL, 10, '2024-01-01 10:00:02.000000'
        );
        INSERT INTO tool_results VALUES (
            'tr1', 's1', 'tc1', 'run_shell',
            '{"cmd":"echo hello world"}', 'hello world', NULL, 42,
            '2024-01-01 10:00:02.500000'
        );
    """)
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_startup_shows_session() -> None:
    """App should mount with one ListItem for the single root session."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        _make_db(db_path)
        app = VizApp(db_path)
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            # Let _populate and _auto_select run
            await pilot.pause(0.2)
            lv = app.query_one("#session-list", viz.ListView)
            items = list(lv.query(viz.ListItem))
            assert len(items) == 1, f"Expected 1 session item, got {len(items)}"
            assert getattr(items[0], "_session_id", None) == "s1"
    finally:
        os.unlink(db_path)


@pytest.mark.anyio
async def test_action_refresh_no_duplicate_ids() -> None:
    """action_refresh called repeatedly must not raise DuplicateIds."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        _make_db(db_path)
        app = VizApp(db_path)
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await pilot.pause(0.2)

            # First refresh
            await app.action_refresh()
            await pilot.pause(0.2)

            # Second refresh — this was the original DuplicateIds crash site
            await app.action_refresh()
            await pilot.pause(0.2)

            lv = app.query_one("#session-list", viz.ListView)
            items = list(lv.query(viz.ListItem))
            assert len(items) == 1, (
                f"Expected 1 session item after double refresh, got {len(items)}"
            )
    finally:
        os.unlink(db_path)


@pytest.mark.anyio
async def test_session_labels_contain_agent_name() -> None:
    """Each ListItem's label should include the agent_name."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        _make_db(db_path)
        app = VizApp(db_path)
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            lv = app.query_one("#session-list", viz.ListView)
            item = list(lv.query(viz.ListItem))[0]
            # The Static inside the ListItem should contain the agent name
            static = item.query_one(viz.Static)
            assert "orchestrator" in str(static.content)
    finally:
        os.unlink(db_path)


@pytest.mark.anyio
async def test_content_pane_exists() -> None:
    """ContentPane should be present in the composed layout."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        _make_db(db_path)
        app = VizApp(db_path)
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await pilot.pause(0.2)
            pane = app.query_one("#content-pane", ContentPane)
            assert pane is not None
    finally:
        os.unlink(db_path)


@pytest.mark.anyio
async def test_content_pane_shows_full_message() -> None:
    """Highlighting a message node must show the full, untruncated content."""
    long_msg = (
        "Please do the thing. This is a long message that would previously "
        "have been truncated at 72 characters."
    )
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        _make_db(db_path)
        app = VizApp(db_path)
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await pilot.pause(0.2)

            # Expand the root session in the tree
            tree = app.query_one("#call-tree", viz.Tree)
            tree.root.expand()
            await pilot.pause(0.1)

            # Expand the Messages group node (first child after task/result)
            for node in tree.root.children:
                if node.data and node.data.get("type") == "messages":
                    node.expand()
                    await pilot.pause(0.1)
                    # Highlight the first message leaf
                    for msg_node in node.children:
                        app.query_one("#call-tree", viz.Tree).post_message(
                            viz.Tree.NodeHighlighted(msg_node)
                        )
                        await pilot.pause(0.1)
                        break
                    break

            pane = app.query_one("#content-pane", ContentPane)
            detail = pane.query_one("#detail-body", viz.Static)
            assert long_msg in str(detail.content), (
                f"Expected full message in pane, got: {detail.content!r}"
            )
    finally:
        os.unlink(db_path)
