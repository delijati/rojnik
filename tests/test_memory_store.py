"""Tests for memory/store.py — MemoryStore CRUD."""


import pytest

from agent_harness.llm.schemas import (
    AssistantMessage,
    SystemMessage,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
)
from agent_harness.memory.store import MemoryStore


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class TestSessions:
    async def test_create_session_returns_id(self, store):
        sid = await store.create_session("agent_a", task="do stuff")
        assert isinstance(sid, str) and len(sid) == 36  # UUID

    async def test_create_session_with_parent(self, store):
        parent = await store.create_session("parent_agent")
        child = await store.create_session("child_agent", parent_session_id=parent)
        # No exception means FK is valid
        assert child != parent

    async def test_close_session_updates_status(self, store):
        from sqlalchemy import select
        from agent_harness.memory.models import Session

        sid = await store.create_session("agent_a")
        await store.close_session(sid, status="completed", result="done", total_tokens=42)

        async with store._session_factory() as db:
            row = await db.get(Session, sid)

        assert row.status == "completed"
        assert row.total_tokens == 42
        assert row.finished_at is not None

    async def test_close_session_stores_result_preview(self, store):
        from sqlalchemy import select
        from agent_harness.memory.models import Session

        sid = await store.create_session("agent_a")
        await store.close_session(sid, result="the answer is 42")

        async with store._session_factory() as db:
            row = await db.get(Session, sid)

        assert "42" in row.result_preview


# ---------------------------------------------------------------------------
# Messages round-trip
# ---------------------------------------------------------------------------

class TestMessages:
    async def test_add_and_retrieve_user_message(self, store):
        sid = await store.create_session("a")
        await store.add_message(sid, UserMessage(content="hello"))
        msgs = await store.get_messages(sid)
        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert msgs[0].content == "hello"

    async def test_add_and_retrieve_system_message(self, store):
        sid = await store.create_session("a")
        await store.add_message(sid, SystemMessage(content="You are helpful."))
        msgs = await store.get_messages(sid)
        assert msgs[0].role == "system"

    async def test_add_and_retrieve_assistant_message(self, store):
        sid = await store.create_session("a")
        await store.add_message(sid, AssistantMessage(content="I can help."))
        msgs = await store.get_messages(sid)
        assert msgs[0].role == "assistant"
        assert msgs[0].content == "I can help."

    async def test_assistant_with_tool_calls_round_trips(self, store):
        sid = await store.create_session("a")
        tc = ToolCallPart(id="tc1", name="add", arguments='{"a":1,"b":2}')
        await store.add_message(sid, AssistantMessage(tool_calls=[tc]))
        msgs = await store.get_messages(sid)
        assert isinstance(msgs[0], AssistantMessage)
        assert msgs[0].tool_calls[0].name == "add"
        assert msgs[0].tool_calls[0].id == "tc1"

    async def test_tool_result_message_round_trips(self, store):
        sid = await store.create_session("a")
        await store.add_message(
            sid, ToolResultMessage(tool_call_id="tc1", content="42")
        )
        msgs = await store.get_messages(sid)
        assert isinstance(msgs[0], ToolResultMessage)
        assert msgs[0].tool_call_id == "tc1"
        assert msgs[0].content == "42"

    async def test_messages_returned_in_insertion_order(self, store):
        sid = await store.create_session("a")
        for content in ["first", "second", "third"]:
            await store.add_message(sid, UserMessage(content=content))
        msgs = await store.get_messages(sid)
        assert [m.content for m in msgs] == ["first", "second", "third"]

    async def test_empty_session_returns_empty_list(self, store):
        sid = await store.create_session("a")
        assert await store.get_messages(sid) == []

    async def test_sessions_are_isolated(self, store):
        s1 = await store.create_session("a")
        s2 = await store.create_session("b")
        await store.add_message(s1, UserMessage(content="only in s1"))
        assert await store.get_messages(s2) == []


# ---------------------------------------------------------------------------
# Tool results
# ---------------------------------------------------------------------------

class TestToolResults:
    async def test_save_tool_result_success(self, store):
        from sqlalchemy import select
        from agent_harness.memory.models import ToolResult

        sid = await store.create_session("a")
        await store.save_tool_result(
            session_id=sid,
            tool_call_id="tc1",
            tool_name="read_file",
            input_json='{"path":"/etc/hosts"}',
            output="127.0.0.1 localhost",
            duration_ms=5,
        )
        async with store._session_factory() as db:
            result = await db.execute(
                select(ToolResult).where(ToolResult.session_id == sid)
            )
            rows = result.scalars().all()

        assert len(rows) == 1
        assert rows[0].tool_name == "read_file"
        assert rows[0].output == "127.0.0.1 localhost"
        assert rows[0].error is None

    async def test_save_tool_result_with_error(self, store):
        from sqlalchemy import select
        from agent_harness.memory.models import ToolResult

        sid = await store.create_session("a")
        await store.save_tool_result(
            session_id=sid,
            tool_call_id="tc2",
            tool_name="shell_exec",
            error="Command timed out",
        )
        async with store._session_factory() as db:
            result = await db.execute(
                select(ToolResult).where(ToolResult.session_id == sid)
            )
            row = result.scalars().first()

        assert row.error == "Command timed out"
        assert row.output is None


# ---------------------------------------------------------------------------
# Singleton behaviour
# ---------------------------------------------------------------------------

class TestSingleton:
    async def test_two_gets_return_same_instance(self, tmp_path):
        db = tmp_path / "singleton.db"
        url = f"sqlite+aiosqlite:///{db}"
        a = await MemoryStore.get(db_url=url)
        b = await MemoryStore.get(db_url=url)
        assert a is b
