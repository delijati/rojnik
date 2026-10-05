"""Tests for Agent lifecycle behavior."""

import asyncio

import pytest
from sqlalchemy import select

from rojnik.agent.agent import Agent
from rojnik.llm.schemas import SystemMessage
from rojnik.memory.models import Session
from rojnik.tools.base import tool
from tests.conftest import MockLLMClient, stop_response, tool_call_response


async def _only_session(store) -> Session:
    async with store._session_factory() as db:
        rows = (await db.execute(select(Session))).scalars().all()
    assert len(rows) == 1
    return rows[0]


class TestAgentLifecycle:
    async def test_llm_error_closes_session(self, store):
        agent = Agent(
            name="failing",
            system_prompt="Fail during the LLM call.",
            llm=MockLLMClient([]),
        )

        with pytest.raises(RuntimeError, match="no more scripted responses"):
            await agent.run("fail")

        session = await _only_session(store)
        assert session.status == "error"
        assert session.finished_at is not None

    async def test_cancellation_closes_session(self, store):
        started = asyncio.Event()

        class BlockingLLM:
            async def chat(self, *args, **kwargs):
                started.set()
                await asyncio.Event().wait()

        agent = Agent(
            name="cancelled",
            system_prompt="Wait forever.",
            llm=BlockingLLM(),  # type: ignore[arg-type]
        )
        task = asyncio.create_task(agent.run("wait"))
        await started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        session = await _only_session(store)
        assert session.status == "cancelled"
        assert session.finished_at is not None


class TestAgentEphemeralSessions:
    @staticmethod
    async def _trace_rows(store) -> list[str]:
        from rojnik.memory.models import DBMessage, ToolResult

        async with store._session_factory() as db:
            sessions = (await db.execute(select(Session.id))).scalars().all()
            messages = (await db.execute(select(DBMessage.id))).scalars().all()
            results = (await db.execute(select(ToolResult.id))).scalars().all()
        return [*sessions, *messages, *results]

    async def test_keep_session_false_discards_completed_run(self, store):
        @tool(description="Read a tank level.")
        async def tank_level(tank: str) -> str:
            return f"{tank}: 4000 l"

        llm = MockLLMClient(
            [
                tool_call_response([("tc1", "tank_level", '{"tank": "T-4"}')]),
                stop_response("T-4 hat 4000 l."),
            ]
        )
        agent = Agent(
            name="ephemeral",
            system_prompt="Business snapshot.",
            tools=[tank_level],
            llm=llm,
            keep_session=False,
        )

        assert await agent.run("Wie voll ist T-4?") == "T-4 hat 4000 l."
        assert await self._trace_rows(store) == []

    async def test_keep_session_false_discards_failed_run(self, store):
        agent = Agent(
            name="ephemeral",
            system_prompt="Fail during the LLM call.",
            llm=MockLLMClient([]),
            keep_session=False,
        )

        with pytest.raises(RuntimeError, match="no more scripted responses"):
            await agent.run("fail")

        assert await self._trace_rows(store) == []

    async def test_keep_session_false_discards_cancelled_run(self, store):
        started = asyncio.Event()

        class BlockingLLM:
            async def chat(self, *args, **kwargs):
                started.set()
                await asyncio.Event().wait()

        agent = Agent(
            name="ephemeral",
            system_prompt="Wait forever.",
            llm=BlockingLLM(),  # type: ignore[arg-type]
            keep_session=False,
        )
        task = asyncio.create_task(agent.run("wait"))
        await started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert await self._trace_rows(store) == []

    async def test_sessions_are_kept_by_default(self, store):
        agent = Agent(
            name="kept",
            system_prompt="Keep the trace.",
            llm=MockLLMClient([stop_response("done")]),
        )

        await agent.run("keep")

        session = await _only_session(store)
        assert session.status == "completed"


class TestAgentSkills:
    @staticmethod
    def _skill(tmp_path, *, name="testing"):
        directory = tmp_path / name
        directory.mkdir()
        path = directory / "SKILL.md"
        path.write_text(
            f"---\nname: {name}\ndescription: Test carefully.\n---\n\n"
            "# Workflow\n\nRun focused tests first.\n",
            encoding="utf-8",
        )
        return path

    async def test_eager_skill_is_in_system_prompt(self, store, tmp_path):
        llm = MockLLMClient([stop_response("done")])
        agent = Agent(
            name="tester",
            system_prompt="Base prompt.",
            skills=[self._skill(tmp_path)],
            llm=llm,
        )

        await agent.run("test this")

        system = llm.calls[0]["messages"][0]
        assert isinstance(system, SystemMessage)
        assert system.content.startswith("Base prompt.")
        assert "Run focused tests first." in system.content
        messages = await store.get_messages((await _only_session(store)).id)
        assert messages[0] == system

    async def test_on_demand_adds_catalog_and_tool(self, store, tmp_path):
        llm = MockLLMClient(
            [
                tool_call_response([("call-1", "load_skill", '{"name":"testing"}')]),
                stop_response("done"),
            ]
        )
        agent = Agent(
            name="tester",
            system_prompt="Base prompt.",
            skills=[self._skill(tmp_path)],
            skill_mode="on_demand",
            llm=llm,
        )

        await agent.run("test this")

        system = llm.calls[0]["messages"][0]
        assert "testing: Test carefully." in system.content
        assert "Run focused tests first." not in system.content
        assert "load_skill" in agent.tool_registry
        second_messages = llm.calls[1]["messages"]
        assert any(
            message.role == "tool" and "Run focused tests first." in message.content
            for message in second_messages
        )

    def test_on_demand_rejects_load_skill_conflict(self, tmp_path):
        @tool(description="Conflicting tool")
        async def load_skill(name: str) -> str:
            return name

        with pytest.raises(ValueError, match="reserved"):
            Agent(
                name="tester",
                system_prompt="Base prompt.",
                tools=[load_skill],
                skills=[self._skill(tmp_path)],
                skill_mode="on_demand",
                llm=MockLLMClient([]),
            )

    def test_no_skills_keeps_tools_unchanged(self):
        agent = Agent(
            name="tester",
            system_prompt="Base prompt.",
            skill_mode="on_demand",
            llm=MockLLMClient([]),
        )

        assert agent.skills == ()
        assert agent.tool_registry.names() == []
