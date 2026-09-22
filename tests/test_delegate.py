"""Tests for the multi-agent delegate tool and AgentRegistry."""


import pytest

from rojnik.agent.agent import Agent, AgentRegistry
from rojnik.tools.builtins.delegate import current_session_id, make_delegate_tool
from rojnik.tools.registry import ToolRegistry
from tests.conftest import MockLLMClient, stop_response, tool_call_response

# ---------------------------------------------------------------------------
# AgentRegistry
# ---------------------------------------------------------------------------

class TestAgentRegistry:
    def _agent(self, name="test"):
        return Agent(
            name=name,
            system_prompt="test",
            llm=MockLLMClient([]),
        )

    def test_register_and_contains(self):
        reg = AgentRegistry()
        a = self._agent("alpha")
        reg.register(a)
        assert "alpha" in reg

    def test_get_returns_agent(self):
        reg = AgentRegistry()
        a = self._agent("alpha")
        reg.register(a)
        assert reg.get("alpha") is a

    def test_get_unknown_raises_keyerror(self):
        reg = AgentRegistry()
        with pytest.raises(KeyError):
            reg.get("ghost")

    def test_register_many(self):
        reg = AgentRegistry()
        reg.register_many(self._agent("a"), self._agent("b"), self._agent("c"))
        assert len(reg) == 3

    def test_names_returns_all(self):
        reg = AgentRegistry()
        reg.register_many(self._agent("x"), self._agent("y"))
        assert set(reg.names()) == {"x", "y"}


# ---------------------------------------------------------------------------
# make_delegate_tool — schema and metadata
# ---------------------------------------------------------------------------

class TestMakeDelegateTool:
    def test_returns_tool_decorated_function(self):
        reg = AgentRegistry()
        delegate = make_delegate_tool(reg)
        assert getattr(delegate, "is_tool", False) is True

    def test_schema_name_is_delegate_to_agent(self):
        reg = AgentRegistry()
        delegate = make_delegate_tool(reg)
        assert delegate.tool_schema.function.name == "delegate_to_agent"

    def test_schema_parameters_include_agent_name_and_task(self):
        reg = AgentRegistry()
        delegate = make_delegate_tool(reg)
        props = delegate.tool_schema.function.parameters.properties
        assert "agent_name" in props
        assert "task" in props

    def test_description_lists_agents(self):
        reg = AgentRegistry()
        reg.register(Agent(name="coder", system_prompt="x", llm=MockLLMClient([])))
        delegate = make_delegate_tool(reg)
        assert "coder" in delegate.tool_schema.function.description


class TestAgentAsTool:
    def test_schema_uses_agent_name(self):
        agent = Agent(name="file_reader", system_prompt="Read files", llm=MockLLMClient([]))
        agent_tool = agent.as_tool()

        assert agent_tool.tool_schema.function.name == "file_reader"
        assert agent_tool.tool_schema.function.parameters.required == ["task"]

    def test_custom_name_and_description(self):
        agent = Agent(name="reader", system_prompt="Read files", llm=MockLLMClient([]))
        agent_tool = agent.as_tool(name="inspect_files", description="Inspect project files")

        assert agent_tool.tool_schema.function.name == "inspect_files"
        assert agent_tool.tool_schema.function.description == "Inspect project files"

    def test_invalid_name_raises(self):
        agent = Agent(name="file reader", system_prompt="Read files", llm=MockLLMClient([]))

        with pytest.raises(ValueError, match="tool names"):
            agent.as_tool()

    async def test_execution_preserves_parent_session(self, store):
        received_parent: list[str | None] = []

        class RecordingAgent(Agent):
            async def run(self, task: str, parent_session_id=None, **kwargs) -> str:
                received_parent.append(parent_session_id)
                return f"result: {task}"

        agent = RecordingAgent(
            name="specialist",
            system_prompt="Specialist",
            llm=MockLLMClient([]),
        )
        agent_tool = agent.as_tool()
        token = current_session_id.set("parent-session")
        try:
            result = await agent_tool(task="do work")
        finally:
            current_session_id.reset(token)

        assert result == "result: do work"
        assert received_parent == ["parent-session"]


# ---------------------------------------------------------------------------
# make_delegate_tool — execution
# ---------------------------------------------------------------------------

class TestDelegateExecution:
    async def test_delegates_to_correct_agent(self, store):
        """Calling delegate should invoke the target agent's run()."""
        called_with: list[str] = []

        class RecordingAgent(Agent):
            async def run(self, task: str, parent_session_id=None) -> str:
                called_with.append(task)
                return f"result: {task}"

        target = RecordingAgent(
            name="specialist",
            system_prompt="I specialise.",
            llm=MockLLMClient([]),
        )
        reg = AgentRegistry()
        reg.register(target)
        delegate = make_delegate_tool(reg)

        result = await delegate(agent_name="specialist", task="do something")
        assert result == "result: do something"
        assert called_with == ["do something"]

    async def test_unknown_agent_returns_error_string(self):
        reg = AgentRegistry()
        delegate = make_delegate_tool(reg)
        result = await delegate(agent_name="ghost", task="hello")
        assert result.startswith("ERROR:")
        assert "ghost" in result

    async def test_parent_session_id_passed_through(self, store):
        """delegate_to_agent should pass the current_session_id as parent_session_id."""
        received_parent: list[str | None] = []

        class RecordingAgent(Agent):
            async def run(self, task: str, parent_session_id=None) -> str:
                received_parent.append(parent_session_id)
                return "ok"

        target = RecordingAgent(
            name="child",
            system_prompt="child",
            llm=MockLLMClient([]),
        )
        reg = AgentRegistry()
        reg.register(target)
        delegate = make_delegate_tool(reg)

        # Simulate what run_loop does — set the context var
        token = current_session_id.set("parent-session-123")
        try:
            await delegate(agent_name="child", task="a task")
        finally:
            current_session_id.reset(token)

        assert received_parent == ["parent-session-123"]


# ---------------------------------------------------------------------------
# End-to-end: orchestrator delegates to a subagent via the loop
# ---------------------------------------------------------------------------

class TestOrchestratorLoop:
    async def test_orchestrator_calls_subagent_and_gets_result(self, store):
        """Full loop with a delegate call: orchestrator → subagent → answer."""
        from rojnik.agent.loop import run_loop
        from rojnik.agent.state import RunState
        from rojnik.llm.schemas import SystemMessage
        from rojnik.memory.context import ContextBuilder

        # The subagent always returns a fixed answer
        subagent_llm = MockLLMClient([stop_response("I found 42 files.")])
        subagent = Agent(
            name="file_reader",
            system_prompt="I read files.",
            llm=subagent_llm,
        )

        registry = AgentRegistry()
        registry.register(subagent)
        delegate = make_delegate_tool(registry)

        orchestrator_tool_registry = ToolRegistry()
        orchestrator_tool_registry.register(delegate)

        # Orchestrator: first response delegates, second response is final
        orchestrator_llm = MockLLMClient([
            tool_call_response([("c1", "delegate_to_agent",
                                 '{"agent_name":"file_reader","task":"count files"}')]),
            stop_response("The subagent found 42 files."),
        ])

        # Create orchestrator session manually
        orch_sid = await store.create_session("orchestrator", task="count files")
        await store.add_message(orch_sid, SystemMessage(content="You coordinate agents."))
        orch_state = RunState(session_id=orch_sid, agent_name="orchestrator")

        result = await run_loop(
            state=orch_state,
            task="How many files are there?",
            llm=orchestrator_llm,
            tool_registry=orchestrator_tool_registry,
            store=store,
            context_builder=ContextBuilder(),
        )

        assert "42" in result
        assert len(orchestrator_llm.calls) == 2
        assert len(subagent_llm.calls) == 1

    async def test_orchestrator_calls_agent_as_tool(self, store):
        subagent_llm = MockLLMClient([stop_response("Direct tool result")])
        subagent = Agent(
            name="specialist",
            system_prompt="Do specialist work.",
            llm=subagent_llm,
        )
        orchestrator_llm = MockLLMClient([
            tool_call_response([("c1", "specialist", '{"task":"do work"}')]),
            stop_response("Used the specialist result."),
        ])
        orchestrator = Agent(
            name="orchestrator",
            system_prompt="Call the specialist.",
            tools=[subagent.as_tool()],
            llm=orchestrator_llm,
        )

        result = await orchestrator.run("delegate this")

        assert result == "Used the specialist result."
        assert len(subagent_llm.calls) == 1

        from sqlalchemy import select

        from rojnik.memory.models import Session

        async with store._session_factory() as db:
            rows = (await db.execute(select(Session))).scalars().all()
        parent = next(row for row in rows if row.agent_name == "orchestrator")
        child = next(row for row in rows if row.agent_name == "specialist")
        assert child.parent_session_id == parent.id
