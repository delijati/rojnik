"""Tests for agent/loop.py — the ReAct loop (fully mocked LLM)."""


import pytest

from rojnik.agent.loop import MaxIterationsError, run_loop
from rojnik.agent.state import RunState
from rojnik.memory.context import ContextBuilder
from rojnik.tools.base import tool
from rojnik.tools.registry import ToolRegistry
from tests.conftest import MockLLMClient, stop_response, tool_call_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _make_state(store, agent_name="test_agent"):
    sid = await store.create_session(agent_name, task="test task")
    from rojnik.llm.schemas import SystemMessage
    await store.add_message(sid, SystemMessage(content="You are a test agent."))
    return RunState(session_id=sid, agent_name=agent_name)


def _registry_with(*tools_list):
    r = ToolRegistry()
    for t in tools_list:
        r.register(t)
    return r


# ---------------------------------------------------------------------------
# Happy path: stop on first response
# ---------------------------------------------------------------------------

class TestLoopStop:
    async def test_returns_content_on_stop(self, store):
        llm = MockLLMClient([stop_response("The answer is 42.")])
        state = await _make_state(store)
        result = await run_loop(
            state=state,
            task="What is 6*7?",
            llm=llm,
            tool_registry=ToolRegistry(),
            store=store,
            context_builder=ContextBuilder(),
        )
        assert result == "The answer is 42."

    async def test_state_status_is_completed(self, store):
        llm = MockLLMClient([stop_response("done")])
        state = await _make_state(store)
        await run_loop(
            state=state, task="x", llm=llm,
            tool_registry=ToolRegistry(), store=store,
            context_builder=ContextBuilder(),
        )
        assert state.status == "completed"

    async def test_tokens_accumulated(self, store):
        llm = MockLLMClient([stop_response(tokens=20)])
        state = await _make_state(store)
        await run_loop(
            state=state, task="x", llm=llm,
            tool_registry=ToolRegistry(), store=store,
            context_builder=ContextBuilder(),
        )
        assert state.total_tokens == 40  # 20 prompt + 20 completion

    async def test_only_one_llm_call_on_immediate_stop(self, store):
        llm = MockLLMClient([stop_response()])
        state = await _make_state(store)
        await run_loop(
            state=state, task="x", llm=llm,
            tool_registry=ToolRegistry(), store=store,
            context_builder=ContextBuilder(),
        )
        assert len(llm.calls) == 1

    async def test_response_persisted_in_store(self, store):
        llm = MockLLMClient([stop_response("persisted answer")])
        state = await _make_state(store)
        await run_loop(
            state=state, task="x", llm=llm,
            tool_registry=ToolRegistry(), store=store,
            context_builder=ContextBuilder(),
        )
        msgs = await store.get_messages(state.session_id)
        roles = [m.role for m in msgs]
        assert "assistant" in roles


# ---------------------------------------------------------------------------
# Tool call → stop (two-iteration loop)
# ---------------------------------------------------------------------------

class TestLoopWithTools:
    async def test_tool_call_then_stop(self, store):
        @tool(description="add")
        async def add(a: int, b: int) -> int:
            return a + b

        llm = MockLLMClient([
            tool_call_response([("c1", "add", '{"a":3,"b":4}')]),
            stop_response("The sum is 7."),
        ])
        state = await _make_state(store)
        result = await run_loop(
            state=state, task="add 3 and 4",
            llm=llm, tool_registry=_registry_with(add),
            store=store, context_builder=ContextBuilder(),
        )
        assert result == "The sum is 7."
        assert len(llm.calls) == 2

    async def test_tool_result_message_persisted(self, store):
        @tool(description="const")
        async def const() -> str:
            return "hello"

        llm = MockLLMClient([
            tool_call_response([("c1", "const", "{}")]),
            stop_response("done"),
        ])
        state = await _make_state(store)
        await run_loop(
            state=state, task="x",
            llm=llm, tool_registry=_registry_with(const),
            store=store, context_builder=ContextBuilder(),
        )
        msgs = await store.get_messages(state.session_id)
        roles = [m.role for m in msgs]
        assert "tool" in roles

    async def test_parallel_tool_calls_both_executed(self, store):
        """Two tool calls in one response should both be dispatched."""
        call_log: list[str] = []

        @tool(description="log a")
        async def log_a() -> str:
            call_log.append("a")
            return "a_result"

        @tool(description="log b")
        async def log_b() -> str:
            call_log.append("b")
            return "b_result"

        llm = MockLLMClient([
            tool_call_response([
                ("c1", "log_a", "{}"),
                ("c2", "log_b", "{}"),
            ]),
            stop_response("both done"),
        ])
        state = await _make_state(store)
        await run_loop(
            state=state, task="run both",
            llm=llm, tool_registry=_registry_with(log_a, log_b),
            store=store, context_builder=ContextBuilder(),
        )
        assert "a" in call_log
        assert "b" in call_log

    async def test_tool_error_does_not_crash_loop(self, store):
        """A tool that raises should return an ERROR: string, not crash the loop."""
        @tool(description="boom")
        async def boom() -> str:
            raise RuntimeError("deliberate")

        llm = MockLLMClient([
            tool_call_response([("c1", "boom", "{}")]),
            stop_response("handled it"),
        ])
        state = await _make_state(store)
        result = await run_loop(
            state=state, task="x",
            llm=llm, tool_registry=_registry_with(boom),
            store=store, context_builder=ContextBuilder(),
        )
        assert result == "handled it"

        # Verify the error was persisted in tool_results
        from sqlalchemy import select

        from rojnik.memory.models import ToolResult
        async with store._session_factory() as db:
            res = await db.execute(
                select(ToolResult).where(ToolResult.session_id == state.session_id)
            )
            row = res.scalars().first()
        assert row is not None
        assert row.error is not None


# ---------------------------------------------------------------------------
# Max iterations
# ---------------------------------------------------------------------------

class TestMaxIterations:
    async def test_raises_after_limit(self, store):
        @tool(description="loop forever")
        async def noop() -> str:
            return "still going"

        # Always return a tool call — loop can never stop
        responses = [
            tool_call_response([("c1", "noop", "{}")]) for _ in range(10)
        ]
        llm = MockLLMClient(responses)
        state = await _make_state(store)

        with pytest.raises(MaxIterationsError):
            await run_loop(
                state=state, task="x",
                llm=llm, tool_registry=_registry_with(noop),
                store=store, context_builder=ContextBuilder(),
                max_iterations=3,
            )

    async def test_session_status_is_max_iterations(self, store):
        @tool(description="noop")
        async def noop() -> str:
            return "x"

        responses = [tool_call_response([("c1", "noop", "{}")]) for _ in range(5)]
        llm = MockLLMClient(responses)
        state = await _make_state(store)

        with pytest.raises(MaxIterationsError):
            await run_loop(
                state=state, task="x",
                llm=llm, tool_registry=_registry_with(noop),
                store=store, context_builder=ContextBuilder(),
                max_iterations=2,
            )

        assert state.status == "max_iterations"

    async def test_exact_limit_executes_all_iterations(self, store):
        """A loop with max_iterations=N should make exactly N LLM calls before raising."""
        @tool(description="noop")
        async def noop() -> str:
            return "x"

        n = 4
        responses = [tool_call_response([("c1", "noop", "{}")]) for _ in range(n + 1)]
        llm = MockLLMClient(responses)
        state = await _make_state(store)

        with pytest.raises(MaxIterationsError):
            await run_loop(
                state=state, task="x",
                llm=llm, tool_registry=_registry_with(noop),
                store=store, context_builder=ContextBuilder(),
                max_iterations=n,
            )

        assert len(llm.calls) == n


# ---------------------------------------------------------------------------
# Unexpected finish reason
# ---------------------------------------------------------------------------

class TestUnexpectedFinish:
    async def test_length_finish_reason_raises(self, store):
        from rojnik.llm.schemas import LLMResponse
        llm = MockLLMClient([
            LLMResponse(finish_reason="length", content="truncated text", model="mock")
        ])
        state = await _make_state(store)
        with pytest.raises(RuntimeError, match="token limit reached"):
            await run_loop(
                state=state, task="x",
                llm=llm, tool_registry=ToolRegistry(),
                store=store, context_builder=ContextBuilder(),
            )
        assert state.status == "error"


# ---------------------------------------------------------------------------
# Streaming (on_chunk)
# ---------------------------------------------------------------------------

class TestStreaming:
    async def test_on_chunk_receives_all_content(self, store):
        """Chunks forwarded by on_chunk must concatenate to the full response."""
        llm = MockLLMClient([stop_response("Hello world")])
        state = await _make_state(store)
        chunks: list[str] = []
        result = await run_loop(
            state=state,
            task="say hello",
            llm=llm,
            tool_registry=ToolRegistry(),
            store=store,
            context_builder=ContextBuilder(),
            on_chunk=chunks.append,
        )
        assert result == "Hello world"
        assert "".join(chunks) == "Hello world"

    async def test_on_chunk_not_called_for_tool_call_turn(self, store):
        """on_chunk is only called when the LLM produces text, not on tool-call turns."""
        @tool(description="const")
        async def const() -> str:
            return "result"

        chunks: list[str] = []
        llm = MockLLMClient([
            tool_call_response([("c1", "const", "{}")]),  # no content → no on_chunk
            stop_response("done"),
        ])
        state = await _make_state(store)
        await run_loop(
            state=state,
            task="x",
            llm=llm,
            tool_registry=_registry_with(const),
            store=store,
            context_builder=ContextBuilder(),
            on_chunk=chunks.append,
        )
        # Only the final "stop" response ("done") should produce chunks
        assert "".join(chunks) == "done"

    async def test_on_chunk_none_does_not_break_normal_flow(self, store):
        """Passing on_chunk=None (the default) must behave identically to before."""
        llm = MockLLMClient([stop_response("ok")])
        state = await _make_state(store)
        result = await run_loop(
            state=state,
            task="x",
            llm=llm,
            tool_registry=ToolRegistry(),
            store=store,
            context_builder=ContextBuilder(),
            on_chunk=None,
        )
        assert result == "ok"


# ---------------------------------------------------------------------------
# Structured output (response_format)
# ---------------------------------------------------------------------------

class TestResponseFormat:
    async def test_response_format_forwarded_to_llm(self, store):
        """response_format must appear verbatim in the recorded LLM call."""
        from pydantic import BaseModel as BM

        class MyOutput(BM):
            answer: str

        llm = MockLLMClient([stop_response('{"answer": "42"}')])
        state = await _make_state(store)
        await run_loop(
            state=state,
            task="x",
            llm=llm,
            tool_registry=ToolRegistry(),
            store=store,
            context_builder=ContextBuilder(),
            response_format=MyOutput,
        )
        assert llm.calls[0]["response_format"] is MyOutput

    async def test_response_format_none_by_default(self, store):
        """When response_format is omitted the LLM call must receive None."""
        llm = MockLLMClient([stop_response("x")])
        state = await _make_state(store)
        await run_loop(
            state=state,
            task="x",
            llm=llm,
            tool_registry=ToolRegistry(),
            store=store,
            context_builder=ContextBuilder(),
        )
        assert llm.calls[0]["response_format"] is None


# ---------------------------------------------------------------------------
# on_tool_call callback
# ---------------------------------------------------------------------------

class TestOnToolCall:
    async def test_callback_fires_for_each_tool_call(self, store):
        """on_tool_call must fire once per tool call with the correct name and args."""
        @tool(description="add")
        async def add(a: int, b: int) -> int:
            return a + b

        @tool(description="const")
        async def const() -> str:
            return "hello"

        fired: list[tuple[str, str]] = []
        llm = MockLLMClient([
            tool_call_response([
                ("c1", "add", '{"a":1,"b":2}'),
                ("c2", "const", "{}"),
            ]),
            stop_response("done"),
        ])
        state = await _make_state(store)
        await run_loop(
            state=state,
            task="x",
            llm=llm,
            tool_registry=_registry_with(add, const),
            store=store,
            context_builder=ContextBuilder(),
            on_tool_call=lambda name, args: fired.append((name, args)),
        )
        assert len(fired) == 2
        assert fired[0] == ("add", '{"a":1,"b":2}')
        assert fired[1] == ("const", "{}")

    async def test_callback_none_does_not_break_loop(self, store):
        """Passing on_tool_call=None (the default) must not affect loop behaviour."""
        @tool(description="noop")
        async def noop() -> str:
            return "x"

        llm = MockLLMClient([
            tool_call_response([("c1", "noop", "{}")]),
            stop_response("ok"),
        ])
        state = await _make_state(store)
        result = await run_loop(
            state=state,
            task="x",
            llm=llm,
            tool_registry=_registry_with(noop),
            store=store,
            context_builder=ContextBuilder(),
            on_tool_call=None,
        )
        assert result == "ok"
