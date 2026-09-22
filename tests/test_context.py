"""Tests for memory/context.py — ContextBuilder token-budget trimming."""

import pytest

from rojnik.llm.schemas import (
    AssistantMessage,
    SystemMessage,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
)
from rojnik.memory.context import ContextBudgetError, ContextBuilder


class TestContextBuilder:
    async def test_empty_session_returns_empty(self, store):
        sid = await store.create_session("a")
        builder = ContextBuilder(token_budget=1000)
        window = await builder.build(sid, store)
        assert window == []

    async def test_messages_returned_in_order(self, store):
        sid = await store.create_session("a")
        await store.add_message(sid, UserMessage(content="first"))
        await store.add_message(sid, AssistantMessage(content="second"))
        builder = ContextBuilder(token_budget=10_000)
        window = await builder.build(sid, store)
        assert [m.role for m in window] == ["user", "assistant"]

    async def test_system_message_always_kept(self, store):
        """System message must survive even when budget is very tight."""
        sid = await store.create_session("a")
        await store.add_message(sid, SystemMessage(content="Be helpful."))
        # Add many user messages that should get trimmed
        for i in range(20):
            await store.add_message(sid, UserMessage(content=f"message number {i} " * 20))

        builder = ContextBuilder(token_budget=50)  # very tight
        with pytest.raises(ContextBudgetError):
            await builder.build(sid, store)

    async def test_oldest_messages_trimmed_first(self, store):
        """With a tight budget, newer messages should be kept over older ones."""
        sid = await store.create_session("a")
        await store.add_message(sid, UserMessage(content="old " * 100))
        await store.add_message(sid, UserMessage(content="new"))

        builder = ContextBuilder(token_budget=30)
        window = await builder.build(sid, store)

        contents = [m.content for m in window if hasattr(m, "content")]
        # "new" should be present; the large old message should be trimmed
        assert "new" in contents

    async def test_extra_messages_appended(self, store):
        sid = await store.create_session("a")
        await store.add_message(sid, UserMessage(content="stored"))

        extra = UserMessage(content="injected")
        builder = ContextBuilder(token_budget=10_000)
        window = await builder.build(sid, store, extra_messages=[extra])

        contents = [m.content for m in window if hasattr(m, "content")]
        assert "stored" in contents
        assert "injected" in contents
        # Extra messages come after stored history
        assert contents.index("injected") > contents.index("stored")

    async def test_fits_within_budget(self, store):
        """Total token count of the window must not exceed the budget."""
        import tiktoken

        sid = await store.create_session("a")
        for i in range(50):
            await store.add_message(sid, UserMessage(content=f"message {i}"))

        budget = 200
        builder = ContextBuilder(token_budget=budget)
        window = await builder.build(sid, store)

        enc = tiktoken.get_encoding("cl100k_base")
        total = sum(
            4 + len(enc.encode(m.content or ""))
            for m in window
            if hasattr(m, "content") and m.content
        )
        assert total <= budget + 50  # small slack for overhead estimation

    async def test_tool_call_and_results_are_trimmed_as_one_group(self, store):
        sid = await store.create_session("a")
        call = ToolCallPart(id="call-1", name="large_tool", arguments="{}")
        await store.add_message(sid, AssistantMessage(tool_calls=[call]))
        await store.add_message(
            sid,
            ToolResultMessage(tool_call_id="call-1", content="large result " * 100),
        )
        await store.add_message(sid, UserMessage(content="new request"))

        window = await ContextBuilder(token_budget=30).build(sid, store)

        assert [message.role for message in window] == ["user"]

    async def test_complete_tool_call_group_is_kept(self, store):
        sid = await store.create_session("a")
        calls = [
            ToolCallPart(id="call-1", name="first", arguments="{}"),
            ToolCallPart(id="call-2", name="second", arguments="{}"),
        ]
        await store.add_message(sid, AssistantMessage(tool_calls=calls))
        await store.add_message(sid, ToolResultMessage(tool_call_id="call-1", content="one"))
        await store.add_message(sid, ToolResultMessage(tool_call_id="call-2", content="two"))

        window = await ContextBuilder(token_budget=1000).build(sid, store)

        assert [message.role for message in window] == ["assistant", "tool", "tool"]

    async def test_latest_user_message_is_mandatory(self, store):
        sid = await store.create_session("a")
        await store.add_message(sid, UserMessage(content="old " * 100))
        await store.add_message(sid, UserMessage(content="latest"))

        window = await ContextBuilder(token_budget=20).build(sid, store)

        assert [message.content for message in window] == ["latest"]

    async def test_oversized_system_and_latest_user_raise(self, store):
        sid = await store.create_session("a")
        await store.add_message(sid, SystemMessage(content="instructions " * 100))
        await store.add_message(sid, UserMessage(content="current task"))

        with pytest.raises(ContextBudgetError, match="latest user message"):
            await ContextBuilder(token_budget=20).build(sid, store)
