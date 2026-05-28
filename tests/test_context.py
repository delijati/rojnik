"""Tests for memory/context.py — ContextBuilder token-budget trimming."""


import pytest

from rojnik.llm.schemas import AssistantMessage, SystemMessage, UserMessage
from rojnik.memory.context import ContextBuilder


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
        window = await builder.build(sid, store)

        assert len(window) >= 1
        assert window[0].role == "system"

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
