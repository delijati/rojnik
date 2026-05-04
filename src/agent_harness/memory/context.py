"""
Context window builder.

Loads a session's message history from the database and trims it to fit
within a token budget before passing it to the LLM.

Strategy
--------
- The system message is always kept (it's never trimmed).
- Messages are trimmed oldest-first (after the system message) until the
  total estimated token count fits within the budget.
- Token counts are estimated using tiktoken for accuracy.

Usage
-----
    builder = ContextBuilder(token_budget=100_000, model="gpt-4o")
    window = await builder.build(session_id, store, extra_messages=[user_msg])
"""


from loguru import logger

import tiktoken

from agent_harness.config import settings
from agent_harness.llm.schemas import Message, SystemMessage
from agent_harness.memory.store import MemoryStore


def _count_tokens(text: str, encoder: tiktoken.Encoding) -> int:
    """Encode *text* and return the number of tokens."""
    try:
        return len(encoder.encode(text))
    except Exception:
        # Fallback: rough approximation (4 chars ≈ 1 token)
        return max(1, len(text) // 4)


def _message_token_count(msg: Message, encoder: tiktoken.Encoding) -> int:
    """Estimate the token cost of a single message (content + overhead)."""
    overhead = 4  # per-message overhead in OpenAI's token counting
    content = ""
    if hasattr(msg, "content") and msg.content:
        content += msg.content
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            content += tc.name + tc.arguments
    return overhead + _count_tokens(content, encoder)


class ContextBuilder:
    """Builds a token-trimmed list of messages for the LLM call."""

    def __init__(
        self,
        token_budget: int | None = None,
        model: str | None = None,
    ) -> None:
        self._budget = token_budget or settings.context_token_budget
        model_name = model or settings.model
        # tiktoken encoding: use the model's real encoding, fall back to cl100k
        try:
            self._encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self._encoder = tiktoken.get_encoding("cl100k_base")

    async def build(
        self,
        session_id: str,
        store: MemoryStore,
        extra_messages: list[Message] | None = None,
    ) -> list[Message]:
        """
        Load history from *store*, append *extra_messages*, then trim to budget.

        Parameters
        ----------
        session_id:
            The session whose history to load.
        store:
            MemoryStore instance.
        extra_messages:
            New messages (e.g. the current user turn) to append *before* trimming.
            These are always included at the end.

        Returns
        -------
        A list of Message objects ready to pass to LLMClient.chat().
        """
        history = await store.get_messages(session_id)
        all_messages = history + (extra_messages or [])

        if not all_messages:
            return []

        # Separate system message (always kept)
        system_msgs: list[Message] = []
        non_system: list[Message] = []
        for msg in all_messages:
            if isinstance(msg, SystemMessage):
                system_msgs.append(msg)
            else:
                non_system.append(msg)

        system_tokens = sum(_message_token_count(m, self._encoder) for m in system_msgs)
        remaining_budget = self._budget - system_tokens

        # Greedily include messages newest-first, then reverse.
        kept: list[Message] = []
        tokens_used = 0
        trimmed = 0

        for msg in reversed(non_system):
            cost = _message_token_count(msg, self._encoder)
            if tokens_used + cost <= remaining_budget:
                kept.append(msg)
                tokens_used += cost
            else:
                trimmed += 1

        if trimmed:
            logger.debug(
                "context.trimmed",
                session_id=session_id,
                trimmed_messages=trimmed,
                kept_messages=len(kept),
                token_budget=self._budget,
                tokens_used=system_tokens + tokens_used,
            )

        kept.reverse()  # back to chronological order
        return system_msgs + kept
