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


import tiktoken
from loguru import logger

from rojnik.config import settings
from rojnik.llm.schemas import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from rojnik.memory.store import MemoryStore


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


class ContextBudgetError(ValueError):
    """Raised when mandatory system and user messages exceed the token budget."""


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

        # Tool calls and their results must stay together; providers reject
        # orphaned tool results or assistant calls with missing responses.
        groups: list[list[Message]] = []
        index = 0
        while index < len(non_system):
            msg = non_system[index]
            group = [msg]
            index += 1
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                call_ids = {call.id for call in msg.tool_calls}
                while index < len(non_system):
                    result = non_system[index]
                    if not isinstance(result, ToolResultMessage):
                        break
                    if result.tool_call_id not in call_ids:
                        break
                    group.append(result)
                    index += 1
            groups.append(group)

        latest_user_group: int | None = None
        for group_index in range(len(groups) - 1, -1, -1):
            if any(isinstance(msg, UserMessage) for msg in groups[group_index]):
                latest_user_group = group_index
                break

        system_tokens = sum(_message_token_count(m, self._encoder) for m in system_msgs)
        mandatory_tokens = system_tokens
        if latest_user_group is not None:
            mandatory_tokens += sum(
                _message_token_count(msg, self._encoder) for msg in groups[latest_user_group]
            )
        if mandatory_tokens > self._budget:
            raise ContextBudgetError(
                "System instructions and the latest user message require "
                f"{mandatory_tokens} tokens, exceeding the {self._budget}-token context budget."
            )
        remaining_budget = self._budget - system_tokens

        # Greedily include complete groups newest-first, then reverse.
        kept_group_indexes: set[int] = set()
        tokens_used = 0
        if latest_user_group is not None:
            kept_group_indexes.add(latest_user_group)
            tokens_used = sum(
                _message_token_count(msg, self._encoder) for msg in groups[latest_user_group]
            )
        trimmed = 0

        for group_index in range(len(groups) - 1, -1, -1):
            if group_index == latest_user_group:
                continue
            group = groups[group_index]
            cost = sum(_message_token_count(msg, self._encoder) for msg in group)
            if tokens_used + cost <= remaining_budget:
                kept_group_indexes.add(group_index)
                tokens_used += cost
            else:
                trimmed += len(group)

        if trimmed:
            logger.debug(
                "context.trimmed",
                session_id=session_id,
                trimmed_messages=trimmed,
                kept_messages=sum(len(groups[index]) for index in kept_group_indexes),
                token_budget=self._budget,
                tokens_used=system_tokens + tokens_used,
            )

        kept = [msg for index, group in enumerate(groups) if index in kept_group_indexes for msg in group]
        return system_msgs + kept
