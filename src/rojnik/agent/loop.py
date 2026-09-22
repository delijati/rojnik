"""
The core ReAct agent loop.

This is a standalone async function shared by every Agent instance.
It is purposely decoupled from the Agent class so it's easy to test,
override, or replace with a different loop strategy.

Loop flow
---------
1.  Build the context window from memory + new user message.
2.  Call the LLM with the agent's registered tools.
3a. finish_reason == "stop"   → persist, return the content.
3b. finish_reason == "tool_calls" →
        - dispatch ALL tool calls concurrently (asyncio.gather)
        - persist tool results
        - append assistant + tool result messages
        - loop back to step 2
4.  If iteration >= max_iterations → persist with status "max_iterations", raise.
"""

import asyncio
import time
from typing import TYPE_CHECKING, Callable

from loguru import logger

from rojnik.agent.state import RunState
from rojnik.config import settings
from rojnik.llm.schemas import (
    AssistantMessage,
    ResponseFormat,
    ToolResultMessage,
    UserMessage,
)
from rojnik.memory.context import ContextBuilder
from rojnik.memory.store import MemoryStore
from rojnik.tools.builtins.delegate import current_session_id

if TYPE_CHECKING:
    from rojnik.llm.client import LLMClient
    from rojnik.tools.registry import ToolRegistry


class MaxIterationsError(Exception):
    """Raised when the agent loop exceeds the configured iteration limit."""


async def run_loop(
    *,
    state: RunState,
    task: str,
    llm: "LLMClient",
    tool_registry: "ToolRegistry",
    store: MemoryStore,
    context_builder: ContextBuilder,
    max_iterations: int | None = None,
    on_chunk: Callable[[str], None] | None = None,
    on_tool_call: Callable[[str, str], None] | None = None,
    response_format: ResponseFormat = None,
) -> str:
    """
    Execute the ReAct loop for one agent run.

    Parameters
    ----------
    state:           Pre-initialised RunState (session already created in store).
    task:            The user's task / prompt for this run.
    llm:             LLMClient instance to use for LLM calls.
    tool_registry:   The agent's registered tools.
    store:           MemoryStore for persistence.
    context_builder: Builds the token-trimmed message window.
    max_iterations:  Override the default from settings.
    on_chunk:        If provided, streamed text tokens are forwarded here as
                     they arrive.  Only called on ``finish_reason == "stop"``
                     turns; tool-call turns produce no text stream.
    on_tool_call:    If provided, called once per tool call just before
                     dispatch with ``(tool_name, arguments_json)``.  Useful
                     for real-time UI feedback.  Called for every agent in a
                     multi-agent graph when stored on the Agent instance.
    response_format: Optional structured-output constraint forwarded to the
                     LLM on every call.  See ``LLMClient.chat()`` for details.

    Returns
    -------
    The agent's final text response.
    """
    limit = max_iterations if max_iterations is not None else settings.max_iterations
    user_message = UserMessage(content=task)

    # Persist the initial user message
    await store.add_message(state.session_id, user_message)

    logger.info(
        "loop.start",
        agent=state.agent_name,
        session_id=state.session_id,
        task_preview=task[:120],
        max_iterations=limit,
    )

    while state.iteration < limit:
        state.iteration += 1

        logger.debug(
            "loop.iteration",
            agent=state.agent_name,
            session_id=state.session_id,
            iteration=state.iteration,
        )

        # Build context window (trim to token budget)
        window = await context_builder.build(
            session_id=state.session_id,
            store=store,
        )
        state.messages = window

        # ----------------------------------------------------------------
        # LLM call
        # ----------------------------------------------------------------
        tools = tool_registry.schemas() if len(tool_registry) > 0 else None
        response = await llm.chat(
            state.messages,
            tools,
            on_chunk=on_chunk,
            response_format=response_format,
        )
        state.add_tokens(response.prompt_tokens, response.completion_tokens)

        # ----------------------------------------------------------------
        # Stop — we have a final answer
        # ----------------------------------------------------------------
        if response.finish_reason == "stop":
            final = response.content or ""
            assistant_msg = AssistantMessage(content=final)
            await store.add_message(state.session_id, assistant_msg)

            state.status = "completed"
            state.result = final

            await store.close_session(
                state.session_id,
                status="completed",
                result=final,
                total_tokens=state.total_tokens,
            )

            logger.info(
                "loop.done",
                agent=state.agent_name,
                session_id=state.session_id,
                iterations=state.iteration,
                total_tokens=state.total_tokens,
                result_preview=final[:120],
            )
            return final

        # ----------------------------------------------------------------
        # Tool calls — dispatch all concurrently, then loop
        # ----------------------------------------------------------------
        if response.has_tool_calls:
            # Persist the assistant message (with tool_calls)
            assistant_msg = AssistantMessage(
                content=response.content,
                tool_calls=response.tool_calls,
            )
            await store.add_message(state.session_id, assistant_msg)

            logger.debug(
                "loop.tool_calls",
                agent=state.agent_name,
                session_id=state.session_id,
                tools=[tc.name for tc in response.tool_calls],
            )

            # Notify caller about each tool call before dispatching
            if on_tool_call is not None:
                for tc in response.tool_calls:
                    on_tool_call(tc.name, tc.arguments)

            # Dispatch all tool calls in parallel
            # Set the context var so delegate_to_agent can read the session ID
            token = current_session_id.set(state.session_id)
            t0 = time.perf_counter()
            dispatch_tasks = [tool_registry.dispatch(call) for call in response.tool_calls]
            results: list[tuple[str, str]] = await asyncio.gather(*dispatch_tasks)
            current_session_id.reset(token)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            logger.debug(
                "loop.tool_calls_done",
                agent=state.agent_name,
                session_id=state.session_id,
                n_tools=len(results),
                elapsed_ms=elapsed_ms,
            )

            # Persist tool results and append result messages
            for call, (call_id, result_str) in zip(response.tool_calls, results):
                duration_per_tool = elapsed_ms // max(len(results), 1)
                is_error = result_str.startswith("ERROR:")
                await store.save_tool_result(
                    session_id=state.session_id,
                    tool_call_id=call_id,
                    tool_name=call.name,
                    input_json=call.arguments,
                    output=None if is_error else result_str,
                    error=result_str if is_error else None,
                    duration_ms=duration_per_tool,
                )

                tool_result_msg = ToolResultMessage(
                    tool_call_id=call_id,
                    content=result_str,
                )
                await store.add_message(state.session_id, tool_result_msg)

            # Loop back for the next LLM call
            continue

        # ----------------------------------------------------------------
        # Unexpected finish reason (e.g. "length", "content_filter")
        # ----------------------------------------------------------------
        logger.warning(
            "loop.unexpected_finish",
            agent=state.agent_name,
            finish_reason=response.finish_reason,
            session_id=state.session_id,
        )
        partial = response.content or ""
        state.status = "error"
        state.result = partial
        _reason_labels: dict[str, str] = {
            "length": "token limit reached",
            "content_filter": "content filtered by provider",
        }
        label = _reason_labels.get(response.finish_reason, response.finish_reason)
        detail = f"  partial output: {partial[:200]}" if partial else ""
        # Store a readable error string in the DB so visualisers can display it
        error_msg = f"{label.capitalize()}{detail}"
        await store.close_session(
            state.session_id,
            status="error",
            result=error_msg,
            total_tokens=state.total_tokens,
        )
        raise RuntimeError(
            f"Agent '{state.agent_name}' stopped: {label}"
            f" (finish_reason={response.finish_reason!r}){detail}"
        )

    # ----------------------------------------------------------------
    # Max iterations exceeded
    # ----------------------------------------------------------------
    state.status = "max_iterations"
    await store.close_session(
        state.session_id,
        status="max_iterations",
        result=f"Exceeded {limit} iterations",
        total_tokens=state.total_tokens,
    )
    logger.error(
        "loop.max_iterations",
        agent=state.agent_name,
        session_id=state.session_id,
        limit=limit,
    )
    raise MaxIterationsError(
        f"Agent {state.agent_name!r} exceeded {limit} iterations (session {state.session_id})."
    )
