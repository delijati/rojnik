"""
delegate_to_agent — the multi-agent bridge tool.

This module is intentionally NOT a plain @tool-decorated function at
module level, because the delegate tool needs a reference to an
AgentRegistry, which is only known at runtime.

Instead, call make_delegate_tool(registry) to get a @tool-decorated
async function bound to a specific registry.  Pass the result to
Agent(tools=[..., make_delegate_tool(registry)]).

Usage
-----
    from agent_harness.agent.agent import AgentRegistry
    from agent_harness.tools.builtins.delegate import make_delegate_tool

    registry = AgentRegistry()
    registry.register(file_agent)
    registry.register(shell_agent)

    orchestrator = Agent(
        name="orchestrator",
        system_prompt="...",
        tools=[make_delegate_tool(registry)],
    )

How it works
------------
When the LLM calls delegate_to_agent(agent_name="file_reader", task="..."),
the harness:
1. Looks up the agent in the registry.
2. Calls agent.run(task, parent_session_id=current_session_id).
3. Returns the subagent's final text response as the tool result.

Because the orchestrator's run_loop dispatches tool calls with
asyncio.gather(), multiple delegate calls in the same LLM response
run the subagents in parallel.

The `_current_session_id` context variable is set by the run_loop
just before dispatching tool calls so the delegate tool can attach
the correct parent_session_id.
"""


from contextvars import ContextVar
from typing import TYPE_CHECKING, Callable, Any

from loguru import logger

from agent_harness.tools.base import tool

if TYPE_CHECKING:
    from agent_harness.agent.agent import AgentRegistry

# Context variable: the run_loop sets this before dispatching tool calls.
# The delegate tool reads it to wire up parent_session_id in SQLite.
current_session_id: ContextVar[str | None] = ContextVar(
    "current_session_id", default=None
)


def make_delegate_tool(registry: "AgentRegistry") -> Callable[..., Any]:
    """
    Build a delegate_to_agent @tool bound to *registry*.

    Parameters
    ----------
    registry:
        The AgentRegistry that will be consulted when the LLM calls the tool.

    Returns
    -------
    An async @tool-decorated function that can be registered on any Agent.
    """

    @tool(description=(
        "Delegate a task to a specialist subagent and return its response. "
        f"Available agents: {', '.join(registry.names()) if len(registry) else '(none registered yet)'}. "
        "Use this when the task requires a specialist capability you don't have directly. "
        "The subagent will run its own ReAct loop and return a complete answer."
    ))
    async def delegate_to_agent(agent_name: str, task: str) -> str:
        if agent_name not in registry:
            available = registry.names()
            return (
                f"ERROR: No agent named {agent_name!r}. "
                f"Available agents: {available}"
            )

        parent_sid = current_session_id.get()
        logger.info(
            "agent.delegate",
            parent_agent_session=parent_sid,
            target_agent=agent_name,
            task_preview=task[:120],
        )

        target = registry.get(agent_name)
        result = await target.run(task, parent_session_id=parent_sid)
        return result

    return delegate_to_agent
