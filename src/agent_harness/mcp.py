"""
agent_harness.mcp — MCP (Model Context Protocol) bridge.

Converts tools exposed by any MCP server into agent-harness @tool-compatible
callables that can be registered directly with a ToolRegistry or passed to an
Agent.

Usage
-----
    from agent_harness.mcp import mcp_to_tools

    # inside an async function, with a live ClientSession:
    tools = await mcp_to_tools(session)
    agent = Agent(name="my_agent", tools=tools, llm=llm)

Each callable returned by ``mcp_to_tools`` is branded with:
    .is_tool    = True         — consumed by ToolRegistry.register()
    .tool_schema = ToolSchema  — forwarded to the LLM as a tool definition

Dependency
----------
Requires the ``mcp`` package (``pip install mcp``).  The module raises a clear
``ImportError`` at import time if it is not installed so that the rest of the
harness can be used without it.
"""

from __future__ import annotations

try:
    from mcp import ClientSession
    from mcp import types as mcp_types
except ImportError as _err:
    raise ImportError(
        "agent_harness.mcp requires the 'mcp' package.  "
        "Install it with:  pip install mcp\n"
        "Or:  pip install 'agent-harness[mcp]'"
    ) from _err

from typing import Any, Callable

from agent_harness.llm.schemas import ToolFunctionSchema, ToolParameterSchema, ToolSchema


def _mcp_tool_to_schema(t: mcp_types.Tool) -> ToolSchema:
    """Convert an MCP ``Tool`` descriptor into a harness ``ToolSchema``."""
    raw: dict[str, Any] = t.inputSchema if isinstance(t.inputSchema, dict) else {}
    return ToolSchema(
        type="function",
        function=ToolFunctionSchema(
            name=t.name,
            description=t.description or t.name,
            parameters=ToolParameterSchema(
                type="object",
                properties=raw.get("properties", {}),
                required=raw.get("required", []),
            ),
        ),
    )


def _make_mcp_caller(session: ClientSession, tool_name: str) -> Callable:
    """
    Return an async callable that forwards ``**kwargs`` to the named MCP tool.

    The returned function is *not* decorated with ``@tool``; instead it carries
    the ``.is_tool`` and ``.tool_schema`` attributes directly so it can be
    passed anywhere a ``@tool``-decorated function is accepted.
    """
    async def _call(**kwargs: Any) -> str:
        result = await session.call_tool(tool_name, kwargs or None)
        if result.isError:
            parts = [
                b.text
                for b in result.content
                if isinstance(b, mcp_types.TextContent) and b.text
            ]
            raise RuntimeError(
                f"MCP tool '{tool_name}' returned error: "
                + (" ".join(parts) or "(no detail)")
            )
        parts = [
            b.text
            for b in result.content
            if isinstance(b, mcp_types.TextContent) and b.text
        ]
        return "\n".join(parts) or "(empty response)"

    _call.__name__ = tool_name
    return _call


async def mcp_to_tools(session: ClientSession) -> list[Callable]:
    """
    Discover all tools exposed by an initialised MCP session and return them
    as a list of agent-harness ``@tool``-compatible callables.

    Parameters
    ----------
    session:
        An already-initialised ``mcp.ClientSession``.  The session must remain
        open for as long as the returned callables are used.

    Returns
    -------
    list[Callable]
        One async callable per MCP tool, each branded with:
        * ``.is_tool = True``
        * ``.tool_schema = ToolSchema(...)``
    """
    listed = await session.list_tools()
    tools: list[Callable] = []
    for t in listed.tools:
        fn = _make_mcp_caller(session, t.name)
        fn.is_tool = True                         # type: ignore[attr-defined]
        fn.tool_schema = _mcp_tool_to_schema(t)  # type: ignore[attr-defined]
        tools.append(fn)
    return tools
