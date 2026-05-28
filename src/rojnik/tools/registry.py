"""
ToolRegistry — per-agent registry of available tools.

Usage
-----
    registry = ToolRegistry()
    registry.register(read_file)
    registry.register(shell_exec)

    # Get schemas for the LLM `tools=` param
    schemas = registry.schemas()

    # Dispatch a tool call from the LLM
    result = await registry.dispatch("read_file", '{"path": "/etc/hosts"}')
"""


import json
from typing import Any, Callable

from loguru import logger

from rojnik.llm.schemas import ToolCallPart, ToolSchema


class ToolNotFoundError(Exception):
    """Raised when the LLM calls a tool that isn't registered."""


class ToolRegistry:
    """
    Holds a mapping of tool name → (schema, async callable).
    Each Agent owns one ToolRegistry instance.
    """

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSchema, Callable[..., Any]]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, fn: Callable[..., Any]) -> None:
        """
        Register a tool function.  The function must have been decorated
        with @tool (i.e. it must have `.is_tool == True` and `.tool_schema`).
        """
        if not getattr(fn, "is_tool", False):
            raise TypeError(
                f"{fn.__name__!r} is not a @tool-decorated function. "
                "Decorate it with @tool(description='...') first."
            )
        schema: ToolSchema = fn.tool_schema  # type: ignore[attr-defined]
        name = schema.function.name

        if name in self._tools:
            logger.warning("tool.registry.overwrite", tool_name=name)

        self._tools[name] = (schema, fn)
        logger.debug("tool.registry.registered", tool_name=name)

    def register_many(self, *fns: Callable[..., Any]) -> None:
        for fn in fns:
            self.register(fn)

    # ------------------------------------------------------------------
    # Schema access
    # ------------------------------------------------------------------

    def schemas(self) -> list[ToolSchema]:
        """Return all registered tool schemas (for `tools=` in LLM calls)."""
        return [schema for schema, _ in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def dispatch(self, call: ToolCallPart) -> tuple[str, str]:
        """
        Execute the tool referenced by *call*.

        Returns
        -------
        (tool_call_id, result_str)
            result_str is always a string — either the tool's return value
            converted to str, or an error message prefixed with "ERROR: ".
        """
        name = call.name
        if name not in self._tools:
            raise ToolNotFoundError(f"No tool named {name!r} is registered.")

        _, fn = self._tools[name]

        try:
            raw_args: dict[str, Any] = json.loads(call.arguments) if call.arguments else {}
        except json.JSONDecodeError as exc:
            err = f"ERROR: Could not parse tool arguments as JSON — {exc}"
            logger.error("tool.dispatch.bad_json", tool_name=name, arguments=call.arguments)
            return call.id, err

        logger.debug("tool.dispatch", tool_name=name, args=raw_args)

        try:
            result = await fn(**raw_args)
            result_str = result if isinstance(result, str) else json.dumps(result, default=str)
        except Exception as exc:
            result_str = f"ERROR: {type(exc).__name__}: {exc}"
            logger.error(
                "tool.dispatch.error",
                tool_name=name,
                error=str(exc),
                exc_info=True,
            )

        logger.debug(
            "tool.dispatch.result",
            tool_name=name,
            result_preview=result_str[:200],
        )
        return call.id, result_str
