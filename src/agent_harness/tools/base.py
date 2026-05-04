"""
@tool decorator.

Usage
-----
    from agent_harness.tools.base import tool

    @tool(description="Return the square of a number")
    async def square(n: float) -> float:
        return n * n

    # or for sync functions (will be run in a thread executor)
    @tool(description="Read a file")
    def read_file(path: str) -> str:
        ...

What the decorator does
-----------------------
1. Introspects the function signature via `inspect`.
2. Builds a Pydantic model dynamically from the parameters.
3. Derives an OpenAI-compatible JSON Schema from that model.
4. Wraps the function so that:
   - inputs are validated with Pydantic before calling the real function.
   - sync functions are dispatched to asyncio's default thread executor
     so the event loop is never blocked.
5. Attaches `.tool_schema` (a ToolSchema) and `.is_tool = True` to the wrapper.

The registry picks up any callable that has `.is_tool == True`.
"""


import asyncio
import inspect
from functools import wraps
from typing import Any, Callable, get_type_hints

# inspect.iscoroutinefunction is the preferred API from Python 3.12+;
# asyncio.iscoroutinefunction is deprecated and will be removed in 3.16.
_is_coroutine = inspect.iscoroutinefunction

from pydantic import create_model
from pydantic.fields import FieldInfo

from agent_harness.llm.schemas import (
    ToolFunctionSchema,
    ToolParameterSchema,
    ToolSchema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SKIP_PARAMS = frozenset({"return", "self", "cls"})

# JSON Schema types for Python builtins
_PY_TO_JSON_TYPE: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _build_tool_schema(fn: Callable[..., Any], description: str) -> ToolSchema:
    """
    Inspect *fn* and produce a ToolSchema (OpenAI function-calling format).
    """
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in _SKIP_PARAMS:
            continue

        annotation = hints.get(name, Any)
        json_type = _PY_TO_JSON_TYPE.get(annotation, "string")

        prop: dict[str, Any] = {"type": json_type}

        # Pull description from param default if it's a FieldInfo (pydantic Field)
        if isinstance(param.default, FieldInfo) and param.default.description:
            prop["description"] = param.default.description

        properties[name] = prop

        if param.default is inspect.Parameter.empty or isinstance(param.default, FieldInfo):
            if not (isinstance(param.default, FieldInfo) and param.default.default is not inspect.Parameter.empty):
                required.append(name)

    return ToolSchema(
        type="function",
        function=ToolFunctionSchema(
            name=fn.__name__,
            description=description,
            parameters=ToolParameterSchema(
                type="object",
                properties=properties,
                required=required,
            ),
        ),
    )


def _build_pydantic_validator(fn: Callable[..., Any]) -> type:
    """
    Build a Pydantic model from *fn*'s signature for input validation.
    """
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    sig = inspect.signature(fn)
    fields: dict[str, Any] = {}

    for name, param in sig.parameters.items():
        if name in _SKIP_PARAMS:
            continue
        annotation = hints.get(name, Any)
        if param.default is inspect.Parameter.empty:
            fields[name] = (annotation, ...)
        elif isinstance(param.default, FieldInfo):
            fields[name] = (annotation, param.default)
        else:
            fields[name] = (annotation, param.default)

    return create_model(f"_{fn.__name__}_Input", **fields)


# ---------------------------------------------------------------------------
# The @tool decorator
# ---------------------------------------------------------------------------

def tool(description: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator that marks a function as an agent tool.

    Parameters
    ----------
    description:
        Human-readable description passed to the LLM in the tool schema.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        schema = _build_tool_schema(fn, description)
        validator_model = _build_pydantic_validator(fn)
        is_async = _is_coroutine(fn)

        @wraps(fn)
        async def wrapper(**kwargs: Any) -> Any:
            # Validate inputs
            validated = validator_model(**kwargs)
            clean_kwargs = validated.model_dump()

            if is_async:
                return await fn(**clean_kwargs)
            else:
                # Run sync function in thread executor to avoid blocking
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: fn(**clean_kwargs))

        # Attach metadata for the registry to consume
        wrapper.is_tool = True  # type: ignore[attr-defined]
        wrapper.tool_schema = schema  # type: ignore[attr-defined]
        wrapper.__wrapped__ = fn  # type: ignore[attr-defined]

        return wrapper

    return decorator
