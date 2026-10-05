"""Tests for tools/base.py — the @tool decorator."""

import inspect
import json
from typing import Any

import pytest
from pydantic import Field

from rojnik.llm.schemas import ToolCallPart
from rojnik.tools.base import tool
from rojnik.tools.registry import ToolRegistry


class TestSchemaGeneration:
    def test_name_from_function(self):
        @tool(description="t")
        async def my_func(x: int) -> int:
            return x

        assert my_func.tool_schema.function.name == "my_func"

    def test_description_set(self):
        @tool(description="Does the thing")
        async def fn(x: str) -> str:
            return x

        assert fn.tool_schema.function.description == "Does the thing"

    def test_required_params(self):
        @tool(description="t")
        async def fn(a: int, b: str) -> str:
            return str(a) + b

        req = fn.tool_schema.function.parameters.required
        assert "a" in req and "b" in req

    def test_optional_param_not_required(self):
        @tool(description="t")
        async def fn(path: str, encoding: str = "utf-8") -> str:
            return path

        req = fn.tool_schema.function.parameters.required
        props = fn.tool_schema.function.parameters.properties
        assert "path" in req
        assert "encoding" not in req
        assert "encoding" in props

    def test_int_maps_to_integer(self):
        @tool(description="t")
        async def fn(n: int) -> int:
            return n

        assert fn.tool_schema.function.parameters.properties["n"]["type"] == "integer"

    def test_float_maps_to_number(self):
        @tool(description="t")
        async def fn(x: float) -> float:
            return x

        assert fn.tool_schema.function.parameters.properties["x"]["type"] == "number"

    def test_bool_maps_to_boolean(self):
        @tool(description="t")
        async def fn(flag: bool) -> bool:
            return flag

        assert fn.tool_schema.function.parameters.properties["flag"]["type"] == "boolean"

    def test_str_maps_to_string(self):
        @tool(description="t")
        async def fn(s: str) -> str:
            return s

        assert fn.tool_schema.function.parameters.properties["s"]["type"] == "string"

    def test_is_tool_attribute(self):
        @tool(description="t")
        async def fn() -> None:
            pass

        assert fn.is_tool is True

    def test_tool_schema_attribute_present(self):
        @tool(description="t")
        async def fn(x: int) -> int:
            return x

        assert fn.tool_schema is not None

    def test_openai_dict_shape(self):
        @tool(description="Test shape")
        async def fn(query: str, limit: int = 10) -> str:
            return query

        d = fn.tool_schema.to_openai_dict()
        assert d["type"] == "function"
        assert "query" in d["function"]["parameters"]["properties"]

    def test_function_name_preserved(self):
        @tool(description="t")
        async def uniquely_named(x: int) -> int:
            return x

        assert uniquely_named.__name__ == "uniquely_named"


class TestExecution:
    async def test_async_tool_executes(self):
        @tool(description="add")
        async def add(a: int, b: int) -> int:
            return a + b

        assert await add(a=2, b=3) == 5

    async def test_sync_tool_executes_via_executor(self):
        @tool(description="mul")
        def multiply(x: int, y: int) -> int:
            return x * y

        assert await multiply(x=4, y=5) == 20

    async def test_default_value_applied(self):
        @tool(description="greet")
        async def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        assert await greet(name="World") == "Hello, World!"

    async def test_returns_none_without_crash(self):
        @tool(description="noop")
        async def noop() -> None:
            pass

        assert await noop() is None

    async def test_exception_in_body_propagates(self):
        @tool(description="boom")
        async def boom() -> str:
            raise ValueError("on purpose")

        with pytest.raises(ValueError, match="on purpose"):
            await boom()


class TestArgumentForwarding:
    """The wrapper forwards exactly the arguments the model sent."""

    @staticmethod
    def _kwargs_tool():
        async def update(**kwargs: Any) -> dict[str, Any]:
            return kwargs

        update.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [
                inspect.Parameter("item_id", inspect.Parameter.KEYWORD_ONLY),
                inspect.Parameter("name", inspect.Parameter.KEYWORD_ONLY, default=None),
                inspect.Parameter("due", inspect.Parameter.KEYWORD_ONLY, default=None),
            ]
        )
        return tool(description="partial update")(update)

    async def test_omitted_optional_arguments_are_not_forwarded(self):
        update = self._kwargs_tool()

        assert await update(item_id=7, name="Neu") == {"item_id": 7, "name": "Neu"}

    async def test_explicit_null_is_forwarded(self):
        update = self._kwargs_tool()

        assert await update(item_id=7, due=None) == {"item_id": 7, "due": None}

    async def test_registry_dispatch_forwards_only_sent_arguments(self):
        registry = ToolRegistry()
        registry.register(self._kwargs_tool())

        _, result = await registry.dispatch(
            ToolCallPart(id="c1", name="update", arguments='{"item_id": 7, "name": "Neu"}')
        )

        assert json.loads(result) == {"item_id": 7, "name": "Neu"}

    async def test_field_default_is_applied_when_omitted(self):
        @tool(description="limit")
        async def search(query: str, limit: int = Field(5, description="max hits")) -> str:
            return f"{query}:{limit}"

        assert await search(query="x") == "x:5"
        assert await search(query="x", limit=2) == "x:2"

    async def test_sync_tool_receives_only_sent_arguments(self):
        @tool(description="greet")
        def greet(name: str, greeting: str = "Hallo") -> str:
            return f"{greeting}, {name}!"

        assert await greet(name="Welt") == "Hallo, Welt!"
