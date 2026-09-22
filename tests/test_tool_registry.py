"""Tests for tools/registry.py — ToolRegistry."""


import json

import pytest

from rojnik.llm.schemas import ToolCallPart
from rojnik.tools.base import tool
from rojnik.tools.registry import ToolNotFoundError, ToolRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    return ToolRegistry()


@pytest.fixture
def add_tool():
    @tool(description="Add two integers")
    async def add(a: int, b: int) -> int:
        return a + b
    return add


@pytest.fixture
def echo_tool():
    @tool(description="Echo a string")
    async def echo(msg: str) -> str:
        return msg
    return echo


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_adds_tool(self, registry, add_tool):
        registry.register(add_tool)
        assert "add" in registry

    def test_len_reflects_registered_tools(self, registry, add_tool, echo_tool):
        registry.register(add_tool)
        registry.register(echo_tool)
        assert len(registry) == 2

    def test_names_returns_registered_names(self, registry, add_tool, echo_tool):
        registry.register_many(add_tool, echo_tool)
        assert set(registry.names()) == {"add", "echo"}

    def test_register_non_tool_raises_typeerror(self, registry):
        async def plain():
            pass
        with pytest.raises(TypeError, match="not a @tool-decorated"):
            registry.register(plain)

    def test_schemas_returns_one_per_tool(self, registry, add_tool, echo_tool):
        registry.register_many(add_tool, echo_tool)
        schemas = registry.schemas()
        assert len(schemas) == 2
        names = {s.function.name for s in schemas}
        assert names == {"add", "echo"}


# ---------------------------------------------------------------------------
# Dispatch — success paths
# ---------------------------------------------------------------------------

class TestDispatch:
    async def test_dispatch_async_tool(self, registry, add_tool):
        registry.register(add_tool)
        call = ToolCallPart(id="c1", name="add", arguments='{"a": 3, "b": 4}')
        call_id, result = await registry.dispatch(call)
        assert call_id == "c1"
        assert result == "7"

    async def test_dispatch_returns_string(self, registry, echo_tool):
        registry.register(echo_tool)
        call = ToolCallPart(id="c2", name="echo", arguments='{"msg": "hi"}')
        _, result = await registry.dispatch(call)
        assert isinstance(result, str)
        assert result == "hi"

    async def test_dispatch_non_string_result_is_json(self, registry):
        @tool(description="list")
        async def make_list(n: int) -> list:
            return list(range(n))

        registry.register(make_list)
        call = ToolCallPart(id="c3", name="make_list", arguments='{"n": 3}')
        _, result = await registry.dispatch(call)
        assert json.loads(result) == [0, 1, 2]

    async def test_dispatch_empty_args(self, registry):
        @tool(description="const")
        async def const() -> str:
            return "always"

        registry.register(const)
        call = ToolCallPart(id="c4", name="const", arguments="{}")
        _, result = await registry.dispatch(call)
        assert result == "always"


# ---------------------------------------------------------------------------
# Dispatch — error paths
# ---------------------------------------------------------------------------

class TestDispatchErrors:
    async def test_tool_not_found_raises(self, registry):
        call = ToolCallPart(id="c5", name="ghost", arguments="{}")
        with pytest.raises(ToolNotFoundError):
            await registry.dispatch(call)

    async def test_bad_json_returns_error_string(self, registry, add_tool):
        registry.register(add_tool)
        call = ToolCallPart(id="c6", name="add", arguments="NOT JSON {")
        _, result = await registry.dispatch(call)
        assert result.startswith("ERROR:")

    async def test_tool_exception_returns_error_string(self, registry):
        @tool(description="always fails")
        async def fail() -> str:
            raise RuntimeError("deliberate failure")

        registry.register(fail)
        call = ToolCallPart(id="c7", name="fail", arguments="{}")
        _, result = await registry.dispatch(call)
        assert result.startswith("ERROR:")
        assert "deliberate failure" in result
