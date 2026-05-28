"""
tests/test_mcp.py — integration tests for mcp_server.py and the MCP bridge.

The bridge (mcp_to_tools and friends) lives in rojnik.mcp and is
imported directly; no code is inlined here.

Covers:
  1. Tool listing — names, schemas.
  2. get_time — ISO 8601 output.
  3. roll_dice — correct output, total == sum, error paths.
  4. Bridge — branded callables; callable invocation; error propagation.

Run with:
  /work/venv/bin/python -m pytest tests/test_mcp.py -v
"""

from __future__ import annotations

import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, get_default_environment, stdio_client  # noqa: E402

from rojnik.mcp import mcp_to_tools  # noqa: E402

_SERVER = str(Path(__file__).parent.parent / "examples" / "mcp_server.py")


# ---------------------------------------------------------------------------
# Inline connection helper — keeps the entire anyio lifecycle inside a single
# test function, avoiding cross-task cancel-scope teardown errors.
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _connect():
    # stdio_client only inherits a safe env-var whitelist; PYTHONPATH is not
    # included.  Propagate it explicitly so the server subprocess can import
    # packages installed in a virtual-environment that is on PYTHONPATH.
    env = get_default_environment()
    if "PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    params = StdioServerParameters(command=sys.executable, args=[_SERVER], env=env)
    errlog = open(os.devnull, "w")
    try:
        async with stdio_client(params, errlog=errlog) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                yield session
    finally:
        errlog.close()


# ---------------------------------------------------------------------------
# 1. Tool listing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_tools_returns_expected_names():
    async with _connect() as session:
        result = await session.list_tools()
        assert {t.name for t in result.tools} == {"get_time", "roll_dice"}


@pytest.mark.asyncio
async def test_get_time_schema_has_empty_properties():
    async with _connect() as session:
        result = await session.list_tools()
        tool = next(t for t in result.tools if t.name == "get_time")
        schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
        assert schema.get("type") == "object"
        assert schema.get("properties") == {}


@pytest.mark.asyncio
async def test_roll_dice_schema_has_sides_and_count():
    async with _connect() as session:
        result = await session.list_tools()
        tool = next(t for t in result.tools if t.name == "roll_dice")
        schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
        assert "sides" in schema.get("properties", {})
        assert "count" in schema.get("properties", {})
        assert schema.get("required") == ["sides"]


# ---------------------------------------------------------------------------
# 2. get_time
# ---------------------------------------------------------------------------

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")


@pytest.mark.asyncio
async def test_get_time_returns_iso8601():
    async with _connect() as session:
        result = await session.call_tool("get_time", {})
        assert not result.isError
        text = result.content[0].text
        assert _ISO_RE.match(text), f"Not ISO 8601: {text!r}"


@pytest.mark.asyncio
async def test_get_time_contains_utc_offset():
    async with _connect() as session:
        result = await session.call_tool("get_time", {})
        assert "+00:00" in result.content[0].text


# ---------------------------------------------------------------------------
# 3. roll_dice — happy paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_roll_dice_single_die():
    async with _connect() as session:
        result = await session.call_tool("roll_dice", {"sides": 6})
        assert not result.isError
        text = result.content[0].text
        assert "1d6" in text and "total" in text


@pytest.mark.asyncio
async def test_roll_dice_multiple():
    async with _connect() as session:
        result = await session.call_tool("roll_dice", {"sides": 20, "count": 4})
        assert not result.isError
        text = result.content[0].text
        assert "4d20" in text
        rolls = [int(x.strip()) for x in re.search(r"\[(.+?)\]", text).group(1).split(",")]
        assert len(rolls) == 4 and all(1 <= r <= 20 for r in rolls)


@pytest.mark.asyncio
async def test_roll_dice_total_is_sum():
    for _ in range(5):
        async with _connect() as session:
            result = await session.call_tool("roll_dice", {"sides": 6, "count": 3})
            text = result.content[0].text
            rolls = [int(x.strip()) for x in re.search(r"\[(.+?)\]", text).group(1).split(",")]
            total = int(re.search(r"total:\s*(\d+)", text).group(1))
            assert sum(rolls) == total


# ---------------------------------------------------------------------------
# 4. roll_dice — error paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_roll_dice_sides_below_2_is_error():
    async with _connect() as session:
        assert (await session.call_tool("roll_dice", {"sides": 1})).isError


@pytest.mark.asyncio
async def test_roll_dice_count_zero_is_error():
    async with _connect() as session:
        assert (await session.call_tool("roll_dice", {"sides": 6, "count": 0})).isError


@pytest.mark.asyncio
async def test_roll_dice_count_over_100_is_error():
    async with _connect() as session:
        assert (await session.call_tool("roll_dice", {"sides": 6, "count": 101})).isError


# ---------------------------------------------------------------------------
# 5 & 6. MCP bridge (using the inlined helpers above)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bridge_creates_branded_callables():
    async with _connect() as session:
        tools = await mcp_to_tools(session)
        assert len(tools) == 2
        names = {t.tool_schema.function.name for t in tools}
        assert names == {"get_time", "roll_dice"}
        for t in tools:
            assert t.is_tool is True
            assert t.tool_schema is not None


@pytest.mark.asyncio
async def test_bridge_callable_get_time():
    async with _connect() as session:
        tools = await mcp_to_tools(session)
        get_time = next(t for t in tools if t.__name__ == "get_time")
        assert _ISO_RE.match(await get_time())


@pytest.mark.asyncio
async def test_bridge_callable_roll_dice():
    async with _connect() as session:
        tools = await mcp_to_tools(session)
        roll_dice = next(t for t in tools if t.__name__ == "roll_dice")
        result = await roll_dice(sides=6, count=2)
        assert "2d6" in result and "total" in result


@pytest.mark.asyncio
async def test_bridge_callable_raises_on_error():
    async with _connect() as session:
        tools = await mcp_to_tools(session)
        roll_dice = next(t for t in tools if t.__name__ == "roll_dice")
        with pytest.raises(RuntimeError, match="error"):
            await roll_dice(sides=1)
