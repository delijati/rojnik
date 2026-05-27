#!/usr/bin/env python3
"""
mcp_server.py — minimal MCP stdio server used by mcp_agent.py.

Exposes two tools over the Model Context Protocol (stdio transport):

  get_time()                      → current UTC time (ISO 8601)
  roll_dice(sides, count)         → roll `count` dice with `sides` sides

Run standalone (for manual testing with the MCP Inspector or any client):

    /work/venv/bin/python examples/mcp_server.py

The server communicates exclusively over stdin/stdout (MCP stdio transport).
All diagnostic output goes to stderr so it does not pollute the JSON-RPC stream.
"""

from __future__ import annotations

import asyncio
import random
import sys
from datetime import datetime, timezone

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

server = Server("demo-tools")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_time",
            description="Return the current UTC time in ISO 8601 format.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        types.Tool(
            name="roll_dice",
            description=(
                "Simulate rolling `count` dice each having `sides` sides. "
                "Returns each individual roll and the total."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sides": {
                        "type": "integer",
                        "description": "Number of faces per die (e.g. 6, 20).",
                    },
                    "count": {
                        "type": "integer",
                        "description": "How many dice to roll (default 1).",
                    },
                },
                "required": ["sides"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "get_time":
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return [types.TextContent(type="text", text=now)]

    if name == "roll_dice":
        sides = int(arguments.get("sides", 6))
        count = int(arguments.get("count", 1))
        if sides < 2:
            raise ValueError(f"sides must be >= 2, got {sides}")
        if count < 1 or count > 100:
            raise ValueError(f"count must be 1–100, got {count}")
        rolls = [random.randint(1, sides) for _ in range(count)]
        total = sum(rolls)
        text = f"Rolled {count}d{sides}: {rolls}  →  total: {total}"
        return [types.TextContent(type="text", text=text)]

    raise ValueError(f"Unknown tool: {name!r}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    print(f"[mcp_server] starting (pid {__import__('os').getpid()})", file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
