#!/usr/bin/env python3
"""Minimal MCP 2.x stdio server used by the coding-agent example."""

from __future__ import annotations

import random
from datetime import UTC, datetime

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

mcp = MCPServer("demo-tools")


@mcp.tool()
def get_time() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@mcp.tool()
def roll_dice(sides: int, count: int = 1) -> str:
    """Roll `count` dice with `sides` faces and return each roll and the total."""
    if sides < 2:
        raise ToolError(f"sides must be >= 2, got {sides}")
    if count < 1 or count > 100:
        raise ToolError(f"count must be 1-100, got {count}")
    rolls = [random.randint(1, sides) for _ in range(count)]
    return f"Rolled {count}d{sides}: {rolls}  ->  total: {sum(rolls)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
