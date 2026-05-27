"""
Coding agent example — demonstrates the full multi-agent harness with
support for OpenAI, DeepSeek, and any local OpenAI-compatible LLM server.

Architecture
------------
  orchestrator
    ├── file_reader     (read_file, list_directory)
    ├── shell_executor  (shell_exec)
    └── mcp_agent       (tools from the MCP server)

An MCP server is always spawned alongside the harness.  The bundled
mcp_server.py (get_time, roll_dice) is used by default; pass --mcp-server
to use a different one.

Provider quick-start
--------------------
  # OpenAI (default) — uses bundled mcp_server.py
  OPENAI_API_KEY=sk-... python examples/coding_agent.py

  # DeepSeek
  python examples/coding_agent.py --provider deepseek --api-key sk-...

  # Local llama.cpp
  python examples/coding_agent.py --provider local \\
      --base-url http://localhost:8080/v1 --model qwen2.5-coder

  # Custom MCP server
  OPENAI_API_KEY=sk-... python examples/coding_agent.py \\
      --mcp-server /usr/local/bin/my-mcp-server

  # Custom task
  OPENAI_API_KEY=sk-... python examples/coding_agent.py \\
      "What time is it, and how many files are in /work?"

Environment variables (alternative to CLI flags)
-------------------------------------------------
  AGENT_PROVIDER   openai | deepseek | local    (default: openai)
  AGENT_MODEL      model identifier             (default: per provider)
  AGENT_BASE_URL   custom API endpoint URL      (default: per provider)
  AGENT_API_KEY    generic API key fallback
  OPENAI_API_KEY   API key for openai provider
  DEEPSEEK_API_KEY API key for deepseek provider
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Provider / model / key must be set in the environment BEFORE agent_harness
# is imported, because config.py creates a module-level singleton at import
# time.  We parse CLI args first, push them into os.environ, then import.
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-agent coding assistant — supports OpenAI, DeepSeek, "
            "and local LLMs, with MCP server tools always available."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "deepseek", "local"],
        default=None,
        help="LLM provider (overrides AGENT_PROVIDER). Choices: openai (default), deepseek, local.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier, e.g. gpt-4o, deepseek-chat, llama3 (overrides AGENT_MODEL).",
    )
    parser.add_argument(
        "--base-url",
        dest="base_url",
        default=None,
        help="Base URL of the OpenAI-compatible endpoint (overrides AGENT_BASE_URL).",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="API key (overrides OPENAI_API_KEY / DEEPSEEK_API_KEY / AGENT_API_KEY).",
    )
    parser.add_argument(
        "--mcp-server",
        dest="mcp_server",
        nargs="+",
        metavar="WORD",
        default=None,
        help=(
            "MCP server command to spawn (default: python examples/mcp_server.py).  "
            "Example: --mcp-server /usr/local/bin/my-mcp-server"
        ),
    )
    parser.add_argument(
        "task",
        nargs="*",
        help="Task for the agent (default: a built-in demo task).",
    )
    return parser.parse_args()


def _apply_args_to_env(args: argparse.Namespace) -> None:
    if args.provider:
        os.environ["AGENT_PROVIDER"] = args.provider
    if args.model:
        os.environ["AGENT_MODEL"] = args.model
    if args.base_url:
        os.environ["AGENT_BASE_URL"] = args.base_url
    if args.api_key:
        provider = args.provider or os.environ.get("AGENT_PROVIDER", "openai")
        if provider == "openai":
            os.environ["OPENAI_API_KEY"] = args.api_key
        elif provider == "deepseek":
            os.environ["DEEPSEEK_API_KEY"] = args.api_key
        else:
            os.environ["AGENT_API_KEY"] = args.api_key


_args = _parse_args()
_apply_args_to_env(_args)

sys.path.insert(0, "/work")

import agent_harness  # noqa: F401 — triggers logging setup  # noqa: E402

from agent_harness.agent.agent import Agent, AgentRegistry  # noqa: E402
from agent_harness.llm.client import LLMClient  # noqa: E402
from agent_harness.mcp import mcp_to_tools  # noqa: E402
from agent_harness.tools.builtins.delegate import make_delegate_tool  # noqa: E402
from agent_harness.tools.builtins.files import list_directory, read_file  # noqa: E402
from agent_harness.tools.builtins.shell import shell_exec  # noqa: E402

from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402

# Default MCP server — the bundled demo server next to this file
_DEFAULT_MCP_CMD = [sys.executable, str(Path(__file__).parent / "mcp_server.py")]


# ---------------------------------------------------------------------------
# Harness assembly
# ---------------------------------------------------------------------------

def build_harness(llm: LLMClient, mcp_tools: list) -> Agent:
    """Assemble the multi-agent system and return the orchestrator."""

    tool_names = [t.tool_schema.function.name for t in mcp_tools]  # type: ignore[attr-defined]

    file_agent = Agent(
        name="file_reader",
        system_prompt=(
            "You are a specialist at reading and summarising files and directories. "
            "Use your tools to explore the filesystem and return clear, concise results. "
            "Always use absolute or unambiguous paths."
        ),
        tools=[read_file, list_directory],
        llm=llm,
    )

    shell_agent = Agent(
        name="shell_executor",
        system_prompt=(
            "You are a specialist at running shell commands and interpreting their output. "
            "Run the requested command, capture all output, and return a clear summary. "
            "Always prefer non-destructive commands unless explicitly asked otherwise."
        ),
        tools=[shell_exec],
        llm=llm,
    )

    mcp_agent = Agent(
        name="mcp_agent",
        system_prompt=(
            "You are a specialist that calls external tools exposed via MCP "
            f"(Model Context Protocol).  Available tools: {', '.join(tool_names)}.  "
            "Use them to answer the request, then return a clear summary."
        ),
        tools=mcp_tools,
        llm=llm,
    )

    registry = AgentRegistry()
    registry.register_many(file_agent, shell_agent, mcp_agent)

    orchestrator = Agent(
        name="orchestrator",
        system_prompt=(
            "You are a senior software engineer coordinating a team of specialist agents:\n"
            "  - file_reader: reads files and lists directories.\n"
            "  - shell_executor: runs shell commands.\n"
            f"  - mcp_agent: calls external MCP tools ({', '.join(tool_names)}).\n\n"
            "When given a task:\n"
            "1. Break it into subtasks.\n"
            "2. Delegate each subtask to the right agent via delegate_to_agent.\n"
            "3. Synthesise the results into a clear final answer.\n\n"
            "You can delegate to multiple agents in a single response to run them in parallel."
        ),
        tools=[make_delegate_tool(registry)],
        llm=llm,
    )

    return orchestrator


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_DEFAULT_TASK = (
    "What time is it right now?  Also list the contents of /work and tell me "
    "what Python version is available."
)


async def main() -> None:
    from agent_harness.config import settings

    task = " ".join(_args.task) if _args.task else _DEFAULT_TASK
    cmd = _args.mcp_server or _DEFAULT_MCP_CMD

    print(
        f"\nProvider  : {settings.provider}"
        f"\nModel     : {settings.model}"
        f"\nBase URL  : {settings.get_base_url() or 'https://api.openai.com/v1 (default)'}"
        f"\nMCP cmd   : {' '.join(cmd)}"
        f"\nTask      : {task}"
        f"\n{'=' * 60}"
    )

    llm = LLMClient()
    params = StdioServerParameters(command=cmd[0], args=cmd[1:])
    errlog = open(os.devnull, "w")
    try:
        async with stdio_client(params, errlog=errlog) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                mcp_tools = await mcp_to_tools(session)
                print(f"MCP tools : {[t.tool_schema.function.name for t in mcp_tools]}")  # type: ignore[attr-defined]
                print("=" * 60)
                result = await build_harness(llm, mcp_tools).run(task)
    finally:
        errlog.close()

    print(f"\n{'=' * 60}\nResult:\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
