"""
Coding agent example — demonstrates a multi-agent rojnik harness with
support for OpenAI, DeepSeek, and any local OpenAI-compatible LLM server.

Architecture
------------
  orchestrator
    ├── file_reader     (read_file, list_directory)
    ├── shell_executor  (shell_exec)
    └── mcp_agent       (optional tools from an MCP server)

Pass --mcp-server to add an MCP specialist. The core example does not require
the optional MCP dependency.

Provider quick-start
--------------------
  # OpenAI (default)
  OPENAI_API_KEY=sk-... python examples/coding_agent.py

  # DeepSeek
  python examples/coding_agent.py --provider deepseek --api-key sk-...

  # Local llama.cpp
  python examples/coding_agent.py --provider local \\
      --base-url http://localhost:8080/v1 --model qwen2.5-coder

  # Bundled MCP demo server
  OPENAI_API_KEY=sk-... python examples/coding_agent.py \\
      --mcp-server "python examples/mcp_server.py"

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
import shlex
from typing import Literal

# ---------------------------------------------------------------------------
# Provider / model / key must be set in the environment BEFORE rojnik
# is imported, because config.py creates a module-level singleton at import
# time.  We parse CLI args first, push them into os.environ, then import.
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-agent coding assistant — supports OpenAI, DeepSeek, "
            "local LLMs, and optional MCP server tools."
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
        metavar="COMMAND",
        default=None,
        help=(
            "Optional MCP server command, parsed with shell-like quoting. "
            "Example: --mcp-server 'python examples/mcp_server.py'"
        ),
    )
    parser.add_argument(
        "--skill",
        action="append",
        default=[],
        metavar="PATH",
        help="SKILL.md file or containing directory. Repeat for multiple skills.",
    )
    parser.add_argument(
        "--skill-mode",
        choices=["eager", "on_demand"],
        default="eager",
        help="Include full skills in the prompt or load them with a tool. Default: eager.",
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

import rojnik  # noqa: E402, F401 - triggers logging setup
from rojnik.agent.agent import Agent  # noqa: E402
from rojnik.llm.client import LLMClient  # noqa: E402
from rojnik.tools.builtins.files import list_directory, read_file  # noqa: E402
from rojnik.tools.builtins.shell import shell_exec  # noqa: E402

# ---------------------------------------------------------------------------
# Harness assembly
# ---------------------------------------------------------------------------

def build_harness(
    llm: LLMClient,
    mcp_tools: list | None = None,
    skills: list[str] | None = None,
    skill_mode: Literal["eager", "on_demand"] = "eager",
) -> Agent:
    """Assemble the multi-agent system and return the orchestrator."""

    file_agent = Agent(
        name="file_reader",
        system_prompt=(
            "You are a specialist at reading and summarising files and directories. "
            "Use your tools to explore the filesystem and return clear, concise results. "
            "Always use absolute or unambiguous paths."
        ),
        tools=[read_file, list_directory],
        llm=llm,
        skills=skills,
        skill_mode=skill_mode,
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

    specialists = [file_agent, shell_agent]
    descriptions = [
        "  - file_reader: reads files and lists directories.",
        "  - shell_executor: runs shell commands.",
    ]

    if mcp_tools:
        tool_names = [t.tool_schema.function.name for t in mcp_tools]  # type: ignore[attr-defined]
        specialists.append(
            Agent(
                name="mcp_agent",
                system_prompt=(
                    "You call external tools exposed via MCP. "
                    f"Available tools: {', '.join(tool_names)}. "
                    "Use them to answer the request, then return a clear summary."
                ),
                tools=mcp_tools,
                llm=llm,
            )
        )
        descriptions.append(f"  - mcp_agent: calls MCP tools ({', '.join(tool_names)}).")

    orchestrator = Agent(
        name="orchestrator",
        system_prompt=(
            "You are a senior software engineer coordinating specialist agents:\n"
            + "\n".join(descriptions)
            + "\n\n"
            "When given a task:\n"
            "1. Break it into subtasks.\n"
            "2. Call the right agent tool for each subtask.\n"
            "3. Synthesise the results into a clear final answer.\n\n"
            "You can call multiple agents in one response to run them in parallel."
        ),
        tools=[agent.as_tool() for agent in specialists],
        llm=llm,
    )

    return orchestrator


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_DEFAULT_TASK = (
    "List the contents of the current directory and tell me what Python version is available."
)


async def main() -> None:
    from rojnik.config import settings

    task = " ".join(_args.task) if _args.task else _DEFAULT_TASK
    print(
        f"\nProvider  : {settings.provider}"
        f"\nModel     : {settings.model}"
        f"\nBase URL  : {settings.get_base_url() or 'https://api.openai.com/v1 (default)'}"
        f"\nMCP cmd   : {_args.mcp_server or '(disabled)'}"
        f"\nTask      : {task}"
        f"\n{'=' * 60}"
    )

    llm = LLMClient()
    if _args.mcp_server:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        from rojnik.mcp import mcp_to_tools

        cmd = shlex.split(_args.mcp_server)
        if not cmd:
            raise ValueError("--mcp-server command cannot be empty")
        params = StdioServerParameters(command=cmd[0], args=cmd[1:])
        with open(os.devnull, "w") as errlog:
            async with stdio_client(params, errlog=errlog) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    mcp_tools = await mcp_to_tools(session)
                    print(
                        "MCP tools : "
                        f"{[t.tool_schema.function.name for t in mcp_tools]}"  # type: ignore[attr-defined]
                    )
                    print("=" * 60)
                    result = await build_harness(
                        llm,
                        mcp_tools,
                        skills=_args.skill,
                        skill_mode=_args.skill_mode,
                    ).run(task)
    else:
        result = await build_harness(
            llm,
            skills=_args.skill,
            skill_mode=_args.skill_mode,
        ).run(task)

    print(f"\n{'=' * 60}\nResult:\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
