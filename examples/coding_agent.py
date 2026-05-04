"""
Coding agent example — demonstrates the full multi-agent harness with
support for OpenAI, DeepSeek, and any local OpenAI-compatible LLM server.

Architecture
------------
  orchestrator
    ├── file_agent      (read_file, list_directory)
    └── shell_agent     (shell_exec)

The orchestrator delegates file reading and shell commands to
specialist agents, then synthesises the results.

Provider quick-start
--------------------
  # OpenAI (default)
  OPENAI_API_KEY=sk-... python examples/coding_agent.py

  # DeepSeek
  python examples/coding_agent.py --provider deepseek --api-key sk-...

  # Local Ollama (llama3 running on the default port)
  python examples/coding_agent.py --provider local --base-url http://localhost:11434/v1 --model llama3

  # Local LM Studio
  python examples/coding_agent.py --provider local --base-url http://localhost:1234/v1 --model local-model

  # Custom task
  OPENAI_API_KEY=sk-... python examples/coding_agent.py --provider openai \
      "List the files in /tmp and tell me how many there are"

Environment variables (alternative to CLI flags)
-------------------------------------------------
  AGENT_PROVIDER   openai | deepseek | local    (default: openai)
  AGENT_MODEL      model identifier             (default: per provider)
  AGENT_BASE_URL   custom API endpoint URL      (default: per provider)
  AGENT_API_KEY    generic API key fallback
  OPENAI_API_KEY   API key for openai provider
  DEEPSEEK_API_KEY API key for deepseek provider
"""


import argparse
import asyncio
import os
import sys

# ---------------------------------------------------------------------------
# Provider / model / key must be set in the environment BEFORE agent_harness
# is imported, because config.py creates a module-level singleton at import
# time.  We parse CLI args first, push them into os.environ, then import.
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-agent coding assistant — supports OpenAI, DeepSeek, and local LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "deepseek", "local"],
        default=None,
        help=(
            "LLM provider to use.  "
            "Overrides AGENT_PROVIDER env var.  "
            "Choices: openai (default), deepseek, local."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model identifier, e.g. gpt-4o, deepseek-chat, llama3.  "
            "Overrides AGENT_MODEL env var.  "
            "Defaults: openai→gpt-4o, deepseek→deepseek-chat, local→llama3."
        ),
    )
    parser.add_argument(
        "--base-url",
        dest="base_url",
        default=None,
        help=(
            "Base URL of the OpenAI-compatible Chat Completions endpoint.  "
            "Overrides AGENT_BASE_URL env var.  "
            "Examples: http://localhost:11434/v1 (Ollama), "
            "http://localhost:1234/v1 (LM Studio), "
            "https://api.deepseek.com/v1 (DeepSeek, set automatically)."
        ),
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help=(
            "API key for the provider.  "
            "Overrides OPENAI_API_KEY / DEEPSEEK_API_KEY / AGENT_API_KEY.  "
            "Not required for local servers that don't enforce auth."
        ),
    )
    # Positional: the task to run (everything after flags)
    parser.add_argument(
        "task",
        nargs="*",
        help="Task description for the agent.  Defaults to a built-in demo task.",
    )
    return parser.parse_args()


def _apply_args_to_env(args: argparse.Namespace) -> None:
    """
    Push CLI overrides into os.environ so that config.Settings picks them up
    when it is instantiated during agent_harness import.
    """
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


# Parse and apply CLI args before importing agent_harness
_args = _parse_args()
_apply_args_to_env(_args)

# Add the workspace root to sys.path so agent_harness is importable
# without installing it as a package.
sys.path.insert(0, "/work")

import agent_harness  # noqa: F401 — triggers logging setup  # noqa: E402

from agent_harness.agent.agent import Agent, AgentRegistry  # noqa: E402
from agent_harness.llm.client import LLMClient  # noqa: E402
from agent_harness.tools.builtins.delegate import make_delegate_tool  # noqa: E402
from agent_harness.tools.builtins.files import list_directory, read_file  # noqa: E402
from agent_harness.tools.builtins.shell import shell_exec  # noqa: E402


def build_harness(llm: LLMClient) -> Agent:
    """
    Assemble the multi-agent system and return the orchestrator.

    Parameters
    ----------
    llm:
        Shared LLMClient instance.  All agents in the harness use the same
        client so they talk to the same provider / model.
    """

    # ------------------------------------------------------------------
    # Specialist agents
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    registry = AgentRegistry()
    registry.register_many(file_agent, shell_agent)

    # ------------------------------------------------------------------
    # Orchestrator — its only tool is delegate_to_agent
    # ------------------------------------------------------------------

    delegate = make_delegate_tool(registry)

    orchestrator = Agent(
        name="orchestrator",
        system_prompt=(
            "You are a senior software engineer coordinating a team of specialist agents. "
            "You have access to two agents:\n"
            "  - file_reader: reads files and lists directories.\n"
            "  - shell_executor: runs shell commands.\n\n"
            "When the user gives you a task:\n"
            "1. Break it into subtasks.\n"
            "2. Delegate each subtask to the appropriate agent using delegate_to_agent.\n"
            "3. Synthesise the results into a clear final answer.\n\n"
            "You can call multiple agents in a single response to run them in parallel."
        ),
        tools=[delegate],
        llm=llm,
    )

    return orchestrator


async def main() -> None:
    from agent_harness.config import settings

    task = (
        " ".join(_args.task)
        if _args.task
        else (
            "List the contents of the /work directory and then run "
            "'python3 --version' to tell me what Python version is available."
        )
    )

    print(
        f"\nProvider : {settings.provider}"
        f"\nModel    : {settings.model}"
        f"\nBase URL : {settings.get_base_url() or 'https://api.openai.com/v1 (default)'}"
        f"\nTask     : {task}"
        f"\n{'=' * 60}"
    )

    # Build a single LLMClient shared by the whole harness
    llm = LLMClient()

    orchestrator = build_harness(llm)
    result = await orchestrator.run(task)

    print(f"\n{'=' * 60}\nResult:\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
