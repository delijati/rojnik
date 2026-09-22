<p align="center">
  <img src="logo.png" alt="rojnik logo" width="220">
</p>

# rojnik

A small async Python harness for building tool-using and multi-agent LLM applications without a workflow framework. It provides an explicit ReAct loop, OpenAI-compatible tool calling, SQLite-backed run history, structured output, streaming callbacks, and optional MCP integration.

## Features

- Async agents backed by OpenAI or any OpenAI-compatible API
- `@tool` schema generation and Pydantic argument validation
- Agents exposed directly as function tools with `Agent.as_tool()`
- Dynamic delegation through `make_delegate_tool(AgentRegistry)`
- Concurrent tool calls with parent/child session tracking
- SQLite persistence for messages, tool results, and execution trees
- Token-budget context trimming that preserves complete tool-call groups
- Streaming callbacks and Pydantic structured-output schemas
- Explicit `SKILL.md` support with eager and on-demand loading
- Optional Model Context Protocol bridge

## Requirements

- Python 3.11 or newer
- An API key for a hosted provider, or a local OpenAI-compatible server

## Installation

```bash
git clone https://github.com/delijati/rojnik
cd rojnik
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Install MCP support only when needed:

```bash
pip install -e ".[mcp]"
```

## Single Agent

```python
import asyncio

from rojnik import Agent
from rojnik.tools.builtins import list_directory, read_file

agent = Agent(
    name="file_reader",
    system_prompt="Read and summarize files accurately.",
    tools=[read_file, list_directory],
)

result = asyncio.run(agent.run("Summarize pyproject.toml"))
print(result)
```

The default provider is OpenAI and reads `OPENAI_API_KEY`.

## Agents As Tools

`Agent.as_tool()` exposes an agent as its own LLM function. The function name defaults to the agent name and accepts one required `task` string.

```python
import asyncio

from rojnik import Agent
from rojnik.tools.builtins import list_directory, read_file, shell_exec

file_agent = Agent(
    name="file_reader",
    system_prompt="Inspect files and return concise findings.",
    tools=[read_file, list_directory],
)

shell_agent = Agent(
    name="shell_executor",
    system_prompt="Run safe shell commands and explain their output.",
    tools=[shell_exec],
)

orchestrator = Agent(
    name="orchestrator",
    system_prompt=(
        "Coordinate file_reader and shell_executor. Call both in one response "
        "when their work can run in parallel."
    ),
    tools=[file_agent.as_tool(), shell_agent.as_tool()],
)

result = asyncio.run(orchestrator.run("Inspect the project and report its Python version."))
print(result)
```

You can override the exposed function metadata:

```python
tool = file_agent.as_tool(
    name="inspect_files",
    description="Inspect project files and return a concise report.",
)
```

Tool names must contain only letters, numbers, underscores, and hyphens. Every subagent run creates a child session linked to the calling agent's session.

## Dynamic Delegation

For a dynamic set of agents, expose one generic `delegate_to_agent(agent_name, task)` tool instead:

```python
from rojnik import Agent, AgentRegistry
from rojnik.tools.builtins import make_delegate_tool

registry = AgentRegistry()
registry.register_many(file_agent, shell_agent)

orchestrator = Agent(
    name="orchestrator",
    system_prompt="Delegate work to the registered specialists.",
    tools=[make_delegate_tool(registry)],
)
```

Use direct agent tools when the specialist set is stable and you want explicit schemas. Use the generic delegate tool when agents are registered dynamically.

## Custom Tools

```python
from rojnik import tool

@tool(description="Return the sum of two integers")
async def add(a: int, b: int) -> int:
    return a + b
```

Sync functions are supported and run in the event loop's thread executor.

## Skills

Agents can use explicitly configured Agent Skills. Each skill is a `SKILL.md`
file with YAML front matter and Markdown instructions:

```markdown
---
name: code-review
description: Review Python changes for correctness and missing tests.
---

# Code Review

Read the implementation and tests before reporting findings.
```

Pass either the file or its containing directory. Eager mode, the default,
adds complete instructions to the agent's system prompt:

```python
reviewer = Agent(
    name="reviewer",
    system_prompt="Review code carefully.",
    skills=["examples/skills/code-review"],
    skill_mode="eager",
)
```

On-demand mode initially exposes only skill names and descriptions. It adds a
constrained `load_skill(name)` tool that can load only configured skills:

```python
reviewer = Agent(
    name="reviewer",
    system_prompt="Load relevant skills before using them.",
    skills=["examples/skills/code-review/SKILL.md"],
    skill_mode="on_demand",
)
```

Skills are validated and loaded when the `Agent` is constructed. Reconstruct
the agent to pick up file changes. The default limits are 32 skills, 64 KiB
per file, and 256 KiB of combined instructions; constructor arguments can
lower or raise those limits.

Skill contents are trusted system-level instructions. In eager mode they are
persisted in the system message; in on-demand mode loaded instructions are
persisted as tool results. Both forms are sent to the configured LLM. Do not
load untrusted skills or put credentials in them. Skill loading never executes
referenced scripts or automatically reads sibling files.

## Providers

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
```

### DeepSeek

```bash
export AGENT_PROVIDER=deepseek
export DEEPSEEK_API_KEY=sk-...
export AGENT_MODEL=deepseek-chat
```

### Local OpenAI-Compatible Server

```bash
export AGENT_PROVIDER=local
export AGENT_BASE_URL=http://localhost:8080/v1
export AGENT_MODEL=qwen2.5-coder
```

Common settings:

```bash
export AGENT_DB_URL=sqlite+aiosqlite:///./agent.db
export AGENT_LOG_LEVEL=INFO
export AGENT_LOG_FILE=agent.log
```

## Streaming And Structured Output

```python
from pydantic import BaseModel

class Report(BaseModel):
    summary: str
    files_checked: int

chunks: list[str] = []
result = await agent.run(
    "Inspect this project",
    on_chunk=chunks.append,
    response_format=Report,
)
```

`result` remains the provider's final text, typically JSON when a structured response format is used.

## MCP

Install the optional dependency, then convert a connected MCP session's tools:

```python
from rojnik.mcp import mcp_to_tools

mcp_tools = await mcp_to_tools(session)
mcp_agent = Agent(
    name="mcp_agent",
    system_prompt="Use the connected MCP tools.",
    tools=mcp_tools,
)
```

See [`examples/README.md`](examples/README.md) for the optional MCP server command and runnable examples.

## Architecture

```text
Agent.run(task)
  -> create SQLite session
  -> build token-limited message context
  -> call LLM with registered function schemas
  -> execute returned tools concurrently
  -> persist assistant messages and tool results
  -> repeat until a final response or configured limit
```

Core modules:

| Module | Responsibility |
|---|---|
| `agent/agent.py` | `Agent`, `Agent.as_tool()`, and `AgentRegistry` |
| `agent/loop.py` | ReAct loop and concurrent tool dispatch |
| `tools/base.py` | `@tool` schema generation and validation |
| `tools/registry.py` | Tool registration and execution |
| `skills.py` | Validated `SKILL.md` loading and constrained skill tools |
| `memory/store.py` | Async SQLite persistence |
| `memory/context.py` | Token-budget context construction |
| `llm/client.py` | OpenAI-compatible async client, retry, and streaming |
| `mcp.py` | Optional MCP-to-rojnik tool bridge |

## Development

```bash
pip install -e ".[dev,mcp]"
ruff check .
pytest -q
python -m build
```

The test suite uses a mock LLM and does not make provider API calls. MCP tests launch the bundled local stdio server.

## Security

`shell_exec`, `read_file`, and `list_directory` operate with the permissions of the current process. Do not expose them to untrusted prompts without sandboxing or filesystem restrictions. The `examples/run_*_bwrap.sh` scripts demonstrate Linux-only bubblewrap isolation.

## License

[MIT](LICENSE)
