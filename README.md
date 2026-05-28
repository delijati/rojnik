# rojnik

A minimal, from-scratch async agentic harness in Python. No LangChain. No LangGraph. Just OpenAI, Pydantic, SQLAlchemy, and Loguru — wired together deliberately.

## Table of Contents

- [Architecture](#architecture)
- [Component Breakdown](#component-breakdown)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Alternative Providers](#alternative-providers)
- [Testing Against terminal-bench](#testing-against-terminal-bench)
- [Suggested Improvements](#suggested-improvements)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User / Entry Point                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  agent.run(task)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Agent (orchestrator)                        │
│  system_prompt · ToolRegistry([delegate_to_agent])                  │
└──────────┬───────────────────────────────────────────────┬──────────┘
           │                                               │
           ▼                                               ▼
    ┌─────────────┐                                ┌─────────────┐
    │  file_agent │                                │ shell_agent │
    │  read_file  │                                │ shell_exec  │
    │  list_dir   │                                └─────────────┘
    └─────────────┘
           │  both agents share ▼
┌──────────┴───────────────────────────────────────────────┐
│                       run_loop()                          │
│                                                           │
│  while iterations < max:                                  │
│    window  = context_builder.build(session_id, store)     │
│    response = await llm.chat(window, tool_schemas)        │
│                                                           │
│    if stop      → persist → return content                │
│    if tool_calls → asyncio.gather(dispatch all) → loop    │
│    if other     → persist error → return                  │
└───────────┬──────────────────────────┬────────────────────┘
            │                          │
            ▼                          ▼
  ┌──────────────────┐      ┌─────────────────────┐
  │    LLMClient     │      │    MemoryStore       │
  │  AsyncOpenAI     │      │  SQLite + SQLAlchemy │
  │  retry / log     │      │  sessions            │
  └──────────────────┘      │  messages            │
                            │  tool_results        │
  ┌──────────────────┐      └─────────────────────┘
  │  ContextBuilder  │
  │  tiktoken trim   │      ┌─────────────────────┐
  └──────────────────┘      │  Loguru              │
                            │  console (human)     │
  ┌──────────────────┐      │  file (JSON)         │
  │  ToolRegistry    │      └─────────────────────┘
  │  @tool decorator │
  │  dispatch()      │
  └──────────────────┘
```

### Multi-Agent Data Flow

When the orchestrator calls `delegate_to_agent("file_reader", task)`:

```
orchestrator.run_loop
  └─ LLM returns tool_calls: [delegate("file_reader", ...), delegate("shell_executor", ...)]
       └─ asyncio.gather(
            file_agent.run(task, parent_session_id=orchestrator_session),
            shell_agent.run(task, parent_session_id=orchestrator_session),
          )
            └─ each subagent runs its own run_loop
            └─ subagent sessions are linked to parent via parent_session_id in SQLite
       └─ both results returned as tool result messages
  └─ LLM sees both results → returns final "stop" response
```

---

## Component Breakdown

| File | Role |
|---|---|
| `config.py` | `Settings` dataclass — all knobs in one place, overridable via env vars |
| `logging_setup.py` | Two Loguru sinks: coloured human console + rotating newline-delimited JSON file |
| `llm/schemas.py` | Canonical Pydantic types: `Message`, `LLMResponse`, `ToolSchema` |
| `llm/client.py` | `LLMClient` — `async chat()`, exponential backoff retry on 429 / 5xx |
| `tools/base.py` | `@tool` decorator: inspects signature → builds OpenAI JSON schema + Pydantic validator |
| `tools/registry.py` | `ToolRegistry` — per-agent `{name: (schema, fn)}`, `dispatch()` returns `(id, str)` |
| `tools/builtins/files.py` | `read_file`, `list_directory` |
| `tools/builtins/shell.py` | `shell_exec` — async subprocess, configurable timeout |
| `tools/builtins/delegate.py` | `make_delegate_tool(registry)` — multi-agent bridge via `contextvars` |
| `memory/models.py` | SQLAlchemy ORM: `Session`, `DBMessage`, `ToolResult` |
| `memory/store.py` | `MemoryStore` — async CRUD singleton, lazy DB init |
| `memory/context.py` | `ContextBuilder` — tiktoken-based token-budget trimming, oldest-first |
| `agent/state.py` | `RunState` — live Pydantic state for one loop (not persisted directly) |
| `agent/loop.py` | `run_loop()` — the ReAct loop, decoupled from Agent for easy testing |
| `agent/agent.py` | `Agent` + `AgentRegistry` |

### Key Design Decisions

**`@tool` is just a decorator.** No base classes, no registries to import. Decorating a function generates its OpenAI schema and wraps it with Pydantic validation automatically. Sync functions run in `asyncio`'s thread executor so they never block the event loop.

**`run_loop()` is a plain async function.** It takes the LLM client, tool registry, store, and context builder as arguments. This makes it trivially testable with a mock LLM — no monkey-patching required.

**`delegate_to_agent` uses `contextvars`.** When the loop calls `asyncio.gather()` across multiple parallel subagent delegations, each coroutine gets its own `ContextVar` value for `current_session_id`. No global mutation, no race conditions.

**SQLite stores the full execution tree.** Every session records a `parent_session_id`. You can reconstruct the entire call graph for any run with a single recursive CTE query.

---

## Installation

```bash
# From source (editable, for development)
git clone https://github.com/your-org/rojnik
cd rojnik
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# From PyPI (once published)
pip install rojnik
```

Set your API key for the provider you intend to use:

```bash
# OpenAI (default)
export OPENAI_API_KEY=sk-...

# DeepSeek
export AGENT_PROVIDER=deepseek
export DEEPSEEK_API_KEY=sk-...

# llama.cpp — no key needed, just point at the server
export AGENT_PROVIDER=local
export AGENT_BASE_URL=http://localhost:8080/v1
export AGENT_MODEL=qwen2.5-coder
```

Optional environment variables:

```bash
export AGENT_MODEL=gpt-4o          # default: per provider
export AGENT_DB_URL=sqlite+aiosqlite:///./agent.db
export AGENT_LOG_LEVEL=DEBUG
export AGENT_LOG_FILE=agent.log
```

---

## Quick Start

### Single agent

```python
import asyncio
from rojnik.agent.agent import Agent
from rojnik.tools.builtins.files import read_file, list_directory
from rojnik.tools.builtins.shell import shell_exec

agent = Agent(
    name="coder",
    system_prompt="You are a senior software engineer.",
    tools=[read_file, list_directory, shell_exec],
)

result = asyncio.run(agent.run("List the Python files in /tmp and count them."))
print(result)
```

### Multi-agent (orchestrator pattern)

```python
import asyncio
from rojnik.agent.agent import Agent, AgentRegistry
from rojnik.tools.builtins.files import read_file, list_directory
from rojnik.tools.builtins.shell import shell_exec
from rojnik.tools.builtins.delegate import make_delegate_tool

# Specialist agents
file_agent = Agent(
    name="file_reader",
    system_prompt="You read and summarise files and directories.",
    tools=[read_file, list_directory],
)
shell_agent = Agent(
    name="shell_executor",
    system_prompt="You run shell commands and interpret their output.",
    tools=[shell_exec],
)

registry = AgentRegistry()
registry.register_many(file_agent, shell_agent)

# Orchestrator only knows how to delegate
orchestrator = Agent(
    name="orchestrator",
    system_prompt=(
        "You coordinate specialist agents to complete tasks. "
        "Available: file_reader, shell_executor."
    ),
    tools=[make_delegate_tool(registry)],
)

result = asyncio.run(orchestrator.run("What Python version is installed?"))
print(result)
```

### Custom tool

```python
from rojnik.tools.base import tool

@tool(description="Fetch the title of a web page")
async def fetch_title(url: str) -> str:
    import httpx
    async with httpx.AsyncClient() as client:
        r = await client.get(url, timeout=10)
    from html.parser import HTMLParser
    # ... extract <title> ...
    return r.text[:200]

agent = Agent(name="researcher", system_prompt="...", tools=[fetch_title])
```

---

## Alternative Providers

`LLMClient` wraps the `openai` Python SDK and accepts a `base_url` parameter,
so any OpenAI-compatible endpoint works without subclassing.

### DeepSeek

```bash
AGENT_PROVIDER=deepseek DEEPSEEK_API_KEY=sk-... python examples/coding_agent.py
```

```python
from rojnik.llm.client import LLMClient

llm = LLMClient(
    api_key="sk-...",
    model="deepseek-chat",          # or deepseek-reasoner
    base_url="https://api.deepseek.com/v1",
)
```

### llama.cpp

Start the llama.cpp server with a model that supports tool calling:

```bash
llama-server \
    --model ~/.cache/llama/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
    --port 8080
```

```bash
AGENT_PROVIDER=local \
AGENT_BASE_URL=http://localhost:8080/v1 \
AGENT_MODEL=qwen2.5-coder \
    python examples/coding_agent.py
```

```python
from rojnik.llm.client import LLMClient

llm = LLMClient(
    model="qwen2.5-coder",
    base_url="http://localhost:8080/v1",
    # no api_key needed unless the server enforces one
)
```

---

## Testing Against terminal-bench

[terminal-bench](https://github.com/tcapelle/terminal_benchmark) is a benchmark suite of tasks that require an agent to accomplish real goals in a sandboxed terminal environment. Tasks range from shell scripting to software installation to data processing. Each task has a verifier that checks the environment state after the agent finishes.

### Adapter

Create a thin adapter that maps terminal-bench's task API to `rojnik.Agent.run()`:

```python
# examples/terminal_bench_adapter.py
import asyncio
import json
from pathlib import Path

from rojnik.agent.agent import Agent
from rojnik.tools.builtins.files import list_directory, read_file
from rojnik.tools.builtins.shell import shell_exec


def make_coding_agent() -> Agent:
    return Agent(
        name="terminal_agent",
        system_prompt=(
            "You are an expert at completing terminal-based tasks. "
            "You have access to shell execution, file reading, and directory listing. "
            "Complete the task precisely as described. "
            "When you are done, say so explicitly."
        ),
        tools=[shell_exec, read_file, list_directory],
        max_iterations=30,
    )


async def run_task(task_description: str, workdir: str = ".") -> str:
    """Run a single terminal-bench task and return the agent's response."""
    agent = make_coding_agent()
    # terminal-bench typically sets a working directory per task;
    # pass it as context in the task description or via a custom tool.
    return await agent.run(
        f"Working directory: {workdir}\n\nTask: {task_description}"
    )


async def run_benchmark(tasks_file: str) -> dict:
    """
    Run all tasks from a terminal-bench tasks JSON file.

    Expected tasks_file format (one task per line, JSONL):
        {"id": "task_001", "description": "...", "workdir": "/sandbox/task_001"}

    Returns a dict mapping task_id → agent_response.
    """
    results = {}
    tasks = [
        json.loads(line)
        for line in Path(tasks_file).read_text().splitlines()
        if line.strip()
    ]

    for task in tasks:
        print(f"Running task {task['id']}...")
        try:
            response = await run_task(task["description"], task.get("workdir", "."))
            results[task["id"]] = {"status": "ok", "response": response}
        except Exception as exc:
            results[task["id"]] = {"status": "error", "error": str(exc)}

    return results


if __name__ == "__main__":
    import sys
    tasks_file = sys.argv[1] if len(sys.argv) > 1 else "tasks.jsonl"
    results = asyncio.run(run_benchmark(tasks_file))
    print(json.dumps(results, indent=2))
```

### Running terminal-bench

```bash
# 1. Install terminal-bench
pip install terminal-bench    # or clone from the repo

# 2. Run evaluation
python examples/terminal_bench_adapter.py tasks.jsonl > results.jsonl

# 3. Score results with terminal-bench's evaluator
tb-eval score --results results.jsonl --tasks tasks.jsonl
```

### Things to tune for benchmark runs

| Setting | Recommendation |
|---|---|
| `max_iterations` | Increase to 30–50 for complex tasks |
| `context_token_budget` | Keep high (128k+) so prior tool results aren't trimmed |
| `model` | `gpt-4o` or `deepseek-chat` for best tool-calling accuracy |
| `shell_exec` timeout | Increase to 120s for long-running commands |
| System prompt | Be explicit: "When the task is complete, output DONE." |

### Parsing success from agent output

terminal-bench verifiers check the environment state directly (not the agent's text output), so you generally don't need to parse the agent's response — just let the verifier run. However, if you want to do token-efficiency analysis:

```python
import re

def extract_done_signal(response: str) -> bool:
    """Check if the agent explicitly declared it finished."""
    return bool(re.search(r"\bDONE\b|task.{0,20}complete", response, re.IGNORECASE))
```

---

## Suggested Improvements

These are the most impactful additions, roughly ordered by value.

### 1. Streaming responses

Currently the harness waits for the full response before processing. Streaming lets you show partial output to the user and detect tool calls earlier.

```python
# In LLMClient.chat(), use stream=True and accumulate chunks:
async with self._client.chat.completions.stream(...) as stream:
    async for event in stream:
        yield event   # or buffer and yield assembled chunks
```

Requires changing `run_loop` to handle an async generator from `chat()`.

### 2. Structured output / constrained decoding

For tasks where the agent must return data in a specific schema (e.g., a JSON report), use OpenAI's `response_format={"type": "json_schema", "json_schema": ...}` or Pydantic's `.model_json_schema()` to enforce the output structure. This eliminates brittle string parsing.

### 3. Long-term semantic memory

The current `ContextBuilder` does recency-based trimming — old messages fall off. For long-running agents, add a vector store (ChromaDB, Qdrant, pgvector) so the agent can retrieve relevant past context by semantic similarity rather than recency.

```
memory/
├── context.py          ← current recency trimmer
├── vector_store.py     ← NEW: embed + retrieve by similarity
└── hybrid_context.py   ← NEW: recency window + semantic recall merged
```

### 4. Tool call result caching

Identical tool calls (same name + same arguments) within a session often produce the same result. A simple `dict` cache keyed on `(tool_name, arguments_hash)` avoids redundant file reads and subprocess calls, which meaningfully reduces latency on tasks with repetitive exploration.

### 5. Agent-level timeouts and budget limits

The loop has a max-iteration guard, but no wall-clock timeout or token-cost budget. Add these to `RunState` and check them each iteration:

```python
@dataclass
class BudgetPolicy:
    max_wall_seconds: float = 300.0
    max_total_tokens: int = 500_000
    max_tool_cost_usd: float = 1.00
```

### 6. Interrupt / human-in-the-loop

Add a hook that the loop calls after each iteration before making the next LLM call. If the hook returns `"pause"`, the loop serialises its state to SQLite and exits. A subsequent call can resume from the saved state.

```python
async def run_loop(..., on_iteration=None):
    ...
    if on_iteration:
        signal = await on_iteration(state)
        if signal == "pause":
            await store.save_checkpoint(state)
            return
```

### 7. Parallel sub-task fan-out from the orchestrator

Right now, parallel execution requires the LLM to emit multiple tool calls in a single response. An alternative is to let the orchestrator explicitly request parallel execution via a `run_parallel(tasks: list[dict])` tool that fans out and `asyncio.gather`s the results automatically.

### 8. Evaluation and replay

Every run is already persisted in SQLite. Add a replay mode that re-runs a session with a different model or prompt to compare outputs without re-executing tools — useful for prompt engineering and model upgrades.

### 9. Proper package extras for providers

```toml
[project.optional-dependencies]
deepseek = []          # uses the openai SDK — no extra install needed
local    = []          # llama.cpp / any OpenAI-compatible server
```

### 10. Observability with OpenTelemetry

The Loguru JSON file is a good start, but for production use, emit spans to an OTLP backend (Jaeger, Grafana Tempo, Honeycomb). Each `run_loop` iteration becomes a span; tool calls become child spans with input/output attributes.
