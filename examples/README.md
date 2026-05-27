# examples

## Prerequisites

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

---

## coding_agent.py

Multi-agent coding assistant.  Architecture:

```
orchestrator
  ├── file_reader     (read_file, list_directory)
  ├── shell_executor  (shell_exec)
  └── mcp_agent       (any MCP server tools — only when --mcp-server is given)
```

### OpenAI

```bash
OPENAI_API_KEY=sk-... python examples/coding_agent.py
```

Custom task:

```bash
OPENAI_API_KEY=sk-... python examples/coding_agent.py \
    "Read pyproject.toml and tell me the project version"
```

### DeepSeek

```bash
python examples/coding_agent.py \
    --provider deepseek \
    --api-key sk-...
```

Uses `deepseek-chat` by default. For the reasoning model:

```bash
python examples/coding_agent.py \
    --provider deepseek \
    --model deepseek-reasoner \
    --api-key sk-...
```

### Local — llama.cpp

Start the llama.cpp server with a model that supports tool calling:

```bash
llama-server \
    --model ~/.cache/llama/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
    --port 8080
```

Then run the agent:

```bash
python examples/coding_agent.py \
    --provider local \
    --base-url http://localhost:8080/v1 \
    --model qwen2.5-coder
```

### With an MCP server

Pass `--mcp-server COMMAND [ARGS...]` to spawn any MCP-compatible server and
bridge its tools into the harness as a dedicated `mcp_agent` specialist.

`mcp_server.py` is a bundled demo server that exposes `get_time` and
`roll_dice`:

```bash
OPENAI_API_KEY=sk-... python examples/coding_agent.py \
    --mcp-server python examples/mcp_server.py \
    "What time is it? Also roll 3d6 for me."
```

Any compliant MCP server works the same way:

```bash
OPENAI_API_KEY=sk-... python examples/coding_agent.py \
    --mcp-server /usr/local/bin/my-mcp-server --arg1 val1 \
    "Use the server tools to answer my question"
```

The `mcp` package must be installed (`pip install mcp`; included in
`pip install -e ".[dev]"`).

---

## mcp_server.py

Standalone demo MCP server (stdio transport).  Exposes two tools:

| Tool | Description |
|------|-------------|
| `get_time()` | Returns the current UTC time in ISO 8601 format |
| `roll_dice(sides, count)` | Simulates rolling dice and returns individual rolls and total |

Intended to be used with `coding_agent.py --mcp-server`, but it can connect
to any MCP client (e.g. the [MCP Inspector](https://github.com/modelcontextprotocol/inspector)):

```bash
python examples/mcp_server.py
```

---

## run_coding_agent_bwrap.sh

Runs `coding_agent.py` inside a bubblewrap sandbox. Same flags as above,
including `--mcp-server` (the subprocess is spawned inside the sandbox).

```bash
chmod +x examples/run_coding_agent_bwrap.sh

# DeepSeek
./examples/run_coding_agent_bwrap.sh \
    --provider deepseek \
    --api-key sk-...

# llama.cpp (server must be running on the host)
./examples/run_coding_agent_bwrap.sh \
    --provider local \
    --base-url http://localhost:8080/v1 \
    --model qwen2.5-coder

# With the demo MCP server
OPENAI_API_KEY=sk-... ./examples/run_coding_agent_bwrap.sh \
    --mcp-server python examples/mcp_server.py \
    "What time is it?"
```

The current directory is bind-mounted read-write at `/work` inside the sandbox.

---

## viz.py

Terminal call-tree visualiser for SQLite run databases produced by agent runs.

Requires `textual>=8.0` (included in `pip install -e ".[dev]"`).

```bash
# view ./agent.db (default)
python3 examples/viz.py

# view a specific database
python3 examples/viz.py --db /path/to/agent.db
```

Keys:

| Key | Action |
|-----|--------|
| `↑` / `↓` | Navigate sessions / tree nodes |
| `enter` | Expand / collapse tree node |
| `r` | Refresh from database |
| `q` | Quit |
