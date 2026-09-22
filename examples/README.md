# Examples

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Add MCP support when running the MCP example:

```bash
pip install -e ".[mcp]"
```

## Coding Agent

`coding_agent.py` builds an orchestrator with `file_reader` and `shell_executor` exposed as direct function tools through `Agent.as_tool()`.

```bash
OPENAI_API_KEY=sk-... python examples/coding_agent.py \
    "Read pyproject.toml and report the project version"
```

DeepSeek:

```bash
python examples/coding_agent.py \
    --provider deepseek \
    --api-key sk-... \
    "Inspect this repository"
```

Local OpenAI-compatible server:

```bash
python examples/coding_agent.py \
    --provider local \
    --base-url http://localhost:8080/v1 \
    --model qwen2.5-coder \
    "Inspect this repository"
```

### Skills

Load the bundled code-review skill eagerly into the file specialist's system
prompt:

```bash
OPENAI_API_KEY=sk-... python examples/coding_agent.py \
    --skill examples/skills/code-review \
    --skill-mode eager \
    "Review this repository"
```

Use on-demand mode to advertise the skill catalog and load complete
instructions only when selected:

```bash
OPENAI_API_KEY=sk-... python examples/coding_agent.py \
    --skill examples/skills/code-review/SKILL.md \
    --skill-mode on_demand \
    "Review this repository"
```

Repeat `--skill PATH` to configure multiple skills. Paths are explicit; the
examples do not scan the project or home directory automatically.

### Optional MCP Specialist

The `--mcp-server` value is one quoted command parsed with shell-like quoting. The bundled server exposes `get_time` and `roll_dice`:

```bash
OPENAI_API_KEY=sk-... python examples/coding_agent.py \
    --mcp-server "python examples/mcp_server.py" \
    "What time is it? Also roll 3d6."
```

Another stdio MCP server works the same way:

```bash
python examples/coding_agent.py \
    --mcp-server "/usr/local/bin/my-mcp-server --arg1 value1" \
    "Use the MCP tools to answer my question"
```

## Interactive Chat

```bash
OPENAI_API_KEY=sk-... python examples/chat.py
```

The Textual interface streams responses, displays tool activity, and stores history in `chat.db`. Inspect that history with:

```bash
python examples/viz.py --db chat.db
```

## Bubblewrap

The `run_chat_bwrap.sh` and `run_coding_agent_bwrap.sh` scripts are Linux-only examples for isolating filesystem and shell tools. They mount the current repository at `/work` inside the sandbox and currently expect a virtual environment named `venv` in the repository root.

## MCP Server

Run the bundled stdio server directly for use with an MCP client or inspector:

```bash
python examples/mcp_server.py
```
