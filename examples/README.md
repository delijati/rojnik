# examples

## Prerequisites

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

---

## coding_agent.py

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

---

## run_coding_agent_bwrap.sh

Runs `coding_agent.py` inside a bubblewrap sandbox. Same flags as above.

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
