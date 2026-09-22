# AGENTS.md

## Project

`rojnik` is a small async Python agent harness. Keep the implementation explicit and framework-free. Prefer focused changes over new abstractions.

## Layout

- `src/rojnik/agent/`: agent configuration, run state, and ReAct loop
- `src/rojnik/llm/`: OpenAI-compatible client and canonical schemas
- `src/rojnik/tools/`: tool decorator, registry, and built-in tools
- `src/rojnik/memory/`: SQLite models, persistence, and context trimming
- `src/rojnik/skills.py`: validated `SKILL.md` loading and prompt/tool integration
- `src/rojnik/mcp.py`: optional MCP bridge
- `tests/`: unit and integration tests using a mock LLM
- `examples/`: runnable CLI, TUI, MCP, and sandbox examples

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,mcp]"
```

## Verification

Run these before submitting changes:

```bash
ruff check .
pytest -q
python -m build
```

Tests must not call hosted LLM APIs. Use `MockLLMClient` from `tests/conftest.py`.

## Design Rules

- Keep `run_loop()` independent from `Agent` so it remains easy to test.
- Tools must be async callables branded by `@tool`, MCP conversion, or `Agent.as_tool()`.
- Preserve assistant tool-call messages and matching tool-result messages as an atomic context group.
- Always restore `ContextVar` tokens in `finally` blocks.
- Every created session must end as `completed`, `error`, `cancelled`, or `max_iterations`.
- Preserve parent session linkage when agents invoke agents.
- Direct agent tools use one required `task: str` argument.
- Use `make_delegate_tool()` when the set of agents is dynamic; use `Agent.as_tool()` for explicit per-agent schemas.
- Keep MCP optional. Core imports and non-MCP examples must work without the `mcp` package.
- Skills use explicit paths, required YAML `name`/`description`, and trusted Markdown instructions.
- Keep skill filesystem access constrained to the configured catalog; never accept model-provided paths.
- Reject oversized skill instructions instead of silently truncating them.
- Do not make network access, filesystem access, or shell execution less explicit.

## Style

- Target Python 3.11+.
- Use type hints for public APIs.
- Prefer small functions and direct control flow.
- Add tests for behavior changes and failure paths.
- Keep docs and executable examples aligned with the actual API.
- Do not commit databases, logs, virtual environments, caches, `dist/`, or generated egg-info.

## Safety

`shell_exec`, `read_file`, and `list_directory` use host-process permissions. Treat them as trusted-environment tools unless the caller provides sandboxing. Never add credentials, API keys, local database contents, or logs to the repository.

`SKILL.md` contents become privileged instructions and are sent to the configured LLM. Only load trusted skills, and never execute skill-referenced scripts implicitly.
