"""
Interactive coding chat — Textual TUI backed by the rojnik.

Architecture
------------
  orchestrator
    ├── file_reader    (read_file, list_directory)
    └── shell_executor (shell_exec)

Tool calls from every agent are shown in the chat log as they start.
The final answer streams token-by-token into a preview bar above the input.

Provider quick-start
--------------------
  # OpenAI (default)
  OPENAI_API_KEY=sk-... /work/venv/bin/python examples/chat.py

  # DeepSeek
  /work/venv/bin/python examples/chat.py --provider deepseek --api-key sk-...

  # Local llama.cpp / Ollama
  /work/venv/bin/python examples/chat.py --provider local \\
      --base-url http://localhost:8080/v1 --model mistral

Keys
----
  enter    send message
  ctrl+c   cancel running agent, or quit when idle

Session history is written to --db (default: chat.db).
Browse it afterwards with:
  /work/venv/bin/python examples/viz.py --db chat.db
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# CLI args — must be parsed BEFORE importing rojnik so env vars are set
# ---------------------------------------------------------------------------

import argparse


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive coding chat backed by the rojnik.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "deepseek", "local"],
        default=None,
        help="LLM provider. Overrides AGENT_PROVIDER. Default: openai.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier. Overrides AGENT_MODEL.",
    )
    parser.add_argument(
        "--base-url",
        dest="base_url",
        default=None,
        help="Base URL for the Chat Completions endpoint. Overrides AGENT_BASE_URL.",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="API key for the provider. Overrides OPENAI_API_KEY / DEEPSEEK_API_KEY.",
    )
    parser.add_argument(
        "--db",
        default="chat.db",
        help="SQLite file for session history. Default: chat.db.",
    )
    return parser.parse_args()


def _apply_args_to_env(args: argparse.Namespace) -> None:
    """Push CLI overrides into os.environ before rojnik is imported."""
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
    # Set the DB path so config.Settings picks it up at import time
    db_path = str(Path(args.db).resolve())
    os.environ["AGENT_DB_URL"] = f"sqlite+aiosqlite:///{db_path}"

    # When running as a TUI, Textual owns the terminal — any direct write to
    # stderr from loguru corrupts the display.  Silence the console sink by
    # defaulting to CRITICAL (fires essentially never).  The JSON file sink
    # in logging_setup.py is hardcoded to DEBUG and is unaffected.
    # Use a separate log file so TUI logs don't mix with coding_agent.log.
    os.environ.setdefault("AGENT_LOG_LEVEL", "CRITICAL")
    os.environ.setdefault("AGENT_LOG_FILE", "chat.log")


_args = _parse_args()
_apply_args_to_env(_args)

# ---------------------------------------------------------------------------
# rojnik imports (after env setup)
# ---------------------------------------------------------------------------

sys.path.insert(0, "/work")

import rojnik  # noqa: F401  — triggers logging setup  # noqa: E402

from rojnik.agent.agent import Agent, AgentRegistry  # noqa: E402
from rojnik.llm.client import LLMClient  # noqa: E402
from rojnik.tools.builtins.delegate import make_delegate_tool  # noqa: E402
from rojnik.tools.builtins.files import list_directory, read_file  # noqa: E402
from rojnik.tools.builtins.shell import shell_exec  # noqa: E402

# ---------------------------------------------------------------------------
# Textual / Rich imports
# ---------------------------------------------------------------------------

from rich.markup import escape as markup_escape  # noqa: E402
from textual import on, work  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.containers import Horizontal  # noqa: E402
from textual.widgets import Footer, Header, Input, RichLog  # noqa: E402
from textual.worker import Worker  # noqa: E402

# ---------------------------------------------------------------------------
# Tool call formatter
# ---------------------------------------------------------------------------


def _format_tool_call(name: str, args_json: str) -> str:
    """Return a single Rich-markup line summarising a tool call."""
    try:
        args = json.loads(args_json)
    except Exception:
        args = {}

    if name == "delegate_to_agent":
        agent_name = markup_escape(str(args.get("agent_name", "?")))
        task = markup_escape(str(args.get("task", ""))[:72])
        return f"  [dim green][[>> {agent_name}]][/]  {task}"

    icons = {"shell_exec": "sh", "read_file": "read", "list_directory": "ls"}
    icon = icons.get(name, name)

    if name == "shell_exec":
        cmd = markup_escape(str(args.get("command", ""))[:80])
        return f"  [dim cyan][[{icon}]][/]  {cmd}"
    if name == "read_file":
        return f"  [dim cyan][[{icon}]][/]  {markup_escape(str(args.get('path', '')))}"
    if name == "list_directory":
        return f"  [dim cyan][[{icon}]][/]  {markup_escape(str(args.get('path', '.')))}"

    rest = markup_escape(args_json[:60]) if args_json else ""
    return f"  [dim cyan][[{icon}]][/]  {rest}"


# ---------------------------------------------------------------------------
# Agent harness setup
# ---------------------------------------------------------------------------


def build_harness(
    llm: LLMClient,
    on_tool_call: Callable[[str, str], None],
) -> Agent:
    """
    Assemble the multi-agent system and return the orchestrator.

    All three agents receive the same *on_tool_call* callback so tool
    calls from every level of the graph appear in the chat log.
    """
    file_agent = Agent(
        name="file_reader",
        system_prompt=(
            "You are a specialist at reading and summarising files and directories. "
            "Use your tools to explore the filesystem and return clear, concise results. "
            "Always use absolute or unambiguous paths."
        ),
        tools=[read_file, list_directory],
        llm=llm,
        on_tool_call=on_tool_call,
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
        on_tool_call=on_tool_call,
    )

    registry = AgentRegistry()
    registry.register_many(file_agent, shell_agent)

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
        on_tool_call=on_tool_call,
    )

    return orchestrator


# ---------------------------------------------------------------------------
# Textual app
# ---------------------------------------------------------------------------

_CSS = """
Screen {
    background: $surface-darken-1;
    layout: vertical;
}

#main {
    height: 1fr;
}

#log {
    width: 2fr;
    background: $surface;
    padding: 0 1;
    border-right: solid $panel-lighten-1;
}

#tools {
    width: 1fr;
    background: $surface-darken-1;
    padding: 0 1;
}

#stream {
    height: auto;
    max-height: 8;
    min-height: 2;
    background: $panel-darken-1;
    padding: 0 1;
    border-top: solid $primary-darken-3;
}

#input {
    background: $panel-darken-1;
    border: tall $panel-lighten-2;
}
"""


class ChatApp(App):
    TITLE = "agent-chat"
    CSS = _CSS
    BINDINGS = [Binding("ctrl+c", "cancel_or_quit", "Cancel/Quit", priority=True)]

    def __init__(self, db_path: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._db_path = db_path
        self._agent: Agent | None = None
        self._stream_buffer: list[str] = []
        self._current_worker: Worker | None = None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            yield RichLog(id="log", markup=True, wrap=True, highlight=False)
            yield RichLog(id="tools", markup=True, wrap=True, highlight=False)
        yield RichLog(id="stream", markup=True, wrap=True, highlight=False, auto_scroll=True)
        yield Input(placeholder="Type a message…", id="input", disabled=True)
        yield Footer()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self.sub_title = Path(self._db_path).name
        self._log: RichLog = self.query_one("#log", RichLog)
        self._tools: RichLog = self.query_one("#tools", RichLog)
        self._stream: RichLog = self.query_one("#stream", RichLog)
        self._input: Input = self.query_one("#input", Input)
        self._stream.display = False
        self._tools.write("[dim]activity[/]")
        self._setup_agent()

    # ------------------------------------------------------------------
    # Cancel / quit
    # ------------------------------------------------------------------

    def action_cancel_or_quit(self) -> None:
        """Ctrl+C: cancel the running agent if active, otherwise quit."""
        if self._current_worker is not None and self._current_worker.is_running:
            self._current_worker.cancel()
        else:
            self.exit()

    @work
    async def _setup_agent(self) -> None:
        """Initialise the DB and build the agent graph in a background worker."""
        from rojnik.memory.store import MemoryStore

        await MemoryStore.get()
        llm = LLMClient()
        self._agent = build_harness(llm, on_tool_call=self._on_tool_call)

        self._log.write(f"[dim]Ready  ·  DB: {self._db_path}[/]")
        self._log.write("[dim]Type a message and press enter.[/]")
        self._input.disabled = False
        self._input.focus()

    # ------------------------------------------------------------------
    # Tool-call display
    # ------------------------------------------------------------------

    def _on_tool_call(self, name: str, args_json: str) -> None:
        """Called synchronously from within the async agent worker."""
        self._tools.write(_format_tool_call(name, args_json))

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    @on(Input.Submitted, "#input")
    def _on_submit(self, event: Input.Submitted) -> None:
        task = event.value.strip()
        if not task or self._agent is None:
            return
        event.input.clear()
        event.input.disabled = True
        self._current_worker = self._run_agent(task)

    # ------------------------------------------------------------------
    # Agent worker
    # ------------------------------------------------------------------

    @work(exclusive=True)
    async def _run_agent(self, task: str) -> None:
        # Blank line separates exchanges in the chat log
        self._log.write("")
        self._log.write(f"[bold cyan][[you]][/]  {markup_escape(task)}")
        # Dim separator in the activity panel so tool calls are grouped by exchange
        preview = (task[:48] + "…") if len(task) > 48 else task
        self._tools.write(f"[dim]── {markup_escape(preview)}[/]")
        self._stream_buffer.clear()
        self._stream.display = True
        self._stream.clear()
        self._stream.write("[dim]thinking…[/]")

        def on_chunk(text: str) -> None:
            self._stream_buffer.append(text)
            full = "".join(self._stream_buffer)
            self._stream.clear()
            self._stream.write(
                f"[bold yellow][[agent]][/]  {markup_escape(full)}[bold yellow]▋[/]"
            )

        try:
            result = await self._agent.run(task, on_chunk=on_chunk)
            final = "".join(self._stream_buffer) or result or ""
            self._stream.display = False
            self._stream.clear()
            self._log.write(
                f"[bold yellow][[agent]][/]  "
                f"{markup_escape(final or '(no response)')}"
            )
        except asyncio.CancelledError:
            self._stream.display = False
            self._stream.clear()
            self._log.write("[dim](stopped)[/]")
            raise
        except Exception as exc:
            self._stream.display = False
            self._stream.clear()
            self._log.write(f"[bold red][[error]][/]  {markup_escape(str(exc))}")
        finally:
            self._input.disabled = False
            self._input.focus()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    db_path = str(Path(_args.db).resolve())
    ChatApp(db_path=db_path).run()


if __name__ == "__main__":
    main()
