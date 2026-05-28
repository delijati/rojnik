#!/usr/bin/env python3
"""
viz.py — terminal call-tree visualiser for rojnik SQLite runs.

Usage
-----
    python3 examples/viz.py                   # reads ./agent.db
    python3 examples/viz.py --db /path/to/agent.db

Keys
----
    ↑ / ↓        navigate sessions / tree nodes
    enter        expand / collapse node
    r            refresh from database
    q            quit
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static, Tree
from textual.widgets.tree import TreeNode


# ---------------------------------------------------------------------------
# Data layer — plain sqlite3, no SQLAlchemy needed
# ---------------------------------------------------------------------------

@dataclass
class SessionRow:
    id: str
    agent_name: str
    parent_session_id: str | None
    status: str
    task_preview: str | None
    result_preview: str | None
    created_at: str
    finished_at: str | None
    total_tokens: int

    @property
    def duration_s(self) -> str:
        if not self.finished_at:
            return "…"
        try:
            fmt = "%Y-%m-%d %H:%M:%S.%f"
            t0 = datetime.strptime(self.created_at[:26], fmt)
            t1 = datetime.strptime(self.finished_at[:26], fmt)
            secs = (t1 - t0).total_seconds()
            return f"{secs:.1f}s"
        except Exception:
            return "?"

    @property
    def created_hm(self) -> str:
        try:
            return self.created_at[11:16]
        except Exception:
            return ""


def _row_to_session(row: tuple) -> SessionRow:
    return SessionRow(*row)


def db_connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def load_root_sessions(con: sqlite3.Connection) -> list[SessionRow]:
    cur = con.execute(
        """
        SELECT id, agent_name, parent_session_id, status,
               task_preview, result_preview, created_at, finished_at, total_tokens
          FROM sessions
         WHERE parent_session_id IS NULL
         ORDER BY created_at DESC
        """
    )
    return [SessionRow(**dict(r)) for r in cur.fetchall()]


def load_children(con: sqlite3.Connection, parent_id: str) -> list[SessionRow]:
    cur = con.execute(
        """
        SELECT id, agent_name, parent_session_id, status,
               task_preview, result_preview, created_at, finished_at, total_tokens
          FROM sessions
         WHERE parent_session_id = ?
         ORDER BY created_at
        """,
        (parent_id,),
    )
    return [SessionRow(**dict(r)) for r in cur.fetchall()]


def load_messages(con: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
    cur = con.execute(
        """
        SELECT id, seq, role, content, tool_calls_json, tool_call_id, token_count
          FROM messages
         WHERE session_id = ?
         ORDER BY seq
        """,
        (session_id,),
    )
    return [dict(r) for r in cur.fetchall()]


def load_tool_results(con: sqlite3.Connection, session_id: str) -> dict[str, dict]:
    """Returns a mapping tool_call_id → tool_result row."""
    cur = con.execute(
        """
        SELECT tool_call_id, tool_name, input_json, output, error, duration_ms
          FROM tool_results
         WHERE session_id = ?
        """,
        (session_id,),
    )
    return {r["tool_call_id"]: dict(r) for r in cur.fetchall()}


# ---------------------------------------------------------------------------
# Rich label helpers
# ---------------------------------------------------------------------------

STATUS_STYLE = {
    "completed":      "bold green",
    "running":        "bold yellow",
    "error":          "bold red",
    "max_iterations": "bold red",
}

# Statuses that represent a terminated-with-error run
_ERROR_STATUSES = {"error", "max_iterations"}

ROLE_STYLE = {
    "system":    ("dim",    "SYS "),
    "user":      ("cyan",   "USR "),
    "assistant": ("yellow", "AST "),
    "tool":      ("green",  "TOOL"),
}


def _status_badge(status: str) -> str:
    style = STATUS_STYLE.get(status, "white")
    label = "max-iter" if status == "max_iterations" else status
    return f"[{style}]{label}[/]"


def _session_label(s: SessionRow) -> str:
    """Single-line label used in the call tree."""
    badge  = _status_badge(s.status)
    tok    = f"[dim]{s.total_tokens:,} tok[/]"
    dur    = f"[dim]{s.duration_s}[/]"
    time_s = f"  [dim]{s.created_hm}[/]" if s.created_hm else ""
    # Append a short error hint so it's readable inline in the tree
    err_hint = ""
    if s.status in _ERROR_STATUSES and s.result_preview:
        hint = s.result_preview.replace("\n", " ")[:50]
        err_hint = f"  [dim red]{hint}[/]"
    return f"[bold]{s.agent_name}[/]  {badge}  {tok}  {dur}{time_s}{err_hint}"


def _session_list_text(s: SessionRow) -> str:
    """Multi-line rich text card for the session list panel."""
    badge  = _status_badge(s.status)
    tok    = f"[dim]{s.total_tokens:,} tok[/]"
    dur    = f"[dim]{s.duration_s}[/]"
    time_s = f"[dim]{s.created_hm}[/]  " if s.created_hm else ""

    parts = [
        f"{time_s}[bold]{s.agent_name}[/]  {badge}",
        f"  {tok}  {dur}",
    ]

    if s.task_preview:
        flat  = s.task_preview.replace("\n", " ")
        short = flat[:60].rstrip() + ("…" if len(flat) > 60 else "")
        parts.append(f"  [dim cyan]Task:[/] [dim]{short}[/]")

    if s.result_preview:
        flat  = s.result_preview.replace("\n", " ")
        short = flat[:60].rstrip() + ("…" if len(flat) > 60 else "")
        if s.status in _ERROR_STATUSES:
            parts.append(f"  [bold red]Error:[/] [dim red]{short}[/]")
        else:
            parts.append(f"  [dim green]Result:[/] [dim]{short}[/]")
    elif s.status in _ERROR_STATUSES:
        parts.append(f"  [bold red]Error:[/] [dim red](no detail stored)[/]")

    if s.status == "running" and not s.finished_at:
        parts.append("  [yellow dim](abandoned — never closed)[/]")

    return "\n".join(parts)


def _truncate(text: str | None, width: int = 80) -> str:
    if not text:
        return "[dim](empty)[/]"
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[:width] + "[dim]…[/]"


def _message_label(msg: dict) -> str:
    role = msg["role"]
    style, tag = ROLE_STYLE.get(role, ("white", role[:4].upper()))
    content = _truncate(msg.get("content") or "", 72)
    tok = f"[dim]{msg['token_count']}t[/] " if msg["token_count"] else ""
    return f"[{style}][{tag}][/] {tok}{content}"


def _tool_call_label(tc: dict, result: dict | None) -> str:
    name = tc.get("name", "?")
    dur  = f"  [dim]{result['duration_ms']}ms[/]" if result else ""
    err  = "  [red][err][/]" if (result and result.get("error")) else ""
    return f"[magenta][CALL][/] [bold]{name}[/]{dur}{err}"


# ---------------------------------------------------------------------------
# Session panel (left)
# ---------------------------------------------------------------------------

class SessionPanel(Vertical):
    """Left panel — list of root sessions."""

    DEFAULT_CSS = """
    SessionPanel {
        width: 28%;
        border-right: tall $panel-lighten-2;
    }
    SessionPanel > Label {
        padding: 0 1;
        background: $panel-darken-1;
        color: $text-muted;
        text-style: bold;
        width: 100%;
    }
    SessionPanel ListView {
        height: 1fr;
        background: $surface;
        overflow-y: auto;
    }
    SessionPanel ListItem {
        padding: 0 1 1 1;
        border-bottom: dashed $panel-lighten-1;
    }
    """

    def __init__(self, sessions: list[SessionRow], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sessions = sessions

    def compose(self) -> ComposeResult:
        yield Label(" Sessions")
        yield ListView(id="session-list")

    def _make_items(self) -> list[ListItem]:
        items = []
        for s in self._sessions:
            item = ListItem(Static(_session_list_text(s)))
            item._session_id = s.id  # type: ignore[attr-defined]
            items.append(item)
        return items

    def on_mount(self) -> None:
        self.call_after_refresh(self._populate)

    async def _populate(self) -> None:
        lv = self.query_one("#session-list", ListView)
        await lv.clear()
        for item in self._make_items():
            lv.append(item)

    async def refresh_sessions(self, sessions: list[SessionRow]) -> None:
        self._sessions = sessions
        await self._populate()


# ---------------------------------------------------------------------------
# Tree panel (right)
# ---------------------------------------------------------------------------

class TreePanel(Vertical):
    """Right panel — expandable call tree."""

    DEFAULT_CSS = """
    TreePanel {
        width: 100%;
        height: 1fr;
    }
    TreePanel > Label {
        padding: 0 1;
        background: $panel-darken-1;
        color: $text-muted;
        text-style: bold;
        width: 100%;
    }
    TreePanel Tree {
        height: 1fr;
        background: $surface;
        padding: 0 1;
        overflow-y: auto;
    }
    """

    def __init__(self, con: sqlite3.Connection, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._con = con

    def compose(self) -> ComposeResult:
        yield Label(" Call Tree")
        yield Tree("(select a session)", id="call-tree")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_session(self, session_id: str) -> None:
        """Build the tree for a root session."""
        tree: Tree = self.query_one("#call-tree", Tree)
        tree.clear()
        cur = self._con.execute(
            """SELECT id, agent_name, parent_session_id, status,
                      task_preview, result_preview, created_at, finished_at, total_tokens
                 FROM sessions WHERE id = ?""",
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            tree.root.set_label("session not found")
            return

        s = SessionRow(**dict(row))
        tree.root.set_label(_session_label(s))
        tree.root.data = {"type": "session", "row": s, "loaded": False}
        tree.root.expand()

    # ------------------------------------------------------------------
    # Lazy expansion
    # ------------------------------------------------------------------

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        node: TreeNode = event.node
        data: dict | None = node.data
        if data is None or data.get("loaded"):
            return

        ntype = data.get("type")
        if ntype == "session":
            self._expand_session(node, data["row"])
        elif ntype == "messages":
            self._expand_messages(node, data["session_id"])

    def _expand_session(self, node: TreeNode, s: SessionRow) -> None:
        node.data["loaded"] = True  # type: ignore[index]

        # task / result / error info lines
        if s.task_preview:
            task_text = s.task_preview.replace("\n", " ")
            node.add_leaf(
                f"[dim]Task:[/] {_truncate(task_text, 90)}",
                data={"type": "text_leaf", "title": "Task", "text": s.task_preview},
            )
        if s.result_preview:
            if s.status in _ERROR_STATUSES:
                node.add_leaf(
                    f"[bold red]Error:[/] {_truncate(s.result_preview, 90)}",
                    data={"type": "text_leaf", "title": "Error", "text": s.result_preview},
                )
            else:
                node.add_leaf(
                    f"[dim]Result:[/] {_truncate(s.result_preview, 90)}",
                    data={"type": "text_leaf", "title": "Result", "text": s.result_preview},
                )

        # messages container
        msgs = load_messages(self._con, s.id)
        if msgs:
            m_node = node.add(
                f"[dim]Messages ({len(msgs)})[/]",
                data={"type": "messages", "session_id": s.id, "loaded": False},
            )
            m_node.allow_expand = True

        # child sessions
        children = load_children(self._con, s.id)
        if children:
            sub_node = node.add("[dim]Subagents[/]", data={"type": "group", "loaded": True})
            sub_node.expand()
            for child in children:
                cn = sub_node.add(
                    _session_label(child),
                    data={"type": "session", "row": child, "loaded": False},
                )
                cn.allow_expand = True

    def _expand_messages(self, node: TreeNode, session_id: str) -> None:
        node.data["loaded"] = True  # type: ignore[index]

        msgs         = load_messages(self._con, session_id)
        tool_results = load_tool_results(self._con, session_id)

        for msg in msgs:
            role         = msg["role"]
            full_content = msg.get("content") or ""
            msg_data = {
                "type":        "msg",
                "role":        role,
                "full_content": full_content,
                "token_count": msg.get("token_count") or 0,
            }

            if role == "assistant" and msg.get("tool_calls_json"):
                try:
                    calls = json.loads(msg["tool_calls_json"])
                except Exception:
                    calls = []
                msg_data["tool_calls"] = calls
                a_node = node.add(_message_label(msg), data=msg_data)
                for tc in calls:
                    result = tool_results.get(tc.get("id", ""))
                    a_node.add_leaf(
                        _tool_call_label(tc, result),
                        data={
                            "type":       "tool_call",
                            "name":       tc.get("name", "?"),
                            "full_args":  tc.get("arguments") or "",
                            "full_output": (result.get("output") or "") if result else "",
                            "full_error":  (result.get("error")  or "") if result else "",
                            "duration_ms": result.get("duration_ms") if result else None,
                        },
                    )
            else:
                node.add_leaf(_message_label(msg), data=msg_data)


# ---------------------------------------------------------------------------
# Content pane (bottom-right) — full text of highlighted tree node
# ---------------------------------------------------------------------------

class ContentPane(Vertical):
    """Shows full, untruncated content of whichever tree node is highlighted."""

    DEFAULT_CSS = """
    ContentPane {
        width: 100%;
        height: 1fr;
        border-top: tall $panel-lighten-2;
    }
    ContentPane > Label {
        padding: 0 1;
        background: $panel-darken-1;
        color: $text-muted;
        text-style: bold;
        width: 100%;
    }
    ContentPane ScrollableContainer {
        height: 1fr;
        background: $surface;
        overflow-y: auto;
    }
    ContentPane Static {
        padding: 0 1;
        width: 100%;
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label(" Detail")
        with ScrollableContainer():
            yield Static(
                "[dim]Navigate the tree to view full content.[/]",
                id="detail-body",
            )

    def show_text(self, text: str) -> None:
        self.query_one("#detail-body", Static).update(text)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

class VizApp(App):
    TITLE = "rojnik · call tree"
    CSS = """
    Screen {
        background: $surface-darken-1;
    }
    #body {
        height: 1fr;
    }
    #right-col {
        width: 72%;
    }
    """
    BINDINGS = [
        Binding("q", "quit",    "Quit",    priority=True),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self, db_path: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._db_path = db_path
        self._con     = db_connect(db_path)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        sessions = load_root_sessions(self._con)
        with Horizontal(id="body"):
            yield SessionPanel(sessions, id="session-panel")
            with Vertical(id="right-col"):
                yield TreePanel(self._con, id="tree-panel")
                yield ContentPane(id="content-pane")
        yield Footer()

    def on_mount(self) -> None:
        short = Path(self._db_path).name
        self.sub_title = short
        # auto-select after the first frame so SessionPanel._populate has run
        self.call_after_refresh(self._auto_select)

    def _auto_select(self) -> None:
        lv = self.query_one("#session-list", ListView)
        if lv.children:
            lv.index = 0
            first_item = lv.children[0]
            sid = getattr(first_item, "_session_id", None)
            if sid:
                self.query_one("#tree-panel", TreePanel).load_session(sid)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    @on(ListView.Highlighted, "#session-list")
    def session_selected(self, event: ListView.Highlighted) -> None:
        if event.item is None:
            return
        sid = getattr(event.item, "_session_id", None)
        if sid:
            self.query_one("#tree-panel", TreePanel).load_session(sid)

    @on(Tree.NodeHighlighted, "#call-tree")
    def node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        pane = self.query_one("#content-pane", ContentPane)
        data: dict | None = event.node.data
        if not data:
            pane.show_text("")
            return

        ntype = data.get("type")

        if ntype == "session":
            s: SessionRow = data["row"]
            finished = s.finished_at[11:16] if s.finished_at else None
            fin_str  = f"  finished {finished}" if finished else "  [yellow](not closed)[/]"
            parts = [
                f"[bold]{s.agent_name}[/]  {_status_badge(s.status)}"
                f"  [dim]{s.total_tokens:,} tok  {s.duration_s}{fin_str}[/]",
            ]
            if s.status in _ERROR_STATUSES:
                err_text = s.result_preview or "(no detail stored)"
                parts += ["", "[bold red]── Error ──[/]", f"[red]{err_text}[/]"]
                parts += ["", "[dim]This run ended with an error and was not resumed.[/]",
                          "[dim]Any work not yet delegated was skipped.[/]"]
            else:
                if s.task_preview:
                    parts += ["", "[dim]── Task ──[/]", s.task_preview]
                if s.result_preview:
                    parts += ["", "[dim]── Result ──[/]", s.result_preview]
            pane.show_text("\n".join(parts))

        elif ntype == "text_leaf":
            title = data.get("title", "")
            text  = data.get("text") or ""
            pane.show_text(f"[dim]── {title} ──[/]\n\n{text}")

        elif ntype == "msg":
            role  = data.get("role", "")
            style, tag = ROLE_STYLE.get(role, ("white", "MSG "))
            tok   = data.get("token_count") or 0
            tok_s = f"  [dim]{tok} tokens[/]" if tok else ""
            body  = data.get("full_content") or "[dim](no content)[/]"
            pane.show_text(f"[{style}][{tag}][/]{tok_s}\n\n{body}")

        elif ntype == "tool_call":
            name     = data.get("name", "?")
            dur      = data.get("duration_ms")
            dur_s    = f"  [dim]{dur}ms[/]" if dur is not None else ""
            parts    = [f"[magenta][CALL][/] [bold]{name}[/]{dur_s}"]
            raw_args = data.get("full_args") or ""
            if raw_args:
                try:
                    raw_args = json.dumps(json.loads(raw_args), indent=2)
                except Exception:
                    pass
                parts += ["", "[dim]── Arguments ──[/]", raw_args]
            err = data.get("full_error") or ""
            out = data.get("full_output") or ""
            if err:
                parts += ["", "[red]── Error ──[/]", err]
            elif out:
                parts += ["", "[dim]── Output ──[/]", out]
            pane.show_text("\n".join(parts))

        else:
            pane.show_text("")

    async def action_refresh(self) -> None:
        self._con.close()
        self._con = db_connect(self._db_path)
        sessions = load_root_sessions(self._con)
        panel = self.query_one("#session-panel", SessionPanel)
        await panel.refresh_sessions(sessions)
        # update the tree's connection too
        tree_panel = self.query_one("#tree-panel", TreePanel)
        tree_panel._con = self._con
        self.notify("Refreshed from database.", timeout=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Terminal call-tree visualiser for rojnik SQLite runs."
    )
    parser.add_argument(
        "--db",
        default="agent.db",
        metavar="PATH",
        help="Path to the SQLite database (default: ./agent.db)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: database not found: {db_path}")
        raise SystemExit(1)

    VizApp(str(db_path)).run()


if __name__ == "__main__":
    main()
