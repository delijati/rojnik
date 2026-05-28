"""
SQLAlchemy ORM models for persistent agent memory.

Tables
------
sessions     — one row per agent run (supports nested subagent calls).
messages     — every message in a session, in order.
tool_results — one row per tool invocation within a session.

The parent_session_id FK on sessions lets you reconstruct the full
execution tree when a subagent is spawned by a parent agent.
"""


from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Session(Base):
    """One agent run."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    agent_name: Mapped[str] = mapped_column(String(128), nullable=False)
    parent_session_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="running"
    )  # running | completed | error | max_iterations
    task_preview: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_preview: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)

    messages: Mapped[list["DBMessage"]] = relationship(
        "DBMessage", back_populates="session", order_by="DBMessage.seq"
    )
    tool_results: Mapped[list["ToolResult"]] = relationship(
        "ToolResult", back_populates="session"
    )


class DBMessage(Base):
    """A single message in a session's conversation history."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)  # ordering within session
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # For assistant messages with tool_calls: serialised JSON list
    tool_calls_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # For tool result messages
    tool_call_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    session: Mapped[Session] = relationship("Session", back_populates="messages")


class ToolResult(Base):
    """One tool invocation within a session."""

    __tablename__ = "tool_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    tool_call_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    input_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    output: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    session: Mapped[Session] = relationship("Session", back_populates="tool_results")
