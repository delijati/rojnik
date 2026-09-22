"""
MemoryStore — async CRUD operations over SQLite via SQLAlchemy async.

One MemoryStore instance is shared across the entire harness (singleton).
It lazily creates the database and tables on first use.

Usage
-----
    store = await MemoryStore.get()

    session_id = await store.create_session("orchestrator", task="Fix the bug")
    await store.add_message(session_id, role="user", content="Fix the bug")
    history = await store.get_messages(session_id)
    await store.close_session(session_id, status="completed", result="Done")
"""


import json
import uuid
from datetime import UTC, datetime
from typing import Any, Optional

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from rojnik.config import settings
from rojnik.llm.schemas import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
)
from rojnik.memory.models import Base, DBMessage, Session, ToolResult


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MemoryStore:
    """Async SQLAlchemy store backed by SQLite."""

    _instance: Optional["MemoryStore"] = None

    def __init__(self, engine: AsyncEngine, async_session_factory: Any) -> None:
        self._engine = engine
        self._session_factory = async_session_factory

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    async def get(cls, db_url: str | None = None) -> "MemoryStore":
        """Return the shared MemoryStore, creating it on first call."""
        if cls._instance is None:
            url = db_url or settings.db_url
            engine = create_async_engine(url, echo=False)
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            cls._instance = cls(engine, factory)
            logger.debug("memory.store.initialised", db_url=url)
        return cls._instance

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    async def create_session(
        self,
        agent_name: str,
        task: str | None = None,
        parent_session_id: str | None = None,
    ) -> str:
        session_id = _new_id()
        async with self._session_factory() as db:
            async with db.begin():
                db.add(
                    Session(
                        id=session_id,
                        agent_name=agent_name,
                        parent_session_id=parent_session_id,
                        status="running",
                        task_preview=task[:500] if task else None,
                    )
                )
        logger.debug(
            "memory.session.created",
            session_id=session_id,
            agent_name=agent_name,
            parent_session_id=parent_session_id,
        )
        return session_id

    async def close_session(
        self,
        session_id: str,
        status: str = "completed",
        result: str | None = None,
        total_tokens: int = 0,
    ) -> None:
        async with self._session_factory() as db:
            async with db.begin():
                await db.execute(
                    update(Session)
                    .where(Session.id == session_id)
                    .values(
                        status=status,
                        result_preview=result[:500] if result else None,
                        finished_at=_utcnow(),
                        total_tokens=total_tokens,
                    )
                )
        logger.debug(
            "memory.session.closed",
            session_id=session_id,
            status=status,
            total_tokens=total_tokens,
        )

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    async def _next_seq(self, db: AsyncSession, session_id: str) -> int:
        result = await db.execute(
            select(DBMessage.seq)
            .where(DBMessage.session_id == session_id)
            .order_by(DBMessage.seq.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        return (row or 0) + 1

    async def add_message(
        self,
        session_id: str,
        msg: Message,
        token_count: int = 0,
    ) -> str:
        msg_id = _new_id()
        async with self._session_factory() as db:
            async with db.begin():
                seq = await self._next_seq(db, session_id)
                role = msg.role
                content: str | None = None
                tool_calls_json: str | None = None
                tool_call_id: str | None = None

                if role in ("system", "user"):
                    content = msg.content  # type: ignore[union-attr]
                elif role == "assistant":
                    assert isinstance(msg, AssistantMessage)
                    content = msg.content
                    if msg.tool_calls:
                        tool_calls_json = json.dumps(
                            [tc.model_dump() for tc in msg.tool_calls]
                        )
                elif role == "tool":
                    assert isinstance(msg, ToolResultMessage)
                    content = msg.content
                    tool_call_id = msg.tool_call_id

                db.add(
                    DBMessage(
                        id=msg_id,
                        session_id=session_id,
                        seq=seq,
                        role=role,
                        content=content,
                        tool_calls_json=tool_calls_json,
                        tool_call_id=tool_call_id,
                        token_count=token_count,
                    )
                )
        return msg_id

    async def get_messages(self, session_id: str) -> list[Message]:
        """Return all messages for a session, reconstructed as Message objects."""
        async with self._session_factory() as db:
            result = await db.execute(
                select(DBMessage)
                .where(DBMessage.session_id == session_id)
                .order_by(DBMessage.seq)
            )
            rows = result.scalars().all()

        messages: list[Message] = []
        for row in rows:
            if row.role == "system":
                messages.append(SystemMessage(content=row.content or ""))
            elif row.role == "user":
                messages.append(UserMessage(content=row.content or ""))
            elif row.role == "assistant":
                tool_calls: list[ToolCallPart] = []
                if row.tool_calls_json:
                    for tc_dict in json.loads(row.tool_calls_json):
                        tool_calls.append(ToolCallPart(**tc_dict))
                messages.append(
                    AssistantMessage(content=row.content, tool_calls=tool_calls)
                )
            elif row.role == "tool":
                messages.append(
                    ToolResultMessage(
                        tool_call_id=row.tool_call_id or "",
                        content=row.content or "",
                    )
                )
        return messages

    # ------------------------------------------------------------------
    # Tool results
    # ------------------------------------------------------------------

    async def save_tool_result(
        self,
        session_id: str,
        tool_call_id: str,
        tool_name: str,
        input_json: str | None = None,
        output: str | None = None,
        error: str | None = None,
        duration_ms: int = 0,
    ) -> None:
        async with self._session_factory() as db:
            async with db.begin():
                db.add(
                    ToolResult(
                        id=_new_id(),
                        session_id=session_id,
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        input_json=input_json,
                        output=output[:4096] if output else None,
                        error=error,
                        duration_ms=duration_ms,
                    )
                )
