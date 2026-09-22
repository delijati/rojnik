"""
Agent class and AgentRegistry.

Agent
-----
An Agent is a named, reusable object that:
  - Has a system prompt describing its role.
  - Has a ToolRegistry with specific tools it can call.
  - Shares a single LLMClient and MemoryStore with the rest of the harness.
  - Exposes a `run(task)` coroutine that executes the ReAct loop.

AgentRegistry
-------------
A global mapping of agent name → Agent, used by the delegate tool
so any agent can spawn any other by name.

Usage
-----
    from rojnik.agent.agent import Agent, AgentRegistry

    file_agent = Agent(
        name="file_reader",
        system_prompt="You read and summarise files.",
        tools=[read_file, list_directory],
    )

    registry = AgentRegistry()
    registry.register(file_agent)

    result = await file_agent.run("Summarise /etc/hosts")
"""

import asyncio
import re
from collections.abc import Callable, Sequence
from typing import Any, Literal

from loguru import logger

from rojnik.agent.loop import run_loop
from rojnik.agent.state import RunState
from rojnik.llm.client import LLMClient
from rojnik.llm.schemas import ResponseFormat, SystemMessage
from rojnik.memory.context import ContextBuilder
from rojnik.memory.store import MemoryStore
from rojnik.skills import (
    DEFAULT_MAX_SKILL_BYTES,
    DEFAULT_MAX_SKILLS,
    DEFAULT_MAX_TOTAL_SKILL_BYTES,
    SkillSource,
    compose_skill_prompt,
    load_skills,
    make_load_skill_tool,
)
from rojnik.tools.base import tool
from rojnik.tools.registry import ToolRegistry

_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class Agent:
    """
    A configured agent: system prompt + tools + shared infrastructure.

    Parameters
    ----------
    name:
        Short identifier used in logging and the agent registry.
    system_prompt:
        Instruction text sent as the system message on every run.
    tools:
        List of @tool-decorated callables the agent may invoke.
    llm:
        Optional LLMClient override (defaults to a shared instance).
    max_iterations:
        Override the global max_iterations for this agent.
    on_tool_call:
        If provided, called once per tool call just before dispatch with
        ``(tool_name, arguments_json)``.  Stored on the instance so it
        fires for sub-agents too (sub-agents are called via
        ``delegate_to_agent`` without extra kwargs).
    skills:
        Explicit ``SKILL.md`` paths, directories containing ``SKILL.md``, or
        preloaded Skill objects. Skills are validated at construction time.
    skill_mode:
        ``"eager"`` includes full instructions in the system prompt.
        ``"on_demand"`` advertises the catalog and adds ``load_skill``.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list[Callable[..., Any]] | None = None,
        llm: LLMClient | None = None,
        max_iterations: int | None = None,
        on_tool_call: Callable[[str, str], None] | None = None,
        skills: Sequence[SkillSource] | None = None,
        skill_mode: Literal["eager", "on_demand"] = "eager",
        max_skill_bytes: int = DEFAULT_MAX_SKILL_BYTES,
        max_total_skill_bytes: int = DEFAULT_MAX_TOTAL_SKILL_BYTES,
        max_skills: int = DEFAULT_MAX_SKILLS,
    ) -> None:
        if skill_mode not in ("eager", "on_demand"):
            raise ValueError("skill_mode must be 'eager' or 'on_demand'")
        self.name = name
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self._on_tool_call = on_tool_call
        self.skills = load_skills(
            skills or (),
            max_skill_bytes=max_skill_bytes,
            max_total_skill_bytes=max_total_skill_bytes,
            max_skills=max_skills,
        )
        self.skill_mode = skill_mode

        # Tool registry — populated from the tools list
        self.tool_registry = ToolRegistry()
        for fn in tools or []:
            self.tool_registry.register(fn)
        if self.skills and skill_mode == "on_demand":
            if "load_skill" in self.tool_registry:
                raise ValueError(
                    "Tool name 'load_skill' is reserved when skill_mode='on_demand'."
                )
            self.tool_registry.register(make_load_skill_tool(self.skills))

        # LLM client — shared singleton by default
        self._llm = llm or _shared_llm()

        # Context builder — shared settings
        self._context_builder = ContextBuilder()

        logger.debug(
            "agent.created",
            name=name,
            n_tools=len(self.tool_registry),
            tools=self.tool_registry.names(),
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(
        self,
        task: str,
        parent_session_id: str | None = None,
        *,
        on_chunk: Callable[[str], None] | None = None,
        response_format: ResponseFormat = None,
    ) -> str:
        """
        Execute one task through the ReAct loop.

        Parameters
        ----------
        task:
            The user's request for this agent.
        parent_session_id:
            If this agent was spawned by another agent's delegate tool,
            pass the parent session ID for traceability in SQLite.
        on_chunk:
            If provided, streamed text tokens are forwarded here as they
            arrive from the LLM.  See ``LLMClient.chat()`` for details.
        response_format:
            Optional structured-output constraint forwarded to the LLM on
            every call.  See ``LLMClient.chat()`` for details.

        Returns
        -------
        The agent's final text response.
        """
        store = await MemoryStore.get()

        # Create a new session in SQLite
        session_id = await store.create_session(
            agent_name=self.name,
            task=task,
            parent_session_id=parent_session_id,
        )

        state = RunState(session_id=session_id, agent_name=self.name)

        try:
            effective_prompt = compose_skill_prompt(
                self.system_prompt,
                self.skills,
                eager=self.skill_mode == "eager",
            )
            await store.add_message(session_id, SystemMessage(content=effective_prompt))

            logger.info(
                "agent.run.start",
                agent=self.name,
                session_id=session_id,
                parent_session_id=parent_session_id,
                task_preview=task[:120],
            )

            result = await run_loop(
                state=state,
                task=task,
                llm=self._llm,
                tool_registry=self.tool_registry,
                store=store,
                context_builder=self._context_builder,
                max_iterations=self.max_iterations,
                on_chunk=on_chunk,
                on_tool_call=self._on_tool_call,
                response_format=response_format,
            )
        except asyncio.CancelledError:
            state.status = "cancelled"
            await store.close_session(
                session_id,
                status="cancelled",
                result="Agent run cancelled",
                total_tokens=state.total_tokens,
            )
            raise
        except Exception as exc:
            if state.status == "running":
                state.status = "error"
                state.error = str(exc)
                await store.close_session(
                    session_id,
                    status="error",
                    result=f"{type(exc).__name__}: {exc}",
                    total_tokens=state.total_tokens,
                )
            raise

        logger.info(
            "agent.run.done",
            agent=self.name,
            session_id=session_id,
            total_tokens=state.total_tokens,
        )
        return result

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[..., Any]:
        """Expose this agent as an LLM function tool accepting a ``task`` string."""
        tool_name = name or self.name
        if not _TOOL_NAME_PATTERN.fullmatch(tool_name):
            raise ValueError(
                "Agent tool names may contain only letters, numbers, underscores, and hyphens."
            )

        async def run_agent(task: str) -> str:
            from rojnik.tools.builtins.delegate import current_session_id

            return await self.run(
                task,
                parent_session_id=current_session_id.get(),
            )

        run_agent.__name__ = tool_name
        return tool(
            description=description
            or f"Delegate a task to the {self.name} agent and return its complete response."
        )(run_agent)

    def __repr__(self) -> str:
        return (
            f"Agent(name={self.name!r}, "
            f"tools={self.tool_registry.names()}, "
            f"skills={[skill.name for skill in self.skills]!r})"
        )


# ---------------------------------------------------------------------------
# Shared LLM client singleton
# ---------------------------------------------------------------------------

_llm_instance: LLMClient | None = None


def _shared_llm() -> LLMClient:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMClient()
    return _llm_instance


# ---------------------------------------------------------------------------
# AgentRegistry
# ---------------------------------------------------------------------------

class AgentRegistry:
    """
    Global registry mapping agent name → Agent instance.
    Used by the delegate tool to look up agents by name.
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        if agent.name in self._agents:
            logger.warning("agent.registry.overwrite", agent_name=agent.name)
        self._agents[agent.name] = agent
        logger.debug("agent.registry.registered", agent_name=agent.name)

    def register_many(self, *agents: Agent) -> None:
        for agent in agents:
            self.register(agent)

    def get(self, name: str) -> Agent:
        if name not in self._agents:
            raise KeyError(
                f"No agent named {name!r} is registered. "
                f"Available agents: {list(self._agents.keys())}"
            )
        return self._agents[name]

    def names(self) -> list[str]:
        return list(self._agents.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __len__(self) -> int:
        return len(self._agents)

    def __repr__(self) -> str:
        return f"AgentRegistry(agents={self.names()})"
