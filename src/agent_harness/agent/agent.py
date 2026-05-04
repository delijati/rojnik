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
    from agent_harness.agent.agent import Agent, AgentRegistry

    file_agent = Agent(
        name="file_reader",
        system_prompt="You read and summarise files.",
        tools=[read_file, list_directory],
    )

    registry = AgentRegistry()
    registry.register(file_agent)

    result = await file_agent.run("Summarise /etc/hosts")
"""

from typing import Callable, Any

from loguru import logger

from agent_harness.agent.loop import run_loop
from agent_harness.agent.state import RunState
from agent_harness.llm.client import LLMClient
from agent_harness.llm.schemas import SystemMessage
from agent_harness.memory.context import ContextBuilder
from agent_harness.memory.store import MemoryStore
from agent_harness.tools.registry import ToolRegistry


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
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list[Callable[..., Any]] | None = None,
        llm: LLMClient | None = None,
        max_iterations: int | None = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

        # Tool registry — populated from the tools list
        self.tool_registry = ToolRegistry()
        for fn in tools or []:
            self.tool_registry.register(fn)

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

        # Persist the system message
        system_msg = SystemMessage(content=self.system_prompt)
        await store.add_message(session_id, system_msg)

        state = RunState(session_id=session_id, agent_name=self.name)

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
        )

        logger.info(
            "agent.run.done",
            agent=self.name,
            session_id=session_id,
            total_tokens=state.total_tokens,
        )
        return result

    def __repr__(self) -> str:
        return (
            f"Agent(name={self.name!r}, "
            f"tools={self.tool_registry.names()})"
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
