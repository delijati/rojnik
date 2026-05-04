"""
Shared fixtures for the agent_harness test suite.

Key concerns
------------
- MemoryStore is a singleton; we reset it before and after every test so tests
  are fully isolated.
- Agent._shared_llm is also reset so tests that construct Agents don't
  accidentally share LLM state.
- MockLLMClient lets us script exact LLM responses without hitting any API.
"""


import os
import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")

import agent_harness  # noqa: F401  — triggers logging setup

from agent_harness.llm.schemas import LLMResponse, ToolCallPart
from agent_harness.memory.store import MemoryStore


# ---------------------------------------------------------------------------
# Singleton reset — runs around EVERY test automatically
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singletons():
    """Wipe module-level singletons so every test starts clean."""
    import agent_harness.agent.agent as agent_module

    MemoryStore._instance = None
    agent_module._llm_instance = None
    yield
    MemoryStore._instance = None
    agent_module._llm_instance = None


# ---------------------------------------------------------------------------
# Fresh SQLite store per test (backed by a temp file)
# ---------------------------------------------------------------------------

@pytest.fixture
async def store(tmp_path):
    db_file = tmp_path / "test.db"
    return await MemoryStore.get(db_url=f"sqlite+aiosqlite:///{db_file}")


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------

class MockLLMClient:
    """
    Scriptable LLM stub for tests.

    Instantiate with a list of LLMResponse objects.  Each call to chat()
    pops and returns the next one.  Raises RuntimeError if exhausted.
    """

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def chat(self, messages, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self._responses:
            raise RuntimeError("MockLLMClient: no more scripted responses")
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Convenience response builders
# ---------------------------------------------------------------------------

def stop_response(content: str = "All done.", tokens: int = 10) -> LLMResponse:
    return LLMResponse(
        finish_reason="stop",
        content=content,
        prompt_tokens=tokens,
        completion_tokens=tokens,
        total_tokens=tokens * 2,
        model="mock",
    )


def tool_call_response(
    calls: list[tuple[str, str, str]],  # (id, name, arguments_json)
) -> LLMResponse:
    return LLMResponse(
        finish_reason="tool_calls",
        tool_calls=[
            ToolCallPart(id=cid, name=name, arguments=args)
            for cid, name, args in calls
        ],
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        model="mock",
    )
