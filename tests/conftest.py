"""
Shared fixtures for the rojnik test suite.

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

import rojnik  # noqa: F401  — triggers logging setup

from rojnik.llm.schemas import LLMResponse, ToolCallPart
from rojnik.memory.store import MemoryStore


# ---------------------------------------------------------------------------
# Singleton reset — runs around EVERY test automatically
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singletons():
    """Wipe module-level singletons so every test starts clean."""
    import rojnik.agent.agent as agent_module

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

    When *on_chunk* is supplied the stub simulates streaming by calling
    it once per character of the response content (if any).  The recorded
    call dict includes ``response_format`` so tests can assert it was
    forwarded correctly.
    """

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def chat(self, messages, tools=None, *, on_chunk=None, response_format=None):
        self.calls.append({
            "messages": list(messages),
            "tools": tools,
            "response_format": response_format,
        })
        if not self._responses:
            raise RuntimeError("MockLLMClient: no more scripted responses")
        resp = self._responses.pop(0)
        # Simulate streaming: call on_chunk once per character of content
        if on_chunk is not None and resp.content:
            for ch in resp.content:
                on_chunk(ch)
        return resp


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
