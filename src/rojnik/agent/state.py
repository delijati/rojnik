"""
RunState — the live state of a single agent run.

This is a Pydantic model that is passed through the loop and mutated
on each iteration. It is NOT persisted directly (the MemoryStore handles
persistence); it's the working memory for one run.
"""


from typing import Literal

from pydantic import BaseModel, Field

from rojnik.llm.schemas import Message

RunStatus = Literal["running", "completed", "error", "cancelled", "max_iterations"]


class RunState(BaseModel):
    """Live state for one pass through the agent loop."""

    # Identity
    session_id: str
    agent_name: str

    # Conversation window (system prompt + trimmed history + current turn)
    messages: list[Message] = Field(default_factory=list)

    # Loop control
    iteration: int = 0
    status: RunStatus = "running"

    # Accumulated token usage for this run
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    # Final answer once the loop exits cleanly
    result: str | None = None
    error: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def add_tokens(self, prompt: int, completion: int) -> None:
        self.total_prompt_tokens += prompt
        self.total_completion_tokens += completion
