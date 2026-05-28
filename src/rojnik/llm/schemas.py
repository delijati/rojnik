"""
Pydantic models that map closely to the OpenAI Chat Completions API.

These are the canonical types passed around the harness. The LLMClient
converts between these and the raw openai SDK types internally.
"""


from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str


class ToolCallPart(BaseModel):
    """A single function call within an assistant message."""
    id: str                  # OpenAI call ID, e.g. "call_abc123"
    name: str                # Function name
    arguments: str           # Raw JSON string — validated by the tool registry


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None        # None when finish_reason == "tool_calls"
    tool_calls: list[ToolCallPart] = Field(default_factory=list)


class ToolResultMessage(BaseModel):
    """Carries a tool's output back to the model."""
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str             # Stringified result (or error message)


# Union of all message types that can appear in a conversation.
Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage


# ---------------------------------------------------------------------------
# LLM response
# ---------------------------------------------------------------------------

class LLMResponse(BaseModel):
    """Normalised response from a single chat completion call."""

    finish_reason: Literal["stop", "tool_calls", "length", "content_filter", "error"]

    # Populated when finish_reason == "stop"
    content: str | None = None

    # Populated when finish_reason == "tool_calls"
    tool_calls: list[ToolCallPart] = Field(default_factory=list)

    # Token usage reported by OpenAI
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Raw model name echoed back (useful for logging)
    model: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return self.finish_reason == "tool_calls" and bool(self.tool_calls)


# ---------------------------------------------------------------------------
# Tool schema (OpenAI function-calling format)
# ---------------------------------------------------------------------------

class ToolParameterSchema(BaseModel):
    """JSON Schema object describing a tool's parameters."""
    type: Literal["object"] = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class ToolFunctionSchema(BaseModel):
    name: str
    description: str
    parameters: ToolParameterSchema


class ToolSchema(BaseModel):
    """Full tool descriptor in the format expected by the OpenAI `tools=` param."""
    type: Literal["function"] = "function"
    function: ToolFunctionSchema

    def to_openai_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Response format (structured output)
# ---------------------------------------------------------------------------

# ResponseFormat can be:
#   - None                  → no constraint (default behaviour)
#   - {"type": "json_object"}
#   - {"type": "json_schema", "json_schema": {...}}
#   - A Pydantic BaseModel subclass → auto-converted to json_schema
ResponseFormat = type[BaseModel] | dict[str, Any] | None


def normalize_response_format(fmt: ResponseFormat) -> dict[str, Any] | None:
    """Convert a *ResponseFormat* value to the dict the OpenAI API expects.

    - ``None``           → ``None`` (no constraint)
    - ``dict``           → passed through unchanged
    - Pydantic subclass  → ``{"type": "json_schema", "json_schema": {...}}``
    """
    if fmt is None:
        return None
    if isinstance(fmt, dict):
        return fmt
    if isinstance(fmt, type) and issubclass(fmt, BaseModel):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": fmt.__name__,
                "strict": True,
                "schema": fmt.model_json_schema(),
            },
        }
    raise TypeError(f"Unsupported response_format type: {type(fmt)!r}")
