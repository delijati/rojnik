"""Tests for llm/schemas.py — Pydantic message and response models."""


import pytest
from pydantic import ValidationError

from agent_harness.llm.schemas import (
    AssistantMessage,
    LLMResponse,
    SystemMessage,
    ToolCallPart,
    ToolFunctionSchema,
    ToolParameterSchema,
    ToolResultMessage,
    ToolSchema,
    UserMessage,
)


class TestMessageTypes:
    def test_system_role_is_fixed(self):
        assert SystemMessage(content="sys").role == "system"

    def test_user_role_is_fixed(self):
        assert UserMessage(content="hi").role == "user"

    def test_assistant_defaults(self):
        m = AssistantMessage()
        assert m.role == "assistant"
        assert m.content is None
        assert m.tool_calls == []

    def test_assistant_with_tool_calls(self):
        tc = ToolCallPart(id="c1", name="add", arguments='{"a":1}')
        m = AssistantMessage(tool_calls=[tc])
        assert m.tool_calls[0].name == "add"

    def test_tool_result_message(self):
        m = ToolResultMessage(tool_call_id="c1", content="42")
        assert m.role == "tool"
        assert m.tool_call_id == "c1"

    def test_system_wrong_role_rejected(self):
        with pytest.raises(ValidationError):
            SystemMessage(role="user", content="x")  # type: ignore


class TestLLMResponse:
    def test_stop_has_no_tool_calls(self):
        r = LLMResponse(finish_reason="stop", content="hi")
        assert r.has_tool_calls is False

    def test_tool_calls_response_has_tool_calls(self):
        tc = ToolCallPart(id="x", name="fn", arguments="{}")
        r = LLMResponse(finish_reason="tool_calls", tool_calls=[tc])
        assert r.has_tool_calls is True

    def test_stop_with_calls_list_is_still_false(self):
        """has_tool_calls gates on finish_reason, not just the list."""
        tc = ToolCallPart(id="x", name="fn", arguments="{}")
        assert not LLMResponse(finish_reason="stop", tool_calls=[tc]).has_tool_calls

    def test_tool_calls_but_empty_list_is_false(self):
        assert not LLMResponse(finish_reason="tool_calls", tool_calls=[]).has_tool_calls

    def test_token_defaults_zero(self):
        r = LLMResponse(finish_reason="stop")
        assert r.prompt_tokens == r.completion_tokens == 0

    def test_invalid_finish_reason(self):
        with pytest.raises(ValidationError):
            LLMResponse(finish_reason="banana")  # type: ignore


class TestToolSchema:
    def _schema(self):
        return ToolSchema(
            type="function",
            function=ToolFunctionSchema(
                name="do_it",
                description="Does it",
                parameters=ToolParameterSchema(
                    properties={"x": {"type": "integer"}},
                    required=["x"],
                ),
            ),
        )

    def test_to_openai_dict_top_level(self):
        d = self._schema().to_openai_dict()
        assert d["type"] == "function"
        assert d["function"]["name"] == "do_it"

    def test_required_present_in_dict(self):
        d = self._schema().to_openai_dict()
        assert "x" in d["function"]["parameters"]["required"]
