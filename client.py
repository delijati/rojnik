"""
Async OpenAI client wrapper.

Responsibilities:
  - Hold a single AsyncOpenAI instance.
  - Provide a clean async chat() method: list[Message] + list[ToolSchema] → LLMResponse.
  - Retry on transient errors (rate limits, network) with exponential back-off.
  - Emit structured log events on every request and response.
"""

import time
import asyncio

from typing import Any

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError

from loguru import logger
from agent_harness.config import settings
from agent_harness.llm.schemas import (
    AssistantMessage,
    LLMResponse,
    Message,
    ToolCallPart,
    ToolSchema,
)


def _messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert our internal Message types to the plain dicts the openai SDK expects."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.role
        if role in ("system", "user"):
            result.append({"role": role, "content": msg.content})  # type: ignore[union-attr]
        elif role == "assistant":
            assert isinstance(msg, AssistantMessage)
            entry: dict[str, Any] = {"role": "assistant"}
            if msg.content:
                entry["content"] = msg.content
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            result.append(entry)
        elif role == "tool":
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,  # type: ignore[union-attr]
                    "content": msg.content,  # type: ignore[union-attr]
                }
            )
    return result


class LLMClient:
    """Thin async wrapper around the OpenAI Chat Completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        max_retries: int | None = None,
        retry_wait_seconds: float | None = None,
    ) -> None:
        self._model = model or settings.model
        self._max_tokens = max_tokens or settings.max_tokens
        self._max_retries = max_retries if max_retries is not None else settings.max_retries
        self._retry_wait = (
            retry_wait_seconds if retry_wait_seconds is not None else settings.retry_wait_seconds
        )
        self._client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion request and return a normalised LLMResponse.

        Retries up to max_retries times on rate-limit and connection errors.
        """
        tools_payload = [t.to_openai_dict() for t in tools] if tools else None
        openai_messages = _messages_to_openai(messages)

        logger.debug(
            "llm.request",
            model=self._model,
            n_messages=len(openai_messages),
            n_tools=len(tools_payload) if tools_payload else 0,
        )

        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 2):  # +2 so last attempt is attempt N+1
            t0 = time.perf_counter()
            try:
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "messages": openai_messages,
                    "max_tokens": self._max_tokens,
                }
                if tools_payload:
                    kwargs["tools"] = tools_payload
                    kwargs["tool_choice"] = "auto"

                response = await self._client.chat.completions.create(**kwargs)

                latency_ms = int((time.perf_counter() - t0) * 1000)
                choice = response.choices[0]
                usage = response.usage

                # --- Parse tool calls ---
                tool_calls: list[ToolCallPart] = []
                if choice.message.tool_calls:
                    for tc in choice.message.tool_calls:
                        tool_calls.append(
                            ToolCallPart(
                                id=tc.id,
                                name=tc.function.name,
                                arguments=tc.function.arguments,
                            )
                        )

                finish_reason = choice.finish_reason or "stop"

                llm_response = LLMResponse(
                    finish_reason=finish_reason,  # type: ignore[arg-type]
                    content=choice.message.content,
                    tool_calls=tool_calls,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    model=response.model,
                )

                logger.debug(
                    "llm.response",
                    finish_reason=finish_reason,
                    n_tool_calls=len(tool_calls),
                    prompt_tokens=llm_response.prompt_tokens,
                    completion_tokens=llm_response.completion_tokens,
                    latency_ms=latency_ms,
                    attempt=attempt,
                )
                return llm_response

            except RateLimitError as exc:
                last_error = exc
                wait = self._retry_wait * (2 ** (attempt - 1))
                logger.warning(
                    "llm.rate_limit — retrying",
                    attempt=attempt,
                    wait_s=wait,
                )
                await asyncio.sleep(wait)

            except APIConnectionError as exc:
                last_error = exc
                wait = self._retry_wait * (2 ** (attempt - 1))
                logger.warning(
                    "llm.connection_error — retrying",
                    attempt=attempt,
                    wait_s=wait,
                    error=str(exc),
                )
                await asyncio.sleep(wait)

            except APIStatusError as exc:
                # 5xx errors are retryable; 4xx (except 429) are not
                if exc.status_code >= 500:
                    last_error = exc
                    wait = self._retry_wait * (2 ** (attempt - 1))
                    logger.warning(
                        "llm.server_error — retrying",
                        status=exc.status_code,
                        attempt=attempt,
                        wait_s=wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error("llm.client_error", status=exc.status_code, body=exc.body)
                    raise

        logger.error("llm.max_retries_exceeded", max_retries=self._max_retries)
        raise RuntimeError(f"LLM call failed after {self._max_retries} retries") from last_error
