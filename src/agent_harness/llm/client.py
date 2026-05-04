"""
Async LLM client wrapper.

Responsibilities:
  - Hold a single AsyncOpenAI instance.
  - Provide a clean async chat() method: list[Message] + list[ToolSchema] → LLMResponse.
  - Retry on transient errors (rate limits, network) with exponential back-off.
  - Emit structured log events on every request and response.

Provider support
----------------
The client is OpenAI-compatible and supports any provider that speaks the
OpenAI Chat Completions API, including:
  - OpenAI          (default, base_url=None)
  - DeepSeek        (base_url="https://api.deepseek.com/v1")
  - Local servers   (Ollama, LM Studio, vLLM — supply a custom base_url)

All of these are driven by the global ``settings`` object (config.py), but
every parameter can also be passed directly to the constructor so that a
caller can create a fully custom client without touching env vars.
"""


import asyncio
import time
from typing import Any

from loguru import logger
from openai import AsyncOpenAI, APIConnectionError, APIStatusError, RateLimitError

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
    """
    Thin async wrapper around any OpenAI-compatible Chat Completions API.

    Parameters
    ----------
    api_key:
        API key for the provider.  Defaults to ``settings.get_api_key()``.
    model:
        Model identifier, e.g. "gpt-4o", "deepseek-chat", "llama3".
        Defaults to ``settings.model``.
    base_url:
        Base URL of the Chat Completions endpoint.  Pass ``None`` to use the
        official OpenAI API.  Examples:
          - DeepSeek:  "https://api.deepseek.com/v1"
          - Ollama:    "http://localhost:11434/v1"
          - LM Studio: "http://localhost:1234/v1"
        Defaults to ``settings.get_base_url()``.
    max_tokens:
        Maximum tokens in the completion.  Defaults to ``settings.max_tokens``.
    max_retries:
        Number of retry attempts on transient errors.
    retry_wait_seconds:
        Base wait time (doubled on each successive attempt).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
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

        # Resolve base_url: explicit arg → settings default → None (= OpenAI)
        resolved_base_url: str | None = base_url
        if resolved_base_url is None:
            resolved_base_url = settings.get_base_url()

        resolved_api_key = api_key or settings.get_api_key()

        client_kwargs: dict[str, Any] = {"api_key": resolved_api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        self._client = AsyncOpenAI(**client_kwargs)

        logger.debug(
            "llm.client.created",
            model=self._model,
            provider=settings.provider,
            base_url=resolved_base_url or "https://api.openai.com/v1",
        )

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
