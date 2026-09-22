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
  - Local servers   (llama.cpp — supply a custom base_url)

All of these are driven by the global ``settings`` object (config.py), but
every parameter can also be passed directly to the constructor so that a
caller can create a fully custom client without touching env vars.
"""


import asyncio
import time
from collections.abc import Callable
from typing import Any

from loguru import logger
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError

from rojnik.config import settings
from rojnik.llm.schemas import (
    AssistantMessage,
    LLMResponse,
    Message,
    ResponseFormat,
    ToolCallPart,
    ToolSchema,
    normalize_response_format,
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
          - DeepSeek:   "https://api.deepseek.com/v1"
          - llama.cpp:  "http://localhost:8080/v1"
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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        on_chunk: Callable[[str], None] | None = None,
        response_format: ResponseFormat = None,
    ) -> LLMResponse:
        """
        Send a chat completion request and return a normalised LLMResponse.

        Parameters
        ----------
        messages:
            Conversation history to send.
        tools:
            Optional list of tool schemas the model may call.
        on_chunk:
            If provided, enables **streaming mode**.  The callable is invoked
            once per text token as the model streams its reply.  No retry is
            performed once streaming has started; a single attempt is made.
        response_format:
            Optional structured-output constraint.  May be:

              - ``None``                    – no constraint (default).
              - ``dict``                    – passed through as-is
                                              (e.g. ``{"type": "json_object"}``).
              - Pydantic ``BaseModel`` subclass – auto-converted to a
                ``json_schema`` descriptor with ``strict=True``.

        Notes
        -----
        Retry logic (up to *max_retries* attempts with exponential back-off)
        applies only in **blocking** mode (``on_chunk=None``).  In streaming
        mode a single attempt is made; if the stream errors mid-way the
        exception propagates to the caller.
        """
        tools_payload = [t.to_openai_dict() for t in tools] if tools else None
        openai_messages = _messages_to_openai(messages)
        fmt = normalize_response_format(response_format)

        logger.debug(
            "llm.request",
            model=self._model,
            n_messages=len(openai_messages),
            n_tools=len(tools_payload) if tools_payload else 0,
            streaming=on_chunk is not None,
        )

        if on_chunk is not None:
            return await self._chat_stream(openai_messages, tools_payload, fmt, on_chunk)
        return await self._chat_blocking(openai_messages, tools_payload, fmt)

    # ------------------------------------------------------------------
    # Private: blocking mode (with retry)
    # ------------------------------------------------------------------

    async def _chat_blocking(
        self,
        openai_messages: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]] | None,
        response_format: dict[str, Any] | None,
    ) -> LLMResponse:
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
                if response_format is not None:
                    kwargs["response_format"] = response_format

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

    # ------------------------------------------------------------------
    # Private: streaming mode (single attempt, no mid-stream retry)
    # ------------------------------------------------------------------

    async def _chat_stream(
        self,
        openai_messages: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]] | None,
        response_format: dict[str, Any] | None,
        on_chunk: Callable[[str], None],
    ) -> LLMResponse:
        t0 = time.perf_counter()

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": openai_messages,
            "max_tokens": self._max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools_payload:
            kwargs["tools"] = tools_payload
            kwargs["tool_choice"] = "auto"
        if response_format is not None:
            kwargs["response_format"] = response_format

        # Accumulators
        content_parts: list[str] = []
        finish_reason: str = "stop"
        model_name: str = self._model
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0

        # Tool-call accumulation: delta index → {id, name, argument chunks}
        tc_accum: dict[int, dict[str, Any]] = {}

        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            # Usage arrives in the final synthetic chunk when stream_options is set
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0
                total_tokens = chunk.usage.total_tokens or 0

            if chunk.model:
                model_name = chunk.model

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason

            delta = choice.delta

            # Text content
            if delta.content:
                content_parts.append(delta.content)
                on_chunk(delta.content)

            # Tool-call deltas (indexed; accumulate id, name, argument fragments)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_accum:
                        tc_accum[idx] = {"id": "", "name": "", "arguments": []}
                    if tc_delta.id:
                        tc_accum[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc_accum[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc_accum[idx]["arguments"].append(tc_delta.function.arguments)

        latency_ms = int((time.perf_counter() - t0) * 1000)

        # Assemble tool calls (preserve original index order)
        tool_calls: list[ToolCallPart] = [
            ToolCallPart(
                id=tc["id"],
                name=tc["name"],
                arguments="".join(tc["arguments"]),
            )
            for tc in (tc_accum[i] for i in sorted(tc_accum))
        ]

        content = "".join(content_parts) if content_parts else None

        llm_response = LLMResponse(
            finish_reason=finish_reason,  # type: ignore[arg-type]
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=model_name,
        )

        logger.debug(
            "llm.response",
            finish_reason=finish_reason,
            n_tool_calls=len(tool_calls),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            streaming=True,
        )
        return llm_response
