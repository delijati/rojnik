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
import inspect
import json
import re
import time
import weakref
from collections.abc import Callable
from typing import Any, TypeGuard

from loguru import logger
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError

from rojnik.config import settings
from rojnik.llm.schemas import (
    AssistantMessage,
    ImagePart,
    LLMResponse,
    Message,
    ResponseFormat,
    TextPart,
    ToolCallPart,
    ToolSchema,
    Transcription,
    UserMessage,
    normalize_response_format,
)

_TRANSIENT_STATUSES = frozenset({408, 409, 429})
_STRUCTURED_OUTPUT_PARAMS = ("tool_choice", "response_format", "json_schema")
_FENCE = re.compile(r"\s*```[\w-]*\s*(.*?)\s*```\s*", re.DOTALL)


class LLMError(RuntimeError):
    """Provider error surfaced by rojnik — callers never see SDK exception types.

    ``status`` is the HTTP status (``None`` for connection failures or
    exhausted retries), ``model`` the requested model.
    """

    def __init__(self, message: str, *, status: int | None = None, model: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.model = model

    @property
    def retryable(self) -> bool:
        """Transient failure worth retrying later: no response (connection, timeout),
        request timeout, conflict, rate limit or server error — never a rejected request."""
        return self.status is None or self.status in _TRANSIENT_STATUSES or self.status >= 500


def _status_error(exc: APIStatusError, model: str, what: str) -> LLMError:
    body = getattr(exc, "message", None) or str(exc)
    return LLMError(
        f"{what} {exc.status_code} (model={model}): {str(body)[:500]}",
        status=exc.status_code,
        model=model,
    )


def _is_json_schema(response_format: dict[str, Any] | None) -> TypeGuard[dict[str, Any]]:
    return response_format is not None and response_format.get("type") == "json_schema"


def _rejects_structured_output(exc: APIStatusError) -> bool:
    text = f"{exc.message} {exc.body}".lower()
    return (
        exc.status_code == 400
        and "support" in text
        and any(p in text for p in _STRUCTURED_OUTPUT_PARAMS)
    )


def _schema_in_prompt(
    openai_messages: list[dict[str, Any]], response_format: dict[str, Any]
) -> list[dict[str, Any]]:
    schema = json.dumps(response_format["json_schema"].get("schema", {}), ensure_ascii=False)
    rule = (
        "Respond with exactly one JSON object that matches this JSON schema, "
        f"without markdown or any other text:\n{schema}"
    )
    if openai_messages and openai_messages[0]["role"] == "system":
        first, *rest = openai_messages
        return [{**first, "content": f"{first['content']}\n\n{rule}"}, *rest]
    return [{"role": "system", "content": rule}, *openai_messages]


def _cached_tokens(usage: Any) -> int:
    """Prompt tokens the provider served from its prompt cache (0 if not reported).

    OpenAI-compatible servers report ``usage.prompt_tokens_details.cached_tokens``;
    DeepSeek uses ``prompt_cache_hit_tokens``; some servers put ``cached_tokens`` or
    ``num_cached_tokens`` directly on ``usage``.
    """
    if usage is None:
        return 0
    details = getattr(usage, "prompt_tokens_details", None)
    if isinstance(details, dict):
        value = details.get("cached_tokens")
    else:
        value = getattr(details, "cached_tokens", None)
    for name in ("prompt_cache_hit_tokens", "cached_tokens", "num_cached_tokens"):
        if value is None:
            value = getattr(usage, name, None)
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


# One SDK client (and with it one HTTP connection pool) per event loop and endpoint, so
# consecutive calls reuse TLS connections instead of opening a new one per LLMClient.
_Pool = dict[tuple[Any, ...], AsyncOpenAI]
_SHARED_CLIENTS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _Pool] = (
    weakref.WeakKeyDictionary()
)


def _shared_client(kwargs: dict[str, Any]) -> AsyncOpenAI:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return AsyncOpenAI(**kwargs)
    key = (kwargs.get("api_key"), kwargs.get("base_url"), kwargs.get("timeout"))
    pool = _SHARED_CLIENTS.setdefault(loop, {})
    client = pool.get(key)
    if client is None or client.is_closed():
        client = pool[key] = AsyncOpenAI(**kwargs)
    return client


async def close_shared_clients() -> None:
    """Close the shared connection pools of the running event loop (call on shutdown)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    for client in _SHARED_CLIENTS.pop(loop, {}).values():
        await client.close()


# A rate limit (429) is waited out longer than other transient failures: on top of
# `max_retries`, this many more attempts, each honouring the server's Retry-After and capped.
_RATE_LIMIT_EXTRA_ATTEMPTS = 4
_MAX_RETRY_WAIT_SECONDS = 30.0


def _retry_after(exc: Exception | None) -> float | None:
    """Seconds the server asked us to wait (``retry-after-ms`` or ``retry-after``)."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        try:
            value = headers.get(name)
            if value is not None:
                return max(0.0, float(value) * scale)
        except (TypeError, ValueError):
            continue
    return None


# Extra request fields an endpoint rejected, per (base_url, model): sent once, then dropped
# for every later request of the process (e.g. `prompt_cache_key` on a gateway without it).
_REJECTED_FIELDS: dict[tuple[str | None, str], set[str]] = {}
# Endpoints (base_url, model) that rejected native json_schema output: every later client of
# the process puts the schema into the prompt right away instead of paying one 400 each.
_SCHEMA_IN_PROMPT: set[tuple[str | None, str]] = set()


def _rejected_field(exc: APIStatusError, extra_body: dict[str, Any] | None) -> str | None:
    """The extra field a 400/422 response complains about, if any."""
    if not extra_body or exc.status_code not in (400, 422):
        return None
    text = f"{getattr(exc, 'message', '')} {getattr(exc, 'body', '')}".lower()
    return next((name for name in extra_body if name.lower() in text), None)


def _without(extra_body: dict[str, Any] | None, field: str) -> dict[str, Any] | None:
    remaining = {name: value for name, value in (extra_body or {}).items() if name != field}
    return remaining or None


def _text_content(content: Any) -> str | None:
    """Some OpenAI-compatible servers return content blocks; keep only the text blocks."""
    if content is None or isinstance(content, str):
        return content
    texts = [
        b.get("text") or "" for b in content if isinstance(b, dict) and b.get("type") == "text"
    ]
    return "".join(texts) or None


def _user_content(msg: UserMessage) -> str | list[dict[str, Any]]:
    """Plain text stays a string; multimodal content becomes OpenAI content parts."""
    if isinstance(msg.content, str):
        return msg.content
    parts: list[dict[str, Any]] = []
    for part in msg.content:
        if isinstance(part, TextPart):
            parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            parts.append(
                {"type": "image_url", "image_url": {"url": part.data_url(), "detail": part.detail}}
            )
    return parts


def _messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert our internal Message types to the plain dicts the openai SDK expects."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.role
        if role == "system":
            result.append({"role": role, "content": msg.content})  # type: ignore[union-attr]
        elif role == "user":
            assert isinstance(msg, UserMessage)
            result.append({"role": "user", "content": _user_content(msg)})
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


def _audio_seconds(response: Any) -> float | None:
    """Billed audio duration of a transcription response, if the provider reports one."""
    if isinstance(response, str):
        return None
    usage = getattr(response, "usage", None)
    if isinstance(response, dict):
        usage = response.get("usage")
    candidates: list[Any] = []
    for name in ("seconds", "prompt_audio_seconds"):
        candidates.append(
            usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        )
    candidates.append(
        response.get("duration")
        if isinstance(response, dict)
        else getattr(response, "duration", None)
    )
    for value in candidates:
        try:
            if value is not None and float(value) >= 0:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


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
    timeout:
        Request timeout in seconds (provider SDK default when ``None``).
    extra_body:
        Provider-specific request fields sent with every chat call (e.g.
        ``{"prompt_cache_key": "session-42"}``). A field the endpoint rejects with 400/422
        is dropped, remembered per endpoint and model, and the request retried once.
    share_connection:
        Reuse one HTTP connection pool per event loop, endpoint, key and timeout across
        all ``LLMClient`` instances (default). ``False`` gives this instance its own pool;
        close it with :meth:`aclose`. Shared pools close with :func:`close_shared_clients`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
        max_retries: int | None = None,
        retry_wait_seconds: float | None = None,
        timeout: float | None = None,
        share_connection: bool = True,
        extra_body: dict[str, Any] | None = None,
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

        # The SDK would retry 2× by itself before our own retry loop retries again; rojnik
        # owns retries (counted, logged, with back-off), so the SDK never retries.
        client_kwargs: dict[str, Any] = {"api_key": resolved_api_key, "max_retries": 0}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        if timeout is not None:
            client_kwargs["timeout"] = timeout

        self._client_kwargs = client_kwargs
        self._share_connection = share_connection
        self._own_client: AsyncOpenAI | None = None
        self._client_override: Any = None
        self._extra_body = dict(extra_body or {})

        logger.debug(
            "llm.client.created",
            model=self._model,
            provider=settings.provider,
            base_url=resolved_base_url or "https://api.openai.com/v1",
        )

    def _retry_delay(self, failure: int, exc: Exception | None = None) -> float:
        """Exponential back-off for the n-th failure, at least the server's Retry-After,
        never more than ``_MAX_RETRY_WAIT_SECONDS``."""
        wait = max(self._retry_wait * (2 ** (failure - 1)), _retry_after(exc) or 0.0)
        return min(wait, _MAX_RETRY_WAIT_SECONDS)

    @property
    def _schema_in_prompt(self) -> bool:
        """This endpoint and model need the schema in the prompt (remembered per process)."""
        return (self._client_kwargs.get("base_url"), self._model) in _SCHEMA_IN_PROMPT

    @_schema_in_prompt.setter
    def _schema_in_prompt(self, value: bool) -> None:
        key = (self._client_kwargs.get("base_url"), self._model)
        if value:
            _SCHEMA_IN_PROMPT.add(key)
        else:
            _SCHEMA_IN_PROMPT.discard(key)

    @property
    def _client(self) -> Any:
        """The SDK client: shared per event loop and endpoint unless disabled."""
        if self._client_override is not None:
            return self._client_override
        if self._share_connection:
            return _shared_client(self._client_kwargs)
        if self._own_client is None:
            self._own_client = AsyncOpenAI(**self._client_kwargs)
        return self._own_client

    @_client.setter
    def _client(self, value: Any) -> None:
        self._client_override = value

    async def aclose(self) -> None:
        """Close this instance's own connection pool (shared pools stay open)."""
        if self._own_client is not None:
            await self._own_client.close()
            self._own_client = None

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
        extra_body: dict[str, Any] | None = None,
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
        extra_body:
            Provider-specific request fields for this call, merged over the client's
            ``extra_body`` (see the constructor).

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

        extra = self._extra_fields(extra_body)
        if on_chunk is not None:
            return await self._chat_stream(openai_messages, tools_payload, fmt, on_chunk, extra)
        if self._schema_in_prompt and _is_json_schema(fmt):
            return await self._chat_schema_in_prompt(openai_messages, tools_payload, fmt, extra)
        return await self._chat_blocking(openai_messages, tools_payload, fmt, extra)

    def _endpoint(self) -> tuple[str | None, str]:
        return (self._client_kwargs.get("base_url"), self._model)

    def _extra_fields(self, extra_body: dict[str, Any] | None) -> dict[str, Any] | None:
        rejected = _REJECTED_FIELDS.get(self._endpoint(), set())
        merged = {**self._extra_body, **(extra_body or {})}
        kept = {name: value for name, value in merged.items() if name not in rejected}
        return kept or None

    def _note_rejected(self, exc: APIStatusError, extra_body: dict[str, Any] | None) -> str | None:
        """Name of the extra field the error rejects (remembered for this endpoint)."""
        field = _rejected_field(exc, extra_body)
        if field is not None:
            _REJECTED_FIELDS.setdefault(self._endpoint(), set()).add(field)
            logger.info("llm.extra_field_rejected — dropped", field=field, model=self._model)
        return field

    async def _chat_schema_in_prompt(
        self,
        openai_messages: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]] | None,
        response_format: dict[str, Any],
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        response = await self._chat_blocking(
            _schema_in_prompt(openai_messages, response_format), tools_payload, None, extra_body
        )
        fenced = _FENCE.fullmatch(response.content or "")
        return response.model_copy(update={"content": fenced.group(1)}) if fenced else response

    # ------------------------------------------------------------------
    # Private: blocking mode (with retry)
    # ------------------------------------------------------------------

    async def _chat_blocking(
        self,
        openai_messages: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]] | None,
        response_format: dict[str, Any] | None,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        last_error: Exception | None = None
        failures = rate_limits = 0
        for attempt in range(1, self._max_retries + _RATE_LIMIT_EXTRA_ATTEMPTS + 2):
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
                if extra_body:
                    kwargs["extra_body"] = extra_body

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
                    content=_text_content(choice.message.content),
                    tool_calls=tool_calls,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    cached_tokens=_cached_tokens(usage),
                    model=response.model,
                )

                logger.debug(
                    "llm.response",
                    finish_reason=finish_reason,
                    n_tool_calls=len(tool_calls),
                    prompt_tokens=llm_response.prompt_tokens,
                    cached_tokens=llm_response.cached_tokens,
                    completion_tokens=llm_response.completion_tokens,
                    latency_ms=latency_ms,
                    attempt=attempt,
                )
                return llm_response

            except RateLimitError as exc:
                last_error = exc
                rate_limits += 1
                if rate_limits > self._max_retries + _RATE_LIMIT_EXTRA_ATTEMPTS:
                    break
                wait = self._retry_delay(rate_limits, exc)
                logger.warning(
                    "llm.rate_limit — retrying",
                    attempt=attempt,
                    wait_s=wait,
                )
                await asyncio.sleep(wait)

            except APIConnectionError as exc:
                last_error = exc
                failures += 1
                if failures > self._max_retries:
                    break
                wait = self._retry_delay(failures)
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
                    failures += 1
                    if failures > self._max_retries:
                        break
                    wait = self._retry_delay(failures)
                    logger.warning(
                        "llm.server_error — retrying",
                        status=exc.status_code,
                        attempt=attempt,
                        wait_s=wait,
                    )
                    await asyncio.sleep(wait)
                elif (field := self._note_rejected(exc, extra_body)) is not None:
                    return await self._chat_blocking(
                        openai_messages,
                        tools_payload,
                        response_format,
                        _without(extra_body, field),
                    )
                elif _is_json_schema(response_format) and _rejects_structured_output(exc):
                    self._schema_in_prompt = True
                    logger.info("llm.json_schema_unsupported — schema in prompt", model=self._model)
                    return await self._chat_schema_in_prompt(
                        openai_messages, tools_payload, response_format, extra_body
                    )
                else:
                    logger.error("llm.client_error", status=exc.status_code, body=exc.body)
                    raise _status_error(exc, self._model, "LLM") from exc

        logger.error("llm.max_retries_exceeded", max_retries=self._max_retries)
        status = last_error.status_code if isinstance(last_error, APIStatusError) else None
        raise LLMError(
            f"LLM call failed after {self._max_retries} retries: {last_error}",
            status=status,
            model=self._model,
        ) from last_error

    # ------------------------------------------------------------------
    # Public: speech-to-text
    # ------------------------------------------------------------------

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "audio.webm",
        content_type: str = "audio/webm",
        language: str | None = None,
        model: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """Transcribe audio via an OpenAI-compatible ``/audio/transcriptions`` endpoint.

        Same retry policy as :meth:`chat` (rate limits, connection errors, 5xx).
        Returns the plain transcript text; :meth:`transcribe_detailed` also returns the
        billed audio duration.
        """
        result = await self.transcribe_detailed(
            audio,
            filename=filename,
            content_type=content_type,
            language=language,
            model=model,
            extra_body=extra_body,
        )
        return result.text

    async def transcribe_detailed(
        self,
        audio: bytes,
        *,
        filename: str = "audio.webm",
        content_type: str = "audio/webm",
        language: str | None = None,
        model: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> Transcription:
        """Like :meth:`transcribe`, with the audio seconds the provider reports (JSON
        response: ``usage.seconds``, ``usage.prompt_audio_seconds`` or ``duration``).
        ``extra_body``: provider fields such as Mistral ``context_bias`` (key terms)."""
        model_name = model or self._model
        safe_name = filename.replace("\\", "/").rsplit("/", 1)[-1] or "audio.webm"
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 2):
            try:
                kwargs: dict[str, Any] = {
                    "model": model_name,
                    "file": (safe_name, audio, content_type),
                    "response_format": "json",
                }
                if language:
                    kwargs["language"] = language
                if extra_body:
                    kwargs["extra_body"] = extra_body
                response = await self._client.audio.transcriptions.create(**kwargs)
                if isinstance(response, str):
                    text = response
                elif isinstance(response, dict):
                    text = response.get("text") or ""
                else:
                    text = getattr(response, "text", "")
                logger.debug("llm.transcribe", model=model_name, attempt=attempt)
                return Transcription(
                    text=str(text or "").strip(),
                    audio_seconds=_audio_seconds(response),
                    model=model_name,
                )
            except (RateLimitError, APIConnectionError) as exc:
                last_error = exc
            except APIStatusError as exc:
                if exc.status_code < 500:
                    raise _status_error(exc, model_name, "ASR") from exc
                last_error = exc
            if attempt > self._max_retries:
                break
            wait = self._retry_delay(attempt, last_error)
            logger.warning("llm.transcribe — retrying", attempt=attempt, wait_s=wait)
            await asyncio.sleep(wait)
        status = last_error.status_code if isinstance(last_error, APIStatusError) else None
        raise LLMError(
            f"ASR failed after {self._max_retries} retries: {last_error}",
            status=status,
            model=model_name,
        ) from last_error

    async def ocr(
        self,
        image: ImagePart,
        *,
        schema: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        """Mistral OCR with document annotations; one document read, no chat fallback."""
        body = {
            "model": self._model,
            "document": {"type": "image_url", "image_url": image.data_url()},
            "include_image_base64": False,
            "document_annotation_format": {
                "type": "json_schema",
                "json_schema": {"name": "observed_document", "strict": True, "schema": schema},
            },
            "document_annotation_prompt": prompt,
        }
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post("/ocr", body=body, cast_to=dict)
                if not isinstance(response, dict):
                    raise LLMError("OCR response must be an object", model=self._model)
                return response
            except (RateLimitError, APIConnectionError) as exc:
                last_error = exc
            except APIStatusError as exc:
                if exc.status_code < 500:
                    raise _status_error(exc, self._model, "OCR") from exc
                last_error = exc
            if attempt < self._max_retries:
                await asyncio.sleep(self._retry_delay(attempt + 1, last_error))
        status = last_error.status_code if isinstance(last_error, APIStatusError) else None
        raise LLMError(
            f"OCR failed after {self._max_retries} retries: {last_error}",
            status=status,
            model=self._model,
        ) from last_error

    # ------------------------------------------------------------------
    # Private: streaming mode (single attempt, no mid-stream retry)
    # ------------------------------------------------------------------

    async def _chat_stream(
        self,
        openai_messages: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]] | None,
        response_format: dict[str, Any] | None,
        on_chunk: Callable[[str], None],
        extra_body: dict[str, Any] | None = None,
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
        if extra_body:
            kwargs["extra_body"] = extra_body

        # Accumulators
        content_parts: list[str] = []
        finish_reason: str = "stop"
        model_name: str = self._model
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0
        cached_tokens: int = 0
        first_token_ms: int | None = None

        # Tool-call accumulation: delta index → {id, name, argument chunks}
        tc_accum: dict[int, dict[str, Any]] = {}

        try:
            stream = await self._client.chat.completions.create(**kwargs)
        except APIStatusError as exc:
            if (field := self._note_rejected(exc, extra_body)) is not None:
                return await self._chat_stream(
                    openai_messages,
                    tools_payload,
                    response_format,
                    on_chunk,
                    _without(extra_body, field),
                )
            raise _status_error(exc, self._model, "LLM") from exc
        except APIConnectionError as exc:
            raise LLMError(f"LLM connection failed: {exc}", model=self._model) from exc
        # Close the response even when the caller cancels mid-stream, so the
        # connection returns to the (shared) pool instead of lingering until GC.
        try:
            async for chunk in stream:
                # Usage arrives in the final synthetic chunk when stream_options is set
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens or 0
                    completion_tokens = chunk.usage.completion_tokens or 0
                    total_tokens = chunk.usage.total_tokens or 0
                    cached_tokens = _cached_tokens(chunk.usage)

                if chunk.model:
                    model_name = chunk.model

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                delta = choice.delta

                # Text content
                text = _text_content(delta.content)
                if text:
                    if first_token_ms is None:
                        first_token_ms = int((time.perf_counter() - t0) * 1000)
                    content_parts.append(text)
                    on_chunk(text)

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
        finally:
            closed = getattr(stream, "close", lambda: None)()
            if inspect.isawaitable(closed):
                await closed

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
            cached_tokens=cached_tokens,
            model=model_name,
        )

        logger.debug(
            "llm.response",
            finish_reason=finish_reason,
            n_tool_calls=len(tool_calls),
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            first_token_ms=first_token_ms,
            streaming=True,
        )
        return llm_response
