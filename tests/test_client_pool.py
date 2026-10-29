"""Prompt-cache token reporting and connection reuse in LLMClient."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from rojnik.llm import LLMClient, UserMessage, close_shared_clients
from rojnik.llm.client import _cached_tokens


def _usage(**extra: Any) -> SimpleNamespace:
    return SimpleNamespace(prompt_tokens=100, completion_tokens=5, total_tokens=105, **extra)


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (_usage(prompt_tokens_details=SimpleNamespace(cached_tokens=64)), 64),
        (_usage(prompt_tokens_details={"cached_tokens": 32}), 32),
        (_usage(prompt_cache_hit_tokens=16), 16),
        (_usage(num_cached_tokens=8), 8),
        (_usage(prompt_tokens_details=SimpleNamespace(cached_tokens=None)), 0),
        (_usage(cached_tokens="kaputt"), 0),
        (_usage(), 0),
        (None, 0),
    ],
)
def test_cached_tokens_are_read_from_every_known_usage_shape(usage: Any, expected: int) -> None:
    assert _cached_tokens(usage) == expected


class _Completions:
    def __init__(self, result: Any) -> None:
        self.result = result

    async def create(self, **kwargs: Any) -> Any:
        return self.result


def _with(result: Any) -> LLMClient:
    client = LLMClient(api_key="test", model="m", base_url="https://llm.example/v1")
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions(result)))
    return client


async def test_blocking_response_reports_cached_prompt_tokens() -> None:
    message = SimpleNamespace(content="ok", tool_calls=None)
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=_usage(prompt_tokens_details=SimpleNamespace(cached_tokens=96)),
        model="m",
    )

    response = await _with(completion).chat([UserMessage(content="hi")])

    assert (response.prompt_tokens, response.cached_tokens) == (100, 96)


class _Stream:
    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks
        self.closed = False

    def __aiter__(self) -> "_Stream":
        return self

    async def __anext__(self) -> Any:
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)

    async def close(self) -> None:
        self.closed = True


def _chunk(text: str | None = None, usage: Any = None) -> SimpleNamespace:
    delta = SimpleNamespace(content=text, tool_calls=None)
    choices = [SimpleNamespace(delta=delta, finish_reason=None)] if text is not None else []
    return SimpleNamespace(choices=choices, usage=usage, model="m")


async def test_stream_reports_cached_tokens_and_closes_the_response() -> None:
    stream = _Stream(
        [_chunk("Hal"), _chunk("lo"), _chunk(usage=_usage(prompt_cache_hit_tokens=80))]
    )
    seen: list[str] = []

    response = await _with(stream).chat([UserMessage(content="hi")], on_chunk=seen.append)

    assert (response.content, response.cached_tokens) == ("Hallo", 80)
    assert seen == ["Hal", "lo"] and stream.closed


async def test_stream_is_closed_when_the_caller_aborts_mid_stream() -> None:
    stream = _Stream([_chunk("a"), _chunk("b")])

    def abort(_text: str) -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await _with(stream).chat([UserMessage(content="hi")], on_chunk=abort)

    assert stream.closed, "connection goes back to the pool"


async def test_clients_of_one_endpoint_share_one_connection_pool_per_loop() -> None:
    first = LLMClient(api_key="k", model="a", base_url="https://llm.example/v1", timeout=30)
    second = LLMClient(api_key="k", model="b", base_url="https://llm.example/v1", timeout=30)
    other_timeout = LLMClient(api_key="k", model="a", base_url="https://llm.example/v1", timeout=5)
    other_key = LLMClient(api_key="x", model="a", base_url="https://llm.example/v1", timeout=30)

    assert first._client is second._client
    assert first._client is not other_timeout._client
    assert first._client is not other_key._client

    shared = first._client
    await close_shared_clients()
    assert shared.is_closed()
    assert first._client is not shared and not first._client.is_closed()
    await close_shared_clients()


def test_each_event_loop_gets_its_own_pool() -> None:
    client = LLMClient(api_key="k", model="a", base_url="https://llm.example/v1")

    async def current() -> Any:
        sdk = client._client
        await close_shared_clients()
        return sdk

    assert asyncio.run(current()) is not asyncio.run(current())


async def test_unshared_client_owns_and_closes_its_pool() -> None:
    client = LLMClient(
        api_key="k", model="a", base_url="https://llm.example/v1", share_connection=False
    )
    shared = LLMClient(api_key="k", model="a", base_url="https://llm.example/v1")

    own = client._client
    assert own is client._client and own is not shared._client
    await client.aclose()
    assert own.is_closed()
    assert not shared._client.is_closed()
    await close_shared_clients()
