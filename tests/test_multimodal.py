"""Multimodal user content, transcription and provider-neutral errors in LLMClient."""

from types import SimpleNamespace
from typing import Any

import pytest
from openai import APIConnectionError, APIStatusError

from rojnik.llm import ImagePart, LLMClient, LLMError, SystemMessage, TextPart, UserMessage
from rojnik.llm.client import _messages_to_openai
from rojnik.memory.context import _message_token_count


def _completion(content: str = "{}") -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        model="vision-model",
    )


def _status_error(status: int) -> APIStatusError:
    # Built without the SDK's HTTP layer (its transport package differs between versions).
    error = APIStatusError.__new__(APIStatusError)
    error.status_code = status
    error.message = "nope"
    error.body = {"error": "nope"}
    return error


def _connection_error() -> APIConnectionError:
    return APIConnectionError.__new__(APIConnectionError)


class _Completions:
    def __init__(self, results: list[Any]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _client(*, chat: list[Any] | None = None, audio: list[Any] | None = None) -> LLMClient:
    client = LLMClient(api_key="test", model="vision-model", base_url="https://llm.example/v1")
    client._retry_wait = 0  # noqa: SLF001
    client._client = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        chat=SimpleNamespace(completions=_Completions(chat or [])),
        audio=SimpleNamespace(transcriptions=_Completions(audio or [])),
    )
    return client


def test_plain_text_user_message_stays_a_string() -> None:
    assert _messages_to_openai([UserMessage(content="hallo")]) == [
        {"role": "user", "content": "hallo"}
    ]


def test_image_part_becomes_openai_image_url_data_url() -> None:
    message = UserMessage(
        content=[
            ImagePart(data=b"\x89PNG", media_type="image/png; q=1", detail="high"),
            TextPart(text="Lies das Dokument."),
        ]
    )

    [converted] = _messages_to_openai([message])

    assert converted["role"] == "user"
    image, text = converted["content"]
    assert image == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,iVBORw==", "detail": "high"},
    }
    assert text == {"type": "text", "text": "Lies das Dokument."}


def test_non_image_media_type_falls_back_to_jpeg() -> None:
    assert (
        ImagePart(data=b"x", media_type="application/pdf")
        .data_url()
        .startswith("data:image/jpeg;base64,")
    )


def test_text_view_ignores_images_for_memory_and_token_counting() -> None:
    message = UserMessage(content=[ImagePart(data=b"x" * 10_000), TextPart(text="kurz")])
    assert message.text == "kurz"
    encoder = SimpleNamespace(encode=lambda text: list(text))  # no tokenizer download

    assert _message_token_count(message, encoder) == 4 + len("kurz")  # type: ignore[arg-type]


async def test_chat_with_image_and_structured_output() -> None:
    client = _client(chat=[_completion('{"observed_text": "pH 3,4"}')])
    schema = {"type": "json_schema", "json_schema": {"name": "doc", "strict": True, "schema": {}}}

    response = await client.chat(
        [SystemMessage(content="OCR"), UserMessage(content=[ImagePart(data=b"img")])],
        response_format=schema,
    )

    call = client._client.chat.completions.calls[0]  # noqa: SLF001
    assert call["response_format"] is schema
    assert call["messages"][1]["content"][0]["type"] == "image_url"
    assert response.content == '{"observed_text": "pH 3,4"}'
    assert response.total_tokens == 5


async def test_client_error_is_raised_as_llm_error_without_retry() -> None:
    client = _client(chat=[_status_error(400)])

    with pytest.raises(LLMError) as caught:
        await client.chat([UserMessage(content="x")])

    assert caught.value.status == 400
    assert caught.value.model == "vision-model"
    assert len(client._client.chat.completions.calls) == 1  # noqa: SLF001


async def test_server_errors_retry_then_raise_llm_error() -> None:
    client = _client(chat=[_status_error(503)] * 10)
    client._max_retries = 1  # noqa: SLF001

    with pytest.raises(LLMError) as caught:
        await client.chat([UserMessage(content="x")])

    assert caught.value.status == 503
    assert len(client._client.chat.completions.calls) == 2  # noqa: SLF001


async def test_transcribe_sends_file_and_language_and_returns_text() -> None:
    client = _client(audio=["  Hallo Keller  "])

    text = await client.transcribe(
        b"audio",
        filename="C:\\aufnahmen\\diktat.webm",
        content_type="audio/webm",
        language="de",
        model="whisper-1",
    )

    call = client._client.audio.transcriptions.calls[0]  # noqa: SLF001
    assert text == "Hallo Keller"
    assert call["model"] == "whisper-1"
    assert call["file"] == ("diktat.webm", b"audio", "audio/webm")
    assert call["language"] == "de"
    assert call["response_format"] == "text"


async def test_transcribe_retries_connection_errors_and_maps_client_errors() -> None:
    flaky = _client(audio=[_connection_error(), SimpleNamespace(text="ok")])
    assert await flaky.transcribe(b"a") == "ok"

    rejected = _client(audio=[_status_error(413)])
    with pytest.raises(LLMError) as caught:
        await rejected.transcribe(b"a", model="whisper-1")
    assert (caught.value.status, caught.value.model) == (413, "whisper-1")
    assert "ASR 413" in caught.value.message
