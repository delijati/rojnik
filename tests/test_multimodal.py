"""Multimodal user content, transcription and provider-neutral errors in LLMClient."""

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, BadRequestError

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


def _status_error(status: int, detail: str = "nope") -> APIStatusError:
    # Built without the SDK's HTTP layer (its transport package differs between versions).
    cls = BadRequestError if status == 400 else APIStatusError
    error = cls.__new__(cls)
    error.status_code = status
    error.body = {"detail": detail}
    error.message = f"Error code: {status} - {error.body}"
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


async def test_content_blocks_keep_only_text_and_drop_thinking() -> None:
    blocks = [
        {"type": "thinking", "thinking": "hmm", "signature": "x"},
        {"type": "text", "text": '{"entries":[]}'},
    ]
    client = _client(chat=[_completion(blocks)])  # type: ignore[arg-type]

    response = await client.chat([UserMessage(content="x")])

    assert response.content == '{"entries":[]}'


async def test_ocr_posts_image_and_annotation_and_retries_transient_failure() -> None:
    client = _client()
    transport = _Completions([_status_error(503), {"pages": [], "document_annotation": "{}"}])

    async def post(path: str, **kwargs: Any) -> Any:
        assert path == "/ocr"
        return await transport.create(**kwargs)

    client._client.post = post  # noqa: SLF001
    result = await client.ocr(
        ImagePart(data=b"img", media_type="image/png"), schema={}, prompt="OCR"
    )
    assert len(transport.calls) == 2
    body = transport.calls[0]["body"]
    assert body["model"] == "vision-model"
    assert body["document"] == {"type": "image_url", "image_url": "data:image/png;base64,aW1n"}
    assert body["document_annotation_format"]["json_schema"]["schema"] == {}
    assert body["document_annotation_prompt"] == "OCR"
    assert result["document_annotation"] == "{}"


async def test_ocr_preserves_access_error_without_retry() -> None:
    client = _client()
    transport = _Completions([_status_error(403)])
    client._client.post = lambda path, **kwargs: transport.create(**kwargs)  # noqa: SLF001
    with pytest.raises(LLMError) as caught:
        await client.ocr(ImagePart(data=b"img"), schema={}, prompt="OCR")
    assert caught.value.status == 403
    assert len(transport.calls) == 1


@pytest.fixture(autouse=True)
def _forget_schema_fallbacks() -> Any:
    from rojnik.llm import client as client_module

    client_module._SCHEMA_IN_PROMPT.clear()
    yield
    client_module._SCHEMA_IN_PROMPT.clear()


async def test_schema_fallback_is_remembered_per_endpoint_and_model() -> None:
    schema = {"type": "json_schema", "json_schema": {"name": "a", "schema": {"type": "object"}}}
    rejected = _status_error(400, 'tool_choice: type "tool" and "any" are not supported')
    first = _client(chat=[rejected, _completion('{"a": 1}')])
    await first.chat([UserMessage(content="x")], response_format=schema)

    later = _client(chat=[_completion('{"a": 2}')])
    await later.chat([UserMessage(content="y")], response_format=schema)
    other = LLMClient(api_key="test", model="other-model", base_url="https://llm.example/v1")

    assert "response_format" not in later._client.chat.completions.calls[0]  # noqa: SLF001
    assert not other._schema_in_prompt  # noqa: SLF001


def test_the_sdk_never_retries_on_its_own() -> None:
    client = LLMClient(api_key="test", model="m", base_url="https://llm.example/v1")
    assert client._client_kwargs["max_retries"] == 0  # noqa: SLF001
    assert client._client.max_retries == 0  # noqa: SLF001


async def test_rejected_json_schema_falls_back_to_schema_in_prompt_once() -> None:
    rejected = _status_error(400, 'tool_choice: type "tool" and "any" are not supported')
    client = _client(
        chat=[rejected, _completion('```json\n{"a": 1}\n```'), _completion('{"a": 2}')]
    )
    schema = {"type": "json_schema", "json_schema": {"name": "a", "schema": {"type": "object"}}}

    first = await client.chat(
        [SystemMessage(content="OCR"), UserMessage(content="x")], response_format=schema
    )
    second = await client.chat([UserMessage(content="y")], response_format=schema)

    calls = client._client.chat.completions.calls  # noqa: SLF001
    assert [("response_format" in call) for call in calls] == [True, False, False]
    assert calls[1]["messages"][0]["content"].startswith("OCR\n\n")
    assert '{"type": "object"}' in calls[1]["messages"][0]["content"]
    assert calls[2]["messages"][0]["role"] == "system"
    assert (first.content, second.content) == ('{"a": 1}', '{"a": 2}')

    invalid = _client(chat=[_status_error(400, "Invalid schema for response_format 'a'")])
    with pytest.raises(LLMError):
        await invalid.chat([UserMessage(content="x")], response_format=schema)
    assert len(invalid._client.chat.completions.calls) == 1  # noqa: SLF001


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
    assert caught.value.retryable
    assert len(client._client.chat.completions.calls) == 2  # noqa: SLF001


async def test_client_errors_are_not_retryable() -> None:
    client = _client(chat=[_status_error(400)])

    with pytest.raises(LLMError) as caught:
        await client.chat([UserMessage(content="x")])

    assert not caught.value.retryable


async def test_connection_failures_after_retries_are_retryable() -> None:
    client = _client(chat=[_connection_error()] * 10)
    client._max_retries = 1  # noqa: SLF001

    with pytest.raises(LLMError) as caught:
        await client.chat([UserMessage(content="x")])

    assert caught.value.status is None
    assert caught.value.retryable


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(None, True), (408, True), (409, True), (429, True), (500, True), (503, True)]
    + [(400, False), (401, False), (403, False), (404, False), (413, False), (422, False)],
)
def test_llm_error_marks_transient_failures_retryable(status: int | None, retryable: bool) -> None:
    assert LLMError("x", status=status).retryable is retryable


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
    assert call["response_format"] == "json"


@pytest.mark.parametrize(
    ("response", "seconds"),
    [
        (SimpleNamespace(text="a", usage=SimpleNamespace(type="duration", seconds=12)), 12.0),
        ({"text": "a", "usage": {"prompt_audio_seconds": 7.5}}, 7.5),
        (SimpleNamespace(text="a", duration=3.25), 3.25),
        (SimpleNamespace(text="a"), None),
        ("plain text", None),
    ],
)
async def test_transcribe_detailed_reports_the_billed_audio_seconds(
    response: Any, seconds: float | None
) -> None:
    client = _client(audio=[response])

    result = await client.transcribe_detailed(b"audio", model="voxtral")

    assert result.audio_seconds == seconds and result.model == "voxtral"
    assert result.text == ("plain text" if isinstance(response, str) else "a")


async def test_transcribe_retries_connection_errors_and_maps_client_errors() -> None:
    flaky = _client(audio=[_connection_error(), SimpleNamespace(text="ok")])
    assert await flaky.transcribe(b"a") == "ok"

    rejected = _client(audio=[_status_error(413)])
    with pytest.raises(LLMError) as caught:
        await rejected.transcribe(b"a", model="whisper-1")
    assert (caught.value.status, caught.value.model) == (413, "whisper-1")
    assert "ASR 413" in caught.value.message


def _rate_limited(retry_after: str | None = None) -> Any:
    from openai import RateLimitError

    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://x"))
    return RateLimitError("rate limited", response=response, body=None)


async def test_rate_limits_are_waited_out_longer_than_other_failures() -> None:
    """Without the SDK's own retries, a 429 gets extra attempts (and its Retry-After)."""
    client = _client(chat=[_rate_limited()] * 4 + [_completion("ok")])
    client._max_retries = 1  # noqa: SLF001

    reply = await client.chat([UserMessage(content="x")])

    assert reply.content == "ok"
    assert len(client._client.chat.completions.calls) == 5  # noqa: SLF001

    exhausted = _client(chat=[_rate_limited()] * 20)
    exhausted._max_retries = 1  # noqa: SLF001
    with pytest.raises(LLMError) as caught:
        await exhausted.chat([UserMessage(content="x")])
    assert caught.value.status == 429
    assert len(exhausted._client.chat.completions.calls) == 1 + 1 + 4  # noqa: SLF001


def test_retry_delay_honours_retry_after_and_is_capped() -> None:
    client = LLMClient(api_key="t", model="m", base_url="https://llm.example/v1")
    client._retry_wait = 2  # noqa: SLF001
    assert client._retry_delay(1) == 2  # noqa: SLF001
    assert client._retry_delay(3) == 8  # noqa: SLF001
    assert client._retry_delay(1, _rate_limited("12")) == 12  # noqa: SLF001
    assert client._retry_delay(9) == 30, "höchstens 30 s"  # noqa: SLF001
    assert client._retry_delay(1, _rate_limited("soon")) == 2  # noqa: SLF001


async def test_transcribe_forwards_provider_fields_such_as_context_bias() -> None:
    client = _client(audio=[SimpleNamespace(text="Steingasse")])

    await client.transcribe(b"a", extra_body={"context_bias": ["Steingasse", "Convini_UK"]})

    call = client._client.audio.transcriptions.calls[0]  # noqa: SLF001
    assert call["extra_body"] == {"context_bias": ["Steingasse", "Convini_UK"]}


async def test_transcribe_drops_a_provider_field_the_endpoint_rejects() -> None:
    from rojnik.llm import client as client_module

    client_module._REJECTED_FIELDS.clear()
    rejected = _status_error(400, "Unrecognized request argument supplied: context_bias")
    client = _client(audio=[rejected, SimpleNamespace(text="ok"), SimpleNamespace(text="ok")])

    assert await client.transcribe(b"a", extra_body={"context_bias": ["Steingasse"]}) == "ok"
    assert await client.transcribe(b"a", extra_body={"context_bias": ["Steingasse"]}) == "ok"

    calls = client._client.audio.transcriptions.calls  # noqa: SLF001
    assert ["extra_body" in call for call in calls] == [True, False, False]
    client_module._REJECTED_FIELDS.clear()
