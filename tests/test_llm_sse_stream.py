"""Unit tests for the shared chat-completions stream reader and both stream parsers.

Covers plan 001.core-fixes, step 003 (stream parsing): the SSE reader rules of DEC-001-10
(incremental UTF-8, LF / CRLF / CR terminators, multi-line events, ignored fields, final
event), transport failures while reading (DEC-001-7, DEC-001-1), error frames and malformed
frames (DEC-001-9), the early-end rule (DEC-001-1), reasoning plus content in one piece
(DEC-001-11), the kept reasoning rules (US-026) and receiver failures (DEC-000-23).

Two seams are used: a fake streamed response whose ``content.iter_chunked`` yields canned
bytes (drives the real reader), and a stubbed ``_iter_sse_data`` that yields canned data
payloads (drives a parser alone; the parser then gets ``response=None``).

DoD-14 (the existing ``tests/test_llm_tools_reasoning_stream.py`` passes unchanged) is covered
by that file itself, which this step does not edit.
"""

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pytest
from yarl import URL

from llm.exceptions import LLMError
from llm.openai import OpenAIAPIClient

STREAM_URL = "http://stream.test.invalid/v1/chat/completions"


# --------------------------------------------------------------------------- helpers


class _FakeContent:
    """Stand-in for ``aiohttp.StreamReader``: yields the given byte reads in order.

    The read boundaries are exactly the given chunks, whatever chunk size the reader asks
    for, so a test controls where a character or a line terminator is split. When ``error``
    is set it is raised after the last chunk, simulating a transfer cut mid-body.
    """

    def __init__(self, chunks: list[bytes], error: BaseException | None = None) -> None:
        self._chunks = chunks
        self._error = error

    async def _gen(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error

    def iter_chunked(self, _n: int) -> AsyncIterator[bytes]:
        return self._gen()

    def iter_any(self) -> AsyncIterator[bytes]:
        return self._gen()


class _FakeResponse:
    """Minimal open 2xx streamed response: a URL (the reader names it in errors) and a body."""

    def __init__(self, chunks: list[bytes], error: BaseException | None = None) -> None:
        self.url = URL(STREAM_URL)
        self.real_url = self.url
        self.status = 200
        self.headers = {"Content-Type": "text/event-stream"}
        self.content = _FakeContent(chunks, error)


def _client() -> OpenAIAPIClient:
    """A chat-completions client with an explicit test credential (never from the environment)."""
    return OpenAIAPIClient(model="gpt-4o-mini", bearer_token="sk-test")


async def _read_events(chunks: list[bytes], error: BaseException | None = None) -> list[str]:
    """Run the real reader over the given byte reads and collect every yielded payload."""
    client = _client()
    return [data async for data in client._iter_sse_data(_FakeResponse(chunks, error))]


def _piece(
    content: str | None = None,
    reasoning: str | None = None,
    finish_reason: str | None = None,
    **extra: Any,
) -> str:
    """One chat-completions stream piece as a JSON data payload.

    Non-ASCII text is kept as raw characters (``ensure_ascii=False``), as real servers send it,
    so the encoded body holds the UTF-8 bytes and a test can cut a read inside a character.
    """
    delta: dict[str, Any] = dict(extra)
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    piece = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
    return json.dumps(piece, ensure_ascii=False)


def _stubbed_client(monkeypatch, payloads: list[str]) -> OpenAIAPIClient:
    """A client whose reader is replaced by one yielding the given data payloads."""
    client = _client()

    async def fake_iter_sse_data(_response: Any) -> AsyncIterator[str]:
        for payload in payloads:
            yield payload

    monkeypatch.setattr(client, "_iter_sse_data", fake_iter_sse_data)
    return client


class _Recorder:
    """Async receiver that records every piece it is given, in order."""

    def __init__(self) -> None:
        self.pieces: list[str] = []

    async def __call__(self, text: str) -> None:
        self.pieces.append(text)


async def _run_plain(monkeypatch, payloads: list[str]) -> tuple[str, list[str]]:
    client = _stubbed_client(monkeypatch, payloads)
    recorder = _Recorder()
    text = await client._stream_openai_chat_response(None, recorder)
    return text, recorder.pieces


async def _run_tools(monkeypatch, payloads: list[str]) -> tuple[str, list[dict], list[str]]:
    client = _stubbed_client(monkeypatch, payloads)
    recorder = _Recorder()
    content, tool_calls = await client._stream_openai_tools_response(None, recorder)
    return content, tool_calls, recorder.pieces


def _sse(*payloads: str, terminator: str = "\n") -> bytes:
    """Encode payloads as data-only SSE events with the given line terminator."""
    return "".join(f"data: {p}{terminator}{terminator}" for p in payloads).encode("utf-8")


# --------------------------------------------------------------------------- DoD-1


@pytest.mark.asyncio
async def test_openai_reader_keeps_multibyte_char_split_across_reads__DoD1():
    """A multi-byte character split between two reads is yielded whole by the reader (DoD-1)."""
    body = _sse(_piece(content="naïve 日本 🎉"), "[DONE]")
    for char in ("ï", "日", "🎉"):
        encoded = char.encode("utf-8")
        cut = body.index(encoded) + 1  # inside the character's byte sequence
        events = await _read_events([body[:cut], body[cut:]])
        assert events[-1] == "[DONE]"
        assert json.loads(events[0])["choices"][0]["delta"]["content"] == "naïve 日本 🎉"


@pytest.mark.asyncio
async def test_openai_plain_stream_delivers_split_multibyte_chars_whole__DoD1():
    """Split characters reach the receiver whole and are in the returned text (DoD-1)."""
    text_a = "Привет, "
    text_b = "日本語 🎉!"
    body = _sse(_piece(content=text_a), _piece(content=text_b, finish_reason="stop"), "[DONE]")
    # Cut inside the 2-byte "П", inside the 3-byte "本" and inside the 4-byte emoji.
    cuts = sorted(body.index(ch.encode("utf-8")) + 1 for ch in ("П", "本", "🎉"))
    chunks = []
    start = 0
    for cut in cuts:
        chunks.append(body[start:cut])
        start = cut
    chunks.append(body[start:])

    client = _client()
    recorder = _Recorder()
    text = await client._stream_openai_chat_response(_FakeResponse(chunks), recorder)

    assert recorder.pieces == [text_a, text_b]
    assert text == text_a + text_b
    assert "�" not in text


# --------------------------------------------------------------------------- DoD-2


@pytest.mark.asyncio
@pytest.mark.parametrize("terminator", ["\n", "\r\n", "\r"], ids=["LF", "CRLF", "CR"])
async def test_openai_reader_same_events_for_lf_crlf_cr__DoD2(terminator):
    """LF, CRLF and bare-CR terminated streams yield the same events (DoD-2)."""
    payloads = [_piece(content="one"), _piece(content="two", finish_reason="stop"), "[DONE]"]
    events = await _read_events([_sse(*payloads, terminator=terminator)])
    assert events == payloads


@pytest.mark.asyncio
async def test_openai_reader_crlf_split_across_reads_is_one_terminator__DoD2():
    """A CR ending one read and an LF starting the next is one terminator, not an early event end (DoD-2)."""
    # One event with two data lines; the CRLF after the first line is split between reads.
    # Treated as CR + LF it would form a blank line and dispatch "first" on its own.
    chunks = [b"data: first\r", b"\ndata: second\r\n\r\n", b"data: [DONE]\r\n\r\n"]
    events = await _read_events(chunks)
    assert events == ["first\nsecond", "[DONE]"]


@pytest.mark.asyncio
async def test_openai_reader_crlf_split_between_events__DoD2():
    """A CRLF split across reads inside the blank dispatch line still yields each event once (DoD-2)."""
    chunks = [b"data: a\r\n\r", b"\ndata: b\r\n\r\n"]
    events = await _read_events(chunks)
    assert events == ["a", "b"]


# --------------------------------------------------------------------------- DoD-3


@pytest.mark.asyncio
async def test_openai_reader_joins_multiline_data_with_newline__DoD3():
    """An event whose data spans several data lines is yielded once, lines joined by a newline (DoD-3)."""
    chunks = [b"data: line one\ndata: line two\ndata: line three\n\n", b"data: [DONE]\n\n"]
    events = await _read_events(chunks)
    assert events == ["line one\nline two\nline three", "[DONE]"]


@pytest.mark.asyncio
async def test_openai_reader_multiline_json_event_parses__DoD3():
    """A JSON payload spread over several data lines arrives as one parseable payload (DoD-3)."""
    obj = {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]}
    pretty = json.dumps(obj, indent=2)
    raw = "".join(f"data: {line}\n" for line in pretty.split("\n")) + "\n"
    events = await _read_events([raw.encode("utf-8")])
    assert len(events) == 1
    assert json.loads(events[0]) == obj


# --------------------------------------------------------------------------- DoD-4


@pytest.mark.asyncio
async def test_openai_reader_ignores_comments_and_non_data_fields__DoD4():
    """Comment, event, id and retry lines never appear in yielded payloads (DoD-4)."""
    raw = (
        ": keep-alive comment\n"
        "\n"
        "event: message\n"
        "id: 17\n"
        "retry: 3000\n"
        "data: payload-1\n"
        "\n"
        "event: ping\n"
        "id: 18\n"
        "\n"
        ":another comment\n"
        "data: payload-2\n"
        "\n"
    )
    events = await _read_events([raw.encode("utf-8")])
    assert events == ["payload-1", "payload-2"]


@pytest.mark.asyncio
async def test_openai_reader_yields_final_event_without_blank_line__DoD4():
    """A final event not followed by a blank line is still yielded (DoD-4)."""
    events = await _read_events([b"data: first\n\n", b"data: last"])
    assert events == ["first", "last"]


@pytest.mark.asyncio
async def test_openai_reader_yields_final_event_with_single_terminator__DoD4():
    """A final event ended by one line terminator but no blank line is still yielded (DoD-4)."""
    events = await _read_events([b"data: first\n\ndata: [DONE]\n"])
    assert events == ["first", "[DONE]"]


# --------------------------------------------------------------------------- DoD-5


def _transport_errors() -> list[Any]:
    return [
        pytest.param(aiohttp.ClientPayloadError("Response payload is not completed"), id="payload-error"),
        pytest.param(aiohttp.ServerDisconnectedError(), id="disconnect"),
        pytest.param(TimeoutError(), id="timeout"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", _transport_errors())
async def test_openai_reader_transport_failure_raises_llm_error__DoD5(error):
    """A payload error, disconnect or timeout while reading raises LLMError naming the address (DoD-5)."""
    chunks = [b"data: " + _piece(content="partial").encode("utf-8") + b"\n\n"]
    with pytest.raises(LLMError) as info:
        await _read_events(chunks, error=error)
    err = info.value
    # DEC-001-7: chained from the original, names the response URL, never "status <n>".
    assert err.__cause__ is error
    assert STREAM_URL in str(err)
    assert re.search(r"status \d", str(err)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("error", _transport_errors())
async def test_openai_plain_stream_transport_failure_raises_llm_error__DoD5(error):
    """A transfer cut while the plain parser reads surfaces as LLMError (DoD-5)."""
    chunks = [_sse(_piece(content="partial"))]
    client = _client()
    with pytest.raises(LLMError):
        await client._stream_openai_chat_response(_FakeResponse(chunks, error), _Recorder())


@pytest.mark.asyncio
async def test_openai_reader_transport_failure_before_any_data__DoD5():
    """A transport failure on the very first read also raises LLMError (DoD-5)."""
    with pytest.raises(LLMError):
        await _read_events([], error=aiohttp.ServerDisconnectedError())


# --------------------------------------------------------------------------- DoD-6


@pytest.mark.asyncio
async def test_openai_plain_error_frame_raises_with_reason__DoD6(monkeypatch):
    """An error frame raises LLMError carrying the server's reason; no later piece reaches the receiver (DoD-6)."""
    client = _stubbed_client(
        monkeypatch,
        [
            _piece(content="Hello"),
            json.dumps({"error": {"message": "model overloaded, retry later", "type": "server_error", "code": 503}}),
            _piece(content=" after error", finish_reason="stop"),
            "[DONE]",
        ],
    )
    recorder = _Recorder()
    with pytest.raises(LLMError) as info:
        await client._stream_openai_chat_response(None, recorder)
    assert "model overloaded, retry later" in str(info.value)
    assert recorder.pieces == ["Hello"]


@pytest.mark.asyncio
async def test_openai_plain_error_frame_non_object_error_value__DoD6(monkeypatch):
    """An error value that is not an object with a message is still reported, as its JSON text (DoD-6)."""
    client = _stubbed_client(monkeypatch, [json.dumps({"error": "backend exploded"}), "[DONE]"])
    recorder = _Recorder()
    with pytest.raises(LLMError) as info:
        await client._stream_openai_chat_response(None, recorder)
    assert "backend exploded" in str(info.value)
    assert recorder.pieces == []


@pytest.mark.asyncio
async def test_openai_plain_error_frame_with_choices_still_raises__DoD6(monkeypatch):
    """An error frame that also carries choices raises and delivers nothing from that frame (DoD-6)."""
    frame = {
        "error": {"message": "context length exceeded"},
        "choices": [{"index": 0, "delta": {"content": "leaked"}, "finish_reason": None}],
    }
    client = _stubbed_client(monkeypatch, [json.dumps(frame), "[DONE]"])
    recorder = _Recorder()
    with pytest.raises(LLMError) as info:
        await client._stream_openai_chat_response(None, recorder)
    assert "context length exceeded" in str(info.value)
    assert "leaked" not in "".join(recorder.pieces)


# --------------------------------------------------------------------------- DoD-7


@pytest.mark.asyncio
async def test_openai_tools_error_frame_raises_with_reason__DoD7(monkeypatch):
    """Tools parser: an error frame raises LLMError carrying the server's reason (DoD-7)."""
    client = _stubbed_client(
        monkeypatch,
        [
            _piece(content="Partial"),
            json.dumps({"error": {"message": "upstream tool server failed", "type": "server_error"}}),
            _piece(content=" more", finish_reason="stop"),
            "[DONE]",
        ],
    )
    with pytest.raises(LLMError) as info:
        await client._stream_openai_tools_response(None, _Recorder())
    assert "upstream tool server failed" in str(info.value)


@pytest.mark.asyncio
async def test_openai_tools_error_frame_non_object_error_value__DoD7(monkeypatch):
    """Tools parser: a non-object error value is reported as its JSON text (DoD-7)."""
    client = _stubbed_client(monkeypatch, [json.dumps({"error": "quota gone"}), "[DONE]"])
    with pytest.raises(LLMError) as info:
        await client._stream_openai_tools_response(None, _Recorder())
    assert "quota gone" in str(info.value)


# --------------------------------------------------------------------------- DoD-8


@pytest.mark.asyncio
async def test_openai_plain_stream_without_completion_signal_raises__DoD8(monkeypatch):
    """Plain parser: no finish reason and no [DONE] raises LLMError (DoD-8)."""
    client = _stubbed_client(monkeypatch, [_piece(content="cut "), _piece(content="off")])
    with pytest.raises(LLMError):
        await client._stream_openai_chat_response(None, _Recorder())


@pytest.mark.asyncio
async def test_openai_plain_empty_stream_raises__DoD8(monkeypatch):
    """Plain parser: a stream with no data at all has no completion signal and raises (DoD-8)."""
    client = _stubbed_client(monkeypatch, [])
    with pytest.raises(LLMError):
        await client._stream_openai_chat_response(None, _Recorder())


@pytest.mark.asyncio
async def test_openai_plain_stream_only_skipped_frames_raises__DoD8(monkeypatch):
    """Plain parser: a stream of only skipped frames still fails the end check (DoD-8, DEC-001-9)."""
    client = _stubbed_client(monkeypatch, ["not json", json.dumps({"usage": {"total_tokens": 3}})])
    with pytest.raises(LLMError):
        await client._stream_openai_chat_response(None, _Recorder())


@pytest.mark.asyncio
async def test_openai_plain_stream_done_only_returns__DoD8(monkeypatch):
    """Plain parser: [DONE] without any finish reason returns normally (DoD-8)."""
    text, pieces = await _run_plain(monkeypatch, [_piece(content="Hello "), _piece(content="world"), "[DONE]"])
    assert text == "Hello world"
    assert pieces == ["Hello ", "world"]


@pytest.mark.asyncio
async def test_openai_plain_stream_finish_reason_only_returns__DoD8(monkeypatch):
    """Plain parser: a finish reason on the last choice without [DONE] returns normally (DoD-8)."""
    text, pieces = await _run_plain(
        monkeypatch, [_piece(content="Hello "), _piece(content="world", finish_reason="stop")]
    )
    assert text == "Hello world"
    assert pieces == ["Hello ", "world"]


@pytest.mark.asyncio
async def test_openai_plain_stream_finish_reason_on_empty_delta_returns__DoD8(monkeypatch):
    """Plain parser: a final piece with an empty delta and a finish reason counts as complete (DoD-8)."""
    text, _pieces = await _run_plain(monkeypatch, [_piece(content="Done."), _piece(finish_reason="length")])
    assert text == "Done."


# --------------------------------------------------------------------------- DoD-9


@pytest.mark.asyncio
async def test_openai_tools_stream_without_completion_signal_raises__DoD9(monkeypatch):
    """Tools parser: no finish reason and no [DONE] raises LLMError (DoD-9)."""
    client = _stubbed_client(monkeypatch, [_piece(content="cut "), _piece(content="off")])
    with pytest.raises(LLMError):
        await client._stream_openai_tools_response(None, _Recorder())


@pytest.mark.asyncio
async def test_openai_tools_empty_stream_raises__DoD9(monkeypatch):
    """Tools parser: an empty stream raises LLMError (DoD-9)."""
    client = _stubbed_client(monkeypatch, [])
    with pytest.raises(LLMError):
        await client._stream_openai_tools_response(None, _Recorder())


@pytest.mark.asyncio
async def test_openai_tools_stream_done_only_returns__DoD9(monkeypatch):
    """Tools parser: [DONE] without any finish reason returns normally (DoD-9)."""
    content, tool_calls, pieces = await _run_tools(
        monkeypatch, [_piece(content="Hi "), _piece(content="there"), "[DONE]"]
    )
    assert content == "Hi there"
    assert tool_calls == []
    assert pieces == ["Hi ", "there"]


@pytest.mark.asyncio
async def test_openai_tools_stream_finish_reason_only_returns__DoD9(monkeypatch):
    """Tools parser: a tool-call round ending with a finish reason and no [DONE] returns normally (DoD-9)."""
    call = {"index": 0, "id": "call_1", "function": {"name": "fetch_record", "arguments": '{"id": 7}'}}
    content, tool_calls, _pieces = await _run_tools(
        monkeypatch,
        [_piece(tool_calls=[call]), _piece(finish_reason="tool_calls")],
    )
    assert content == ""
    assert tool_calls == [
        {"id": "call_1", "type": "function", "function": {"name": "fetch_record", "arguments": '{"id": 7}'}}
    ]


@pytest.mark.asyncio
async def test_openai_tools_stream_finish_reason_on_content_returns__DoD9(monkeypatch):
    """Tools parser: a finish reason on the last content piece without [DONE] returns normally (DoD-9)."""
    content, _tool_calls, _pieces = await _run_tools(
        monkeypatch, [_piece(content="Answer "), _piece(content="42", finish_reason="stop")]
    )
    assert content == "Answer 42"


# --------------------------------------------------------------------------- DoD-10


@pytest.mark.asyncio
async def test_openai_plain_piece_with_reasoning_and_content_opening_block__DoD10(monkeypatch):
    """A first piece with both reasoning and content gives '<think>'+reasoning, then '</think>'+content (DoD-10)."""
    text, pieces = await _run_plain(
        monkeypatch,
        [_piece(reasoning="Short thought.", content="The answer."), _piece(content=" More.", finish_reason="stop")],
    )
    assert pieces == ["<think>Short thought.", "</think>The answer.", " More."]
    assert text == "".join(pieces)


@pytest.mark.asyncio
async def test_openai_plain_piece_with_reasoning_and_content_inside_open_block__DoD10(monkeypatch):
    """With the block already open, the reasoning comes bare, then '</think>' + content (DoD-10)."""
    text, pieces = await _run_plain(
        monkeypatch,
        [
            _piece(reasoning="Step one. "),
            _piece(reasoning="Step two.", content="Result."),
            "[DONE]",
        ],
    )
    assert pieces == ["<think>Step one. ", "Step two.", "</think>Result."]
    assert text == "".join(pieces)
    assert text == "<think>Step one. Step two.</think>Result."


# --------------------------------------------------------------------------- DoD-11


def _malformed_payloads() -> list[str]:
    """Payloads DEC-001-9 says are skipped: non-JSON, JSON non-objects, objects without choices or error."""
    return [
        "this is not json",
        "{broken json",
        "[1, 2, 3]",
        "42",
        '"just a string"',
        "null",
        json.dumps({"usage": {"prompt_tokens": 5, "completion_tokens": 2}}),
        json.dumps({"choices": [], "usage": {"total_tokens": 7}}),
    ]


@pytest.mark.asyncio
async def test_openai_plain_skips_malformed_payloads__DoD11(monkeypatch):
    """Plain parser skips malformed payloads without raising; other text arrives intact (DoD-11)."""
    bad = _malformed_payloads()
    payloads = [_piece(content="Alpha ")] + bad[:4] + [_piece(content="Beta ")] + bad[4:]
    payloads += [_piece(content="Gamma", finish_reason="stop"), "[DONE]"]
    text, pieces = await _run_plain(monkeypatch, payloads)
    assert pieces == ["Alpha ", "Beta ", "Gamma"]
    assert text == "Alpha Beta Gamma"


@pytest.mark.asyncio
async def test_openai_tools_skips_malformed_payloads__DoD11(monkeypatch):
    """Tools parser skips malformed payloads without raising; other text arrives intact (DoD-11)."""
    bad = _malformed_payloads()
    payloads = [_piece(content="Alpha ")] + bad[:4] + [_piece(content="Beta ")] + bad[4:]
    payloads += [_piece(content="Gamma", finish_reason="stop"), "[DONE]"]
    content, tool_calls, pieces = await _run_tools(monkeypatch, payloads)
    assert pieces == ["Alpha ", "Beta ", "Gamma"]
    assert content == "Alpha Beta Gamma"
    assert tool_calls == []


# --------------------------------------------------------------------------- DoD-12


@pytest.mark.asyncio
async def test_openai_plain_reasoning_one_block_before_answer__DoD12(monkeypatch):
    """Reasoning over several pieces forms one marked block before the answer (DoD-12)."""
    text, pieces = await _run_plain(
        monkeypatch,
        [
            _piece(reasoning="First "),
            _piece(reasoning="second."),
            _piece(content="Answer "),
            _piece(content="here.", finish_reason="stop"),
            "[DONE]",
        ],
    )
    assert text == "<think>First second.</think>Answer here."
    assert text.count("<think>") == 1
    assert text.count("</think>") == 1
    assert "".join(pieces) == text


@pytest.mark.asyncio
async def test_openai_plain_reasoning_closed_at_end_if_open__DoD12(monkeypatch):
    """A block still open at the end of the stream is closed, for the receiver and the returned text (DoD-12)."""
    text, pieces = await _run_plain(monkeypatch, [_piece(reasoning="Only thinking."), "[DONE]"])
    assert text == "<think>Only thinking.</think>"
    assert "".join(pieces) == text


@pytest.mark.asyncio
async def test_openai_plain_no_markers_without_reasoning__DoD12(monkeypatch):
    """A model that does not reason gets no markers at all (DoD-12)."""
    text, pieces = await _run_plain(
        monkeypatch, [_piece(content="Plain "), _piece(content="answer.", finish_reason="stop"), "[DONE]"]
    )
    assert text == "Plain answer."
    assert pieces == ["Plain ", "answer."]
    assert "<think>" not in text and "</think>" not in text


# --------------------------------------------------------------------------- DoD-13


class _ReceiverBoom(Exception):
    """App-side failure raised by a test receiver; must propagate unchanged."""


def _failing_receiver(fail_on: int, error: BaseException):
    """Receiver that records pieces and raises ``error`` on the ``fail_on``-th call (1-based)."""
    pieces: list[str] = []

    async def receiver(text: str) -> None:
        pieces.append(text)
        if len(pieces) == fail_on:
            raise error

    return receiver, pieces


@pytest.mark.asyncio
async def test_openai_plain_receiver_failure_propagates__DoD13(monkeypatch):
    """Plain parser: a receiver failure ends the call with that same exception; nothing later is delivered (DoD-13)."""
    error = _ReceiverBoom("receiver broke")
    receiver, pieces = _failing_receiver(2, error)
    client = _stubbed_client(
        monkeypatch, [_piece(content="one "), _piece(content="two "), _piece(content="three"), "[DONE]"]
    )
    with pytest.raises(_ReceiverBoom) as info:
        await client._stream_openai_chat_response(None, receiver)
    assert info.value is error
    assert pieces == ["one ", "two "]


@pytest.mark.asyncio
async def test_openai_tools_receiver_failure_propagates__DoD13(monkeypatch):
    """Tools parser: a receiver failure ends the call with that same exception; nothing later is delivered (DoD-13)."""
    error = _ReceiverBoom("receiver broke")
    receiver, pieces = _failing_receiver(2, error)
    client = _stubbed_client(
        monkeypatch, [_piece(content="one "), _piece(content="two "), _piece(content="three"), "[DONE]"]
    )
    with pytest.raises(_ReceiverBoom) as info:
        await client._stream_openai_tools_response(None, receiver)
    assert info.value is error
    assert pieces == ["one ", "two "]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_factory",
    [
        pytest.param(lambda: TimeoutError("receiver timed out"), id="timeout"),
        pytest.param(lambda: aiohttp.ClientPayloadError("receiver payload"), id="client-error"),
    ],
)
@pytest.mark.parametrize("parser", ["plain", "tools"])
async def test_openai_receiver_transport_like_failure_not_wrapped__DoD13(parser, error_factory):
    """Through the real reader, a receiver raising a transport-like error gets it back unwrapped (DoD-13, DEC-001-7)."""
    error = error_factory()
    receiver, pieces = _failing_receiver(1, error)
    body = _sse(_piece(content="first"), _piece(content="second", finish_reason="stop"), "[DONE]")
    client = _client()
    response = _FakeResponse([body])
    with pytest.raises(type(error)) as info:
        if parser == "plain":
            await client._stream_openai_chat_response(response, receiver)
        else:
            await client._stream_openai_tools_response(response, receiver)
    assert info.value is error
    assert not isinstance(info.value, LLMError)
    assert pieces == ["first"]
