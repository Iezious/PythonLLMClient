"""Unit tests for llama-swap Responses chat: a reply cut at the app's own output limit is returned.

Per DEC-arch-13 and DEC-003-17..20, covers ``LlamaSwapAPIClient``'s chat-style
calls (``chat``, ``generate``, ``chat_json``; streamed and not), the module-level reply reader
``_responses_reply_text`` and the stream parser ``_stream_responses_response``: a reply or ``response.incomplete``
event whose reason is exactly ``max_output_tokens`` is returned like a completed reply when the app set the
output limit (``max_tokens`` or the ``context_length`` alias, call option or default), and stays ``LLMError``
when the app set no limit, for every other incomplete reason, and for failed / error / unfinished streams.
The server is faked through the DEC-003-16 seams (``_request``, ``_stream_post``, ``_iter_sse_data``).
Each test names its DoD id. DoD-12 (no app limit) is covered by the existing incomplete-reply tests, not here.
"""

import copy
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from llm.exceptions import LLMError
from llm.llama_swap import LlamaSwapAPIClient, _responses_reply_text

_BASE_URL = "http://localhost:8080/v1"
_MODEL = "some-model"

_OPEN = "<think>"
_CLOSE = "</think>"

_MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "Hello"}]

# The one reason that counts as a cut at the output limit (DEC-003-18).
_CUT_REASON = "max_output_tokens"
_CUT_DETAILS: dict[str, Any] = {"reason": _CUT_REASON}

# Every way the app sets the output limit (DEC-003-17): (client default options, call options).
_LIMIT_WAYS = [
    pytest.param(None, {"max_tokens": 16}, id="call-max_tokens"),
    pytest.param({"max_tokens": 16}, None, id="default-max_tokens"),
    pytest.param(None, {"context_length": 16}, id="call-context_length"),
    pytest.param({"context_length": 16}, None, id="default-context_length"),
]

# Every way the app sets no output limit (DEC-003-17, DEC-000-5): never set, or a default cleared per call.
_NO_LIMIT_WAYS = [
    pytest.param(None, None, id="never-set"),
    pytest.param(None, {"temperature": 0.2}, id="never-set-other-options"),
    pytest.param({"max_tokens": 16}, {"max_tokens": None}, id="default-cleared-with-none"),
    pytest.param({"context_length": 16}, {"context_length": None}, id="alias-default-cleared-with-none"),
]

# A plain call-option limit, used where the way the limit is set is not under test.
_LIMIT = {"max_tokens": 16}


# ---------------------------------------------------------------------------
# Reply and event builders (Responses wire vocabulary)
# ---------------------------------------------------------------------------


def _message_item(*texts: str | None) -> dict[str, Any]:
    """A Responses ``message`` output item with one ``output_text`` part per text."""
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": t, "annotations": []} for t in texts],
    }


def _reasoning_item(text: str) -> dict[str, Any]:
    """A reasoning output item in llama.cpp's shape (ignored by text extraction, DEC-003-12)."""
    return {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": text}]}


def _reply(*items: dict[str, Any], status: str = "completed", **extra: Any) -> dict[str, Any]:
    """A non-streamed Responses reply mapping with the given output items and status."""
    return {"object": "response", "status": status, "output": list(items), **extra}


def _cut_reply(*items: dict[str, Any]) -> dict[str, Any]:
    """A non-streamed reply cut at the output limit: status incomplete, reason ``max_output_tokens``."""
    return _reply(*items, status="incomplete", incomplete_details=dict(_CUT_DETAILS))


def _ev(event_type: str, **fields: Any) -> str:
    """One stream payload string: a JSON event object with ``type`` and the given fields."""
    return json.dumps({"type": event_type, **fields})


def _text(delta: str) -> str:
    """An output-text delta event."""
    return _ev("response.output_text.delta", item_id="msg_1", output_index=1, content_index=0, delta=delta)


def _reasoning(delta: str) -> str:
    """A reasoning-text delta event."""
    return _ev("response.reasoning_text.delta", item_id="rs_1", output_index=0, content_index=0, delta=delta)


def _incomplete_event(details: Any = None, *, with_details: bool = True) -> str:
    """A ``response.incomplete`` event; ``with_details=False`` omits ``incomplete_details`` entirely."""
    response: dict[str, Any] = {"id": "resp_1", "status": "incomplete", "output": []}
    if with_details:
        response["incomplete_details"] = details
    return _ev("response.incomplete", response=response)


_CUT_EVENT = _incomplete_event(dict(_CUT_DETAILS))
_COMPLETED = _ev("response.completed", response={"id": "resp_1", "status": "completed", "output": []})
_LATE_FAILED = _ev("response.failed", response={"status": "failed", "error": {"message": "late-failure-q1"}})


# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch):
    """Start every test with tracing off so a developer's shell env cannot change behaviour (DEC-000-48)."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _client(default_options: dict[str, Any] | None = None) -> LlamaSwapAPIClient:
    """Build a llama-swap Responses client on the fixed test address (no credential needed)."""
    return LlamaSwapAPIClient(
        model=_MODEL, base_url=_BASE_URL, default_options=copy.deepcopy(default_options) if default_options else None
    )


class _Recorder:
    """Async receiver collecting every piece it is handed, in order."""

    def __init__(self) -> None:
        self.pieces: list[str] = []

    async def __call__(self, piece: str) -> None:
        self.pieces.append(piece)


def _fake_request(monkeypatch, client: LlamaSwapAPIClient, reply: Any) -> list[dict[str, Any]]:
    """Replace ``_request`` with a fake returning a copy of ``reply``; returns the captured requests."""
    calls: list[dict[str, Any]] = []

    async def fake_request(method, path, **kw):
        calls.append({"method": method, "path": path, "json": copy.deepcopy(kw.get("json"))})
        return copy.deepcopy(reply)

    monkeypatch.setattr(client, "_request", fake_request)
    return calls


def _fake_sse(monkeypatch, client: LlamaSwapAPIClient, payloads: list[str]) -> None:
    """Replace ``_iter_sse_data`` with a fake yielding ``payloads`` (SSE data strings without ``data:``)."""

    async def fake_iter_sse_data(_response):
        for payload in payloads:
            yield payload

    monkeypatch.setattr(client, "_iter_sse_data", fake_iter_sse_data)


def _fake_stream_post(monkeypatch, client: LlamaSwapAPIClient) -> list[dict[str, Any]]:
    """Replace ``_stream_post`` with a fake async context manager capturing (path, payload).

    The yielded response is a placeholder: the parser only hands it to the faked ``_iter_sse_data``
    (DEC-003-16).
    """
    calls: list[dict[str, Any]] = []

    @asynccontextmanager
    async def fake_stream_post(path, payload):
        calls.append({"path": path, "payload": copy.deepcopy(payload)})
        yield None

    monkeypatch.setattr(client, "_stream_post", fake_stream_post)
    return calls


async def _nonstream_chat(monkeypatch, reply: Any, default_options=None, options=None) -> str:
    """Run one non-streamed ``chat`` against ``reply`` on a client with the given settings."""
    client = _client(default_options)
    _fake_request(monkeypatch, client, reply)
    return await client.chat(copy.deepcopy(_MESSAGES), options=copy.deepcopy(options))


async def _stream_chat(
    monkeypatch, payloads: list[str], receiver: Any, default_options=None, options=None
) -> str:
    """Run one streamed ``chat`` over canned payloads on a client with the given settings."""
    client = _client(default_options)
    _fake_stream_post(monkeypatch, client)
    _fake_sse(monkeypatch, client, payloads)
    return await client.chat(copy.deepcopy(_MESSAGES), options=copy.deepcopy(options), stream=True, on_delta=receiver)


# ---------------------------------------------------------------------------
# DoD-1 — non-streamed chat with a call-option max_tokens returns the cut text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_chat_cut_at_call_max_tokens_returns_text_as_completed__DoD1(monkeypatch):
    """DoD-1: a cut reply with ``max_tokens`` as a call option returns the text a completed reply would."""
    items = (_reasoning_item("hidden reasoning"), _message_item("Hello", ", "), _message_item("world"))

    cut = await _nonstream_chat(monkeypatch, _cut_reply(*items), options={"max_tokens": 16})
    completed = await _nonstream_chat(monkeypatch, _reply(*items), options={"max_tokens": 16})

    assert cut == "Hello, world"
    assert cut == completed


@pytest.mark.asyncio
async def test_llamaswap_chat_cut_without_text_returns_empty__DoD1(monkeypatch):
    """DoD-1: a cut reply with no message text returns empty text, as a completed one does (DEC-003-19)."""
    result = await _nonstream_chat(monkeypatch, _cut_reply(_reasoning_item("only thinking")), options=_LIMIT)

    assert result == ""


def test_llamaswap_reader_with_flag_extracts_cut_text_as_completed__DoD1():
    """DoD-1: the reader with the flag on extracts a cut reply's text exactly as from a completed reply."""
    items = (_message_item("Partial", " answer"), _reasoning_item("r"), _message_item(" here"))

    cut = _responses_reply_text(_cut_reply(*items), output_limit_set=True)
    completed = _responses_reply_text(_reply(*items), output_limit_set=True)

    assert cut == "Partial answer here"
    assert cut == completed


# ---------------------------------------------------------------------------
# DoD-2 — every way the app sets the limit (defaults, context_length alias)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("default_options", "options"), _LIMIT_WAYS)
async def test_llamaswap_chat_cut_returns_text_for_every_way_the_limit_is_set__DoD2(
    monkeypatch, default_options, options
):
    """DoD-2: limit as call option or default, via ``max_tokens`` or ``context_length``: cut text returned."""
    result = await _nonstream_chat(
        monkeypatch, _cut_reply(_message_item("Cut ", "text")), default_options=default_options, options=options
    )

    assert result == "Cut text"


# ---------------------------------------------------------------------------
# DoD-3 — non-streamed generate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("default_options", "options"), _LIMIT_WAYS)
async def test_llamaswap_generate_cut_returns_text__DoD3(monkeypatch, default_options, options):
    """DoD-3: non-streamed ``generate`` with the app's limit set returns the cut text."""
    client = _client(default_options)
    _fake_request(monkeypatch, client, _cut_reply(_message_item("Generated ", "part")))

    result = await client.generate("Write a line.", options=copy.deepcopy(options))

    assert result == "Generated part"


# ---------------------------------------------------------------------------
# DoD-4 — chat_json
# ---------------------------------------------------------------------------

_SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}}}


@pytest.mark.asyncio
@pytest.mark.parametrize(("default_options", "options"), _LIMIT_WAYS)
async def test_llamaswap_chat_json_without_shape_returns_cut_text__DoD4(monkeypatch, default_options, options):
    """DoD-4: ``chat_json`` without an expected shape and with the app's limit set returns the cut text."""
    cut_text = '{"answer": 4'
    client = _client(default_options)
    _fake_request(monkeypatch, client, _cut_reply(_message_item(cut_text)))

    result = await client.chat_json(copy.deepcopy(_MESSAGES), options=copy.deepcopy(options))

    assert result == cut_text


@pytest.mark.asyncio
async def test_llamaswap_chat_json_with_shape_cut_invalid_json_fails_as_completed__DoD4(monkeypatch):
    """DoD-4: with an expected shape, a cut answer that is not valid JSON ends with the completed-reply error."""
    invalid = '{"answer": 4'

    completed_client = _client()
    _fake_request(monkeypatch, completed_client, _reply(_message_item(invalid)))
    with pytest.raises(LLMError) as completed_exc:
        await completed_client.chat_json(copy.deepcopy(_MESSAGES), response_schema=_SCHEMA, options=dict(_LIMIT))

    cut_client = _client()
    _fake_request(monkeypatch, cut_client, _cut_reply(_message_item(invalid)))
    with pytest.raises(LLMError) as cut_exc:
        await cut_client.chat_json(copy.deepcopy(_MESSAGES), response_schema=_SCHEMA, options=dict(_LIMIT))

    assert type(cut_exc.value) is type(completed_exc.value)
    assert str(cut_exc.value) == str(completed_exc.value)
    assert getattr(cut_exc.value, "description", None) == getattr(completed_exc.value, "description", None)
    assert _CUT_REASON not in str(cut_exc.value)


@pytest.mark.asyncio
async def test_llamaswap_chat_json_with_shape_cut_valid_json_is_returned__DoD4(monkeypatch):
    """DoD-4: with an expected shape, a cut answer that is still valid JSON passes the unchanged check."""
    valid = '{"answer": 4}'
    client = _client()
    _fake_request(monkeypatch, client, _cut_reply(_message_item(valid)))

    result = await client.chat_json(copy.deepcopy(_MESSAGES), response_schema=_SCHEMA, options=dict(_LIMIT))

    assert result == valid


# ---------------------------------------------------------------------------
# DoD-5 — no app limit: a cut for the output limit stays LLMError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("default_options", "options"), _NO_LIMIT_WAYS)
async def test_llamaswap_chat_cut_without_app_limit_raises_with_reason__DoD5(monkeypatch, default_options, options):
    """DoD-5: non-streamed, with no app limit a cut reply ends with LLMError carrying the server's reason."""
    with pytest.raises(LLMError) as excinfo:
        await _nonstream_chat(
            monkeypatch, _cut_reply(_message_item("partial")), default_options=default_options, options=options
        )

    assert _CUT_REASON in str(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["generate", "chat_json"])
async def test_llamaswap_generate_and_chat_json_cut_without_app_limit_raise__DoD5(monkeypatch, operation):
    """DoD-5: ``generate`` and ``chat_json`` with a default limit cleared per call raise on a cut reply."""
    client = _client({"max_tokens": 16})
    _fake_request(monkeypatch, client, _cut_reply(_message_item('{"a": 1}')))

    with pytest.raises(LLMError) as excinfo:
        if operation == "generate":
            await client.generate("Prompt", options={"max_tokens": None})
        else:
            await client.chat_json(copy.deepcopy(_MESSAGES), options={"max_tokens": None})

    assert _CUT_REASON in str(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(("default_options", "options"), _NO_LIMIT_WAYS)
async def test_llamaswap_streamed_chat_cut_without_app_limit_raises_with_reason__DoD5(
    monkeypatch, default_options, options
):
    """DoD-5: streamed, no app limit: a cut event ends with LLMError carrying the reason; nothing after it."""
    rec = _Recorder()
    with pytest.raises(LLMError) as excinfo:
        await _stream_chat(
            monkeypatch,
            [_text("before"), _CUT_EVENT, _text("after"), _COMPLETED],
            rec,
            default_options=default_options,
            options=options,
        )

    assert _CUT_REASON in str(excinfo.value)
    assert rec.pieces == ["before"]


# ---------------------------------------------------------------------------
# DoD-6 — app limit set, but another incomplete reason / no reason / no details: LLMError
# ---------------------------------------------------------------------------

# (incomplete details or a "missing" marker, text the error must carry or None when only LLMError is required).
_MISSING = object()
_OTHER_INCOMPLETE = [
    pytest.param({"reason": "content_filter"}, "content_filter", id="other-reason"),
    pytest.param({"reason": "MAX_OUTPUT_TOKENS"}, "MAX_OUTPUT_TOKENS", id="different-case"),
    pytest.param({"reason": "max_output_tokens_exceeded"}, "max_output_tokens_exceeded", id="prefix-only"),
    pytest.param({"reason": 42}, None, id="non-string-reason"),
    pytest.param({"detail": "trunc-k3"}, "trunc-k3", id="no-reason"),
    pytest.param(None, None, id="null-details"),
    pytest.param(_MISSING, None, id="missing-details"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("details", "carried"), _OTHER_INCOMPLETE)
async def test_llamaswap_chat_other_incomplete_with_limit_raises__DoD6(monkeypatch, details, carried):
    """DoD-6: non-streamed, with the app's limit set, any non-matching incomplete reply ends with LLMError."""
    reply = _reply(_message_item("partial"), status="incomplete")
    if details is not _MISSING:
        reply["incomplete_details"] = copy.deepcopy(details)

    with pytest.raises(LLMError) as excinfo:
        await _nonstream_chat(monkeypatch, reply, options=_LIMIT)

    if carried is not None:
        assert carried in str(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(("details", "carried"), _OTHER_INCOMPLETE)
async def test_llamaswap_streamed_chat_other_incomplete_with_limit_raises__DoD6(monkeypatch, details, carried):
    """DoD-6: streamed, with the app's limit set, any non-matching incomplete event ends with LLMError."""
    if details is _MISSING:
        event = _incomplete_event(with_details=False)
    else:
        event = _incomplete_event(copy.deepcopy(details))
    rec = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await _stream_chat(monkeypatch, [_text("before"), event, _text("after"), _COMPLETED], rec, options=_LIMIT)

    if carried is not None:
        assert carried in str(excinfo.value)
    assert rec.pieces == ["before"]


# ---------------------------------------------------------------------------
# DoD-7 — app limit set: failed reply / failed event / error event / unfinished stream stay LLMError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_chat_failed_reply_with_limit_raises_with_reason__DoD7(monkeypatch):
    """DoD-7: non-streamed, with the app's limit set, a failed reply ends with LLMError carrying the reason."""
    reply = _reply(status="failed", error={"code": "server_error", "message": "engine-crashed-f4"})

    with pytest.raises(LLMError) as excinfo:
        await _nonstream_chat(monkeypatch, reply, options=_LIMIT)

    assert "engine-crashed-f4" in str(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "reason"),
    [
        pytest.param(
            _ev("response.failed", response={"status": "failed", "error": {"code": "s", "message": "boom-f7"}}),
            "boom-f7",
            id="failed-event",
        ),
        pytest.param(_ev("error", code="rate_limit", message="slow-down-e2"), "slow-down-e2", id="error-event"),
    ],
)
async def test_llamaswap_streamed_failure_event_with_limit_raises_with_reason__DoD7(monkeypatch, event, reason):
    """DoD-7: streamed, with the app's limit set, a failed or error event ends with LLMError carrying the reason."""
    rec = _Recorder()
    with pytest.raises(LLMError) as excinfo:
        await _stream_chat(monkeypatch, [_text("before"), event, _text("after"), _COMPLETED], rec, options=_LIMIT)

    assert reason in str(excinfo.value)
    assert rec.pieces == ["before"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payloads",
    [
        pytest.param([_text("partial"), _text(" answer")], id="no-completion-event"),
        pytest.param([_text("partial"), "[DONE]"], id="done-sentinel-only"),
        pytest.param([], id="empty-stream"),
    ],
)
async def test_llamaswap_streamed_unfinished_stream_with_limit_raises__DoD7(monkeypatch, payloads):
    """DoD-7: streamed, with the app's limit set, a stream ending without a completion event ends with LLMError."""
    with pytest.raises(LLMError):
        await _stream_chat(monkeypatch, payloads, _Recorder(), options=_LIMIT)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payloads",
    [
        pytest.param(
            [_ev("response.failed", response={"status": "failed", "error": {"message": "boom-p1"}})], id="failed"
        ),
        pytest.param([_ev("error", message="boom-p1")], id="error"),
    ],
)
async def test_llamaswap_parser_with_flag_failure_events_raise__DoD7(monkeypatch, payloads):
    """DoD-7: the parser with the flag on still raises LLMError with the reason on failed and error events."""
    client = _client()
    _fake_sse(monkeypatch, client, [_text("x"), *payloads, _COMPLETED])

    with pytest.raises(LLMError) as excinfo:
        await client._stream_responses_response(None, _Recorder(), output_limit_set=True)

    assert "boom-p1" in str(excinfo.value)


@pytest.mark.asyncio
async def test_llamaswap_parser_with_flag_unfinished_stream_raises__DoD7(monkeypatch):
    """DoD-7: the parser with the flag on still raises LLMError when the stream ends without completion."""
    client = _client()
    _fake_sse(monkeypatch, client, [_text("x"), "[DONE]"])

    with pytest.raises(LLMError):
        await client._stream_responses_response(None, _Recorder(), output_limit_set=True)


# ---------------------------------------------------------------------------
# DoD-8 — streamed chat with the app's limit: pieces before the cut, nothing after, return == received
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("default_options", "options"), _LIMIT_WAYS)
async def test_llamaswap_streamed_chat_cut_returns_pieces_received__DoD8(monkeypatch, default_options, options):
    """DoD-8: pieces before a matching cut event reach the receiver in order; nothing after; return == received."""
    rec = _Recorder()
    payloads = [_text("One "), _text("two "), _text("thr"), _CUT_EVENT, _text("AFTER"), _LATE_FAILED, _COMPLETED]

    result = await _stream_chat(monkeypatch, payloads, rec, default_options=default_options, options=options)

    assert rec.pieces == ["One ", "two ", "thr"]
    assert result == "One two thr"
    assert result == "".join(rec.pieces)


@pytest.mark.asyncio
async def test_llamaswap_streamed_chat_cut_without_receiver_returns_text__DoD8(monkeypatch):
    """DoD-8: with no receiver, a streamed cut at the app's limit returns the text streamed before the cut."""
    result = await _stream_chat(monkeypatch, [_text("a"), _text("b"), _CUT_EVENT, _text("c")], None, options=_LIMIT)

    assert result == "ab"


@pytest.mark.asyncio
async def test_llamaswap_parser_with_flag_ends_at_cut_event__DoD8(monkeypatch):
    """DoD-8: the parser with the flag on ends at a matching cut event as at completion; nothing after it."""
    client = _client()
    _fake_sse(monkeypatch, client, [_text("x"), _text("y"), _CUT_EVENT, _text("z"), _LATE_FAILED])
    rec = _Recorder()

    result = await client._stream_responses_response(None, rec, output_limit_set=True)

    assert rec.pieces == ["x", "y"]
    assert result == "xy"


# ---------------------------------------------------------------------------
# DoD-9 — open reasoning block closed at the cut; streamed generate == one-message chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_streamed_cut_inside_reasoning_closes_block_with_own_piece__DoD9(monkeypatch):
    """DoD-9: a reasoning block open at a matching cut is closed by a ``</think>`` piece of its own."""
    rec = _Recorder()
    payloads = [_reasoning("Let me"), _reasoning(" think"), _CUT_EVENT, _text("AFTER"), _COMPLETED]

    result = await _stream_chat(monkeypatch, payloads, rec, options=_LIMIT)

    assert rec.pieces == [f"{_OPEN}Let me", " think", _CLOSE]
    assert result == f"{_OPEN}Let me think{_CLOSE}"
    assert result == "".join(rec.pieces)


@pytest.mark.asyncio
async def test_llamaswap_streamed_cut_in_second_reasoning_run_closes_it__DoD9(monkeypatch):
    """DoD-9: text then a new reasoning run open at the cut: the run is closed by its own ``</think>`` piece."""
    rec = _Recorder()
    payloads = [_reasoning("r1"), _text("t1"), _reasoning("r2"), _CUT_EVENT, _text("AFTER")]

    result = await _stream_chat(monkeypatch, payloads, rec, options=_LIMIT)

    assert rec.pieces == [f"{_OPEN}r1", f"{_CLOSE}t1", f"{_OPEN}r2", _CLOSE]
    assert result == "".join(rec.pieces)


@pytest.mark.asyncio
async def test_llamaswap_parser_with_flag_closes_open_block_at_cut__DoD9(monkeypatch):
    """DoD-9: the parser with the flag on closes an open reasoning block at the cut with its own piece."""
    client = _client()
    _fake_sse(monkeypatch, client, [_reasoning("deep"), _CUT_EVENT, _reasoning("more")])
    rec = _Recorder()

    result = await client._stream_responses_response(None, rec, output_limit_set=True)

    assert rec.pieces == [f"{_OPEN}deep", _CLOSE]
    assert result == f"{_OPEN}deep{_CLOSE}"


@pytest.mark.asyncio
async def test_llamaswap_streamed_generate_cut_matches_one_message_chat__DoD9(monkeypatch):
    """DoD-9: streamed ``generate`` cut at the app's limit gives the pieces, return and request of the chat."""
    payloads = [_reasoning("hm"), _text("Gen "), _text("ou"), _CUT_EVENT, _text("AFTER")]

    chat_client = _client()
    chat_calls = _fake_stream_post(monkeypatch, chat_client)
    _fake_sse(monkeypatch, chat_client, payloads)
    chat_rec = _Recorder()
    chat_result = await chat_client.chat(
        [{"role": "user", "content": "Write a line."}],
        system="Be terse.",
        options=dict(_LIMIT),
        stream=True,
        on_delta=chat_rec,
    )

    gen_client = _client()
    gen_calls = _fake_stream_post(monkeypatch, gen_client)
    _fake_sse(monkeypatch, gen_client, payloads)
    gen_rec = _Recorder()
    gen_result = await gen_client.generate(
        "Write a line.", system="Be terse.", options=dict(_LIMIT), stream=True, on_delta=gen_rec
    )

    assert chat_rec.pieces == [f"{_OPEN}hm", f"{_CLOSE}Gen ", "ou"]
    assert gen_rec.pieces == chat_rec.pieces
    assert gen_result == chat_result == "".join(gen_rec.pieces)
    assert len(gen_calls) == 1
    assert gen_calls[0]["payload"] == chat_calls[0]["payload"]


@pytest.mark.asyncio
async def test_llamaswap_streamed_generate_cut_inside_reasoning_closes_block__DoD9(monkeypatch):
    """DoD-9: streamed ``generate`` with a reasoning block open at the cut closes it with its own piece."""
    client = _client({"context_length": 16})
    _fake_stream_post(monkeypatch, client)
    _fake_sse(monkeypatch, client, [_reasoning("thinking"), _CUT_EVENT])
    rec = _Recorder()

    result = await client.generate("Prompt", stream=True, on_delta=rec)

    assert rec.pieces == [f"{_OPEN}thinking", _CLOSE]
    assert result == f"{_OPEN}thinking{_CLOSE}"


# ---------------------------------------------------------------------------
# DoD-10 — reader and parser without the flag keep the old rule
# ---------------------------------------------------------------------------


def test_llamaswap_reader_without_flag_raises_on_cut__DoD10():
    """DoD-10: the reader called without the flag raises LLMError with the reason on a cut reply."""
    with pytest.raises(LLMError) as excinfo:
        _responses_reply_text(_cut_reply(_message_item("partial")))

    assert _CUT_REASON in str(excinfo.value)


def test_llamaswap_reader_with_flag_off_raises_on_cut__DoD10():
    """DoD-10: the reader with the flag explicitly off behaves as its default: LLMError with the reason."""
    with pytest.raises(LLMError) as excinfo:
        _responses_reply_text(_cut_reply(_message_item("partial")), output_limit_set=False)

    assert _CUT_REASON in str(excinfo.value)


@pytest.mark.asyncio
async def test_llamaswap_parser_without_flag_raises_on_cut_event__DoD10(monkeypatch):
    """DoD-10: the parser called without the flag raises LLMError with the reason at a cut event."""
    client = _client()
    _fake_sse(monkeypatch, client, [_text("before"), _CUT_EVENT, _text("after"), _COMPLETED])
    rec = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await client._stream_responses_response(None, rec)

    assert _CUT_REASON in str(excinfo.value)
    assert rec.pieces == ["before"]


@pytest.mark.asyncio
async def test_llamaswap_parser_with_flag_off_raises_on_cut_event__DoD10(monkeypatch):
    """DoD-10: the parser with the flag explicitly off raises LLMError with the reason at a cut event."""
    client = _client()
    _fake_sse(monkeypatch, client, [_reasoning("r"), _CUT_EVENT, _COMPLETED])
    rec = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await client._stream_responses_response(None, rec, output_limit_set=False)

    assert _CUT_REASON in str(excinfo.value)
    assert rec.pieces == [f"{_OPEN}r"]


# ---------------------------------------------------------------------------
# DoD-11 — tracing: the cut reply is returned and recorded as the call's response
# ---------------------------------------------------------------------------

# Section titles of the trace record (docs/architecture/data-model.md § Trace record; as in test_llm_trace.py).
_HEADINGS = frozenset({"SYSTEM PROMPT", "MESSAGE HISTORY", "REQUEST", "RESPONSE", "CHUNK", "ERROR", "END TRACE"})


def _sections(text: str) -> list[tuple[str, list[str]]]:
    """Split a record into ``(title, body lines)``; blank and pure rule lines are layout and skipped."""
    sections: list[tuple[str, list[str]]] = [("HEADER", [])]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) <= {"=", "-"}:
            continue
        if stripped in _HEADINGS:
            sections.append((stripped, []))
        else:
            sections[-1][1].append(stripped)
    return sections


def _single_record(folder: Path) -> str:
    """The text of the one trace record written under ``folder``."""
    records = sorted(folder.rglob("*.log"))
    assert len(records) == 1, records
    return records[0].read_text(encoding="utf-8")


def _traced_client(monkeypatch, folder: Path) -> LlamaSwapAPIClient:
    """A client with tracing on into ``folder``; tracing is read at construction, so env is set first."""
    folder.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(folder))
    return _client()


@pytest.mark.asyncio
async def test_llamaswap_traced_cut_reply_is_returned_and_recorded_as_response__DoD11(monkeypatch, tmp_path):
    """DoD-11: traced non-streamed chat returns a cut reply and records it as the response, like a completed one."""
    text = "Traced cut answer 9c2e"

    completed_client = _traced_client(monkeypatch, tmp_path / "completed")
    _fake_request(monkeypatch, completed_client, _reply(_message_item(text)))
    completed_result = await completed_client.chat(copy.deepcopy(_MESSAGES), options=dict(_LIMIT))

    cut_client = _traced_client(monkeypatch, tmp_path / "cut")
    _fake_request(monkeypatch, cut_client, _cut_reply(_message_item(text)))
    cut_result = await cut_client.chat(copy.deepcopy(_MESSAGES), options=dict(_LIMIT))

    assert cut_result == completed_result == text
    cut_sections = _sections(_single_record(tmp_path / "cut"))
    completed_sections = _sections(_single_record(tmp_path / "completed"))
    cut_titles = [title for title, _ in cut_sections]
    assert "ERROR" not in cut_titles
    assert cut_titles == [title for title, _ in completed_sections]
    cut_responses = [body for title, body in cut_sections if title == "RESPONSE"]
    assert cut_responses == [body for title, body in completed_sections if title == "RESPONSE"]
    assert any(text in line for body in cut_responses for line in body)


@pytest.mark.asyncio
async def test_llamaswap_traced_streamed_cut_is_returned_and_recorded_as_response__DoD11(monkeypatch, tmp_path):
    """DoD-11: traced streamed chat cut at the app's limit returns the text and records it like a completed one."""
    pieces = [_text("Streamed "), _text("cut 4b1d")]

    completed_client = _traced_client(monkeypatch, tmp_path / "completed")
    _fake_stream_post(monkeypatch, completed_client)
    _fake_sse(monkeypatch, completed_client, [*pieces, _COMPLETED])
    completed_rec = _Recorder()
    completed_result = await completed_client.chat(
        copy.deepcopy(_MESSAGES), options=dict(_LIMIT), stream=True, on_delta=completed_rec
    )

    cut_client = _traced_client(monkeypatch, tmp_path / "cut")
    _fake_stream_post(monkeypatch, cut_client)
    _fake_sse(monkeypatch, cut_client, [*pieces, _CUT_EVENT, _text("AFTER")])
    cut_rec = _Recorder()
    cut_result = await cut_client.chat(copy.deepcopy(_MESSAGES), options=dict(_LIMIT), stream=True, on_delta=cut_rec)

    assert cut_result == completed_result == "Streamed cut 4b1d"
    assert cut_rec.pieces == completed_rec.pieces
    cut_sections = _sections(_single_record(tmp_path / "cut"))
    completed_sections = _sections(_single_record(tmp_path / "completed"))
    cut_titles = [title for title, _ in cut_sections]
    assert "ERROR" not in cut_titles
    assert cut_titles == [title for title, _ in completed_sections]
    cut_responses = [body for title, body in cut_sections if title == "RESPONSE"]
    assert cut_responses == [body for title, body in completed_sections if title == "RESPONSE"]
    assert cut_responses, "the cut reply is recorded as the call's response"
