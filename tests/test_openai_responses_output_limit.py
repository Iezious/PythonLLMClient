"""Unit tests for a reply cut at the app's own output limit on the OpenAI Responses class
(DEC-arch-13, DEC-003-17..20).

When the app set the output limit (``max_tokens`` or its ``context_length`` alias, as a call option or a client
default), a non-streamed reply with status ``incomplete`` and reason exactly ``max_output_tokens``, or a matching
``response.incomplete`` stream event, is returned like a completed reply through ``chat``, ``generate`` and
``chat_json``. With no app limit, and for every other early end, the call still ends with ``LLMError``. The
server is faked through the ``_request`` seam (non-streamed) or ``_stream_post`` + ``_iter_sse_data``
(streamed), per DEC-003-16. Every expected value comes from the step's Definition of done and the cited
decisions.
"""

import copy
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from llm.exceptions import LLMError
from llm.openai import OpenAIAPIClient, _responses_reply_text

BASE_URL = "https://api.example.test/v1"
CREDENTIAL = "sk-test"
MODEL = "gpt-6-luna"

USER_MESSAGES = [{"role": "user", "content": "Tell me a long story."}]
PROMPT = "Tell me a long story."
ANSWER_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}

LIMIT = 16
CUT_REASON = "max_output_tokens"

# The ways the app can set the output limit (DEC-003-17): (client defaults, call options).
LIMIT_SOURCES = [
    pytest.param(None, {"max_tokens": LIMIT}, id="call-max_tokens"),
    pytest.param({"max_tokens": LIMIT}, None, id="default-max_tokens"),
    pytest.param(None, {"context_length": LIMIT}, id="call-context_length"),
    pytest.param({"context_length": LIMIT}, None, id="default-context_length"),
]

# The ways the app sets no limit (DEC-003-17, DEC-000-5): never set, or a default cleared per call with None.
NO_LIMIT_SOURCES = [
    pytest.param(None, None, id="never-set"),
    pytest.param({"max_tokens": LIMIT}, {"max_tokens": None}, id="default-max_tokens-cleared"),
    pytest.param({"context_length": LIMIT}, {"context_length": None}, id="default-context_length-cleared"),
]


# ---------------------------------------------------------------------------
# Non-streamed reply builders (DEC-003-12 shapes)
# ---------------------------------------------------------------------------


def _message_item(*texts: str) -> dict:
    """A Responses ``message`` output item with one ``output_text`` part per text."""
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": t} for t in texts]}


def _reasoning_item(text: str) -> dict:
    """A ``reasoning`` output item; DEC-003-12 says it never contributes to the extracted text."""
    return {"type": "reasoning", "summary": [{"type": "summary_text", "text": text}]}


# Output of a reply stopped part-way: reasoning ignored, message parts concatenated with no separator
# (DEC-003-12), so the extracted text is CUT_TEXT.
CUT_OUTPUT = [
    _reasoning_item("thinking hidden 51"),
    _message_item("Once upon a time, ", "there was a"),
    _message_item(" dragon who"),
]
CUT_TEXT = "Once upon a time, there was a dragon who"


def _reply(output: list[dict], *, status: str, **extra: Any) -> dict:
    """A non-streamed Responses reply mapping."""
    body: dict[str, Any] = {"id": "resp_1", "object": "response", "status": status, "output": copy.deepcopy(output)}
    body.update(extra)
    return body


def _completed(output: list[dict] = CUT_OUTPUT) -> dict:
    """A completed reply with the given output."""
    return _reply(output, status="completed")


def _cut(output: list[dict] = CUT_OUTPUT) -> dict:
    """A reply the server stopped at the output limit (status incomplete, reason exactly max_output_tokens)."""
    return _reply(output, status="incomplete", incomplete_details={"reason": CUT_REASON})


def _client(defaults: dict | None = None) -> OpenAIAPIClient:
    """A Responses client built directly with the app-provided test credential (tests/AGENTS.md)."""
    return OpenAIAPIClient(model=MODEL, base_url=BASE_URL, bearer_token=CREDENTIAL, default_options=defaults)


def _serve(monkeypatch, client: OpenAIAPIClient, reply: dict) -> list[dict]:
    """Replace the ``_request`` seam (DEC-003-16) with a fake answering ``reply``; records each call's
    method, path and a deep copy of the body so later mutation by the client cannot alter what was seen."""
    records: list[dict] = []

    async def fake_request(method, path, **kw):
        records.append({"method": method, "path": path, "json": copy.deepcopy(kw.get("json"))})
        return copy.deepcopy(reply)

    monkeypatch.setattr(client, "_request", fake_request)
    return records


# ---------------------------------------------------------------------------
# Stream event builders and seams (DEC-003-13, DEC-003-16)
# ---------------------------------------------------------------------------


def _event(event: dict[str, Any]) -> str:
    """One SSE data payload: the JSON text of a Responses stream event."""
    return json.dumps(event)


def _text(delta: str) -> str:
    """An output-text delta event."""
    return _event({"type": "response.output_text.delta", "item_id": "msg_1", "delta": delta})


def _reasoning(delta: str) -> str:
    """A reasoning-text delta event."""
    return _event({"type": "response.reasoning_text.delta", "item_id": "rs_1", "delta": delta})


def _incomplete_event(details: Any = None, *, with_details: bool = True) -> str:
    """A ``response.incomplete`` event; ``with_details=False`` omits ``incomplete_details`` entirely."""
    response: dict[str, Any] = {"id": "resp_1", "status": "incomplete"}
    if with_details:
        response["incomplete_details"] = details
    return _event({"type": "response.incomplete", "response": response})


CREATED = _event({"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}})
COMPLETED = _event({"type": "response.completed", "response": {"id": "resp_1", "status": "completed"}})
CUT_EVENT = _incomplete_event({"reason": CUT_REASON})
FAILED_EVENT = _event(
    {
        "type": "response.failed",
        "response": {"id": "resp_1", "status": "failed", "error": {"code": "server_error", "message": "Model fell 83"}},
    }
)
ERROR_EVENT = _event({"type": "error", "code": "rate_limit_exceeded", "message": "Slow down 29"})


class _Recorder:
    """Async receiver that records every piece it is given, in order."""

    def __init__(self) -> None:
        self.pieces: list[str] = []

    async def __call__(self, piece: str) -> None:
        self.pieces.append(piece)


def _install_sse(monkeypatch, client: OpenAIAPIClient, payloads: list[str]) -> None:
    """Replace ``_iter_sse_data`` with a fake yielding ``payloads`` as SSE data strings (DEC-003-16)."""

    async def fake_iter_sse_data(_response):
        for payload in payloads:
            yield payload

    monkeypatch.setattr(client, "_iter_sse_data", fake_iter_sse_data)


def _install_stream(monkeypatch, client: OpenAIAPIClient, payloads: list[str]) -> list[dict]:
    """Replace ``_stream_post`` with a fake async context manager recording ``(path, payload)`` and yielding a
    placeholder response, plus the ``_iter_sse_data`` fake (DEC-003-16)."""
    calls: list[dict] = []

    @asynccontextmanager
    async def fake_stream_post(path, payload):
        calls.append({"path": path, "payload": copy.deepcopy(payload)})
        yield None

    monkeypatch.setattr(client, "_stream_post", fake_stream_post)
    _install_sse(monkeypatch, client, payloads)
    return calls


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch):
    """Clear the trace switches so tests start untraced whatever the developer's environment; the DoD-11
    tests turn them on with ``monkeypatch.setenv`` before building their clients."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


# ---------------------------------------------------------------------------
# DoD-1 / DoD-2 — non-streamed chat returns the cut text when the app set the limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_chat_cut_at_call_limit_returns_text_as_completed__DoD1(monkeypatch):
    """DoD-1: with ``max_tokens`` as a call option, a reply cut for the output limit returns the text extracted
    exactly as from a completed reply with the same output (DEC-003-19, DEC-003-12)."""
    completed_client = _client()
    _serve(monkeypatch, completed_client, _completed())
    expected = await completed_client.chat(list(USER_MESSAGES), options={"max_tokens": LIMIT})

    client = _client()
    records = _serve(monkeypatch, client, _cut())
    result = await client.chat(list(USER_MESSAGES), options={"max_tokens": LIMIT})

    assert result == CUT_TEXT
    assert result == expected
    assert len(records) == 1 and records[0]["path"] == "/responses"


@pytest.mark.asyncio
async def test_openai_chat_cut_at_limit_without_text_returns_empty__DoD1(monkeypatch):
    """DoD-1: a cut reply holding no message text returns ``""``, as a completed reply without text does
    (DEC-003-19, DEC-003-12)."""
    client = _client()
    _serve(monkeypatch, client, _cut([_reasoning_item("only thinking")]))

    assert await client.chat(list(USER_MESSAGES), options={"max_tokens": LIMIT}) == ""


def test_openai_reply_reader_with_flag_on_extracts_cut_text__DoD1():
    """DoD-1 (reader level): with ``output_limit_set=True`` the reader extracts a cut reply's text as for a
    completed reply (Interface intent, DEC-003-19)."""
    assert _responses_reply_text(_cut(), output_limit_set=True) == CUT_TEXT
    assert _responses_reply_text(_cut(), output_limit_set=True) == _responses_reply_text(_completed())


@pytest.mark.asyncio
@pytest.mark.parametrize(("defaults", "options"), LIMIT_SOURCES)
async def test_openai_chat_cut_returns_text_for_every_limit_source__DoD2(monkeypatch, defaults, options):
    """DoD-2: the cut text is returned whether the limit is a call option or a client default, under
    ``max_tokens`` or the ``context_length`` alias (DEC-003-17)."""
    client = _client(defaults)
    _serve(monkeypatch, client, _cut())

    assert await client.chat(list(USER_MESSAGES), options=options) == CUT_TEXT


# ---------------------------------------------------------------------------
# DoD-3 — non-streamed generate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("defaults", "options"), LIMIT_SOURCES[:2])
async def test_openai_generate_cut_at_limit_returns_text__DoD3(monkeypatch, defaults, options):
    """DoD-3: non-streamed ``generate`` with the app's limit set returns the cut text (DEC-000-8)."""
    client = _client(defaults)
    _serve(monkeypatch, client, _cut())

    assert await client.generate(PROMPT, options=options) == CUT_TEXT


# ---------------------------------------------------------------------------
# DoD-4 — chat_json
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("defaults", "options"), LIMIT_SOURCES[:2])
async def test_openai_chat_json_without_shape_returns_cut_text__DoD4(monkeypatch, defaults, options):
    """DoD-4: ``chat_json`` without an expected shape, with the app's limit set, returns the cut text even
    though it is not valid JSON (no parse check without a shape, DEC-000-37)."""
    cut_json = '{"answer": "a dragon wh'
    client = _client(defaults)
    _serve(monkeypatch, client, _cut([_message_item(cut_json)]))

    assert await client.chat_json(list(USER_MESSAGES), options=options) == cut_json


@pytest.mark.asyncio
async def test_openai_chat_json_with_shape_cut_invalid_json_fails_like_completed__DoD4(monkeypatch):
    """DoD-4: with an expected shape and the app's limit set, a cut answer that is not valid JSON ends with the
    same error as a completed invalid-JSON answer (DEC-002-7, DEC-003-19)."""
    cut_json = '{"answer": "a dragon wh'
    output = [_message_item(cut_json)]

    completed_client = _client()
    _serve(monkeypatch, completed_client, _completed(output))
    with pytest.raises(LLMError) as completed_exc:
        await completed_client.chat_json(
            list(USER_MESSAGES), response_schema=ANSWER_SCHEMA, options={"max_tokens": LIMIT}
        )

    client = _client()
    _serve(monkeypatch, client, _cut(output))
    with pytest.raises(LLMError) as cut_exc:
        await client.chat_json(list(USER_MESSAGES), response_schema=ANSWER_SCHEMA, options={"max_tokens": LIMIT})

    assert type(cut_exc.value) is type(completed_exc.value)
    assert str(cut_exc.value) == str(completed_exc.value)
    assert getattr(cut_exc.value, "description", None) == getattr(completed_exc.value, "description", None)
    # The JSON check is what failed, not the incomplete status: the server's reason is not the error.
    assert getattr(cut_exc.value, "description", None) != CUT_REASON


@pytest.mark.asyncio
async def test_openai_chat_json_with_shape_cut_valid_json_returned__DoD4(monkeypatch):
    """DoD-4: with an expected shape and the app's limit set, a cut answer that still parses as JSON passes the
    unchanged check and is returned (DEC-003-19: the parse check runs as before)."""
    cut_json = '{"answer": "short"}'
    client = _client()
    _serve(monkeypatch, client, _cut([_message_item(cut_json)]))

    result = await client.chat_json(list(USER_MESSAGES), response_schema=ANSWER_SCHEMA, options={"max_tokens": LIMIT})

    assert result == cut_json


# ---------------------------------------------------------------------------
# DoD-5 — no app limit: a cut for the output limit stays LLMError, streamed and not
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("defaults", "options"), NO_LIMIT_SOURCES)
async def test_openai_chat_cut_without_app_limit_raises_with_reason__DoD5(monkeypatch, defaults, options):
    """DoD-5: with no app limit (never set, or a default cleared per call with ``None``), non-streamed ``chat``
    on a reply cut for the output limit ends with ``LLMError`` carrying the server's reason (DEC-003-17)."""
    client = _client(defaults)
    records = _serve(monkeypatch, client, _cut())

    with pytest.raises(LLMError) as excinfo:
        await client.chat(list(USER_MESSAGES), options=options)

    assert excinfo.value.description == CUT_REASON
    assert CUT_REASON in str(excinfo.value)
    # The merge's verdict: no limit was sent (DEC-003-17, DEC-000-5).
    assert "max_output_tokens" not in records[0]["json"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("defaults", "options"), NO_LIMIT_SOURCES)
async def test_openai_chat_json_cut_without_app_limit_raises_with_reason__DoD5(monkeypatch, defaults, options):
    """DoD-5: ``chat_json`` with no app limit on a reply cut for the output limit ends with ``LLMError``
    carrying the server's reason."""
    client = _client(defaults)
    _serve(monkeypatch, client, _cut([_message_item('{"answer": "x"}')]))

    with pytest.raises(LLMError) as excinfo:
        await client.chat_json(list(USER_MESSAGES), options=options)

    assert excinfo.value.description == CUT_REASON


@pytest.mark.asyncio
@pytest.mark.parametrize(("defaults", "options"), NO_LIMIT_SOURCES)
async def test_openai_streamed_chat_cut_without_app_limit_raises_with_reason__DoD5(monkeypatch, defaults, options):
    """DoD-5: streamed ``chat`` with no app limit ends with ``LLMError`` carrying the server's reason at a
    ``response.incomplete`` event for the output limit; nothing after the event is delivered."""
    client = _client(defaults)
    _install_stream(monkeypatch, client, [CREATED, _text("Before"), CUT_EVENT, _text("After"), COMPLETED])
    recorder = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await client.chat(list(USER_MESSAGES), options=options, stream=True, on_delta=recorder)

    assert excinfo.value.description == CUT_REASON
    assert recorder.pieces == ["Before"]


# ---------------------------------------------------------------------------
# DoD-6 — app limit set, but any other incomplete reason / no reason / no details stays LLMError
# ---------------------------------------------------------------------------


# (incomplete_details present?, its value, expected description or None when only LLMError is asserted).
# String reasons carry the reason; a mapping without a reason carries the details as JSON text (DEC-003-12/13).
_OTHER_INCOMPLETE = [
    pytest.param(True, {"reason": "content_filter"}, "content_filter", id="content-filter"),
    pytest.param(True, {"reason": "MAX_OUTPUT_TOKENS"}, "MAX_OUTPUT_TOKENS", id="different-case"),
    pytest.param(True, {"reason": "max_output_tokens_soft"}, "max_output_tokens_soft", id="prefix-match"),
    pytest.param(True, {"reason": 7}, None, id="non-string-reason"),
    pytest.param(True, {"x": "cut"}, json.dumps({"x": "cut"}), id="no-reason"),
    pytest.param(True, None, None, id="null-details"),
    pytest.param(False, None, None, id="missing-details"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("has_details", "details", "description"), _OTHER_INCOMPLETE)
async def test_openai_chat_other_incomplete_with_app_limit_raises__DoD6(
    monkeypatch, has_details, details, description
):
    """DoD-6: with the app's limit set, a non-streamed incomplete reply whose reason is any other value, has no
    reason, or has no incomplete details ends with ``LLMError`` as before (DEC-003-18)."""
    extra = {"incomplete_details": details} if has_details else {}
    client = _client()
    _serve(monkeypatch, client, _reply(CUT_OUTPUT, status="incomplete", **extra))

    with pytest.raises(LLMError) as excinfo:
        await client.chat(list(USER_MESSAGES), options={"max_tokens": LIMIT})

    if description is not None:
        if description.startswith("{"):
            assert json.loads(excinfo.value.description) == details
        else:
            assert excinfo.value.description == description


@pytest.mark.asyncio
@pytest.mark.parametrize(("has_details", "details", "description"), _OTHER_INCOMPLETE)
async def test_openai_streamed_chat_other_incomplete_with_app_limit_raises__DoD6(
    monkeypatch, has_details, details, description
):
    """DoD-6: with the app's limit set, a streamed ``response.incomplete`` event whose reason is any other
    value, has no reason, or has no incomplete details ends with ``LLMError`` as before; nothing after it is
    delivered (DEC-003-18, DEC-003-13)."""
    client = _client()
    event = _incomplete_event(details, with_details=has_details)
    _install_stream(monkeypatch, client, [_text("Before"), event, _text("After"), COMPLETED])
    recorder = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await client.chat(list(USER_MESSAGES), options={"max_tokens": LIMIT}, stream=True, on_delta=recorder)

    assert recorder.pieces == ["Before"]
    if description is not None:
        if description.startswith("{"):
            assert json.loads(excinfo.value.description) == details
        else:
            assert excinfo.value.description == description


# ---------------------------------------------------------------------------
# DoD-7 — app limit set: failed reply / failed event / error event / missing completion stay LLMError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_chat_failed_reply_with_app_limit_raises_with_reason__DoD7(monkeypatch):
    """DoD-7: with the app's limit set, a failed non-streamed reply still ends with ``LLMError`` carrying the
    server's message (DEC-003-12)."""
    reply = _reply(CUT_OUTPUT, status="failed", error={"code": "server_error", "message": "Engine stalled 64"})
    client = _client()
    _serve(monkeypatch, client, reply)

    with pytest.raises(LLMError) as excinfo:
        await client.chat(list(USER_MESSAGES), options={"max_tokens": LIMIT})

    assert excinfo.value.description == "Engine stalled 64"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "reason"),
    [
        pytest.param(FAILED_EVENT, "Model fell 83", id="failed-event"),
        pytest.param(ERROR_EVENT, "Slow down 29", id="error-event"),
    ],
)
async def test_openai_streamed_chat_failure_event_with_app_limit_raises__DoD7(monkeypatch, event, reason):
    """DoD-7: with the app's limit set, a failed event and an error event still end streamed ``chat`` with
    ``LLMError`` carrying the server's reason; nothing after is delivered (DEC-003-13)."""
    client = _client()
    _install_stream(monkeypatch, client, [_text("Before"), event, _text("After"), COMPLETED])
    recorder = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await client.chat(list(USER_MESSAGES), options={"max_tokens": LIMIT}, stream=True, on_delta=recorder)

    assert excinfo.value.description == reason
    assert recorder.pieces == ["Before"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payloads",
    [
        pytest.param([CREATED, _text("A"), _text("B")], id="ends-after-text"),
        pytest.param([_text("A"), "[DONE]"], id="done-sentinel"),
        pytest.param([], id="empty-stream"),
    ],
)
async def test_openai_streamed_chat_without_completion_with_app_limit_raises__DoD7(monkeypatch, payloads):
    """DoD-7: with the app's limit set, a stream that ends without a completion event still ends with
    ``LLMError``, also after a ``[DONE]`` sentinel (DEC-003-13)."""
    client = _client()
    _install_stream(monkeypatch, client, payloads)

    with pytest.raises(LLMError):
        await client.chat(list(USER_MESSAGES), options={"max_tokens": LIMIT}, stream=True, on_delta=_Recorder())


# ---------------------------------------------------------------------------
# DoD-8 — streamed chat: pieces before the matching incomplete event, nothing after, return == concatenation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("defaults", "options"), LIMIT_SOURCES)
async def test_openai_streamed_chat_cut_at_limit_returns_delivered_text__DoD8(monkeypatch, defaults, options):
    """DoD-8: streamed ``chat`` with the app's limit set delivers the text pieces before a matching incomplete
    event in order, nothing after it, and returns exactly their concatenation (US-062.AC-2/AC-3, DEC-003-19)."""
    client = _client(defaults)
    calls = _install_stream(
        monkeypatch,
        client,
        [CREATED, _text("Once "), _text("upon "), _text("a ti"), CUT_EVENT, _text("me"), COMPLETED],
    )
    recorder = _Recorder()

    result = await client.chat(list(USER_MESSAGES), options=options, stream=True, on_delta=recorder)

    assert recorder.pieces == ["Once ", "upon ", "a ti"]
    assert result == "Once upon a ti"
    assert result == "".join(recorder.pieces)
    assert len(calls) == 1 and calls[0]["path"] == "/responses"


@pytest.mark.asyncio
async def test_openai_streamed_chat_cut_at_limit_without_receiver_returns_text__DoD8(monkeypatch):
    """DoD-8: with no receiver, streamed ``chat`` cut at the app's limit returns the text before the event."""
    client = _client()
    _install_stream(monkeypatch, client, [_text("Half "), _text("told"), CUT_EVENT, _text(" tale")])

    assert await client.chat(list(USER_MESSAGES), options={"max_tokens": LIMIT}, stream=True) == "Half told"


# ---------------------------------------------------------------------------
# DoD-9 — open reasoning block closed by its own marker piece; streamed generate == one-message chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payloads", "joined"),
    [
        pytest.param([_reasoning("Let me "), _reasoning("think"), CUT_EVENT], "<think>Let me think</think>", id="only"),
        pytest.param(
            [_text("Answer: "), _reasoning("hmm"), CUT_EVENT, _text("late")],
            "Answer: <think>hmm</think>",
            id="after-text",
        ),
    ],
)
async def test_openai_streamed_cut_closes_open_reasoning_block__DoD9(monkeypatch, payloads, joined):
    """DoD-9: a reasoning block still open at a matching incomplete event is closed by a ``</think>`` piece of
    its own, and the return equals what the receiver got (DEC-000-20, DEC-003-19)."""
    client = _client()
    _install_stream(monkeypatch, client, payloads)
    recorder = _Recorder()

    result = await client.chat(list(USER_MESSAGES), options={"max_tokens": LIMIT}, stream=True, on_delta=recorder)

    assert recorder.pieces[-1] == "</think>"
    assert "".join(recorder.pieces) == joined
    assert result == "".join(recorder.pieces)


@pytest.mark.asyncio
async def test_openai_streamed_generate_cut_matches_one_message_chat__DoD9(monkeypatch):
    """DoD-9: streamed ``generate`` cut at the app's limit behaves as the one-message ``chat``: same request,
    same pieces, same return (DEC-000-8)."""
    payloads = [_reasoning("plan"), _text("Once upon"), CUT_EVENT, _text(" never")]
    options = {"max_tokens": LIMIT}

    gen_client = _client()
    gen_calls = _install_stream(monkeypatch, gen_client, payloads)
    gen_recorder = _Recorder()
    gen_result = await gen_client.generate(PROMPT, options=options, stream=True, on_delta=gen_recorder)

    chat_client = _client()
    chat_calls = _install_stream(monkeypatch, chat_client, payloads)
    chat_recorder = _Recorder()
    chat_result = await chat_client.chat(
        [{"role": "user", "content": PROMPT}], options=options, stream=True, on_delta=chat_recorder
    )

    assert gen_recorder.pieces == chat_recorder.pieces
    assert gen_result == chat_result == "".join(gen_recorder.pieces)
    assert gen_calls == chat_calls
    assert "".join(gen_recorder.pieces) == "<think>plan</think>Once upon"


# ---------------------------------------------------------------------------
# DoD-10 — reader and parser with the flag at its default keep the old rule
# ---------------------------------------------------------------------------


def test_openai_reply_reader_default_flag_cut_raises_with_reason__DoD10():
    """DoD-10: the reader called without ``output_limit_set`` treats a cut for the output limit as before:
    ``LLMError`` with the reason (DEC-003-17)."""
    with pytest.raises(LLMError) as excinfo:
        _responses_reply_text(_cut())

    assert excinfo.value.description == CUT_REASON


@pytest.mark.asyncio
async def test_openai_stream_parser_default_flag_cut_raises_with_reason__DoD10(monkeypatch):
    """DoD-10: the parser called without ``output_limit_set`` treats a matching incomplete event as before:
    ``LLMError`` with the reason, nothing after delivered (DEC-003-17)."""
    client = _client()
    _install_sse(monkeypatch, client, [_text("Before"), CUT_EVENT, _text("After"), COMPLETED])
    recorder = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await client._read_responses_stream(None, recorder)

    assert excinfo.value.description == CUT_REASON
    assert recorder.pieces == ["Before"]


@pytest.mark.asyncio
async def test_openai_stream_parser_flag_on_cut_ends_like_completed__DoD10(monkeypatch):
    """DoD-10 contrast (Interface intent): the parser called with ``output_limit_set=True`` ends at a matching
    incomplete event as at ``response.completed`` and returns what the receiver got (DEC-003-19)."""
    client = _client()
    _install_sse(monkeypatch, client, [_text("Before"), CUT_EVENT, _text("After"), COMPLETED])
    recorder = _Recorder()

    result = await client._read_responses_stream(None, recorder, output_limit_set=True)

    assert recorder.pieces == ["Before"]
    assert result == "Before"


# ---------------------------------------------------------------------------
# DoD-11 — tracing: a cut reply at the app's limit is recorded as the call's response
# ---------------------------------------------------------------------------


# Section titles of the trace record (docs/architecture/data-model.md § Trace record; as in test_llm_trace.py).
_HEADINGS = frozenset({"SYSTEM PROMPT", "MESSAGE HISTORY", "REQUEST", "RESPONSE", "CHUNK", "ERROR", "END TRACE"})


def _sections(text: str) -> list[tuple[str, list[str]]]:
    """Split a record into ``(title, stripped body lines)``; blank and pure rule lines are layout, skipped."""
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


def _only_record(folder: Path) -> str:
    """The text of the single trace record written under ``folder``."""
    logs = sorted(folder.rglob("*.log"))
    assert len(logs) == 1, logs
    return logs[0].read_text(encoding="utf-8")


def _traced_client(monkeypatch, folder: Path) -> OpenAIAPIClient:
    """A client traced into ``folder``; the tracer reads its switches at construction (DEC-003-16), so they
    are set first, and each client gets its own folder so the two records cannot be confused."""
    folder.mkdir()
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(folder))
    return _client()


@pytest.mark.asyncio
async def test_openai_traced_chat_cut_at_limit_recorded_as_response__DoD11(monkeypatch, tmp_path):
    """DoD-11: with tracing on, a non-streamed reply cut at the app's limit is returned and recorded as the
    call's response, the same as a completed reply with the same output, with no error section (DEC-000-48)."""
    completed_client = _traced_client(monkeypatch, tmp_path / "completed")
    _serve(monkeypatch, completed_client, _completed())
    assert await completed_client.chat(list(USER_MESSAGES), options={"max_tokens": LIMIT}) == CUT_TEXT

    cut_client = _traced_client(monkeypatch, tmp_path / "cut")
    _serve(monkeypatch, cut_client, _cut())
    assert await cut_client.chat(list(USER_MESSAGES), options={"max_tokens": LIMIT}) == CUT_TEXT

    cut_sections = _sections(_only_record(tmp_path / "cut"))
    completed_sections = _sections(_only_record(tmp_path / "completed"))
    cut_titles = [title for title, _ in cut_sections]
    assert "ERROR" not in cut_titles
    cut_responses = [body for title, body in cut_sections if title == "RESPONSE"]
    assert cut_responses == [body for title, body in completed_sections if title == "RESPONSE"]
    assert any(CUT_TEXT in " ".join(body) for body in cut_responses)


@pytest.mark.asyncio
async def test_openai_traced_streamed_chat_cut_at_limit_recorded_as_response__DoD11(monkeypatch, tmp_path):
    """DoD-11: with tracing on, a streamed reply cut at the app's limit returns what the receiver got and is
    recorded as the call's response, as the same stream ending ``response.completed`` is, with no error
    section (DEC-000-48, DEC-003-19)."""
    pieces = [_text("Once "), _text("upon")]

    completed_client = _traced_client(monkeypatch, tmp_path / "completed")
    _install_stream(monkeypatch, completed_client, [*pieces, COMPLETED])
    await completed_client.chat(list(USER_MESSAGES), options={"max_tokens": LIMIT}, stream=True, on_delta=_Recorder())

    cut_client = _traced_client(monkeypatch, tmp_path / "cut")
    _install_stream(monkeypatch, cut_client, [*pieces, CUT_EVENT, _text(" never")])
    recorder = _Recorder()
    result = await cut_client.chat(list(USER_MESSAGES), options={"max_tokens": LIMIT}, stream=True, on_delta=recorder)

    assert recorder.pieces == ["Once ", "upon"]
    assert result == "Once upon"
    cut_sections = _sections(_only_record(tmp_path / "cut"))
    completed_sections = _sections(_only_record(tmp_path / "completed"))
    assert "ERROR" not in [title for title, _ in cut_sections]
    cut_responses = [body for title, body in cut_sections if title == "RESPONSE"]
    assert cut_responses
    assert cut_responses == [body for title, body in completed_sections if title == "RESPONSE"]
