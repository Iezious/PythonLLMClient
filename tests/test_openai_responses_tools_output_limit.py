"""Unit tests for a tool round cut at the app's own output limit on the OpenAI Responses class
(DEC-arch-13 tool part, DEC-004-15..18, DEC-003-17..20).

When the app set the output limit (``max_tokens`` or its ``context_length`` alias, as a call option or a client
default), a ``chat_with_tools`` round that the server stops for the output limit (non-streamed: status
``incomplete`` with reason exactly ``max_output_tokens``; streamed: the matching ``response.incomplete`` event) is
the answer when the round holds no tool call, and ends the call with ``LLMError`` when it holds one. With no app
limit, and for every other early end, the round still raises. Covers the non-streamed tool-round reader
``_read_responses_tool_round`` and the tool-round stream parser ``OpenAIAPIClient._read_responses_tool_stream``
directly, and ``chat_with_tools`` streamed and not. The server is faked through per-round ``_request`` replies
or per-round ``_stream_post`` + ``_iter_sse_data`` event lists (DEC-004-14). Every expected value comes from the
step's Definition of done and the cited decisions.
"""

import copy
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest

from llm.exceptions import LLMError
from llm.openai import OpenAIAPIClient, _read_responses_tool_round

BASE_URL = "https://api.example.test/v1"
CREDENTIAL = "sk-test"
MODEL = "gpt-6-luna"

USER_MESSAGES = [{"role": "user", "content": "What is stored under key k-1? Tell the whole story."}]

LIMIT = 64
CUT_REASON = "max_output_tokens"

# Answer text split into parts: non-streamed as one output_text part each (concatenated with no separator,
# DEC-003-12), streamed as one text delta each, so both paths yield CUT_TEXT.
CUT_PARTS = ["Once upon a time, ", "there was a", " dragon who"]
CUT_TEXT = "".join(CUT_PARTS)

# A text delta placed after a round's end event: no parser may deliver it (DEC-004-9, DEC-004-17).
LATE = "NEVER-DELIVERED"

# The ways the app sets the output limit (DEC-003-17, DEC-004-15): (client defaults, call options).
LIMIT_SOURCES = [
    pytest.param(None, {"max_tokens": LIMIT}, id="call-max_tokens"),
    pytest.param({"max_tokens": LIMIT}, None, id="default-max_tokens"),
    pytest.param(None, {"context_length": LIMIT}, id="call-context_length"),
    pytest.param({"context_length": LIMIT}, None, id="default-context_length"),
]

# The ways the app sets no limit: never set, or a default cleared per call with None (DEC-003-17).
NO_LIMIT_SOURCES = [
    pytest.param(None, None, id="never-set"),
    pytest.param({"max_tokens": LIMIT}, {"max_tokens": None}, id="default-max_tokens-cleared"),
    pytest.param({"context_length": LIMIT}, {"context_length": None}, id="default-context_length-cleared"),
]

STREAM_MODES = [pytest.param(False, id="non-streamed"), pytest.param(True, id="streamed")]

APP_LIMIT = {"max_tokens": LIMIT}


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def _lookup_def() -> dict:
    """The app's nested definition of the single ``lookup`` tool (DEC-004-5)."""
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Fetch a stored record by its key.",
            "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
        },
    }


class _Tools:
    """The ``lookup`` tool, recording the key of every invocation in order, so a test sees which tools ran and
    how many times."""

    def __init__(self) -> None:
        self.keys: list[str] = []

    def lookup(self, key: str) -> str:
        self.keys.append(key)
        return f"rec-{key}"


# ---------------------------------------------------------------------------
# Stream events (DEC-004-9 vocabulary)
# ---------------------------------------------------------------------------


def _ev(type_: str, **fields: Any) -> str:
    """One SSE data payload: the JSON text of a Responses stream event."""
    return json.dumps({"type": type_, **fields})


def _response(status: str, **extra: Any) -> dict:
    """The ``response`` object of a lifecycle event."""
    return {"id": "resp_s", "status": status, **extra}


def _text_delta(delta: str) -> str:
    return _ev("response.output_text.delta", item_id="msg_1", output_index=1, content_index=0, delta=delta)


def _reasoning_delta(delta: str) -> str:
    return _ev("response.reasoning_text.delta", item_id="rs_1", output_index=0, content_index=0, delta=delta)


def _incomplete_event(details: Any = None, *, with_details: bool = True) -> str:
    """A ``response.incomplete`` event; ``with_details=False`` omits ``incomplete_details`` entirely."""
    response = _response("incomplete")
    if with_details:
        response["incomplete_details"] = details
    return _ev("response.incomplete", response=response)


def _function_call_item(item_id: str, call_id: str, arguments: str, status: str = "completed") -> dict:
    """A ``function_call`` output item for the ``lookup`` tool, arguments as JSON text."""
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": "lookup",
        "arguments": arguments,
        "status": status,
    }


def _added(index: int, item: dict) -> str:
    return _ev("response.output_item.added", output_index=index, item=item)


def _done(index: int, item: dict) -> str:
    return _ev("response.output_item.done", output_index=index, item=item)


def _args_delta(index: int, item_id: str, delta: str) -> str:
    return _ev("response.function_call_arguments.delta", item_id=item_id, output_index=index, delta=delta)


def _args_done(index: int, item_id: str, arguments: str) -> str:
    return _ev("response.function_call_arguments.done", item_id=item_id, output_index=index, arguments=arguments)


CREATED = _ev("response.created", response=_response("in_progress"))
COMPLETED_EVENT = _ev("response.completed", response=_response("completed"))
CUT_EVENT = _incomplete_event({"reason": CUT_REASON})
LATE_EVENT = _text_delta(LATE)
FAILED_EVENT = _ev(
    "response.failed", response=_response("failed", error={"code": "server_error", "message": "Engine stalled 64"})
)
ERROR_EVENT = _ev("error", code="rate_limit_exceeded", message="Slow down 29")


# ---------------------------------------------------------------------------
# Round builders: one round as a non-streamed reply and as the equivalent event list
# ---------------------------------------------------------------------------


class _Ending(NamedTuple):
    """How a round ends: the reply's status fields (non-streamed) and the events from the end on (streamed).
    Event endings are followed by a late text delta and a completion, so reading past the end would show."""

    reply_fields: dict[str, Any]
    events: list[str]


COMPLETED_END = _Ending({"status": "completed"}, [COMPLETED_EVENT, LATE_EVENT])
CUT_END = _Ending(
    {"status": "incomplete", "incomplete_details": {"reason": CUT_REASON}}, [CUT_EVENT, LATE_EVENT, COMPLETED_EVENT]
)
FAILED_END = _Ending(
    {"status": "failed", "error": {"code": "server_error", "message": "Engine stalled 64"}},
    [FAILED_EVENT, LATE_EVENT, COMPLETED_EVENT],
)


def _incomplete_end(details: Any, *, with_details: bool = True) -> _Ending:
    """An incomplete ending with the given details (omitted entirely when ``with_details`` is false)."""
    fields: dict[str, Any] = {"status": "incomplete"}
    if with_details:
        fields["incomplete_details"] = details
    return _Ending(fields, [_incomplete_event(details, with_details=with_details), LATE_EVENT, COMPLETED_EVENT])


class _Round(NamedTuple):
    """One canned round: the non-streamed reply and the streamed event list with the same output."""

    reply: dict[str, Any]
    events: list[str]


def _message_item(*parts: str) -> dict:
    """An assistant ``message`` output item with one ``output_text`` part per text."""
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": p, "annotations": []} for p in parts],
    }


def _reasoning_item(text: str) -> dict:
    """A ``reasoning`` output item; never part of the round text (DEC-003-12, DEC-004-8)."""
    return {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": text}]}


def _reply(output: list[dict], end: _Ending) -> dict:
    return {"id": "resp_1", "object": "response", "output": output, **copy.deepcopy(end.reply_fields)}


def _answer_round(text: list[str], *, reasoning: list[str] = (), end: _Ending = CUT_END) -> _Round:
    """A round with optional reasoning, then answer text, and no tool call."""
    output: list[dict] = []
    events = [CREATED]
    if reasoning:
        output.append(_reasoning_item("".join(reasoning)))
        events += [_reasoning_delta(chunk) for chunk in reasoning]
    if text:
        output.append(_message_item(*text))
        events += [_text_delta(chunk) for chunk in text]
    return _Round(_reply(output, end), events + end.events)


def _call_round(
    calls: list[tuple[str, str]], *, text: list[str] = (), reasoning: list[str] = (), end: _Ending = COMPLETED_END
) -> _Round:
    """A round with optional reasoning and text, then one ``lookup`` call per ``(call_id, arguments text)``.

    Streamed, each call is an added item, two argument fragments, an arguments-done and an item-done event."""
    output: list[dict] = []
    events = [CREATED]
    if reasoning:
        output.append(_reasoning_item("".join(reasoning)))
        events += [_reasoning_delta(chunk) for chunk in reasoning]
    if text:
        output.append(_message_item(*text))
        events += [_text_delta(chunk) for chunk in text]
    for n, (call_id, arguments) in enumerate(calls):
        index = 2 + n
        item_id = f"fc_{call_id}"
        half = len(arguments) // 2
        output.append(_function_call_item(item_id, call_id, arguments))
        events += [
            _added(index, _function_call_item(item_id, call_id, "", status="in_progress")),
            _args_delta(index, item_id, arguments[:half]),
            _args_delta(index, item_id, arguments[half:]),
            _args_done(index, item_id, arguments),
            _done(index, _function_call_item(item_id, call_id, arguments)),
        ]
    return _Round(_reply(output, end), events + end.events)


def _stream_only(events: list[str]) -> _Round:
    """A round used on the streamed path only (event orders with no non-streamed counterpart)."""
    return _Round({}, events)


FIRST_CALL = ("call_1", '{"key": "k-1"}')


# ---------------------------------------------------------------------------
# Client, receiver and seams (DEC-004-14)
# ---------------------------------------------------------------------------


def _client(defaults: dict | None = None) -> OpenAIAPIClient:
    """A Responses client built directly with the app-provided test credential (tests/AGENTS.md)."""
    return OpenAIAPIClient(model=MODEL, base_url=BASE_URL, bearer_token=CREDENTIAL, default_options=defaults)


class _Recorder:
    """Async receiver that records every piece it is given, in order."""

    def __init__(self) -> None:
        self.pieces: list[str] = []

    async def __call__(self, piece: str) -> None:
        self.pieces.append(piece)


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch):
    """Clear the trace switches so tests start untraced whatever the developer's environment; the DoD-11
    tests turn them on with ``monkeypatch.setenv`` before building their clients."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _install_rounds(monkeypatch, client: OpenAIAPIClient, rounds: list[_Round], *, stream: bool) -> list[dict]:
    """Serve one canned round per request and record each request (path and deep-copied body).

    Non-streamed: ``_request`` answers each round's reply. Streamed: ``_stream_post`` opens one round and
    ``_iter_sse_data`` yields that round's events. The other path fails the test if used, and a request beyond
    the canned rounds fails it too, so "no further request is sent" is observable as the record count.
    """
    pending = [copy.deepcopy(r) for r in rounds]
    sent: list[dict] = []

    def _next_round() -> _Round:
        if not pending:
            raise AssertionError("more rounds were requested than canned rounds exist")
        return pending.pop(0)

    if stream:
        opened: list[list[str]] = []

        @asynccontextmanager
        async def fake_stream_post(path, payload):
            sent.append({"path": path, "json": copy.deepcopy(payload)})
            opened.append(_next_round().events)
            yield SimpleNamespace(status=200)

        async def fake_iter_sse_data(_response):
            for payload in opened.pop(0):
                yield payload

        async def no_request(*_args, **_kw):
            raise AssertionError("a streamed tool round must not use the non-streamed request path")

        monkeypatch.setattr(client, "_stream_post", fake_stream_post)
        monkeypatch.setattr(client, "_iter_sse_data", fake_iter_sse_data)
        monkeypatch.setattr(client, "_request", no_request)
    else:

        async def fake_request(method, path, **kw):
            sent.append({"method": method, "path": path, "json": copy.deepcopy(kw.get("json"))})
            return _next_round().reply

        def no_stream_post(*_args, **_kw):
            raise AssertionError("a non-streamed tool round must not open a stream")

        monkeypatch.setattr(client, "_request", fake_request)
        monkeypatch.setattr(client, "_stream_post", no_stream_post)
    return sent


def _install_sse(monkeypatch, client: OpenAIAPIClient, events: list[str]) -> None:
    """Replace ``_iter_sse_data`` with a fake yielding ``events``, for direct parser calls (DEC-004-14)."""

    async def fake_iter_sse_data(_response):
        for payload in events:
            yield payload

    monkeypatch.setattr(client, "_iter_sse_data", fake_iter_sse_data)


async def _run(
    client: OpenAIAPIClient,
    tools: _Tools,
    *,
    stream: bool,
    options: dict | None = None,
    on_delta: _Recorder | None = None,
    max_loops: int = 5,
) -> str:
    """Run ``chat_with_tools`` over fresh copies of the caller data with the ``lookup`` tool."""
    return await client.chat_with_tools(
        copy.deepcopy(USER_MESSAGES),
        tools_definitions=[_lookup_def()],
        tools={"lookup": tools.lookup},
        max_loops=max_loops,
        options=options,
        stream=stream,
        on_delta=on_delta if stream else None,
    )


def _carries(exc: LLMError, reason: str) -> bool:
    """Whether the error carries the server's reason, in its text or as its description."""
    return reason in str(exc) or getattr(exc, "description", None) == reason


def _no_markers(text: str) -> bool:
    return "<think>" not in text and "</think>" not in text


# ---------------------------------------------------------------------------
# DoD-1 — non-streamed, call option: a cut first round without calls is the answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_cut_first_round_without_call_returns_text_as_completed__DoD1(monkeypatch):
    """DoD-1: with ``max_tokens`` as a call option, a first round cut at the app's limit holding answer text and
    no call makes ``chat_with_tools`` return that text, the same as a completed round with the same output; one
    request only and no tool runs (US-062.AC-4, DEC-004-16)."""
    completed_client = _client()
    _install_rounds(
        monkeypatch, completed_client, [_answer_round(CUT_PARTS, reasoning=["hidden"], end=COMPLETED_END)],
        stream=False,
    )
    completed = await _run(completed_client, _Tools(), stream=False, options=APP_LIMIT)

    client = _client()
    sent = _install_rounds(monkeypatch, client, [_answer_round(CUT_PARTS, reasoning=["hidden"])], stream=False)
    tools = _Tools()

    result = await _run(client, tools, stream=False, options=APP_LIMIT)

    assert result == CUT_TEXT
    assert result == completed
    assert len(sent) == 1
    assert tools.keys == []


def test_openai_tool_round_reader_flag_on_cut_without_call_returns_text__DoD1():
    """DoD-1 (Interface intent): the reader called with ``output_limit_set=True`` on a cut reply with no
    ``function_call`` item returns the round text, extracted as from a completed reply, and no tool calls."""
    round_ = _read_responses_tool_round(_answer_round(CUT_PARTS, reasoning=["hidden"]).reply, output_limit_set=True)

    assert round_.text == CUT_TEXT
    assert round_.tool_calls == []


# ---------------------------------------------------------------------------
# DoD-2 — non-streamed: a cut second round after a tool round is the answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_cut_second_round_after_tool_round_returns_its_text__DoD2(monkeypatch):
    """DoD-2: after a first round whose tool ran, a second round cut at the app's limit with no call returns
    the second round's text; the first round's tool ran exactly once (US-062.AC-4, DEC-004-16, DEC-004-18)."""
    client = _client()
    rounds = [_call_round([FIRST_CALL]), _answer_round(["Record k-1 ", "holds rec-k-1, and"])]
    sent = _install_rounds(monkeypatch, client, rounds, stream=False)
    tools = _Tools()

    result = await _run(client, tools, stream=False, options=APP_LIMIT)

    assert result == "Record k-1 holds rec-k-1, and"
    assert tools.keys == ["k-1"]
    assert len(sent) == 2


# ---------------------------------------------------------------------------
# DoD-3 — DoD-1 for every limit source, streamed and not
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
@pytest.mark.parametrize(("defaults", "options"), LIMIT_SOURCES)
async def test_openai_cut_answer_round_returned_for_every_limit_source__DoD3(monkeypatch, defaults, options, stream):
    """DoD-3: a first round cut at the app's limit with answer text and no call is returned, streamed and not,
    whether the limit is a call option or a client default, under ``max_tokens`` or the ``context_length``
    alias; one request only, no tool runs (US-062.AC-4, DEC-004-15, DEC-003-17)."""
    client = _client(defaults)
    sent = _install_rounds(monkeypatch, client, [_answer_round(CUT_PARTS, reasoning=["hidden"])], stream=stream)
    tools = _Tools()

    result = await _run(client, tools, stream=stream, options=options, on_delta=_Recorder())

    assert result == CUT_TEXT
    assert len(sent) == 1
    assert tools.keys == []


# ---------------------------------------------------------------------------
# DoD-4 — non-streamed: a cut round holding tool calls ends with LLMError
# ---------------------------------------------------------------------------

# One complete call, or a complete call followed by one whose arguments were cut off.
CUT_CALLS = [
    pytest.param([("call_2", '{"key": "k-2"}')], id="one-call"),
    pytest.param([("call_2", '{"key": "k-2"}'), ("call_3", '{"key": "k-')], id="two-calls"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("round_no", [1, 2], ids=["first-round", "later-round"])
@pytest.mark.parametrize("calls", CUT_CALLS)
async def test_openai_cut_round_with_calls_raises_and_runs_none__DoD4(monkeypatch, calls, round_no):
    """DoD-4: with the app's limit set, a non-streamed round cut at the limit that holds tool calls ends the
    call with ``LLMError`` carrying the server's reason; none of its tools runs and no further request is sent;
    in a later round, the earlier round's tool ran exactly once (US-062.AC-4, DEC-004-16, DEC-004-18)."""
    cut = _call_round(calls, text=["Let me check."], end=CUT_END)
    rounds = [cut] if round_no == 1 else [_call_round([FIRST_CALL]), cut]
    client = _client()
    sent = _install_rounds(monkeypatch, client, rounds, stream=False)
    tools = _Tools()

    with pytest.raises(LLMError) as excinfo:
        await _run(client, tools, stream=False, options=APP_LIMIT)

    assert _carries(excinfo.value, CUT_REASON)
    assert tools.keys == ([] if round_no == 1 else ["k-1"])
    assert len(sent) == round_no


@pytest.mark.parametrize("calls", CUT_CALLS)
def test_openai_tool_round_reader_flag_on_cut_with_calls_raises__DoD4(calls):
    """DoD-4 (Interface intent): the reader called with ``output_limit_set=True`` on a cut reply holding
    ``function_call`` items raises ``LLMError`` carrying the server's reason (DEC-004-16)."""
    reply = _call_round(calls, text=["Let me check."], end=CUT_END).reply

    with pytest.raises(LLMError) as excinfo:
        _read_responses_tool_round(reply, output_limit_set=True)

    assert _carries(excinfo.value, CUT_REASON)


# ---------------------------------------------------------------------------
# DoD-5 — streamed: a cut round with no started call ends like completed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_streamed_cut_first_round_returns_joined_text_deltas__DoD5(monkeypatch):
    """DoD-5: streamed, with the app's limit set, a first round cut with no call started returns its text
    deltas joined; the receiver got every piece in order and nothing after the event; one request, no tool
    runs (US-062.AC-4, DEC-004-17)."""
    client = _client()
    sent = _install_rounds(monkeypatch, client, [_answer_round(["Once ", "upon"])], stream=True)
    tools = _Tools()
    recorder = _Recorder()

    result = await _run(client, tools, stream=True, options=APP_LIMIT, on_delta=recorder)

    assert result == "Once upon"
    assert recorder.pieces == ["Once ", "upon"]
    assert len(sent) == 1
    assert tools.keys == []


@pytest.mark.asyncio
async def test_openai_streamed_cut_later_round_returns_text_without_reasoning__DoD5(monkeypatch):
    """DoD-5: streamed, after a tool round, a later round cut with no call started returns that round's text
    deltas joined with no markers and no reasoning; the receiver got every piece of every round in order, each
    round's reasoning as its own closed block, and nothing after the event (US-062.AC-4, US-027.AC-1,
    DEC-004-17, DEC-000-20)."""
    client = _client()
    rounds = [
        _call_round([FIRST_CALL], reasoning=["Pl", "an"]),
        _answer_round(["Fi", "nal"], reasoning=["Chk"]),
    ]
    sent = _install_rounds(monkeypatch, client, rounds, stream=True)
    tools = _Tools()
    recorder = _Recorder()

    result = await _run(client, tools, stream=True, options=APP_LIMIT, on_delta=recorder)

    assert result == "Final"
    assert _no_markers(result)
    assert recorder.pieces == ["<think>Pl", "an", "</think>", "<think>Chk", "</think>Fi", "nal"]
    assert tools.keys == ["k-1"]
    assert len(sent) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("events", "pieces", "text"),
    [
        pytest.param(
            [CREATED, _text_delta("Ans"), _text_delta("wer"), _reasoning_delta("hm"), _reasoning_delta("m"),
             CUT_EVENT, LATE_EVENT, COMPLETED_EVENT],
            ["Ans", "wer", "<think>hm", "m", "</think>"],
            "Answer",
            id="text-then-open-reasoning",
        ),
        pytest.param(
            [CREATED, _reasoning_delta("hmm"), CUT_EVENT, LATE_EVENT],
            ["<think>hmm", "</think>"],
            "",
            id="only-open-reasoning",
        ),
    ],
)
async def test_openai_streamed_cut_closes_open_reasoning_block__DoD5(monkeypatch, events, pieces, text):
    """DoD-5: a reasoning block still open at a matching incomplete event is closed by a ``</think>`` piece of
    its own; nothing after the event is delivered and the returned answer is the text deltas only
    (US-027.AC-1, DEC-004-17, DEC-000-20)."""
    client = _client()
    _install_rounds(monkeypatch, client, [_stream_only(events)], stream=True)
    recorder = _Recorder()

    result = await _run(client, _Tools(), stream=True, options=APP_LIMIT, on_delta=recorder)

    assert recorder.pieces == pieces
    assert result == text


@pytest.mark.asyncio
async def test_openai_tool_stream_parser_flag_on_cut_without_call_ends_like_completed__DoD5(monkeypatch):
    """DoD-5 (Interface intent): the parser called with ``output_limit_set=True`` ends at a matching incomplete
    event with no call started as at completion: round text is the text deltas, no tool calls, pieces up to the
    event only (DEC-004-17)."""
    client = _client()
    _install_sse(monkeypatch, client, _answer_round(["Fi", "nal"], reasoning=["Chk"]).events)
    recorder = _Recorder()

    round_ = await client._read_responses_tool_stream(None, recorder, output_limit_set=True)

    assert round_.text == "Final"
    assert round_.tool_calls == []
    assert recorder.pieces == ["<think>Chk", "</think>Fi", "nal"]


# ---------------------------------------------------------------------------
# DoD-6 — streamed: a cut round with a started call ends with LLMError
# ---------------------------------------------------------------------------

_CUT_ITEM_ID = "fc_cut"

# Every way a call counts as started (DEC-004-17): an added call item, with or without a fragment, an
# arguments fragment alone (no added item), or a done call item alone.
STARTED_CALLS = [
    pytest.param([_added(2, _function_call_item(_CUT_ITEM_ID, "call_9", "", status="in_progress"))], id="added"),
    pytest.param(
        [_added(2, _function_call_item(_CUT_ITEM_ID, "call_9", "", status="in_progress")),
         _args_delta(2, _CUT_ITEM_ID, '{"key": "k-')],
        id="added-with-fragment",
    ),
    pytest.param([_args_delta(2, _CUT_ITEM_ID, '{"key": "k-')], id="fragment-alone"),
    pytest.param([_done(2, _function_call_item(_CUT_ITEM_ID, "call_9", '{"key": "k-9"}'))], id="done-item"),
]


def _started_call_events(started: list[str]) -> list[str]:
    """A round with text, then a reasoning block left open, then a started call, then the matching cut event
    followed by late pieces. Expected pieces: ``["Before", "<think>Plan"]`` and nothing more."""
    return [CREATED, _text_delta("Before"), _reasoning_delta("Plan"), *started, CUT_EVENT, LATE_EVENT,
            _text_delta("After"), COMPLETED_EVENT]


PIECES_BEFORE_CUT = ["Before", "<think>Plan"]


@pytest.mark.asyncio
@pytest.mark.parametrize("started", STARTED_CALLS)
async def test_openai_tool_stream_parser_cut_with_started_call_raises_and_stops__DoD6(monkeypatch, started):
    """DoD-6: the parser with the flag on raises ``LLMError`` carrying the server's reason at a matching
    incomplete event once a call has started, and delivers no further piece, not even a closing marker
    (US-062.AC-4, DEC-004-17)."""
    client = _client()
    _install_sse(monkeypatch, client, _started_call_events(started))
    recorder = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await client._read_responses_tool_stream(None, recorder, output_limit_set=True)

    assert _carries(excinfo.value, CUT_REASON)
    assert recorder.pieces == PIECES_BEFORE_CUT


@pytest.mark.asyncio
@pytest.mark.parametrize("round_no", [1, 2], ids=["first-round", "later-round"])
@pytest.mark.parametrize("started", STARTED_CALLS)
async def test_openai_streamed_cut_round_with_started_call_raises_runs_none__DoD6(monkeypatch, started, round_no):
    """DoD-6: streamed, with the app's limit set, a round cut after a call started ends the call with
    ``LLMError`` carrying the server's reason; no piece after the event, no tool of that round runs, no further
    request; in a later round the earlier round's tool ran exactly once (US-062.AC-4, DEC-004-17, DEC-004-18)."""
    cut = _stream_only(_started_call_events(started))
    rounds = [cut] if round_no == 1 else [_call_round([FIRST_CALL]), cut]
    client = _client()
    sent = _install_rounds(monkeypatch, client, rounds, stream=True)
    tools = _Tools()
    recorder = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await _run(client, tools, stream=True, options=APP_LIMIT, on_delta=recorder)

    assert _carries(excinfo.value, CUT_REASON)
    # The first round of the later-round case has no text or reasoning, so it delivers no piece.
    assert recorder.pieces == PIECES_BEFORE_CUT
    assert tools.keys == ([] if round_no == 1 else ["k-1"])
    assert len(sent) == round_no


# ---------------------------------------------------------------------------
# DoD-7 — no app limit: a cut round stays LLMError, streamed and not, with and without calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("with_calls", [False, True], ids=["no-calls", "with-calls"])
@pytest.mark.parametrize("stream", STREAM_MODES)
@pytest.mark.parametrize(("defaults", "options"), NO_LIMIT_SOURCES)
async def test_openai_cut_round_without_app_limit_raises_with_reason__DoD7(
    monkeypatch, defaults, options, stream, with_calls
):
    """DoD-7: with no app limit (never set, or a default cleared per call with ``None``), a round cut for the
    output limit ends with ``LLMError`` carrying the server's reason, streamed and not, with and without tool
    calls; no tool of that round runs (US-062.AC-6, DEC-004-15, DEC-003-17)."""
    if with_calls:
        cut = _call_round([("call_2", '{"key": "k-2"}')], text=["Checking."], end=CUT_END)
    else:
        cut = _answer_round(CUT_PARTS)
    client = _client(defaults)
    sent = _install_rounds(monkeypatch, client, [cut], stream=stream)
    tools = _Tools()

    with pytest.raises(LLMError) as excinfo:
        await _run(client, tools, stream=stream, options=options, on_delta=_Recorder())

    assert _carries(excinfo.value, CUT_REASON)
    assert tools.keys == []
    assert len(sent) == 1


# ---------------------------------------------------------------------------
# DoD-8 — app limit set: every other early end stays LLMError, streamed and not
# ---------------------------------------------------------------------------

# (ending, reason the error must carry, or None when only LLMError is asserted).
_OTHER_INCOMPLETE = [
    pytest.param(_incomplete_end({"reason": "content_filter"}), "content_filter", id="content-filter"),
    pytest.param(_incomplete_end({"reason": "MAX_OUTPUT_TOKENS"}), "MAX_OUTPUT_TOKENS", id="different-case"),
    pytest.param(_incomplete_end({"reason": "max_output_tokens_soft"}), "max_output_tokens_soft", id="prefix"),
    pytest.param(_incomplete_end({"reason": 7}), None, id="non-string-reason"),
    pytest.param(_incomplete_end({"x": "cut"}), None, id="no-reason"),
    pytest.param(_incomplete_end(None), None, id="null-details"),
    pytest.param(_incomplete_end(None, with_details=False), None, id="missing-details"),
]

NON_STREAMED_OTHER_ENDS = [*_OTHER_INCOMPLETE, pytest.param(FAILED_END, "Engine stalled 64", id="failed-reply")]

STREAMED_OTHER_ENDS = [
    *_OTHER_INCOMPLETE,
    pytest.param(FAILED_END, "Engine stalled 64", id="failed-event"),
    pytest.param(_Ending({}, [ERROR_EVENT, LATE_EVENT, COMPLETED_EVENT]), "Slow down 29", id="error-event"),
    pytest.param(_Ending({}, []), None, id="no-completion-event"),
    pytest.param(_Ending({}, ["[DONE]"]), None, id="done-sentinel-only"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("ending", "reason"), NON_STREAMED_OTHER_ENDS)
async def test_openai_other_early_end_with_app_limit_raises__DoD8(monkeypatch, ending, reason):
    """DoD-8: non-streamed, with the app's limit set, an incomplete round with any other reason, no reason or no
    incomplete details, and a failed reply, still end with ``LLMError`` (US-024.AC-2, DEC-003-18, DEC-004-8)."""
    client = _client()
    sent = _install_rounds(monkeypatch, client, [_answer_round(["Partial ", "answer"], end=ending)], stream=False)

    with pytest.raises(LLMError) as excinfo:
        await _run(client, _Tools(), stream=False, options=APP_LIMIT)

    if reason is not None:
        assert _carries(excinfo.value, reason)
    assert len(sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("ending", "reason"), STREAMED_OTHER_ENDS)
async def test_openai_streamed_other_early_end_with_app_limit_raises__DoD8(monkeypatch, ending, reason):
    """DoD-8: streamed, with the app's limit set, an incomplete event with any other reason, no reason or no
    incomplete details, a failed event, an error event, and a stream ending without a completion event still
    end with ``LLMError``; nothing after a failure event is delivered (US-024.AC-2, DEC-003-18, DEC-004-9)."""
    client = _client()
    sent = _install_rounds(monkeypatch, client, [_answer_round(["Partial ", "answer"], end=ending)], stream=True)
    recorder = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await _run(client, _Tools(), stream=True, options=APP_LIMIT, on_delta=recorder)

    if reason is not None:
        assert _carries(excinfo.value, reason)
    assert recorder.pieces == ["Partial ", "answer"]
    assert len(sent) == 1


@pytest.mark.parametrize(("ending", "reason"), NON_STREAMED_OTHER_ENDS)
def test_openai_tool_round_reader_flag_on_other_early_end_raises__DoD8(ending, reason):
    """DoD-8: the reader with the flag on still raises ``LLMError`` for every other early end (DEC-003-18)."""
    with pytest.raises(LLMError) as excinfo:
        _read_responses_tool_round(_answer_round(["Partial ", "answer"], end=ending).reply, output_limit_set=True)

    if reason is not None:
        assert _carries(excinfo.value, reason)


# ---------------------------------------------------------------------------
# DoD-9 — reader and parser at the flag's default keep the old rule
# ---------------------------------------------------------------------------

DEFAULT_FLAG_ROUNDS = [
    pytest.param(_answer_round(CUT_PARTS), id="no-calls"),
    pytest.param(_call_round([("call_2", '{"key": "k-2"}')], text=["Checking."], end=CUT_END), id="with-calls"),
]


@pytest.mark.parametrize("round_", DEFAULT_FLAG_ROUNDS)
def test_openai_tool_round_reader_default_flag_cut_raises_with_reason__DoD9(round_):
    """DoD-9: the reader called without ``output_limit_set`` treats a cut round as before: ``LLMError`` with the
    reason (DEC-004-15)."""
    with pytest.raises(LLMError) as excinfo:
        _read_responses_tool_round(copy.deepcopy(round_.reply))

    assert _carries(excinfo.value, CUT_REASON)


@pytest.mark.asyncio
@pytest.mark.parametrize("round_", DEFAULT_FLAG_ROUNDS)
async def test_openai_tool_stream_parser_default_flag_cut_raises_with_reason__DoD9(monkeypatch, round_):
    """DoD-9: the parser called without ``output_limit_set`` treats a matching incomplete event as before:
    ``LLMError`` with the reason, and the late piece after the event is not delivered (DEC-004-15)."""
    client = _client()
    _install_sse(monkeypatch, client, list(round_.events))
    recorder = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await client._read_responses_tool_stream(None, recorder)

    assert _carries(excinfo.value, CUT_REASON)
    assert LATE not in recorder.pieces


# ---------------------------------------------------------------------------
# DoD-10 — max_loops 1: a cut answer round is returned, no round-limit error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_cut_answer_round_on_last_allowed_round_returns_text__DoD10(monkeypatch, stream):
    """DoD-10: with ``max_loops`` 1 and the app's limit set, a first round cut at the limit with no call returns
    its text and raises no round-limit error (DEC-004-18, DEC-002-2)."""
    client = _client()
    sent = _install_rounds(monkeypatch, client, [_answer_round(CUT_PARTS)], stream=stream)

    result = await _run(client, _Tools(), stream=stream, options=APP_LIMIT, on_delta=_Recorder(), max_loops=1)

    assert result == CUT_TEXT
    assert len(sent) == 1


# ---------------------------------------------------------------------------
# DoD-11 — tracing: a cut answer round is recorded as the call's response
# ---------------------------------------------------------------------------

# Section titles of the trace record (docs/architecture/data-model.md § Trace record); tool sections carry the
# tool name after the title.
_HEADINGS = frozenset({"SYSTEM PROMPT", "MESSAGE HISTORY", "REQUEST", "RESPONSE", "CHUNK", "ERROR", "END TRACE"})
_HEADING_PREFIXES = ("TOOL CALL:", "TOOL RESULT:")


def _sections(text: str) -> list[tuple[str, list[str]]]:
    """Split a record into ``(title, stripped body lines)``; blank and pure rule lines are layout, skipped."""
    sections: list[tuple[str, list[str]]] = [("HEADER", [])]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) <= {"=", "-"}:
            continue
        if stripped in _HEADINGS or stripped.startswith(_HEADING_PREFIXES):
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
    """A client traced into ``folder``; the tracer reads its switches when the client is built, so they are set
    first, and each client gets its own folder so the two records cannot be confused."""
    folder.mkdir()
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(folder))
    return _client()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_traced_cut_answer_round_recorded_as_response__DoD11(monkeypatch, tmp_path, stream):
    """DoD-11: with tracing on, a cut answer round is returned and recorded as the call's response, the same as
    a completed answer round with the same output, with no error section (DEC-004-18, DEC-004-10, DEC-000-48)."""
    completed_client = _traced_client(monkeypatch, tmp_path / "completed")
    _install_rounds(monkeypatch, completed_client, [_answer_round(CUT_PARTS, end=COMPLETED_END)], stream=stream)
    completed = await _run(completed_client, _Tools(), stream=stream, options=APP_LIMIT, on_delta=_Recorder())

    cut_client = _traced_client(monkeypatch, tmp_path / "cut")
    _install_rounds(monkeypatch, cut_client, [_answer_round(CUT_PARTS)], stream=stream)
    result = await _run(cut_client, _Tools(), stream=stream, options=APP_LIMIT, on_delta=_Recorder())

    assert result == completed == CUT_TEXT
    cut_sections = _sections(_only_record(tmp_path / "cut"))
    completed_sections = _sections(_only_record(tmp_path / "completed"))
    assert "ERROR" not in [title for title, _ in cut_sections]
    cut_responses = [body for title, body in cut_sections if title == "RESPONSE"]
    assert cut_responses
    assert cut_responses == [body for title, body in completed_sections if title == "RESPONSE"]
    assert any(CUT_TEXT in " ".join(body) for body in cut_responses)
