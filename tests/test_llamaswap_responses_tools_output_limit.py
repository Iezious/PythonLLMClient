"""Unit tests for llama-swap Responses tool conversations: a round cut at the app's own output limit.

Feature 004.responses-tools, step 006 (DEC-arch-13 tool part; DEC-004-15..18, DEC-003-17..20). Covers
``LlamaSwapAPIClient.chat_with_tools`` (streamed and not), the module-level tool-round reader
``_responses_tool_round`` and the tool-round stream parser ``_stream_responses_tool_round``: with the app's
limit set (``max_tokens`` or the ``context_length`` alias, call option or client default), a round whose
incomplete reason is exactly ``max_output_tokens`` is the answer when it holds no tool call (streamed: no call
started) and ends the call with ``LLMError`` when it holds one. With no app limit, for every other early end,
and for the reader / parser called without the flag, the round stays ``LLMError``.

The server is faked through the DEC-004-14 seams: a per-round ``_request`` queue for non-streamed rounds, and a
per-round ``_iter_sse_data`` queue behind a fake ``_stream_post`` for streamed rounds. Each test names its DoD id.
DoD-12 (existing suite unchanged) is covered by running the existing suite, not here.
"""

import copy
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from llm.exceptions import LLMError
from llm.llama_swap import LlamaSwapAPIClient, _responses_tool_round

_BASE_URL = "http://localhost:8080/v1"
_MODEL = "some-model"

_OPEN = "<think>"
_CLOSE = "</think>"

_MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "Look up the record alpha-7 and tell me what it says."}]

# The one reason that counts as a cut at the output limit (DEC-003-18).
_CUT_REASON = "max_output_tokens"

# Every way the app sets the output limit (DEC-003-17, DEC-004-15): (client default options, call options).
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

_STREAM_MODES = [pytest.param(False, id="non-streamed"), pytest.param(True, id="streamed")]
_ROUND_POSITIONS = [pytest.param(0, id="first-round"), pytest.param(1, id="later-round")]

# The full argument text of the earlier round's fetch_record call, and the tool-log entry it produces.
_FULL_ARGS = '{"key": "alpha-7"}'
_EARLIER_RUN = ("fetch_record", {"key": "alpha-7"})


# ---------------------------------------------------------------------------
# Item, reply and event builders (Responses wire vocabulary, llama.cpp shapes)
# ---------------------------------------------------------------------------


def _message_item(item_id: str, *texts: str | None) -> dict[str, Any]:
    """A finished assistant message item with one ``output_text`` part per text."""
    return {
        "type": "message",
        "id": item_id,
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": t, "annotations": []} for t in texts],
    }


def _reasoning_item(item_id: str, text: str) -> dict[str, Any]:
    """A finished llama.cpp reasoning item (plain reasoning text, empty summary and encrypted content)."""
    return {
        "type": "reasoning",
        "id": item_id,
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
        "encrypted_content": "",
    }


def _call_item(item_id: str, call_id: str, name: str, arguments: str) -> dict[str, Any]:
    """A finished function-call item with its full arguments."""
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def _reply(*items: dict[str, Any], status: str = "completed", **extra: Any) -> dict[str, Any]:
    """A non-streamed Responses reply with the given output items and status."""
    return {"id": "resp_1", "object": "response", "status": status, "output": list(items), **extra}


def _cut_reply(*items: dict[str, Any]) -> dict[str, Any]:
    """A non-streamed round reply cut at the output limit: status incomplete, reason ``max_output_tokens``."""
    return _reply(*items, status="incomplete", incomplete_details={"reason": _CUT_REASON})


def _ev(event_type: str, **fields: Any) -> str:
    """One stream payload string: a JSON event object with ``type`` and the given fields."""
    return json.dumps({"type": event_type, **fields})


_CREATED = _ev("response.created", response={"id": "resp_1", "status": "in_progress", "output": []})


def _completed(*output: dict[str, Any]) -> str:
    """The completed event, carrying the round's output."""
    return _ev("response.completed", response={"id": "resp_1", "status": "completed", "output": list(output)})


def _incomplete_event(details: Any = None, *, with_details: bool = True) -> str:
    """A ``response.incomplete`` event; ``with_details=False`` omits ``incomplete_details`` entirely."""
    response: dict[str, Any] = {"id": "resp_1", "status": "incomplete", "output": []}
    if with_details:
        response["incomplete_details"] = details
    return _ev("response.incomplete", response=response)


_CUT_EVENT = _incomplete_event({"reason": _CUT_REASON})
_LATE_FAILED = _ev("response.failed", response={"status": "failed", "error": {"message": "late-failure-q1"}})


def _text_delta(delta: str, index: int = 1, item_id: str = "msg_1") -> str:
    """An output-text delta."""
    return _ev("response.output_text.delta", item_id=item_id, output_index=index, content_index=0, delta=delta)


def _reasoning_delta(delta: str, index: int = 0, item_id: str = "rs_1") -> str:
    """A reasoning-text delta."""
    return _ev("response.reasoning_text.delta", item_id=item_id, output_index=index, content_index=0, delta=delta)


def _item_done(index: int, item: dict[str, Any]) -> str:
    """An ``output_item.done`` event for any item."""
    return _ev("response.output_item.done", output_index=index, item=item)


def _call_added(index: int, item_id: str, call_id: str, name: str) -> str:
    """The function-call ``output_item.added`` event: name and call id present, arguments empty."""
    item = {"type": "function_call", "id": item_id, "call_id": call_id, "name": name, "arguments": ""}
    return _ev("response.output_item.added", output_index=index, item={**item, "status": "in_progress"})


def _args_delta(delta: str, index: int, item_id: str) -> str:
    """A function-call arguments delta."""
    return _ev("response.function_call_arguments.delta", item_id=item_id, output_index=index, delta=delta)


def _call_round_events(item_id: str, call_id: str, name: str, arguments: str) -> list[str]:
    """A complete streamed tool round holding one call at position 1 (no text, no reasoning)."""
    call = _call_item(item_id, call_id, name, arguments)
    return [
        _CREATED,
        _call_added(1, item_id, call_id, name),
        _args_delta(arguments, 1, item_id),
        _item_done(1, call),
        _completed(call),
    ]


# The earlier tool round used by later-round tests, streamed and non-streamed: one fetch_record call.
_EARLIER_CALL = _call_item("fc_1", "call_1", "fetch_record", _FULL_ARGS)
_EARLIER_REPLY = _reply(_EARLIER_CALL)


def _earlier_stream_round() -> list[str]:
    """The streamed form of the earlier tool round (delivers no piece: it holds no text or reasoning)."""
    return _call_round_events("fc_1", "call_1", "fetch_record", _FULL_ARGS)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _tool_def(name: str, param: str) -> dict[str, Any]:
    """An app tool definition in the nested shape with one required text parameter."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Look something up with {name}.",
            "parameters": {"type": "object", "properties": {param: {"type": "string"}}, "required": [param]},
        },
    }


_TOOL_DEFS = [_tool_def("fetch_record", "key"), _tool_def("query_index", "term")]


class _Tools:
    """Two neutral tools that record every invocation, in order, with their arguments."""

    def __init__(self) -> None:
        self.log: list[tuple[str, dict[str, Any]]] = []

    def fetch_record(self, key: str) -> str:
        self.log.append(("fetch_record", {"key": key}))
        return f"record:{key}"

    def query_index(self, term: str) -> str:
        self.log.append(("query_index", {"term": term}))
        return f"index:{term}"

    def mapping(self) -> dict[str, Any]:
        """The ``tools`` argument for ``chat_with_tools``."""
        return {"fetch_record": self.fetch_record, "query_index": self.query_index}


# ---------------------------------------------------------------------------
# Fixtures and fakes (DEC-004-14)
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


def _install_request_rounds(monkeypatch, client: LlamaSwapAPIClient, replies: list[Any]) -> list[dict[str, Any]]:
    """Replace ``_request`` with a per-round fake (one canned reply per call) and forbid ``_stream_post``.

    Replies and captured bodies are deep-copied so neither side can alter the other. A request beyond the
    canned replies fails the test, which is how "no further request is sent" is enforced.
    """
    pending = [copy.deepcopy(r) for r in replies]
    calls: list[dict[str, Any]] = []

    async def fake_request(method, path, **kw):
        calls.append({"method": method, "path": path, "json": copy.deepcopy(kw.get("json"))})
        if not pending:
            pytest.fail(f"unexpected extra non-streamed request #{len(calls)} to {path}")
        return pending.pop(0)

    def forbidden_stream_post(path, payload):
        pytest.fail(f"a non-streamed tool round opened a stream to {path}")

    monkeypatch.setattr(client, "_request", fake_request)
    monkeypatch.setattr(client, "_stream_post", forbidden_stream_post)
    return calls


def _install_stream_rounds(monkeypatch, client: LlamaSwapAPIClient, rounds: list[list[str]]) -> list[dict[str, Any]]:
    """Install a per-round ``_iter_sse_data`` queue behind a fake ``_stream_post``; forbid ``_request``.

    Each opened stream reads the next round's payload strings; reading an unscripted round fails the test.
    The yielded response is a placeholder the code only hands to the faked ``_iter_sse_data`` (DEC-004-14).
    Returns the captured (path, payload) of every opened stream.
    """
    pending = [list(r) for r in rounds]
    calls: list[dict[str, Any]] = []

    async def fake_iter_sse_data(_response):
        if not pending:
            pytest.fail("the stream of an unscripted round was read")
        for payload in pending.pop(0):
            yield payload

    @asynccontextmanager
    async def fake_stream_post(path, payload):
        calls.append({"path": path, "payload": copy.deepcopy(payload)})
        yield None

    async def forbidden_request(method, path, **kw):
        pytest.fail(f"a streamed tool round sent a non-streamed request: {method} {path}")

    monkeypatch.setattr(client, "_iter_sse_data", fake_iter_sse_data)
    monkeypatch.setattr(client, "_stream_post", fake_stream_post)
    monkeypatch.setattr(client, "_request", forbidden_request)
    return calls


def _fake_sse(monkeypatch, client: LlamaSwapAPIClient, payloads: list[str]) -> None:
    """Replace ``_iter_sse_data`` with a fake yielding one round of ``payloads`` (direct parser tests)."""

    async def fake_iter_sse_data(_response):
        for payload in payloads:
            yield payload

    monkeypatch.setattr(client, "_iter_sse_data", fake_iter_sse_data)


async def _run(
    client: LlamaSwapAPIClient,
    tools: _Tools,
    *,
    stream: bool = False,
    on_delta: Any = None,
    options: dict[str, Any] | None = None,
    **kw: Any,
) -> str:
    """Run ``chat_with_tools`` on copies of the standard messages and tool definitions."""
    return await client.chat_with_tools(
        copy.deepcopy(_MESSAGES),
        tools_definitions=copy.deepcopy(_TOOL_DEFS),
        tools=tools.mapping(),
        options=copy.deepcopy(options),
        stream=stream,
        on_delta=on_delta,
        **kw,
    )


# ---------------------------------------------------------------------------
# DoD-1 — non-streamed, call-option max_tokens: a cut first round without calls is the answer
# ---------------------------------------------------------------------------

# A first round's output: reasoning (not part of the text) and two messages, the first with two parts.
_ANSWER_ITEMS = (
    _reasoning_item("rs_1", "hidden reasoning"),
    _message_item("msg_1", "Hello", ", "),
    _message_item("msg_2", "world"),
)


@pytest.mark.asyncio
async def test_llamaswap_tools_cut_first_round_without_calls_returns_text__DoD1(monkeypatch):
    """DoD-1: a cut first round without tool calls returns its text, exactly as a completed round with the same
    output would; one request is sent and no tool runs (US-062.AC-4, DEC-004-16)."""
    cut_client = _client()
    cut_calls = _install_request_rounds(monkeypatch, cut_client, [_cut_reply(*_ANSWER_ITEMS)])
    cut_tools = _Tools()
    cut_result = await _run(cut_client, cut_tools, options={"max_tokens": 16})

    completed_client = _client()
    _install_request_rounds(monkeypatch, completed_client, [_reply(*_ANSWER_ITEMS)])
    completed_result = await _run(completed_client, _Tools(), options={"max_tokens": 16})

    assert cut_result == "Hello, world"
    assert cut_result == completed_result
    assert len(cut_calls) == 1
    assert cut_tools.log == []


@pytest.mark.asyncio
async def test_llamaswap_tools_cut_first_round_without_text_returns_empty__DoD1(monkeypatch):
    """DoD-1: a cut first round with no message text returns empty text, as a completed one does (DEC-003-19)."""
    client = _client()
    calls = _install_request_rounds(monkeypatch, client, [_cut_reply(_reasoning_item("rs_1", "only thinking"))])
    tools = _Tools()

    result = await _run(client, tools, options=_LIMIT)

    assert result == ""
    assert len(calls) == 1
    assert tools.log == []


def test_llamaswap_tool_round_reader_with_flag_reads_cut_round_as_completed__DoD1():
    """DoD-1: the reader with the flag on returns a cut round's text, extracted as from a completed round, and no
    tool calls (DEC-004-16)."""
    cut_text, cut_calls, _cut_reasoning = _responses_tool_round(_cut_reply(*_ANSWER_ITEMS), output_limit_set=True)
    done_text, _done_calls, _done_reasoning = _responses_tool_round(_reply(*_ANSWER_ITEMS), output_limit_set=True)

    assert cut_text == "Hello, world"
    assert cut_text == done_text
    assert cut_calls == []


# ---------------------------------------------------------------------------
# DoD-2 — non-streamed: a cut later round without calls is the answer; the earlier tool ran once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_tools_cut_second_round_without_calls_returns_its_text__DoD2(monkeypatch):
    """DoD-2: after a round whose tool ran, a second round cut at the app's limit without tool calls returns
    the second round's text; the first round's tool ran exactly once (US-062.AC-4, DEC-004-16, DEC-004-18)."""
    client = _client()
    calls = _install_request_rounds(
        monkeypatch, client, [_EARLIER_REPLY, _cut_reply(_message_item("msg_2", "The record ", "says hi"))]
    )
    tools = _Tools()

    result = await _run(client, tools, options=_LIMIT)

    assert result == "The record says hi"
    assert tools.log == [_EARLIER_RUN]
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# DoD-3 — DoD-1 for every way the limit is set, streamed and not
# ---------------------------------------------------------------------------


def _answer_stream(*, ending: str) -> list[str]:
    """A streamed answer round: reasoning, then text in two deltas, then ``ending`` and trailing events."""
    return [
        _CREATED,
        _reasoning_delta("Hmm", 0, "rs_1"),
        _text_delta("Cut ", 1, "msg_1"),
        _text_delta("text", 1, "msg_1"),
        ending,
        _text_delta("AFTER", 1, "msg_1"),
        _LATE_FAILED,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", _STREAM_MODES)
@pytest.mark.parametrize(("default_options", "options"), _LIMIT_WAYS)
async def test_llamaswap_tools_cut_first_round_returns_text_for_every_way_the_limit_is_set__DoD3(
    monkeypatch, default_options, options, stream
):
    """DoD-3: with the limit as call option or default, via ``max_tokens`` or ``context_length``, a cut first round
    without calls returns its text as a completed round would; one request, no tool run (US-062.AC-4, DEC-004-15)."""
    tools = _Tools()
    client = _client(default_options)
    if stream:
        calls = _install_stream_rounds(monkeypatch, client, [_answer_stream(ending=_CUT_EVENT)])
        rec = _Recorder()
        result = await _run(client, tools, stream=True, on_delta=rec, options=options)

        done_client = _client(default_options)
        _install_stream_rounds(monkeypatch, done_client, [_answer_stream(ending=_completed())])
        done_rec = _Recorder()
        done_result = await _run(done_client, _Tools(), stream=True, on_delta=done_rec, options=options)

        assert rec.pieces == done_rec.pieces
    else:
        items = (_reasoning_item("rs_1", "Hmm"), _message_item("msg_1", "Cut ", "text"))
        calls = _install_request_rounds(monkeypatch, client, [_cut_reply(*items)])
        result = await _run(client, tools, options=options)

        done_client = _client(default_options)
        _install_request_rounds(monkeypatch, done_client, [_reply(*items)])
        done_result = await _run(done_client, _Tools(), options=options)

    assert result == "Cut text"
    assert result == done_result
    assert len(calls) == 1
    assert tools.log == []


# ---------------------------------------------------------------------------
# DoD-4 — non-streamed: a cut round holding tool calls ends with LLMError; none of its tools run
# ---------------------------------------------------------------------------

_CUT_CALL_SETS = [
    pytest.param(
        (_message_item("msg_c", "Let me check."), _call_item("fc_q", "call_q", "query_index", '{"term": "be')),
        id="one-call",
    ),
    pytest.param(
        (
            _call_item("fc_a", "call_a", "fetch_record", '{"key": "beta-2"}'),
            _call_item("fc_q", "call_q", "query_index", '{"term": "gam'),
        ),
        id="two-calls",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("position", _ROUND_POSITIONS)
@pytest.mark.parametrize("cut_items", _CUT_CALL_SETS)
async def test_llamaswap_tools_cut_round_with_calls_raises_and_runs_none__DoD4(monkeypatch, cut_items, position):
    """DoD-4: with the app's limit set, a cut round holding tool calls ends with LLMError carrying the server's
    reason; no tool of that round runs and no further request is sent; earlier rounds' tools ran exactly once
    (US-062.AC-4, DEC-004-16, DEC-004-18)."""
    replies = [_cut_reply(*cut_items)] if position == 0 else [_EARLIER_REPLY, _cut_reply(*cut_items)]
    client = _client()
    calls = _install_request_rounds(monkeypatch, client, replies)
    tools = _Tools()

    with pytest.raises(LLMError) as excinfo:
        await _run(client, tools, options=_LIMIT)

    assert _CUT_REASON in str(excinfo.value)
    assert tools.log == ([] if position == 0 else [_EARLIER_RUN])
    assert len(calls) == position + 1


@pytest.mark.parametrize("cut_items", _CUT_CALL_SETS)
def test_llamaswap_tool_round_reader_with_flag_raises_on_cut_round_with_calls__DoD4(cut_items):
    """DoD-4: the reader with the flag on raises LLMError carrying the reason for a cut round holding calls
    (DEC-004-16)."""
    with pytest.raises(LLMError) as excinfo:
        _responses_tool_round(_cut_reply(*cut_items), output_limit_set=True)

    assert _CUT_REASON in str(excinfo.value)


# ---------------------------------------------------------------------------
# DoD-5 — streamed: a cut round with no started call is the answer; pieces in order; open block closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("position", _ROUND_POSITIONS)
async def test_llamaswap_streamed_tools_cut_round_without_call_returns_text_deltas__DoD5(monkeypatch, position):
    """DoD-5: a streamed round cut at the app's limit with no call started returns its text deltas joined (no
    markers, no reasoning); the receiver got every piece of every round in order and nothing after the event
    (US-062.AC-4, US-027.AC-1, DEC-004-17, DEC-000-20)."""
    earlier = [
        _CREATED,
        _reasoning_delta("Need ", 0, "rs_1"),
        _reasoning_delta("the record.", 0, "rs_1"),
        _call_added(1, "fc_1", "call_1", "fetch_record"),
        _args_delta(_FULL_ARGS, 1, "fc_1"),
        _item_done(1, _EARLIER_CALL),
        _item_done(0, _reasoning_item("rs_1", "Need the record.")),
        _completed(_reasoning_item("rs_1", "Need the record."), _EARLIER_CALL),
    ]
    earlier_pieces = [f"{_OPEN}Need ", "the record.", _CLOSE]
    cut = [
        _CREATED,
        _reasoning_delta("Hmm", 0, "rs_2"),
        _text_delta("Ans", 1, "msg_2"),
        _text_delta("wer", 1, "msg_2"),
        _CUT_EVENT,
        _text_delta("AFTER", 1, "msg_2"),
        _reasoning_delta("LATE", 0, "rs_2"),
        _LATE_FAILED,
    ]
    cut_pieces = [f"{_OPEN}Hmm", f"{_CLOSE}Ans", "wer"]
    rounds = [cut] if position == 0 else [earlier, cut]
    client = _client()
    calls = _install_stream_rounds(monkeypatch, client, rounds)
    tools = _Tools()
    rec = _Recorder()

    result = await _run(client, tools, stream=True, on_delta=rec, options=_LIMIT)

    assert result == "Answer"
    assert rec.pieces == (cut_pieces if position == 0 else [*earlier_pieces, *cut_pieces])
    assert tools.log == ([] if position == 0 else [_EARLIER_RUN])
    assert len(calls) == position + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("position", _ROUND_POSITIONS)
async def test_llamaswap_streamed_tools_cut_inside_reasoning_closes_block_with_own_piece__DoD5(monkeypatch, position):
    """DoD-5: a reasoning block still open at the cut is closed by a closing-marker piece of its own; nothing after
    the event is delivered; the returned text holds no reasoning or markers (DEC-004-17, DEC-000-20)."""
    cut = [
        _CREATED,
        _text_delta("Part", 1, "msg_2"),
        _reasoning_delta("more", 0, "rs_2"),
        _CUT_EVENT,
        _text_delta("AFTER", 1, "msg_2"),
        _completed(),
    ]
    rounds = [cut] if position == 0 else [_earlier_stream_round(), cut]
    client = _client()
    calls = _install_stream_rounds(monkeypatch, client, rounds)
    tools = _Tools()
    rec = _Recorder()

    result = await _run(client, tools, stream=True, on_delta=rec, options=_LIMIT)

    assert result == "Part"
    assert rec.pieces == ["Part", f"{_OPEN}more", _CLOSE]
    assert tools.log == ([] if position == 0 else [_EARLIER_RUN])
    assert len(calls) == position + 1


@pytest.mark.asyncio
async def test_llamaswap_tool_round_parser_with_flag_ends_at_cut_without_call__DoD5(monkeypatch):
    """DoD-5: the parser with the flag on ends a round at a matching cut with no call started: text deltas only,
    no calls, an open block closed by its own piece, nothing after the event (DEC-004-17)."""
    client = _client()
    _fake_sse(
        monkeypatch,
        client,
        [_text_delta("x"), _text_delta("y"), _reasoning_delta("r"), _CUT_EVENT, _text_delta("z"), _LATE_FAILED],
    )
    rec = _Recorder()

    text, calls, _reasoning = await client._stream_responses_tool_round(None, rec, output_limit_set=True)

    assert text == "xy"
    assert calls == []
    assert rec.pieces == ["x", "y", f"{_OPEN}r", _CLOSE]


# ---------------------------------------------------------------------------
# DoD-6 — streamed: a cut round with a started call ends with LLMError; nothing delivered after the event
# ---------------------------------------------------------------------------

# Ways a call has started before the cut (DEC-004-17): an added call item, an arguments fragment alone (no item
# was added for it), or a done call item alone. A reasoning block is open first, so "not even a closing marker"
# is observable.
_STARTED_CALLS = [
    pytest.param(
        [_call_added(1, "fc_q", "call_q", "query_index"), _args_delta('{"term": "be', 1, "fc_q")], id="call-added"
    ),
    pytest.param([_args_delta('{"term": "be', 1, "fc_q")], id="arguments-fragment-alone"),
    pytest.param(
        [_item_done(1, _call_item("fc_q", "call_q", "query_index", '{"term": "beta-3"}'))], id="call-done-alone"
    ),
]


def _started_call_round(started: list[str], ending: str) -> list[str]:
    """A round: an open reasoning block, then the started-call events, then ``ending`` and trailing events."""
    return [
        _CREATED,
        _reasoning_delta("pondering", 0, "rs_q"),
        *started,
        ending,
        _text_delta("AFTER", 2, "msg_q"),
        _completed(),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("position", _ROUND_POSITIONS)
@pytest.mark.parametrize("started", _STARTED_CALLS)
async def test_llamaswap_streamed_tools_cut_round_with_started_call_raises__DoD6(monkeypatch, started, position):
    """DoD-6: a streamed round cut at the app's limit after a call started ends with LLMError carrying the reason;
    no piece after the event (not even a closing marker); no tool of that round runs; no further request; earlier
    rounds' tools ran exactly once (US-062.AC-4, DEC-004-17, DEC-004-18)."""
    cut = _started_call_round(started, _CUT_EVENT)
    rounds = [cut] if position == 0 else [_earlier_stream_round(), cut]
    client = _client()
    calls = _install_stream_rounds(monkeypatch, client, rounds)
    tools = _Tools()
    rec = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await _run(client, tools, stream=True, on_delta=rec, options=_LIMIT)

    assert _CUT_REASON in str(excinfo.value)
    assert rec.pieces == [f"{_OPEN}pondering"]
    assert tools.log == ([] if position == 0 else [_EARLIER_RUN])
    assert len(calls) == position + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("started", _STARTED_CALLS)
async def test_llamaswap_tool_round_parser_with_flag_raises_on_cut_after_started_call__DoD6(monkeypatch, started):
    """DoD-6: the parser with the flag on raises LLMError with the reason at a cut after a started call and
    delivers nothing after the event, not even a closing marker (DEC-004-17)."""
    client = _client()
    _fake_sse(monkeypatch, client, _started_call_round(started, _CUT_EVENT))
    rec = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await client._stream_responses_tool_round(None, rec, output_limit_set=True)

    assert _CUT_REASON in str(excinfo.value)
    assert rec.pieces == [f"{_OPEN}pondering"]


# ---------------------------------------------------------------------------
# DoD-7 — no app limit: a cut round stays LLMError, streamed and not, with and without calls
# ---------------------------------------------------------------------------


def _cut_round(stream: bool, with_calls: bool) -> Any:
    """A first round cut at the output limit, in the given mode, holding a call or only text."""
    if stream:
        started = [_call_added(1, "fc_q", "call_q", "query_index"), _args_delta('{"term": "x"}', 1, "fc_q")]
        events = [_CREATED, _text_delta("partial", 0, "msg_1"), *(started if with_calls else []), _CUT_EVENT]
        return [*events, _text_delta("AFTER", 0, "msg_1"), _completed()]
    items = [_message_item("msg_1", "partial")]
    if with_calls:
        items.append(_call_item("fc_q", "call_q", "query_index", '{"term": "x"}'))
    return _cut_reply(*items)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_calls", [pytest.param(False, id="no-calls"), pytest.param(True, id="with-calls")])
@pytest.mark.parametrize("stream", _STREAM_MODES)
@pytest.mark.parametrize(("default_options", "options"), _NO_LIMIT_WAYS)
async def test_llamaswap_tools_cut_round_without_app_limit_raises_with_reason__DoD7(
    monkeypatch, default_options, options, stream, with_calls
):
    """DoD-7: with no app limit (never set, or a default cleared with ``None``), a round cut for the output limit
    ends with LLMError carrying the server's reason, streamed and not, with and without calls; no tool of that
    round runs (US-062.AC-6, DEC-004-15, DEC-003-17)."""
    client = _client(default_options)
    tools = _Tools()
    rec = _Recorder()
    if stream:
        calls = _install_stream_rounds(monkeypatch, client, [_cut_round(True, with_calls)])
    else:
        calls = _install_request_rounds(monkeypatch, client, [_cut_round(False, with_calls)])

    with pytest.raises(LLMError) as excinfo:
        await _run(client, tools, stream=stream, on_delta=rec if stream else None, options=options)

    assert _CUT_REASON in str(excinfo.value)
    assert tools.log == []
    assert len(calls) == 1
    if stream:
        assert rec.pieces == ["partial"]


# ---------------------------------------------------------------------------
# DoD-8 — app limit set: every other early end of a round stays LLMError, streamed and not
# ---------------------------------------------------------------------------

# (incomplete details or the "missing" marker, text the error must carry or None when only LLMError is required).
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


def _other_incomplete_reply(details: Any) -> dict[str, Any]:
    """A non-streamed incomplete round reply holding only text, with the given details (or none at all)."""
    reply = _reply(_message_item("msg_1", "partial"), status="incomplete")
    if details is not _MISSING:
        reply["incomplete_details"] = copy.deepcopy(details)
    return reply


def _other_incomplete_event(details: Any) -> str:
    """A ``response.incomplete`` event with the given details (or none at all)."""
    if details is _MISSING:
        return _incomplete_event(with_details=False)
    return _incomplete_event(copy.deepcopy(details))


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", _STREAM_MODES)
@pytest.mark.parametrize(("details", "carried"), _OTHER_INCOMPLETE)
async def test_llamaswap_tools_other_incomplete_round_with_limit_raises__DoD8(monkeypatch, details, carried, stream):
    """DoD-8: with the app's limit set, an incomplete round whose reason is any other value, has no reason or has
    no incomplete details ends with LLMError, streamed and not (US-024.AC-2, DEC-003-18, DEC-004-8, DEC-004-9)."""
    client = _client()
    tools = _Tools()
    rec = _Recorder()
    if stream:
        payloads = [_CREATED, _text_delta("partial"), _other_incomplete_event(details), _text_delta("AFTER")]
        calls = _install_stream_rounds(monkeypatch, client, [[*payloads, _completed()]])
    else:
        calls = _install_request_rounds(monkeypatch, client, [_other_incomplete_reply(details)])

    with pytest.raises(LLMError) as excinfo:
        await _run(client, tools, stream=stream, on_delta=rec if stream else None, options=_LIMIT)

    if carried is not None:
        assert carried in str(excinfo.value)
    assert len(calls) == 1
    if stream:
        assert rec.pieces == ["partial"]


@pytest.mark.parametrize(("details", "carried"), _OTHER_INCOMPLETE)
def test_llamaswap_tool_round_reader_with_flag_other_incomplete_raises__DoD8(details, carried):
    """DoD-8: the reader with the flag on still raises LLMError for every non-matching incomplete round."""
    with pytest.raises(LLMError) as excinfo:
        _responses_tool_round(_other_incomplete_reply(details), output_limit_set=True)

    if carried is not None:
        assert carried in str(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(("details", "carried"), _OTHER_INCOMPLETE)
async def test_llamaswap_tool_round_parser_with_flag_other_incomplete_raises__DoD8(monkeypatch, details, carried):
    """DoD-8: the parser with the flag on still raises LLMError for every non-matching incomplete event and
    delivers nothing after it."""
    client = _client()
    _fake_sse(monkeypatch, client, [_text_delta("before"), _other_incomplete_event(details), _text_delta("after")])
    rec = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await client._stream_responses_tool_round(None, rec, output_limit_set=True)

    if carried is not None:
        assert carried in str(excinfo.value)
    assert rec.pieces == ["before"]


@pytest.mark.asyncio
async def test_llamaswap_tools_failed_round_reply_with_limit_raises_with_reason__DoD8(monkeypatch):
    """DoD-8: non-streamed, with the app's limit set, a failed round reply ends with LLMError carrying the reason."""
    client = _client()
    calls = _install_request_rounds(
        monkeypatch, client, [_reply(status="failed", error={"code": "server_error", "message": "engine-crashed-f4"})]
    )

    with pytest.raises(LLMError) as excinfo:
        await _run(client, _Tools(), options=_LIMIT)

    assert "engine-crashed-f4" in str(excinfo.value)
    assert len(calls) == 1


def test_llamaswap_tool_round_reader_with_flag_failed_reply_raises__DoD8():
    """DoD-8: the reader with the flag on still raises LLMError carrying the reason for a failed reply."""
    with pytest.raises(LLMError) as excinfo:
        _responses_tool_round(_reply(status="failed", error={"message": "engine-crashed-f5"}), output_limit_set=True)

    assert "engine-crashed-f5" in str(excinfo.value)


_FAILURE_EVENTS = [
    pytest.param(
        _ev("response.failed", response={"status": "failed", "error": {"code": "s", "message": "boom-f7"}}),
        "boom-f7",
        id="failed-event",
    ),
    pytest.param(_ev("error", code="rate_limit", message="slow-down-e2"), "slow-down-e2", id="error-event"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("event", "reason"), _FAILURE_EVENTS)
async def test_llamaswap_streamed_tools_failure_event_with_limit_raises_with_reason__DoD8(monkeypatch, event, reason):
    """DoD-8: streamed, with the app's limit set, a failed or error event ends with LLMError carrying the reason;
    nothing after it is delivered."""
    client = _client()
    calls = _install_stream_rounds(
        monkeypatch, client, [[_CREATED, _text_delta("before"), event, _text_delta("after"), _completed()]]
    )
    rec = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await _run(client, _Tools(), stream=True, on_delta=rec, options=_LIMIT)

    assert reason in str(excinfo.value)
    assert rec.pieces == ["before"]
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("event", "reason"), _FAILURE_EVENTS)
async def test_llamaswap_tool_round_parser_with_flag_failure_event_raises__DoD8(monkeypatch, event, reason):
    """DoD-8: the parser with the flag on still raises LLMError with the reason on failed and error events."""
    client = _client()
    _fake_sse(monkeypatch, client, [_text_delta("before"), event, _text_delta("after"), _completed()])
    rec = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await client._stream_responses_tool_round(None, rec, output_limit_set=True)

    assert reason in str(excinfo.value)
    assert rec.pieces == ["before"]


_EARLY_ENDINGS = [
    pytest.param([_CREATED, _text_delta("partial"), _text_delta(" answer")], id="no-completion-event"),
    pytest.param([_CREATED, _text_delta("partial"), "[DONE]"], id="done-sentinel-only"),
    pytest.param([], id="empty-stream"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("payloads", _EARLY_ENDINGS)
async def test_llamaswap_streamed_tools_unfinished_round_with_limit_raises__DoD8(monkeypatch, payloads):
    """DoD-8: streamed, with the app's limit set, a round stream ending without a completion event ends with
    LLMError and no further request is sent."""
    client = _client()
    calls = _install_stream_rounds(monkeypatch, client, [payloads])

    with pytest.raises(LLMError):
        await _run(client, _Tools(), stream=True, on_delta=_Recorder(), options=_LIMIT)

    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("payloads", _EARLY_ENDINGS)
async def test_llamaswap_tool_round_parser_with_flag_unfinished_stream_raises__DoD8(monkeypatch, payloads):
    """DoD-8: the parser with the flag on still raises LLMError when the stream ends without a completion event."""
    client = _client()
    _fake_sse(monkeypatch, client, payloads)

    with pytest.raises(LLMError):
        await client._stream_responses_tool_round(None, _Recorder(), output_limit_set=True)


# ---------------------------------------------------------------------------
# DoD-9 — the reader and the parser without the flag keep the old rule
# ---------------------------------------------------------------------------

_FLAG_DEFAULTS = [pytest.param({}, id="flag-omitted"), pytest.param({"output_limit_set": False}, id="flag-false")]


@pytest.mark.parametrize("flag", _FLAG_DEFAULTS)
@pytest.mark.parametrize(
    "items",
    [
        pytest.param((_message_item("msg_1", "partial"),), id="no-calls"),
        pytest.param(
            (_message_item("msg_1", "partial"), _call_item("fc_q", "call_q", "query_index", '{"term": "x"}')),
            id="with-calls",
        ),
    ],
)
def test_llamaswap_tool_round_reader_without_flag_raises_on_cut__DoD9(items, flag):
    """DoD-9: the reader called without the flag (or with it off) raises LLMError with the reason on a cut round
    (DEC-004-15)."""
    with pytest.raises(LLMError) as excinfo:
        _responses_tool_round(_cut_reply(*items), **flag)

    assert _CUT_REASON in str(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", _FLAG_DEFAULTS)
async def test_llamaswap_tool_round_parser_without_flag_raises_on_cut__DoD9(monkeypatch, flag):
    """DoD-9: the parser called without the flag (or with it off) raises LLMError with the reason at a cut event
    with no call started, and delivers nothing after it (DEC-004-15)."""
    client = _client()
    _fake_sse(monkeypatch, client, [_text_delta("before"), _CUT_EVENT, _text_delta("after"), _completed()])
    rec = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await client._stream_responses_tool_round(None, rec, **flag)

    assert _CUT_REASON in str(excinfo.value)
    assert rec.pieces == ["before"]


# ---------------------------------------------------------------------------
# DoD-10 — max_loops 1: a cut answer round is returned, no round-limit error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", _STREAM_MODES)
async def test_llamaswap_tools_cut_answer_on_only_allowed_round_returns_text__DoD10(monkeypatch, stream):
    """DoD-10: with ``max_loops`` 1 and the app's limit set, a first round cut without calls returns its text and
    raises no round-limit error (DEC-004-18, DEC-002-2)."""
    client = _client()
    tools = _Tools()
    if stream:
        calls = _install_stream_rounds(
            monkeypatch, client, [[_CREATED, _text_delta("Only "), _text_delta("round"), _CUT_EVENT]]
        )
        result = await _run(client, tools, stream=True, on_delta=_Recorder(), options=_LIMIT, max_loops=1)
    else:
        calls = _install_request_rounds(monkeypatch, client, [_cut_reply(_message_item("msg_1", "Only round"))])
        result = await _run(client, tools, options=_LIMIT, max_loops=1)

    assert result == "Only round"
    assert len(calls) == 1
    assert tools.log == []


# ---------------------------------------------------------------------------
# DoD-11 — tracing: a cut answer round is returned and recorded as the call's response
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


def _assert_recorded_like_completed(cut_folder: Path, completed_folder: Path, text: str) -> None:
    """The cut record has no error section, the same sections as the completed one and the same response."""
    cut_sections = _sections(_single_record(cut_folder))
    completed_sections = _sections(_single_record(completed_folder))
    cut_titles = [title for title, _ in cut_sections]
    assert "ERROR" not in cut_titles
    assert cut_titles == [title for title, _ in completed_sections]
    cut_responses = [body for title, body in cut_sections if title == "RESPONSE"]
    assert cut_responses == [body for title, body in completed_sections if title == "RESPONSE"]
    assert any(text in line for body in cut_responses for line in body)


@pytest.mark.asyncio
async def test_llamaswap_traced_tools_cut_answer_is_returned_and_recorded_as_response__DoD11(monkeypatch, tmp_path):
    """DoD-11: traced non-streamed tool call: a cut answer round is returned and recorded as the call's response,
    as a completed answer round is (DEC-004-18, DEC-004-10, DEC-000-48)."""
    text = "Traced cut answer 7f3a"

    completed_client = _traced_client(monkeypatch, tmp_path / "completed")
    _install_request_rounds(monkeypatch, completed_client, [_reply(_message_item("msg_1", text))])
    completed_result = await _run(completed_client, _Tools(), options=_LIMIT)

    cut_client = _traced_client(monkeypatch, tmp_path / "cut")
    _install_request_rounds(monkeypatch, cut_client, [_cut_reply(_message_item("msg_1", text))])
    cut_result = await _run(cut_client, _Tools(), options=_LIMIT)

    assert cut_result == completed_result == text
    _assert_recorded_like_completed(tmp_path / "cut", tmp_path / "completed", text)


@pytest.mark.asyncio
async def test_llamaswap_traced_streamed_tools_cut_answer_is_returned_and_recorded__DoD11(monkeypatch, tmp_path):
    """DoD-11: traced streamed tool call: a cut answer round is returned and recorded as the call's response,
    as a completed answer round is (DEC-004-18, DEC-004-10, DEC-000-48)."""
    pieces = [_CREATED, _text_delta("Streamed "), _text_delta("cut 4b1d")]

    completed_client = _traced_client(monkeypatch, tmp_path / "completed")
    _install_stream_rounds(monkeypatch, completed_client, [[*pieces, _completed()]])
    completed_rec = _Recorder()
    completed_result = await _run(completed_client, _Tools(), stream=True, on_delta=completed_rec, options=_LIMIT)

    cut_client = _traced_client(monkeypatch, tmp_path / "cut")
    _install_stream_rounds(monkeypatch, cut_client, [[*pieces, _CUT_EVENT, _text_delta("AFTER")]])
    cut_rec = _Recorder()
    cut_result = await _run(cut_client, _Tools(), stream=True, on_delta=cut_rec, options=_LIMIT)

    assert cut_result == completed_result == "Streamed cut 4b1d"
    assert cut_rec.pieces == completed_rec.pieces
    _assert_recorded_like_completed(tmp_path / "cut", tmp_path / "completed", "Streamed cut 4b1d")
