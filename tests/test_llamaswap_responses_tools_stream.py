"""Unit tests for streamed tool conversations on the llama-swap Responses class (feature 004.responses-tools, step 004).

Covers ``LlamaSwapAPIClient.chat_with_tools(stream=True)`` and its tool-round stream parser
``_stream_responses_tool_round`` (DEC-004-9): one streamed ``/responses`` request per round with the
non-streamed body plus ``stream`` true, text and reasoning pieces per round with fresh think markers,
call assembly from tool-call events (including the llama.cpp sequence with out-of-order item
completion and no arguments-done event), reasoning carry to later rounds (DEC-004-7), failure events
and early end, non-2xx on a later round, receiver rules, structured answers and tracing parity.

The server is faked through the DEC-004-14 seams: a per-round queue behind a fake ``_stream_post``
(async context manager, bodies deep-copied) and a fake ``_iter_sse_data``; non-streamed parity runs use
a per-round fake ``_request``. Every test names its DoD id.
"""

import asyncio
import copy
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest
from yarl import URL

from llm.exceptions import LLMError
from llm.llama_swap import LlamaSwapAPIClient

_BASE_URL = "http://localhost:8080/v1"
_RESPONSES_URL = f"{_BASE_URL}/responses"
_MODEL = "some-model"
_SECRET = "sk-llamaswap-secret-token-456"

_OPEN = "<think>"
_CLOSE = "</think>"

_USER_TEXT = "Look up the record alpha-7 and tell me what it says."
_MESSAGES = [{"role": "user", "content": _USER_TEXT}]

# The full argument text of the llama.cpp sample call, and the three deltas it arrives in.
_FULL_ARGS = '{"key": "alpha-7"}'
_ARG_DELTAS = ['{"key"', ': "alp', 'ha-7"}']


# ---------------------------------------------------------------------------
# Event and item builders (context.md § Responses wire vocabulary for tools)
# ---------------------------------------------------------------------------


def _ev(event_type: str, **fields: Any) -> str:
    """One stream payload string: a JSON event object with ``type`` and the given fields."""
    return json.dumps({"type": event_type, **fields})


_CREATED = _ev(
    "response.created", response={"id": "resp_1", "object": "response", "status": "in_progress", "output": []}
)
_IN_PROGRESS = _ev(
    "response.in_progress", response={"id": "resp_1", "object": "response", "status": "in_progress", "output": []}
)


def _completed(*output: dict[str, Any]) -> str:
    """The completed event, carrying the round's full output as llama.cpp sends it."""
    response = {"id": "resp_1", "object": "response", "status": "completed", "output": list(output)}
    return _ev("response.completed", response=response)


def _reasoning_item(item_id: str, text: str) -> dict[str, Any]:
    """A finished llama.cpp reasoning item: plain reasoning text, empty summary and encrypted content, no status."""
    return {
        "type": "reasoning",
        "id": item_id,
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
        "encrypted_content": "",
    }


def _reasoning_added(index: int, item_id: str) -> str:
    """The reasoning ``output_item.added`` event exactly as llama.cpp sends it (004.context.md probe details)."""
    item = {
        "type": "reasoning",
        "id": item_id,
        "status": "in_progress",
        "content": [],
        "summary": [],
        "encrypted_content": "",
    }
    return _ev("response.output_item.added", output_index=index, item=item)


def _reasoning_delta(delta: str, index: int = 0, item_id: str = "rs_1") -> str:
    """A reasoning-text delta, carrying ``item_id``, ``output_index`` and ``content_index``."""
    return _ev("response.reasoning_text.delta", item_id=item_id, output_index=index, content_index=0, delta=delta)


def _summary_delta(delta: str, index: int = 0, item_id: str = "rs_1") -> str:
    """A reasoning-summary delta (handled as reasoning, DEC-004-9)."""
    return _ev(
        "response.reasoning_summary_text.delta", item_id=item_id, output_index=index, summary_index=0, delta=delta
    )


def _item_done(index: int, item: dict[str, Any]) -> str:
    """An ``output_item.done`` event for any item."""
    return _ev("response.output_item.done", output_index=index, item=item)


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


def _call_added(index: int, item_id: str, call_id: str, name: str) -> str:
    """The function-call ``output_item.added`` event: name and call id present, arguments empty."""
    item = {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": name,
        "arguments": "",
        "status": "in_progress",
    }
    return _ev("response.output_item.added", output_index=index, item=item)


def _args_delta(delta: str, index: int, item_id: str) -> str:
    """A function-call arguments delta (``delta``, ``item_id``, ``output_index``)."""
    return _ev("response.function_call_arguments.delta", item_id=item_id, output_index=index, delta=delta)


def _args_done(index: int, item_id: str, arguments: str) -> str:
    """A function-call arguments-done event (OpenAI sends it; llama.cpp does not)."""
    return _ev("response.function_call_arguments.done", item_id=item_id, output_index=index, arguments=arguments)


def _message_item(item_id: str, text: str) -> dict[str, Any]:
    """A finished assistant message item with one output-text part."""
    return {
        "type": "message",
        "id": item_id,
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _message_added(index: int, item_id: str = "msg_1") -> str:
    """The message ``output_item.added`` event."""
    item = {"type": "message", "id": item_id, "role": "assistant", "status": "in_progress", "content": []}
    return _ev("response.output_item.added", output_index=index, item=item)


def _text_delta(delta: str, index: int = 1, item_id: str = "msg_1") -> str:
    """An output-text delta."""
    return _ev("response.output_text.delta", item_id=item_id, output_index=index, content_index=0, delta=delta)


# The reasoning item the llama.cpp sample round finishes with (re-sent verbatim on later rounds).
_R1 = _reasoning_item("rs_1", "Need the record.")
_CALL_1 = _call_item("fc_1", "call_1", "fetch_record", _FULL_ARGS)


def _llamacpp_tool_round() -> list[str]:
    """The llama.cpp tool-round sequence (step DoD-4, context.md § wire vocabulary), in this exact order.

    created, in progress; reasoning added at 0 then its deltas; call added at 1 with name, call id and
    empty arguments, then argument deltas; the call done with full arguments; the reasoning done after the
    call's deltas; completed. No arguments-done event and no ``[DONE]``.
    """
    return [
        _CREATED,
        _IN_PROGRESS,
        _reasoning_added(0, "rs_1"),
        _reasoning_delta("Need ", 0, "rs_1"),
        _reasoning_delta("the record.", 0, "rs_1"),
        _call_added(1, "fc_1", "call_1", "fetch_record"),
        *[_args_delta(d, 1, "fc_1") for d in _ARG_DELTAS],
        _item_done(1, _CALL_1),
        _item_done(0, _R1),
        _completed(_R1, _CALL_1),
    ]


def _call_round(item_id: str, call_id: str, name: str, arguments: str, *, text: str = "") -> list[str]:
    """A tool round without reasoning: optional text at position 0, then one call at position 1."""
    events = [_CREATED]
    output: list[dict[str, Any]] = []
    if text:
        events += [_message_added(0), _text_delta(text, 0), _item_done(0, _message_item("msg_1", text))]
        output.append(_message_item("msg_1", text))
    call = _call_item(item_id, call_id, name, arguments)
    events += [_call_added(1, item_id, call_id, name), _args_delta(arguments, 1, item_id), _item_done(1, call)]
    output.append(call)
    return [*events, _completed(*output)]


def _answer_round(*texts: str, reasoning: tuple[str, ...] = (), reasoning_id: str = "rs_9") -> list[str]:
    """A final round: optional reasoning at position 0, then a message whose text arrives as ``texts``."""
    events = [_CREATED, _IN_PROGRESS]
    output: list[dict[str, Any]] = []
    msg_index = 0
    if reasoning:
        item = _reasoning_item(reasoning_id, "".join(reasoning))
        events += [_reasoning_added(0, reasoning_id), *[_reasoning_delta(r, 0, reasoning_id) for r in reasoning]]
        events.append(_item_done(0, item))
        output.append(item)
        msg_index = 1
    message = _message_item("msg_9", "".join(texts))
    events += [_message_added(msg_index, "msg_9"), *[_text_delta(t, msg_index, "msg_9") for t in texts]]
    events.append(_item_done(msg_index, message))
    output.append(message)
    return [*events, _completed(*output)]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _tool_def(name: str, param: str) -> dict[str, Any]:
    """An app tool definition in today's nested shape with one required text parameter."""
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
# Fakes (DEC-004-14)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch):
    """Start every test with tracing off so a developer's shell env cannot change behaviour (DEC-000-48)."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _client(**kw: Any) -> LlamaSwapAPIClient:
    """Build a llama-swap Responses client on the fixed test address (credential optional)."""
    return LlamaSwapAPIClient(model=_MODEL, base_url=_BASE_URL, **kw)


class _Recorder:
    """Async receiver collecting every piece it is handed, in order."""

    def __init__(self) -> None:
        self.pieces: list[str] = []

    async def __call__(self, piece: str) -> None:
        self.pieces.append(piece)


def _fake_sse_rounds(monkeypatch, client: LlamaSwapAPIClient, rounds: list[list[str]]) -> None:
    """Replace ``_iter_sse_data`` with a per-round queue: each read yields the next round's payload strings.

    Payloads are SSE data strings without ``data:``. Reading more rounds than scripted fails the test.
    """
    pending = [list(r) for r in rounds]

    async def fake_iter_sse_data(_response):
        if not pending:
            pytest.fail("the stream of an unscripted round was read")
        for payload in pending.pop(0):
            yield payload

    monkeypatch.setattr(client, "_iter_sse_data", fake_iter_sse_data)


def _fake_stream_post(monkeypatch, client: LlamaSwapAPIClient) -> list[dict[str, Any]]:
    """Replace ``_stream_post`` with a fake async context manager; returns the captured (path, payload) calls.

    The yielded response is a placeholder that the code only hands to the faked ``_iter_sse_data``
    (DEC-004-14). Payloads are deep-copied so later mutation by the loop cannot change what was captured.
    """
    calls: list[dict[str, Any]] = []

    @asynccontextmanager
    async def fake_stream_post(path, payload):
        calls.append({"path": path, "payload": copy.deepcopy(payload)})
        yield None

    monkeypatch.setattr(client, "_stream_post", fake_stream_post)
    return calls


def _install_stream_rounds(monkeypatch, client: LlamaSwapAPIClient, rounds: list[list[str]]) -> list[dict[str, Any]]:
    """Install both streamed fakes and forbid ``_request``: streamed rounds must not use the non-streamed seam."""
    _fake_sse_rounds(monkeypatch, client, rounds)
    calls = _fake_stream_post(monkeypatch, client)

    async def forbidden_request(method, path, **kw):
        pytest.fail(f"a streamed tool round sent a non-streamed request: {method} {path}")

    monkeypatch.setattr(client, "_request", forbidden_request)
    return calls


def _install_request_rounds(
    monkeypatch, client: LlamaSwapAPIClient, replies: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Replace ``_request`` with a per-round fake returning one canned reply per call; bodies deep-copied."""
    pending = [copy.deepcopy(r) for r in replies]
    calls: list[dict[str, Any]] = []

    async def fake_request(method, path, **kw):
        calls.append({"method": method, "path": path, "json": copy.deepcopy(kw.get("json"))})
        if not pending:
            pytest.fail("an unscripted non-streamed round was requested")
        return pending.pop(0)

    monkeypatch.setattr(client, "_request", fake_request)
    return calls


def _reply(*items: dict[str, Any]) -> dict[str, Any]:
    """A completed non-streamed Responses reply with the given output items."""
    return {"id": "resp_1", "object": "response", "status": "completed", "output": list(items)}


async def _parse(
    monkeypatch, payloads: list[str], receiver: Any = None
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Drive the tool-round parser directly on one round of canned payloads (DEC-004-14)."""
    client = _client()
    _fake_sse_rounds(monkeypatch, client, [payloads])
    return await client._stream_responses_tool_round(None, receiver)


async def _stream_tools(client: LlamaSwapAPIClient, tools: _Tools, on_delta: Any = None, **kw: Any) -> str:
    """Run a streamed tool conversation on a copy of the standard messages."""
    return await client.chat_with_tools(
        copy.deepcopy(_MESSAGES),
        tools_definitions=copy.deepcopy(_TOOL_DEFS),
        tools=tools.mapping(),
        stream=True,
        on_delta=on_delta,
        **kw,
    )


# ---------------------------------------------------------------------------
# DoD-1 — one streamed request per round, body = non-streamed body with stream true
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_format", [pytest.param(None, id="no-format"), pytest.param({"type": "json_object"}, id="json-object")]
)
async def test_llamaswap_streamed_rounds_send_nonstreamed_body_with_stream_true__DoD1(monkeypatch, response_format):
    """DoD-1: each streamed round opens one request to /responses; its body is the non-streamed round body
    with ``stream`` true (DEC-004-10, DEC-004-14)."""
    options = {"temperature": 0.2, "max_tokens": 256, "enable_thinking": True}
    answer = '{"ok": true}'

    plain = _client()
    requests = _install_request_rounds(
        monkeypatch, plain, [_reply(_R1, _CALL_1), _reply(_message_item("msg_9", answer))]
    )
    plain_tools = _Tools()
    plain_result = await plain.chat_with_tools(
        copy.deepcopy(_MESSAGES),
        tools_definitions=copy.deepcopy(_TOOL_DEFS),
        tools=plain_tools.mapping(),
        system="Be brief.",
        options=dict(options),
        response_format=copy.deepcopy(response_format),
    )
    assert len(requests) == 2
    assert all(r["json"]["stream"] is False for r in requests)

    streamed = _client()
    calls = _install_stream_rounds(monkeypatch, streamed, [_llamacpp_tool_round(), _answer_round(answer)])
    streamed_tools = _Tools()
    streamed_result = await streamed.chat_with_tools(
        copy.deepcopy(_MESSAGES),
        tools_definitions=copy.deepcopy(_TOOL_DEFS),
        tools=streamed_tools.mapping(),
        system="Be brief.",
        options=dict(options),
        stream=True,
        response_format=copy.deepcopy(response_format),
    )

    assert len(calls) == 2
    assert [c["path"] for c in calls] == ["/responses", "/responses"]
    for streamed_call, plain_call in zip(calls, requests):
        assert streamed_call["payload"] == {**plain_call["json"], "stream": True}
    assert streamed_result == plain_result == answer
    assert streamed_tools.log == plain_tools.log


# ---------------------------------------------------------------------------
# DoD-2 — text pieces of every round; only the final round's text is returned
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_streamed_tools_receiver_gets_every_rounds_text_and_call_returns_final__DoD2(monkeypatch):
    """DoD-2: the receiver gets the text pieces of every round in order; the call returns only the final
    round's text (US-023.AC-1, US-023.AC-2)."""
    client = _client()
    rounds = [
        _call_round("fc_1", "call_1", "fetch_record", _FULL_ARGS, text="Checking. "),
        _answer_round("Found ", "it."),
    ]
    _install_stream_rounds(monkeypatch, client, rounds)
    rec = _Recorder()

    result = await _stream_tools(client, _Tools(), rec)

    assert rec.pieces == ["Checking. ", "Found ", "it."]
    assert result == "Found it."


# ---------------------------------------------------------------------------
# DoD-3 — reasoning blocks per round
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_streamed_tools_each_rounds_reasoning_is_its_own_closed_block__DoD3(monkeypatch):
    """DoD-3: each round's reasoning arrives as its own block, closed before the next round's pieces even when
    the round ends in calls only; the answer holds no reasoning or marker (US-027.AC-1, US-027.AC-2,
    US-058.AC-5, DEC-004-9)."""
    client = _client()
    rounds = [_llamacpp_tool_round(), _answer_round("Answer", reasoning=("Got ", "it."))]
    _install_stream_rounds(monkeypatch, client, rounds)
    rec = _Recorder()

    result = await _stream_tools(client, _Tools(), rec)

    assert rec.pieces == [f"{_OPEN}Need ", "the record.", _CLOSE, f"{_OPEN}Got ", "it.", f"{_CLOSE}Answer"]
    assert result == "Answer"
    assert _OPEN not in result and _CLOSE not in result


@pytest.mark.asyncio
async def test_llamaswap_streamed_tools_summary_deltas_are_reasoning__DoD3(monkeypatch):
    """DoD-3: reasoning-summary deltas reach the receiver as reasoning, one closed block per round, and never
    reach the returned answer (US-027.AC-1, DEC-004-9)."""
    client = _client()
    round_1 = [
        _CREATED,
        _reasoning_added(0, "rs_1"),
        _summary_delta("Plan ", 0, "rs_1"),
        _summary_delta("lookup.", 0, "rs_1"),
        _call_added(1, "fc_1", "call_1", "fetch_record"),
        _args_delta(_FULL_ARGS, 1, "fc_1"),
        _item_done(1, _CALL_1),
        _item_done(0, _R1),
        _completed(_R1, _CALL_1),
    ]
    round_2 = [
        _CREATED,
        _summary_delta("Wrap up.", 0, "rs_2"),
        _text_delta("Done", 1, "msg_2"),
        _completed(),
    ]
    _install_stream_rounds(monkeypatch, client, [round_1, round_2])
    rec = _Recorder()

    result = await _stream_tools(client, _Tools(), rec)

    assert rec.pieces == [f"{_OPEN}Plan ", "lookup.", _CLOSE, f"{_OPEN}Wrap up.", f"{_CLOSE}Done"]
    assert result == "Done"


@pytest.mark.asyncio
async def test_llamaswap_tool_round_parser_closes_open_block_at_completed_and_keeps_text_clean__DoD3(monkeypatch):
    """DoD-3: the parser closes a block still open at completion with its own piece; the round text it returns
    holds text deltas only, no reasoning and no marker (DEC-004-9)."""
    rec = _Recorder()
    payloads = [
        _CREATED,
        _reasoning_delta("think", 0, "rs_1"),
        _text_delta("Hi", 1, "msg_1"),
        _reasoning_delta("more", 2, "rs_2"),
        _completed(),
    ]

    text, calls, _reasoning = await _parse(monkeypatch, payloads, rec)

    assert rec.pieces == [f"{_OPEN}think", f"{_CLOSE}Hi", f"{_OPEN}more", _CLOSE]
    assert text == "Hi"
    assert calls == []


# ---------------------------------------------------------------------------
# DoD-4 — the llama.cpp sequence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_llamacpp_sequence_runs_tool_with_full_arguments_and_resends_reasoning__DoD4(monkeypatch):
    """DoD-4: the llama.cpp sequence (reasoning done after the call's deltas, no arguments-done, no [DONE])
    runs the tool with the full arguments, and the next round re-sends the reasoning item ahead of the call
    (DEC-004-9, DEC-004-7, US-058.AC-3)."""
    client = _client()
    calls = _install_stream_rounds(monkeypatch, client, [_llamacpp_tool_round(), _answer_round("The record says hi.")])
    tools = _Tools()

    result = await _stream_tools(client, tools, _Recorder())

    assert tools.log == [("fetch_record", {"key": "alpha-7"})]
    assert result == "The record says hi."
    assert len(calls) == 2
    assert calls[1]["payload"]["input"] == [
        {"role": "user", "content": _USER_TEXT},
        _R1,
        {"type": "function_call", "call_id": "call_1", "name": "fetch_record", "arguments": _FULL_ARGS},
        {"type": "function_call_output", "call_id": "call_1", "output": "record:alpha-7"},
    ]


@pytest.mark.asyncio
async def test_llamaswap_tool_round_parser_reads_llamacpp_sequence__DoD4(monkeypatch):
    """DoD-4: driven directly, the parser returns no round text, the one call with its name, call id and full
    arguments, the reasoning carry item as finished, and delivers the reasoning as one closed block (DEC-004-9)."""
    rec = _Recorder()

    text, calls, reasoning = await _parse(monkeypatch, _llamacpp_tool_round(), rec)

    assert text == ""
    assert len(calls) == 1
    assert calls[0]["call_id"] == "call_1"
    assert calls[0]["name"] == "fetch_record"
    assert calls[0]["arguments"] == _FULL_ARGS
    assert reasoning == [_R1]
    assert rec.pieces == [f"{_OPEN}Need ", "the record.", _CLOSE]


@pytest.mark.asyncio
async def test_llamaswap_tool_round_parser_assembles_arguments_from_deltas_alone__DoD4(monkeypatch):
    """DoD-4: with no done event for the call, the argument deltas alone assemble the full arguments
    (llama.cpp sends no arguments-done; DEC-004-9)."""
    payloads = [
        _CREATED,
        _call_added(1, "fc_1", "call_1", "fetch_record"),
        *[_args_delta(d, 1, "fc_1") for d in _ARG_DELTAS],
        _completed(),
    ]

    _text, calls, _reasoning = await _parse(monkeypatch, payloads)

    assert len(calls) == 1
    assert calls[0]["arguments"] == _FULL_ARGS


# ---------------------------------------------------------------------------
# DoD-5 — done events replace the assembled argument text
# ---------------------------------------------------------------------------


def _round_with_replacing_done(done_event: str) -> list[str]:
    """A round whose argument deltas assemble wrong text, then a done event carries the full arguments."""
    return [
        _CREATED,
        _call_added(1, "fc_1", "call_1", "fetch_record"),
        _args_delta('{"key": "wro', 1, "fc_1"),
        _args_delta('ng"}', 1, "fc_1"),
        done_event,
        _completed(),
    ]


_REPLACING_DONE_EVENTS = [
    pytest.param(_args_done(1, "fc_1", _FULL_ARGS), id="arguments-done"),
    pytest.param(_item_done(1, _CALL_1), id="output-item-done"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("done_event", _REPLACING_DONE_EVENTS)
async def test_llamaswap_tool_round_parser_done_event_replaces_assembled_arguments__DoD5(monkeypatch, done_event):
    """DoD-5: an arguments-done event or a function-call output-item-done event with full arguments replaces
    the assembled argument text (DEC-004-9)."""
    _text, calls, _reasoning = await _parse(monkeypatch, _round_with_replacing_done(done_event))

    assert len(calls) == 1
    assert calls[0]["arguments"] == _FULL_ARGS


@pytest.mark.asyncio
@pytest.mark.parametrize("done_event", _REPLACING_DONE_EVENTS)
async def test_llamaswap_streamed_tools_run_with_done_event_arguments__DoD5(monkeypatch, done_event):
    """DoD-5: through ``chat_with_tools`` the tool runs with the done event's arguments, not the deltas'
    (DEC-004-9)."""
    client = _client()
    _install_stream_rounds(monkeypatch, client, [_round_with_replacing_done(done_event), _answer_round("ok")])
    tools = _Tools()

    await _stream_tools(client, tools)

    assert tools.log == [("fetch_record", {"key": "alpha-7"})]


# ---------------------------------------------------------------------------
# DoD-6 — two calls at positions 1 and 2 behind reasoning at 0
# ---------------------------------------------------------------------------

_CALL_A = _call_item("fc_a", "call_a", "fetch_record", '{"key": "alpha-7"}')
_CALL_B = _call_item("fc_b", "call_b", "query_index", '{"term": "beta-3"}')
_R_TWO = _reasoning_item("rs_1", "Two lookups needed.")


def _two_call_round() -> list[str]:
    """Reasoning at 0, call A at 1, call B at 2 (separate, not interleaved), reasoning done last."""
    return [
        _CREATED,
        _IN_PROGRESS,
        _reasoning_added(0, "rs_1"),
        _reasoning_delta("Two lookups needed.", 0, "rs_1"),
        _call_added(1, "fc_a", "call_a", "fetch_record"),
        _args_delta('{"key": ', 1, "fc_a"),
        _args_delta('"alpha-7"}', 1, "fc_a"),
        _item_done(1, _CALL_A),
        _call_added(2, "fc_b", "call_b", "query_index"),
        _args_delta('{"term": ', 2, "fc_b"),
        _args_delta('"beta-3"}', 2, "fc_b"),
        _item_done(2, _CALL_B),
        _item_done(0, _R_TWO),
        _completed(_R_TWO, _CALL_A, _CALL_B),
    ]


@pytest.mark.asyncio
async def test_llamaswap_streamed_two_calls_run_in_order_with_own_call_ids__DoD6(monkeypatch):
    """DoD-6: two calls at output positions 1 and 2 behind reasoning at 0 run as two calls in order, each with
    its own name and arguments; their outputs carry their own call_id (US-035.AC-4, DEC-arch-10, DEC-004-9)."""
    client = _client()
    calls = _install_stream_rounds(monkeypatch, client, [_two_call_round(), _answer_round("Both found.")])
    tools = _Tools()

    result = await _stream_tools(client, tools)

    assert tools.log == [("fetch_record", {"key": "alpha-7"}), ("query_index", {"term": "beta-3"})]
    assert result == "Both found."
    assert calls[1]["payload"]["input"] == [
        {"role": "user", "content": _USER_TEXT},
        _R_TWO,
        {"type": "function_call", "call_id": "call_a", "name": "fetch_record", "arguments": '{"key": "alpha-7"}'},
        {"type": "function_call", "call_id": "call_b", "name": "query_index", "arguments": '{"term": "beta-3"}'},
        {"type": "function_call_output", "call_id": "call_a", "output": "record:alpha-7"},
        {"type": "function_call_output", "call_id": "call_b", "output": "index:beta-3"},
    ]


@pytest.mark.asyncio
async def test_llamaswap_tool_round_parser_returns_two_dense_calls_in_order__DoD6(monkeypatch):
    """DoD-6: driven directly, the parser returns exactly two calls in order of appearance, each with its own
    call id, name and arguments (DEC-arch-10, DEC-004-9)."""
    _text, calls, reasoning = await _parse(monkeypatch, _two_call_round())

    assert [(c["call_id"], c["name"], c["arguments"]) for c in calls] == [
        ("call_a", "fetch_record", '{"key": "alpha-7"}'),
        ("call_b", "query_index", '{"term": "beta-3"}'),
    ]
    assert reasoning == [_R_TWO]


# ---------------------------------------------------------------------------
# DoD-7 — reasoning carried verbatim to every later round, ordered by position, same layout as non-streamed
# ---------------------------------------------------------------------------

_RA1 = _reasoning_item("rs_a1", "First thought.")
_RA2 = _reasoning_item("rs_a2", "Second thought.")
_RB = _reasoning_item("rs_b", "Now the other one.")
_RC = _reasoning_item("rs_c", "Final thought.")
_CALL_R1 = _call_item("fc_1", "call_1", "fetch_record", '{"key": "alpha-7"}')
_CALL_R2 = _call_item("fc_2", "call_2", "fetch_record", '{"key": "beta-3"}')


def _three_round_stream() -> list[list[str]]:
    """Three streamed rounds; in round 1 two reasoning items (positions 0 and 1) finish in reverse order."""
    round_1 = [
        _CREATED,
        _reasoning_added(0, "rs_a1"),
        _reasoning_delta("First thought.", 0, "rs_a1"),
        _reasoning_added(1, "rs_a2"),
        _reasoning_delta("Second thought.", 1, "rs_a2"),
        _call_added(2, "fc_1", "call_1", "fetch_record"),
        _args_delta('{"key": "alpha-7"}', 2, "fc_1"),
        _item_done(2, _CALL_R1),
        _item_done(1, _RA2),
        _item_done(0, _RA1),
        _completed(_RA1, _RA2, _CALL_R1),
    ]
    round_2 = [
        _CREATED,
        _reasoning_added(0, "rs_b"),
        _reasoning_delta("Now the other one.", 0, "rs_b"),
        _call_added(1, "fc_2", "call_2", "fetch_record"),
        _args_delta('{"key": "beta-3"}', 1, "fc_2"),
        _item_done(1, _CALL_R2),
        _item_done(0, _RB),
        _completed(_RB, _CALL_R2),
    ]
    round_3 = _answer_round("All done.", reasoning=("Final thought.",), reasoning_id="rs_c")
    return [round_1, round_2, round_3]


def _three_round_replies() -> list[dict[str, Any]]:
    """The same three rounds as non-streamed replies (reasoning items as the done events carry them)."""
    return [
        _reply(_RA1, _RA2, _CALL_R1),
        _reply(_RB, _CALL_R2),
        _reply(_reasoning_item("rs_c", "Final thought."), _message_item("msg_9", "All done.")),
    ]


@pytest.mark.asyncio
async def test_llamaswap_streamed_reasoning_carried_to_every_later_round_ordered_by_position__DoD7(monkeypatch):
    """DoD-7: reasoning items from a round's done events are re-sent verbatim on every later round, ahead of
    that round's calls and ordered by output position (US-058.AC-3, DEC-004-7)."""
    client = _client()
    calls = _install_stream_rounds(monkeypatch, client, _three_round_stream())

    result = await _stream_tools(client, _Tools())

    user = {"role": "user", "content": _USER_TEXT}
    fc_1 = {"type": "function_call", "call_id": "call_1", "name": "fetch_record", "arguments": '{"key": "alpha-7"}'}
    out_1 = {"type": "function_call_output", "call_id": "call_1", "output": "record:alpha-7"}
    fc_2 = {"type": "function_call", "call_id": "call_2", "name": "fetch_record", "arguments": '{"key": "beta-3"}'}
    out_2 = {"type": "function_call_output", "call_id": "call_2", "output": "record:beta-3"}

    assert result == "All done."
    assert len(calls) == 3
    assert calls[0]["payload"]["input"] == [user]
    assert calls[1]["payload"]["input"] == [user, _RA1, _RA2, fc_1, out_1]
    assert calls[2]["payload"]["input"] == [user, _RA1, _RA2, fc_1, out_1, _RB, fc_2, out_2]


@pytest.mark.asyncio
async def test_llamaswap_streamed_round_input_matches_nonstreamed_layout__DoD7(monkeypatch):
    """DoD-7: for the same rounds, the streamed path's re-sent input has the same layout as the non-streamed
    path's (US-058.AC-3, DEC-004-7)."""
    plain = _client()
    requests = _install_request_rounds(monkeypatch, plain, _three_round_replies())
    plain_result = await plain.chat_with_tools(
        copy.deepcopy(_MESSAGES), tools_definitions=copy.deepcopy(_TOOL_DEFS), tools=_Tools().mapping()
    )

    streamed = _client()
    calls = _install_stream_rounds(monkeypatch, streamed, _three_round_stream())
    streamed_result = await _stream_tools(streamed, _Tools())

    assert [c["payload"]["input"] for c in calls] == [r["json"]["input"] for r in requests]
    assert streamed_result == plain_result


@pytest.mark.asyncio
async def test_llamaswap_tool_round_parser_orders_carry_items_by_output_position__DoD7(monkeypatch):
    """DoD-7: driven directly, the parser returns carry items verbatim and ordered by output position even when
    they finish in reverse order (DEC-004-9, DEC-004-7)."""
    _text, _calls, reasoning = await _parse(monkeypatch, _three_round_stream()[0])

    assert reasoning == [_RA1, _RA2]


# ---------------------------------------------------------------------------
# DoD-8 — no tool-call text reaches the receiver
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_streamed_tools_never_deliver_call_names_or_arguments__DoD8(monkeypatch):
    """DoD-8: no tool-call name or argument text reaches the receiver; it gets text and reasoning only
    (DEC-004-9)."""
    client = _client()
    _install_stream_rounds(monkeypatch, client, [_two_call_round(), _answer_round("Both found.")])
    rec = _Recorder()

    await _stream_tools(client, _Tools(), rec)

    assert rec.pieces == [f"{_OPEN}Two lookups needed.", _CLOSE, "Both found."]
    for piece in rec.pieces:
        for fragment in ("fetch_record", "query_index", "alpha-7", "beta-3", '"key"', '"term"', "call_a", "call_b"):
            assert fragment not in piece


@pytest.mark.asyncio
async def test_llamaswap_tool_round_parser_delivers_no_call_fragments__DoD8(monkeypatch):
    """DoD-8: driven directly on a call-only round, the parser hands the receiver no piece at all (DEC-004-9)."""
    rec = _Recorder()
    payloads = [
        _CREATED,
        _call_added(0, "fc_1", "call_1", "fetch_record"),
        *[_args_delta(d, 0, "fc_1") for d in _ARG_DELTAS],
        _item_done(0, _CALL_1),
        _completed(_CALL_1),
    ]

    await _parse(monkeypatch, payloads, rec)

    assert rec.pieces == []


# ---------------------------------------------------------------------------
# DoD-9 — failure events and early end in any round
# ---------------------------------------------------------------------------

_FAILURE_EVENTS = [
    pytest.param(
        _ev("response.failed", response={"status": "failed", "error": {"code": "server", "message": "boom-xyz"}}),
        "boom-xyz",
        id="failed",
    ),
    pytest.param(
        _ev("response.incomplete", response={"status": "incomplete", "incomplete_details": {"reason": "cut-r9"}}),
        "cut-r9",
        id="incomplete",
    ),
    pytest.param(_ev("error", code="rate_limit", message="slow-down-z2"), "slow-down-z2", id="error"),
]

_ROUND_POSITIONS = [pytest.param(0, id="first-round"), pytest.param(1, id="later-round")]


def _broken_round(ending: list[str]) -> list[str]:
    """A round with a complete query_index call and one text piece, then ``ending`` instead of a normal close."""
    call = _call_item("fc_q", "call_q", "query_index", '{"term": "beta-3"}')
    return [
        _CREATED,
        _call_added(1, "fc_q", "call_q", "query_index"),
        _args_delta('{"term": "beta-3"}', 1, "fc_q"),
        _item_done(1, call),
        _text_delta("before", 2, "msg_1"),
        *ending,
    ]


def _rounds_with_broken(position: int, broken: list[str]) -> list[list[str]]:
    """Put the broken round first, or after one good fetch_record round."""
    good = _call_round("fc_1", "call_1", "fetch_record", _FULL_ARGS)
    return [broken] if position == 0 else [good, broken]


@pytest.mark.asyncio
@pytest.mark.parametrize("position", _ROUND_POSITIONS)
@pytest.mark.parametrize(("event", "reason"), _FAILURE_EVENTS)
async def test_llamaswap_streamed_tools_failure_event_ends_call_with_reason__DoD9(monkeypatch, event, reason, position):
    """DoD-9: a failed, incomplete or error event in any round ends the call with LLMError carrying the server's
    reason; no tool of that round runs and no later piece is delivered (US-060.AC-2, US-024.AC-2, DEC-004-9)."""
    client = _client()
    broken = _broken_round([event, _text_delta("after", 2, "msg_1"), _completed()])
    calls = _install_stream_rounds(monkeypatch, client, _rounds_with_broken(position, broken))
    tools = _Tools()
    rec = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await _stream_tools(client, tools, rec)

    assert reason in str(excinfo.value)
    assert all(name != "query_index" for name, _args in tools.log)
    assert rec.pieces == ["before"]
    assert len(calls) == position + 1


_EARLY_ENDINGS = [pytest.param([], id="no-completed"), pytest.param(["[DONE]"], id="done-sentinel")]


@pytest.mark.asyncio
@pytest.mark.parametrize("position", _ROUND_POSITIONS)
@pytest.mark.parametrize("ending", _EARLY_ENDINGS)
async def test_llamaswap_streamed_tools_round_without_completed_ends_with_llmerror__DoD9(monkeypatch, ending, position):
    """DoD-9: a round stream that ends without the completed event, including one with a ``[DONE]`` sentinel,
    ends the call with LLMError; no tool of that round runs (US-060.AC-2, US-024.AC-2, DEC-004-9)."""
    client = _client()
    calls = _install_stream_rounds(monkeypatch, client, _rounds_with_broken(position, _broken_round(ending)))
    tools = _Tools()

    with pytest.raises(LLMError):
        await _stream_tools(client, tools, _Recorder())

    assert all(name != "query_index" for name, _args in tools.log)
    assert len(calls) == position + 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("event", "reason"), _FAILURE_EVENTS)
async def test_llamaswap_tool_round_parser_failure_event_raises_and_stops_delivery__DoD9(monkeypatch, event, reason):
    """DoD-9: driven directly, a failure event raises LLMError with the server's reason and nothing after it
    is delivered (DEC-004-9)."""
    rec = _Recorder()
    with pytest.raises(LLMError) as excinfo:
        await _parse(monkeypatch, [_text_delta("before"), event, _text_delta("after"), _completed()], rec)

    assert reason in str(excinfo.value)
    assert rec.pieces == ["before"]


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", _EARLY_ENDINGS)
async def test_llamaswap_tool_round_parser_end_without_completed_raises__DoD9(monkeypatch, ending):
    """DoD-9: driven directly, a stream ending without the completed event (``[DONE]`` does not count) raises
    LLMError (DEC-004-9)."""
    with pytest.raises(LLMError):
        await _parse(monkeypatch, [_text_delta("partial"), *ending], _Recorder())


# ---------------------------------------------------------------------------
# DoD-10 — non-2xx on a later streamed round
# ---------------------------------------------------------------------------


class _FakeContent:
    """Stand-in for ``aiohttp.StreamReader`` serving the body in small chunks."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    async def iter_chunked(self, n: int):
        size = max(1, min(n, 7))
        for i in range(0, len(self._body), size):
            yield self._body[i : i + size]

    async def iter_any(self):
        yield self._body

    async def read(self, n: int = -1) -> bytes:
        return self._body


class _FakeResponse:
    """Minimal ``aiohttp.ClientResponse`` stand-in: status, headers, body reads, status check, release."""

    def __init__(self, url: str, status: int, body: bytes, content_type: str) -> None:
        self.url = URL(url)
        self.real_url = self.url
        self.status = status
        self.reason = "OK" if status < 400 else "Not Found"
        self.headers = {"Content-Type": content_type}
        self.content_type = content_type
        self.content = _FakeContent(body)
        self.history = ()
        self.closed = False
        self._body = body
        # ClientResponseError only needs ``real_url`` from the request info for its text.
        self.request_info = SimpleNamespace(url=self.url, real_url=self.url, method="POST", headers={})

    @property
    def ok(self) -> bool:
        return self.status < 400

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                self.request_info, self.history, status=self.status, message=self.reason, headers=self.headers
            )

    async def text(self, encoding: str | None = None, errors: str = "strict") -> str:
        return self._body.decode(encoding or "utf-8", errors)

    async def read(self) -> bytes:
        return self._body

    async def json(self, **kw: Any) -> Any:
        return json.loads(self._body)

    def release(self) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True


class _FakeRequestContext:
    """Mimics aiohttp's request context manager: usable with ``async with`` and with ``await``."""

    def __init__(self, open_fn) -> None:
        self._open_fn = open_fn
        self._resp: _FakeResponse | None = None

    async def _open(self) -> _FakeResponse:
        self._resp = self._open_fn()
        return self._resp

    def __await__(self):
        return self._open().__await__()

    async def __aenter__(self) -> _FakeResponse:
        return await self._open()

    async def __aexit__(self, *exc: object) -> bool:
        if self._resp is not None:
            self._resp.release()
        return False


class _FakeSession:
    """Fake ``aiohttp.ClientSession`` assigned to ``client.session``; every request gets the same reply."""

    def __init__(self, *, status: int, body: bytes, content_type: str) -> None:
        self.status = status
        self.body = body
        self.content_type = content_type
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def request(self, method: str, url: Any, **kw: Any) -> _FakeRequestContext:
        self.calls.append({"method": method.upper(), "url": str(url), **kw})
        return _FakeRequestContext(lambda: _FakeResponse(str(url), self.status, self.body, self.content_type))

    def post(self, url: Any, **kw: Any) -> _FakeRequestContext:
        return self.request("POST", url, **kw)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_llamaswap_streamed_tools_later_round_404_raises_llmerror_with_address_and_reply__DoD10(monkeypatch):
    """DoD-10: a non-2xx reply on a later streamed round (a 404 with a JSON error body) ends with LLMError
    carrying the address and the server's reply, never the credential (US-060.AC-1, DEC-arch-5)."""
    client = _client(bearer_token=_SECRET)
    body = json.dumps({"error": {"code": 404, "message": "model-gone-q8 not found", "type": "not_found_error"}})
    session = _FakeSession(status=404, body=body.encode("utf-8"), content_type="application/json")
    client.session = session
    _fake_sse_rounds(monkeypatch, client, [_llamacpp_tool_round()])

    # Round 1 is served by a placeholder response; later rounds go through the real streamed-request helper,
    # which reaches the fake session and gets the 404.
    real_stream_post = client._stream_post
    paths: list[str] = []

    @asynccontextmanager
    async def first_round_faked(path, payload):
        paths.append(path)
        if len(paths) == 1:
            yield None
        else:
            async with real_stream_post(path, payload) as response:
                yield response

    monkeypatch.setattr(client, "_stream_post", first_round_faked)
    tools = _Tools()

    with pytest.raises(LLMError) as excinfo:
        await _stream_tools(client, tools, _Recorder())

    err = excinfo.value
    rendered = str(err)
    assert _RESPONSES_URL in rendered
    assert "model-gone-q8 not found" in rendered
    assert "404" in rendered
    assert _SECRET not in f"{rendered} {err.message} {err.description} {err.args!r}"
    assert tools.log == [("fetch_record", {"key": "alpha-7"})]
    assert paths == ["/responses", "/responses"]
    assert len(session.calls) == 1


# ---------------------------------------------------------------------------
# DoD-11 — receiver rules
# ---------------------------------------------------------------------------


class _ReceiverBoom(Exception):
    """A receiver-specific failure, distinct from any library error."""


@pytest.mark.asyncio
async def test_llamaswap_streamed_tools_receiver_failure_propagates_unchanged__DoD11(monkeypatch):
    """DoD-11: a receiver failure ends the call with that same exception and no later piece is delivered
    (US-022.AC-1, DEC-000-23)."""
    client = _client()
    calls = _install_stream_rounds(
        monkeypatch, client, [_call_round("fc_1", "call_1", "fetch_record", _FULL_ARGS), _answer_round("a", "b", "c")]
    )
    seen: list[str] = []
    boom = _ReceiverBoom("receiver broke")

    async def receiver(piece: str) -> None:
        seen.append(piece)
        if piece == "b":
            raise boom

    with pytest.raises(_ReceiverBoom) as excinfo:
        await _stream_tools(client, _Tools(), receiver)

    assert excinfo.value is boom
    assert seen == ["a", "b"]
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_llamaswap_streamed_tools_receiver_failure_in_tool_round_ends_call__DoD11(monkeypatch):
    """DoD-11: a receiver failing on a tool round's reasoning ends the call with that exception; nothing later
    is delivered and no later round is requested (US-022.AC-1, DEC-000-23)."""
    client = _client()
    calls = _install_stream_rounds(monkeypatch, client, [_llamacpp_tool_round(), _answer_round("never")])
    seen: list[str] = []
    boom = _ReceiverBoom("receiver broke early")

    async def receiver(piece: str) -> None:
        seen.append(piece)
        raise boom

    with pytest.raises(_ReceiverBoom) as excinfo:
        await _stream_tools(client, _Tools(), receiver)

    assert excinfo.value is boom
    assert seen == [f"{_OPEN}Need "]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_llamaswap_streamed_tools_async_receiver_is_awaited__DoD11(monkeypatch):
    """DoD-11: an async receiver is awaited; it records only after yielding control, so the full ordered piece
    list proves each call was awaited (US-022.AC-2, DEC-000-23)."""
    client = _client()
    _install_stream_rounds(monkeypatch, client, [_llamacpp_tool_round(), _answer_round("x", "y")])
    seen: list[str] = []

    async def receiver(piece: str) -> None:
        await asyncio.sleep(0)
        seen.append(piece)

    result = await _stream_tools(client, _Tools(), receiver)

    assert seen == [f"{_OPEN}Need ", "the record.", _CLOSE, "x", "y"]
    assert result == "xy"


# ---------------------------------------------------------------------------
# DoD-12 — structured final answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_streamed_tools_with_format_return_final_round_text_pieces_joined__DoD12(monkeypatch):
    """DoD-12: with a requested format, the final round's text pieces join to the returned structured answer
    (US-037.AC-4)."""
    client = _client()
    final_pieces = ['{"record": ', '"record:alpha-7"', "}"]
    _install_stream_rounds(monkeypatch, client, [_llamacpp_tool_round(), _answer_round(*final_pieces)])
    rec = _Recorder()

    result = await _stream_tools(client, _Tools(), rec, response_format={"type": "json_object"})

    assert result == "".join(final_pieces)
    assert json.loads(result) == {"record": "record:alpha-7"}
    assert rec.pieces[-len(final_pieces) :] == final_pieces


# ---------------------------------------------------------------------------
# DoD-13 — no receiver, and tracing parity
# ---------------------------------------------------------------------------


def _parity_rounds() -> list[list[str]]:
    """Two rounds with reasoning in both, used to compare runs with and without a receiver or tracing."""
    return [_llamacpp_tool_round(), _answer_round("Final ", "answer.", reasoning=("Got ", "it."))]


@pytest.mark.asyncio
async def test_llamaswap_streamed_tools_without_receiver_return_same_answer__DoD13(monkeypatch):
    """DoD-13: a streamed tool call without a receiver returns the same answer as with one (US-023.AC-2)."""
    with_rec = _client()
    _install_stream_rounds(monkeypatch, with_rec, _parity_rounds())
    with_result = await _stream_tools(with_rec, _Tools(), _Recorder())

    without = _client()
    _install_stream_rounds(monkeypatch, without, _parity_rounds())
    tools = _Tools()
    without_result = await _stream_tools(without, tools, None)

    assert without_result == with_result == "Final answer."
    assert tools.log == [("fetch_record", {"key": "alpha-7"})]


@pytest.mark.asyncio
async def test_llamaswap_streamed_tools_tracing_keeps_pieces_and_answer__DoD13(monkeypatch, tmp_path):
    """DoD-13: with tracing enabled the receiver gets the same pieces and the call returns the same answer as
    without tracing (US-050.AC-4, DEC-004-10)."""
    untraced = _client()
    _install_stream_rounds(monkeypatch, untraced, _parity_rounds())
    untraced_rec = _Recorder()
    untraced_result = await _stream_tools(untraced, _Tools(), untraced_rec)

    # Tracing is read when the client is built, so the env is set before the traced client exists.
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(tmp_path))
    traced = _client()
    _install_stream_rounds(monkeypatch, traced, _parity_rounds())
    traced_rec = _Recorder()
    traced_result = await _stream_tools(traced, _Tools(), traced_rec)

    assert traced_rec.pieces == untraced_rec.pieces
    assert traced_result == untraced_result == "Final answer."


@pytest.mark.asyncio
async def test_llamaswap_streamed_tools_tracing_without_receiver_returns_same_answer__DoD13(monkeypatch, tmp_path):
    """DoD-13: with tracing enabled and no receiver, the streamed tool call returns the same answer
    (US-050.AC-4, DEC-004-10)."""
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(tmp_path))
    client = _client()
    _install_stream_rounds(monkeypatch, client, _parity_rounds())

    result = await _stream_tools(client, _Tools(), None)

    assert result == "Final answer."
