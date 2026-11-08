"""Unit tests for streamed tool conversations on the OpenAI Responses class (DEC-004-9).

Covers the tool-round stream parser ``OpenAIAPIClient._read_responses_tool_stream`` and the streamed branch of
``chat_with_tools``: one streamed ``/responses`` request per round whose body is the non-streamed round body with
``stream`` true (DEC-004-10), text and reasoning pieces with fresh per-round ``<think>`` / ``</think>`` markers,
call assembly from tool-call events, reasoning carry items ordered by output position (DEC-004-7, DEC-004-9),
failures inside rounds, receiver rules (DEC-000-23), requested formats and tracing. Fake streams reproduce both
the OpenAI event sequence and the llama.cpp sequence seen in a probe (DEC-004-9).
Every expected value comes from the step's Definition of done and the cited decisions.
"""

import asyncio
import copy
import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest
from multidict import CIMultiDict
from yarl import URL

from llm.exceptions import LLMError
from llm.openai import OpenAIAPIClient

BASE_URL = "https://api.example.test/v1"
RESPONSES_URL = f"{BASE_URL}/responses"
CREDENTIAL = "sk-test"
MODEL = "gpt-6-luna"

PROMPT = "Look up record A7 and tell me what it holds."
MESSAGES = [{"role": "user", "content": PROMPT}]
SYSTEM = "You use the tools you are given."
# Thinking on, so the reasoning setting is part of every round's body (DEC-003-9, DEC-004-10).
OPTIONS = {"temperature": 0.25, "max_tokens": 512, "enable_thinking": True}

# Both stream dialects (DEC-004-9, DEC-003-13; llama.cpp order from a probe): OpenAI (summary deltas,
# arguments-done events, encrypted reasoning) and llama.cpp (reasoning-text deltas, no arguments-done,
# reasoning finishing after the other items).
FLAVOURS = [
    pytest.param("openai", id="openai-sequence"),
    pytest.param("llamacpp", id="llamacpp-sequence"),
]

ARGS_A7 = ['{"key": ', '"A7"}']
ARGS_A7_TEXT = '{"key": "A7"}'


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _tool_def(name: str, arg: str) -> dict[str, Any]:
    """An app tool definition in today's nested chat-completions shape (DEC-004-5)."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Tool {name}.",
            "parameters": {"type": "object", "properties": {arg: {"type": "string"}}, "required": [arg]},
        },
    }


TOOL_DEFS = [_tool_def("fetch_record", "key"), _tool_def("count_items", "category")]


def _record_result(key: str) -> str:
    """The text ``fetch_record`` returns for ``key``; a plain string is sent as given (DEC-002-5)."""
    return f"record {key}: payload-{key}"


def _count_result(category: str) -> str:
    """The text ``count_items`` returns for ``category``."""
    return f"{category}: 3 items"


class _ToolLog:
    """Both advertised tools, recording every invocation in order as ``(name, kwargs)`` so a test can see which
    tools ran, in which order and with which arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def fetch_record(self, key: str) -> str:
        self.calls.append(("fetch_record", {"key": key}))
        return _record_result(key)

    def count_items(self, category: str) -> str:
        self.calls.append(("count_items", {"category": category}))
        return _count_result(category)

    def callables(self) -> dict[str, Callable[..., Any]]:
        # Every advertised tool needs a callable, else the pre-flight check refuses the call (DEC-004-11).
        return {"fetch_record": self.fetch_record, "count_items": self.count_items}


# ---------------------------------------------------------------------------
# Stream event builders (Responses tool-round vocabulary: DEC-004-9, DEC-003-13)
# ---------------------------------------------------------------------------


def _ev(type_: str, **fields: Any) -> str:
    """One SSE data payload: the JSON text of a Responses stream event."""
    return json.dumps({"type": type_, **fields})


CREATED = _ev("response.created", response={"id": "resp_s", "status": "in_progress"})
IN_PROGRESS = _ev("response.in_progress", response={"id": "resp_s", "status": "in_progress"})
COMPLETED = _ev("response.completed", response={"id": "resp_s", "status": "completed"})

_FAILED = {
    "type": "response.failed",
    "response": {"id": "resp_s", "status": "failed", "error": {"code": "server_error", "message": "Model crashed"}},
}
_INCOMPLETE = {
    "type": "response.incomplete",
    "response": {"id": "resp_s", "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
}
_ERROR = {"type": "error", "code": "rate_limit_exceeded", "message": "Rate limit reached"}

FAILURES = [
    pytest.param(_FAILED, "Model crashed", id="failed"),
    pytest.param(_INCOMPLETE, "max_output_tokens", id="incomplete"),
    pytest.param(_ERROR, "Rate limit reached", id="error"),
]


def _added(index: int, item: dict[str, Any]) -> str:
    return _ev("response.output_item.added", output_index=index, item=item)


def _done(index: int, item: dict[str, Any]) -> str:
    return _ev("response.output_item.done", output_index=index, item=item)


def _text_delta(index: int, item_id: str, delta: str) -> str:
    return _ev("response.output_text.delta", item_id=item_id, output_index=index, content_index=0, delta=delta)


def _reasoning_delta(flavour: str, index: int, item_id: str, delta: str) -> str:
    """OpenAI streams reasoning as summary deltas (DEC-003-13); llama.cpp as reasoning-text deltas."""
    if flavour == "openai":
        return _ev(
            "response.reasoning_summary_text.delta", item_id=item_id, output_index=index, summary_index=0, delta=delta
        )
    return _ev("response.reasoning_text.delta", item_id=item_id, output_index=index, content_index=0, delta=delta)


def _args_delta(index: int, item_id: str, delta: str) -> str:
    return _ev("response.function_call_arguments.delta", item_id=item_id, output_index=index, delta=delta)


def _args_done(index: int, item_id: str, arguments: str) -> str:
    return _ev("response.function_call_arguments.done", item_id=item_id, output_index=index, arguments=arguments)


def _reasoning_item(flavour: str, item_id: str, text: str) -> dict[str, Any]:
    """A finished reasoning item: OpenAI fills summary and encrypted content; llama.cpp has reasoning-text content,
    an empty summary, empty encrypted content and no status (wire vocabulary)."""
    if flavour == "openai":
        return {
            "type": "reasoning",
            "id": item_id,
            "summary": [{"type": "summary_text", "text": text}],
            "encrypted_content": f"enc-{item_id}-opaque",
        }
    return {
        "type": "reasoning",
        "id": item_id,
        "content": [{"type": "reasoning_text", "text": text}],
        "summary": [],
        "encrypted_content": "",
    }


def _function_call_item(item_id: str, call_id: str, name: str, arguments: str, status: str = "completed") -> dict:
    """A function-call output item: server item id (``fc_…``), ``call_id`` (``call_…``), name, JSON-text
    arguments and status (wire vocabulary)."""
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": status,
    }


def _message_item(item_id: str, text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": item_id,
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _reasoning_part(flavour: str, index: int, item_id: str, chunks: list[str]) -> tuple[list[str], str, dict]:
    """Events of one reasoning item: ``(added + deltas, done event, finished item)``."""
    item = _reasoning_item(flavour, item_id, "".join(chunks))
    opening = [_added(index, {"type": "reasoning", "id": item_id, "summary": []})]
    if flavour == "openai":
        # An event type the tool-round parser ignores (DEC-004-9: every other type is ignored).
        opening.append(
            _ev("response.reasoning_summary_part.added", item_id=item_id, output_index=index, summary_index=0,
                part={"type": "summary_text", "text": ""})
        )
    opening += [_reasoning_delta(flavour, index, item_id, c) for c in chunks]
    return opening, _done(index, item), item


def _message_part(index: int, item_id: str, chunks: list[str]) -> tuple[list[str], str, dict]:
    """Events of one assistant message item: ``(added + deltas + ignorable done events, done event, item)``."""
    text = "".join(chunks)
    item = _message_item(item_id, text)
    opening = [
        _added(index, {"type": "message", "id": item_id, "role": "assistant", "status": "in_progress", "content": []}),
        _ev("response.content_part.added", item_id=item_id, output_index=index, content_index=0,
            part={"type": "output_text", "text": ""}),
        *[_text_delta(index, item_id, c) for c in chunks],
        _ev("response.output_text.done", item_id=item_id, output_index=index, content_index=0, text=text),
    ]
    return opening, _done(index, item), item


def _call_part(flavour: str, index: int, item_id: str, call_id: str, name: str, chunks: list[str]):
    """Events of one function-call item: added (with name and call id, empty arguments), argument deltas, on
    OpenAI an arguments-done event; then the done event with the full arguments (wire vocabulary)."""
    arguments = "".join(chunks)
    opening = [_added(index, _function_call_item(item_id, call_id, name, "", status="in_progress"))]
    opening += [_args_delta(index, item_id, c) for c in chunks]
    if flavour == "openai":
        opening.append(_args_done(index, item_id, arguments))
    item = _function_call_item(item_id, call_id, name, arguments)
    return opening, _done(index, item), item


def _round(
    flavour: str,
    tag: str,
    *,
    reasoning: list[str] | None = None,
    text: list[str] | None = None,
    calls: list[tuple[str, str, list[str]]] = (),
    late_reasoning_done: bool | None = None,
    complete: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """One round's stream events and the equivalent non-streamed reply.

    Items get output positions in order: reasoning (if any), message (if any text), then each call
    ``(call_id, name, argument chunks)``. With ``late_reasoning_done`` (default: on for llama.cpp) the reasoning
    item's done event is sent after every other item finished, as observed in the probe.
    """
    if late_reasoning_done is None:
        late_reasoning_done = flavour == "llamacpp"
    events = [CREATED, IN_PROGRESS]
    deferred: list[str] = []
    items: list[dict[str, Any]] = []
    index = 0
    if reasoning:
        opening, done, item = _reasoning_part(flavour, index, f"rs_{tag}", reasoning)
        events += opening
        (deferred if late_reasoning_done else events).append(done)
        items.append(item)
        index += 1
    if text:
        opening, done, item = _message_part(index, f"msg_{tag}", text)
        events += [*opening, done]
        items.append(item)
        index += 1
    for n, (call_id, name, chunks) in enumerate(calls):
        opening, done, item = _call_part(flavour, index, f"fc_{tag}_{n}", call_id, name, chunks)
        events += [*opening, done]
        items.append(item)
        index += 1
    events += deferred
    if complete:
        events.append(COMPLETED)
    reply = {"id": f"resp_{tag}", "object": "response", "status": "completed", "output": items}
    return events, reply


def _of_type(reply: dict[str, Any], item_type: str) -> list[dict[str, Any]]:
    """The reply's output items of one type, in output order."""
    return [item for item in reply["output"] if item["type"] == item_type]


def _fc_input(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    """The function-call input item a later round must carry (DEC-004-6)."""
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}


def _fco_input(call_id: str, output: str) -> dict[str, Any]:
    """The function-call-output input item a later round must carry (DEC-004-6)."""
    return {"type": "function_call_output", "call_id": call_id, "output": output}


USER_ITEM = {"role": "user", "content": PROMPT}


# ---------------------------------------------------------------------------
# Client, receiver and seam fakes (DEC-004-14)
# ---------------------------------------------------------------------------


class _Recorder:
    """Async receiver that records every piece it is given, in order."""

    def __init__(self) -> None:
        self.pieces: list[str] = []

    async def __call__(self, piece: str) -> None:
        self.pieces.append(piece)


class _ReceiverBoom(Exception):
    """A receiver-side failure distinct from any library error."""


def _client() -> OpenAIAPIClient:
    """A Responses-class client built directly with an explicit test credential (tests/AGENTS.md)."""
    return OpenAIAPIClient(model=MODEL, base_url=BASE_URL, bearer_token=CREDENTIAL)


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch):
    """Clear the trace switches so every test starts untraced regardless of the developer's environment; the
    tracing test (DoD-12) turns them back on with ``monkeypatch.setenv`` before building its client."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _install_sse_rounds(monkeypatch, client: OpenAIAPIClient, rounds: list[list[str]], log: list[str] | None = None):
    """Replace ``_iter_sse_data`` with a per-round queue (DEC-004-14): each time a round's stream is read, the next
    canned event list is yielded. A read past the last canned round fails with ``IndexError``.

    When ``log`` is given, ``"read"`` is appended each time the parser asks for the next payload, so a test can see
    whether a receiver call finished before the next read (DEC-000-23).
    """
    pending = [list(events) for events in rounds]

    async def fake_iter_sse_data(_response):
        events = pending.pop(0)
        for payload in events:
            if log is not None:
                log.append("read")
            yield payload
        if log is not None:
            log.append("read")

    monkeypatch.setattr(client, "_iter_sse_data", fake_iter_sse_data)


def _install_stream_rounds(monkeypatch, client: OpenAIAPIClient, rounds: list[list[str]]) -> list[dict]:
    """Replace ``_stream_post`` with a fake async context manager recording ``path`` and a deep copy of each body
    (against aliasing), and ``_iter_sse_data`` with the per-round queue; returns the recorded calls."""
    calls: list[dict] = []

    @asynccontextmanager
    async def fake_stream_post(path, payload):
        calls.append({"path": path, "payload": copy.deepcopy(payload)})
        yield SimpleNamespace(round=len(calls))

    monkeypatch.setattr(client, "_stream_post", fake_stream_post)
    _install_sse_rounds(monkeypatch, client, rounds)
    return calls


def _install_request_rounds(monkeypatch, client: OpenAIAPIClient, replies: list[dict]) -> list[dict]:
    """Replace ``_request`` with a fake answering one canned reply per call and recording each body (deep copy);
    used to obtain the non-streamed bodies a streamed run is compared against (DEC-004-14)."""
    pending = [copy.deepcopy(reply) for reply in replies]
    records: list[dict] = []

    async def fake_request(method, path, **kw):
        records.append({"method": method, "path": path, "json": copy.deepcopy(kw.get("json"))})
        return pending.pop(0)

    monkeypatch.setattr(client, "_request", fake_request)
    return records


def _forbid_request(monkeypatch, client: OpenAIAPIClient) -> list[dict]:
    """Record any non-streamed request; a streamed tool call must send none (DoD-1)."""
    records: list[dict] = []

    async def fake_request(method, path, **kw):
        records.append({"method": method, "path": path})
        raise AssertionError("a streamed tool round must not use the non-streamed request path")

    monkeypatch.setattr(client, "_request", fake_request)
    return records


async def _streamed(client: OpenAIAPIClient, tools: _ToolLog, on_delta=None, **kw: Any) -> str:
    """Run a streamed ``chat_with_tools`` over a fresh copy of the caller data."""
    return await client.chat_with_tools(
        list(MESSAGES),
        tools_definitions=copy.deepcopy(TOOL_DEFS),
        tools=tools.callables(),
        stream=True,
        on_delta=on_delta,
        **kw,
    )


async def _non_streamed(client: OpenAIAPIClient, tools: _ToolLog, **kw: Any) -> str:
    """Run a non-streamed ``chat_with_tools`` over a fresh copy of the caller data."""
    return await client.chat_with_tools(
        list(MESSAGES), tools_definitions=copy.deepcopy(TOOL_DEFS), tools=tools.callables(), **kw
    )


def _no_markers(text: str) -> bool:
    return "<think>" not in text and "</think>" not in text


# ---------------------------------------------------------------------------
# Fake HTTP session (DoD-9: the real _stream_post runs against it)
# ---------------------------------------------------------------------------


class _Noop:
    """An awaitable that does nothing, so ``release()`` works whether or not the caller awaits it."""

    def __await__(self):
        return iter(())


class _FakeContent:
    """Stands in for ``aiohttp.StreamReader``: yields the body as one chunk."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_chunked(self, n: int):
        for chunk in self._chunks:
            yield chunk

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class _FakeResponse:
    """A minimal ``aiohttp.ClientResponse`` stand-in: status, headers, body reads and streamed content."""

    def __init__(self, url: str, status: int = 200, body: bytes = b"", content_type: str = "application/json") -> None:
        self.url = URL(url)
        self.real_url = self.url
        self.status = status
        self.reason = "OK" if status < 400 else "Not Found"
        self.content_type = content_type
        self.headers = CIMultiDict({"Content-Type": content_type})
        self._body = body
        self.content = _FakeContent([body])

    @property
    def ok(self) -> bool:
        return self.status < 400

    async def text(self, *args, **kwargs) -> str:
        return self._body.decode("utf-8")

    async def read(self) -> bytes:
        return self._body

    async def json(self, *args, **kwargs) -> Any:
        return json.loads(self._body.decode("utf-8"))

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=SimpleNamespace(real_url=self.url, url=self.url, method="POST", headers={}),
                history=(),
                status=self.status,
                message=self.reason,
            )

    def release(self) -> _Noop:
        return _Noop()

    def close(self) -> None:
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False


class _FakeRequestContext:
    """What ``session.request/post`` return: usable with ``async with`` or ``await``."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def _open(self) -> _FakeResponse:
        return self._response

    def __await__(self):
        return self._open().__await__()

    async def __aenter__(self) -> _FakeResponse:
        return await self._open()

    async def __aexit__(self, *exc_info) -> bool:
        return False


class _FakeSession:
    """Fake ``aiohttp.ClientSession`` assigned to ``client.session`` (the client reuses an existing session).

    ``responder(method, url, kwargs)`` returns the ``_FakeResponse`` for each request; calls are recorded.
    """

    def __init__(self, responder: Callable[[str, str, dict], _FakeResponse]) -> None:
        self._responder = responder
        self.calls: list[dict] = []
        self.closed = False

    def request(self, method: str, url: Any, **kwargs) -> _FakeRequestContext:
        self.calls.append({"method": method, "url": str(url), "kwargs": kwargs})
        return _FakeRequestContext(self._responder(method, str(url), kwargs))

    def post(self, url: Any, **kwargs) -> _FakeRequestContext:
        return self.request("POST", url, **kwargs)

    def get(self, url: Any, **kwargs) -> _FakeRequestContext:
        return self.request("GET", url, **kwargs)

    async def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# DoD-1 — one streamed /responses request per round; body = non-streamed body with stream true
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_openai_streamed_tool_rounds_post_non_streamed_bodies_with_stream_true__DoD1(monkeypatch, flavour):
    """DoD-1: each streamed round opens exactly one streamed request to ``/responses`` and no non-streamed one; its
    body equals the non-streamed round body except that ``stream`` is true (DEC-004-10, DEC-004-14)."""
    r1_events, r1_reply = _round(
        flavour, "r1", reasoning=["Need ", "the record."], text=["Checking."],
        calls=[("call_1", "fetch_record", ARGS_A7)],
    )
    r2_events, r2_reply = _round(flavour, "r2", reasoning=["Got it."], text=["Record ", "found."])

    plain = _client()
    records = _install_request_rounds(monkeypatch, plain, [r1_reply, r2_reply])
    plain_result = await _non_streamed(plain, _ToolLog(), system=SYSTEM, options=dict(OPTIONS))

    streamed = _client()
    calls = _install_stream_rounds(monkeypatch, streamed, [r1_events, r2_events])
    forbidden = _forbid_request(monkeypatch, streamed)
    streamed_result = await _streamed(streamed, _ToolLog(), _Recorder(), system=SYSTEM, options=dict(OPTIONS))

    assert len(records) == 2
    assert all(r["method"] == "POST" and r["path"] == "/responses" for r in records)
    assert len(calls) == 2
    assert forbidden == []
    for record, call in zip(records, calls):
        assert call["path"] == "/responses"
        assert record["json"]["stream"] is False
        assert call["payload"]["stream"] is True
        assert call["payload"] == {**record["json"], "stream": True}
    assert streamed_result == plain_result == "Record found."


# ---------------------------------------------------------------------------
# DoD-2 — every round's text pieces reach the receiver; only the final round's text is returned
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_openai_streamed_tools_deliver_every_round_text_return_final__DoD2(monkeypatch, flavour):
    """DoD-2: the receiver gets the text pieces of every round in order; the call returns only the final round's
    text (US-023.AC-1, US-023.AC-2)."""
    r1, _ = _round(flavour, "r1", text=["Let me ", "check."], calls=[("call_1", "fetch_record", ARGS_A7)])
    r2, _ = _round(flavour, "r2", text=["Found ", "it."])
    client = _client()
    _install_stream_rounds(monkeypatch, client, [r1, r2])
    recorder = _Recorder()

    result = await _streamed(client, _ToolLog(), recorder)

    assert recorder.pieces == ["Let me ", "check.", "Found ", "it."]
    assert result == "Found it."


@pytest.mark.asyncio
@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_openai_tool_stream_parser_returns_round_text__DoD2(monkeypatch, flavour):
    """DoD-2: the round parser delivers the round's text pieces in order and returns their concatenation as the
    round text (US-023.AC-1, DEC-004-9)."""
    events, _ = _round(flavour, "r1", text=["Hel", "lo", " world"])
    client = _client()
    _install_sse_rounds(monkeypatch, client, [events])
    recorder = _Recorder()

    result = await client._read_responses_tool_stream(None, recorder)

    assert recorder.pieces == ["Hel", "lo", " world"]
    assert result.text == "Hello world"
    assert result.tool_calls == []
    assert result.reasoning == []


# ---------------------------------------------------------------------------
# DoD-3 — reasoning pieces: one closed block per round, never in the answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_openai_streamed_tools_reasoning_block_per_round__DoD3(monkeypatch, flavour):
    """DoD-3: each round's reasoning arrives as its own block — opening marker first, closed before that round's
    text; a round ending in reasoning followed only by calls has its block closed before the next round's pieces;
    the returned answer has no reasoning and no marker (US-027.AC-1, US-027.AC-2, US-058.AC-5, DEC-004-9)."""
    r1, _ = _round(flavour, "r1", reasoning=["Pl", "an"], calls=[("call_1", "fetch_record", ARGS_A7)])
    r2, _ = _round(flavour, "r2", reasoning=["Chk"], text=["Do", "ne"])
    client = _client()
    _install_stream_rounds(monkeypatch, client, [r1, r2])
    recorder = _Recorder()

    result = await _streamed(client, _ToolLog(), recorder)

    assert recorder.pieces == ["<think>Pl", "an", "</think>", "<think>Chk", "</think>Do", "ne"]
    assert result == "Done"
    assert _no_markers(result)
    assert "Plan" not in result and "Chk" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_openai_tool_stream_parser_marks_reasoning_and_keeps_text_only__DoD3(monkeypatch, flavour):
    """DoD-3: the parser delivers reasoning with markers (``<think>`` on the first reasoning piece, ``</think>`` on
    the first text piece after it) and returns round text without reasoning or markers (DEC-004-9, DEC-003-13)."""
    events, _ = _round(flavour, "r1", reasoning=["Think", "ing"], text=["Ans", "wer"])
    client = _client()
    _install_sse_rounds(monkeypatch, client, [events])
    recorder = _Recorder()

    result = await client._read_responses_tool_stream(None, recorder)

    assert recorder.pieces == ["<think>Think", "ing", "</think>Ans", "wer"]
    assert result.text == "Answer"


@pytest.mark.asyncio
@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_openai_tool_stream_parser_closes_open_block_at_completion__DoD3(monkeypatch, flavour):
    """DoD-3: a round of reasoning followed only by a call closes its block with a ``</think>`` piece at
    completion; the round text is empty (US-027.AC-1, DEC-004-9)."""
    events, _ = _round(flavour, "r1", reasoning=["a", "b"], calls=[("call_1", "fetch_record", ARGS_A7)])
    client = _client()
    _install_sse_rounds(monkeypatch, client, [events])
    recorder = _Recorder()

    result = await client._read_responses_tool_stream(None, recorder)

    assert recorder.pieces == ["<think>a", "b", "</think>"]
    assert result.text == ""


@pytest.mark.asyncio
async def test_openai_tool_stream_parser_fresh_marker_state_each_round__DoD3(monkeypatch):
    """DoD-3: marker state is fresh per round — a second parser run that starts with text delivers no closing
    marker, and one that starts with reasoning opens a new block (DEC-004-9)."""
    first, _ = _round("openai", "r1", reasoning=["R1"], calls=[("call_1", "fetch_record", ARGS_A7)])
    second, _ = _round("openai", "r2", text=["T2"])
    third, _ = _round("llamacpp", "r3", reasoning=["R3"], text=["T3"])
    client = _client()
    _install_sse_rounds(monkeypatch, client, [first, second, third])
    recorder = _Recorder()

    await client._read_responses_tool_stream(None, recorder)
    await client._read_responses_tool_stream(None, recorder)
    await client._read_responses_tool_stream(None, recorder)

    assert recorder.pieces == ["<think>R1", "</think>", "T2", "<think>R3", "</think>T3"]


# ---------------------------------------------------------------------------
# DoD-4 — calls assembled from added + deltas; full arguments on done events replace the assembled text
# ---------------------------------------------------------------------------


def _expected_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    """A tool call in the internal assistant-turn shape that a tool round returns (DEC-004-11)."""
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial", "chunks"),
    [
        pytest.param("", ['{"key": ', '"A7"}'], id="deltas-only"),
        pytest.param('{"key": ', ['"A7"}'], id="initial-arguments-then-deltas"),
    ],
)
async def test_openai_tool_stream_parser_assembles_call_from_added_and_deltas__DoD4(monkeypatch, initial, chunks):
    """DoD-4: a call is assembled from its output-item-added event (call id, name, initial arguments) and its
    argument deltas, when no done event carries the arguments (DEC-004-9)."""
    events = [
        CREATED,
        _added(0, _function_call_item("fc_1", "call_a", "fetch_record", initial, status="in_progress")),
        *[_args_delta(0, "fc_1", c) for c in chunks],
        COMPLETED,
    ]
    client = _client()
    _install_sse_rounds(monkeypatch, client, [events])

    result = await client._read_responses_tool_stream(None, _Recorder())

    assert result.tool_calls == [_expected_call("call_a", "fetch_record", ARGS_A7_TEXT)]
    assert result.text == ""


def _replacing_events(replacer: str) -> list[str]:
    """A call whose deltas assemble a draft, then a done event (arguments-done or output-item-done) carrying the
    full arguments for key B9."""
    full = '{"key": "B9"}'
    events = [
        CREATED,
        _added(0, _function_call_item("fc_1", "call_b", "fetch_record", "", status="in_progress")),
        _args_delta(0, "fc_1", '{"key": '),
        _args_delta(0, "fc_1", '"DRAFT"}'),
    ]
    if replacer == "arguments-done":
        events.append(_args_done(0, "fc_1", full))
    else:
        events.append(_done(0, _function_call_item("fc_1", "call_b", "fetch_record", full)))
    events.append(COMPLETED)
    return events


REPLACERS = [pytest.param("arguments-done", id="arguments-done"), pytest.param("item-done", id="item-done")]


@pytest.mark.asyncio
@pytest.mark.parametrize("replacer", REPLACERS)
async def test_openai_tool_stream_parser_done_arguments_replace_assembled__DoD4(monkeypatch, replacer):
    """DoD-4: full arguments on an arguments-done event or on a function-call output-item-done event replace the
    text assembled from the deltas (DEC-004-9)."""
    client = _client()
    _install_sse_rounds(monkeypatch, client, [_replacing_events(replacer)])

    result = await client._read_responses_tool_stream(None, _Recorder())

    assert result.tool_calls == [_expected_call("call_b", "fetch_record", '{"key": "B9"}')]


@pytest.mark.asyncio
@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_openai_streamed_tools_run_assembled_call_and_resend_it__DoD4(monkeypatch, flavour):
    """DoD-4: the tool runs with the assembled arguments and the next round's input holds that call with its
    ``call_id``, name and arguments (DEC-004-9, DEC-004-6)."""
    r1, _ = _round(flavour, "r1", calls=[("call_1", "fetch_record", ARGS_A7)])
    r2, _ = _round(flavour, "r2", text=["Done."])
    client = _client()
    calls = _install_stream_rounds(monkeypatch, client, [r1, r2])
    tools = _ToolLog()

    result = await _streamed(client, tools, _Recorder())

    assert tools.calls == [("fetch_record", {"key": "A7"})]
    assert _fc_input("call_1", "fetch_record", ARGS_A7_TEXT) in calls[1]["payload"]["input"]
    assert result == "Done."


@pytest.mark.asyncio
@pytest.mark.parametrize("replacer", REPLACERS)
async def test_openai_streamed_tools_run_with_replaced_arguments__DoD4(monkeypatch, replacer):
    """DoD-4: when a done event carries full arguments, the tool runs with those and the next round re-sends them,
    not the assembled draft (DEC-004-9, DEC-004-6)."""
    r2, _ = _round("openai", "r2", text=["Done."])
    client = _client()
    calls = _install_stream_rounds(monkeypatch, client, [_replacing_events(replacer), r2])
    tools = _ToolLog()

    await _streamed(client, tools, _Recorder())

    assert tools.calls == [("fetch_record", {"key": "B9"})]
    assert _fc_input("call_b", "fetch_record", '{"key": "B9"}') in calls[1]["payload"]["input"]


# ---------------------------------------------------------------------------
# DoD-5 — two calls in one round behind a reasoning item
# ---------------------------------------------------------------------------


TWO_CALLS = [
    ("call_a", "fetch_record", ['{"key": ', '"A1"}']),
    ("call_b", "count_items", ['{"category": ', '"tools"}']),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_openai_tool_stream_parser_two_calls_dense_in_order__DoD5(monkeypatch, flavour):
    """DoD-5: two calls at output positions 1 and 2 behind reasoning at 0 come back as two separate calls, dense
    and in order, each with its own call id, name and arguments (US-035.AC-4, DEC-arch-10, DEC-004-9)."""
    events, _ = _round(flavour, "r1", reasoning=["Two lookups."], calls=TWO_CALLS)
    client = _client()
    _install_sse_rounds(monkeypatch, client, [events])

    result = await client._read_responses_tool_stream(None, _Recorder())

    assert result.tool_calls == [
        _expected_call("call_a", "fetch_record", '{"key": "A1"}'),
        _expected_call("call_b", "count_items", '{"category": "tools"}'),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_openai_streamed_tools_two_calls_run_in_order_with_own_call_ids__DoD5(monkeypatch, flavour):
    """DoD-5: both calls run in order with their own name and arguments, and their outputs carry their own
    ``call_id`` on the next round (US-035.AC-4, DEC-arch-10, DEC-004-9, DEC-004-6)."""
    r1, r1_reply = _round(flavour, "r1", reasoning=["Two lookups."], calls=TWO_CALLS)
    r2, _ = _round(flavour, "r2", text=["Both done."])
    client = _client()
    calls = _install_stream_rounds(monkeypatch, client, [r1, r2])
    tools = _ToolLog()

    result = await _streamed(client, tools, _Recorder())

    assert tools.calls == [("fetch_record", {"key": "A1"}), ("count_items", {"category": "tools"})]
    assert calls[1]["payload"]["input"] == [
        USER_ITEM,
        _of_type(r1_reply, "reasoning")[0],
        _fc_input("call_a", "fetch_record", '{"key": "A1"}'),
        _fc_input("call_b", "count_items", '{"category": "tools"}'),
        _fco_input("call_a", _record_result("A1")),
        _fco_input("call_b", _count_result("tools")),
    ]
    assert result == "Both done."


# ---------------------------------------------------------------------------
# DoD-6 — reasoning carry items re-sent verbatim, by output position, same layout as non-streamed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_openai_streamed_tools_resend_reasoning_verbatim_ahead_of_calls__DoD6(monkeypatch, flavour):
    """DoD-6: the reasoning item from the round's reasoning output-item-done event is re-sent on the next round
    verbatim and ahead of that round's function call, also when it finished after the call's deltas
    (US-058.AC-3, DEC-004-7, DEC-004-9)."""
    r1, r1_reply = _round(
        flavour, "r1", reasoning=["Need ", "the record."], calls=[("call_1", "fetch_record", ARGS_A7)],
        late_reasoning_done=True,
    )
    r2, _ = _round(flavour, "r2", text=["Done."])
    client = _client()
    calls = _install_stream_rounds(monkeypatch, client, [r1, r2])

    await _streamed(client, _ToolLog(), _Recorder())

    reasoning_item = _of_type(r1_reply, "reasoning")[0]
    assert calls[1]["payload"]["input"] == [
        USER_ITEM,
        reasoning_item,
        _fc_input("call_1", "fetch_record", ARGS_A7_TEXT),
        _fco_input("call_1", _record_result("A7")),
    ]
    # The first round has no earlier reasoning to carry.
    assert calls[0]["payload"]["input"] == [USER_ITEM]


def _two_reasoning_round(flavour: str) -> tuple[list[str], dict[str, Any]]:
    """A round with reasoning at positions 0 and 1 and a call at 2, where reasoning 1 finishes first and
    reasoning 0 finishes last (after the call's deltas), plus the equivalent non-streamed reply."""
    rs0_open, rs0_done, rs0 = _reasoning_part(flavour, 0, "rs_a", ["First ", "thought."])
    rs1_open, rs1_done, rs1 = _reasoning_part(flavour, 1, "rs_b", ["Second thought."])
    fc_open, fc_done, fc = _call_part(flavour, 2, "fc_1", "call_1", "fetch_record", ARGS_A7)
    events = [CREATED, *rs0_open, *rs1_open, rs1_done, *fc_open, fc_done, rs0_done, COMPLETED]
    reply = {"id": "resp_r1", "object": "response", "status": "completed", "output": [rs0, rs1, fc]}
    return events, reply


@pytest.mark.asyncio
@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_openai_tool_stream_parser_orders_carry_items_by_output_position__DoD6(monkeypatch, flavour):
    """DoD-6: carry items come back verbatim and ordered by output position, not by the order the items
    finished (DEC-004-9, DEC-004-7)."""
    events, reply = _two_reasoning_round(flavour)
    client = _client()
    _install_sse_rounds(monkeypatch, client, [events])

    result = await client._read_responses_tool_stream(None, _Recorder())

    assert result.reasoning == _of_type(reply, "reasoning")


@pytest.mark.asyncio
@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_openai_streamed_tools_resent_input_matches_non_streamed_layout__DoD6(monkeypatch, flavour):
    """DoD-6: with reasoning items finishing out of order, the re-sent input has the same layout as the
    non-streamed path's for the same round: reasoning in output order, then the call, then its output
    (US-058.AC-3, DEC-004-7, DEC-004-9)."""
    r1_events, r1_reply = _two_reasoning_round(flavour)
    r2_events, r2_reply = _round(flavour, "r2", text=["Done."])

    plain = _client()
    records = _install_request_rounds(monkeypatch, plain, [r1_reply, r2_reply])
    await _non_streamed(plain, _ToolLog())

    streamed = _client()
    calls = _install_stream_rounds(monkeypatch, streamed, [r1_events, r2_events])
    await _streamed(streamed, _ToolLog(), _Recorder())

    rs0, rs1 = _of_type(r1_reply, "reasoning")
    expected = [USER_ITEM, rs0, rs1, _fc_input("call_1", "fetch_record", ARGS_A7_TEXT),
                _fco_input("call_1", _record_result("A7"))]
    assert calls[1]["payload"]["input"] == expected
    assert calls[1]["payload"]["input"] == records[1]["json"]["input"]


# ---------------------------------------------------------------------------
# DoD-7 — no tool-call name or argument text reaches the receiver
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_openai_streamed_tools_never_deliver_call_fragments__DoD7(monkeypatch, flavour):
    """DoD-7: neither a tool-call name nor any argument text reaches the receiver; it gets text and reasoning
    only (DEC-004-9)."""
    r1, _ = _round(
        flavour, "r1", reasoning=["Plan."], text=["Looking."],
        calls=[("call_1", "fetch_record", ['{"key": ', '"ZQX417"}'])],
    )
    r2, _ = _round(flavour, "r2", text=["Done."])
    client = _client()
    _install_stream_rounds(monkeypatch, client, [r1, r2])
    recorder = _Recorder()

    await _streamed(client, _ToolLog(), recorder)

    received = "".join(recorder.pieces)
    for fragment in ("fetch_record", "ZQX417", '{"key"'):
        assert fragment not in received
        assert all(fragment not in piece for piece in recorder.pieces)
    assert recorder.pieces == ["<think>Plan.", "</think>Looking.", "Done."]


# ---------------------------------------------------------------------------
# DoD-8 — failure events and early end in any round
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("event", "reason"), FAILURES)
async def test_openai_tool_stream_parser_failure_event_raises_with_reason__DoD8(monkeypatch, event, reason):
    """DoD-8: a failed, incomplete or error event ends the round with ``LLMError`` carrying the server's reason;
    no later piece is delivered (US-060.AC-2, DEC-004-9)."""
    opening, done, _ = _call_part("openai", 1, "fc_1", "call_1", "fetch_record", ARGS_A7)
    events = [CREATED, _text_delta(0, "msg_1", "Before"), *opening, done, json.dumps(event),
              _text_delta(0, "msg_1", "After"), _reasoning_delta("openai", 2, "rs_1", "late"), COMPLETED]
    client = _client()
    _install_sse_rounds(monkeypatch, client, [events])
    recorder = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await client._read_responses_tool_stream(None, recorder)

    assert reason in str(excinfo.value)
    assert recorder.pieces == ["Before"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("event", "reason"), FAILURES)
async def test_openai_streamed_tools_failure_in_first_round_runs_no_tool__DoD8(monkeypatch, event, reason):
    """DoD-8: a failure event in the first round ends the call with ``LLMError`` carrying the server's reason; that
    round's tool does not run, no later piece is delivered and no further round is requested
    (US-060.AC-2, DEC-004-9)."""
    r1, _ = _round("openai", "r1", text=["Before"], calls=[("call_1", "fetch_record", ARGS_A7)], complete=False)
    r1 += [json.dumps(event), _text_delta(0, "msg_r1", "After"), COMPLETED]
    r2, _ = _round("openai", "r2", text=["Never."])
    client = _client()
    calls = _install_stream_rounds(monkeypatch, client, [r1, r2])
    tools = _ToolLog()
    recorder = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await _streamed(client, tools, recorder)

    assert reason in str(excinfo.value)
    assert tools.calls == []
    assert recorder.pieces == ["Before"]
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("event", "reason"), FAILURES)
async def test_openai_streamed_tools_failure_in_later_round_runs_none_of_its_tools__DoD8(monkeypatch, event, reason):
    """DoD-8: a failure event in the second round ends the call with ``LLMError`` carrying the server's reason;
    the second round's tool does not run and no piece after the event is delivered (US-060.AC-2, DEC-004-9)."""
    r1, _ = _round("llamacpp", "r1", text=["First."], calls=[("call_1", "fetch_record", ARGS_A7)])
    r2, _ = _round(
        "llamacpp", "r2", text=["Partial"], calls=[("call_2", "fetch_record", ['{"key": "B2"}'])], complete=False
    )
    r2 += [json.dumps(event), _text_delta(0, "msg_r2", "After"), COMPLETED]
    r3, _ = _round("llamacpp", "r3", text=["Never."])
    client = _client()
    calls = _install_stream_rounds(monkeypatch, client, [r1, r2, r3])
    tools = _ToolLog()
    recorder = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await _streamed(client, tools, recorder)

    assert reason in str(excinfo.value)
    assert tools.calls == [("fetch_record", {"key": "A7"})]
    assert recorder.pieces == ["First.", "Partial"]
    assert len(calls) == 2


EARLY_ENDS = [pytest.param([], id="no-completed"), pytest.param(["[DONE]"], id="done-sentinel")]


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", EARLY_ENDS)
@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_openai_tool_stream_parser_end_without_completed_raises__DoD8(monkeypatch, flavour, tail):
    """DoD-8: a round stream ending without ``response.completed`` raises ``LLMError``, also after a ``[DONE]``
    sentinel (US-024.AC-2, DEC-004-9)."""
    events, _ = _round(flavour, "r1", text=["Cut"], calls=[("call_1", "fetch_record", ARGS_A7)], complete=False)
    client = _client()
    _install_sse_rounds(monkeypatch, client, [events + tail])

    with pytest.raises(LLMError):
        await client._read_responses_tool_stream(None, _Recorder())


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", EARLY_ENDS)
@pytest.mark.parametrize("cut_round", [pytest.param(1, id="first-round"), pytest.param(2, id="second-round")])
async def test_openai_streamed_tools_round_without_completed_runs_no_tool__DoD8(monkeypatch, cut_round, tail):
    """DoD-8: a round stream that ends without the completed event (with or without ``[DONE]``) ends the call with
    ``LLMError``; no tool of that round runs (US-024.AC-2, US-060.AC-2, DEC-004-9)."""
    r1, _ = _round("openai", "r1", calls=[("call_1", "fetch_record", ARGS_A7)], complete=cut_round != 1)
    r2, _ = _round("openai", "r2", calls=[("call_2", "fetch_record", ['{"key": "B2"}'])], complete=cut_round != 2)
    r3, _ = _round("openai", "r3", text=["Never."])
    rounds = [r1 + (tail if cut_round == 1 else []), r2 + (tail if cut_round == 2 else []), r3]
    client = _client()
    calls = _install_stream_rounds(monkeypatch, client, rounds)
    tools = _ToolLog()

    with pytest.raises(LLMError):
        await _streamed(client, tools, _Recorder())

    expected_runs = [] if cut_round == 1 else [("fetch_record", {"key": "A7"})]
    assert tools.calls == expected_runs
    assert len(calls) == cut_round


# ---------------------------------------------------------------------------
# DoD-9 — non-2xx reply on a later streamed round
# ---------------------------------------------------------------------------


_SERVER_ERROR_BODY = json.dumps(
    {"error": {"message": "Item 'rs_r1' could not be read", "type": "invalid_request_error", "code": None}}
)


@pytest.mark.asyncio
async def test_openai_streamed_tools_non_2xx_on_later_round_raises_with_address_not_credential__DoD9(monkeypatch):
    """DoD-9: a non-2xx reply on the second streamed round ends with ``LLMError`` carrying the address and the
    server's reply and not the credential; nothing more reaches the receiver (US-060.AC-1, US-060.AC-4,
    DEC-arch-5)."""
    client = _client()
    r1, _ = _round("openai", "r1", reasoning=["Plan."], text=["First."], calls=[("call_1", "fetch_record", ARGS_A7)])
    # Only the event reader is faked; the real _stream_post runs against a fake session: round 1 is answered
    # 200 (its events come from the fake reader), round 2 is answered 404 with the server's error body.
    _install_sse_rounds(monkeypatch, client, [r1])
    error_bytes = _SERVER_ERROR_BODY.encode("utf-8")

    def responder(method: str, url: str, kwargs: dict) -> _FakeResponse:
        if len(session.calls) == 1:
            return _FakeResponse(url, 200, b"", "text/event-stream")
        return _FakeResponse(url, 404, error_bytes, "application/json")

    session = _FakeSession(responder)
    client.session = session
    tools = _ToolLog()
    recorder = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await _streamed(client, tools, recorder)

    err = excinfo.value
    rendered = str(err)
    assert len(session.calls) == 2
    assert RESPONSES_URL in rendered
    assert "404" in rendered
    assert "could not be read" in rendered
    assert CREDENTIAL not in f"{rendered} {err.message} {err.description} {err.args!r}"
    assert tools.calls == [("fetch_record", {"key": "A7"})]
    assert recorder.pieces == ["<think>Plan.", "</think>First."]


# ---------------------------------------------------------------------------
# DoD-10 — receiver failures propagate unchanged; async receiver awaited
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_tool_stream_parser_receiver_failure_propagates_unchanged__DoD10(monkeypatch):
    """DoD-10: a receiver failure ends the round with that same exception and no later piece is delivered
    (US-022.AC-1, US-022.AC-2, DEC-000-23)."""
    events, _ = _round("openai", "r1", text=["a", "b", "c"])
    client = _client()
    _install_sse_rounds(monkeypatch, client, [events])
    boom = _ReceiverBoom("receiver broke")
    seen: list[str] = []

    async def failing_receiver(piece: str) -> None:
        seen.append(piece)
        if piece == "b":
            raise boom

    with pytest.raises(_ReceiverBoom) as excinfo:
        await client._read_responses_tool_stream(None, failing_receiver)

    assert excinfo.value is boom
    assert seen == ["a", "b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_openai_streamed_tools_receiver_failure_in_later_round_propagates__DoD10(monkeypatch, flavour):
    """DoD-10: a receiver failure in the second round ends the call with that same exception, not wrapped, and no
    later piece is delivered (US-022.AC-2, DEC-000-23)."""
    r1, _ = _round(flavour, "r1", text=["one"], calls=[("call_1", "fetch_record", ARGS_A7)])
    r2, _ = _round(flavour, "r2", text=["two", "three"])
    client = _client()
    _install_stream_rounds(monkeypatch, client, [r1, r2])
    boom = _ReceiverBoom("receiver broke in round 2")
    seen: list[str] = []

    async def failing_receiver(piece: str) -> None:
        seen.append(piece)
        if piece == "two":
            raise boom

    with pytest.raises(_ReceiverBoom) as excinfo:
        await _streamed(client, _ToolLog(), failing_receiver)

    assert excinfo.value is boom
    assert seen == ["one", "two"]


@pytest.mark.asyncio
async def test_openai_tool_stream_parser_awaits_async_receiver_before_next_read__DoD10(monkeypatch):
    """DoD-10: an async receiver is awaited — each piece is fully handled (past an ``await`` inside the receiver)
    before the next payload is read (US-022.AC-1, DEC-000-23)."""
    client = _client()
    log: list[str] = []
    _install_sse_rounds(
        monkeypatch, client, [[_text_delta(0, "msg_1", "a"), _text_delta(0, "msg_1", "b"), COMPLETED]], log=log
    )

    async def slow_receiver(piece: str) -> None:
        # Yield to the loop first: an un-awaited coroutine would never reach the append below.
        await asyncio.sleep(0)
        log.append(f"got:{piece}")

    result = await client._read_responses_tool_stream(None, slow_receiver)

    assert result.text == "ab"
    # Three payload reads (a, b, completed); the parser returns at completion, so no fourth read is required.
    assert log[:5] == ["read", "got:a", "read", "got:b", "read"]


# ---------------------------------------------------------------------------
# DoD-11 — requested format: final round's pieces join to the structured answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_openai_streamed_tools_with_format_return_joined_final_pieces__DoD11(monkeypatch, flavour):
    """DoD-11: with a requested format, the final round's text pieces join to the returned structured answer
    (US-037.AC-4); the format is sent on every round (DEC-004-10)."""
    final_chunks = ['{"record": ', '"A7", ', '"value": 42}']
    r1, _ = _round(flavour, "r1", calls=[("call_1", "fetch_record", ARGS_A7)])
    r2, _ = _round(flavour, "r2", text=final_chunks)
    client = _client()
    calls = _install_stream_rounds(monkeypatch, client, [r1, r2])
    recorder = _Recorder()

    result = await _streamed(client, _ToolLog(), recorder, response_format={"type": "json_object"})

    assert recorder.pieces[-3:] == final_chunks
    assert result == "".join(final_chunks)
    assert json.loads(result) == {"record": "A7", "value": 42}
    assert all(call["payload"]["text"] == {"format": {"type": "json_object"}} for call in calls)


# ---------------------------------------------------------------------------
# DoD-12 — same answer without a receiver; tracing changes neither pieces nor answer
# ---------------------------------------------------------------------------


def _conversation(flavour: str) -> list[list[str]]:
    """A two-round conversation with reasoning and text in both rounds."""
    r1, _ = _round(flavour, "r1", reasoning=["Why"], text=["Looking"], calls=[("call_1", "fetch_record", ARGS_A7)])
    r2, _ = _round(flavour, "r2", reasoning=["Because"], text=["Fin", "al."])
    return [r1, r2]


@pytest.mark.asyncio
@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_openai_streamed_tools_without_receiver_return_same_answer__DoD12(monkeypatch, flavour):
    """DoD-12: a streamed tool call without a receiver returns the same answer as with one (US-023.AC-2)."""
    with_receiver = _client()
    _install_stream_rounds(monkeypatch, with_receiver, _conversation(flavour))
    with_tools = _ToolLog()
    with_result = await _streamed(with_receiver, with_tools, _Recorder())

    without_receiver = _client()
    _install_stream_rounds(monkeypatch, without_receiver, _conversation(flavour))
    without_tools = _ToolLog()
    without_result = await _streamed(without_receiver, without_tools)

    assert with_result == without_result == "Final."
    assert with_tools.calls == without_tools.calls == [("fetch_record", {"key": "A7"})]


@pytest.mark.asyncio
@pytest.mark.parametrize("flavour", FLAVOURS)
async def test_openai_streamed_tools_with_tracing_match_untraced__DoD12(monkeypatch, tmp_path, flavour):
    """DoD-12: with tracing enabled the receiver gets the same pieces and the call returns the same answer as
    without tracing (US-050.AC-4, DEC-004-10)."""
    plain = _client()
    _install_stream_rounds(monkeypatch, plain, _conversation(flavour))
    plain_recorder = _Recorder()
    plain_result = await _streamed(plain, _ToolLog(), plain_recorder, system=SYSTEM)

    # The tracer reads its switches at construction (DEC-003-16), so set them before building the traced client.
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(tmp_path))
    traced = _client()
    _install_stream_rounds(monkeypatch, traced, _conversation(flavour))
    traced_recorder = _Recorder()
    traced_result = await _streamed(traced, _ToolLog(), traced_recorder, system=SYSTEM)

    expected = ["<think>Why", "</think>Looking", "<think>Because", "</think>Fin", "al."]
    assert plain_recorder.pieces == expected
    assert traced_recorder.pieces == expected
    assert traced_result == plain_result == "Final."
