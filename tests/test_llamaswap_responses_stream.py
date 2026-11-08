"""Unit tests for streamed chat on the llama-swap Responses class (DEC-003-13).

Covers ``LlamaSwapAPIClient``'s Responses stream handling (DEC-003-13): the streamed request to
``/responses`` and its body, text and reasoning pieces with per-run think markers, completion and
early-end rules, failure events, skipped payloads, non-2xx replies, receiver rules, ``generate``
through ``chat`` and tracing parity. The server is faked through the DEC-003-16 seams
(``_iter_sse_data``, ``_stream_post``, ``_request`` or a fake session). Each test names its DoD id.
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
_SECRET = "sk-llamaswap-secret-token-123"

_OPEN = "<think>"
_CLOSE = "</think>"

_MESSAGES = [{"role": "user", "content": "Hello"}]


# ---------------------------------------------------------------------------
# Event builders (Responses stream vocabulary: DEC-003-13, DEC-arch-11)
# ---------------------------------------------------------------------------


def _ev(event_type: str, **fields: Any) -> str:
    """One stream payload string: a JSON event object with ``type`` and the given fields."""
    return json.dumps({"type": event_type, **fields})


def _text(delta: str) -> str:
    """An output-text delta event."""
    return _ev("response.output_text.delta", item_id="msg_1", output_index=1, content_index=0, delta=delta)


def _reasoning(delta: str) -> str:
    """A reasoning-text delta event (what llama-swap sends while the model reasons)."""
    return _ev("response.reasoning_text.delta", item_id="rs_1", output_index=0, content_index=0, delta=delta)


def _summary(delta: str) -> str:
    """A reasoning-summary delta event (handled exactly like reasoning text, DEC-003-13)."""
    return _ev("response.reasoning_summary_text.delta", item_id="rs_1", summary_index=0, delta=delta)


_COMPLETED = _ev("response.completed", response={"id": "resp_1", "status": "completed", "output": []})


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch):
    """Start every test with tracing off so a developer's shell env cannot change behaviour (DEC-000-48)."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _client(**kw: Any) -> LlamaSwapAPIClient:
    """Build a llama-swap Responses client on the fixed test address."""
    return LlamaSwapAPIClient(model=_MODEL, base_url=_BASE_URL, **kw)


class _Recorder:
    """Async receiver collecting every piece it is handed, in order."""

    def __init__(self) -> None:
        self.pieces: list[str] = []

    async def __call__(self, piece: str) -> None:
        self.pieces.append(piece)


def _fake_sse(monkeypatch, client: LlamaSwapAPIClient, payloads: list[str]) -> None:
    """Replace ``_iter_sse_data`` with a fake yielding ``payloads`` (SSE data strings without ``data:``)."""

    async def fake_iter_sse_data(_response):
        for payload in payloads:
            yield payload

    monkeypatch.setattr(client, "_iter_sse_data", fake_iter_sse_data)


def _fake_stream_post(monkeypatch, client: LlamaSwapAPIClient) -> list[dict[str, Any]]:
    """Replace ``_stream_post`` with a fake async context manager; returns the captured (path, payload) calls.

    The yielded response is a placeholder: the parser only hands it to the faked ``_iter_sse_data``
    (DEC-003-16). Payloads are deep-copied so later mutation cannot change what was captured.
    """
    calls: list[dict[str, Any]] = []

    @asynccontextmanager
    async def fake_stream_post(path, payload):
        calls.append({"path": path, "payload": copy.deepcopy(payload)})
        yield None

    monkeypatch.setattr(client, "_stream_post", fake_stream_post)
    return calls


def _fake_request(monkeypatch, client: LlamaSwapAPIClient, reply: Any) -> list[dict[str, Any]]:
    """Replace ``_request`` with a fake returning ``reply``; returns the captured requests (body deep-copied)."""
    calls: list[dict[str, Any]] = []

    async def fake_request(method, path, **kw):
        calls.append({"method": method, "path": path, "json": copy.deepcopy(kw.get("json"))})
        return reply

    monkeypatch.setattr(client, "_request", fake_request)
    return calls


def _nonstream_reply(text: str) -> dict[str, Any]:
    """A completed non-streamed Responses reply carrying one message with one output-text part."""
    return {
        "id": "resp_1",
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
    }


# Real-session fakes, used only for the non-2xx case so the base streamed-request helper runs for real.


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


async def _parse(monkeypatch, payloads: list[str], receiver: Any = None) -> str:
    """Drive the parser directly on canned payloads with a placeholder response (DEC-003-16)."""
    client = _client()
    _fake_sse(monkeypatch, client, payloads)
    return await client._stream_responses_response(None, receiver)


# ---------------------------------------------------------------------------
# DoD-1 — the streamed request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_streamed_chat_opens_one_request_to_responses_with_stream_true__DoD1(monkeypatch):
    """DoD-1: one streamed request to /responses whose body is the non-streamed body with stream true."""
    options = {"temperature": 0.3, "max_tokens": 64, "enable_thinking": True}
    messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]

    plain = _client()
    captured = _fake_request(monkeypatch, plain, _nonstream_reply("Hi"))
    await plain.chat(copy.deepcopy(messages), system="Be brief.", options=dict(options))
    assert len(captured) == 1
    nonstream_body = captured[0]["json"]
    assert nonstream_body["stream"] is False

    streamed = _client()
    calls = _fake_stream_post(monkeypatch, streamed)
    _fake_sse(monkeypatch, streamed, [_text("Hi"), _COMPLETED])
    await streamed.chat(copy.deepcopy(messages), system="Be brief.", options=dict(options), stream=True)

    assert len(calls) == 1
    assert calls[0]["path"] == "/responses"
    assert calls[0]["payload"] == {**nonstream_body, "stream": True}


# ---------------------------------------------------------------------------
# DoD-2 — output text pieces
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_output_text_deltas_reach_receiver_in_order__DoD2(monkeypatch):
    """DoD-2: output-text deltas reach the receiver as pieces in order; the return is their concatenation."""
    rec = _Recorder()
    result = await _parse(monkeypatch, [_text("Hel"), _text("lo, "), _text("world"), _COMPLETED], rec)

    assert rec.pieces == ["Hel", "lo, ", "world"]
    assert result == "Hello, world"


@pytest.mark.asyncio
async def test_llamaswap_streamed_chat_delivers_text_pieces_and_returns_them__DoD2(monkeypatch):
    """DoD-2: through ``chat(stream=True)`` the receiver gets the text pieces and the call returns them joined."""
    client = _client()
    _fake_stream_post(monkeypatch, client)
    _fake_sse(monkeypatch, client, [_text("One "), _text("two"), _COMPLETED])
    rec = _Recorder()

    result = await client.chat(copy.deepcopy(_MESSAGES), stream=True, on_delta=rec)

    assert rec.pieces == ["One ", "two"]
    assert result == "One two"


# ---------------------------------------------------------------------------
# DoD-3 — reasoning pieces and markers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_reasoning_then_text_marks_one_block__DoD3(monkeypatch):
    """DoD-3: the first reasoning piece opens the block, the first text piece after it closes it."""
    rec = _Recorder()
    payloads = [_reasoning("Let me"), _reasoning(" think."), _text("Answer"), _COMPLETED]
    result = await _parse(monkeypatch, payloads, rec)

    assert rec.pieces == [f"{_OPEN}Let me", " think.", f"{_CLOSE}Answer"]
    assert result == "".join(rec.pieces)
    assert result == f"{_OPEN}Let me think.{_CLOSE}Answer"


@pytest.mark.asyncio
async def test_llamaswap_reasoning_summary_deltas_are_reasoning__DoD3(monkeypatch):
    """DoD-3: reasoning-summary deltas are delivered as reasoning, with the same markers."""
    rec = _Recorder()
    result = await _parse(monkeypatch, [_summary("Plan"), _summary(" steps"), _text("Done"), _COMPLETED], rec)

    assert rec.pieces == [f"{_OPEN}Plan", " steps", f"{_CLOSE}Done"]
    assert result == "".join(rec.pieces)


@pytest.mark.asyncio
async def test_llamaswap_reasoning_and_summary_share_one_run_without_separator__DoD3(monkeypatch):
    """DoD-3: reasoning text then summary text in one run stay one block with no separator inserted."""
    rec = _Recorder()
    result = await _parse(monkeypatch, [_reasoning("a"), _summary("b"), _text("c"), _COMPLETED], rec)

    assert rec.pieces == [f"{_OPEN}a", "b", f"{_CLOSE}c"]
    assert result == f"{_OPEN}ab{_CLOSE}c"


@pytest.mark.asyncio
async def test_llamaswap_interleaved_reasoning_and_text_give_two_blocks__DoD3(monkeypatch):
    """DoD-3: reasoning, text, reasoning, text gives two separate reasoning blocks."""
    rec = _Recorder()
    payloads = [_reasoning("r1"), _text("t1"), _reasoning("r2"), _text("t2"), _COMPLETED]
    result = await _parse(monkeypatch, payloads, rec)

    assert rec.pieces == [f"{_OPEN}r1", f"{_CLOSE}t1", f"{_OPEN}r2", f"{_CLOSE}t2"]
    assert result == f"{_OPEN}r1{_CLOSE}t1{_OPEN}r2{_CLOSE}t2"
    assert result.count(_OPEN) == 2
    assert result.count(_CLOSE) == 2


@pytest.mark.asyncio
async def test_llamaswap_open_block_at_completion_closed_by_own_piece__DoD3(monkeypatch):
    """DoD-3: a block still open at completion is closed by a final closing-marker piece of its own."""
    rec = _Recorder()
    result = await _parse(monkeypatch, [_reasoning("only reasoning"), _COMPLETED], rec)

    assert rec.pieces == [f"{_OPEN}only reasoning", _CLOSE]
    assert result == f"{_OPEN}only reasoning{_CLOSE}"


@pytest.mark.asyncio
async def test_llamaswap_text_then_reasoning_at_end_closed_at_completion__DoD3(monkeypatch):
    """DoD-3: reasoning after text opens a new block, closed by the final closing piece at completion."""
    rec = _Recorder()
    result = await _parse(monkeypatch, [_text("t"), _reasoning("r"), _COMPLETED], rec)

    assert rec.pieces == ["t", f"{_OPEN}r", _CLOSE]
    assert result == "".join(rec.pieces)


# ---------------------------------------------------------------------------
# DoD-4 — the observed llama-swap event sequence
# ---------------------------------------------------------------------------


def _llamaswap_observed_sequence() -> list[str]:
    """The event sequence observed on llama-swap: lifecycle, reasoning, message, done events, completed.

    The ``*.done`` events repeat the full text; they must not be delivered again (DEC-003-13). No ``[DONE]``.
    """
    reasoning_item = {"type": "reasoning", "id": "rs_1", "summary": [], "content": []}
    message_item = {"type": "message", "id": "msg_1", "role": "assistant", "content": []}
    return [
        _ev("response.created", response={"id": "resp_1", "status": "in_progress", "output": []}),
        _ev("response.in_progress", response={"id": "resp_1", "status": "in_progress", "output": []}),
        _ev("response.output_item.added", output_index=0, item=reasoning_item),
        _reasoning("Thinking"),
        _reasoning(" hard."),
        _ev("response.reasoning_text.done", item_id="rs_1", content_index=0, text="Thinking hard."),
        _ev("response.output_item.done", output_index=0, item={**reasoning_item, "status": "completed"}),
        _ev("response.output_item.added", output_index=1, item=message_item),
        _ev(
            "response.content_part.added",
            item_id="msg_1",
            output_index=1,
            content_index=0,
            part={"type": "output_text", "text": ""},
        ),
        _text("Hello"),
        _text(" there"),
        _ev("response.output_text.done", item_id="msg_1", output_index=1, content_index=0, text="Hello there"),
        _ev(
            "response.content_part.done",
            item_id="msg_1",
            output_index=1,
            content_index=0,
            part={"type": "output_text", "text": "Hello there"},
        ),
        _ev("response.output_item.done", output_index=1, item={**message_item, "status": "completed"}),
        _COMPLETED,
    ]


@pytest.mark.asyncio
async def test_llamaswap_observed_event_sequence_returns_reasoning_block_then_answer__DoD4(monkeypatch):
    """DoD-4: the observed llama-swap sequence returns the reasoning block followed by the answer text."""
    rec = _Recorder()
    result = await _parse(monkeypatch, _llamaswap_observed_sequence(), rec)

    assert result == f"{_OPEN}Thinking hard.{_CLOSE}Hello there"
    assert "".join(rec.pieces) == result


# ---------------------------------------------------------------------------
# DoD-5 — no reasoning, no markers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_stream_without_reasoning_has_no_markers__DoD5(monkeypatch):
    """DoD-5: a stream with no reasoning events gives no think markers in any piece or the return."""
    rec = _Recorder()
    payloads = [
        _ev("response.created", response={"id": "resp_1", "status": "in_progress"}),
        _text("Plain"),
        _text(" answer"),
        _ev("response.output_text.done", text="Plain answer"),
        _COMPLETED,
    ]
    result = await _parse(monkeypatch, payloads, rec)

    assert rec.pieces == ["Plain", " answer"]
    assert result == "Plain answer"
    assert all(_OPEN not in p and _CLOSE not in p for p in rec.pieces)


# ---------------------------------------------------------------------------
# DoD-6 — failure events
# ---------------------------------------------------------------------------

_FAILURE_EVENTS = [
    pytest.param(
        _ev("response.failed", response={"status": "failed", "error": {"code": "server", "message": "boom-xyz"}}),
        "boom-xyz",
        id="failed-message",
    ),
    pytest.param(
        _ev("response.failed", response={"status": "failed", "error": {"code": "fail-code-q7"}}),
        "fail-code-q7",
        id="failed-no-message-json",
    ),
    pytest.param(
        _ev("response.incomplete", response={"status": "incomplete", "incomplete_details": {"reason": "cut-r9"}}),
        "cut-r9",
        id="incomplete-reason",
    ),
    pytest.param(
        _ev("response.incomplete", response={"status": "incomplete", "incomplete_details": {"detail": "trunc-k3"}}),
        "trunc-k3",
        id="incomplete-no-reason-json",
    ),
    pytest.param(_ev("error", code="rate_limit", message="slow-down-z2"), "slow-down-z2", id="error-message"),
    pytest.param(_ev("error", code="err-code-w5"), "err-code-w5", id="error-no-message-json"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("event", "reason"), _FAILURE_EVENTS)
async def test_llamaswap_failure_event_raises_llmerror_with_reason__DoD6(monkeypatch, event, reason):
    """DoD-6: failed / incomplete / error events raise LLMError with the server's reason; nothing after arrives."""
    rec = _Recorder()
    with pytest.raises(LLMError) as excinfo:
        await _parse(monkeypatch, [_text("before"), event, _text("after"), _COMPLETED], rec)

    assert reason in str(excinfo.value)
    assert rec.pieces == ["before"]


@pytest.mark.asyncio
async def test_llamaswap_failure_inside_reasoning_delivers_no_closing_piece__DoD6(monkeypatch):
    """DoD-6: a failure while a reasoning block is open delivers nothing more, not even a closing marker."""
    rec = _Recorder()
    failed = _ev("response.failed", response={"status": "failed", "error": {"message": "engine-died"}})
    with pytest.raises(LLMError) as excinfo:
        await _parse(monkeypatch, [_reasoning("r"), failed, _text("t"), _COMPLETED], rec)

    assert "engine-died" in str(excinfo.value)
    assert rec.pieces == [f"{_OPEN}r"]


@pytest.mark.asyncio
async def test_llamaswap_streamed_chat_failure_event_raises_llmerror__DoD6(monkeypatch):
    """DoD-6: through ``chat(stream=True)`` a failed event ends the call with LLMError carrying the reason."""
    client = _client()
    _fake_stream_post(monkeypatch, client)
    failed = _ev("response.failed", response={"status": "failed", "error": {"message": "chat-fail-m1"}})
    _fake_sse(monkeypatch, client, [_text("x"), failed, _text("y"), _COMPLETED])
    rec = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await client.chat(copy.deepcopy(_MESSAGES), stream=True, on_delta=rec)

    assert "chat-fail-m1" in str(excinfo.value)
    assert rec.pieces == ["x"]


# ---------------------------------------------------------------------------
# DoD-7 — completion and early end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_stream_ending_without_completed_raises__DoD7(monkeypatch):
    """DoD-7: a stream that ends without the completed event raises LLMError."""
    rec = _Recorder()
    with pytest.raises(LLMError):
        await _parse(monkeypatch, [_text("partial"), _text(" answer")], rec)

    assert rec.pieces == ["partial", " answer"]


@pytest.mark.asyncio
async def test_llamaswap_done_sentinel_is_not_completion__DoD7(monkeypatch):
    """DoD-7: a ``[DONE]`` sentinel without the completed event still raises LLMError."""
    with pytest.raises(LLMError):
        await _parse(monkeypatch, [_text("partial"), "[DONE]"], _Recorder())


@pytest.mark.asyncio
async def test_llamaswap_empty_stream_raises__DoD7(monkeypatch):
    """DoD-7: a stream with no events at all never completed, so it raises LLMError."""
    with pytest.raises(LLMError):
        await _parse(monkeypatch, [], _Recorder())


@pytest.mark.asyncio
async def test_llamaswap_completed_returns_and_ignores_anything_after__DoD7(monkeypatch):
    """DoD-7: the completed event ends the call normally; text and failures after it are not delivered."""
    rec = _Recorder()
    failed = _ev("response.failed", response={"status": "failed", "error": {"message": "late"}})
    result = await _parse(monkeypatch, [_text("done"), _COMPLETED, _text(" extra"), failed, _reasoning("r")], rec)

    assert result == "done"
    assert rec.pieces == ["done"]


# ---------------------------------------------------------------------------
# DoD-8 — skipped payloads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_unreadable_and_unknown_payloads_are_skipped__DoD8(monkeypatch):
    """DoD-8: unknown types, non-JSON, non-object and type-less payloads and empty deltas are skipped silently."""
    rec = _Recorder()
    payloads = [
        _ev("response.some_future_event", delta="ignored-unknown"),
        "not json at all {",
        "[DONE]",
        "[1, 2, 3]",
        '"a string"',
        "42",
        "null",
        json.dumps({"delta": "no-type"}),
        json.dumps({"type": 7, "delta": "non-string-type"}),
        _ev("response.output_text.delta", delta=""),
        _ev("response.output_text.delta"),
        _ev("response.reasoning_text.delta", delta=""),
        _ev("response.reasoning_summary_text.delta"),
        _text("kept"),
        _COMPLETED,
    ]
    result = await _parse(monkeypatch, payloads, rec)

    assert rec.pieces == ["kept"]
    assert result == "kept"


@pytest.mark.asyncio
async def test_llamaswap_only_skippable_payloads_give_empty_text__DoD8(monkeypatch):
    """DoD-8: a completed stream of only skippable payloads makes no receiver call and returns empty text."""
    rec = _Recorder()
    payloads = ["garbage", _ev("response.in_progress"), _ev("response.output_text.delta", delta=""), _COMPLETED]
    result = await _parse(monkeypatch, payloads, rec)

    assert rec.pieces == []
    assert result == ""


# ---------------------------------------------------------------------------
# DoD-9 — non-2xx reply to the streamed request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_streamed_chat_404_raises_llmerror_with_address_and_reply__DoD9(monkeypatch):
    """DoD-9: a 404 with a JSON error body raises LLMError with the address and reply, never the credential."""
    client = _client(bearer_token=_SECRET)
    body = json.dumps({"error": {"code": 404, "message": "model-missing-p4 not found", "type": "not_found_error"}})
    session = _FakeSession(status=404, body=body.encode("utf-8"), content_type="application/json")
    client.session = session
    rec = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await client.chat(copy.deepcopy(_MESSAGES), stream=True, on_delta=rec)

    err = excinfo.value
    text = str(err)
    assert _RESPONSES_URL in text
    assert "model-missing-p4 not found" in text
    assert _SECRET not in text
    assert _SECRET not in (err.message or "")
    assert _SECRET not in (err.description or "")
    assert rec.pieces == []
    assert len(session.calls) == 1


# ---------------------------------------------------------------------------
# DoD-10 — receiver rules
# ---------------------------------------------------------------------------


class _ReceiverBoom(Exception):
    """A receiver-specific failure, distinct from any library error."""


@pytest.mark.asyncio
async def test_llamaswap_receiver_failure_propagates_and_stops_delivery__DoD10(monkeypatch):
    """DoD-10: a receiver failure ends the parse with that same exception; no later piece is delivered."""
    seen: list[str] = []
    boom = _ReceiverBoom("receiver broke")

    async def receiver(piece: str) -> None:
        seen.append(piece)
        if piece == "b":
            raise boom

    with pytest.raises(_ReceiverBoom) as excinfo:
        await _parse(monkeypatch, [_text("a"), _text("b"), _text("c"), _COMPLETED], receiver)

    assert excinfo.value is boom
    assert seen == ["a", "b"]


@pytest.mark.asyncio
async def test_llamaswap_streamed_chat_receiver_failure_propagates_unchanged__DoD10(monkeypatch):
    """DoD-10: through ``chat(stream=True)`` the receiver's own exception ends the call unchanged."""
    client = _client()
    _fake_stream_post(monkeypatch, client)
    _fake_sse(monkeypatch, client, [_text("a"), _text("b"), _text("c"), _COMPLETED])
    seen: list[str] = []
    boom = _ReceiverBoom("receiver broke in chat")

    async def receiver(piece: str) -> None:
        seen.append(piece)
        raise boom

    with pytest.raises(_ReceiverBoom) as excinfo:
        await client.chat(copy.deepcopy(_MESSAGES), stream=True, on_delta=receiver)

    assert excinfo.value is boom
    assert seen == ["a"]


@pytest.mark.asyncio
async def test_llamaswap_async_receiver_is_awaited__DoD10(monkeypatch):
    """DoD-10: an async receiver is awaited; it records only after yielding control, so pieces prove the await."""
    seen: list[str] = []

    async def receiver(piece: str) -> None:
        await asyncio.sleep(0)
        seen.append(piece)

    result = await _parse(monkeypatch, [_reasoning("r"), _text("x"), _text("y"), _COMPLETED], receiver)

    assert seen == [f"{_OPEN}r", f"{_CLOSE}x", "y"]
    assert result == "".join(seen)


@pytest.mark.asyncio
async def test_llamaswap_plain_receiver_is_accepted__DoD10(monkeypatch):
    """DoD-10: a plain (non-async) receiver is called for each piece; only awaitable results are awaited."""
    seen: list[str] = []

    def receiver(piece: str) -> None:
        seen.append(piece)

    result = await _parse(monkeypatch, [_text("x"), _text("y"), _COMPLETED], receiver)

    assert seen == ["x", "y"]
    assert result == "xy"


# ---------------------------------------------------------------------------
# DoD-11 — no receiver, and generate through chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_streamed_chat_without_receiver_returns_full_text__DoD11(monkeypatch):
    """DoD-11: a streamed chat without a receiver returns the full text, reasoning marked."""
    client = _client()
    _fake_stream_post(monkeypatch, client)
    _fake_sse(monkeypatch, client, [_reasoning("why"), _text("Full "), _text("text"), _COMPLETED])

    result = await client.chat(copy.deepcopy(_MESSAGES), stream=True)

    assert result == f"{_OPEN}why{_CLOSE}Full text"


@pytest.mark.asyncio
async def test_llamaswap_parser_without_receiver_returns_full_text__DoD11(monkeypatch):
    """DoD-11: the parser with no receiver still returns everything it would have delivered."""
    result = await _parse(monkeypatch, [_text("a"), _reasoning("b"), _COMPLETED], None)

    assert result == f"a{_OPEN}b{_CLOSE}"


@pytest.mark.asyncio
async def test_llamaswap_streamed_generate_matches_one_message_chat__DoD11(monkeypatch):
    """DoD-11: a streamed generate gives the same pieces, return and request as the one-message chat."""
    payloads = [_reasoning("hm"), _text("Gen "), _text("out"), _COMPLETED]

    chat_client = _client()
    chat_calls = _fake_stream_post(monkeypatch, chat_client)
    _fake_sse(monkeypatch, chat_client, payloads)
    chat_rec = _Recorder()
    chat_result = await chat_client.chat(
        [{"role": "user", "content": "Write a line."}], system="Be terse.", stream=True, on_delta=chat_rec
    )

    gen_client = _client()
    gen_calls = _fake_stream_post(monkeypatch, gen_client)
    _fake_sse(monkeypatch, gen_client, payloads)
    gen_rec = _Recorder()
    gen_result = await gen_client.generate("Write a line.", system="Be terse.", stream=True, on_delta=gen_rec)

    assert gen_rec.pieces == chat_rec.pieces
    assert gen_result == chat_result
    assert len(gen_calls) == 1
    assert gen_calls[0]["path"] == "/responses"
    assert gen_calls[0]["payload"] == chat_calls[0]["payload"]


# ---------------------------------------------------------------------------
# DoD-12 — tracing parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_tracing_keeps_pieces_and_return__DoD12(monkeypatch, tmp_path):
    """DoD-12: with tracing enabled the receiver gets the same pieces and the call returns the same text."""
    payloads = [_reasoning("trace "), _reasoning("me"), _text("Traced "), _text("answer"), _COMPLETED]

    untraced = _client()
    _fake_stream_post(monkeypatch, untraced)
    _fake_sse(monkeypatch, untraced, payloads)
    untraced_rec = _Recorder()
    untraced_result = await untraced.chat(copy.deepcopy(_MESSAGES), stream=True, on_delta=untraced_rec)

    # Tracing is read at construction (DEC-003-16), so the env is set before the traced client is built.
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(tmp_path))
    traced = _client()
    _fake_stream_post(monkeypatch, traced)
    _fake_sse(monkeypatch, traced, payloads)
    traced_rec = _Recorder()
    traced_result = await traced.chat(copy.deepcopy(_MESSAGES), stream=True, on_delta=traced_rec)

    assert traced_rec.pieces == untraced_rec.pieces
    assert traced_result == untraced_result
    assert traced_result == f"{_OPEN}trace me{_CLOSE}Traced answer"


@pytest.mark.asyncio
async def test_llamaswap_tracing_without_receiver_returns_same_text__DoD12(monkeypatch, tmp_path):
    """DoD-12: with tracing enabled and no receiver, the streamed chat still returns the full text."""
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(tmp_path))
    client = _client()
    _fake_stream_post(monkeypatch, client)
    _fake_sse(monkeypatch, client, [_text("no "), _text("receiver"), _COMPLETED])

    result = await client.chat(copy.deepcopy(_MESSAGES), stream=True)

    assert result == "no receiver"
