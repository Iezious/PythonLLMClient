"""Unit tests for streamed chat on the OpenAI Responses class (DEC-003-13).

Covers the Responses stream parser ``OpenAIAPIClient._read_responses_stream`` and the streamed ``chat`` /
``generate`` path: the single streamed request to ``/responses`` (DEC-003-11, DEC-003-16), text and reasoning
pieces with per-run ``<think>`` / ``</think>`` markers (DEC-000-20, DEC-003-13), failure events and early
stream end as ``LLMError``, skipped payloads, non-2xx replies, receiver failures (DEC-000-23) and tracing
(DEC-000-48). Every expected value comes from the step's Definition of done and the cited decisions.
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
MODEL = "gpt-4o-mini"

MESSAGES = [
    {"role": "user", "content": "Name a colour."},
    {"role": "assistant", "content": "Blue."},
    {"role": "user", "content": "Another one."},
]
SYSTEM = "You answer in one word."
OPTIONS = {"temperature": 0.25, "max_tokens": 64}
PROMPT = "Name a fruit."


# ---------------------------------------------------------------------------
# Stream event builders (Responses wire vocabulary: DEC-003-13, DEC-arch-11)
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


def _summary(delta: str) -> str:
    """A reasoning-summary delta event."""
    return _event({"type": "response.reasoning_summary_text.delta", "item_id": "rs_1", "delta": delta})


CREATED = _event({"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}})
IN_PROGRESS = _event({"type": "response.in_progress", "response": {"id": "resp_1", "status": "in_progress"}})
COMPLETED = _event({"type": "response.completed", "response": {"id": "resp_1", "status": "completed"}})


class _Recorder:
    """Async receiver that records every piece it is given, in order."""

    def __init__(self) -> None:
        self.pieces: list[str] = []

    async def __call__(self, piece: str) -> None:
        self.pieces.append(piece)


def _client() -> OpenAIAPIClient:
    """A Responses-class client built directly with an explicit test credential (tests/AGENTS.md)."""
    return OpenAIAPIClient(model=MODEL, base_url=BASE_URL, bearer_token=CREDENTIAL)


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch):
    """Clear the trace switches so every test starts untraced regardless of the developer's environment;
    the tracing test (DoD-11) turns them back on with ``monkeypatch.setenv`` before building its client."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _install_sse(monkeypatch, client: OpenAIAPIClient, payloads: list[str], log: list[str] | None = None) -> None:
    """Replace the ``_iter_sse_data`` seam (DEC-003-16) with a fake yielding ``payloads`` as SSE data strings.

    When ``log`` is given, ``"read"`` is appended each time the parser asks for the next payload, so a test can
    see whether a receiver call finished before the next read (DEC-000-23).
    """

    async def fake_iter_sse_data(_response):
        for payload in payloads:
            if log is not None:
                log.append("read")
            yield payload
        if log is not None:
            log.append("read")

    monkeypatch.setattr(client, "_iter_sse_data", fake_iter_sse_data)


def _install_stream_post(monkeypatch, client: OpenAIAPIClient, payloads: list[str]) -> list[dict]:
    """Replace ``_stream_post`` with a fake async context manager (DEC-003-16) and ``_iter_sse_data`` with canned
    payloads; returns the list of recorded ``(path, payload)`` calls (payload deep-copied against aliasing)."""
    calls: list[dict] = []

    @asynccontextmanager
    async def fake_stream_post(path, payload):
        calls.append({"path": path, "payload": copy.deepcopy(payload)})
        yield None

    monkeypatch.setattr(client, "_stream_post", fake_stream_post)
    _install_sse(monkeypatch, client, payloads)
    return calls


def _capture_requests(monkeypatch, client: OpenAIAPIClient, reply: dict) -> list[dict]:
    """Replace ``_request`` with a fake recording ``method``, ``path`` and a deep copy of the JSON body and
    answering with ``reply``; used to obtain the non-streamed body a streamed call is compared against."""
    records: list[dict] = []

    async def fake_request(method, path, **kw):
        records.append({"method": method, "path": path, "json": copy.deepcopy(kw.get("json"))})
        return reply

    monkeypatch.setattr(client, "_request", fake_request)
    return records


# A completed non-streamed Responses reply (DEC-003-12 shape) used only to let the non-streamed call finish.
_NON_STREAMED_REPLY = {
    "status": "completed",
    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Red."}]}],
}


# ---------------------------------------------------------------------------
# Fake HTTP session (for DoD-8: the real _stream_post runs against it)
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


def _error_session(status: int, body: str) -> _FakeSession:
    """A session answering every request with a non-2xx reply whose body is ``body``."""
    data = body.encode("utf-8")
    return _FakeSession(lambda method, url, kw: _FakeResponse(url, status, data, "application/json"))


# ---------------------------------------------------------------------------
# DoD-1 — one streamed request to /responses, body = non-streamed body with stream true
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_streamed_chat_posts_non_streamed_body_with_stream_true__DoD1(monkeypatch):
    """A streamed chat opens exactly one streamed request to ``/responses`` whose body equals the non-streamed
    body except ``stream`` is true (DoD-1, DEC-003-11, DEC-003-16)."""
    client = _client()
    records = _capture_requests(monkeypatch, client, _NON_STREAMED_REPLY)
    await client.chat(list(MESSAGES), system=SYSTEM, options=dict(OPTIONS))
    assert len(records) == 1
    assert records[0]["method"] == "POST"
    assert records[0]["path"] == "/responses"
    non_streamed_body = records[0]["json"]
    assert non_streamed_body["stream"] is False

    calls = _install_stream_post(monkeypatch, client, [_text("Green."), COMPLETED])
    await client.chat(list(MESSAGES), system=SYSTEM, options=dict(OPTIONS), stream=True, on_delta=_Recorder())

    assert len(calls) == 1
    assert calls[0]["path"] == "/responses"
    assert calls[0]["payload"]["stream"] is True
    assert calls[0]["payload"] == {**non_streamed_body, "stream": True}
    # The streamed call must not also send a non-streamed request.
    assert len(records) == 1


# ---------------------------------------------------------------------------
# DoD-2 — output-text deltas in order; return = concatenation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_stream_text_deltas_reach_receiver_in_order__DoD2(monkeypatch):
    """Output-text deltas reach the receiver as pieces in order and the call returns their concatenation
    (DoD-2, US-008.AC-1, US-057.AC-1, DEC-003-13)."""
    client = _client()
    _install_sse(monkeypatch, client, [CREATED, IN_PROGRESS, _text("Hel"), _text("lo"), _text(" world"), COMPLETED])
    recorder = _Recorder()

    result = await client._read_responses_stream(None, recorder)

    assert recorder.pieces == ["Hel", "lo", " world"]
    assert result == "Hello world"


@pytest.mark.asyncio
async def test_openai_streamed_chat_delivers_text_pieces_and_returns_them__DoD2(monkeypatch):
    """Through ``chat(stream=True)``, text deltas reach the receiver in order and the call returns their
    concatenation (DoD-2, US-008.AC-1)."""
    client = _client()
    _install_stream_post(monkeypatch, client, [CREATED, _text("One"), _text(", two"), COMPLETED])
    recorder = _Recorder()

    result = await client.chat(list(MESSAGES), stream=True, on_delta=recorder)

    assert recorder.pieces == ["One", ", two"]
    assert result == "One, two"


# ---------------------------------------------------------------------------
# DoD-3 — reasoning pieces and per-run markers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_stream_reasoning_text_then_answer_is_marked__DoD3(monkeypatch):
    """Reasoning-text deltas reach the receiver as reasoning: the first reasoning piece starts with ``<think>``,
    the first text piece after it starts with ``</think>``; the return equals everything delivered
    (DoD-3, US-026.AC-1, US-026.AC-2, DEC-000-20)."""
    client = _client()
    _install_sse(monkeypatch, client, [_reasoning("Think"), _reasoning("ing"), _text("Ans"), _text("wer"), COMPLETED])
    recorder = _Recorder()

    result = await client._read_responses_stream(None, recorder)

    assert recorder.pieces == ["<think>Think", "ing", "</think>Ans", "wer"]
    assert result == "".join(recorder.pieces)
    assert result == "<think>Thinking</think>Answer"


@pytest.mark.asyncio
async def test_openai_stream_reasoning_summary_then_answer_is_marked__DoD3(monkeypatch):
    """Reasoning-summary deltas are reasoning too, with no separator inserted between parts
    (DoD-3, US-059.AC-1, DEC-003-13)."""
    client = _client()
    _install_sse(monkeypatch, client, [_summary("Step one."), _summary("Step two."), _text("Done"), COMPLETED])
    recorder = _Recorder()

    result = await client._read_responses_stream(None, recorder)

    assert recorder.pieces == ["<think>Step one.", "Step two.", "</think>Done"]
    assert result == "<think>Step one.Step two.</think>Done"


@pytest.mark.asyncio
async def test_openai_stream_summary_and_reasoning_text_in_one_run_form_one_block__DoD3(monkeypatch):
    """Summary and reasoning-text deltas in one run belong to the same reasoning block: one opening marker only
    (DoD-3, DEC-003-13)."""
    client = _client()
    _install_sse(monkeypatch, client, [_summary("S"), _reasoning("R"), _text("T"), COMPLETED])
    recorder = _Recorder()

    result = await client._read_responses_stream(None, recorder)

    assert recorder.pieces == ["<think>S", "R", "</think>T"]
    assert result == "<think>SR</think>T"


@pytest.mark.asyncio
async def test_openai_stream_interleaved_reasoning_gives_two_blocks__DoD3(monkeypatch):
    """Reasoning, text, reasoning, text gives two separate reasoning blocks (DoD-3, US-026.AC-4, DEC-000-20)."""
    client = _client()
    _install_sse(monkeypatch, client, [_reasoning("R1"), _text("T1"), _reasoning("R2"), _text("T2"), COMPLETED])
    recorder = _Recorder()

    result = await client._read_responses_stream(None, recorder)

    assert recorder.pieces == ["<think>R1", "</think>T1", "<think>R2", "</think>T2"]
    assert result == "<think>R1</think>T1<think>R2</think>T2"


@pytest.mark.asyncio
async def test_openai_stream_open_block_at_completion_is_closed_by_own_piece__DoD3(monkeypatch):
    """A reasoning block still open at completion is closed by a final ``</think>`` piece of its own; the return
    equals everything the receiver got (DoD-3, DEC-003-13)."""
    client = _client()
    _install_sse(monkeypatch, client, [_text("Pre"), _reasoning("Late thought"), COMPLETED])
    recorder = _Recorder()

    result = await client._read_responses_stream(None, recorder)

    assert recorder.pieces == ["Pre", "<think>Late thought", "</think>"]
    assert result == "Pre<think>Late thought</think>"


@pytest.mark.asyncio
async def test_openai_stream_reasoning_only_is_closed_at_completion__DoD3(monkeypatch):
    """A stream carrying only reasoning delivers the opening marker on the first piece and a closing-marker piece
    at completion (DoD-3, US-026.AC-1)."""
    client = _client()
    _install_sse(monkeypatch, client, [_reasoning("a"), _reasoning("b"), COMPLETED])
    recorder = _Recorder()

    result = await client._read_responses_stream(None, recorder)

    assert recorder.pieces == ["<think>a", "b", "</think>"]
    assert result == "<think>ab</think>"


# ---------------------------------------------------------------------------
# DoD-4 — no reasoning, no markers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_stream_without_reasoning_has_no_markers__DoD4(monkeypatch):
    """A stream with no reasoning events gives no ``<think>`` / ``</think>`` markers (DoD-4, US-026.AC-3)."""
    client = _client()
    _install_sse(monkeypatch, client, [CREATED, _text("Plain"), _text(" answer"), COMPLETED])
    recorder = _Recorder()

    result = await client._read_responses_stream(None, recorder)

    assert recorder.pieces == ["Plain", " answer"]
    assert result == "Plain answer"
    for piece in [*recorder.pieces, result]:
        assert "<think>" not in piece
        assert "</think>" not in piece


# ---------------------------------------------------------------------------
# DoD-5 — failed / incomplete / error events raise LLMError with the server's reason
# ---------------------------------------------------------------------------


_FAILED = {
    "type": "response.failed",
    "response": {"id": "resp_1", "status": "failed", "error": {"code": "server_error", "message": "Model crashed"}},
}
_INCOMPLETE = {
    "type": "response.incomplete",
    "response": {"id": "resp_1", "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
}
_ERROR = {"type": "error", "code": "rate_limit_exceeded", "message": "Rate limit reached"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "reason"),
    [
        pytest.param(_FAILED, "Model crashed", id="failed"),
        pytest.param(_INCOMPLETE, "max_output_tokens", id="incomplete"),
        pytest.param(_ERROR, "Rate limit reached", id="error"),
    ],
)
async def test_openai_stream_failure_event_raises_with_server_reason__DoD5(monkeypatch, event, reason):
    """A failed, an incomplete and an error event each raise ``LLMError`` whose description is the server's
    reason, and no piece after the event reaches the receiver (DoD-5, US-060.AC-2, DEC-003-13)."""
    client = _client()
    _install_sse(monkeypatch, client, [_text("Before"), _event(event), _text("After"), _reasoning("x"), COMPLETED])
    recorder = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await client._read_responses_stream(None, recorder)

    assert excinfo.value.description == reason
    assert reason in str(excinfo.value)
    assert recorder.pieces == ["Before"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "reason_value"),
    [
        pytest.param(
            {"type": "response.failed", "response": {"status": "failed", "error": {"code": "server_error_7q"}}},
            {"code": "server_error_7q"},
            id="failed-without-string-message",
        ),
        pytest.param(
            {"type": "response.incomplete", "response": {"status": "incomplete", "incomplete_details": {"x": "cut"}}},
            {"x": "cut"},
            id="incomplete-without-reason",
        ),
        pytest.param(
            {"type": "error", "code": "overloaded_3k"},
            {"type": "error", "code": "overloaded_3k"},
            id="error-without-string-message",
        ),
    ],
)
async def test_openai_stream_failure_event_without_message_carries_reason_as_json__DoD5(
    monkeypatch, event, reason_value
):
    """When the failure carries no string message / reason, the description is the reason value as JSON text
    (DoD-5, DEC-003-13): the failed event's ``response.error``, the incomplete event's
    ``response.incomplete_details``, the error event itself."""
    client = _client()
    _install_sse(monkeypatch, client, [_event(event), _text("After"), COMPLETED])
    recorder = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await client._read_responses_stream(None, recorder)

    assert json.loads(excinfo.value.description) == reason_value
    assert recorder.pieces == []


@pytest.mark.asyncio
async def test_openai_streamed_chat_failure_event_raises_llm_error__DoD5(monkeypatch):
    """Through ``chat(stream=True)``, a failed event ends the call with ``LLMError`` carrying the server's reason
    (DoD-5, US-060.AC-2)."""
    client = _client()
    _install_stream_post(monkeypatch, client, [_text("Partial"), _event(_FAILED), _text("Never")])
    recorder = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await client.chat(list(MESSAGES), stream=True, on_delta=recorder)

    assert "Model crashed" in str(excinfo.value)
    assert recorder.pieces == ["Partial"]


# ---------------------------------------------------------------------------
# DoD-6 — completion: missing completed raises (also after [DONE]); after completed nothing is delivered
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payloads",
    [
        pytest.param([CREATED, _text("A"), _text("B")], id="ends-after-text"),
        pytest.param([_text("A"), "[DONE]"], id="done-sentinel"),
        pytest.param([_reasoning("R"), _text("A"), "[DONE]"], id="done-after-reasoning"),
        pytest.param([], id="empty-stream"),
    ],
)
async def test_openai_stream_ending_without_completed_raises__DoD6(monkeypatch, payloads):
    """A stream that ends without ``response.completed`` raises ``LLMError``, also when a ``[DONE]`` sentinel
    arrived (DoD-6, US-024.AC-2, DEC-003-13)."""
    client = _client()
    _install_sse(monkeypatch, client, payloads)

    with pytest.raises(LLMError):
        await client._read_responses_stream(None, _Recorder())


@pytest.mark.asyncio
async def test_openai_stream_completed_returns_and_ignores_later_events__DoD6(monkeypatch):
    """A stream ending with ``response.completed`` returns normally; nothing after it is delivered, not even a
    later failure event (DoD-6, DEC-003-13)."""
    client = _client()
    _install_sse(monkeypatch, client, [_text("A"), COMPLETED, _text("Z"), _reasoning("late"), _event(_FAILED)])
    recorder = _Recorder()

    result = await client._read_responses_stream(None, recorder)

    assert recorder.pieces == ["A"]
    assert result == "A"


@pytest.mark.asyncio
async def test_openai_streamed_chat_without_completed_raises__DoD6(monkeypatch):
    """Through ``chat(stream=True)``, a stream cut before ``response.completed`` ends with ``LLMError``
    (DoD-6, US-024.AC-2)."""
    client = _client()
    _install_stream_post(monkeypatch, client, [_text("Cut"), "[DONE]"])

    with pytest.raises(LLMError):
        await client.chat(list(MESSAGES), stream=True, on_delta=_Recorder())


# ---------------------------------------------------------------------------
# DoD-7 — skipped payloads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "junk",
    [
        pytest.param("not json at all", id="non-json"),
        pytest.param("[DONE]", id="done-sentinel"),
        pytest.param("{truncated", id="broken-json"),
        pytest.param("42", id="number"),
        pytest.param('"a string"', id="string"),
        pytest.param("[1, 2, 3]", id="list"),
        pytest.param("null", id="null"),
        pytest.param(_event({"delta": "X"}), id="missing-type"),
        pytest.param(_event({"type": 7, "delta": "X"}), id="non-string-type"),
        pytest.param(_event({"type": "response.brand_new_event", "delta": "X"}), id="unknown-type"),
        pytest.param(_event({"type": "response.output_item.added", "item": {"type": "message"}}), id="item-added"),
        pytest.param(_event({"type": "response.content_part.added", "part": {"type": "output_text"}}), id="part"),
        pytest.param(_event({"type": "response.output_text.done", "text": "X"}), id="text-done"),
        pytest.param(_event({"type": "response.output_item.done", "item": {"type": "message"}}), id="item-done"),
        pytest.param(_event({"type": "response.reasoning_text.done", "text": "X"}), id="reasoning-done"),
    ],
)
async def test_openai_stream_skips_unusable_payloads__DoD7(monkeypatch, junk):
    """Unknown event types, non-JSON payloads and non-object payloads are skipped without error and without a
    receiver call (DoD-7, DEC-003-13)."""
    client = _client()
    _install_sse(monkeypatch, client, [junk, _text("ok"), junk, COMPLETED])
    recorder = _Recorder()

    result = await client._read_responses_stream(None, recorder)

    assert recorder.pieces == ["ok"]
    assert result == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "empty",
    [
        pytest.param(_event({"type": "response.output_text.delta", "delta": ""}), id="empty-text"),
        pytest.param(_event({"type": "response.output_text.delta"}), id="missing-text"),
        pytest.param(_event({"type": "response.reasoning_text.delta", "delta": ""}), id="empty-reasoning"),
        pytest.param(_event({"type": "response.reasoning_text.delta"}), id="missing-reasoning"),
        pytest.param(_event({"type": "response.reasoning_summary_text.delta", "delta": ""}), id="empty-summary"),
        pytest.param(_event({"type": "response.reasoning_summary_text.delta"}), id="missing-summary"),
    ],
)
async def test_openai_stream_empty_deltas_deliver_nothing__DoD7(monkeypatch, empty):
    """Empty or missing deltas deliver nothing: no receiver call, no marker (DoD-7, DEC-003-13)."""
    client = _client()
    _install_sse(monkeypatch, client, [empty, _text("ok"), empty, COMPLETED])
    recorder = _Recorder()

    result = await client._read_responses_stream(None, recorder)

    assert recorder.pieces == ["ok"]
    assert result == "ok"


# ---------------------------------------------------------------------------
# DoD-8 — non-2xx streamed reply
# ---------------------------------------------------------------------------


_SERVER_ERROR_BODY = json.dumps(
    {"error": {"message": "The model 'gpt-zz' does not exist", "type": "invalid_request_error", "code": None}}
)


@pytest.mark.asyncio
async def test_openai_streamed_chat_non_2xx_raises_with_address_and_reply_not_credential__DoD8(monkeypatch):
    """A streamed request answered with a non-2xx reply raises ``LLMError`` carrying the address and the server's
    reply, and never the credential (DoD-8, US-060.AC-1, US-060.AC-4, DEC-arch-5)."""
    client = _client()
    # A fake session so the real streamed-request helper runs (DEC-003-16); _stream_post is not patched.
    client.session = _error_session(404, _SERVER_ERROR_BODY)
    recorder = _Recorder()

    with pytest.raises(LLMError) as excinfo:
        await client.chat(list(MESSAGES), stream=True, on_delta=recorder)

    err = excinfo.value
    rendered = str(err)
    assert RESPONSES_URL in rendered
    assert "404" in rendered
    assert _SERVER_ERROR_BODY in rendered
    assert CREDENTIAL not in f"{rendered} {err.message} {err.description} {err.args!r}"
    assert recorder.pieces == []


# ---------------------------------------------------------------------------
# DoD-9 — receiver failures propagate unchanged; async receiver awaited
# ---------------------------------------------------------------------------


class _ReceiverBoom(Exception):
    """A receiver-side failure distinct from any library error."""


@pytest.mark.asyncio
async def test_openai_stream_receiver_failure_propagates_unchanged__DoD9(monkeypatch):
    """A receiver failure ends the call with that same exception and no later piece is delivered
    (DoD-9, US-022.AC-1, US-022.AC-2, DEC-000-23)."""
    client = _client()
    _install_sse(monkeypatch, client, [_text("a"), _text("b"), _text("c"), COMPLETED])
    boom = _ReceiverBoom("receiver broke")
    seen: list[str] = []

    async def failing_receiver(piece: str) -> None:
        seen.append(piece)
        if piece == "b":
            raise boom

    with pytest.raises(_ReceiverBoom) as excinfo:
        await client._read_responses_stream(None, failing_receiver)

    assert excinfo.value is boom
    assert seen == ["a", "b"]


@pytest.mark.asyncio
async def test_openai_streamed_chat_receiver_failure_propagates_unchanged__DoD9(monkeypatch):
    """Through ``chat(stream=True)``, a receiver failure surfaces as the same exception, not wrapped
    (DoD-9, US-022.AC-2, DEC-000-23)."""
    client = _client()
    _install_stream_post(monkeypatch, client, [_reasoning("r"), _text("a"), _text("b"), COMPLETED])
    boom = _ReceiverBoom("receiver broke in chat")
    seen: list[str] = []

    async def failing_receiver(piece: str) -> None:
        seen.append(piece)
        raise boom

    with pytest.raises(_ReceiverBoom) as excinfo:
        await client.chat(list(MESSAGES), stream=True, on_delta=failing_receiver)

    assert excinfo.value is boom
    assert seen == ["<think>r"]


@pytest.mark.asyncio
async def test_openai_stream_async_receiver_is_awaited_before_next_read__DoD9(monkeypatch):
    """An async receiver is awaited: each piece is fully handled (past an ``await`` inside the receiver) before
    the next payload is read (DoD-9, US-022.AC-1, DEC-000-23)."""
    client = _client()
    log: list[str] = []
    _install_sse(monkeypatch, client, [_text("a"), _text("b"), COMPLETED], log=log)

    async def slow_receiver(piece: str) -> None:
        # Yield to the loop first: an un-awaited coroutine would never reach the append below.
        await asyncio.sleep(0)
        log.append(f"got:{piece}")

    result = await client._read_responses_stream(None, slow_receiver)

    assert result == "ab"
    # Three payload reads (a, b, completed); the parser stops at completion, so no fourth read is required.
    assert log[:5] == ["read", "got:a", "read", "got:b", "read"]


# ---------------------------------------------------------------------------
# DoD-10 — no receiver; streamed generate equals one-message chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_stream_parser_without_receiver_returns_full_text__DoD10(monkeypatch):
    """The parser without a receiver returns the full text, markers included (DoD-10, US-021.AC-1)."""
    client = _client()
    _install_sse(monkeypatch, client, [_reasoning("R"), _text("T1"), _text("T2"), COMPLETED])

    result = await client._read_responses_stream(None, None)

    assert result == "<think>R</think>T1T2"


@pytest.mark.asyncio
async def test_openai_streamed_chat_without_receiver_returns_full_text__DoD10(monkeypatch):
    """A streamed ``chat`` without a receiver returns the full text (DoD-10, US-021.AC-1)."""
    client = _client()
    _install_stream_post(monkeypatch, client, [CREATED, _text("Full"), _text(" text"), COMPLETED])

    result = await client.chat(list(MESSAGES), stream=True)

    assert result == "Full text"


@pytest.mark.asyncio
async def test_openai_streamed_generate_matches_one_message_chat__DoD10(monkeypatch):
    """A streamed ``generate`` gives the same pieces, the same return and the same request as the equivalent
    one-message ``chat`` (DoD-10, DEC-000-8, DEC-003-11)."""
    payloads = [_summary("Plan."), _text("Apple"), _text("."), COMPLETED]

    chat_client = _client()
    chat_calls = _install_stream_post(monkeypatch, chat_client, payloads)
    chat_recorder = _Recorder()
    chat_result = await chat_client.chat(
        [{"role": "user", "content": PROMPT}], system=SYSTEM, stream=True, on_delta=chat_recorder
    )

    gen_client = _client()
    gen_calls = _install_stream_post(monkeypatch, gen_client, payloads)
    gen_recorder = _Recorder()
    gen_result = await gen_client.generate(PROMPT, system=SYSTEM, stream=True, on_delta=gen_recorder)

    assert chat_recorder.pieces == ["<think>Plan.", "</think>Apple", "."]
    assert gen_recorder.pieces == chat_recorder.pieces
    assert gen_result == chat_result == "<think>Plan.</think>Apple."
    assert len(gen_calls) == 1
    assert gen_calls[0]["path"] == "/responses"
    assert gen_calls[0]["payload"] == chat_calls[0]["payload"]


# ---------------------------------------------------------------------------
# DoD-11 — tracing does not change pieces or return
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_streamed_chat_with_tracing_matches_untraced__DoD11(monkeypatch, tmp_path):
    """With tracing enabled, the receiver gets the same pieces and the call returns the same text as without
    tracing (DoD-11, US-050.AC-4, DEC-000-48)."""
    payloads = [CREATED, _reasoning("Why"), _text("Because"), _reasoning("More"), _text("."), COMPLETED]

    plain_client = _client()
    _install_stream_post(monkeypatch, plain_client, payloads)
    plain_recorder = _Recorder()
    plain_result = await plain_client.chat(list(MESSAGES), system=SYSTEM, stream=True, on_delta=plain_recorder)

    # The tracer reads its switches at construction (DEC-003-16), so set them before building the traced client.
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(tmp_path))
    traced_client = _client()
    _install_stream_post(monkeypatch, traced_client, payloads)
    traced_recorder = _Recorder()
    traced_result = await traced_client.chat(list(MESSAGES), system=SYSTEM, stream=True, on_delta=traced_recorder)

    expected = ["<think>Why", "</think>Because", "<think>More", "</think>."]
    assert plain_recorder.pieces == expected
    assert traced_recorder.pieces == expected
    assert traced_result == plain_result == "".join(expected)


@pytest.mark.asyncio
async def test_openai_streamed_chat_with_tracing_and_no_receiver_returns_full_text__DoD11(monkeypatch, tmp_path):
    """With tracing enabled and no receiver, the streamed call still returns the full text (DoD-11, DEC-000-48)."""
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(tmp_path))
    client = _client()
    _install_stream_post(monkeypatch, client, [_text("Traced"), _text(" text"), COMPLETED])

    result = await client.chat(list(MESSAGES), stream=True)

    assert result == "Traced text"
