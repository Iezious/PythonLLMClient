"""Silence-limit (``read_timeout``) tests for feature 006 step 003 (DEC-arch-20, DEC-006-5, DEC-006-11, DEC-arch-24).

Two kinds of check, as DEC-006-11 lays out:

- Seam checks (step DoD-6): a fake ``client.session`` records the ``timeout`` each request is opened with. On every
  OpenAI-family class built with ``read_timeout``, a streamed request (``_stream_post``) carries that per-read limit
  next to the unchanged total, a non-streamed request (``_request``) carries the total only, and without
  ``read_timeout`` a streamed request carries no per-read limit (US-068.AC-3).
- Local-server checks (step DoD-7, DoD-8): an in-process ``aiohttp.web`` server on ``127.0.0.1`` with an ephemeral
  port serves a chat-completions SSE stream to a real streamed ``chat``. A stream that goes silent longer than the
  limit ends with ``LLMError`` of kind ``timeout`` (US-068.AC-1); a stream whose pieces keep arriving at shorter gaps
  but lasts longer than the limit in total finishes with the full text (US-068.AC-2).

Step DoD-10: the deprecated ``OllamaAPIClient`` does not declare ``read_timeout``, so passing it is a ``TypeError``.
"""

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from yarl import URL

from llm.exceptions import LLMError
from llm.llama_swap import LlamaSwapAPIClient
from llm.llama_swap_chat_completions import LlamaSwapChatCompletionsClient
from llm.ollama import OllamaAPIClient
from llm.openai import OpenAIAPIClient
from llm.openai_chat_completions import OpenAIChatCompletionsClient

BASE_URL = "http://llm.test:9292/v1"
TOKEN = "sk-test"
MODEL = "test-model"

# Distinctive limits for the seam checks, so a recorded value can only have come from the client's settings.
TOTAL_LIMIT = 33.0
SEAM_READ_LIMIT = 0.75

# DEC-006-11 timings for the local-server checks: silence limit 0.2 s; the "keeps producing" stream sends a frame
# every 0.05 s for 0.5 s in total; the "silent" stream sends one frame then pauses 1.0 s; the total limit stays far
# above both, so only the silence limit can end the silent stream early.
READ_LIMIT = 0.2
STEADY_GAP = 0.05
STEADY_PIECES = [f"piece-{i} " for i in range(11)]  # 10 gaps of 0.05 s = 0.5 s of streaming
SILENT_PAUSE = 1.0
SERVER_TOTAL_LIMIT = 30.0

_OPENAI_FAMILY = [
    pytest.param(OpenAIAPIClient, id="openai_responses"),
    pytest.param(LlamaSwapAPIClient, id="llamaswap_responses"),
    pytest.param(OpenAIChatCompletionsClient, id="openai_chat_completions"),
    pytest.param(LlamaSwapChatCompletionsClient, id="llamaswap_chat_completions"),
]

_CHAT_COMPLETIONS_CLASSES = [
    pytest.param(OpenAIChatCompletionsClient, id="openai"),
    pytest.param(LlamaSwapChatCompletionsClient, id="llamaswap"),
]


@pytest.fixture(autouse=True)
def _no_trace(monkeypatch):
    """Keep the env-driven tracer off so no test writes trace files."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


# --------------------------------------------------------------------------- fake session (seam checks)


class _FakeContent:
    """Stand-in for ``response.content``; the seam checks never read the stream body."""

    async def iter_chunked(self, _n: int) -> AsyncIterator[bytes]:
        for chunk in ():
            yield chunk

    async def iter_any(self) -> AsyncIterator[bytes]:
        for chunk in ():
            yield chunk


class _FakeResponse:
    """A 2xx fake aiohttp response carrying a JSON body (readable as text, bytes or parsed JSON)."""

    def __init__(self, url: str, *, json_body: Any = None, content_type: str = "application/json") -> None:
        self.url = URL(url)
        self.real_url = self.url
        self.status = 200
        self.reason = "OK"
        self.ok = True
        self.content_type = content_type
        self.headers = {"Content-Type": content_type}
        self.content = _FakeContent()
        self._json_body = json_body

    def raise_for_status(self) -> None:
        return None

    async def text(self, *args: Any, **kwargs: Any) -> str:
        return json.dumps(self._json_body)

    async def read(self) -> bytes:
        return json.dumps(self._json_body).encode("utf-8")

    async def json(self, *args: Any, **kwargs: Any) -> Any:
        return self._json_body

    def release(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeRequestContext:
    """What ``session.request`` / ``post`` return; works with both ``async with`` and ``await`` like aiohttp's."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def _open(self) -> _FakeResponse:
        return self._response

    def __await__(self):
        return self._open().__await__()

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _RecordingSession:
    """Fake ``aiohttp.ClientSession`` that records the keyword arguments (``timeout`` among them) of each request."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.closed = False

    def request(self, method: str, url: Any, **kwargs: Any) -> _FakeRequestContext:
        url = str(url)
        self.calls.append((method.upper(), url, kwargs))
        if (kwargs.get("json") or {}).get("stream"):
            return _FakeRequestContext(_FakeResponse(url, content_type="text/event-stream"))
        return _FakeRequestContext(_FakeResponse(url, json_body={"object": "list", "data": []}))

    def post(self, url: Any, **kwargs: Any) -> _FakeRequestContext:
        return self.request("POST", url, **kwargs)

    def get(self, url: Any, **kwargs: Any) -> _FakeRequestContext:
        return self.request("GET", url, **kwargs)

    async def close(self) -> None:
        self.closed = True


def _family_client(cls: type, **kwargs: Any) -> Any:
    """An OpenAI-family client with explicit address, credential and total limit; extra keywords pass through."""
    return cls(model=MODEL, base_url=BASE_URL, timeout=TOTAL_LIMIT, bearer_token=TOKEN, **kwargs)


def _install(client: Any) -> _RecordingSession:
    session = _RecordingSession()
    client.session = session
    return session


def _opened_timeout(session: _RecordingSession) -> aiohttp.ClientTimeout:
    """The time limits the single recorded request was opened with."""
    assert len(session.calls) == 1
    _method, _url, kwargs = session.calls[0]
    timeout = kwargs.get("timeout")
    assert isinstance(timeout, aiohttp.ClientTimeout), f"request opened without a ClientTimeout: {timeout!r}"
    return timeout


async def _open_stream(client: Any) -> None:
    async with client._stream_post("/chat/completions", {"model": MODEL, "stream": True}):
        pass


# --------------------------------------------------------------------------- step DoD-6


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", _OPENAI_FAMILY)
async def test_streamed_request_carries_read_limit_and_total__s003_DoD6(monkeypatch, cls):
    """Step 003 DoD-6: with ``read_timeout`` set, a streamed request has that per-read limit and the unchanged total."""
    client = _family_client(cls, read_timeout=SEAM_READ_LIMIT)
    session = _install(client)

    await _open_stream(client)

    timeout = _opened_timeout(session)
    assert timeout.sock_read == SEAM_READ_LIMIT
    assert timeout.total == TOTAL_LIMIT


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", _OPENAI_FAMILY)
async def test_non_streamed_request_carries_total_only__s003_DoD6(monkeypatch, cls):
    """Step 003 DoD-6: on a client with ``read_timeout``, a non-streamed request has the total limit only."""
    client = _family_client(cls, read_timeout=SEAM_READ_LIMIT)
    session = _install(client)

    await client._request("GET", "/models")

    timeout = _opened_timeout(session)
    assert timeout.total == TOTAL_LIMIT
    assert timeout.sock_read is None
    assert timeout.connect is None
    assert timeout.sock_connect is None


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", _OPENAI_FAMILY)
async def test_streamed_request_without_read_timeout_has_no_read_limit__s003_DoD6(monkeypatch, cls):
    """Step 003 DoD-6 / US-068.AC-3: without ``read_timeout`` a streamed request has no per-read limit, total only."""
    client = _family_client(cls)
    session = _install(client)

    await _open_stream(client)

    timeout = _opened_timeout(session)
    assert timeout.sock_read is None
    assert timeout.total == TOTAL_LIMIT


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", _OPENAI_FAMILY)
async def test_streamed_request_with_explicit_none_has_no_read_limit__s003_DoD6(monkeypatch, cls):
    """Step 003 DoD-6: ``read_timeout=None`` passed explicitly is the same as not passing it (DEC-arch-20)."""
    client = _family_client(cls, read_timeout=None)
    session = _install(client)

    await _open_stream(client)

    timeout = _opened_timeout(session)
    assert timeout.sock_read is None
    assert timeout.total == TOTAL_LIMIT


# --------------------------------------------------------------------------- local streaming server (DEC-006-11)


def _frame(payload: str) -> bytes:
    """One data-only SSE event."""
    return f"data: {payload}\n\n".encode("utf-8")


def _piece(content: str | None = None, finish_reason: str | None = None) -> str:
    """One chat-completions stream piece as a JSON data payload."""
    delta: dict[str, Any] = {} if content is None else {"content": content}
    return json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]})


async def _start_stream(request: web.Request) -> web.StreamResponse:
    await request.read()
    response = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    return response


async def _finish_stream(response: web.StreamResponse) -> None:
    """Send the completion signal (finish reason, then ``[DONE]``) and end the body."""
    await response.write(_frame(_piece(finish_reason="stop")))
    await response.write(_frame("[DONE]"))
    await response.write_eof()


def _steady_handler() -> Callable[[web.Request], Awaitable[web.StreamResponse]]:
    """A stream that keeps producing: one piece every ``STEADY_GAP`` s for 0.5 s, then the completion signal."""

    async def handle(request: web.Request) -> web.StreamResponse:
        response = await _start_stream(request)
        for index, text in enumerate(STEADY_PIECES):
            if index:
                await asyncio.sleep(STEADY_GAP)
            await response.write(_frame(_piece(content=text)))
        await _finish_stream(response)
        return response

    return handle


def _silent_handler(release: asyncio.Event) -> Callable[[web.Request], Awaitable[web.StreamResponse]]:
    """A stream that sends one piece, then goes silent for ``SILENT_PAUSE`` s before completing.

    The pause also ends when ``release`` is set, which the server helper does at teardown, so a client that already
    gave up does not keep the server's shutdown waiting for the rest of the pause.
    """

    async def handle(request: web.Request) -> web.StreamResponse:
        response = await _start_stream(request)
        await response.write(_frame(_piece(content="first ")))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(release.wait(), SILENT_PAUSE)
        # The client may have hung up during the pause; writing to it then fails, which is expected here.
        with contextlib.suppress(ConnectionError, RuntimeError):
            await response.write(_frame(_piece(content="late")))
            await _finish_stream(response)
        return response

    return handle


@contextlib.asynccontextmanager
async def _local_stream_server(
    make_handler: Callable[[asyncio.Event], Callable[[web.Request], Awaitable[web.StreamResponse]]],
) -> AsyncIterator[str]:
    """Serve ``POST /v1/chat/completions`` on 127.0.0.1 at an ephemeral port; yields the client's ``base_url``."""
    release = asyncio.Event()
    app = web.Application()
    app.router.add_post("/v1/chat/completions", make_handler(release))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        release.set()
        await runner.cleanup()


class _Recorder:
    """Async receiver that records every streamed piece, in order."""

    def __init__(self) -> None:
        self.pieces: list[str] = []

    async def __call__(self, text: str) -> None:
        self.pieces.append(text)


def _server_client(cls: type, base_url: str) -> Any:
    """A chat-completions client aimed at the local server, with the DEC-006-11 silence limit and a large total."""
    return cls(
        model=MODEL, base_url=base_url, timeout=SERVER_TOTAL_LIMIT, bearer_token=TOKEN, read_timeout=READ_LIMIT
    )


_USER_MESSAGES = [{"role": "user", "content": "Tell me a long story."}]


# --------------------------------------------------------------------------- step DoD-7


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", _CHAT_COMPLETIONS_CLASSES)
async def test_silent_stream_ends_with_timeout_kind__s003_DoD7(monkeypatch, cls):
    """Step 003 DoD-7 / US-068.AC-1: a stream silent longer than ``read_timeout`` ends with kind ``timeout``."""
    async with _local_stream_server(_silent_handler) as base_url:
        client = _server_client(cls, base_url)
        async with client:
            with pytest.raises(LLMError) as exc_info:
                await client.chat(list(_USER_MESSAGES), stream=True, on_delta=_Recorder())

    assert exc_info.value.kind == "timeout"


# --------------------------------------------------------------------------- step DoD-8


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", _CHAT_COMPLETIONS_CLASSES)
async def test_steady_stream_longer_than_limit_completes__s003_DoD8(monkeypatch, cls):
    """Step 003 DoD-8 / US-068.AC-2: pieces at gaps under ``read_timeout`` but a longer total finish with full text."""
    recorder = _Recorder()
    async with _local_stream_server(lambda release: _steady_handler()) as base_url:
        client = _server_client(cls, base_url)
        async with client:
            started = time.monotonic()
            text = await client.chat(list(_USER_MESSAGES), stream=True, on_delta=recorder)
            elapsed = time.monotonic() - started

    # The whole stream lasted longer than the silence limit, so only the gaps between pieces kept it alive.
    assert elapsed > READ_LIMIT
    assert text == "".join(STEADY_PIECES)
    assert "".join(recorder.pieces) == "".join(STEADY_PIECES)


# --------------------------------------------------------------------------- step DoD-10


def test_ollama_constructor_refuses_read_timeout__s003_DoD10():
    """Step 003 DoD-10: ``OllamaAPIClient`` does not declare ``read_timeout``, so passing it raises ``TypeError``."""
    with pytest.raises(TypeError):
        OllamaAPIClient(model="llama3", base_url="http://localhost:11434", read_timeout=5.0)
