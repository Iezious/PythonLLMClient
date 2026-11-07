"""Request-error tests for the base client (feature 001.core-fixes, step 002).

Covers the transport-failure mapping of ``LLMClient._request`` (DEC-001-7), the unchanged non-2xx mapping
(DEC-000-11), the streamed-request helper ``LLMClient._stream_post`` (DEC-001-8, DEC-arch-5), and the effect of
the transport mapping on the deprecated Ollama embed fallback (DEC-000-42).

The server is faked by assigning a fake ``client.session`` (``_ensure_session`` returns an existing session as is):
its ``request`` / ``post`` hand back an async context manager that either raises a transport exception on entry or
yields a fake response. DoD-11 (the existing unit suite passes unchanged) is covered by the existing files
``tests/test_llm_embeddings.py``, ``tests/test_get_models.py``, ``tests/test_llm_tools_streaming.py`` and
``tests/test_llm_tools_reasoning_stream.py``; no new test is written for it.
"""

import re
from typing import Any

import aiohttp
import pytest
from yarl import URL

from llm.exceptions import LLMError
from llm.llm_client import LLMClient
from llm.ollama import OllamaAPIClient

BASE_URL = "http://llm.test:9292/v1"
TOKEN = "sk-secret-credential-0042"
CLIENT_TIMEOUT = 42.5

_STATUS_PHRASE = re.compile(r"status \d")


class _ReceiverFailure(Exception):
    """A caller-side failure raised inside a ``_stream_post`` block (stands in for a failing receiver)."""


class _FakeContent:
    """Minimal stand-in for ``response.content``; this step never reads the stream itself."""

    async def iter_chunked(self, _n: int):
        for chunk in ():
            yield chunk


class _FakeResponse:
    """Fake aiohttp response: fixed status, body and content type; can fail every body read.

    Reading the body after the response was released fails like a closed connection, so a helper that reads the
    non-2xx reply only after leaving the response context cannot pass (the reply must be read while open).
    """

    def __init__(
        self,
        url: str,
        *,
        status: int = 200,
        body: str = "",
        json_body: Any = None,
        content_type: str = "application/json",
        body_error: BaseException | None = None,
    ) -> None:
        self.url = URL(url)
        self.status = status
        self.reason = "Error" if status >= 400 else "OK"
        self.content_type = content_type
        self.headers = {"Content-Type": content_type}
        self.content = _FakeContent()
        self._body = body
        self._json_body = json_body
        self._body_error = body_error
        self.released = False

    @property
    def ok(self) -> bool:
        return self.status < 400

    def raise_for_status(self) -> None:
        if self.status >= 400:
            info = aiohttp.RequestInfo(url=self.url, method="POST", headers={}, real_url=self.url)
            raise aiohttp.ClientResponseError(info, (), status=self.status, message=self.reason)

    def _check_readable(self) -> None:
        if self._body_error is not None:
            raise self._body_error
        if self.released:
            raise aiohttp.ClientConnectionError("Connection closed")

    async def text(self, *args: Any, **kwargs: Any) -> str:
        self._check_readable()
        return self._body

    async def read(self) -> bytes:
        self._check_readable()
        return self._body.encode("utf-8")

    async def json(self, *args: Any, **kwargs: Any) -> Any:
        self._check_readable()
        return self._json_body

    def release(self) -> None:
        self.released = True

    def close(self) -> None:
        self.released = True


class _FakeRequestContext:
    """What ``session.request`` / ``session.post`` return: raises ``error`` on entry or yields ``response``.

    Supports both ``async with`` and ``await``, as aiohttp's own request context manager does, so the test does not
    depend on which form the client uses.
    """

    def __init__(self, response: _FakeResponse | None, error: BaseException | None) -> None:
        self._response = response
        self._error = error

    async def _open(self) -> _FakeResponse:
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response

    def __await__(self):
        return self._open().__await__()

    async def __aenter__(self) -> _FakeResponse:
        return await self._open()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if self._response is not None:
            self._response.release()
        return False


class _FakeSession:
    """Fake ``aiohttp.ClientSession`` recording every request as ``(method, url, kwargs)``.

    ``responder(method, url)`` returns either a ``_FakeResponse`` or an exception instance to raise on opening.
    """

    def __init__(self, responder) -> None:
        self._responder = responder
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.closed = False

    def request(self, method: str, url: Any, **kwargs: Any) -> _FakeRequestContext:
        method = method.upper()
        url = str(url)
        self.calls.append((method, url, kwargs))
        outcome = self._responder(method, url)
        if isinstance(outcome, BaseException):
            return _FakeRequestContext(None, outcome)
        return _FakeRequestContext(outcome, None)

    def post(self, url: Any, **kwargs: Any) -> _FakeRequestContext:
        return self.request("POST", url, **kwargs)

    def get(self, url: Any, **kwargs: Any) -> _FakeRequestContext:
        return self.request("GET", url, **kwargs)

    async def close(self) -> None:
        self.closed = True


def _client(bearer_token: str | None = TOKEN) -> LLMClient:
    """A base client with a distinctive address, credential and time limit, so each can be traced in assertions."""
    return LLMClient("test-model", BASE_URL, CLIENT_TIMEOUT, bearer_token)


def _install(client: LLMClient, responder) -> _FakeSession:
    session = _FakeSession(responder)
    client.session = session  # type: ignore[assignment]
    return session


def _texts(err: LLMError) -> list[str]:
    """Every text an app could show for the error: ``str()``, message and description."""
    return [str(err), err.message or "", err.description or ""]


def _transport_errors() -> list[BaseException]:
    return [
        aiohttp.ClientConnectionError("Cannot connect to host llm.test:9292"),
        aiohttp.ServerDisconnectedError(),
    ]


def _timeout_errors() -> list[BaseException]:
    return [TimeoutError(), aiohttp.ServerTimeoutError("Timeout on reading data from socket")]


@pytest.fixture(autouse=True)
def _no_trace(monkeypatch):
    """Keep the env-driven tracer off so no test writes trace files."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


# --- _request: transport failures -------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("error", _transport_errors(), ids=lambda e: type(e).__name__)
async def test_request_connection_failure_raises_llm_error_with_address__dod1(monkeypatch, error):
    """DoD-1: a connection failure in ``_request`` raises ``LLMError`` whose text carries the request address."""
    client = _client()
    _install(client, lambda method, url: error)

    with pytest.raises(LLMError) as exc_info:
        await client._request("POST", "/chat/completions", json={"model": "test-model"})

    assert f"{BASE_URL}/chat/completions" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", _timeout_errors(), ids=lambda e: type(e).__name__)
async def test_request_timeout_on_open_raises_llm_error_with_address__dod2(monkeypatch, error):
    """DoD-2: a timeout while opening the request in ``_request`` raises ``LLMError`` carrying the address."""
    client = _client()
    _install(client, lambda method, url: error)

    with pytest.raises(LLMError) as exc_info:
        await client._request("GET", "/models")

    assert f"{BASE_URL}/models" in str(exc_info.value)


@pytest.mark.asyncio
async def test_request_timeout_on_body_read_raises_llm_error_with_address__dod2(monkeypatch):
    """DoD-2: a timeout while reading a 2xx body in ``_request`` raises ``LLMError`` carrying the address."""
    client = _client()
    url = f"{BASE_URL}/models"
    _install(client, lambda method, u: _FakeResponse(u, status=200, body_error=TimeoutError()))

    with pytest.raises(LLMError) as exc_info:
        await client._request("GET", "/models")

    assert url in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "content_type"),
    [
        (aiohttp.ClientPayloadError("Response payload is not completed"), "application/json"),
        (aiohttp.ClientPayloadError("Response payload is not completed"), "application/octet-stream"),
        (aiohttp.ServerDisconnectedError(), "application/json"),
        (aiohttp.ServerDisconnectedError(), "application/octet-stream"),
    ],
    ids=["payload-json", "payload-bytes", "disconnect-json", "disconnect-bytes"],
)
async def test_request_body_read_failure_raises_llm_error__dod3(monkeypatch, error, content_type):
    """DoD-3: a payload error or disconnect while reading a 2xx body in ``_request`` raises ``LLMError``."""
    client = _client()
    _install(client, lambda method, u: _FakeResponse(u, status=200, content_type=content_type, body_error=error))

    with pytest.raises(LLMError):
        await client._request("POST", "/embeddings", json={"model": "test-model", "input": ["a"]})


# --- _request / _stream_post: no credential and no status phrase in transport errors ------------------------------


def _assert_clean_transport_text(err: LLMError) -> None:
    for text in _texts(err):
        assert TOKEN not in text
        assert _STATUS_PHRASE.search(text) is None, text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    _transport_errors() + _timeout_errors(),
    ids=lambda e: type(e).__name__,
)
async def test_request_transport_error_text_has_no_credential_or_status__dod4(monkeypatch, error):
    """DoD-4: a transport ``LLMError`` from ``_request`` has no credential and no "status <n>"."""
    client = _client(TOKEN)
    _install(client, lambda method, url: error)

    with pytest.raises(LLMError) as exc_info:
        await client._request("POST", "/chat/completions", json={"model": "test-model"})

    _assert_clean_transport_text(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [aiohttp.ClientPayloadError("Response payload is not completed"), aiohttp.ServerDisconnectedError()],
    ids=lambda e: type(e).__name__,
)
async def test_request_body_read_error_text_has_no_credential_or_status__dod4(monkeypatch, error):
    """DoD-4: a body-read transport ``LLMError`` from ``_request`` has no credential and no "status <n>"."""
    client = _client(TOKEN)
    _install(client, lambda method, u: _FakeResponse(u, status=200, body_error=error))

    with pytest.raises(LLMError) as exc_info:
        await client._request("GET", "/models")

    _assert_clean_transport_text(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    _transport_errors() + _timeout_errors(),
    ids=lambda e: type(e).__name__,
)
async def test_stream_post_transport_error_text_has_no_credential_or_status__dod4(monkeypatch, error):
    """DoD-4: a transport ``LLMError`` from opening ``_stream_post`` has no credential and no "status <n>"."""
    client = _client(TOKEN)
    _install(client, lambda method, url: error)

    with pytest.raises(LLMError) as exc_info:
        async with client._stream_post("/chat/completions", {"model": "test-model", "stream": True}):
            pass

    _assert_clean_transport_text(exc_info.value)


# --- _request: non-2xx mapping unchanged ------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 500])
async def test_request_non_2xx_keeps_status_message_and_reply_description__dod5(monkeypatch, status):
    """DoD-5: non-2xx keeps message "Request to <url> failed with status <n>" and the reply as description."""
    client = _client()
    reply = f"server says no ({status})"
    _install(client, lambda method, u: _FakeResponse(u, status=status, body=reply, content_type="text/plain"))

    with pytest.raises(LLMError) as exc_info:
        await client._request("POST", "/chat/completions", json={"model": "test-model"})

    err = exc_info.value
    assert err.message == f"Request to {BASE_URL}/chat/completions failed with status {status}"
    assert err.description == reply


# --- _stream_post: non-2xx --------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 503])
async def test_stream_post_non_2xx_raises_with_address_status_and_reply__dod6(monkeypatch, status):
    """DoD-6: a refused stream raises ``LLMError`` with address, status and reply, never the credential."""
    client = _client(TOKEN)
    reply = '{"error": {"message": "model not loaded"}}'
    _install(client, lambda method, u: _FakeResponse(u, status=status, body=reply, content_type="application/json"))

    with pytest.raises(LLMError) as exc_info:
        async with client._stream_post("/chat/completions", {"model": "test-model", "stream": True}):
            pytest.fail("the block must not run for a non-2xx reply")

    text = str(exc_info.value)
    assert f"{BASE_URL}/chat/completions" in text
    assert str(status) in text
    assert reply in text
    for t in _texts(exc_info.value):
        assert TOKEN not in t


# --- _stream_post: 2xx request shape ----------------------------------------------------------------------------


def _header(kwargs: dict[str, Any], name: str) -> str | None:
    """Case-insensitive header lookup in the recorded request keywords."""
    headers = kwargs.get("headers") or {}
    for key, value in dict(headers).items():
        if key.lower() == name.lower():
            return value
    return None


def _timeout_total(kwargs: dict[str, Any]) -> Any:
    timeout = kwargs.get("timeout")
    return getattr(timeout, "total", timeout)


@pytest.mark.asyncio
async def test_stream_post_2xx_yields_response_with_token__dod7(monkeypatch):
    """DoD-7: on 2xx the open response is yielded; POST to base + path, JSON payload, bearer header, time limit."""
    client = _client(TOKEN)
    payload = {"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    responses: list[_FakeResponse] = []

    def responder(method, url):
        resp = _FakeResponse(url, status=200, content_type="text/event-stream")
        responses.append(resp)
        return resp

    session = _install(client, responder)

    async with client._stream_post("/chat/completions", payload) as response:
        assert response is responses[0]
        assert response.released is False

    assert len(session.calls) == 1
    method, url, kwargs = session.calls[0]
    assert method == "POST"
    assert url == f"{BASE_URL}/chat/completions"
    assert kwargs.get("json") == payload
    assert _header(kwargs, "Authorization") == f"Bearer {TOKEN}"
    assert _timeout_total(kwargs) == CLIENT_TIMEOUT


@pytest.mark.asyncio
async def test_stream_post_2xx_without_token_sends_no_authorization__dod7(monkeypatch):
    """DoD-7: a client without a token sends no ``Authorization`` header; address, payload and time limit hold."""
    client = _client(None)
    payload = {"model": "test-model", "messages": [], "stream": True}
    session = _install(client, lambda method, u: _FakeResponse(u, status=200, content_type="text/event-stream"))

    async with client._stream_post("/chat/completions", payload) as response:
        assert response.status == 200

    assert len(session.calls) == 1
    method, url, kwargs = session.calls[0]
    assert method == "POST"
    assert url == f"{BASE_URL}/chat/completions"
    assert kwargs.get("json") == payload
    assert _header(kwargs, "Authorization") is None
    assert _timeout_total(kwargs) == CLIENT_TIMEOUT


# --- _stream_post: transport failure while opening ---------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("error", _transport_errors() + _timeout_errors(), ids=lambda e: type(e).__name__)
async def test_stream_post_open_failure_raises_llm_error_with_address__dod8(monkeypatch, error):
    """DoD-8: a connection failure or timeout opening ``_stream_post`` raises ``LLMError`` with the address."""
    client = _client()
    _install(client, lambda method, url: error)

    with pytest.raises(LLMError) as exc_info:
        async with client._stream_post("/chat/completions", {"model": "test-model", "stream": True}):
            pytest.fail("the block must not run when the request cannot be opened")

    assert f"{BASE_URL}/chat/completions" in str(exc_info.value)


# --- _stream_post: caller's exceptions pass through ---------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        _ReceiverFailure("receiver broke"),
        ValueError("bad value in receiver"),
        # Transport-typed exceptions from the caller's own code are not converted either (DEC-001-7, Rejected).
        TimeoutError("receiver timed out"),
        aiohttp.ClientConnectionError("receiver's own connection failed"),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_stream_post_block_exception_passes_through_unchanged__dod9(monkeypatch, error):
    """DoD-9: an exception raised inside the ``_stream_post`` block reaches the caller as the same object."""
    client = _client()
    _install(client, lambda method, u: _FakeResponse(u, status=200, content_type="text/event-stream"))

    with pytest.raises(BaseException) as exc_info:
        async with client._stream_post("/chat/completions", {"model": "test-model", "stream": True}):
            raise error

    assert exc_info.value is error
    assert type(exc_info.value) is type(error)


# --- Ollama embed fallback (deprecated client, observed through the real _request) ---------------------------------


OLLAMA_URL = "http://ollama.test:11434"


@pytest.mark.asyncio
@pytest.mark.parametrize("error", _transport_errors() + _timeout_errors(), ids=lambda e: type(e).__name__)
async def test_ollama_embed_batch_transport_failure_no_fallback__dod10(monkeypatch, error):
    """DoD-10: a transport failure on the Ollama embed route raises ``LLMError`` and makes no per-text fallback."""
    client = OllamaAPIClient(model="nomic-embed-text", base_url=OLLAMA_URL)
    session = _install(client, lambda method, url: error)

    with pytest.raises(LLMError):
        await client.embed_batch(["a", "b"])

    paths = [url for _method, url, _kw in session.calls]
    assert paths == [f"{OLLAMA_URL}/api/embed"]


@pytest.mark.asyncio
async def test_ollama_embed_batch_404_still_falls_back__dod10(monkeypatch):
    """DoD-10: a 404 from the Ollama embed route still triggers the per-text ``/api/embeddings`` fallback."""
    client = OllamaAPIClient(model="nomic-embed-text", base_url=OLLAMA_URL)
    fallback_calls = 0

    def responder(method, url):
        nonlocal fallback_calls
        if url == f"{OLLAMA_URL}/api/embed":
            return _FakeResponse(url, status=404, body="404 page not found", content_type="text/plain")
        if url == f"{OLLAMA_URL}/api/embeddings":
            fallback_calls += 1
            return _FakeResponse(url, status=200, json_body={"embedding": [float(fallback_calls)]})
        raise AssertionError(f"unexpected request: {method} {url}")

    session = _install(client, responder)

    vectors = await client.embed_batch(["a", "b"])

    paths = [url for _method, url, _kw in session.calls]
    assert paths == [f"{OLLAMA_URL}/api/embed", f"{OLLAMA_URL}/api/embeddings", f"{OLLAMA_URL}/api/embeddings"]
    assert vectors == [[1.0], [2.0]]
