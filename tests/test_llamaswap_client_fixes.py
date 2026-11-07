"""Unit tests for the llama-swap client fixes of feature 001.core-fixes, step 005.

Covers, on ``LlamaSwapChatCompletionsClient``: streamed requests opened through the base streamed-request
helper (credential header, non-2xx and transport failures as ``LLMError``, pieces delivered in
order), null reply content returned as empty text, and the settings rules (no built-in values,
stop / token bias / end-user id / response format sent, response-format precedence, merge rules,
thinking switch, embedding end-user id). Each test names the step-005 DoD clause it covers.

Feature 003.responses-chat, step 001: these tests now build ``LlamaSwapChatCompletionsClient`` (the
chat-completions regression suite, DoD-11), and the last section checks that its ``chat`` posts to
``/chat/completions`` (DoD-9 of that step, tagged ``__s001_DoD9``).
"""

import copy
import json
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest
from yarl import URL

from llm.exceptions import LLMError
from llm.llama_swap_chat_completions import LlamaSwapChatCompletionsClient

_BASE_URL = "http://localhost:8080/v1"
_CHAT_URL = f"{_BASE_URL}/chat/completions"
_MODEL = "some-model"
_SECRET = "sk-llamaswap-secret-token-123"

# Generation settings that must never be sent unless the app set them (US-018.AC-1, DEC-001-13);
# reasoning_effort is included because llama-swap never sends it (DEC-000-19).
_GENERATION_KEYS = (
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "max_tokens",
    "seed",
    "enable_thinking",
    "reasoning_effort",
)

_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        },
    }
]


def _add(a: int, b: int) -> int:
    """Tool used by the tool-loop tests."""
    return a + b


_TOOLS = {"add": _add}

_ADD_CALL = [{"id": "call_1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}}]

# The four offered settings of DEC-001-14, with values distinct enough to recognise in a body.
_OFFERED_SETTINGS: dict[str, Any] = {
    "stop": ["END", "STOP"],
    "logit_bias": {"50256": -100.0},
    "user": "end-user-42",
    "response_format": {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object", "properties": {"x": {"type": "integer"}}}},
    },
}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch):
    """Start every test with tracing off so a developer's shell env cannot change behaviour (DEC-000-4)."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _client(**kw: Any) -> LlamaSwapChatCompletionsClient:
    """Build a llama-swap client on the fixed test address; credential and defaults come from ``kw``."""
    return LlamaSwapChatCompletionsClient(model=_MODEL, base_url=_BASE_URL, **kw)


def _sse(*payloads: str) -> bytes:
    """Encode data payloads as an SSE byte stream, one event per payload."""
    return "".join(f"data: {p}\n\n" for p in payloads).encode("utf-8")


def _delta_frame(content: str | None = None, finish_reason: str | None = None) -> str:
    """One chat-completions stream frame carrying a content piece and/or a finish reason."""
    delta = {"content": content} if content is not None else {}
    return json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]})


def _complete_stream(*pieces: str) -> bytes:
    """A well-formed stream: the pieces, a finish reason and the done frame (so no early-end error)."""
    frames = [_delta_frame(p) for p in pieces]
    frames.append(_delta_frame(finish_reason="stop"))
    frames.append("[DONE]")
    return _sse(*frames)


class _FakeContent:
    """Stand-in for ``aiohttp.StreamReader``: serves the body in small chunks to exercise the reader."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    async def iter_chunked(self, n: int):
        # Small reads (at most 7 bytes) so events and lines span several chunks, as on a real wire.
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
        self.reason = "OK" if status < 400 else "Service Unavailable"
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
    """Mimics aiohttp's request context manager: usable with ``async with`` and with ``await``.

    The response (or the transport failure) is produced on enter, as a real connection would be.
    """

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
    """Fake ``aiohttp.ClientSession`` assigned to ``client.session``; records every request it opens.

    ``raise_on_open`` simulates an unreachable server: the exception is raised when the request opens.
    """

    def __init__(
        self,
        *,
        status: int = 200,
        body: bytes = b"",
        content_type: str = "text/event-stream",
        raise_on_open: BaseException | None = None,
    ) -> None:
        self.status = status
        self.body = body
        self.content_type = content_type
        self.raise_on_open = raise_on_open
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def request(self, method: str, url: Any, **kw: Any) -> _FakeRequestContext:
        self.calls.append({"method": method.upper(), "url": str(url), **kw})

        def _open() -> _FakeResponse:
            if self.raise_on_open is not None:
                raise self.raise_on_open
            return _FakeResponse(str(url), self.status, self.body, self.content_type)

        return _FakeRequestContext(_open)

    def post(self, url: Any, **kw: Any) -> _FakeRequestContext:
        return self.request("POST", url, **kw)

    async def close(self) -> None:
        self.closed = True


def _headers_of(call: dict[str, Any]) -> dict[str, str]:
    """Request headers of a recorded call, with lower-cased names for case-insensitive lookup."""
    return {str(k).lower(): v for k, v in dict(call.get("headers") or {}).items()}


def _reply(content: str | None, tool_calls: list[dict] | None = None) -> dict[str, Any]:
    """A non-streamed chat-completions reply body."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}


def _fake_request(monkeypatch, client: LlamaSwapChatCompletionsClient, replies: list[Any]) -> list[dict[str, Any]]:
    """Replace ``_request`` with a fake returning ``replies`` in order; returns the captured requests.

    Payloads are deep-copied because the client keeps mutating its message list between rounds.
    """
    captured: list[dict[str, Any]] = []

    async def fake_request(method, path, **kw):
        captured.append({"method": method, "path": path, "json": copy.deepcopy(kw.get("json"))})
        return replies[min(len(captured), len(replies)) - 1]

    monkeypatch.setattr(client, "_request", fake_request)
    return captured


async def _run_tools(
    monkeypatch, client: LlamaSwapChatCompletionsClient, *, stream: bool, rounds: list[tuple], **kwargs: Any
) -> tuple[str, list[dict[str, Any]]]:
    """Run ``chat_with_tools`` against fakes of both seams; returns the result and every round's body.

    ``rounds`` is a list of ``(content, tool_calls)`` per server round. Streamed rounds go through
    ``_tools_stream_request``, non-streamed ones through ``_request``.
    """
    bodies: list[dict[str, Any]] = []

    async def fake_request(method, path, **kw):
        assert method == "POST"
        assert path == "/chat/completions"
        bodies.append(copy.deepcopy(kw["json"]))
        content, tool_calls = rounds[len(bodies) - 1]
        return _reply(content, tool_calls)

    async def fake_stream_request(payload, on_delta=None):
        bodies.append(copy.deepcopy(payload))
        content, tool_calls = rounds[len(bodies) - 1]
        return (content if content is not None else "", tool_calls or [])

    monkeypatch.setattr(client, "_request", fake_request)
    monkeypatch.setattr(client, "_tools_stream_request", fake_stream_request)
    result = await client.chat_with_tools(
        [{"role": "user", "content": "What is 1+2?"}],
        tools_definitions=_TOOL_DEFS,
        tools=_TOOLS,
        stream=stream,
        **kwargs,
    )
    return result, bodies


def _assert_no_credential(err: LLMError, secret: str) -> None:
    """The error's message, description and text never contain the credential (US-011.AC-4)."""
    assert secret not in str(err)
    assert secret not in (err.message or "")
    assert secret not in (err.description or "")


_TOOLS_PAYLOAD = {
    "model": _MODEL,
    "messages": [{"role": "user", "content": "What is 1+2?"}],
    "tools": _TOOL_DEFS,
    "stream": True,
}


# ---------------------------------------------------------------------------
# Streamed requests through the base helper (DoD 1-5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_streamed_chat_sends_bearer_when_credential_set__DoD1(monkeypatch):
    """DoD-1: with a credential, a streamed chat request carries ``Authorization: Bearer <credential>``."""
    client = _client(bearer_token=_SECRET)
    session = _FakeSession(body=_complete_stream("hi"))
    client.session = session

    result = await client.chat([{"role": "user", "content": "Hello"}], stream=True)

    assert result == "hi"
    assert len(session.calls) == 1
    assert _headers_of(session.calls[0]).get("authorization") == f"Bearer {_SECRET}"


@pytest.mark.asyncio
async def test_llamaswap_streamed_chat_without_credential_sends_no_auth_header__DoD1(monkeypatch):
    """DoD-1: without a credential, a streamed chat request has no Authorization header and still works."""
    client = _client()
    session = _FakeSession(body=_complete_stream("hello"))
    client.session = session

    result = await client.chat([{"role": "user", "content": "Hello"}], stream=True)

    assert result == "hello"
    assert len(session.calls) == 1
    assert "authorization" not in _headers_of(session.calls[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("token", [_SECRET, None])
async def test_llamaswap_streamed_tool_round_credential_header__DoD1(monkeypatch, token):
    """DoD-1: a streamed tool round carries the bearer header only when the client has a credential."""
    client = _client(bearer_token=token)
    session = _FakeSession(body=_complete_stream("ok"))
    client.session = session

    content, tool_calls = await client._tools_stream_request(copy.deepcopy(_TOOLS_PAYLOAD), None)

    assert content == "ok"
    assert tool_calls == []
    headers = _headers_of(session.calls[0])
    if token:
        assert headers.get("authorization") == f"Bearer {token}"
    else:
        assert "authorization" not in headers


@pytest.mark.asyncio
async def test_llamaswap_streamed_chat_non_2xx_raises_llmerror_with_reply__DoD2(monkeypatch):
    """DoD-2: a non-2xx reply to a streamed chat raises LLMError with address, status and reply, no credential."""
    client = _client(bearer_token=_SECRET)
    client.session = _FakeSession(status=503, body=b"upstream model failed to load", content_type="text/plain")

    with pytest.raises(LLMError) as excinfo:
        await client.chat([{"role": "user", "content": "Hello"}], stream=True)

    err = excinfo.value
    text = str(err)
    assert _CHAT_URL in text
    assert "503" in text
    assert "upstream model failed to load" in text
    _assert_no_credential(err, _SECRET)


@pytest.mark.asyncio
async def test_llamaswap_streamed_tool_round_non_2xx_raises_llmerror_with_reply__DoD3(monkeypatch):
    """DoD-3: a non-2xx reply to a streamed tool round raises LLMError with address, status and reply."""
    client = _client(bearer_token=_SECRET)
    client.session = _FakeSession(status=500, body=b'{"error": "tool template broken"}', content_type="text/plain")

    with pytest.raises(LLMError) as excinfo:
        await client._tools_stream_request(copy.deepcopy(_TOOLS_PAYLOAD), None)

    err = excinfo.value
    text = str(err)
    assert _CHAT_URL in text
    assert "500" in text
    assert "tool template broken" in text
    _assert_no_credential(err, _SECRET)


@pytest.mark.asyncio
async def test_llamaswap_streamed_chat_unreachable_server_raises_llmerror__DoD4(monkeypatch):
    """DoD-4: a streamed chat against an unreachable server raises LLMError naming the address (DEC-001-7)."""
    client = _client(bearer_token=_SECRET)
    refused = aiohttp.ClientConnectionError("Cannot connect to host localhost:8080")
    client.session = _FakeSession(raise_on_open=refused)

    with pytest.raises(LLMError) as excinfo:
        await client.chat([{"role": "user", "content": "Hello"}], stream=True)

    assert _CHAT_URL in str(excinfo.value)
    _assert_no_credential(excinfo.value, _SECRET)


@pytest.mark.asyncio
async def test_llamaswap_streamed_chat_delivers_pieces_in_order__DoD5(monkeypatch):
    """DoD-5: a streamed chat gives the receiver every piece in order and returns their concatenation."""
    client = _client()
    session = _FakeSession(body=_complete_stream("Hel", "lo", ", ", "world"))
    client.session = session
    received: list[str] = []

    async def on_delta(text: str) -> None:
        received.append(text)

    result = await client.chat([{"role": "user", "content": "Hello"}], stream=True, on_delta=on_delta)

    assert received == ["Hel", "lo", ", ", "world"]
    assert result == "Hello, world"
    call = session.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == _CHAT_URL
    assert call["json"]["stream"] is True
    assert call["json"]["model"] == _MODEL


@pytest.mark.asyncio
async def test_llamaswap_streamed_chat_with_tracing_delivers_pieces_in_order__DoD5(monkeypatch, tmp_path):
    """DoD-5: with tracing on (trace wrapper active, DEC-000-48) the receiver still gets every piece in order."""
    # Tracing is env-driven and read when the client is built (DEC-000-4), so set it first.
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(tmp_path))
    client = _client()
    client.session = _FakeSession(body=_complete_stream("a", "b", "c"))
    received: list[str] = []

    async def on_delta(text: str) -> None:
        received.append(text)

    result = await client.chat([{"role": "user", "content": "Hello"}], stream=True, on_delta=on_delta)

    assert received == ["a", "b", "c"]
    assert result == "abc"


# ---------------------------------------------------------------------------
# Null reply content is empty text (DoD 6-8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_chat_null_content_returns_empty_text__DoD6(monkeypatch):
    """DoD-6: a non-streamed chat whose reply content is null returns empty text."""
    client = _client()
    _fake_request(monkeypatch, client, [_reply(None)])

    result = await client.chat([{"role": "user", "content": "Hello"}])

    assert result == ""


@pytest.mark.asyncio
async def test_llamaswap_generate_null_content_returns_empty_text__DoD6(monkeypatch):
    """DoD-6: a non-streamed generate whose reply content is null returns empty text."""
    client = _client()
    _fake_request(monkeypatch, client, [_reply(None)])

    result = await client.generate("Say nothing")

    assert result == ""


@pytest.mark.asyncio
async def test_llamaswap_chat_null_content_with_tracing_returns_empty_text__DoD7(monkeypatch, tmp_path):
    """DoD-7: with tracing on, a non-streamed chat with null reply content returns empty text."""
    # Tracing is env-driven and read when the client is built (DEC-000-4), so set it first.
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(tmp_path))
    client = _client()
    _fake_request(monkeypatch, client, [_reply(None)])

    result = await client.chat([{"role": "user", "content": "Hello"}])

    assert result == ""


@pytest.mark.asyncio
async def test_llamaswap_chat_json_null_content_returns_empty_text__DoD8(monkeypatch):
    """DoD-8: chat_json with null reply content (no expected shape given) returns empty text."""
    client = _client()
    _fake_request(monkeypatch, client, [_reply(None)])

    result = await client.chat_json([{"role": "user", "content": "Give JSON"}])

    assert result == ""


@pytest.mark.asyncio
async def test_llamaswap_chat_with_tools_null_final_answer_returns_empty_text__DoD8(monkeypatch):
    """DoD-8: chat_with_tools whose final answer has null content returns empty text."""
    client = _client()

    result, _ = await _run_tools(monkeypatch, client, stream=False, rounds=[(None, [])])

    assert result == ""


@pytest.mark.asyncio
async def test_llamaswap_chat_with_tools_null_tool_call_turn_enters_next_round_as_empty_text__DoD8(monkeypatch):
    """DoD-8: an assistant tool-call turn with null content appears in the next round's conversation as ''."""
    client = _client()

    result, bodies = await _run_tools(
        monkeypatch, client, stream=False, rounds=[(None, _ADD_CALL), ("The result is 3.", [])]
    )

    assert result == "The result is 3."
    assert len(bodies) == 2
    tool_turns = [m for m in bodies[1]["messages"] if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(tool_turns) == 1
    assert tool_turns[0]["content"] == ""


# ---------------------------------------------------------------------------
# Only the app's settings are sent (DoD 9)
# ---------------------------------------------------------------------------


def _assert_no_generation_settings(body: dict[str, Any]) -> None:
    """No generation setting key appears in a request body (US-018.AC-1)."""
    present = [k for k in _GENERATION_KEYS if k in body]
    assert present == [], f"unexpected generation settings sent: {present}"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_llamaswap_chat_without_settings_sends_no_generation_setting__DoD9(monkeypatch, stream):
    """DoD-9: chat (streamed or not) with no app settings sends no generation setting."""
    client = _client()
    if stream:
        session = _FakeSession(body=_complete_stream("ok"))
        client.session = session
        await client.chat([{"role": "user", "content": "Hello"}], stream=True)
        body = session.calls[0]["json"]
    else:
        captured = _fake_request(monkeypatch, client, [_reply("ok")])
        await client.chat([{"role": "user", "content": "Hello"}])
        body = captured[0]["json"]

    _assert_no_generation_settings(body)


@pytest.mark.asyncio
async def test_llamaswap_chat_json_without_settings_sends_no_generation_setting__DoD9(monkeypatch):
    """DoD-9: chat_json with no app settings sends no generation setting."""
    client = _client()
    captured = _fake_request(monkeypatch, client, [_reply('{"a": 1}')])

    await client.chat_json([{"role": "user", "content": "Give JSON"}])

    _assert_no_generation_settings(captured[0]["json"])


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_llamaswap_chat_with_tools_without_settings_sends_no_generation_setting__DoD9(monkeypatch, stream):
    """DoD-9: chat_with_tools (every round, streamed or not) with no app settings sends no generation setting."""
    client = _client()

    _, bodies = await _run_tools(monkeypatch, client, stream=stream, rounds=[("", _ADD_CALL), ("3", [])])

    assert len(bodies) == 2
    for body in bodies:
        _assert_no_generation_settings(body)


# ---------------------------------------------------------------------------
# Offered settings reach the server (DoD 10)
# ---------------------------------------------------------------------------


def _assert_offered_settings(body: dict[str, Any]) -> None:
    """Every offered setting appears top-level in the body with the app's value (US-019.AC-1)."""
    for key, value in _OFFERED_SETTINGS.items():
        assert body.get(key) == value, f"{key} not sent as set"


def _settings_client(where: str) -> tuple[LlamaSwapChatCompletionsClient, dict[str, Any] | None]:
    """A client plus per-call options carrying the offered settings either as defaults or per call."""
    if where == "default":
        return _client(default_options=copy.deepcopy(_OFFERED_SETTINGS)), None
    return _client(), copy.deepcopy(_OFFERED_SETTINGS)


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["default", "call"])
async def test_llamaswap_chat_sends_offered_settings__DoD10(monkeypatch, where):
    """DoD-10: stop, token bias, end-user id and response format set by the app reach the chat body."""
    client, options = _settings_client(where)
    captured = _fake_request(monkeypatch, client, [_reply("ok")])

    await client.chat([{"role": "user", "content": "Hello"}], options=options)

    _assert_offered_settings(captured[0]["json"])


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["default", "call"])
async def test_llamaswap_streamed_chat_sends_offered_settings__DoD10(monkeypatch, where):
    """DoD-10: the offered settings also reach the body of a streamed chat."""
    client, options = _settings_client(where)
    session = _FakeSession(body=_complete_stream("ok"))
    client.session = session

    await client.chat([{"role": "user", "content": "Hello"}], options=options, stream=True)

    _assert_offered_settings(session.calls[0]["json"])


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["default", "call"])
async def test_llamaswap_generate_sends_offered_settings__DoD10(monkeypatch, where):
    """DoD-10: the offered settings reach the body of generate."""
    client, options = _settings_client(where)
    captured = _fake_request(monkeypatch, client, [_reply("ok")])

    await client.generate("Hello", options=options)

    _assert_offered_settings(captured[0]["json"])


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("where", ["default", "call"])
async def test_llamaswap_chat_with_tools_sends_offered_settings__DoD10(monkeypatch, where, stream):
    """DoD-10: the offered settings reach every round's body of chat_with_tools (no format parameter given)."""
    client, options = _settings_client(where)

    _, bodies = await _run_tools(
        monkeypatch, client, stream=stream, rounds=[("", _ADD_CALL), ("3", [])], options=options
    )

    assert len(bodies) == 2
    for body in bodies:
        _assert_offered_settings(body)


# ---------------------------------------------------------------------------
# Response format precedence (DoD 11, 12)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["default", "call"])
async def test_llamaswap_chat_json_sends_response_format_setting_when_set__DoD11(monkeypatch, where):
    """DoD-11: chat_json sends the app's response-format setting instead of the JSON-object format."""
    fmt = copy.deepcopy(_OFFERED_SETTINGS["response_format"])
    if where == "default":
        client, options = _client(default_options={"response_format": fmt}), None
    else:
        client, options = _client(), {"response_format": fmt}
    captured = _fake_request(monkeypatch, client, [_reply('{"x": 1}')])

    await client.chat_json([{"role": "user", "content": "Give JSON"}], options=options)

    assert captured[0]["json"]["response_format"] == _OFFERED_SETTINGS["response_format"]


@pytest.mark.asyncio
async def test_llamaswap_chat_json_sends_json_object_format_without_setting__DoD11(monkeypatch):
    """DoD-11: chat_json sends the JSON-object format when no response-format setting is set."""
    client = _client()
    captured = _fake_request(monkeypatch, client, [_reply('{"x": 1}')])

    await client.chat_json([{"role": "user", "content": "Give JSON"}])

    assert captured[0]["json"]["response_format"] == {"type": "json_object"}


_PARAM_FORMAT = {"type": "json_schema", "json_schema": {"name": "param_format", "schema": {"type": "object"}}}
_SETTING_FORMAT = {"type": "json_object"}


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_llamaswap_chat_with_tools_format_parameter_wins_over_setting__DoD12(monkeypatch, stream):
    """DoD-12: the response-format parameter is sent on every round even when a setting is set."""
    client = _client(default_options={"response_format": copy.deepcopy(_SETTING_FORMAT)})

    _, bodies = await _run_tools(
        monkeypatch,
        client,
        stream=stream,
        rounds=[("", _ADD_CALL), ("3", [])],
        options={"response_format": copy.deepcopy(_SETTING_FORMAT)},
        response_format=copy.deepcopy(_PARAM_FORMAT),
    )

    assert len(bodies) == 2
    for body in bodies:
        assert body["response_format"] == _PARAM_FORMAT


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_llamaswap_chat_with_tools_sends_setting_when_parameter_absent__DoD12(monkeypatch, stream):
    """DoD-12: without the parameter, the response-format setting is sent on every round."""
    client = _client(default_options={"response_format": copy.deepcopy(_SETTING_FORMAT)})

    _, bodies = await _run_tools(monkeypatch, client, stream=stream, rounds=[("", _ADD_CALL), ("3", [])])

    assert len(bodies) == 2
    for body in bodies:
        assert body["response_format"] == _SETTING_FORMAT


# ---------------------------------------------------------------------------
# Existing merge rules (DoD 13) and thinking switch (DoD 14)
# ---------------------------------------------------------------------------


async def _chat_body(
    monkeypatch, client: LlamaSwapChatCompletionsClient, options: dict[str, Any] | None
) -> dict[str, Any]:
    """Run a non-streamed chat and return the request body it sent."""
    captured = _fake_request(monkeypatch, client, [_reply("ok")])
    await client.chat([{"role": "user", "content": "Hello"}], options=options)
    return captured[0]["json"]


@pytest.mark.asyncio
async def test_llamaswap_call_none_removes_client_default__DoD13(monkeypatch):
    """DoD-13: a per-call None removes the client default; other defaults stay (US-015.AC-1, US-014.AC-2)."""
    client = _client(default_options={"temperature": 0.3, "seed": 7})

    body = await _chat_body(monkeypatch, client, {"temperature": None})

    assert "temperature" not in body
    assert body["seed"] == 7


@pytest.mark.asyncio
async def test_llamaswap_context_length_acts_as_max_tokens__DoD13(monkeypatch):
    """DoD-13: context_length is sent as max_tokens (US-017.AC-1)."""
    client = _client()

    body = await _chat_body(monkeypatch, client, {"context_length": 256})

    assert body["max_tokens"] == 256
    assert "context_length" not in body


@pytest.mark.asyncio
async def test_llamaswap_unknown_setting_not_sent__DoD13(monkeypatch):
    """DoD-13: a setting llama-swap does not accept is not sent, and the call succeeds (US-016.AC-1/AC-2)."""
    client = _client(default_options={"mirostat": 2})

    body = await _chat_body(monkeypatch, client, {"top_k": 40, "not_a_setting": "x"})

    assert "top_k" not in body
    assert "not_a_setting" not in body
    assert "mirostat" not in body


@pytest.mark.asyncio
async def test_llamaswap_call_value_wins_over_client_default__DoD13(monkeypatch):
    """DoD-13: a per-call value replaces the client default; untouched defaults are kept (US-014.AC-2)."""
    client = _client(default_options={"temperature": 0.2, "seed": 1})

    body = await _chat_body(monkeypatch, client, {"temperature": 0.9})

    assert body["temperature"] == 0.9
    assert body["seed"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("default", "call"),
    [({"max_tokens": 100}, {"context_length": 300}), ({"context_length": 100}, {"max_tokens": 300})],
)
async def test_llamaswap_call_value_wins_across_alternative_names__DoD13(monkeypatch, default, call):
    """DoD-13: default under one name, per-call value under the other: the per-call value wins (US-017.AC-2)."""
    client = _client(default_options=default)

    body = await _chat_body(monkeypatch, client, call)

    assert body["max_tokens"] == 300
    assert "context_length" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["default", "call"])
async def test_llamaswap_thinking_on_sends_switch__DoD14(monkeypatch, where):
    """DoD-14: thinking on is sent as the top-level thinking switch, without reasoning_effort (US-025.AC-1)."""
    if where == "default":
        client, options = _client(default_options={"enable_thinking": True}), None
    else:
        client, options = _client(), {"enable_thinking": True}

    body = await _chat_body(monkeypatch, client, options)

    assert body["enable_thinking"] is True
    assert "reasoning_effort" not in body


@pytest.mark.asyncio
async def test_llamaswap_reasoning_effort_never_sent__DoD14(monkeypatch):
    """DoD-14: reasoning_effort set by the app is never sent; the thinking switch still is (DEC-000-19)."""
    client = _client(default_options={"reasoning_effort": "low"})

    body = await _chat_body(monkeypatch, client, {"enable_thinking": True, "reasoning_effort": "high"})

    assert "reasoning_effort" not in body
    assert body["enable_thinking"] is True


@pytest.mark.asyncio
async def test_llamaswap_reasoning_effort_alone_sends_nothing__DoD14(monkeypatch):
    """DoD-14: reasoning_effort without the switch sends neither key."""
    client = _client()

    body = await _chat_body(monkeypatch, client, {"reasoning_effort": "high"})

    assert "reasoning_effort" not in body
    assert "enable_thinking" not in body


@pytest.mark.asyncio
async def test_llamaswap_thinking_off_sent_as_is__DoD14(monkeypatch):
    """DoD-14: the thinking switch set to off is sent as-is (sent only when set, DEC-000-19)."""
    client = _client()

    body = await _chat_body(monkeypatch, client, {"enable_thinking": False})

    assert body["enable_thinking"] is False
    assert "reasoning_effort" not in body


@pytest.mark.asyncio
async def test_llamaswap_thinking_unset_sends_neither__DoD14(monkeypatch):
    """DoD-14: thinking unset sends neither the thinking switch nor reasoning_effort."""
    client = _client()

    body = await _chat_body(monkeypatch, client, None)

    assert "enable_thinking" not in body
    assert "reasoning_effort" not in body


# ---------------------------------------------------------------------------
# Embedding end-user id (DoD 15)
# ---------------------------------------------------------------------------

# Client defaults mixing an end-user id with generation settings that must not reach an embedding.
_EMBED_DEFAULTS = {"user": "client-user", "temperature": 0.5, "top_p": 0.9, "max_tokens": 64, "seed": 3}


@pytest.mark.asyncio
async def test_llamaswap_embed_batch_call_user_wins__DoD15(monkeypatch):
    """DoD-15: embed_batch sends the call's end-user id over the client's, and no generation setting."""
    client = _client(default_options=dict(_EMBED_DEFAULTS))
    captured = _fake_request(monkeypatch, client, [{"data": [{"embedding": [0.1, 0.2]}]}])

    vectors = await client.embed_batch(["x"], options={"user": "call-user", "temperature": 0.1})

    assert vectors == [[0.1, 0.2]]
    body = captured[0]["json"]
    assert captured[0]["path"] == "/embeddings"
    assert body["user"] == "call-user"
    _assert_no_generation_settings(body)


@pytest.mark.asyncio
async def test_llamaswap_embed_batch_client_user_when_call_has_none__DoD15(monkeypatch):
    """DoD-15: embed_batch sends the client's end-user id when the call gives none, and no generation setting."""
    client = _client(default_options=dict(_EMBED_DEFAULTS))
    captured = _fake_request(monkeypatch, client, [{"data": [{"embedding": [0.3]}]}])

    await client.embed_batch(["x"])

    body = captured[0]["json"]
    assert body["user"] == "client-user"
    _assert_no_generation_settings(body)


# ---------------------------------------------------------------------------
# Feature 003.responses-chat, step 001: chat goes to /chat/completions (DoD-9)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_chat_completions_chat_posts_to_chat_completions__s001_DoD9(monkeypatch):
    """003 step 001 DoD-9 (US-056.AC-2): a non-streamed ``chat`` sends one ``POST /chat/completions``."""
    client = _client()
    captured = _fake_request(monkeypatch, client, [_reply("ok")])

    await client.chat([{"role": "user", "content": "Hello"}])

    assert [(c["method"], c["path"]) for c in captured] == [("POST", "/chat/completions")]


@pytest.mark.asyncio
async def test_llamaswap_chat_completions_streamed_chat_posts_to_chat_completions__s001_DoD9():
    """003 step 001 DoD-9 (US-056.AC-2): a streamed ``chat`` also sends its request to ``/chat/completions``."""
    client = _client()
    session = _FakeSession(body=_complete_stream("hi"))
    client.session = session

    await client.chat([{"role": "user", "content": "Hello"}], stream=True)

    assert [(call["method"], call["url"]) for call in session.calls] == [("POST", _CHAT_URL)]
