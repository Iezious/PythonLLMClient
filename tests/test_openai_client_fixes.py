"""Unit tests for the OpenAI client fixes (plan 001.core-fixes, step 004).

Covers the app-provided credential and its refusal (DEC-001-5/6), streamed requests through the base
helper with non-2xx and transport failures as ``LLMError`` (DEC-001-7/8), null reply content as empty
text (DEC-001-12), only app-set settings sent plus the offered keys (DEC-001-13/14), response-format
precedence (DEC-001-2/15), the unchanged merge and thinking rules, and the embedding end-user id rule
(DEC-001-16). Every expected value comes from the step's Definition of done and the cited decisions.

Feature 003.responses-chat, step 001: these tests now build ``OpenAIChatCompletionsClient`` (the
chat-completions regression suite, DoD-11), and the last section checks that its ``chat`` posts to
``/chat/completions`` (DoD-9 of that step, tagged ``__s001_DoD9``).
"""

import json
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest
from multidict import CIMultiDict
from yarl import URL

from llm.exceptions import LLMError
from llm.llm import get_llm_client
from llm.openai_chat_completions import OpenAIChatCompletionsClient

BASE_URL = "https://api.example.test/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"
CREDENTIAL = "sk-app-credential-7f3a"
MODEL = "gpt-4o-mini"

# Generation settings the library used to send on its own (US-018.AC-1, DEC-001-13).
GENERATION_KEYS = (
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "max_tokens",
    "seed",
    "reasoning_effort",
)
# Offered settings that must likewise stay absent when the app did not set them.
OFFERED_KEYS = ("stop", "logit_bias", "user", "enable_thinking")

# The offered settings an app may set (US-019.AC-1, DEC-001-14); response_format is a json_schema
# format so it is distinguishable from the built-in JSON-object request.
APP_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
    },
}
OFFERED_SETTINGS = {
    "stop": ["END"],
    "logit_bias": {"50256": -100},
    "user": "end-user-42",
    "response_format": APP_RESPONSE_FORMAT,
}

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_record",
            "description": "Retrieve a stored record by its key.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    }
]


def _fetch_record(key: str) -> str:
    return "alpha-77"


TOOLS = {"fetch_record": _fetch_record}

TOOL_CALLS = [
    {"id": "call_1", "type": "function", "function": {"name": "fetch_record", "arguments": '{"key": "k1"}'}},
]

USER_MESSAGES = [{"role": "user", "content": "Hello"}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reply(content: str | None, tool_calls: list[dict] | None = None) -> dict:
    """A non-streamed chat-completions reply body with the given message content."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}


def _capture_requests(monkeypatch, client: OpenAIChatCompletionsClient, replies: list[dict]) -> list[dict]:
    """Replace the ``_request`` seam with a fake that records each request and returns canned replies.

    Each record holds ``method``, ``path`` and a copy of the JSON body; messages are copied because the
    tool loop keeps appending to the same list (aliasing would hide what one round sent).
    """
    records: list[dict] = []
    pending = list(replies)

    async def fake_request(method, path, **kw):
        body = dict(kw["json"])
        if "messages" in body:
            body["messages"] = [dict(m) for m in body["messages"]]
        records.append({"method": method, "path": path, "json": body})
        return pending.pop(0)

    monkeypatch.setattr(client, "_request", fake_request)
    return records


def _sse(*payloads: Any) -> bytes:
    """Encode payloads as SSE ``data:`` events; strings are sent verbatim, other values as JSON."""
    parts = []
    for payload in payloads:
        text = payload if isinstance(payload, str) else json.dumps(payload)
        parts.append(f"data: {text}\n\n")
    return "".join(parts).encode("utf-8")


def _delta_frame(content: str) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}


_FINISH_FRAME = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}


class _Noop:
    """An awaitable that does nothing, so ``release()`` works whether or not the caller awaits it."""

    def __await__(self):
        return iter(())


class _FakeContent:
    """Stands in for ``aiohttp.StreamReader``: yields the body as the given byte chunks."""

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
        self.reason = "OK" if status < 400 else "Internal Server Error"
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
    """What ``session.request/post`` return: usable with ``async with`` or ``await``.

    A transport failure is raised when the request is opened, as aiohttp does.
    """

    def __init__(self, response: _FakeResponse | None, exc: BaseException | None) -> None:
        self._response = response
        self._exc = exc

    async def _open(self) -> _FakeResponse:
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return self._response

    def __await__(self):
        return self._open().__await__()

    async def __aenter__(self) -> _FakeResponse:
        return await self._open()

    async def __aexit__(self, *exc_info) -> bool:
        return False


class _FakeSession:
    """Fake ``aiohttp.ClientSession`` assigned to ``client.session`` (the client reuses an existing session).

    ``responder(method, url, kwargs)`` returns a ``_FakeResponse`` or an exception instance to raise on open.
    Every call is recorded so tests can inspect the headers the server would have received.
    """

    def __init__(self, responder) -> None:
        self._responder = responder
        self.calls: list[dict] = []
        self.closed = False

    def request(self, method: str, url: Any, **kwargs) -> _FakeRequestContext:
        self.calls.append({"method": method, "url": str(url), "kwargs": kwargs})
        outcome = self._responder(method, str(url), kwargs)
        if isinstance(outcome, BaseException):
            return _FakeRequestContext(None, outcome)
        return _FakeRequestContext(outcome, None)

    def post(self, url: Any, **kwargs) -> _FakeRequestContext:
        return self.request("POST", url, **kwargs)

    def get(self, url: Any, **kwargs) -> _FakeRequestContext:
        return self.request("GET", url, **kwargs)

    async def close(self) -> None:
        self.closed = True


def _authorization(call: dict) -> str | None:
    """The Authorization header of a recorded session call, matched case-insensitively."""
    headers = call["kwargs"].get("headers") or {}
    for name, value in dict(headers).items():
        if name.lower() == "authorization":
            return value
    return None


def _json_reply_session(content: str = "hi") -> _FakeSession:
    """A session answering every request with a 200 JSON chat reply."""
    body = json.dumps(_reply(content)).encode("utf-8")
    return _FakeSession(lambda method, url, kw: _FakeResponse(url, 200, body, "application/json"))


def _stream_session(body: bytes) -> _FakeSession:
    """A session answering every request with a 200 SSE stream carrying ``body``."""
    return _FakeSession(lambda method, url, kw: _FakeResponse(url, 200, body, "text/event-stream"))


def _error_session(status: int, body: str) -> _FakeSession:
    """A session answering every request with a non-2xx reply whose body is ``body``."""
    data = body.encode("utf-8")
    return _FakeSession(lambda method, url, kw: _FakeResponse(url, status, data, "application/json"))


def _failing_session(exc: BaseException) -> _FakeSession:
    """A session whose every request fails to open with ``exc`` (unreachable server / timeout)."""
    return _FakeSession(lambda method, url, kw: exc)


def _client(**kwargs) -> OpenAIChatCompletionsClient:
    return OpenAIChatCompletionsClient(model=MODEL, base_url=BASE_URL, bearer_token=CREDENTIAL, **kwargs)


@pytest.fixture
def no_env_credential(monkeypatch):
    """Remove OpenAI credentials from the process environment so a refusal cannot depend on it."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPEN_AI_KEY", raising=False)


# ---------------------------------------------------------------------------
# Credential (DoD-1 .. DoD-5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_non_streamed_request_carries_app_credential__DoD1(no_env_credential):
    """DoD-1: a non-streamed chat sends ``Authorization: Bearer <the app's credential>``."""
    client = _client()
    session = _json_reply_session()
    client.session = session

    await client.chat(USER_MESSAGES)

    assert session.calls, "the chat must reach the server"
    assert all(_authorization(call) == f"Bearer {CREDENTIAL}" for call in session.calls)


@pytest.mark.asyncio
async def test_openai_streamed_request_carries_app_credential__DoD1(no_env_credential):
    """DoD-1: a streamed chat also sends ``Authorization: Bearer <the app's credential>``."""
    client = _client()
    session = _stream_session(_sse(_delta_frame("ok"), _FINISH_FRAME, "[DONE]"))
    client.session = session

    await client.chat(USER_MESSAGES, stream=True)

    assert session.calls, "the streamed chat must reach the server"
    assert all(_authorization(call) == f"Bearer {CREDENTIAL}" for call in session.calls)


@pytest.mark.parametrize("missing", [None, ""])
def test_openai_missing_credential_refused_with_guidance__DoD2(no_env_credential, missing):
    """DoD-2: no or empty credential raises ValueError naming ``bearer_token`` and no environment variable."""
    with pytest.raises(ValueError) as excinfo:
        OpenAIChatCompletionsClient(model=MODEL, base_url=BASE_URL, bearer_token=missing)

    message = str(excinfo.value)
    assert "bearer_token" in message
    assert "OPENAI_API_KEY" not in message
    assert "OPEN_AI_KEY" not in message


def test_openai_credential_in_environment_is_not_used__DoD3(monkeypatch):
    """DoD-3: with both environment keys set and no credential passed, building the client is refused."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-openai-api-key-111")
    monkeypatch.setenv("OPEN_AI_KEY", "sk-env-open-ai-key-222")

    with pytest.raises(ValueError):
        OpenAIChatCompletionsClient(model=MODEL, base_url=BASE_URL)


def test_openai_credential_in_environment_empty_argument_refused__DoD3(monkeypatch):
    """DoD-3: an empty credential is refused too, even with environment keys present."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-openai-api-key-111")
    monkeypatch.setenv("OPEN_AI_KEY", "sk-env-open-ai-key-222")

    with pytest.raises(ValueError):
        OpenAIChatCompletionsClient(model=MODEL, base_url=BASE_URL, bearer_token="")


def test_openai_refusal_message_contains_no_credential_value__DoD4(monkeypatch):
    """DoD-4: the refusal message holds no credential value, not even one present in the environment."""
    env_values = ("sk-env-openai-api-key-111", "sk-env-open-ai-key-222")
    monkeypatch.setenv("OPENAI_API_KEY", env_values[0])
    monkeypatch.setenv("OPEN_AI_KEY", env_values[1])

    with pytest.raises(ValueError) as excinfo:
        OpenAIChatCompletionsClient(model=MODEL, base_url=BASE_URL)

    rendered = f"{excinfo.value} {excinfo.value.args!r}"
    for value in env_values:
        assert value not in rendered


@pytest.mark.asyncio
async def test_openai_factory_forwards_credential__DoD5(no_env_credential):
    """DoD-5: ``get_llm_client("openai", ...)`` with a credential gives a client that sends it."""
    client = get_llm_client("openai", BASE_URL, MODEL, bearer_token=CREDENTIAL, protocol="chat_completions")
    assert isinstance(client, OpenAIChatCompletionsClient)
    session = _json_reply_session()
    client.session = session

    await client.chat(USER_MESSAGES)

    assert session.calls
    assert all(_authorization(call) == f"Bearer {CREDENTIAL}" for call in session.calls)


def test_openai_factory_without_credential_raises__DoD5(no_env_credential):
    """DoD-5: ``get_llm_client("openai", ...)`` without a credential raises ValueError."""
    with pytest.raises(ValueError):
        get_llm_client("openai", BASE_URL, MODEL, protocol="chat_completions")


# ---------------------------------------------------------------------------
# Streamed requests through the base helper (DoD-6 .. DoD-9)
# ---------------------------------------------------------------------------

_SERVER_ERROR_BODY = '{"error": {"message": "The server had an error processing your request (ref 91x)."}}'


@pytest.mark.asyncio
async def test_openai_streamed_chat_non_2xx_raises_llmerror__DoD6():
    """DoD-6: a non-2xx streamed chat raises LLMError with address, status and reply, and no credential."""
    client = _client()
    client.session = _error_session(500, _SERVER_ERROR_BODY)

    with pytest.raises(LLMError) as excinfo:
        await client.chat(USER_MESSAGES, stream=True)

    err = excinfo.value
    rendered = str(err)
    assert CHAT_URL in rendered
    assert "500" in rendered
    assert _SERVER_ERROR_BODY in rendered
    everything = f"{rendered} {err.message} {err.description} {err.args!r}"
    assert CREDENTIAL not in everything


@pytest.mark.asyncio
async def test_openai_tools_stream_request_non_2xx_raises_llmerror__DoD7():
    """DoD-7: a streamed tool round with a non-2xx reply raises the same kind of LLMError."""
    client = _client()
    client.session = _error_session(503, _SERVER_ERROR_BODY)
    payload = {"model": MODEL, "messages": list(USER_MESSAGES), "tools": TOOL_DEFS, "stream": True}

    with pytest.raises(LLMError) as excinfo:
        await client._tools_stream_request(payload)

    err = excinfo.value
    rendered = str(err)
    assert CHAT_URL in rendered
    assert "503" in rendered
    assert _SERVER_ERROR_BODY in rendered
    assert CREDENTIAL not in f"{rendered} {err.message} {err.description} {err.args!r}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        aiohttp.ClientConnectionError("Cannot connect to host api.example.test:443"),
        TimeoutError(),
    ],
    ids=["unreachable", "timeout"],
)
async def test_openai_streamed_chat_transport_failure_raises_llmerror__DoD8(failure):
    """DoD-8: a streamed chat against an unreachable (or timed-out) server raises LLMError."""
    client = _client()
    client.session = _failing_session(failure)

    with pytest.raises(LLMError):
        await client.chat(USER_MESSAGES, stream=True)


@pytest.mark.asyncio
async def test_openai_streamed_chat_delivers_pieces_in_order__DoD9():
    """DoD-9: the receiver gets every piece in order and the call returns their concatenation."""
    client = _client()
    pieces = ["Hel", "lo", " world"]
    body = _sse(*[_delta_frame(p) for p in pieces], _FINISH_FRAME, "[DONE]")
    client.session = _stream_session(body)
    received: list[str] = []

    async def on_delta(text: str) -> None:
        received.append(text)

    result = await client.chat(USER_MESSAGES, stream=True, on_delta=on_delta)

    assert received == pieces
    assert result == "Hello world"


# ---------------------------------------------------------------------------
# Null reply content (DoD-10 .. DoD-12)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_chat_null_content_returns_empty_text__DoD10(monkeypatch):
    """DoD-10: a non-streamed chat whose reply content is null returns empty text."""
    client = _client()
    _capture_requests(monkeypatch, client, [_reply(None)])

    assert await client.chat(USER_MESSAGES) == ""


@pytest.mark.asyncio
async def test_openai_generate_null_content_returns_empty_text__DoD10(monkeypatch):
    """DoD-10: a non-streamed generate whose reply content is null returns empty text."""
    client = _client()
    _capture_requests(monkeypatch, client, [_reply(None)])

    assert await client.generate("Say something") == ""


@pytest.mark.asyncio
async def test_openai_chat_null_content_with_tracing_returns_empty_text__DoD11(monkeypatch, tmp_path):
    """DoD-11: with tracing on, a non-streamed chat with null content returns empty text."""
    # The tracer reads its switches when the client is built (DEC-000-4), so set them first.
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(tmp_path))
    client = _client()
    _capture_requests(monkeypatch, client, [_reply(None)])

    assert await client.chat(USER_MESSAGES) == ""


@pytest.mark.asyncio
async def test_openai_chat_json_null_content_returns_empty_text__DoD12(monkeypatch):
    """DoD-12: chat_json with null reply content returns empty text."""
    client = _client()
    _capture_requests(monkeypatch, client, [_reply(None)])

    assert await client.chat_json(USER_MESSAGES) == ""


@pytest.mark.asyncio
async def test_openai_chat_with_tools_null_final_answer_returns_empty_text__DoD12(monkeypatch):
    """DoD-12: chat_with_tools whose final answer has null content returns empty text."""
    client = _client()
    _capture_requests(monkeypatch, client, [_reply(None)])

    result = await client.chat_with_tools(USER_MESSAGES, tools_definitions=TOOL_DEFS, tools=TOOLS)

    assert result == ""


@pytest.mark.asyncio
async def test_openai_tool_call_turn_null_content_enters_next_round_as_empty_text__DoD12(monkeypatch):
    """DoD-12: an assistant tool-call turn with null content is sent in the next round with empty text."""
    client = _client()
    records = _capture_requests(monkeypatch, client, [_reply(None, TOOL_CALLS), _reply("done")])

    result = await client.chat_with_tools(USER_MESSAGES, tools_definitions=TOOL_DEFS, tools=TOOLS)

    assert result == "done"
    assert len(records) == 2
    second_round = records[1]["json"]["messages"]
    tool_call_turns = [m for m in second_round if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(tool_call_turns) == 1
    assert tool_call_turns[0]["content"] == ""


# ---------------------------------------------------------------------------
# Only the app's settings are sent; offered settings reach the server (DoD-13 .. DoD-16)
# ---------------------------------------------------------------------------


def _assert_no_generation_setting(body: dict) -> None:
    for key in GENERATION_KEYS + OFFERED_KEYS:
        assert key not in body, f"{key} was sent although the app did not set it"


@pytest.mark.asyncio
async def test_openai_chat_sends_no_unset_setting__DoD13(monkeypatch):
    """DoD-13: with no settings from the app, chat's request body carries no generation setting."""
    client = _client()
    records = _capture_requests(monkeypatch, client, [_reply("ok")])

    await client.chat(USER_MESSAGES)

    _assert_no_generation_setting(records[0]["json"])
    assert "response_format" not in records[0]["json"]


@pytest.mark.asyncio
async def test_openai_chat_json_sends_no_unset_setting__DoD13(monkeypatch):
    """DoD-13: with no settings from the app, chat_json's request body carries no generation setting."""
    client = _client()
    records = _capture_requests(monkeypatch, client, [_reply('{"a": 1}')])

    await client.chat_json(USER_MESSAGES)

    _assert_no_generation_setting(records[0]["json"])


@pytest.mark.asyncio
async def test_openai_chat_with_tools_sends_no_unset_setting__DoD13(monkeypatch):
    """DoD-13: with no settings from the app, chat_with_tools' request body carries no generation setting."""
    client = _client()
    records = _capture_requests(monkeypatch, client, [_reply("ok")])

    await client.chat_with_tools(USER_MESSAGES, tools_definitions=TOOL_DEFS, tools=TOOLS)

    _assert_no_generation_setting(records[0]["json"])
    assert "response_format" not in records[0]["json"]


def _settings_split(source: str) -> tuple[dict | None, dict | None]:
    """Return (client defaults, per-call options) carrying the offered settings from ``source``."""
    settings = json.loads(json.dumps(OFFERED_SETTINGS))
    return (settings, None) if source == "defaults" else (None, settings)


def _assert_offered_settings_sent(body: dict) -> None:
    for key, value in OFFERED_SETTINGS.items():
        assert body.get(key) == value, f"{key} did not reach the request body"


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["defaults", "call"])
async def test_openai_chat_sends_offered_settings__DoD14(monkeypatch, source):
    """DoD-14: stop, token bias, end-user id and response format set by the app reach chat's body."""
    defaults, call_options = _settings_split(source)
    client = _client(default_options=defaults)
    records = _capture_requests(monkeypatch, client, [_reply("ok")])

    await client.chat(USER_MESSAGES, options=call_options)

    _assert_offered_settings_sent(records[0]["json"])


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["defaults", "call"])
async def test_openai_generate_sends_offered_settings__DoD14(monkeypatch, source):
    """DoD-14: stop, token bias, end-user id and response format set by the app reach generate's body."""
    defaults, call_options = _settings_split(source)
    client = _client(default_options=defaults)
    records = _capture_requests(monkeypatch, client, [_reply("ok")])

    await client.generate("Say something", options=call_options)

    _assert_offered_settings_sent(records[0]["json"])


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["defaults", "call"])
async def test_openai_chat_with_tools_sends_offered_settings__DoD14(monkeypatch, source):
    """DoD-14: stop, token bias, end-user id and response format set by the app reach chat_with_tools' body."""
    defaults, call_options = _settings_split(source)
    client = _client(default_options=defaults)
    records = _capture_requests(monkeypatch, client, [_reply("ok")])

    await client.chat_with_tools(USER_MESSAGES, tools_definitions=TOOL_DEFS, tools=TOOLS, options=call_options)

    _assert_offered_settings_sent(records[0]["json"])


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["defaults", "call"])
async def test_openai_chat_json_sends_response_format_setting__DoD15(monkeypatch, source):
    """DoD-15: chat_json sends the app's response-format setting when one is set."""
    defaults = {"response_format": APP_RESPONSE_FORMAT} if source == "defaults" else None
    call_options = {"response_format": APP_RESPONSE_FORMAT} if source == "call" else None
    client = _client(default_options=defaults)
    records = _capture_requests(monkeypatch, client, [_reply('{"answer": "x"}')])

    await client.chat_json(USER_MESSAGES, options=call_options)

    assert records[0]["json"]["response_format"] == APP_RESPONSE_FORMAT


@pytest.mark.asyncio
async def test_openai_chat_json_sends_json_object_without_setting__DoD15(monkeypatch):
    """DoD-15: without a response-format setting chat_json sends the JSON-object format."""
    client = _client()
    records = _capture_requests(monkeypatch, client, [_reply('{"a": 1}')])

    await client.chat_json(USER_MESSAGES)

    assert records[0]["json"]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_openai_chat_with_tools_parameter_wins_over_setting__DoD16(monkeypatch):
    """DoD-16: the response-format parameter is sent on every round even when a setting is set."""
    parameter_format = {"type": "json_object"}
    client = _client(default_options={"response_format": APP_RESPONSE_FORMAT})
    records = _capture_requests(monkeypatch, client, [_reply(None, TOOL_CALLS), _reply('{"answer": "x"}')])

    await client.chat_with_tools(
        USER_MESSAGES,
        tools_definitions=TOOL_DEFS,
        tools=TOOLS,
        options={"response_format": APP_RESPONSE_FORMAT},
        response_format=parameter_format,
    )

    assert len(records) == 2
    for record in records:
        assert record["json"]["response_format"] == parameter_format


@pytest.mark.asyncio
async def test_openai_chat_with_tools_setting_sent_without_parameter__DoD16(monkeypatch):
    """DoD-16: without the parameter, the response-format setting is sent."""
    client = _client(default_options={"response_format": APP_RESPONSE_FORMAT})
    records = _capture_requests(monkeypatch, client, [_reply('{"answer": "x"}')])

    await client.chat_with_tools(USER_MESSAGES, tools_definitions=TOOL_DEFS, tools=TOOLS)

    assert records[0]["json"]["response_format"] == APP_RESPONSE_FORMAT


# ---------------------------------------------------------------------------
# Existing merge and thinking rules (DoD-17, DoD-18)
# ---------------------------------------------------------------------------


async def _chat_body(monkeypatch, defaults: dict | None, call_options: dict | None) -> dict:
    """Run one non-streamed chat and return the request body it sent."""
    client = _client(default_options=defaults)
    records = _capture_requests(monkeypatch, client, [_reply("ok")])
    await client.chat(USER_MESSAGES, options=call_options)
    return records[0]["json"]


@pytest.mark.asyncio
async def test_openai_call_none_removes_client_default__DoD17(monkeypatch):
    """DoD-17 (US-015.AC-1): a per-call ``None`` removes a client default from the request."""
    body = await _chat_body(monkeypatch, {"temperature": 0.3}, {"temperature": None})

    assert "temperature" not in body


@pytest.mark.asyncio
async def test_openai_context_length_acts_as_max_tokens__DoD17(monkeypatch):
    """DoD-17 (US-017.AC-1): ``context_length`` is sent as ``max_tokens``."""
    body = await _chat_body(monkeypatch, None, {"context_length": 256})

    assert body["max_tokens"] == 256
    assert "context_length" not in body


@pytest.mark.asyncio
async def test_openai_alias_per_call_wins_over_default_under_main_name__DoD17(monkeypatch):
    """DoD-17 (US-017.AC-2): a per-call alias value wins over a client default under the main name."""
    body = await _chat_body(monkeypatch, {"max_tokens": 100}, {"context_length": 200})

    assert body["max_tokens"] == 200


@pytest.mark.asyncio
async def test_openai_unknown_setting_not_sent__DoD17(monkeypatch):
    """DoD-17 (US-016.AC-2): an unknown setting is not sent and the call succeeds."""
    body = await _chat_body(monkeypatch, None, {"top_k": 40})

    assert "top_k" not in body


@pytest.mark.asyncio
async def test_openai_call_value_wins_over_client_default__DoD17(monkeypatch):
    """DoD-17 (US-014.AC-2): a per-call value wins and the other client defaults are kept."""
    body = await _chat_body(monkeypatch, {"temperature": 0.3, "top_p": 0.5}, {"temperature": 0.9})

    assert body["temperature"] == 0.9
    assert body["top_p"] == 0.5


@pytest.mark.asyncio
async def test_openai_thinking_on_without_effort_sends_default_effort__DoD18(monkeypatch):
    """DoD-18 (US-025.AC-3): thinking on with no effort sends the default effort, ``medium`` (DEC-000-19)."""
    body = await _chat_body(monkeypatch, None, {"enable_thinking": True})

    assert body["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_openai_explicit_effort_sent_as_given__DoD18(monkeypatch):
    """DoD-18 (US-025.AC-4): an explicit effort is sent as given, with thinking on or without it."""
    with_thinking = await _chat_body(monkeypatch, None, {"enable_thinking": True, "reasoning_effort": "high"})
    effort_only = await _chat_body(monkeypatch, None, {"reasoning_effort": "low"})

    assert with_thinking["reasoning_effort"] == "high"
    assert effort_only["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_openai_thinking_unset_sends_no_effort__DoD18(monkeypatch):
    """DoD-18 (DEC-000-19): with the thinking switch unset no effort is sent."""
    body = await _chat_body(monkeypatch, None, None)

    assert "reasoning_effort" not in body


# ---------------------------------------------------------------------------
# Embedding end-user id (DoD-19)
# ---------------------------------------------------------------------------

# Client defaults mixing generation settings with an end-user id; none of the generation settings may
# reach an embedding request (US-041.AC-4).
_EMBED_CLIENT_DEFAULTS = {
    "user": "client-user",
    "temperature": 0.4,
    "top_p": 0.8,
    "max_tokens": 64,
    "presence_penalty": 0.1,
    "frequency_penalty": 0.2,
    "seed": 7,
    "stop": ["END"],
    "logit_bias": {"1": 2},
    "reasoning_effort": "low",
}

_EMBED_REPLY = {"data": [{"embedding": [0.1, 0.2]}]}


def _assert_no_generation_in_embedding(body: dict) -> None:
    for key in GENERATION_KEYS + ("stop", "logit_bias", "enable_thinking", "response_format"):
        assert key not in body, f"{key} reached the embedding request"


@pytest.mark.asyncio
async def test_openai_embed_batch_call_user_wins__DoD19(monkeypatch):
    """DoD-19 (US-041.AC-3): the call's end-user id is sent over the client's; no generation setting."""
    client = _client(default_options=dict(_EMBED_CLIENT_DEFAULTS))
    records = _capture_requests(monkeypatch, client, [_EMBED_REPLY])

    await client.embed_batch(["text"], options={"user": "call-user"})

    assert records[0]["path"] == "/embeddings"
    assert records[0]["json"]["user"] == "call-user"
    _assert_no_generation_in_embedding(records[0]["json"])


@pytest.mark.asyncio
async def test_openai_embed_batch_client_user_when_call_has_none__DoD19(monkeypatch):
    """DoD-19 (US-041.AC-4): the client's end-user id is sent when the call has none; no generation setting."""
    client = _client(default_options=dict(_EMBED_CLIENT_DEFAULTS))
    records = _capture_requests(monkeypatch, client, [_EMBED_REPLY])

    await client.embed_batch(["text"])

    assert records[0]["json"]["user"] == "client-user"
    _assert_no_generation_in_embedding(records[0]["json"])


# ---------------------------------------------------------------------------
# Feature 003.responses-chat, step 001: chat goes to /chat/completions (DoD-9)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_chat_completions_chat_posts_to_chat_completions__s001_DoD9(monkeypatch):
    """003 step 001 DoD-9 (US-056.AC-2): a non-streamed ``chat`` sends one ``POST /chat/completions``."""
    client = _client()
    records = _capture_requests(monkeypatch, client, [_reply("ok")])

    await client.chat(USER_MESSAGES)

    assert [(r["method"], r["path"]) for r in records] == [("POST", "/chat/completions")]


@pytest.mark.asyncio
async def test_openai_chat_completions_streamed_chat_posts_to_chat_completions__s001_DoD9():
    """003 step 001 DoD-9 (US-056.AC-2): a streamed ``chat`` also sends its request to ``/chat/completions``."""
    client = _client()
    session = _stream_session(_sse(_delta_frame("ok"), _FINISH_FRAME, "[DONE]"))
    client.session = session

    await client.chat(USER_MESSAGES, stream=True)

    assert [(call["method"], call["url"]) for call in session.calls] == [("POST", CHAT_URL)]
