"""Unit tests for the llama-swap Responses client: settings, request and non-streamed reply.

Feature 003.responses-chat, step 004. Covers ``LlamaSwapAPIClient`` on the Responses protocol for
non-streamed ``chat``, ``generate`` and ``chat_json``: the exact request body (DEC-003-11), the settings
mapping and the ignored-setting warning (DEC-003-9, DEC-003-6), thinking as the reasoning effort without a
summary (DEC-003-1, DEC-003-2), the text-format translation (DEC-003-10), reply extraction and failed /
incomplete replies (DEC-003-12), non-2xx failures, the optional credential, the unchanged model-list /
embedding operations and tracing. Each test names its DoD id. (003's tool-chat refusal test, DoD-14, was
removed by feature 004 step 003, which replaces the refusal with the Responses tool loop.)
"""

import copy
import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest
from multidict import CIMultiDict
from yarl import URL

from llm.exceptions import LLMError, LLMSettingIgnoredWarning
from llm.llama_swap import LlamaSwapAPIClient
from llm.llama_swap_chat_completions import LlamaSwapChatCompletionsClient

_BASE_URL = "http://localhost:8080/v1"
_MODEL = "some-model"
_SECRET = "sk-llamaswap-secret-token-123"

# The exact body keys of a settings-free chat-style request (DEC-003-11).
_BASE_KEYS = {"model", "input", "stream", "store", "include"}
_INCLUDE = ["reasoning.encrypted_content"]

_USER_MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "Hello"}]

# Supported keys without a Responses counterpart, with recognisable values (DEC-003-9).
_NO_COUNTERPART: dict[str, Any] = {
    "presence_penalty": 0.5,
    "frequency_penalty": 0.4,
    "seed": 7,
    "stop": ["END"],
    "logit_bias": {"50256": -100.0},
}


# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch):
    """Start every test with tracing off so a developer's shell env cannot change behaviour (DEC-000-48)."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _client(**kw: Any) -> LlamaSwapAPIClient:
    """Build a llama-swap Responses client on the fixed test address; credential and defaults from ``kw``."""
    return LlamaSwapAPIClient(model=_MODEL, base_url=_BASE_URL, **kw)


def _message_item(*texts: str | None, extra_parts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """A Responses ``message`` output item with one ``output_text`` part per text (``None`` = null text)."""
    parts: list[dict[str, Any]] = [{"type": "output_text", "text": t, "annotations": []} for t in texts]
    parts.extend(extra_parts or [])
    return {"type": "message", "role": "assistant", "status": "completed", "content": parts}


def _reasoning_item(text: str = "Let me think about this.") -> dict[str, Any]:
    """A reasoning item in llama.cpp's shape: ``reasoning_text`` content, empty summary and encrypted content."""
    return {
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
        "encrypted_content": "",
    }


def _reply(*items: dict[str, Any], status: str = "completed", **extra: Any) -> dict[str, Any]:
    """A non-streamed Responses reply mapping with the given output items."""
    return {"object": "response", "status": status, "output": list(items), **extra}


def _text_reply(text: str) -> dict[str, Any]:
    """A completed reply whose single message item holds ``text``."""
    return _reply(_message_item(text))


def _fake_request(monkeypatch, client: Any, replies: list[Any]) -> list[dict[str, Any]]:
    """Replace ``_request`` with a fake returning ``replies`` in order; returns every captured request.

    The JSON body is deep-copied so later mutation by the client cannot alter what the test inspects.
    """
    captured: list[dict[str, Any]] = []

    async def fake_request(method, path, **kw):
        captured.append({"method": method, "path": path, "json": copy.deepcopy(kw.get("json")), "kw": kw})
        return replies[min(len(captured), len(replies)) - 1]

    monkeypatch.setattr(client, "_request", fake_request)
    return captured


async def _chat_body(monkeypatch, client: LlamaSwapAPIClient | None = None, **chat_kw: Any) -> dict[str, Any]:
    """Run one non-streamed ``chat`` against a canned reply and return the single request body sent."""
    if client is None:
        client = _client()
    captured = _fake_request(monkeypatch, client, [_text_reply("ok")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LLMSettingIgnoredWarning)
        await client.chat(chat_kw.pop("messages", _USER_MESSAGES), **chat_kw)
    assert len(captured) == 1
    return captured[0]["json"]


def _ignored(records: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    """Only the library's ignored-setting warnings out of a ``catch_warnings(record=True)`` list."""
    return [w for w in records if issubclass(w.category, LLMSettingIgnoredWarning)]


class _Noop:
    """An awaitable that does nothing, so ``release()`` works whether or not the caller awaits it."""

    def __await__(self):
        return iter(())


class _FakeContent:
    """Stands in for ``aiohttp.StreamReader``: yields the whole body as one chunk."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    async def iter_chunked(self, n: int):
        yield self._body

    async def iter_any(self):
        yield self._body


class _FakeResponse:
    """A minimal ``aiohttp.ClientResponse`` stand-in: status, headers and body reads."""

    def __init__(self, url: str, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.url = URL(url)
        self.real_url = self.url
        self.status = status
        self.reason = "OK" if status < 400 else "Not Found"
        self.content_type = content_type
        self.headers = CIMultiDict({"Content-Type": content_type})
        self._body = body
        self.content = _FakeContent(body)

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
    """What ``session.request/post/get`` return: usable with ``async with`` or ``await``."""

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
    """Fake ``aiohttp.ClientSession`` installed as ``client.session`` so the real ``_request`` runs.

    Answers every request with ``status`` and the JSON ``body``; records method, URL and keyword
    arguments of each call so tests can inspect the address and headers the server would have seen.
    """

    def __init__(self, status: int, body: Any) -> None:
        self._status = status
        self._body = json.dumps(body).encode("utf-8")
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def request(self, method: str, url: Any, **kwargs) -> _FakeRequestContext:
        self.calls.append({"method": method.upper(), "url": str(url), "kwargs": kwargs})
        return _FakeRequestContext(_FakeResponse(str(url), self._status, self._body))

    def post(self, url: Any, **kwargs) -> _FakeRequestContext:
        return self.request("POST", url, **kwargs)

    def get(self, url: Any, **kwargs) -> _FakeRequestContext:
        return self.request("GET", url, **kwargs)

    async def close(self) -> None:
        self.closed = True


def _authorization(call: dict[str, Any]) -> str | None:
    """The Authorization header of a recorded session call, matched case-insensitively."""
    headers = call["kwargs"].get("headers") or {}
    for name, value in dict(headers).items():
        if str(name).lower() == "authorization":
            return value
    return None


# ---------------------------------------------------------------------------
# DoD-1, DoD-2, DoD-3 — request body and input items
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_chat_sends_one_responses_request_with_exact_keys__DoD1(monkeypatch):
    """DoD-1: a settings-free chat sends one POST /responses with exactly model/input/stream/store/include."""
    client = _client()
    captured = _fake_request(monkeypatch, client, [_text_reply("ok")])

    await client.chat(_USER_MESSAGES)

    assert len(captured) == 1
    assert captured[0]["method"] == "POST"
    assert captured[0]["path"] == "/responses"
    body = captured[0]["json"]
    assert set(body) == _BASE_KEYS
    assert body["model"] == _MODEL
    assert body["stream"] is False
    assert body["store"] is False
    assert body["include"] == _INCLUDE


# A conversation with every caller role, and the input items DoD-2 (DEC-003-11 as amended 2026-11-08)
# requires for it behind a head system item: system / user bare role/content, assistant typed ``message``.
_MIXED_MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "Caller system"},
    {"role": "user", "content": "First"},
    {"role": "assistant", "content": "Second"},
    {"role": "user", "content": "Third"},
]
_MIXED_ITEMS_WITH_HEAD: list[dict[str, Any]] = [
    {"role": "system", "content": "Be brief."},
    {"role": "system", "content": "Caller system"},
    {"role": "user", "content": "First"},
    {"type": "message", "role": "assistant", "content": "Second"},
    {"role": "user", "content": "Third"},
]


@pytest.mark.asyncio
async def test_llamaswap_chat_messages_become_items_with_system_first__DoD2(monkeypatch):
    """DoD-2: messages become items in order behind a bare head system item; assistant typed, others bare."""
    messages = copy.deepcopy(_MIXED_MESSAGES)
    original = copy.deepcopy(messages)

    body = await _chat_body(monkeypatch, messages=messages, system="Be brief.")

    assert body["input"] == _MIXED_ITEMS_WITH_HEAD
    assert messages == original


@pytest.mark.asyncio
async def test_llamaswap_chat_item_key_sets_per_role__DoD2(monkeypatch):
    """DoD-2: the assistant item has exactly type/role/content; system, user and head system items are bare."""
    messages = [
        {"role": "system", "content": "S", "name": "sys"},
        {"role": "user", "content": "U", "name": "bob"},
        {"role": "assistant", "content": "A", "name": "helper"},
    ]
    original = copy.deepcopy(messages)

    body = await _chat_body(monkeypatch, messages=messages, system="Head")

    items = body["input"]
    assert len(items) == 4
    assert items[0] == {"role": "system", "content": "Head"}
    assert set(items[0]) == {"role", "content"}
    assert items[1] == {"role": "system", "content": "S"}
    assert set(items[1]) == {"role", "content"}
    assert items[2] == {"role": "user", "content": "U"}
    assert set(items[2]) == {"role", "content"}
    assert items[3] == {"type": "message", "role": "assistant", "content": "A"}
    assert set(items[3]) == {"type", "role", "content"}
    assert messages == original


@pytest.mark.asyncio
async def test_llamaswap_chat_json_builds_the_same_input_items__DoD2(monkeypatch):
    """DoD-2: chat_json builds the same input items as chat (typed assistant, bare system / user / head)."""
    messages = copy.deepcopy(_MIXED_MESSAGES)
    original = copy.deepcopy(messages)
    client = _client()
    captured = _fake_request(monkeypatch, client, [_text_reply('{"a": 1}')])

    await client.chat_json(messages, system="Be brief.")

    assert len(captured) == 1
    assert captured[0]["json"]["input"] == _MIXED_ITEMS_WITH_HEAD
    assert messages == original


@pytest.mark.asyncio
@pytest.mark.parametrize("system", ["", None])
async def test_llamaswap_chat_empty_system_adds_no_item__DoD2(monkeypatch, system):
    """DoD-2: an empty (or absent) system instruction adds no system item; the caller's list is unchanged."""
    messages = [{"role": "user", "content": "Only"}]
    original = copy.deepcopy(messages)

    body = await _chat_body(monkeypatch, messages=messages, system=system)

    assert body["input"] == [{"role": "user", "content": "Only"}]
    assert messages == original


@pytest.mark.asyncio
async def test_llamaswap_chat_items_carry_only_role_and_content__DoD2(monkeypatch):
    """DoD-2: a user input item holds only the role and content keys, values as given (DEC-003-11)."""
    messages = [{"role": "user", "content": "Hi", "name": "bob"}]
    original = copy.deepcopy(messages)

    body = await _chat_body(monkeypatch, messages=messages)

    assert body["input"] == [{"role": "user", "content": "Hi"}]
    assert messages == original


@pytest.mark.asyncio
async def test_llamaswap_generate_sends_system_and_user_prompt_items__DoD3(monkeypatch):
    """DoD-3: generate sends the system item then one user item holding the prompt, to /responses."""
    client = _client()
    captured = _fake_request(monkeypatch, client, [_text_reply("answer")])

    result = await client.generate("Tell me a joke", system="You are funny.")

    assert result == "answer"
    assert len(captured) == 1
    assert captured[0]["path"] == "/responses"
    assert captured[0]["json"]["input"] == [
        {"role": "system", "content": "You are funny."},
        {"role": "user", "content": "Tell me a joke"},
    ]
    assert captured[0]["json"]["stream"] is False


@pytest.mark.asyncio
async def test_llamaswap_generate_without_system_sends_one_user_item__DoD3(monkeypatch):
    """DoD-3: generate without a system instruction sends exactly one user item holding the prompt."""
    client = _client()
    captured = _fake_request(monkeypatch, client, [_text_reply("answer")])

    await client.generate("Tell me a joke")

    assert captured[0]["json"]["input"] == [{"role": "user", "content": "Tell me a joke"}]


# ---------------------------------------------------------------------------
# DoD-4, DoD-5 — reply extraction, failed and incomplete replies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_chat_concatenates_output_text_of_message_items__DoD4(monkeypatch):
    """DoD-4: every output_text part of every message item, in order, with no separator; reasoning ignored."""
    reply = _reply(
        _reasoning_item("secret reasoning"),
        _message_item("Hello", ", ", extra_parts=[{"type": "refusal", "refusal": "REFUSED"}]),
        _reasoning_item("more reasoning"),
        _message_item("world", "!"),
    )
    client = _client()
    _fake_request(monkeypatch, client, [reply])

    result = await client.chat(_USER_MESSAGES)

    assert result == "Hello, world!"
    assert "reasoning" not in result
    assert "<think>" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(_reply(), id="empty-output"),
        pytest.param({"object": "response", "status": "completed"}, id="missing-output"),
        pytest.param({"object": "response", "status": "completed", "output": "nope"}, id="non-list-output"),
        pytest.param(_reply(_reasoning_item()), id="reasoning-only"),
        pytest.param(_reply(_message_item(None)), id="null-text"),
        pytest.param(_reply({"type": "message", "role": "assistant", "content": []}), id="no-parts"),
    ],
)
async def test_llamaswap_chat_reply_without_text_returns_empty__DoD4(monkeypatch, reply):
    """DoD-4: a reply without message items or text returns empty text."""
    client = _client()
    _fake_request(monkeypatch, client, [reply])

    assert await client.chat(_USER_MESSAGES) == ""


@pytest.mark.asyncio
async def test_llamaswap_chat_failed_reply_raises_with_server_message__DoD5(monkeypatch):
    """DoD-5: status failed raises LLMError carrying the server's error message."""
    reply = _reply(status="failed", error={"code": "server_error", "message": "The engine crashed badly"})
    client = _client()
    _fake_request(monkeypatch, client, [reply])

    with pytest.raises(LLMError) as excinfo:
        await client.chat(_USER_MESSAGES)

    assert "The engine crashed badly" in str(excinfo.value)


@pytest.mark.asyncio
async def test_llamaswap_chat_failed_reply_with_non_mapping_error_carries_it__DoD5(monkeypatch):
    """DoD-5: a failed reply whose error is not a mapping still carries the server's error (as JSON text)."""
    reply = _reply(status="failed", error="engine-out-of-memory")
    client = _client()
    _fake_request(monkeypatch, client, [reply])

    with pytest.raises(LLMError) as excinfo:
        await client.chat(_USER_MESSAGES)

    assert "engine-out-of-memory" in str(excinfo.value)


@pytest.mark.asyncio
async def test_llamaswap_chat_incomplete_reply_raises_with_reason__DoD5(monkeypatch):
    """DoD-5: status incomplete raises LLMError carrying the incomplete reason, even with partial text."""
    reply = _reply(
        _message_item("partial ans"),
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
    )
    client = _client()
    _fake_request(monkeypatch, client, [reply])

    with pytest.raises(LLMError) as excinfo:
        await client.chat(_USER_MESSAGES)

    assert "max_output_tokens" in str(excinfo.value)


# ---------------------------------------------------------------------------
# DoD-6, DoD-7, DoD-8 — settings mapping and the ignored-setting warning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_chat_sends_mapped_settings_under_responses_names__DoD6(monkeypatch):
    """DoD-6: temperature, top_p and user keep their names; max_tokens is sent as max_output_tokens."""
    client = _client(default_options={"temperature": 0.3, "top_p": 0.9, "user": "end-user-42", "max_tokens": 128})

    body = await _chat_body(monkeypatch, client)

    assert body["temperature"] == 0.3
    assert body["top_p"] == 0.9
    assert body["user"] == "end-user-42"
    assert body["max_output_tokens"] == 128
    assert "max_tokens" not in body
    assert set(body) == _BASE_KEYS | {"temperature", "top_p", "user", "max_output_tokens"}


@pytest.mark.asyncio
async def test_llamaswap_chat_context_length_is_sent_as_max_output_tokens__DoD6(monkeypatch):
    """DoD-6: the context_length alias is sent as max_output_tokens."""
    body = await _chat_body(monkeypatch, options={"context_length": 256})

    assert body["max_output_tokens"] == 256
    assert "context_length" not in body
    assert "max_tokens" not in body


@pytest.mark.asyncio
async def test_llamaswap_chat_per_call_none_removes_client_default__DoD6(monkeypatch):
    """DoD-6: a per-call None removes a client default from the request."""
    client = _client(default_options={"temperature": 0.5, "max_tokens": 64})

    body = await _chat_body(monkeypatch, client, options={"temperature": None, "max_tokens": None})

    assert "temperature" not in body
    assert "max_output_tokens" not in body


@pytest.mark.asyncio
async def test_llamaswap_chat_per_call_value_wins_over_default__DoD6(monkeypatch):
    """DoD-6: a per-call value wins over the client default."""
    client = _client(default_options={"temperature": 0.5, "top_p": 0.8})

    body = await _chat_body(monkeypatch, client, options={"temperature": 0.9})

    assert body["temperature"] == 0.9
    assert body["top_p"] == 0.8


@pytest.mark.asyncio
async def test_llamaswap_chat_unknown_key_not_sent_and_no_warning__DoD6(monkeypatch):
    """DoD-6: a key outside the settings table is not sent and raises no warning."""
    client = _client()
    captured = _fake_request(monkeypatch, client, [_text_reply("ok")])

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        await client.chat(_USER_MESSAGES, options={"made_up_setting": 1})

    body = captured[0]["json"]
    assert "made_up_setting" not in body
    assert set(body) == _BASE_KEYS
    assert _ignored(records) == []
    assert not any("made_up_setting" in str(w.message) for w in records)


@pytest.mark.asyncio
@pytest.mark.parametrize("key", list(_NO_COUNTERPART))
async def test_llamaswap_chat_no_counterpart_setting_is_not_sent__DoD7(monkeypatch, key):
    """DoD-7: a no-counterpart setting set by the app is absent from the request; the call returns normally."""
    client = _client(default_options={key: _NO_COUNTERPART[key]})
    captured = _fake_request(monkeypatch, client, [_text_reply("fine")])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LLMSettingIgnoredWarning)
        result = await client.chat(_USER_MESSAGES)

    assert result == "fine"
    body = captured[0]["json"]
    assert key not in body
    assert set(body) == _BASE_KEYS


@pytest.mark.asyncio
@pytest.mark.parametrize("key", list(_NO_COUNTERPART))
async def test_llamaswap_no_counterpart_setting_warns_once_per_client__DoD8(monkeypatch, key):
    """DoD-8: one warning naming the setting on the first call; none for it on a second call of that client."""
    client = _client()
    _fake_request(monkeypatch, client, [_text_reply("ok")])

    with warnings.catch_warnings(record=True) as first:
        warnings.simplefilter("always")
        await client.chat(_USER_MESSAGES, options={key: _NO_COUNTERPART[key]})
    with warnings.catch_warnings(record=True) as second:
        warnings.simplefilter("always")
        await client.chat(_USER_MESSAGES, options={key: _NO_COUNTERPART[key]})

    assert len(_ignored(first)) == 1
    assert key in str(_ignored(first)[0].message)
    assert _ignored(second) == []


@pytest.mark.asyncio
async def test_llamaswap_another_no_counterpart_setting_warns_on_its_own__DoD8(monkeypatch):
    """DoD-8: after one setting has warned, a different setting still emits its own warning."""
    client = _client(default_options={"seed": 7})
    _fake_request(monkeypatch, client, [_text_reply("ok")])

    with warnings.catch_warnings(record=True) as first:
        warnings.simplefilter("always")
        await client.chat(_USER_MESSAGES)
    with warnings.catch_warnings(record=True) as second:
        warnings.simplefilter("always")
        await client.chat(_USER_MESSAGES, options={"logit_bias": {"50256": -100.0}})

    assert len(_ignored(first)) == 1
    assert "seed" in str(_ignored(first)[0].message)
    assert len(_ignored(second)) == 1
    assert "logit_bias" in str(_ignored(second)[0].message)
    assert "seed" not in str(_ignored(second)[0].message)


@pytest.mark.asyncio
async def test_llamaswap_second_client_instance_warns_again__DoD8(monkeypatch):
    """DoD-8: the once-only record is per client instance; a second instance warns again."""
    first_client = _client(default_options={"presence_penalty": 0.5})
    second_client = _client(default_options={"presence_penalty": 0.5})
    _fake_request(monkeypatch, first_client, [_text_reply("ok")])
    _fake_request(monkeypatch, second_client, [_text_reply("ok")])

    with warnings.catch_warnings(record=True) as first:
        warnings.simplefilter("always")
        await first_client.chat(_USER_MESSAGES)
    with warnings.catch_warnings(record=True) as second:
        warnings.simplefilter("always")
        await second_client.chat(_USER_MESSAGES)

    assert len(_ignored(first)) == 1
    assert len(_ignored(second)) == 1
    assert "presence_penalty" in str(_ignored(second)[0].message)


@pytest.mark.asyncio
async def test_llamaswap_setting_cleared_to_none_is_not_warned__DoD8(monkeypatch):
    """DoD-8: a no-counterpart default cleared per call with None is not warned about (DEC-003-6)."""
    client = _client(default_options={"seed": 7})
    _fake_request(monkeypatch, client, [_text_reply("ok")])

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        await client.chat(_USER_MESSAGES, options={"seed": None})

    assert _ignored(records) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["generate", "chat_json"])
async def test_llamaswap_generate_and_chat_json_also_warn__DoD8(monkeypatch, operation):
    """DoD-8: the warning is emitted on every chat-style call, also generate and chat_json (DEC-003-6)."""
    client = _client(default_options={"stop": ["END"]})
    _fake_request(monkeypatch, client, [_text_reply('{"a": 1}')])

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        if operation == "generate":
            await client.generate("Hi")
        else:
            await client.chat_json(_USER_MESSAGES)

    assert len(_ignored(records)) == 1
    assert "stop" in str(_ignored(records)[0].message)


# ---------------------------------------------------------------------------
# DoD-9, DoD-10, DoD-11 — thinking switch and reasoning effort
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("options", "expected"),
    [
        pytest.param({"enable_thinking": True}, {"effort": "medium"}, id="on-default-medium"),
        pytest.param({"enable_thinking": True, "reasoning_effort": "high"}, {"effort": "high"}, id="on-explicit"),
    ],
)
async def test_llamaswap_thinking_on_sends_reasoning_effort_without_summary__DoD9(monkeypatch, options, expected):
    """DoD-9: thinking on sends effort medium, or the explicit effort; the reasoning object has no summary."""
    body = await _chat_body(monkeypatch, options=options)

    assert body["reasoning"] == expected
    assert "summary" not in body["reasoning"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("options", "expected"),
    [
        pytest.param({"enable_thinking": False}, {"effort": "none"}, id="off-default-none"),
        pytest.param({"enable_thinking": False, "reasoning_effort": "low"}, {"effort": "low"}, id="off-explicit"),
    ],
)
async def test_llamaswap_thinking_off_sends_effort_none_or_explicit__DoD10(monkeypatch, options, expected):
    """DoD-10: thinking off sends effort none, or the explicit effort when given; no summary."""
    body = await _chat_body(monkeypatch, options=options)

    assert body["reasoning"] == expected


@pytest.mark.asyncio
async def test_llamaswap_thinking_unset_sends_no_reasoning__DoD11(monkeypatch):
    """DoD-11: thinking unset and no effort sends no reasoning field."""
    body = await _chat_body(monkeypatch)

    assert "reasoning" not in body


@pytest.mark.asyncio
async def test_llamaswap_explicit_effort_alone_is_sent_as_reasoning_effort__DoD11(monkeypatch):
    """DoD-11: an explicit effort without the thinking switch is sent as the reasoning effort, no summary."""
    body = await _chat_body(monkeypatch, options={"reasoning_effort": "low"})

    assert body["reasoning"] == {"effort": "low"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        pytest.param({}, id="unset"),
        pytest.param({"reasoning_effort": "low"}, id="effort-only"),
        pytest.param({"enable_thinking": True}, id="on"),
        pytest.param({"enable_thinking": True, "reasoning_effort": "high"}, id="on-explicit"),
        pytest.param({"enable_thinking": False}, id="off"),
        pytest.param({"enable_thinking": False, "reasoning_effort": "low"}, id="off-explicit"),
    ],
)
async def test_llamaswap_thinking_keys_never_sent_top_level__DoD11(monkeypatch, options):
    """DoD-11: neither the thinking switch nor the effort ever appears as a top-level request key."""
    body = await _chat_body(monkeypatch, options=options)

    assert "enable_thinking" not in body
    assert "reasoning_effort" not in body
    assert set(body) <= _BASE_KEYS | {"reasoning"}


# ---------------------------------------------------------------------------
# DoD-12, DoD-13 — response format and chat_json
# ---------------------------------------------------------------------------

_SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_format", "expected"),
    [
        pytest.param({"type": "json_object"}, {"type": "json_object"}, id="json-object"),
        pytest.param(
            {"type": "json_schema", "json_schema": {"name": "answer", "schema": _SCHEMA, "strict": True}},
            {"type": "json_schema", "name": "answer", "schema": _SCHEMA, "strict": True},
            id="json-schema-flattened",
        ),
        pytest.param({"type": "text"}, {"type": "text"}, id="text"),
        pytest.param(
            {"type": "json_schema", "name": "flat", "schema": _SCHEMA},
            {"type": "json_schema", "name": "flat", "schema": _SCHEMA},
            id="already-flat-unchanged",
        ),
        pytest.param({"kind": "custom", "x": 1}, {"kind": "custom", "x": 1}, id="other-unchanged"),
    ],
)
async def test_llamaswap_chat_response_format_sent_as_text_format__DoD12(monkeypatch, response_format, expected):
    """DoD-12: a response-format setting is sent as text.format, translated per DEC-003-10."""
    body = await _chat_body(monkeypatch, options={"response_format": copy.deepcopy(response_format)})

    assert body["text"] == {"format": expected}
    assert "response_format" not in body


@pytest.mark.asyncio
async def test_llamaswap_chat_json_defaults_to_json_object_text_format__DoD13(monkeypatch):
    """DoD-13: chat_json without a response-format setting sends text.format json_object, stream false."""
    client = _client()
    captured = _fake_request(monkeypatch, client, [_text_reply('{"answer": 42}')])

    result = await client.chat_json(_USER_MESSAGES)

    assert result == '{"answer": 42}'
    assert len(captured) == 1
    assert captured[0]["method"] == "POST"
    assert captured[0]["path"] == "/responses"
    body = captured[0]["json"]
    assert body["text"] == {"format": {"type": "json_object"}}
    assert body["stream"] is False
    assert body["input"] == [{"role": "user", "content": "Hello"}]
    assert set(body) == _BASE_KEYS | {"text"}


@pytest.mark.asyncio
async def test_llamaswap_chat_json_sends_translated_response_format_setting__DoD13(monkeypatch):
    """DoD-13: chat_json with a response-format setting sends that setting, translated (nested schema flattened)."""
    response_format = {"type": "json_schema", "json_schema": {"name": "answer", "schema": _SCHEMA}}
    client = _client(default_options={"response_format": response_format})
    captured = _fake_request(monkeypatch, client, [_text_reply('{"x": 1}')])

    result = await client.chat_json(_USER_MESSAGES)

    assert result == '{"x": 1}'
    assert captured[0]["json"]["text"] == {"format": {"type": "json_schema", "name": "answer", "schema": _SCHEMA}}
    assert captured[0]["json"]["stream"] is False


@pytest.mark.asyncio
async def test_llamaswap_chat_json_with_shape_rejects_invalid_json__DoD13(monkeypatch):
    """DoD-13: with an expected shape, a reply that is not valid JSON raises LLMError."""
    client = _client()
    _fake_request(monkeypatch, client, [_text_reply("this is not json")])

    with pytest.raises(LLMError):
        await client.chat_json(_USER_MESSAGES, response_schema=_SCHEMA)


@pytest.mark.asyncio
async def test_llamaswap_chat_json_with_shape_returns_valid_json_unchanged__DoD13(monkeypatch):
    """DoD-13: with an expected shape, a valid single JSON value is returned unchanged."""
    reply_text = '{ "x" : 1 }'
    client = _client()
    _fake_request(monkeypatch, client, [_text_reply(reply_text)])

    result = await client.chat_json(_USER_MESSAGES, response_schema=_SCHEMA)

    assert result == reply_text


# ---------------------------------------------------------------------------
# DoD-15, DoD-16 — non-2xx failures and the optional credential
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_chat_non_2xx_raises_with_address_and_reply__DoD15():
    """DoD-15: a 404 with a JSON error body raises LLMError with the address and reply, never the credential."""
    error_body = {"error": {"code": 404, "message": "could not find suitable model xyz", "type": "not_found_error"}}
    client = _client(bearer_token=_SECRET)
    session = _FakeSession(404, error_body)
    client.session = session

    with pytest.raises(LLMError) as excinfo:
        await client.chat(_USER_MESSAGES)

    err = excinfo.value
    text = str(err)
    assert f"{_BASE_URL}/responses" in text
    assert "could not find suitable model xyz" in text
    assert _SECRET not in text
    assert _SECRET not in repr(err.args)
    assert _SECRET not in str(getattr(err, "description", ""))
    assert len(session.calls) == 1
    assert session.calls[0]["method"] == "POST"
    assert session.calls[0]["url"] == f"{_BASE_URL}/responses"
    assert not any("/chat/completions" in call["url"] for call in session.calls)


@pytest.mark.asyncio
async def test_llamaswap_request_carries_credential_as_bearer_header__DoD16():
    """DoD-16: built with a credential, the request carries it as the bearer Authorization header."""
    client = _client(bearer_token=_SECRET)
    session = _FakeSession(200, _text_reply("hi"))
    client.session = session

    assert await client.chat(_USER_MESSAGES) == "hi"

    assert len(session.calls) == 1
    assert session.calls[0]["url"] == f"{_BASE_URL}/responses"
    assert _authorization(session.calls[0]) == f"Bearer {_SECRET}"


@pytest.mark.asyncio
async def test_llamaswap_request_without_credential_works_without_header__DoD16():
    """DoD-16: built without a credential, the call works and carries no Authorization header."""
    client = _client()
    session = _FakeSession(200, _text_reply("hi"))
    client.session = session

    assert await client.chat(_USER_MESSAGES) == "hi"

    assert len(session.calls) == 1
    assert _authorization(session.calls[0]) is None


# ---------------------------------------------------------------------------
# DoD-17 — unchanged model listing and embeddings
# ---------------------------------------------------------------------------

_MODELS_REPLY = {
    "object": "list",
    "data": [
        {"id": "model-b", "object": "model", "created": 1, "owned_by": "llamaswap"},
        {"id": "model-a", "object": "model", "created": 2, "owned_by": "llamaswap"},
    ],
}
_EMBED_REPLY = {
    "object": "list",
    "data": [
        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
        {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
    ],
}


def _capture_all(monkeypatch, client: Any, reply: Any) -> list[dict[str, Any]]:
    """Replace ``_request`` with a fake answering ``reply``; records method, path and every keyword argument."""
    calls: list[dict[str, Any]] = []

    async def fake_request(method, path, **kw):
        calls.append({"method": method, "path": path, "kw": copy.deepcopy(kw)})
        return copy.deepcopy(reply)

    monkeypatch.setattr(client, "_request", fake_request)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "reply"),
    [
        pytest.param("list_models", _MODELS_REPLY, id="list_models"),
        pytest.param("get_models", _MODELS_REPLY, id="get_models"),
        pytest.param("embed_batch", _EMBED_REPLY, id="embed_batch"),
    ],
)
async def test_llamaswap_unchanged_operations_match_chat_completions_class__DoD17(monkeypatch, operation, reply):
    """DoD-17: list_models / get_models / embed_batch send the same requests and return the same results as
    the chat-completions class for the same canned reply, and emit no warning."""
    # Both clients get the same no-counterpart default so a warning from these operations would show up.
    defaults = {"seed": 7, "stop": ["END"]}
    responses = LlamaSwapAPIClient(_MODEL, _BASE_URL, default_options=dict(defaults))
    chat_completions = LlamaSwapChatCompletionsClient(_MODEL, _BASE_URL, default_options=dict(defaults))
    responses_calls = _capture_all(monkeypatch, responses, reply)
    chat_completions_calls = _capture_all(monkeypatch, chat_completions, reply)

    async def run(client: Any) -> Any:
        if operation == "embed_batch":
            return await client.embed_batch(["first text", "second text"])
        return await getattr(client, operation)()

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        responses_result = await run(responses)
    expected_result = await run(chat_completions)

    assert responses_result == expected_result
    assert responses_calls == chat_completions_calls
    assert len(responses_calls) >= 1
    assert all("/responses" not in call["path"] for call in responses_calls)
    assert _ignored(records) == []


# ---------------------------------------------------------------------------
# DoD-18 — tracing
# ---------------------------------------------------------------------------


def _records(root: Path) -> list[Path]:
    """Every trace record file written under the trace folder."""
    return sorted(root.rglob("*.log"))


@pytest.mark.asyncio
async def test_llamaswap_traced_chat_returns_text_and_records_responses_request__DoD18(monkeypatch, tmp_path):
    """DoD-18: with tracing on, chat returns the same text; the record shows the /responses request and the text."""
    reply_text = "Traced answer 7f3a"
    # Tracing is read at construction, so it is enabled before the client is built (DEC-003-16).
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(tmp_path))
    client = _client()
    captured = _fake_request(monkeypatch, client, [_text_reply(reply_text)])

    result = await client.chat(_USER_MESSAGES)

    assert result == reply_text
    assert len(captured) == 1
    assert captured[0]["path"] == "/responses"
    records = _records(tmp_path)
    assert len(records) == 1, records
    record = records[0].read_text(encoding="utf-8")
    assert "/responses" in record
    assert "/chat/completions" not in record
    assert reply_text in record
