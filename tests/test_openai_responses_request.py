"""Unit tests for the OpenAI Responses class, non-streamed path (plan 003.responses-chat, step 002).

Covers the Responses request body (DEC-003-11), reply text extraction and failed / incomplete replies
(DEC-003-12), the settings mapping and the ignored-setting warning (DEC-003-9, DEC-003-6, DEC-003-1), the
response-format translation (DEC-003-10), the tool-chat refusal (DEC-003-4), non-2xx errors and the
credential (US-060, DEC-001-6), unchanged model listing and embeddings (US-057.AC-2) and tracing. The server
is faked through the ``_request`` seam or a fake session (DEC-003-16). Every expected value comes from the
step's Definition of done and the cited decisions.
"""

import copy
import json
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest
from multidict import CIMultiDict
from yarl import URL

from llm.exceptions import LLMError, LLMSettingIgnoredWarning
from llm.openai import OpenAIAPIClient
from llm.openai_chat_completions import OpenAIChatCompletionsClient

BASE_URL = "https://api.example.test/v1"
RESPONSES_URL = f"{BASE_URL}/responses"
CREDENTIAL = "sk-test"
MODEL = "gpt-6-luna"

USER_MESSAGES = [{"role": "user", "content": "Hello"}]
REPLY_TEXT = "Bonjour from the Responses route 4417."

# The fixed keys of every chat-style Responses body (DEC-003-11, DEC-arch-7).
FIXED_KEYS = {"model", "input", "stream", "store", "include"}

# No-counterpart settings and an app value for each (DEC-003-9).
NO_COUNTERPART_SETTINGS = {
    "presence_penalty": 0.5,
    "frequency_penalty": 0.4,
    "seed": 7,
    "stop": ["END"],
    "logit_bias": {"50256": -100},
}

ANSWER_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client(**kwargs: Any) -> OpenAIAPIClient:
    """A Responses client with the app-provided test credential."""
    return OpenAIAPIClient(model=MODEL, base_url=BASE_URL, bearer_token=CREDENTIAL, **kwargs)


def _message_item(*parts: dict) -> dict:
    """A Responses ``message`` output item holding the given content parts."""
    return {"type": "message", "role": "assistant", "content": list(parts)}


def _text_part(text: str | None) -> dict:
    """An ``output_text`` content part."""
    return {"type": "output_text", "text": text}


def _reasoning_item(text: str) -> dict:
    """A Responses ``reasoning`` output item with a summary and reasoning content."""
    return {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": text}],
        "content": [{"type": "reasoning_text", "text": text}],
        "encrypted_content": "opaque-blob",
    }


def _reply(*items: dict, status: str = "completed", **extra: Any) -> dict:
    """A non-streamed Responses reply mapping with the given output items."""
    body: dict[str, Any] = {"id": "resp_1", "object": "response", "status": status, "output": list(items)}
    body.update(extra)
    return body


def _text_reply(text: str = REPLY_TEXT) -> dict:
    """A completed reply whose single message item carries ``text``."""
    return _reply(_message_item(_text_part(text)))


def _capture_requests(monkeypatch, client: Any, replies: list[Any]) -> list[dict]:
    """Replace the ``_request`` seam with a fake that records each call and returns canned replies in order.

    Each record holds ``method``, ``path``, the JSON body (``json``) and all keyword arguments, deep-copied so a
    later change by the client cannot alter what was observed as sent.
    """
    records: list[dict] = []
    pending = list(replies)

    async def fake_request(method, path, **kw):
        records.append(
            {"method": method, "path": path, "json": copy.deepcopy(kw.get("json")), "kwargs": copy.deepcopy(kw)}
        )
        return copy.deepcopy(pending.pop(0))

    monkeypatch.setattr(client, "_request", fake_request)
    return records


def _only_body(records: list[dict]) -> dict:
    """The body of the single request sent, checking it went to ``POST /responses``."""
    assert len(records) == 1, records
    assert records[0]["method"] == "POST"
    assert records[0]["path"] == "/responses"
    return records[0]["json"]


async def _chat_body(monkeypatch, defaults: dict | None = None, call_options: dict | None = None) -> dict:
    """Run one non-streamed chat and return the body it sent (warnings recorded and discarded)."""
    client = _client(default_options=defaults)
    records = _capture_requests(monkeypatch, client, [_text_reply()])
    with _capture_warnings():
        await client.chat(list(USER_MESSAGES), options=call_options)
    return _only_body(records)


@contextmanager
def _capture_warnings() -> Iterator[list[warnings.WarningMessage]]:
    """Record every warning with all filters "always", so per-location de-duplication cannot hide or fake
    the once-per-client rule (DEC-003-6, tests/AGENTS.md)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def _ignored(caught: list[warnings.WarningMessage]) -> list[str]:
    """Messages of the library's ignored-setting warnings among ``caught``."""
    return [str(w.message) for w in caught if issubclass(w.category, LLMSettingIgnoredWarning)]


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
    """A minimal ``aiohttp.ClientResponse`` stand-in: status, headers and body reads."""

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
    """Fake ``aiohttp.ClientSession`` assigned to ``client.session`` so the real request code runs.

    Every call is recorded (method, URL, keyword arguments) so tests can see where requests went and which
    headers the server would have received.
    """

    def __init__(self, status: int, body: str) -> None:
        self._status = status
        self._body = body.encode("utf-8")
        self.calls: list[dict] = []
        self.closed = False

    def request(self, method: str, url: Any, **kwargs) -> _FakeRequestContext:
        self.calls.append({"method": method, "url": str(url), "kwargs": kwargs})
        return _FakeRequestContext(_FakeResponse(str(url), self._status, self._body))

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


@pytest.fixture
def no_env_credential(monkeypatch):
    """Remove OpenAI credentials from the environment so a refusal cannot depend on it."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPEN_AI_KEY", raising=False)


# ---------------------------------------------------------------------------
# Request body (DoD-1 .. DoD-3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_chat_without_settings_sends_fixed_body__DoD1(monkeypatch):
    """DoD-1: one ``POST /responses`` whose body has exactly model, input, stream false, store false, include."""
    client = _client()
    records = _capture_requests(monkeypatch, client, [_text_reply()])

    await client.chat(list(USER_MESSAGES))

    body = _only_body(records)
    assert set(body) == FIXED_KEYS
    assert body["model"] == MODEL
    assert body["stream"] is False
    assert body["store"] is False
    assert body["include"] == ["reasoning.encrypted_content"]


@pytest.mark.asyncio
async def test_openai_chat_messages_become_input_items_in_order__DoD2(monkeypatch):
    """DoD-2: each caller message is one input item, in order, without a system instruction: user turns bare
    role/content, assistant turns typed ``message`` (DEC-003-11 as amended 2026-11-08)."""
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
    ]
    client = _client()
    records = _capture_requests(monkeypatch, client, [_text_reply()])

    await client.chat(messages)

    assert _only_body(records)["input"] == [
        {"role": "user", "content": "First question"},
        {"type": "message", "role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
    ]


@pytest.mark.asyncio
async def test_openai_chat_system_instruction_heads_input__DoD2(monkeypatch):
    """DoD-2: a truthy system instruction is a bare system-role item at the head, the assistant turn is typed
    ``message``, and the caller's list is unchanged."""
    messages = [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]
    snapshot = copy.deepcopy(messages)
    client = _client()
    records = _capture_requests(monkeypatch, client, [_text_reply()])

    await client.chat(messages, system="Be terse.")

    assert _only_body(records)["input"] == [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Question"},
        {"type": "message", "role": "assistant", "content": "Answer"},
    ]
    assert messages == snapshot


# A conversation with every caller role, so the per-role item rule of DEC-003-11 (amended 2026-11-08) is
# pinned in one place: system and user bare, assistant typed ``message``.
_MIXED_MESSAGES = [
    {"role": "system", "content": "Caller system turn."},
    {"role": "user", "content": "Ask one."},
    {"role": "assistant", "content": "Reply one."},
    {"role": "user", "content": "Ask two."},
]
_MIXED_ITEMS_WITH_HEAD = [
    {"role": "system", "content": "Head instruction."},
    {"role": "system", "content": "Caller system turn."},
    {"role": "user", "content": "Ask one."},
    {"type": "message", "role": "assistant", "content": "Reply one."},
    {"role": "user", "content": "Ask two."},
]


def _assert_item_key_sets(items: list[dict]) -> None:
    """Assistant items carry exactly type/role/content; every other item exactly role/content (DoD-2)."""
    for item in items:
        if item.get("role") == "assistant":
            assert set(item) == {"type", "role", "content"}, item
            assert item["type"] == "message"
        else:
            assert set(item) == {"role", "content"}, item


@pytest.mark.asyncio
async def test_openai_chat_items_typed_only_for_assistant__DoD2(monkeypatch):
    """DoD-2: on chat, caller system and user turns and the head system item are bare role/content items, the
    assistant turn is a ``message`` item with exactly type, role and content; the caller's list is unchanged."""
    messages = copy.deepcopy(_MIXED_MESSAGES)
    snapshot = copy.deepcopy(messages)
    client = _client()
    records = _capture_requests(monkeypatch, client, [_text_reply()])

    await client.chat(messages, system="Head instruction.")

    items = _only_body(records)["input"]
    assert items == _MIXED_ITEMS_WITH_HEAD
    _assert_item_key_sets(items)
    assert messages == snapshot


@pytest.mark.asyncio
async def test_openai_chat_json_builds_same_items__DoD2(monkeypatch):
    """DoD-2: chat_json builds the same input items as chat (bare system/user and head system item, typed
    assistant with exactly three keys); the caller's list is unchanged."""
    messages = copy.deepcopy(_MIXED_MESSAGES)
    snapshot = copy.deepcopy(messages)
    client = _client()
    records = _capture_requests(monkeypatch, client, [_text_reply('{"a": 1}')])

    await client.chat_json(messages, system="Head instruction.")

    items = _only_body(records)["input"]
    assert items == _MIXED_ITEMS_WITH_HEAD
    _assert_item_key_sets(items)
    assert messages == snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("system", ["", None], ids=["empty", "none"])
async def test_openai_chat_empty_system_adds_no_item__DoD2(monkeypatch, system):
    """DoD-2: an empty (or absent) system instruction adds no item; the caller's list is unchanged."""
    messages = [{"role": "user", "content": "Question"}]
    snapshot = copy.deepcopy(messages)
    client = _client()
    records = _capture_requests(monkeypatch, client, [_text_reply()])

    await client.chat(messages, system=system)

    assert _only_body(records)["input"] == [{"role": "user", "content": "Question"}]
    assert messages == snapshot


@pytest.mark.asyncio
async def test_openai_generate_sends_system_and_prompt_items__DoD3(monkeypatch):
    """DoD-3: generate sends the system item then one user item holding the prompt."""
    client = _client()
    records = _capture_requests(monkeypatch, client, [_text_reply()])

    result = await client.generate("Tell me a fact", system="You are a librarian.")

    body = _only_body(records)
    assert body["input"] == [
        {"role": "system", "content": "You are a librarian."},
        {"role": "user", "content": "Tell me a fact"},
    ]
    assert result == REPLY_TEXT


@pytest.mark.asyncio
async def test_openai_generate_without_system_sends_one_user_item__DoD3(monkeypatch):
    """DoD-3: generate without a system instruction sends exactly one user item holding the prompt."""
    client = _client()
    records = _capture_requests(monkeypatch, client, [_text_reply()])

    await client.generate("Tell me a fact")

    assert _only_body(records)["input"] == [{"role": "user", "content": "Tell me a fact"}]


# ---------------------------------------------------------------------------
# Reply extraction (DoD-4, DoD-5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_chat_concatenates_output_text_of_message_items__DoD4(monkeypatch):
    """DoD-4: every output-text part of every message item, in order, concatenated; reasoning never appears."""
    reply = _reply(
        _reasoning_item("SECRET-REASONING-1"),
        _message_item(_text_part("Alpha "), _text_part("beta ")),
        _reasoning_item("SECRET-REASONING-2"),
        _message_item({"type": "refusal", "refusal": "IGNORED-REFUSAL"}, _text_part("gamma")),
    )
    client = _client()
    _capture_requests(monkeypatch, client, [reply])

    result = await client.chat(list(USER_MESSAGES))

    assert result == "Alpha beta gamma"
    assert "SECRET-REASONING" not in result
    assert "<think>" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        _reply(),
        _reply(_reasoning_item("only reasoning")),
        _reply(_message_item(_text_part(None))),
        _reply(_message_item({"type": "output_text"})),
        _reply(_message_item()),
        {"id": "resp_1", "status": "completed"},
        {"id": "resp_1", "status": "completed", "output": "not a list"},
    ],
    ids=["no-items", "reasoning-only", "null-text", "missing-text", "no-parts", "no-output", "output-not-list"],
)
async def test_openai_chat_reply_without_text_returns_empty__DoD4(monkeypatch, reply):
    """DoD-4: a reply without message items or text returns empty text."""
    client = _client()
    _capture_requests(monkeypatch, client, [reply])

    assert await client.chat(list(USER_MESSAGES)) == ""


@pytest.mark.asyncio
async def test_openai_chat_failed_reply_raises_with_server_message__DoD5(monkeypatch):
    """DoD-5: status failed raises LLMError carrying the server's error message."""
    reply = _reply(
        _message_item(_text_part("partial")),
        status="failed",
        error={"code": "server_error", "message": "Model melted down (ref 5521)."},
    )
    client = _client()
    _capture_requests(monkeypatch, client, [reply])

    with pytest.raises(LLMError) as excinfo:
        await client.chat(list(USER_MESSAGES))

    assert "Model melted down (ref 5521)." in str(excinfo.value)


@pytest.mark.asyncio
async def test_openai_chat_incomplete_reply_raises_with_reason__DoD5(monkeypatch):
    """DoD-5: status incomplete raises LLMError carrying the incomplete reason."""
    reply = _reply(
        _message_item(_text_part("cut o")),
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
    )
    client = _client()
    _capture_requests(monkeypatch, client, [reply])

    with pytest.raises(LLMError) as excinfo:
        await client.chat(list(USER_MESSAGES))

    assert "max_output_tokens" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Settings with a counterpart (DoD-6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_temperature_top_p_user_sent_under_their_names__DoD6(monkeypatch):
    """DoD-6: temperature, top-p and end-user id set by the app are sent under their names."""
    body = await _chat_body(monkeypatch, call_options={"temperature": 0.2, "top_p": 0.9, "user": "end-user-42"})

    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9
    assert body["user"] == "end-user-42"


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["max_tokens", "context_length"])
async def test_openai_output_cap_sent_as_max_output_tokens__DoD6(monkeypatch, key):
    """DoD-6: the output-token cap, set as max_tokens or context_length, is sent as max_output_tokens."""
    body = await _chat_body(monkeypatch, call_options={key: 256})

    assert body["max_output_tokens"] == 256
    assert "max_tokens" not in body
    assert "context_length" not in body


@pytest.mark.asyncio
async def test_openai_call_none_removes_client_default__DoD6(monkeypatch):
    """DoD-6 (US-015.AC-1): a per-call None removes a client default from the request."""
    body = await _chat_body(monkeypatch, defaults={"temperature": 0.3}, call_options={"temperature": None})

    assert "temperature" not in body


@pytest.mark.asyncio
async def test_openai_call_value_wins_over_client_default__DoD6(monkeypatch):
    """DoD-6 (US-014.AC-2): a per-call value wins and the other client defaults are kept."""
    defaults = {"temperature": 0.3, "top_p": 0.5}
    body = await _chat_body(monkeypatch, defaults=defaults, call_options={"temperature": 0.9})

    assert body["temperature"] == 0.9
    assert body["top_p"] == 0.5


@pytest.mark.asyncio
async def test_openai_alias_per_call_wins_over_default__DoD6(monkeypatch):
    """DoD-6 (US-017.AC-2): a per-call context_length wins over a client max_tokens default."""
    body = await _chat_body(monkeypatch, defaults={"max_tokens": 100}, call_options={"context_length": 200})

    assert body["max_output_tokens"] == 200


@pytest.mark.asyncio
async def test_openai_unknown_setting_not_sent_and_not_warned__DoD6(monkeypatch):
    """DoD-6 (US-016.AC-2): an unknown key is not sent and raises no ignored-setting warning."""
    client = _client()
    records = _capture_requests(monkeypatch, client, [_text_reply()])

    with _capture_warnings() as caught:
        result = await client.chat(list(USER_MESSAGES), options={"top_k": 40})

    body = _only_body(records)
    assert "top_k" not in body
    assert set(body) == FIXED_KEYS
    assert _ignored(caught) == []
    assert result == REPLY_TEXT


# ---------------------------------------------------------------------------
# No-counterpart settings and the warning (DoD-7, DoD-8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("key", sorted(NO_COUNTERPART_SETTINGS))
async def test_openai_no_counterpart_setting_not_sent__DoD7(monkeypatch, key):
    """DoD-7: a no-counterpart setting set by the app is absent from the request and the call returns normally."""
    client = _client()
    records = _capture_requests(monkeypatch, client, [_text_reply()])

    with _capture_warnings():
        result = await client.chat(list(USER_MESSAGES), options={key: NO_COUNTERPART_SETTINGS[key]})

    body = _only_body(records)
    assert key not in body
    assert set(body) == FIXED_KEYS
    assert result == REPLY_TEXT


@pytest.mark.asyncio
@pytest.mark.parametrize("key", sorted(NO_COUNTERPART_SETTINGS))
async def test_openai_no_counterpart_setting_warns_once_per_client__DoD8(monkeypatch, key):
    """DoD-8: one warning naming the setting on the first call, none on a second call on the same client."""
    client = _client()
    _capture_requests(monkeypatch, client, [_text_reply(), _text_reply()])
    options = {key: NO_COUNTERPART_SETTINGS[key]}

    with _capture_warnings() as first:
        await client.chat(list(USER_MESSAGES), options=options)
    with _capture_warnings() as second:
        await client.chat(list(USER_MESSAGES), options=options)

    first_messages = _ignored(first)
    assert len(first_messages) == 1
    assert key in first_messages[0]
    assert _ignored(second) == []


@pytest.mark.asyncio
async def test_openai_other_setting_emits_its_own_warning__DoD8(monkeypatch):
    """DoD-8: after one setting was warned about, another setting on the same client emits its own warning."""
    client = _client()
    _capture_requests(monkeypatch, client, [_text_reply(), _text_reply()])

    with _capture_warnings() as first:
        await client.chat(list(USER_MESSAGES), options={"seed": 7})
    with _capture_warnings() as second:
        await client.chat(list(USER_MESSAGES), options={"seed": 7, "presence_penalty": 0.5})

    assert len(_ignored(first)) == 1
    second_messages = _ignored(second)
    assert len(second_messages) == 1
    assert "presence_penalty" in second_messages[0]


@pytest.mark.asyncio
async def test_openai_two_settings_in_one_call_each_warn__DoD8(monkeypatch):
    """DoD-8: two no-counterpart settings in one call each emit their own warning."""
    client = _client()
    _capture_requests(monkeypatch, client, [_text_reply()])

    with _capture_warnings() as caught:
        await client.chat(list(USER_MESSAGES), options={"frequency_penalty": 0.4, "logit_bias": {"1": 2}})

    messages = _ignored(caught)
    assert len(messages) == 2
    assert any("frequency_penalty" in m for m in messages)
    assert any("logit_bias" in m for m in messages)


@pytest.mark.asyncio
async def test_openai_second_client_warns_again__DoD8(monkeypatch):
    """DoD-8: a second client instance warns again about a setting the first already warned about."""
    first_client = _client()
    second_client = _client()
    _capture_requests(monkeypatch, first_client, [_text_reply()])
    _capture_requests(monkeypatch, second_client, [_text_reply()])

    with _capture_warnings() as first:
        await first_client.chat(list(USER_MESSAGES), options={"stop": ["END"]})
    with _capture_warnings() as second:
        await second_client.chat(list(USER_MESSAGES), options={"stop": ["END"]})

    assert len(_ignored(first)) == 1
    second_messages = _ignored(second)
    assert len(second_messages) == 1
    assert "stop" in second_messages[0]


@pytest.mark.asyncio
async def test_openai_client_default_warns_like_call_value__DoD8(monkeypatch):
    """DoD-8: a client default warns on the first call, and the same key per call is then not warned again."""
    client = _client(default_options={"seed": 11})
    records = _capture_requests(monkeypatch, client, [_text_reply(), _text_reply()])

    with _capture_warnings() as first:
        await client.chat(list(USER_MESSAGES))
    with _capture_warnings() as second:
        await client.chat(list(USER_MESSAGES), options={"seed": 12})

    first_messages = _ignored(first)
    assert len(first_messages) == 1
    assert "seed" in first_messages[0]
    assert _ignored(second) == []
    assert all("seed" not in record["json"] for record in records)


@pytest.mark.asyncio
async def test_openai_setting_cleared_to_none_not_warned__DoD8(monkeypatch):
    """DoD-8 (DEC-003-6): a no-counterpart client default cleared to None per call is not warned about."""
    client = _client(default_options={"presence_penalty": 0.5})
    _capture_requests(monkeypatch, client, [_text_reply()])

    with _capture_warnings() as caught:
        await client.chat(list(USER_MESSAGES), options={"presence_penalty": None})

    assert _ignored(caught) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["generate", "chat_json"])
async def test_openai_generate_and_chat_json_warn__DoD8(monkeypatch, operation):
    """DoD-8 (DEC-003-6): generate and chat_json also emit the warning for a no-counterpart setting."""
    client = _client()
    records = _capture_requests(monkeypatch, client, [_text_reply('{"a": 1}')])

    with _capture_warnings() as caught:
        if operation == "generate":
            await client.generate("Prompt", options={"seed": 3})
        else:
            await client.chat_json(list(USER_MESSAGES), options={"seed": 3})

    messages = _ignored(caught)
    assert len(messages) == 1
    assert "seed" in messages[0]
    assert "seed" not in _only_body(records)


# ---------------------------------------------------------------------------
# Thinking and reasoning effort (DoD-9 .. DoD-11)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_thinking_on_without_effort_sends_medium_with_summary__DoD9(monkeypatch):
    """DoD-9: thinking on with no effort sends reasoning effort medium with summary auto."""
    body = await _chat_body(monkeypatch, call_options={"enable_thinking": True})

    assert body["reasoning"] == {"effort": "medium", "summary": "auto"}


@pytest.mark.asyncio
async def test_openai_thinking_on_with_effort_sends_that_effort_with_summary__DoD9(monkeypatch):
    """DoD-9: thinking on with an explicit effort sends that effort with summary auto."""
    body = await _chat_body(monkeypatch, call_options={"enable_thinking": True, "reasoning_effort": "high"})

    assert body["reasoning"] == {"effort": "high", "summary": "auto"}


@pytest.mark.asyncio
async def test_openai_thinking_off_sends_effort_none_without_summary__DoD10(monkeypatch):
    """DoD-10: thinking off sends reasoning effort none without a summary."""
    body = await _chat_body(monkeypatch, call_options={"enable_thinking": False})

    assert body["reasoning"] == {"effort": "none"}


@pytest.mark.asyncio
async def test_openai_thinking_off_with_effort_sends_that_effort__DoD10(monkeypatch):
    """DoD-10: thinking off with an explicit effort sends that effort instead, without a summary."""
    body = await _chat_body(monkeypatch, call_options={"enable_thinking": False, "reasoning_effort": "minimal"})

    assert body["reasoning"] == {"effort": "minimal"}


@pytest.mark.asyncio
async def test_openai_thinking_unset_sends_no_reasoning__DoD11(monkeypatch):
    """DoD-11: thinking unset (and no effort) sends no reasoning field."""
    body = await _chat_body(monkeypatch)

    assert "reasoning" not in body


@pytest.mark.asyncio
async def test_openai_thinking_unset_with_effort_sends_effort_without_summary__DoD11(monkeypatch):
    """DoD-11: thinking unset with an explicit effort sends that effort without a summary."""
    body = await _chat_body(monkeypatch, call_options={"reasoning_effort": "low"})

    assert body["reasoning"] == {"effort": "low"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"enable_thinking": True},
        {"enable_thinking": True, "reasoning_effort": "high"},
        {"enable_thinking": False},
        {"enable_thinking": False, "reasoning_effort": "minimal"},
        {"reasoning_effort": "low"},
        {},
    ],
    ids=["on", "on-effort", "off", "off-effort", "effort-only", "unset"],
)
async def test_openai_thinking_and_effort_never_top_level__DoD11(monkeypatch, options):
    """DoD-11: neither the thinking switch nor the effort appears as a top-level key in any case."""
    body = await _chat_body(monkeypatch, call_options=options)

    assert "enable_thinking" not in body
    assert "reasoning_effort" not in body


# ---------------------------------------------------------------------------
# Response format (DoD-12, DoD-13)
# ---------------------------------------------------------------------------

_NESTED_SCHEMA_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "answer", "schema": ANSWER_SCHEMA, "strict": True},
}
_FLAT_SCHEMA_FORMAT = {"type": "json_schema", "name": "answer", "schema": ANSWER_SCHEMA, "strict": True}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        ({"type": "json_object"}, {"type": "json_object"}),
        (_NESTED_SCHEMA_FORMAT, _FLAT_SCHEMA_FORMAT),
        ({"type": "text"}, {"type": "text"}),
        (_FLAT_SCHEMA_FORMAT, _FLAT_SCHEMA_FORMAT),
        ({"type": "vendor_format", "detail": 1}, {"type": "vendor_format", "detail": 1}),
    ],
    ids=["json-object", "nested-json-schema", "text", "flat-json-schema", "other"],
)
async def test_openai_chat_response_format_sent_as_text_format__DoD12(monkeypatch, setting, expected):
    """DoD-12: chat sends the response format as the text format: JSON-object as is, nested schema flattened,
    any other value unchanged."""
    body = await _chat_body(monkeypatch, call_options={"response_format": copy.deepcopy(setting)})

    assert body["text"] == {"format": expected}
    assert "response_format" not in body


@pytest.mark.asyncio
async def test_openai_chat_json_without_setting_sends_json_object__DoD13(monkeypatch):
    """DoD-13: chat_json without a response-format setting sends the JSON-object text format and stream false,
    and returns the reply text."""
    client = _client()
    records = _capture_requests(monkeypatch, client, [_text_reply('{"a": 1}')])

    result = await client.chat_json(list(USER_MESSAGES))

    body = _only_body(records)
    assert body["text"] == {"format": {"type": "json_object"}}
    assert body["stream"] is False
    assert "response_format" not in body
    assert result == '{"a": 1}'


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["defaults", "call"])
async def test_openai_chat_json_sends_translated_setting__DoD13(monkeypatch, source):
    """DoD-13: chat_json sends the translated response-format setting when one is set."""
    setting = {"response_format": copy.deepcopy(_NESTED_SCHEMA_FORMAT)}
    client = _client(default_options=setting if source == "defaults" else None)
    records = _capture_requests(monkeypatch, client, [_text_reply('{"answer": "x"}')])

    result = await client.chat_json(list(USER_MESSAGES), options=setting if source == "call" else None)

    body = _only_body(records)
    assert body["text"] == {"format": _FLAT_SCHEMA_FORMAT}
    assert body["stream"] is False
    assert result == '{"answer": "x"}'


@pytest.mark.asyncio
async def test_openai_chat_json_with_schema_invalid_json_raises__DoD13(monkeypatch):
    """DoD-13: with an expected shape, a reply that is not valid JSON raises LLMError."""
    client = _client()
    _capture_requests(monkeypatch, client, [_text_reply("this is not json {")])

    with pytest.raises(LLMError):
        await client.chat_json(list(USER_MESSAGES), response_schema=ANSWER_SCHEMA)


@pytest.mark.asyncio
async def test_openai_chat_json_with_schema_valid_json_returned_unchanged__DoD13(monkeypatch):
    """DoD-13: with an expected shape, a valid single JSON value is returned unchanged."""
    reply_text = '{ "answer" :  "forty-two" }'
    client = _client()
    _capture_requests(monkeypatch, client, [_text_reply(reply_text)])

    assert await client.chat_json(list(USER_MESSAGES), response_schema=ANSWER_SCHEMA) == reply_text


# ---------------------------------------------------------------------------
# Tool chat refused (DoD-14)
# ---------------------------------------------------------------------------

_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_record",
            "description": "Retrieve a stored record by its key.",
            "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
        },
    }
]


def _fetch_record(key: str) -> str:
    """The offered tool; never run, since the call is refused before any round."""
    return "alpha-77"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non-streamed", "streamed"])
async def test_openai_chat_with_tools_refused_before_any_request__DoD14(monkeypatch, stream):
    """DoD-14: chat_with_tools raises NotImplementedError naming OpenAIChatCompletionsClient, sends nothing and
    warns about nothing, even with no-counterpart settings set."""
    client = _client(default_options={"seed": 7, "stop": ["END"]})
    records = _capture_requests(monkeypatch, client, [_text_reply()])
    stream_calls: list[Any] = []

    def fake_stream_post(*args, **kwargs):
        stream_calls.append((args, kwargs))
        raise AssertionError("no streamed request may be opened")

    monkeypatch.setattr(client, "_stream_post", fake_stream_post)
    # Any request reaching the transport would be recorded here, whichever path it took.
    session = _FakeSession(200, json.dumps(_text_reply()))
    client.session = session

    with _capture_warnings() as caught, pytest.raises(NotImplementedError) as excinfo:
        await client.chat_with_tools(
            list(USER_MESSAGES),
            tools_definitions=_TOOL_DEFS,
            tools={"fetch_record": _fetch_record},
            options={"presence_penalty": 0.5, "logit_bias": {"1": 2}},
            stream=stream,
        )

    assert "OpenAIChatCompletionsClient" in str(excinfo.value)
    assert records == []
    assert stream_calls == []
    assert session.calls == []
    assert _ignored(caught) == []


# ---------------------------------------------------------------------------
# Non-2xx reply and credential (DoD-15, DoD-16)
# ---------------------------------------------------------------------------

_SERVER_ERROR_MESSAGE = "The server had an error processing your request (ref 91x)."
_SERVER_ERROR_BODY = json.dumps({"error": {"message": _SERVER_ERROR_MESSAGE, "type": "server_error"}})


@pytest.mark.asyncio
async def test_openai_chat_non_2xx_raises_llmerror_without_credential__DoD15():
    """DoD-15: a non-2xx reply raises LLMError with the request address and the server's reply, never the
    credential; exactly one request went to /responses and none to /chat/completions."""
    client = _client()
    session = _FakeSession(500, _SERVER_ERROR_BODY)
    client.session = session

    with pytest.raises(LLMError) as excinfo:
        await client.chat(list(USER_MESSAGES))

    err = excinfo.value
    rendered = str(err)
    assert RESPONSES_URL in rendered
    assert _SERVER_ERROR_MESSAGE in rendered
    assert CREDENTIAL not in f"{rendered} {err.message} {err.description} {err.args!r}"
    assert len(session.calls) == 1
    assert session.calls[0]["url"].endswith("/responses")
    assert all("/chat/completions" not in call["url"] for call in session.calls)


@pytest.mark.parametrize("missing", [None, ""], ids=["none", "empty"])
def test_openai_missing_credential_refused__DoD16(no_env_credential, missing):
    """DoD-16: building without a credential raises ValueError."""
    with pytest.raises(ValueError):
        OpenAIAPIClient(model=MODEL, base_url=BASE_URL, bearer_token=missing)


@pytest.mark.asyncio
async def test_openai_request_carries_app_credential__DoD16(no_env_credential):
    """DoD-16: requests carry the app's credential as the bearer header."""
    client = _client()
    session = _FakeSession(200, json.dumps(_text_reply()))
    client.session = session

    result = await client.chat(list(USER_MESSAGES))

    assert result == REPLY_TEXT
    assert session.calls
    assert all(_authorization(call) == f"Bearer {CREDENTIAL}" for call in session.calls)


# ---------------------------------------------------------------------------
# Unchanged operations (DoD-17)
# ---------------------------------------------------------------------------

_MODELS_REPLY = {
    "object": "list",
    "data": [
        {"id": "gpt-6-luna", "object": "model", "created": 1715367049, "owned_by": "system"},
        {"id": "gpt-4o-mini", "object": "model", "created": 1721172741, "owned_by": "system"},
    ],
}
_EMBED_REPLY = {
    "object": "list",
    "data": [
        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
        {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]},
    ],
    "model": "text-embedding-3-small",
}
# Client defaults mixing a no-counterpart setting, generation settings and an end-user id, so the comparison
# also shows that embeddings warn about nothing (DEC-003-6).
_UNCHANGED_DEFAULTS = {"seed": 7, "presence_penalty": 0.1, "temperature": 0.4, "user": "client-user"}


async def _run_unchanged(monkeypatch, client: Any, operation: str) -> tuple[Any, list[dict], list[str]]:
    """Run one unchanged operation against canned replies; return its result, requests and ignored warnings."""
    if operation in ("list_models", "get_models"):
        records = _capture_requests(monkeypatch, client, [_MODELS_REPLY])
    else:
        records = _capture_requests(monkeypatch, client, [_EMBED_REPLY])
    with _capture_warnings() as caught:
        if operation == "list_models":
            result = await client.list_models()
        elif operation == "get_models":
            result = await client.get_models()
        elif operation == "embed_batch":
            result = await client.embed_batch(["first text", "second text"])
        else:
            result = await client.embed_batch(["first text", "second text"], options={"user": "call-user"})
    requests = [{"method": r["method"], "path": r["path"], "kwargs": r["kwargs"]} for r in records]
    return result, requests, _ignored(caught)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["list_models", "get_models", "embed_batch", "embed_batch_with_user"])
async def test_openai_unchanged_operations_match_chat_completions_client__DoD17(monkeypatch, operation):
    """DoD-17: list_models, get_models and embed_batch send the same requests and return the same results as
    OpenAIChatCompletionsClient for the same canned replies, and emit no warning."""
    responses_client = _client(default_options=dict(_UNCHANGED_DEFAULTS))
    reference_client = OpenAIChatCompletionsClient(
        model=MODEL, base_url=BASE_URL, bearer_token=CREDENTIAL, default_options=dict(_UNCHANGED_DEFAULTS)
    )

    result, requests, ignored = await _run_unchanged(monkeypatch, responses_client, operation)
    expected_result, expected_requests, _ = await _run_unchanged(monkeypatch, reference_client, operation)

    assert result == expected_result
    assert requests == expected_requests
    assert ignored == []


# ---------------------------------------------------------------------------
# Tracing (DoD-18)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_traced_chat_records_request_and_text__DoD18(monkeypatch, tmp_path):
    """DoD-18: with tracing on, chat returns the same text and the record shows the /responses request and
    the returned text."""
    # The tracer reads its switches when the client is built, so they are set first (DEC-003-16).
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(tmp_path))
    client = _client()
    records = _capture_requests(monkeypatch, client, [_text_reply()])

    result = await client.chat(list(USER_MESSAGES))

    assert result == REPLY_TEXT
    _only_body(records)
    logs = sorted(Path(tmp_path).rglob("*.log"))
    assert len(logs) == 1, logs
    record = logs[0].read_text(encoding="utf-8")
    assert "/responses" in record
    assert REPLY_TEXT in record
