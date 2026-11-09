"""Unit tests for the llama-swap Responses tool conversation, non-streamed.

Covers ``LlamaSwapAPIClient.chat_with_tools`` with ``stream=False``:
tool flattening (DEC-004-5, DEC-004-1), the per-round ``POST /responses`` body (DEC-004-10), the response
format on every round, calls / results / assistant text as typed input items (DEC-004-6, DEC-004-4),
reasoning carried verbatim to later rounds (DEC-004-7), round reading (DEC-004-8), 002's tool-loop rules
(DEC-004-11), failures, the ignored-setting warning (DEC-004-12), tracing and the end of the 003 refusal.
The server is faked through ``_request`` (one canned reply per round, DEC-004-14) or, where headers or
non-2xx replies matter, through a fake ``aiohttp`` session. Replies use the llama.cpp probe shapes
(DEC-004-8, DEC-004-9; interfaces.md § Consumed: backend APIs). Each test names its DoD id.
"""

import copy
import importlib
import inspect
import json
import re
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest
from multidict import CIMultiDict
from pydantic import BaseModel
from yarl import URL

from llm.exceptions import LLMError, LLMSettingIgnoredWarning
from llm.llama_swap import LlamaSwapAPIClient
from llm.tools import pydantic_to_openai_tool

_BASE_URL = "http://localhost:8080/v1"
_MODEL = "some-model"

# The body keys every Responses request carries without settings (DEC-003-11); tool rounds add ``tools``.
_BASE_KEYS = {"model", "input", "stream", "store", "include"}
_INCLUDE = ["reasoning.encrypted_content"]

# Sentinel for "leave this key out of the item" in the reply builders.
_MISSING = object()

# Supported keys without a Responses counterpart, with recognisable values (DEC-003-9, DEC-004-12).
_NO_COUNTERPART: dict[str, Any] = {
    "presence_penalty": 0.5,
    "frequency_penalty": 0.4,
    "seed": 7,
    "stop": ["END"],
    "logit_bias": {"50256": -100.0},
}


# ---------------------------------------------------------------------------
# Fixtures, reply builders and fakes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch):
    """Start every test with tracing off so a developer's shell env cannot change behaviour (DEC-000-48)."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _client(**kw: Any) -> LlamaSwapAPIClient:
    """A llama-swap Responses client on the fixed test address, built without a credential unless given."""
    return LlamaSwapAPIClient(model=_MODEL, base_url=_BASE_URL, **kw)


def _user(text: str) -> dict[str, Any]:
    """A caller user turn; also the bare input item 003's builder makes for it (DEC-003-11)."""
    return {"role": "user", "content": text}


def _tool_def(name: str, **function_fields: Any) -> dict[str, Any]:
    """An app tool definition in today's nested chat-completions shape."""
    return {"type": "function", "function": {"name": name, **function_fields}}


def _reasoning(text: str, item_id: str = "rs_1") -> dict[str, Any]:
    """A reasoning output item exactly as llama.cpp sends it: plain reasoning text, empty summary and
    empty encrypted content (probe shape, recorded in DEC-003-2's evidence)."""
    return {
        "type": "reasoning",
        "id": item_id,
        "status": "completed",
        "content": [{"type": "reasoning_text", "text": text}],
        "summary": [],
        "encrypted_content": "",
    }


def _call(name: Any, arguments: Any, call_id: str) -> dict[str, Any]:
    """A function_call output item in the probe shape; ``_MISSING`` drops the name or arguments key."""
    item: dict[str, Any] = {"type": "function_call", "id": f"fc_{call_id}", "call_id": call_id}
    if name is not _MISSING:
        item["name"] = name
    if arguments is not _MISSING:
        item["arguments"] = arguments
    item["status"] = "completed"
    return item


def _message(text: str | None) -> dict[str, Any]:
    """An assistant message output item holding one output_text part (``None`` = null text)."""
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _reply(*items: dict[str, Any], status: str = "completed", **extra: Any) -> dict[str, Any]:
    """A non-streamed Responses reply with the given output items."""
    return {"id": "resp_1", "object": "response", "status": status, "output": list(items), **extra}


def _sent_call(name: str, arguments: str, call_id: str) -> dict[str, Any]:
    """The function_call input item DoD-5 requires: exactly type, call_id, name and arguments."""
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}


def _sent_output(call_id: str, output: str) -> dict[str, Any]:
    """The function_call_output input item DoD-5 requires: exactly type, call_id and the text output."""
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _rounds(monkeypatch, client: Any, replies: list[Any]) -> list[dict[str, Any]]:
    """Replace ``_request`` with a per-round fake (DEC-004-14); returns every captured request.

    Each call gets the next canned reply (a deep copy, so the client cannot alter the test's originals and
    "verbatim" comparisons stay meaningful). Bodies are deep-copied when captured so later mutation by the
    client cannot change what the test inspects. A request beyond the canned replies fails the test.
    """
    pending = [copy.deepcopy(r) for r in replies]
    captured: list[dict[str, Any]] = []

    async def fake_request(method, path, **kw):
        captured.append({"method": method, "path": path, "json": copy.deepcopy(kw.get("json"))})
        if not pending:
            raise AssertionError(f"unexpected extra request #{len(captured)} to {path}")
        return pending.pop(0)

    monkeypatch.setattr(client, "_request", fake_request)
    return captured


class _Tool:
    """A recording tool: remembers the keyword arguments of every run and returns ``result``.

    It declares an explicit signature (``params``, default just ``city`` for the ``lookup`` tool) so that
    it accepts exactly those argument names, like an ordinary function, and is not a catch-all callable.
    """

    def __init__(
        self, *params: str, result: Any = "ok", log: list[str] | None = None, label: str = ""
    ) -> None:
        names = params or ("city",)
        self.__signature__ = inspect.Signature(
            [inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, default=None) for n in names]
        )
        self.result = result
        self.calls: list[dict[str, Any]] = []
        self._log = log
        self._label = label

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._log is not None:
            self._log.append(self._label)
        return self.result


def _ignored(records: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    """Only the library's ignored-setting warnings out of a ``catch_warnings(record=True)`` list."""
    return [w for w in records if issubclass(w.category, LLMSettingIgnoredWarning)]


# Fake aiohttp session (copied per file, harvest § B): lets the real ``_request`` run, so the address,
# headers and the non-2xx mapping are the library's own.


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

    Answers the n-th request with the n-th ``(status, JSON body)`` pair (one per tool round) and records
    method, URL and keyword arguments of each call; a request beyond the list fails the test.
    """

    def __init__(self, answers: list[tuple[int, Any]]) -> None:
        self._answers = [(status, json.dumps(body).encode("utf-8")) for status, body in answers]
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def request(self, method: str, url: Any, **kwargs) -> _FakeRequestContext:
        self.calls.append({"method": method.upper(), "url": str(url), "kwargs": kwargs})
        if len(self.calls) > len(self._answers):
            raise AssertionError(f"unexpected extra request #{len(self.calls)} to {url}")
        status, body = self._answers[len(self.calls) - 1]
        return _FakeRequestContext(_FakeResponse(str(url), status, body))

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


# A standard two-round conversation: the model calls ``lookup`` once, then answers.
_ASK = "What is the code for Paris?"
_LOOKUP_DEFS = [
    _tool_def(
        "lookup",
        description="Look up the code of a city.",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    )
]
_R1 = _reasoning("I should call lookup for Paris.", "rs_round1")
_LOOKUP_ARGS = '{"city": "Paris-7731"}'
_TOOL_ROUND = _reply(_R1, _call("lookup", _LOOKUP_ARGS, "call_aa1"))
_FINAL_TEXT = "The code is QX-48812."
_FINAL_ROUND = _reply(_reasoning("Now I can answer.", "rs_final"), _message(_FINAL_TEXT))


# ---------------------------------------------------------------------------
# DoD-1 — tool flattening
# ---------------------------------------------------------------------------


class _Weather(BaseModel):
    """Typed data model turned into a tool definition by ``pydantic_to_openai_tool`` (US-031.AC-1)."""

    city: str
    days: int = 1


@pytest.mark.asyncio
async def test_llamaswap_tool_definitions_are_sent_flattened__DoD1(monkeypatch):
    """DoD-1: nested definitions are flattened (description / parameters only when present, strict copied
    or false), a flat definition is sent unchanged, order is kept, the app's list is unchanged and a
    ``pydantic_to_openai_tool`` definition is accepted (DEC-004-5, DEC-004-1, US-031.AC-1)."""
    params = {"type": "object", "properties": {"q": {"type": "string"}}}
    flat = {"type": "function", "name": "flat_tool", "description": "Already flat.", "parameters": params}
    weather = pydantic_to_openai_tool("weather", "Weather for a city.", _Weather)
    definitions = [
        _tool_def("full", description="Full tool.", parameters=params, strict=True),
        _tool_def("bare"),
        flat,
        _tool_def("not_strict", parameters=params, strict=False),
        weather,
    ]
    original = copy.deepcopy(definitions)
    tools = {name: _Tool() for name in ("full", "bare", "flat_tool", "not_strict", "weather")}
    client = _client()
    captured = _rounds(monkeypatch, client, [_reply(_message("done"))])

    result = await client.chat_with_tools([_user("Hi")], tools_definitions=definitions, tools=tools)

    assert result == "done"
    assert captured[0]["json"]["tools"] == [
        {"type": "function", "name": "full", "description": "Full tool.", "parameters": params, "strict": True},
        {"type": "function", "name": "bare", "strict": False},
        {"type": "function", "name": "flat_tool", "description": "Already flat.", "parameters": params},
        {"type": "function", "name": "not_strict", "parameters": params, "strict": False},
        {
            "type": "function",
            "name": "weather",
            "description": "Weather for a city.",
            "parameters": weather["function"]["parameters"],
            "strict": False,
        },
    ]
    assert definitions == original


@pytest.mark.asyncio
async def test_llamaswap_flattened_tools_sent_on_every_round__DoD1(monkeypatch):
    """DoD-1: the same flattened tool list is sent on every round of the call (DEC-004-5)."""
    client = _client()
    captured = _rounds(monkeypatch, client, [_TOOL_ROUND, _FINAL_ROUND])

    await client.chat_with_tools([_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool()})

    expected = [
        {
            "type": "function",
            "name": "lookup",
            "description": "Look up the code of a city.",
            "parameters": _LOOKUP_DEFS[0]["function"]["parameters"],
            "strict": False,
        }
    ]
    assert len(captured) == 2
    assert all(c["json"]["tools"] == expected for c in captured)


# ---------------------------------------------------------------------------
# DoD-2 — request per round, settings, credential
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_every_round_is_one_responses_post_with_exact_keys__DoD2(monkeypatch):
    """DoD-2: each round is one POST /responses whose body has exactly the DEC-003-11 keys plus tools,
    stream false; nothing goes to /chat/completions (DEC-004-10, US-056.AC-4)."""
    client = _client()
    captured = _rounds(monkeypatch, client, [_TOOL_ROUND, _FINAL_ROUND])

    await client.chat_with_tools([_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool()})

    assert len(captured) == 2
    for request in captured:
        assert request["method"] == "POST"
        assert request["path"] == "/responses"
        body = request["json"]
        assert set(body) == _BASE_KEYS | {"tools"}
        assert body["model"] == _MODEL
        assert body["stream"] is False
        assert body["store"] is False
        assert body["include"] == _INCLUDE
    assert not any("/chat/completions" in r["path"] for r in captured)


@pytest.mark.asyncio
async def test_llamaswap_tool_rounds_carry_mapped_settings__DoD2(monkeypatch):
    """DoD-2: with settings given, every round holds exactly their DEC-003-9 wire names plus tools."""
    client = _client(default_options={"temperature": 0.3, "max_tokens": 128})
    captured = _rounds(monkeypatch, client, [_TOOL_ROUND, _FINAL_ROUND])

    await client.chat_with_tools([_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool()})

    assert len(captured) == 2
    for request in captured:
        body = request["json"]
        assert set(body) == _BASE_KEYS | {"tools", "temperature", "max_output_tokens"}
        assert body["temperature"] == 0.3
        assert body["max_output_tokens"] == 128


@pytest.mark.asyncio
async def test_llamaswap_thinking_on_sends_effort_without_summary_every_round__DoD2(monkeypatch):
    """DoD-2: thinking on sends the reasoning effort with no summary on every round (DEC-003-2)."""
    client = _client()
    captured = _rounds(monkeypatch, client, [_TOOL_ROUND, _FINAL_ROUND])

    await client.chat_with_tools(
        [_user(_ASK)],
        tools_definitions=_LOOKUP_DEFS,
        tools={"lookup": _Tool()},
        options={"enable_thinking": True},
    )

    assert len(captured) == 2
    for request in captured:
        body = request["json"]
        assert body["reasoning"] == {"effort": "medium"}
        assert "summary" not in body["reasoning"]
        assert "enable_thinking" not in body
        assert set(body) == _BASE_KEYS | {"tools", "reasoning"}


@pytest.mark.asyncio
async def test_llamaswap_credentialless_rounds_carry_no_bearer_header__DoD2():
    """DoD-2: built without a credential, every round goes to /responses and carries no Authorization
    header (US-005.AC-2, DEC-004-10)."""
    client = _client()
    session = _FakeSession([(200, _TOOL_ROUND), (200, _FINAL_ROUND)])
    client.session = session
    tool = _Tool(result="QX")

    result = await client.chat_with_tools([_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": tool})

    assert result == _FINAL_TEXT
    assert len(session.calls) == 2
    for call in session.calls:
        assert call["method"] == "POST"
        assert call["url"] == f"{_BASE_URL}/responses"
        assert _authorization(call) is None
    assert not any("/chat/completions" in call["url"] for call in session.calls)


# ---------------------------------------------------------------------------
# DoD-3 — response format on every round
# ---------------------------------------------------------------------------

_SCHEMA = {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}


@pytest.mark.asyncio
async def test_llamaswap_response_format_parameter_wins_on_every_round__DoD3(monkeypatch):
    """DoD-3: the response_format parameter is sent translated as text.format on every round, tool-call
    rounds included, and wins over a response-format setting (US-037.AC-3, DEC-001-15, DEC-003-10)."""
    client = _client(default_options={"response_format": {"type": "json_object"}})
    captured = _rounds(monkeypatch, client, [_TOOL_ROUND, _FINAL_ROUND])

    await client.chat_with_tools(
        [_user(_ASK)],
        tools_definitions=_LOOKUP_DEFS,
        tools={"lookup": _Tool()},
        response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": _SCHEMA}},
    )

    assert len(captured) == 2
    for request in captured:
        body = request["json"]
        assert body["text"] == {"format": {"type": "json_schema", "name": "answer", "schema": _SCHEMA}}
        assert "response_format" not in body


@pytest.mark.asyncio
async def test_llamaswap_response_format_setting_used_without_parameter__DoD3(monkeypatch):
    """DoD-3: with no parameter, the response-format setting is sent as text.format on every round."""
    client = _client(default_options={"response_format": {"type": "json_object"}})
    captured = _rounds(monkeypatch, client, [_TOOL_ROUND, _FINAL_ROUND])

    await client.chat_with_tools([_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool()})

    assert len(captured) == 2
    for request in captured:
        assert request["json"]["text"] == {"format": {"type": "json_object"}}
        assert "response_format" not in request["json"]


@pytest.mark.asyncio
async def test_llamaswap_no_response_format_sends_no_text_field__DoD3(monkeypatch):
    """DoD-3: with neither parameter nor setting, no round's body has a text field."""
    client = _client()
    captured = _rounds(monkeypatch, client, [_TOOL_ROUND, _FINAL_ROUND])

    await client.chat_with_tools([_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool()})

    assert len(captured) == 2
    assert all("text" not in r["json"] for r in captured)


# ---------------------------------------------------------------------------
# DoD-4 — answer on the first round
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_first_round_without_calls_returns_its_text__DoD4(monkeypatch):
    """DoD-4: a first reply without function-call items returns its text; no tool runs and exactly one
    request is sent (US-030.AC-3, DEC-004-8)."""
    tool = _Tool()
    client = _client()
    captured = _rounds(monkeypatch, client, [_reply(_reasoning("No tool needed."), _message("Direct answer."))])

    result = await client.chat_with_tools([_user("Hi")], tools_definitions=_LOOKUP_DEFS, tools={"lookup": tool})

    assert result == "Direct answer."
    assert len(captured) == 1
    assert tool.calls == []


# ---------------------------------------------------------------------------
# DoD-5, DoD-6, DoD-7, DoD-8 — round input items and reasoning carry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_tool_call_round_builds_next_input_in_order__DoD5(monkeypatch):
    """DoD-5: the tool runs with the decoded arguments, the final text is returned, and the next input is
    the caller's items, the round's reasoning verbatim, the function_call item, then its output item
    (US-030.AC-1, US-058.AC-3, DEC-arch-9, DEC-004-6, DEC-004-7)."""
    tool = _Tool(result="QX-48812")
    client = _client()
    captured = _rounds(monkeypatch, client, [_TOOL_ROUND, _FINAL_ROUND])

    result = await client.chat_with_tools([_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": tool})

    assert result == _FINAL_TEXT
    assert tool.calls == [{"city": "Paris-7731"}]
    assert len(captured) == 2
    assert captured[0]["json"]["input"] == [_user(_ASK)]
    assert captured[1]["json"]["input"] == [
        _user(_ASK),
        _R1,
        _sent_call("lookup", _LOOKUP_ARGS, "call_aa1"),
        _sent_output("call_aa1", "QX-48812"),
    ]
    # Exactly as received: plain reasoning text, empty summary, empty encrypted content.
    carried = captured[1]["json"]["input"][1]
    assert carried["summary"] == []
    assert carried["encrypted_content"] == ""


@pytest.mark.asyncio
async def test_llamaswap_system_item_heads_every_round_input__DoD5(monkeypatch):
    """DoD-5: with a system instruction, the caller's items (system first) lead every round's input."""
    client = _client()
    captured = _rounds(monkeypatch, client, [_TOOL_ROUND, _FINAL_ROUND])

    await client.chat_with_tools(
        [_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool(result="r")}, system="Be brief."
    )

    head = [{"role": "system", "content": "Be brief."}, _user(_ASK)]
    assert captured[0]["json"]["input"] == head
    assert captured[1]["json"]["input"] == head + [
        _R1,
        _sent_call("lookup", _LOOKUP_ARGS, "call_aa1"),
        _sent_output("call_aa1", "r"),
    ]


@pytest.mark.asyncio
async def test_llamaswap_round_text_with_calls_sent_as_typed_assistant_item__DoD6(monkeypatch):
    """DoD-6: a tool round's non-empty text is sent as a typed assistant message between that round's
    reasoning and its calls (DEC-004-4, DEC-004-6)."""
    round1 = _reply(_R1, _message("Let me check that."), _call("lookup", _LOOKUP_ARGS, "call_aa1"))
    client = _client()
    captured = _rounds(monkeypatch, client, [round1, _FINAL_ROUND])

    await client.chat_with_tools(
        [_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool(result="QX")}
    )

    assert captured[1]["json"]["input"] == [
        _user(_ASK),
        _R1,
        {"type": "message", "role": "assistant", "content": "Let me check that."},
        _sent_call("lookup", _LOOKUP_ARGS, "call_aa1"),
        _sent_output("call_aa1", "QX"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text_items",
    [
        pytest.param([_message("")], id="empty-text"),
        pytest.param([_message(None)], id="null-text"),
        pytest.param([], id="no-message-item"),
    ],
)
async def test_llamaswap_round_without_text_sends_no_assistant_item__DoD6(monkeypatch, text_items):
    """DoD-6: a tool round with empty or null text sends no assistant item (DEC-004-6)."""
    round1 = _reply(_R1, *text_items, _call("lookup", _LOOKUP_ARGS, "call_aa1"))
    client = _client()
    captured = _rounds(monkeypatch, client, [round1, _FINAL_ROUND])

    await client.chat_with_tools(
        [_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool(result="QX")}
    )

    assert captured[1]["json"]["input"] == [
        _user(_ASK),
        _R1,
        _sent_call("lookup", _LOOKUP_ARGS, "call_aa1"),
        _sent_output("call_aa1", "QX"),
    ]


@pytest.mark.asyncio
async def test_llamaswap_no_assistant_item_is_ever_untyped__DoD6(monkeypatch):
    """DoD-6: every assistant item sent on any round, from the caller's history or a tool round, carries
    type message (llama.cpp refuses untyped assistant items; DEC-004-4)."""
    messages = [_user("Earlier question"), {"role": "assistant", "content": "Earlier answer"}, _user(_ASK)]
    round1 = _reply(_R1, _message("Checking."), _call("lookup", _LOOKUP_ARGS, "call_aa1"))
    client = _client()
    captured = _rounds(monkeypatch, client, [round1, _FINAL_ROUND])

    await client.chat_with_tools(messages, tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool(result="QX")})

    assistant_items = [
        item for request in captured for item in request["json"]["input"] if item.get("role") == "assistant"
    ]
    assert len(assistant_items) == 3  # caller turn in both rounds + the round-1 text in round 2
    assert all(item.get("type") == "message" for item in assistant_items)


@pytest.mark.asyncio
async def test_llamaswap_two_calls_run_in_order_and_are_sent_calls_then_outputs__DoD7(monkeypatch):
    """DoD-7: two calls behind a reasoning item run in order of appearance; the next input holds both calls
    in order, then both outputs, each with its own call_id (DEC-arch-10, DEC-004-8, US-030.AC-1)."""
    order: list[str] = []
    first = _Tool("x", result="A-result", log=order, label="first")
    second = _Tool("y", result="B-result", log=order, label="second")
    definitions = [_tool_def("first_tool"), _tool_def("second_tool")]
    round1 = _reply(
        _R1,
        _call("first_tool", '{"x": "1"}', "call_one"),
        _call("second_tool", '{"y": "2"}', "call_two"),
    )
    client = _client()
    captured = _rounds(monkeypatch, client, [round1, _FINAL_ROUND])

    result = await client.chat_with_tools(
        [_user(_ASK)], tools_definitions=definitions, tools={"first_tool": first, "second_tool": second}
    )

    assert result == _FINAL_TEXT
    assert order == ["first", "second"]
    assert first.calls == [{"x": "1"}]
    assert second.calls == [{"y": "2"}]
    assert captured[1]["json"]["input"] == [
        _user(_ASK),
        _R1,
        _sent_call("first_tool", '{"x": "1"}', "call_one"),
        _sent_call("second_tool", '{"y": "2"}', "call_two"),
        _sent_output("call_one", "A-result"),
        _sent_output("call_two", "B-result"),
    ]


@pytest.mark.asyncio
async def test_llamaswap_reasoning_of_each_round_resent_on_later_rounds__DoD8(monkeypatch):
    """DoD-8: over three rounds, each round's reasoning is re-sent on every later round ahead of that round's
    calls; the final round's reasoning is sent nowhere (US-058.AC-3, DEC-004-7)."""
    r1 = _reasoning("Round one thinking.", "rs_a")
    r2 = _reasoning("Round two thinking.", "rs_b")
    r3 = _reasoning("Final round thinking.", "rs_c")
    replies = [
        _reply(r1, _call("lookup", '{"city": "A"}', "call_r1")),
        _reply(r2, _call("lookup", '{"city": "B"}', "call_r2")),
        _reply(r3, _message("All done.")),
    ]
    tool = _Tool(result="val")
    client = _client()
    captured = _rounds(monkeypatch, client, replies)

    result = await client.chat_with_tools([_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": tool})

    assert result == "All done."
    assert len(captured) == 3
    assert captured[0]["json"]["input"] == [_user(_ASK)]
    round1_items = [r1, _sent_call("lookup", '{"city": "A"}', "call_r1"), _sent_output("call_r1", "val")]
    round2_items = [r2, _sent_call("lookup", '{"city": "B"}', "call_r2"), _sent_output("call_r2", "val")]
    assert captured[1]["json"]["input"] == [_user(_ASK), *round1_items]
    assert captured[2]["json"]["input"] == [_user(_ASK), *round1_items, *round2_items]
    assert all(r3 not in request["json"]["input"] for request in captured)
    assert "Final round thinking." not in result


# ---------------------------------------------------------------------------
# DoD-9, DoD-10 — caller data untouched, caller history with tool turns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_caller_messages_unchanged_and_answer_clean__DoD9(monkeypatch):
    """DoD-9: after the call the caller's list equals its prior contents; the answer holds no reasoning text
    and no marker (US-058.AC-4, US-058.AC-5, US-028.AC-1)."""
    messages = [_user("Earlier"), {"role": "assistant", "content": "Earlier reply"}, _user(_ASK)]
    original = copy.deepcopy(messages)
    client = _client()
    _rounds(monkeypatch, client, [_TOOL_ROUND, _FINAL_ROUND])

    result = await client.chat_with_tools(
        messages, tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool(result="QX")}
    )

    assert messages == original
    assert result == _FINAL_TEXT
    assert "I should call lookup" not in result
    assert "Now I can answer" not in result
    assert "<think>" not in result
    assert "</think>" not in result


@pytest.mark.asyncio
async def test_llamaswap_caller_history_tool_turns_sent_as_items__DoD10(monkeypatch):
    """DoD-10: a caller history holding an assistant tool-call turn and its tool-result turn is sent as
    function_call and function_call_output items (no reasoning; empty content adds no message; DEC-004-6)."""
    messages = [
        _user("First question"),
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_prev", "type": "function", "function": {"name": "lookup", "arguments": '{"city": "Rome"}'}}
            ],
        },
        {"role": "tool", "content": "RM-1", "tool_call_id": "call_prev"},
        _user("And now?"),
    ]
    original = copy.deepcopy(messages)
    client = _client()
    captured = _rounds(monkeypatch, client, [_reply(_message("Answered."))])

    result = await client.chat_with_tools(
        messages, tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool()}
    )

    assert result == "Answered."
    assert captured[0]["json"]["input"] == [
        _user("First question"),
        _sent_call("lookup", '{"city": "Rome"}', "call_prev"),
        _sent_output("call_prev", "RM-1"),
        _user("And now?"),
    ]
    assert messages == original


@pytest.mark.asyncio
async def test_llamaswap_caller_history_tool_turn_with_text_and_mapping_arguments__DoD10(monkeypatch):
    """DoD-10: a caller assistant tool-call turn with text sends a typed assistant item then the call;
    mapping arguments are sent as JSON text (DEC-004-6)."""
    messages = [
        _user("First question"),
        {
            "role": "assistant",
            "content": "Looking it up.",
            "tool_calls": [
                {"id": "call_prev", "type": "function", "function": {"name": "lookup", "arguments": {"city": "Rome"}}}
            ],
        },
        {"role": "tool", "content": "RM-1", "tool_call_id": "call_prev"},
        _user("And now?"),
    ]
    client = _client()
    captured = _rounds(monkeypatch, client, [_reply(_message("Answered."))])

    await client.chat_with_tools(messages, tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool()})

    items = captured[0]["json"]["input"]
    assert len(items) == 5
    assert items[0] == _user("First question")
    assert items[1] == {"type": "message", "role": "assistant", "content": "Looking it up."}
    assert set(items[2]) == {"type", "call_id", "name", "arguments"}
    assert items[2]["type"] == "function_call"
    assert items[2]["call_id"] == "call_prev"
    assert items[2]["name"] == "lookup"
    assert isinstance(items[2]["arguments"], str)
    assert json.loads(items[2]["arguments"]) == {"city": "Rome"}
    assert items[3] == _sent_output("call_prev", "RM-1")
    assert items[4] == _user("And now?")


# ---------------------------------------------------------------------------
# DoD-11 — running tools and converting results
# ---------------------------------------------------------------------------


async def _one_tool_round(monkeypatch, tool: Any) -> tuple[str, dict[str, Any]]:
    """Run one tool round calling ``lookup`` with ``city=Paris-7731`` then a final round; returns the answer
    and the function_call_output item sent on the second round."""
    client = _client()
    captured = _rounds(monkeypatch, client, [_TOOL_ROUND, _FINAL_ROUND])
    result = await client.chat_with_tools([_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": tool})
    outputs = [i for i in captured[1]["json"]["input"] if i.get("type") == "function_call_output"]
    assert len(outputs) == 1
    return result, outputs[0]


@pytest.mark.asyncio
async def test_llamaswap_awaitable_tool_is_awaited__DoD11(monkeypatch):
    """DoD-11: an async tool is awaited and its result is sent as the output text (US-030.AC-2)."""
    seen: list[str] = []

    async def lookup(city: str) -> str:
        seen.append(city)
        return f"async-{city}"

    result, output = await _one_tool_round(monkeypatch, lookup)

    assert result == _FINAL_TEXT
    assert seen == ["Paris-7731"]
    assert output == _sent_output("call_aa1", "async-Paris-7731")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        pytest.param({"city": "Zürich", "code": 7}, id="mapping"),
        pytest.param(["Zürich", 1, None], id="list"),
    ],
)
async def test_llamaswap_mapping_or_list_result_sent_as_json_text__DoD11(monkeypatch, value):
    """DoD-11: a mapping or list result is sent as JSON text with non-ASCII kept (DEC-002-5)."""
    _, output = await _one_tool_round(monkeypatch, _Tool(result=value))

    content = output["output"]
    assert isinstance(content, str)
    assert json.loads(content) == value
    assert "Zürich" in content
    assert "\\u" not in content


class _Custom:
    """A non-JSON result object with its own text form."""

    def __str__(self) -> str:
        return "custom-text-form"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(42, "42", id="int"),
        pytest.param(None, "None", id="none"),
        pytest.param("plain text", "plain text", id="str"),
        pytest.param(_Custom(), "custom-text-form", id="object"),
    ],
)
async def test_llamaswap_other_result_sent_as_its_text_form__DoD11(monkeypatch, value, expected):
    """DoD-11: any other result is sent as its text form; the output field is always text (DEC-002-5)."""
    _, output = await _one_tool_round(monkeypatch, _Tool(result=value))

    assert isinstance(output["output"], str)
    assert output["output"] == expected


# ---------------------------------------------------------------------------
# DoD-12, DoD-13, DoD-14, DoD-15 — tool-loop failures
# ---------------------------------------------------------------------------
# The tool-call failure tests below run in strict mode: they pass ``tool_errors="raise"`` because the
# loop is tolerant by default (DEC-arch-28; plan 006 step 007 DoD-10 keeps their assertions unchanged).


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "definition",
    [
        pytest.param(_tool_def("orphan"), id="nested"),
        pytest.param({"type": "function", "name": "orphan", "parameters": {"type": "object"}}, id="flat"),
    ],
)
async def test_llamaswap_advertised_tool_without_callable_refused_before_request__DoD12(monkeypatch, definition):
    """DoD-12: an advertised tool without a callable, nested or flat, is refused with ValueError before any
    request (US-032.AC-1, DEC-004-5)."""
    client = _client()
    captured = _rounds(monkeypatch, client, [_FINAL_ROUND])

    with pytest.raises(ValueError):
        await client.chat_with_tools(
            [_user(_ASK)], tools_definitions=[*_LOOKUP_DEFS, definition], tools={"lookup": _Tool()}
        )

    assert captured == []


@pytest.mark.asyncio
async def test_llamaswap_call_to_unadvertised_tool_ends_with_llmerror__DoD13(monkeypatch):
    """DoD-13: in strict mode (``tool_errors="raise"``, DEC-arch-28) a call naming a tool not advertised ends
    with LLMError naming it, even when tools holds a callable under that name; that callable does not run
    (US-035.AC-1, DEC-002-3)."""
    hidden = _Tool()
    client = _client()
    _rounds(monkeypatch, client, [_reply(_R1, _call("hidden_tool", "{}", "call_x"))])

    with pytest.raises(LLMError) as excinfo:
        await client.chat_with_tools(
            [_user(_ASK)],
            tools_definitions=_LOOKUP_DEFS,
            tools={"lookup": _Tool(), "hidden_tool": hidden},
            tool_errors="raise",
        )

    assert "hidden_tool" in str(excinfo.value)
    assert hidden.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("name", [pytest.param("", id="empty"), pytest.param(_MISSING, id="missing")])
async def test_llamaswap_call_with_empty_or_missing_name_ends_with_llmerror__DoD13(monkeypatch, name):
    """DoD-13: in strict mode (``tool_errors="raise"``, DEC-arch-28) a call with an empty or missing name ends
    with LLMError; no tool runs (DEC-002-3)."""
    lookup = _Tool()
    client = _client()
    _rounds(monkeypatch, client, [_reply(_R1, _call(name, "{}", "call_x"))])

    with pytest.raises(LLMError):
        await client.chat_with_tools(
            [_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": lookup}, tool_errors="raise"
        )

    assert lookup.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param("{not json", id="not-json"),
        pytest.param('["Paris"]', id="json-list"),
        pytest.param('"Paris"', id="json-string"),
    ],
)
async def test_llamaswap_unreadable_arguments_end_with_llmerror_naming_tool__DoD14(monkeypatch, arguments):
    """DoD-14: in strict mode (``tool_errors="raise"``, DEC-arch-28) argument text that is not JSON, or JSON
    that is not an object, ends with LLMError naming the tool, and the tool does not run (US-035.AC-2,
    DEC-002-4)."""
    lookup = _Tool()
    client = _client()
    _rounds(monkeypatch, client, [_reply(_R1, _call("lookup", arguments, "call_x"))])

    with pytest.raises(LLMError) as excinfo:
        await client.chat_with_tools(
            [_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": lookup}, tool_errors="raise"
        )

    assert "lookup" in str(excinfo.value)
    assert lookup.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [pytest.param("", id="empty"), pytest.param(_MISSING, id="missing")])
async def test_llamaswap_empty_or_missing_arguments_run_tool_without_arguments__DoD14(monkeypatch, arguments):
    """DoD-14: empty or missing arguments run the tool with no arguments (US-030.AC-1, DEC-002-4)."""
    runs: list[str] = []

    def ping() -> str:
        runs.append("ran")
        return "pong"

    client = _client()
    _rounds(monkeypatch, client, [_reply(_R1, _call("ping", arguments, "call_p")), _FINAL_ROUND])

    result = await client.chat_with_tools(
        [_user("Ping it")], tools_definitions=[_tool_def("ping")], tools={"ping": ping}
    )

    assert runs == ["ran"]
    assert result == _FINAL_TEXT


@pytest.mark.asyncio
async def test_llamaswap_unknown_argument_name_ends_with_valueerror__DoD15(monkeypatch):
    """DoD-15: in strict mode (``tool_errors="raise"``, DEC-arch-28) an argument name the callable lacks ends
    with ValueError and the tool does not run (US-033.AC-1, DEC-004-11)."""
    runs: list[str] = []

    def lookup(city: str) -> str:
        runs.append(city)
        return "never"

    client = _client()
    _rounds(monkeypatch, client, [_reply(_R1, _call("lookup", '{"town": "Paris"}', "call_x"))])

    with pytest.raises(ValueError):
        await client.chat_with_tools(
            [_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": lookup}, tool_errors="raise"
        )

    assert runs == []


@pytest.mark.asyncio
async def test_llamaswap_raising_tool_ends_with_runtimeerror_and_no_further_request__DoD15(monkeypatch):
    """DoD-15: in strict mode (``tool_errors="raise"``, DEC-arch-28) a tool that raises ends with RuntimeError
    naming the tool; no further request is sent (US-033.AC-2, US-033.AC-3, DEC-004-11)."""

    def lookup(city: str) -> str:
        raise ZeroDivisionError("boom inside the tool")

    client = _client()
    captured = _rounds(monkeypatch, client, [_TOOL_ROUND, _FINAL_ROUND])

    with pytest.raises(RuntimeError) as excinfo:
        await client.chat_with_tools(
            [_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": lookup}, tool_errors="raise"
        )

    assert "lookup" in str(excinfo.value)
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_llamaswap_unserialisable_result_ends_with_llmerror_naming_tool__DoD15(monkeypatch):
    """DoD-15: in strict mode (``tool_errors="raise"``, DEC-arch-28) a mapping result that cannot be
    serialised to JSON ends with LLMError naming the tool (US-033.AC-4, US-035.AC-3, DEC-002-5)."""
    client = _client()
    captured = _rounds(monkeypatch, client, [_TOOL_ROUND, _FINAL_ROUND])

    with pytest.raises(LLMError) as excinfo:
        await client.chat_with_tools(
            [_user(_ASK)],
            tools_definitions=_LOOKUP_DEFS,
            tools={"lookup": _Tool(result={"when": object()})},
            tool_errors="raise",
        )

    assert "lookup" in str(excinfo.value)
    assert len(captured) == 1


# ---------------------------------------------------------------------------
# DoD-16 — round limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_round_limit_ends_with_runtimeerror_after_two_requests__DoD16(monkeypatch):
    """DoD-16: with max_loops=2 and a model that keeps calling tools, exactly 2 requests are sent and the
    call ends with RuntimeError mentioning max_loops and 2; no answer is returned (US-034.AC-1..AC-4,
    DEC-002-2)."""
    replies = [
        _reply(_reasoning("again", "rs_1"), _call("lookup", '{"city": "A"}', "call_1")),
        _reply(_reasoning("again", "rs_2"), _call("lookup", '{"city": "B"}', "call_2")),
    ]
    client = _client()
    captured = _rounds(monkeypatch, client, replies)
    result: Any = "unset"

    with pytest.raises(RuntimeError) as excinfo:
        result = await client.chat_with_tools(
            [_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool()}, max_loops=2
        )

    message = str(excinfo.value)
    assert len(captured) == 2
    assert "max_loops" in message
    assert re.search(r"(?<!\d)2(?!\d)", message)
    assert result == "unset"


# ---------------------------------------------------------------------------
# DoD-17 — failed, incomplete and non-2xx replies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_failed_round_ends_with_llmerror_carrying_reason__DoD17(monkeypatch):
    """DoD-17: a round reply with status failed ends with LLMError carrying the server's reason
    (US-060.AC-1, DEC-004-8, DEC-003-12)."""
    failed = _reply(status="failed", error={"code": "server_error", "message": "The engine crashed badly"})
    client = _client()
    _rounds(monkeypatch, client, [failed])

    with pytest.raises(LLMError) as excinfo:
        await client.chat_with_tools([_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool()})

    assert "The engine crashed badly" in str(excinfo.value)


@pytest.mark.asyncio
async def test_llamaswap_incomplete_later_round_ends_with_llmerror_carrying_reason__DoD17(monkeypatch):
    """DoD-17: an incomplete reply on a later round (after a tool round) ends with LLMError carrying the
    server's reason, even with partial text (DEC-004-8, DEC-003-12)."""
    incomplete = _reply(_message("partial"), status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    client = _client()
    captured = _rounds(monkeypatch, client, [_TOOL_ROUND, incomplete])

    with pytest.raises(LLMError) as excinfo:
        await client.chat_with_tools(
            [_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool(result="QX")}
        )

    assert "max_output_tokens" in str(excinfo.value)
    assert len(captured) == 2


@pytest.mark.asyncio
async def test_llamaswap_non_2xx_round_ends_with_llmerror_with_address_and_reply__DoD17():
    """DoD-17: a 404 with a JSON error body ends with LLMError carrying the address and the server's reply
    (US-060.AC-1, DEC-000-11; llama.cpp answers an unknown model this way)."""
    error_body = {"error": {"code": 404, "message": "could not find suitable model xyz", "type": "not_found_error"}}
    client = _client()
    session = _FakeSession([(404, error_body)])
    client.session = session

    with pytest.raises(LLMError) as excinfo:
        await client.chat_with_tools([_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool()})

    text = str(excinfo.value)
    assert f"{_BASE_URL}/responses" in text
    assert "could not find suitable model xyz" in text
    assert len(session.calls) == 1


# ---------------------------------------------------------------------------
# DoD-18 — ignored-setting warning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("key", list(_NO_COUNTERPART))
async def test_llamaswap_no_counterpart_setting_absent_and_warned_once_per_client__DoD18(monkeypatch, key):
    """DoD-18: a no-counterpart setting is absent from every round's body and warned about once per client;
    a second tool call and a later chat emit none for it (US-061.AC-1, US-061.AC-2, DEC-004-12)."""
    client = _client(default_options={key: _NO_COUNTERPART[key]})
    captured = _rounds(
        monkeypatch,
        client,
        [_TOOL_ROUND, _FINAL_ROUND, _TOOL_ROUND, _FINAL_ROUND, _reply(_message("chat reply"))],
    )

    def call_tools():
        return client.chat_with_tools(
            [_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool(result="QX")}
        )

    with warnings.catch_warnings(record=True) as first:
        warnings.simplefilter("always")
        assert await call_tools() == _FINAL_TEXT
    with warnings.catch_warnings(record=True) as second:
        warnings.simplefilter("always")
        assert await call_tools() == _FINAL_TEXT
    with warnings.catch_warnings(record=True) as third:
        warnings.simplefilter("always")
        assert await client.chat([_user("Hi")]) == "chat reply"

    assert len(_ignored(first)) == 1
    assert key in str(_ignored(first)[0].message)
    assert not any(key in str(w.message) for w in _ignored(second))
    assert not any(key in str(w.message) for w in _ignored(third))
    assert len(captured) == 5
    assert all(key not in request["json"] for request in captured[:4])


# ---------------------------------------------------------------------------
# DoD-19 — tracing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_traced_tool_call_returns_same_answer_and_records_rounds__DoD19(monkeypatch, tmp_path):
    """DoD-19: with tracing on, the call returns the same answer; the record shows each round's /responses
    request tagged with its round number (from 1), the tool call with its arguments, the tool result and the
    final answer (US-049.AC-4, US-049.AC-12, US-049.AC-5, US-050.AC-4, DEC-004-10)."""
    untraced = _client()
    _rounds(monkeypatch, untraced, [_TOOL_ROUND, _FINAL_ROUND])
    plain_result = await untraced.chat_with_tools(
        [_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool(result="RESULT-5531")}
    )

    # Tracing is read at construction, so it is enabled before the traced client is built (DEC-003-16).
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(tmp_path))
    traced = _client()
    _rounds(monkeypatch, traced, [_TOOL_ROUND, _FINAL_ROUND])
    traced_result = await traced.chat_with_tools(
        [_user(_ASK)], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool(result="RESULT-5531")}
    )

    assert plain_result == _FINAL_TEXT
    assert traced_result == plain_result
    records = sorted(Path(tmp_path).rglob("*.log"))
    assert len(records) == 1, records
    record = records[0].read_text(encoding="utf-8")
    assert "/responses (loop 1)" in record
    assert "/responses (loop 2)" in record
    assert "/responses (loop 0)" not in record
    assert "/chat/completions" not in record
    assert "lookup" in record
    assert "Paris-7731" in record
    assert "RESULT-5531" in record
    assert _FINAL_TEXT in record


# ---------------------------------------------------------------------------
# DoD-20 — the 003 refusal is gone
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_chat_with_tools_no_longer_refused__DoD20(monkeypatch):
    """DoD-20: non-streamed chat_with_tools runs instead of raising NotImplementedError (DEC-003-4 replaced,
    US-057.AC-1)."""
    client = _client()
    captured = _rounds(monkeypatch, client, [_reply(_message("Works now."))])

    result = await client.chat_with_tools(
        [_user("Hi")], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool()}, stream=False
    )

    assert result == "Works now."
    assert len(captured) == 1


def test_llamaswap_003_refusal_test_removed__DoD20():
    """DoD-20: 003's refusal test is gone from tests/test_llamaswap_responses_request.py (DEC-003-4 replaced)."""
    module = importlib.import_module("tests.test_llamaswap_responses_request")

    assert not hasattr(module, "test_llamaswap_chat_with_tools_refused_naming_chat_completions_class__DoD14")
    assert not any("refused" in name for name in dir(module) if name.startswith("test_"))
