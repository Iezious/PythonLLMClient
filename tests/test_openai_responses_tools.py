"""Unit tests for the OpenAI Responses class, non-streamed tool conversation (plan 004.responses-tools, step 001).

Covers ``OpenAIAPIClient.chat_with_tools`` on the Responses protocol: tool flattening (DEC-004-5, DEC-004-1),
the per-round request body and response format (DEC-004-10), round reading (DEC-004-8), the round input items
and the reasoning carried between rounds (DEC-004-6, DEC-004-7, DEC-004-4), stage-001 tool-loop rules
(DEC-004-11, DEC-002-2..5), round failures and non-2xx replies (DEC-003-12), the ignored-setting warning
(DEC-004-12) and tracing. The server is faked through the ``_request`` seam with one canned reply per round,
or through a fake session (DEC-004-14). Replies follow the probe shapes of the plan's wire vocabulary, with
both llama.cpp-style and OpenAI-style reasoning items. Every expected value comes from the step's Definition
of done and the cited decisions; error texts are checked loosely (tool name, key words), never verbatim.
"""

import asyncio
import copy
import json
import re
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest
from multidict import CIMultiDict
from pydantic import BaseModel
from yarl import URL

from llm.exceptions import LLMError, LLMSettingIgnoredWarning
from llm.openai import OpenAIAPIClient
from llm.tools import pydantic_to_openai_tool

BASE_URL = "https://api.example.test/v1"
RESPONSES_URL = f"{BASE_URL}/responses"
CREDENTIAL = "sk-test"
MODEL = "gpt-6-luna"

# The fixed keys of every Responses body (DEC-003-11), to which tool rounds add ``tools`` (DEC-004-10).
FIXED_KEYS = {"model", "input", "stream", "store", "include"}

USER_TEXT = "What is stored under key k-9?"
USER_MESSAGES = [{"role": "user", "content": USER_TEXT}]
USER_ITEM = {"role": "user", "content": USER_TEXT}

FINAL_TEXT = "The record under k-9 says rec-551."

_LOOKUP_PARAMS = {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}

# Sentinel for "key absent" in item builders, distinct from an explicit None.
_MISSING = object()


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_trace_env(monkeypatch):
    """Tracing is off unless a test switches it on, so no test depends on the developer's environment."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _client(**kwargs: Any) -> OpenAIAPIClient:
    """A Responses client with the app-provided test credential."""
    return OpenAIAPIClient(model=MODEL, base_url=BASE_URL, bearer_token=CREDENTIAL, **kwargs)


def _llama_reasoning(item_id: str, text: str) -> dict:
    """A llama.cpp-style reasoning output item: plain reasoning text, empty summary and encrypted content."""
    return {
        "type": "reasoning",
        "id": item_id,
        "status": "completed",
        "content": [{"type": "reasoning_text", "text": text}],
        "summary": [],
        "encrypted_content": "",
    }


def _openai_reasoning(item_id: str, text: str) -> dict:
    """An OpenAI-style reasoning output item: filled summary and opaque encrypted content."""
    return {
        "type": "reasoning",
        "id": item_id,
        "summary": [{"type": "summary_text", "text": text}],
        "encrypted_content": f"gAAAAB-opaque-{item_id}",
    }


# Both server styles of reasoning item, so carry-over is checked verbatim for each (001.context.md).
_REASONING_STYLES = {"llamacpp": _llama_reasoning, "openai": _openai_reasoning}


def _function_call(call_id: str, name: Any = _MISSING, arguments: Any = _MISSING) -> dict:
    """A ``function_call`` output item in the probe shape; ``name`` / ``arguments`` omitted when ``_MISSING``."""
    item: dict[str, Any] = {"type": "function_call", "id": f"fc_{call_id}", "call_id": call_id, "status": "completed"}
    if name is not _MISSING:
        item["name"] = name
    if arguments is not _MISSING:
        item["arguments"] = arguments
    return item


def _message(text: str | None) -> dict:
    """An assistant ``message`` output item with one output-text part."""
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _reply(*items: dict, status: str = "completed", **extra: Any) -> dict:
    """A non-streamed Responses reply with the given output items."""
    body: dict[str, Any] = {"id": "resp_1", "object": "response", "status": status, "output": list(items)}
    body.update(extra)
    return body


def _answer_reply(text: str = FINAL_TEXT, reasoning_id: str = "rs_final") -> dict:
    """An answer round in the probe shape: a reasoning item, then the message."""
    return _reply(_llama_reasoning(reasoning_id, "FINAL-ROUND-REASONING"), _message(text))


def _call_item(call_id: str, name: str, arguments: str) -> dict:
    """The input item a tool call is sent as on later rounds (DEC-004-6)."""
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}


def _output_item(call_id: str, output: str) -> dict:
    """The input item a tool result is sent as (DEC-004-6)."""
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _nested_def(name: str, *, description: Any = _MISSING, parameters: Any = _MISSING, strict: Any = _MISSING) -> dict:
    """An app tool definition in today's nested shape; optional keys omitted when ``_MISSING``."""
    function: dict[str, Any] = {"name": name}
    if description is not _MISSING:
        function["description"] = description
    if parameters is not _MISSING:
        function["parameters"] = parameters
    if strict is not _MISSING:
        function["strict"] = strict
    return {"type": "function", "function": function}


def _lookup_def() -> dict:
    """The nested definition of the ``lookup`` tool used by most tests."""
    return _nested_def("lookup", description="Fetch a stored record by its key.", parameters=_LOOKUP_PARAMS)


def _lookup_tool(calls: list, result: Any = "rec-551"):
    """A sync ``lookup(key)`` tool recording each call's arguments and returning ``result``."""

    def lookup(key: str) -> Any:
        calls.append({"key": key})
        return result

    return lookup


def _capture_rounds(monkeypatch, client: Any, replies: list[Any]) -> list[dict]:
    """Replace ``_request`` with a fake returning one canned reply per call, in order (DEC-004-14).

    Each record holds the method, path and the JSON body deep-copied at send time, so later changes by the
    client cannot alter what was observed as sent. A request beyond the canned replies fails the test.
    """
    records: list[dict] = []
    pending = list(replies)

    async def fake_request(method, path, **kw):
        records.append({"method": method, "path": path, "json": copy.deepcopy(kw.get("json"))})
        if not pending:
            raise AssertionError("more rounds were requested than canned replies exist")
        return copy.deepcopy(pending.pop(0))

    monkeypatch.setattr(client, "_request", fake_request)
    return records


def _inputs(records: list[dict]) -> list[list[dict]]:
    """The ``input`` item list of every recorded round, in order."""
    return [record["json"]["input"] for record in records]


@contextmanager
def _capture_warnings() -> Iterator[list[warnings.WarningMessage]]:
    """Record every warning with filters "always", so de-duplication cannot hide or fake the once-per-client
    rule (DEC-003-6, tests/AGENTS.md)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def _ignored(caught: list[warnings.WarningMessage]) -> list[str]:
    """Messages of the library's ignored-setting warnings among ``caught``."""
    return [str(w.message) for w in caught if issubclass(w.category, LLMSettingIgnoredWarning)]


async def _lookup_conversation(monkeypatch, client: Any, **kwargs: Any) -> tuple[str, list[dict], list]:
    """Run a two-round conversation: one ``lookup`` call, then the answer. Returns (answer, records, calls)."""
    calls: list = []
    replies = [
        _reply(_llama_reasoning("rs_1", "Need the record."), _function_call("call_1", "lookup", '{"key": "k-9"}')),
        _answer_reply(),
    ]
    records = _capture_rounds(monkeypatch, client, replies)
    result = await client.chat_with_tools(
        list(USER_MESSAGES),
        tools_definitions=[_lookup_def()],
        tools={"lookup": _lookup_tool(calls)},
        **kwargs,
    )
    return result, records, calls


# --- Fake HTTP session, so the real ``_request`` runs for the non-2xx test (copied per file, tests/AGENTS.md).


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
    """Fake ``aiohttp.ClientSession`` assigned to ``client.session`` so the real request code runs; every call
    is recorded (method, URL, keyword arguments)."""

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


# ---------------------------------------------------------------------------
# Tool flattening (DoD-1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_nested_tool_definitions_sent_flattened__DoD1(monkeypatch):
    """DoD-1 (DEC-004-5, DEC-004-1, US-031.AC-1): each nested definition is sent flat with type function and
    its name; description and parameters only when present; strict copied when given, else false; order kept."""
    definitions = [
        _nested_def("lookup", description="Fetch a stored record by its key.", parameters=_LOOKUP_PARAMS),
        _nested_def("ping"),
        _nested_def("strict_tool", parameters=_LOOKUP_PARAMS, strict=True),
        _nested_def("lax_tool", description="Explicitly not strict.", strict=False),
    ]
    tools = {name: (lambda **kw: "x") for name in ("lookup", "ping", "strict_tool", "lax_tool")}
    client = _client()
    records = _capture_rounds(monkeypatch, client, [_answer_reply()])

    await client.chat_with_tools(list(USER_MESSAGES), tools_definitions=definitions, tools=tools)

    assert records[0]["json"]["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "Fetch a stored record by its key.",
            "parameters": _LOOKUP_PARAMS,
            "strict": False,
        },
        {"type": "function", "name": "ping", "strict": False},
        {"type": "function", "name": "strict_tool", "parameters": _LOOKUP_PARAMS, "strict": True},
        {"type": "function", "name": "lax_tool", "description": "Explicitly not strict.", "strict": False},
    ]


@pytest.mark.asyncio
async def test_openai_flat_tool_definition_sent_unchanged_in_order__DoD1(monkeypatch):
    """DoD-1 (DEC-004-5, DEC-004-1): a definition already flat is sent unchanged (no strict added), between
    flattened nested ones, in the app's order."""
    flat_without_strict = {"type": "function", "name": "flat_one", "parameters": _LOOKUP_PARAMS}
    flat_with_strict = {"type": "function", "name": "flat_two", "description": "Flat and strict.", "strict": True}
    definitions = [_nested_def("first"), flat_without_strict, _nested_def("last"), flat_with_strict]
    tools = {name: (lambda **kw: "x") for name in ("first", "flat_one", "last", "flat_two")}
    client = _client()
    records = _capture_rounds(monkeypatch, client, [_answer_reply()])

    await client.chat_with_tools(list(USER_MESSAGES), tools_definitions=definitions, tools=tools)

    assert records[0]["json"]["tools"] == [
        {"type": "function", "name": "first", "strict": False},
        {"type": "function", "name": "flat_one", "parameters": _LOOKUP_PARAMS},
        {"type": "function", "name": "last", "strict": False},
        {"type": "function", "name": "flat_two", "description": "Flat and strict.", "strict": True},
    ]


@pytest.mark.asyncio
async def test_openai_tool_definitions_unchanged_after_call__DoD1(monkeypatch):
    """DoD-1 (DEC-004-5): the app's tool definitions are unchanged after a two-round call."""
    definitions = [_lookup_def(), {"type": "function", "name": "flat_one", "parameters": _LOOKUP_PARAMS}]
    snapshot = copy.deepcopy(definitions)
    calls: list = []
    client = _client()
    _capture_rounds(
        monkeypatch,
        client,
        [_reply(_function_call("call_1", "lookup", '{"key": "k-9"}')), _answer_reply()],
    )

    await client.chat_with_tools(
        list(USER_MESSAGES),
        tools_definitions=definitions,
        tools={"lookup": _lookup_tool(calls), "flat_one": lambda **kw: "x"},
    )

    assert definitions == snapshot


class _WeatherQuery(BaseModel):
    """Typed arguments of a tool built with ``pydantic_to_openai_tool`` (US-031)."""

    city: str
    days: int = 1


@pytest.mark.asyncio
async def test_openai_pydantic_tool_definition_accepted_and_flattened__DoD1(monkeypatch):
    """DoD-1 (US-031.AC-1, DEC-004-1): a definition built by ``pydantic_to_openai_tool`` is accepted and sent
    flat with its name, description, parameters and strict false."""
    definition = pydantic_to_openai_tool("weather", "Forecast for a city.", _WeatherQuery)
    nested = copy.deepcopy(definition)
    client = _client()
    records = _capture_rounds(monkeypatch, client, [_answer_reply()])

    result = await client.chat_with_tools(
        list(USER_MESSAGES), tools_definitions=[definition], tools={"weather": lambda city, days=1: "sunny"}
    )

    assert result == FINAL_TEXT
    assert records[0]["json"]["tools"] == [
        {
            "type": "function",
            "name": "weather",
            "description": "Forecast for a city.",
            "parameters": nested["function"]["parameters"],
            "strict": False,
        }
    ]


# ---------------------------------------------------------------------------
# Request per round (DoD-2, DoD-3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_every_round_posts_fixed_body_plus_tools__DoD2(monkeypatch):
    """DoD-2 (DEC-004-10, DEC-003-11): without settings, every round is one POST /responses whose body holds
    exactly the fixed keys plus tools, stream false; nothing goes to /chat/completions."""
    client = _client()

    _, records, _ = await _lookup_conversation(monkeypatch, client)

    assert len(records) == 2
    expected_tools = [
        {
            "type": "function",
            "name": "lookup",
            "description": "Fetch a stored record by its key.",
            "parameters": _LOOKUP_PARAMS,
            "strict": False,
        }
    ]
    for record in records:
        assert record["method"] == "POST"
        assert record["path"] == "/responses"
        body = record["json"]
        assert set(body) == FIXED_KEYS | {"tools"}
        assert body["model"] == MODEL
        assert body["stream"] is False
        assert body["store"] is False
        assert body["include"] == ["reasoning.encrypted_content"]
        assert body["tools"] == expected_tools
    assert all("/chat/completions" not in record["path"] for record in records)


@pytest.mark.asyncio
async def test_openai_every_round_carries_settings_and_thinking__DoD2(monkeypatch):
    """DoD-2 (DEC-004-10, DEC-003-9, US-056.AC-4): with settings and thinking on, every round's body holds
    exactly the fixed keys, the mapped settings and tools; the thinking-on reasoning setting is on every round."""
    client = _client(default_options={"temperature": 0.2})

    _, records, _ = await _lookup_conversation(
        monkeypatch, client, options={"max_tokens": 300, "enable_thinking": True}
    )

    assert len(records) == 2
    for record in records:
        body = record["json"]
        assert record["path"] == "/responses"
        assert set(body) == FIXED_KEYS | {"tools", "temperature", "max_output_tokens", "reasoning"}
        assert body["temperature"] == 0.2
        assert body["max_output_tokens"] == 300
        assert body["reasoning"] == {"effort": "medium", "summary": "auto"}
        assert body["stream"] is False


_NESTED_SCHEMA_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
        "strict": True,
    },
}
# The Responses text format of _NESTED_SCHEMA_FORMAT per DEC-003-10: the nested schema lifted to the top.
_FLAT_SCHEMA_FORMAT = {
    "type": "json_schema",
    "name": "answer",
    "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
    "strict": True,
}


@pytest.mark.asyncio
async def test_openai_response_format_parameter_sent_on_every_round__DoD3(monkeypatch):
    """DoD-3 (US-037.AC-3, DEC-003-10, DEC-004-10): the response_format parameter is sent as the translated
    text format on every round, the tool-call round included, and never as response_format."""
    client = _client()

    _, records, _ = await _lookup_conversation(
        monkeypatch, client, response_format=copy.deepcopy(_NESTED_SCHEMA_FORMAT)
    )

    assert len(records) == 2
    for record in records:
        assert record["json"]["text"] == {"format": _FLAT_SCHEMA_FORMAT}
        assert "response_format" not in record["json"]


@pytest.mark.asyncio
async def test_openai_response_format_parameter_wins_over_setting__DoD3(monkeypatch):
    """DoD-3 (DEC-001-15): the parameter wins over a response-format setting on every round."""
    client = _client(default_options={"response_format": {"type": "json_object"}})

    _, records, _ = await _lookup_conversation(
        monkeypatch, client, response_format=copy.deepcopy(_NESTED_SCHEMA_FORMAT)
    )

    for record in records:
        assert record["json"]["text"] == {"format": _FLAT_SCHEMA_FORMAT}


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["defaults", "call"])
async def test_openai_response_format_setting_used_without_parameter__DoD3(monkeypatch, source):
    """DoD-3 (DEC-001-15, DEC-003-10): with no parameter, the response-format setting is sent translated on
    every round."""
    setting = {"response_format": copy.deepcopy(_NESTED_SCHEMA_FORMAT)}
    client = _client(default_options=setting if source == "defaults" else None)

    _, records, _ = await _lookup_conversation(
        monkeypatch, client, options=setting if source == "call" else None
    )

    assert len(records) == 2
    for record in records:
        assert record["json"]["text"] == {"format": _FLAT_SCHEMA_FORMAT}


@pytest.mark.asyncio
async def test_openai_no_response_format_sends_no_text_field__DoD3(monkeypatch):
    """DoD-3 (DEC-004-10): with neither parameter nor setting, no round carries a text field."""
    client = _client()

    _, records, _ = await _lookup_conversation(monkeypatch, client)

    assert all("text" not in record["json"] for record in records)


# ---------------------------------------------------------------------------
# Rounds, items and reasoning carry (DoD-4 .. DoD-10)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("style", sorted(_REASONING_STYLES))
async def test_openai_first_round_without_calls_returns_its_text__DoD4(monkeypatch, style):
    """DoD-4 (US-030.AC-3, DEC-004-8): a first reply without function-call items returns that round's text,
    runs no tool and sends exactly one request."""
    calls: list = []
    reply = _reply(_REASONING_STYLES[style]("rs_1", "Nothing to look up."), _message("Direct answer 812."))
    client = _client()
    records = _capture_rounds(monkeypatch, client, [reply])

    result = await client.chat_with_tools(
        list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": _lookup_tool(calls)}
    )

    assert result == "Direct answer 812."
    assert calls == []
    assert len(records) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("style", sorted(_REASONING_STYLES))
async def test_openai_function_call_runs_tool_and_next_input_carries_items__DoD5(monkeypatch, style):
    """DoD-5 (US-030.AC-1, US-058.AC-3, DEC-arch-9, DEC-004-6, DEC-004-7): the tool runs with the decoded
    arguments; the next input is the caller's item, the round's reasoning verbatim, the function call, then
    its output with the result text; the final round's text is returned."""
    reasoning = _REASONING_STYLES[style]("rs_1", "I should fetch k-9.")
    calls: list = []
    replies = [
        _reply(copy.deepcopy(reasoning), _function_call("call_abc", "lookup", '{"key": "k-9"}')),
        _answer_reply(),
    ]
    client = _client()
    records = _capture_rounds(monkeypatch, client, replies)

    result = await client.chat_with_tools(
        list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": _lookup_tool(calls)}
    )

    assert result == FINAL_TEXT
    assert calls == [{"key": "k-9"}]
    first, second = _inputs(records)
    assert first == [USER_ITEM]
    assert second == [
        USER_ITEM,
        reasoning,
        _call_item("call_abc", "lookup", '{"key": "k-9"}'),
        _output_item("call_abc", "rec-551"),
    ]


@pytest.mark.asyncio
async def test_openai_system_instruction_heads_every_round_input__DoD5(monkeypatch):
    """DoD-5 (DEC-004-6, DEC-003-11): the system instruction is the bare system item at the head of every
    round's input, ahead of the caller's items."""
    client = _client()

    _, records, _ = await _lookup_conversation(monkeypatch, client, system="Answer briefly.")

    for items in _inputs(records):
        assert items[0] == {"role": "system", "content": "Answer briefly."}
        assert items[1] == USER_ITEM


@pytest.mark.asyncio
async def test_openai_tool_round_text_sent_as_typed_assistant_item__DoD6(monkeypatch):
    """DoD-6 (DEC-004-4, DEC-004-6, DEC-004-8): a tool round's non-empty text is sent as a typed assistant
    message between that round's reasoning and its function calls."""
    reasoning = _openai_reasoning("rs_1", "Check the store.")
    calls: list = []
    replies = [
        _reply(
            copy.deepcopy(reasoning),
            _message("Let me check that record."),
            _function_call("call_1", "lookup", '{"key": "k-9"}'),
        ),
        _answer_reply(),
    ]
    client = _client()
    records = _capture_rounds(monkeypatch, client, replies)

    await client.chat_with_tools(
        list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": _lookup_tool(calls)}
    )

    assert _inputs(records)[1] == [
        USER_ITEM,
        reasoning,
        {"type": "message", "role": "assistant", "content": "Let me check that record."},
        _call_item("call_1", "lookup", '{"key": "k-9"}'),
        _output_item("call_1", "rec-551"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", None], ids=["empty", "null"])
async def test_openai_tool_round_without_text_sends_no_assistant_item__DoD6(monkeypatch, text):
    """DoD-6 (DEC-004-6, DEC-004-8): a tool round with empty or null text sends no assistant item."""
    reasoning = _llama_reasoning("rs_1", "Check the store.")
    calls: list = []
    replies = [
        _reply(copy.deepcopy(reasoning), _message(text), _function_call("call_1", "lookup", '{"key": "k-9"}')),
        _answer_reply(),
    ]
    client = _client()
    records = _capture_rounds(monkeypatch, client, replies)

    await client.chat_with_tools(
        list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": _lookup_tool(calls)}
    )

    second = _inputs(records)[1]
    assert second == [
        USER_ITEM,
        reasoning,
        _call_item("call_1", "lookup", '{"key": "k-9"}'),
        _output_item("call_1", "rec-551"),
    ]
    assert not any(item.get("role") == "assistant" for item in second)


@pytest.mark.asyncio
async def test_openai_two_calls_run_in_order_sequentially__DoD7(monkeypatch):
    """DoD-7 (DEC-arch-10, DEC-004-8, US-030.AC-1): two calls behind a reasoning item run in their order of
    appearance, one after the other; the next input holds both calls in order, then both outputs, each with
    its call's call_id."""
    events: list[str] = []

    # Async tools that yield to the event loop mid-run: if the calls overlapped, the start/end events would
    # interleave, so the event order proves sequential execution.
    async def first_tool(city: str) -> str:
        events.append(f"first:start:{city}")
        await asyncio.sleep(0)
        events.append("first:end")
        return "first-result-17"

    async def second_tool(city: str) -> str:
        events.append(f"second:start:{city}")
        await asyncio.sleep(0)
        events.append("second:end")
        return "second-result-29"

    reasoning = _openai_reasoning("rs_1", "Two lookups needed.")
    replies = [
        _reply(
            copy.deepcopy(reasoning),
            _function_call("call_a", "first_tool", '{"city": "Oslo"}'),
            _function_call("call_b", "second_tool", '{"city": "Lima"}'),
        ),
        _answer_reply(),
    ]
    client = _client()
    records = _capture_rounds(monkeypatch, client, replies)

    result = await client.chat_with_tools(
        list(USER_MESSAGES),
        tools_definitions=[_nested_def("first_tool"), _nested_def("second_tool")],
        tools={"first_tool": first_tool, "second_tool": second_tool},
    )

    assert result == FINAL_TEXT
    assert events == ["first:start:Oslo", "first:end", "second:start:Lima", "second:end"]
    assert _inputs(records)[1] == [
        USER_ITEM,
        reasoning,
        _call_item("call_a", "first_tool", '{"city": "Oslo"}'),
        _call_item("call_b", "second_tool", '{"city": "Lima"}'),
        _output_item("call_a", "first-result-17"),
        _output_item("call_b", "second-result-29"),
    ]


@pytest.mark.asyncio
async def test_openai_reasoning_carried_over_three_rounds__DoD8(monkeypatch):
    """DoD-8 (US-058.AC-3, DEC-004-7): round 1's reasoning is re-sent on rounds 2 and 3 ahead of round 1's
    call, round 2's on round 3 ahead of round 2's call; an item with empty encrypted content and plain
    reasoning text is re-sent unchanged; the final round's reasoning is sent nowhere."""
    first_reasoning = _llama_reasoning("rs_round1", "Plain reasoning text, round one.")
    second_reasoning = _openai_reasoning("rs_round2", "Summary of round two.")
    final_reasoning = _openai_reasoning("rs_round3", "UNSENT-FINAL-REASONING")
    calls: list = []
    replies = [
        _reply(copy.deepcopy(first_reasoning), _function_call("call_1", "lookup", '{"key": "k-1"}')),
        _reply(copy.deepcopy(second_reasoning), _function_call("call_2", "lookup", '{"key": "k-2"}')),
        _reply(copy.deepcopy(final_reasoning), _message(FINAL_TEXT)),
    ]
    client = _client()
    records = _capture_rounds(monkeypatch, client, replies)

    result = await client.chat_with_tools(
        list(USER_MESSAGES),
        tools_definitions=[_lookup_def()],
        tools={"lookup": _lookup_tool(calls, result="rec-x")},
    )

    assert result == FINAL_TEXT
    assert calls == [{"key": "k-1"}, {"key": "k-2"}]
    first, second, third = _inputs(records)
    assert first == [USER_ITEM]
    round_one_items = [
        first_reasoning,
        _call_item("call_1", "lookup", '{"key": "k-1"}'),
        _output_item("call_1", "rec-x"),
    ]
    assert second == [USER_ITEM, *round_one_items]
    assert third == [
        USER_ITEM,
        *round_one_items,
        second_reasoning,
        _call_item("call_2", "lookup", '{"key": "k-2"}'),
        _output_item("call_2", "rec-x"),
    ]
    sent = json.dumps([record["json"] for record in records])
    assert "rs_round3" not in sent
    assert "UNSENT-FINAL-REASONING" not in sent


@pytest.mark.asyncio
async def test_openai_caller_messages_unchanged_and_answer_clean__DoD9(monkeypatch):
    """DoD-9 (US-058.AC-4, US-058.AC-5, US-028.AC-1): after a tool call the caller's message list equals its
    prior contents, and the answer holds no reasoning text and no think markers."""
    messages = [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": USER_TEXT},
    ]
    snapshot = copy.deepcopy(messages)
    calls: list = []
    replies = [
        _reply(_openai_reasoning("rs_1", "HIDDEN-REASONING-ONE"), _function_call("call_1", "lookup", '{"key": "k"}')),
        _reply(_llama_reasoning("rs_2", "HIDDEN-REASONING-TWO"), _message(FINAL_TEXT)),
    ]
    client = _client()
    _capture_rounds(monkeypatch, client, replies)

    result = await client.chat_with_tools(
        messages, tools_definitions=[_lookup_def()], tools={"lookup": _lookup_tool(calls)}, system="Be brief."
    )

    assert messages == snapshot
    assert result == FINAL_TEXT
    assert "HIDDEN-REASONING" not in result
    assert "<think>" not in result
    assert "</think>" not in result


@pytest.mark.asyncio
async def test_openai_caller_tool_history_sent_as_call_items_without_reasoning__DoD10(monkeypatch):
    """DoD-10 (DEC-004-6): a caller history holding an assistant tool-call turn and its tool-result turn is
    sent as function-call and function-call-output items, without any reasoning item."""
    messages = [
        {"role": "user", "content": "Look up k-old."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_old", "type": "function", "function": {"name": "lookup", "arguments": '{"key": "k-old"}'}}
            ],
        },
        {"role": "tool", "content": "old-record-3", "tool_call_id": "call_old"},
        {"role": "user", "content": "And what does it mean?"},
    ]
    snapshot = copy.deepcopy(messages)
    calls: list = []
    client = _client()
    records = _capture_rounds(monkeypatch, client, [_answer_reply()])

    result = await client.chat_with_tools(
        messages, tools_definitions=[_lookup_def()], tools={"lookup": _lookup_tool(calls)}
    )

    assert result == FINAL_TEXT
    items = _inputs(records)[0]
    assert items == [
        {"role": "user", "content": "Look up k-old."},
        _call_item("call_old", "lookup", '{"key": "k-old"}'),
        _output_item("call_old", "old-record-3"),
        {"role": "user", "content": "And what does it mean?"},
    ]
    assert not any(item.get("type") == "reasoning" for item in items)
    assert messages == snapshot


@pytest.mark.asyncio
async def test_openai_caller_tool_history_with_text_and_mapping_arguments__DoD10(monkeypatch):
    """DoD-10 (DEC-004-6): a caller tool-call turn with text sends a typed assistant item before its call;
    mapping arguments are sent as JSON text; no reasoning item is sent."""
    messages = [
        {"role": "user", "content": "Look up k-old."},
        {
            "role": "assistant",
            "content": "Checking the store.",
            "tool_calls": [
                {"id": "call_old", "type": "function", "function": {"name": "lookup", "arguments": {"key": "k-old"}}}
            ],
        },
        {"role": "tool", "content": "old-record-3", "tool_call_id": "call_old"},
    ]
    calls: list = []
    client = _client()
    records = _capture_rounds(monkeypatch, client, [_answer_reply()])

    await client.chat_with_tools(messages, tools_definitions=[_lookup_def()], tools={"lookup": _lookup_tool(calls)})

    items = _inputs(records)[0]
    assert len(items) == 4
    assert items[0] == {"role": "user", "content": "Look up k-old."}
    assert items[1] == {"type": "message", "role": "assistant", "content": "Checking the store."}
    call_item = items[2]
    assert set(call_item) == {"type", "call_id", "name", "arguments"}
    assert call_item["type"] == "function_call"
    assert call_item["call_id"] == "call_old"
    assert call_item["name"] == "lookup"
    assert isinstance(call_item["arguments"], str)
    assert json.loads(call_item["arguments"]) == {"key": "k-old"}
    assert items[3] == _output_item("call_old", "old-record-3")
    assert not any(item.get("type") == "reasoning" for item in items)


# ---------------------------------------------------------------------------
# Tool runs and results (DoD-11)
# ---------------------------------------------------------------------------


async def _single_result_output(monkeypatch, tool) -> str:
    """Run one call of ``tool`` (named ``probe``, no arguments) and return the output text sent next round."""
    client = _client()
    records = _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", "probe", "{}")), _answer_reply()]
    )
    await client.chat_with_tools(list(USER_MESSAGES), tools_definitions=[_nested_def("probe")], tools={"probe": tool})
    outputs = [item for item in _inputs(records)[1] if item.get("type") == "function_call_output"]
    assert len(outputs) == 1
    assert outputs[0]["call_id"] == "call_1"
    return outputs[0]["output"]


@pytest.mark.asyncio
async def test_openai_awaitable_tool_awaited_and_result_sent__DoD11(monkeypatch):
    """DoD-11 (US-030.AC-2): an awaitable tool is awaited and its result sent as the output text."""

    async def probe() -> str:
        await asyncio.sleep(0)
        return "async-result-404"

    assert await _single_result_output(monkeypatch, probe) == "async-result-404"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [{"city": "Zürich", "count": 3}, ["Zürich", 1, {"nested": True}]],
    ids=["mapping", "list"],
)
async def test_openai_mapping_or_list_result_sent_as_json_text__DoD11(monkeypatch, value):
    """DoD-11 (DEC-002-5, US-030.AC-1): a mapping or list result is sent as JSON text with non-ASCII kept."""
    output = await _single_result_output(monkeypatch, lambda: copy.deepcopy(value))

    assert isinstance(output, str)
    assert json.loads(output) == value
    assert "Zürich" in output
    assert "\\u" not in output


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [42, 3.5, True, None], ids=["int", "float", "bool", "none"])
async def test_openai_other_result_sent_as_text_form__DoD11(monkeypatch, value):
    """DoD-11 (DEC-002-5): any other result is sent as its text form; the output is always text."""
    output = await _single_result_output(monkeypatch, lambda: value)

    assert isinstance(output, str)
    assert output == str(value)


# ---------------------------------------------------------------------------
# Loop failures (DoD-12 .. DoD-16)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "definition",
    [_nested_def("missing_tool"), {"type": "function", "name": "missing_tool", "parameters": _LOOKUP_PARAMS}],
    ids=["nested", "flat"],
)
async def test_openai_advertised_tool_without_callable_refused_before_request__DoD12(monkeypatch, definition):
    """DoD-12 (US-032.AC-1, DEC-004-5): an advertised tool without a callable, nested or flat, is refused with
    ValueError before any request."""
    calls: list = []
    client = _client()
    records = _capture_rounds(monkeypatch, client, [_answer_reply()])

    with pytest.raises(ValueError):
        await client.chat_with_tools(
            list(USER_MESSAGES),
            tools_definitions=[_lookup_def(), definition],
            tools={"lookup": _lookup_tool(calls)},
        )

    assert records == []
    assert calls == []


@pytest.mark.asyncio
async def test_openai_call_to_unadvertised_tool_raises_and_does_not_run__DoD13(monkeypatch):
    """DoD-13 (US-035.AC-1, DEC-002-3): a call naming a tool not advertised, even with a callable under that
    name in tools, ends with LLMError naming the tool, and that callable does not run."""
    secret_runs: list = []
    lookup_calls: list = []
    client = _client()
    records = _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", "secret_tool", "{}")), _answer_reply()]
    )

    with pytest.raises(LLMError) as excinfo:
        await client.chat_with_tools(
            list(USER_MESSAGES),
            tools_definitions=[_lookup_def()],
            tools={"lookup": _lookup_tool(lookup_calls), "secret_tool": lambda: secret_runs.append(1)},
        )

    assert "secret_tool" in str(excinfo.value)
    assert secret_runs == []
    assert lookup_calls == []
    assert len(records) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["", _MISSING], ids=["empty", "missing"])
async def test_openai_call_with_empty_or_missing_name_raises__DoD13(monkeypatch, name):
    """DoD-13 (US-035.AC-1, DEC-002-3): a call with an empty or missing name ends with LLMError saying the
    name was empty, and no tool runs."""
    lookup_calls: list = []
    client = _client()
    records = _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", name, '{"key": "k-9"}')), _answer_reply()]
    )

    with pytest.raises(LLMError) as excinfo:
        await client.chat_with_tools(
            list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": _lookup_tool(lookup_calls)}
        )

    assert "empty" in str(excinfo.value).lower()
    assert lookup_calls == []
    assert len(records) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    ["not json {", "[1, 2]", "3", '"k-9"', "null"],
    ids=["invalid-json", "list", "number", "string", "null"],
)
async def test_openai_unreadable_arguments_raise_and_tool_does_not_run__DoD14(monkeypatch, arguments):
    """DoD-14 (US-035.AC-2, DEC-002-4): argument text that is not JSON, or JSON that is not an object, ends
    with LLMError naming the tool, and the tool does not run."""
    lookup_calls: list = []
    client = _client()
    records = _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", "lookup", arguments)), _answer_reply()]
    )

    with pytest.raises(LLMError) as excinfo:
        await client.chat_with_tools(
            list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": _lookup_tool(lookup_calls)}
        )

    assert "lookup" in str(excinfo.value)
    assert lookup_calls == []
    assert len(records) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["", _MISSING], ids=["empty", "missing"])
async def test_openai_empty_or_missing_arguments_run_tool_without_arguments__DoD14(monkeypatch, arguments):
    """DoD-14 (US-030.AC-1, DEC-002-4, DEC-004-6): empty or missing arguments run the tool with no
    arguments; the call is re-sent with empty argument text."""
    runs: list = []

    def ping() -> str:
        runs.append("ran")
        return "pong-61"

    client = _client()
    records = _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", "ping", arguments)), _answer_reply()]
    )

    result = await client.chat_with_tools(
        list(USER_MESSAGES), tools_definitions=[_nested_def("ping")], tools={"ping": ping}
    )

    assert result == FINAL_TEXT
    assert runs == ["ran"]
    assert _inputs(records)[1] == [USER_ITEM, _call_item("call_1", "ping", ""), _output_item("call_1", "pong-61")]


@pytest.mark.asyncio
async def test_openai_unknown_argument_name_raises_value_error__DoD15(monkeypatch):
    """DoD-15 (US-033.AC-1, DEC-004-11): an argument name the callable lacks ends with ValueError and the
    tool does not run."""
    lookup_calls: list = []
    client = _client()
    records = _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", "lookup", '{"other": 1}')), _answer_reply()]
    )

    with pytest.raises(ValueError):
        await client.chat_with_tools(
            list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": _lookup_tool(lookup_calls)}
        )

    assert lookup_calls == []
    assert len(records) == 1


@pytest.mark.asyncio
async def test_openai_raising_tool_ends_with_runtime_error__DoD15(monkeypatch):
    """DoD-15 (US-033.AC-2, US-033.AC-3, DEC-004-11): a tool that raises ends with RuntimeError naming the
    tool, and no further request is sent."""

    def lookup(key: str) -> str:
        raise OSError("disk unavailable 77")

    client = _client()
    records = _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", "lookup", '{"key": "k-9"}')), _answer_reply()]
    )

    with pytest.raises(RuntimeError) as excinfo:
        await client.chat_with_tools(list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": lookup})

    assert "lookup" in str(excinfo.value)
    assert len(records) == 1


@pytest.mark.asyncio
async def test_openai_unserialisable_mapping_result_raises__DoD15(monkeypatch):
    """DoD-15 (US-033.AC-4, US-035.AC-3, DEC-002-5): a mapping result that cannot be serialised to JSON ends
    with LLMError naming the tool, and no further request is sent."""
    client = _client()
    records = _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", "lookup", '{"key": "k-9"}')), _answer_reply()]
    )

    with pytest.raises(LLMError) as excinfo:
        await client.chat_with_tools(
            list(USER_MESSAGES),
            tools_definitions=[_lookup_def()],
            tools={"lookup": lambda key: {"values": {1, 2, 3}}},
        )

    assert "lookup" in str(excinfo.value)
    assert len(records) == 1


@pytest.mark.asyncio
async def test_openai_round_limit_reached_raises_runtime_error__DoD16(monkeypatch):
    """DoD-16 (US-034.AC-1..AC-4, DEC-002-2): with max_loops 2 and a model that keeps asking for tools,
    exactly 2 requests are sent, then RuntimeError mentioning max_loops and 2; no answer is returned."""
    calls: list = []
    replies = [
        _reply(_llama_reasoning(f"rs_{n}", "Again."), _function_call(f"call_{n}", "lookup", '{"key": "k"}'))
        for n in range(1, 4)
    ]
    client = _client()
    records = _capture_rounds(monkeypatch, client, replies)

    with pytest.raises(RuntimeError) as excinfo:
        await client.chat_with_tools(
            list(USER_MESSAGES),
            tools_definitions=[_lookup_def()],
            tools={"lookup": _lookup_tool(calls)},
            max_loops=2,
        )

    message = str(excinfo.value)
    assert len(records) == 2
    assert "max_loops" in message
    assert re.search(r"(?<!\d)2(?!\d)", message)


# ---------------------------------------------------------------------------
# Failed rounds and non-2xx (DoD-17)
# ---------------------------------------------------------------------------

_FAILED_REPLY = _reply(status="failed", error={"code": "server_error", "message": "Model melted down (ref 5521)."})
_INCOMPLETE_REPLY = _reply(
    _message("cut o"), status="incomplete", incomplete_details={"reason": "max_output_tokens"}
)
_ROUND_FAILURES = {
    "failed": (_FAILED_REPLY, "Model melted down (ref 5521)."),
    "incomplete": (_INCOMPLETE_REPLY, "max_output_tokens"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(_ROUND_FAILURES))
@pytest.mark.parametrize("round_no", [1, 2], ids=["first-round", "second-round"])
async def test_openai_failed_or_incomplete_round_raises_with_reason__DoD17(monkeypatch, kind, round_no):
    """DoD-17 (US-060.AC-1, DEC-004-8, DEC-003-12): a round reply with status failed or incomplete ends with
    LLMError carrying the server's reason, on the first round and on a second round."""
    failure, reason = _ROUND_FAILURES[kind]
    calls: list = []
    tool_round = _reply(_llama_reasoning("rs_1", "Lookup."), _function_call("call_1", "lookup", '{"key": "k-9"}'))
    replies = [failure] if round_no == 1 else [tool_round, failure]
    client = _client()
    records = _capture_rounds(monkeypatch, client, replies)

    with pytest.raises(LLMError) as excinfo:
        await client.chat_with_tools(
            list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": _lookup_tool(calls)}
        )

    assert reason in str(excinfo.value)
    assert len(records) == round_no


_SERVER_ERROR_MESSAGE = "The server had an error processing your request (ref 91x)."
_SERVER_ERROR_BODY = json.dumps({"error": {"message": _SERVER_ERROR_MESSAGE, "type": "server_error"}})


@pytest.mark.asyncio
async def test_openai_tool_round_non_2xx_raises_without_credential__DoD17():
    """DoD-17 (US-060.AC-1, US-060.AC-4): a non-2xx reply ends with LLMError carrying the request address and
    the server's reply, never the credential; the request went to /responses only."""
    calls: list = []
    client = _client()
    session = _FakeSession(500, _SERVER_ERROR_BODY)
    client.session = session

    with pytest.raises(LLMError) as excinfo:
        await client.chat_with_tools(
            list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": _lookup_tool(calls)}
        )

    err = excinfo.value
    rendered = str(err)
    assert RESPONSES_URL in rendered
    assert _SERVER_ERROR_MESSAGE in rendered
    assert CREDENTIAL not in f"{rendered} {err.message} {err.description} {err.args!r}"
    assert len(session.calls) == 1
    assert session.calls[0]["url"].endswith("/responses")
    assert calls == []


# ---------------------------------------------------------------------------
# Ignored-setting warning (DoD-18)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_no_counterpart_setting_absent_and_warned_once__DoD18(monkeypatch):
    """DoD-18 (US-061.AC-1, DEC-004-12): a no-counterpart setting is absent from every round's body and emits
    one warning for the client, naming it."""
    client = _client(default_options={"seed": 7})

    with _capture_warnings() as caught:
        _, records, _ = await _lookup_conversation(monkeypatch, client)

    assert len(records) == 2
    assert all("seed" not in record["json"] for record in records)
    messages = _ignored(caught)
    assert len(messages) == 1
    assert "seed" in messages[0]


@pytest.mark.asyncio
async def test_openai_second_tool_call_and_later_chat_emit_no_warning__DoD18(monkeypatch):
    """DoD-18 (US-061.AC-2, DEC-004-12): a second tool call on the same client, and a chat after a tool call,
    emit no warning for that setting, and neither sends it."""
    client = _client()
    options = {"presence_penalty": 0.5}
    calls: list = []
    one_round = [_reply(_function_call("call_1", "lookup", '{"key": "k-9"}')), _answer_reply()]
    records = _capture_rounds(monkeypatch, client, [*one_round, *one_round, _answer_reply()])
    kwargs = {"tools_definitions": [_lookup_def()], "tools": {"lookup": _lookup_tool(calls)}, "options": options}

    with _capture_warnings() as first:
        await client.chat_with_tools(list(USER_MESSAGES), **kwargs)
    with _capture_warnings() as second:
        await client.chat_with_tools(list(USER_MESSAGES), **kwargs)
    with _capture_warnings() as third:
        await client.chat(list(USER_MESSAGES), options=options)

    assert len(_ignored(first)) == 1
    assert _ignored(second) == []
    assert _ignored(third) == []
    assert len(records) == 5
    assert all("presence_penalty" not in record["json"] for record in records)


@pytest.mark.asyncio
async def test_openai_tool_call_after_chat_emits_no_warning__DoD18(monkeypatch):
    """DoD-18 (DEC-004-12): the per-client record is shared, so a tool call after a chat that already warned
    about a setting emits no warning for it."""
    client = _client()
    calls: list = []
    records = _capture_rounds(monkeypatch, client, [_answer_reply(), _answer_reply()])

    with _capture_warnings() as first:
        await client.chat(list(USER_MESSAGES), options={"stop": ["END"]})
    with _capture_warnings() as second:
        await client.chat_with_tools(
            list(USER_MESSAGES),
            tools_definitions=[_lookup_def()],
            tools={"lookup": _lookup_tool(calls)},
            options={"stop": ["END"]},
        )

    assert len(_ignored(first)) == 1
    assert _ignored(second) == []
    assert all("stop" not in record["json"] for record in records)


# ---------------------------------------------------------------------------
# Tracing (DoD-19)
# ---------------------------------------------------------------------------


async def _trace_conversation(monkeypatch, client: Any) -> tuple[str, list]:
    """A two-round conversation with distinctive argument and result values, for the trace record."""
    calls: list = []
    replies = [
        _reply(_openai_reasoning("rs_1", "Fetch it."), _function_call("call_1", "lookup", '{"key": "trace-key-31"}')),
        _reply(_llama_reasoning("rs_2", "Answering."), _message("Traced final answer 6602.")),
    ]
    _capture_rounds(monkeypatch, client, replies)
    result = await client.chat_with_tools(
        list(USER_MESSAGES),
        tools_definitions=[_lookup_def()],
        tools={"lookup": _lookup_tool(calls, result="trace-result-88")},
    )
    return result, calls


@pytest.mark.asyncio
async def test_openai_traced_tool_call_records_rounds_calls_and_answer__DoD19(monkeypatch, tmp_path):
    """DoD-19 (US-049.AC-4, US-049.AC-5, US-049.AC-12, US-050.AC-4, DEC-004-10): with tracing on the call
    returns the same answer, and the record shows each round's /responses request tagged with its round
    number from 1, the tool call with its arguments, the tool result and the final answer."""
    untraced_result, _ = await _trace_conversation(monkeypatch, _client())

    # The tracer reads its switches when the client is built, so they are set first.
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(tmp_path))
    traced_result, calls = await _trace_conversation(monkeypatch, _client())

    assert traced_result == untraced_result == "Traced final answer 6602."
    assert calls == [{"key": "trace-key-31"}]
    logs = sorted(Path(tmp_path).rglob("*.log"))
    assert len(logs) == 1, logs
    record = logs[0].read_text(encoding="utf-8")
    assert "/responses (loop 1)" in record
    assert "/responses (loop 2)" in record
    assert "/responses (loop 0)" not in record
    assert "lookup" in record
    assert "trace-key-31" in record
    assert "trace-result-88" in record
    assert "Traced final answer 6602." in record


# ---------------------------------------------------------------------------
# Refusal replaced (DoD-20)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_chat_with_tools_no_longer_refused__DoD20(monkeypatch):
    """DoD-20 (DEC-003-4 replaced, US-057.AC-1): chat_with_tools no longer raises NotImplementedError; a
    non-streamed tool conversation returns the final answer."""
    client = _client()

    try:
        result, records, calls = await _lookup_conversation(monkeypatch, client, stream=False)
    except NotImplementedError as exc:  # pragma: no cover - the failure this test exists to catch
        pytest.fail(f"chat_with_tools is still refused: {exc}")

    assert result == FINAL_TEXT
    assert calls == [{"key": "k-9"}]
    assert len(records) == 2
