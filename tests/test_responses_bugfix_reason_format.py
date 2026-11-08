"""Bug-fix reproduction tests for the Responses classes (feature 003.responses-chat, bug fix 2026-11-08).

Bug A (DEC-003-10): ``LlamaSwapAPIClient`` must send a
nested JSON-schema response format as ``text.format`` with ``type`` ``json_schema``; a ``type`` key inside the
nested mapping must not replace it (same rule as ``OpenAIAPIClient``).

Bug B (DEC-003-21, amending DEC-003-12 / DEC-003-13): the ``LLMError`` raised for an incomplete reply or event
has as description the incomplete reason only when ``incomplete_details`` is a mapping whose ``reason`` is a
non-empty string; otherwise the description is the ``incomplete_details`` value as JSON text. Checked on both
Responses classes, streamed and non-streamed, chat and tool rounds. No app output limit is set anywhere here,
so no reply counts as a cut (DEC-003-18).
"""

import copy
import json
from typing import Any

import pytest

from llm.exceptions import LLMError
from llm.llama_swap import LlamaSwapAPIClient
from llm.llama_swap import _responses_text_format as _llamaswap_text_format
from llm.openai import OpenAIAPIClient
from llm.openai import _responses_text_format as _openai_text_format

_OPENAI_BASE_URL = "https://api.example.test/v1"
_LLAMASWAP_BASE_URL = "http://localhost:8080/v1"
_MODEL = "some-model"
_CREDENTIAL = "sk-test"
_MESSAGES = [{"role": "user", "content": "Hello there"}]

_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
# A nested JSON-schema format whose inner mapping carries its own "type" key (Bug A).
_NESTED_WITH_TYPE = {"type": "json_schema", "json_schema": {"name": "a", "schema": _SCHEMA, "type": "x"}}
# DEC-003-10: nested keys lifted beside "type", which stays "json_schema"; the nested "type" is dropped.
_EXPECTED_FORMAT = {"type": "json_schema", "name": "a", "schema": _SCHEMA}

_LOOKUP_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up the code of a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]

# Incomplete-details cases (DEC-003-21), all ASCII so json.dumps text is unambiguous.
# ("text", reason): description must equal the reason; ("json", None): description must be JSON text that
# decodes back to the details value.
_REASON_CASES = [
    pytest.param({"reason": "content_filter"}, "text", "content_filter", id="string-reason"),
    pytest.param({"reason": ""}, "json", None, id="empty-reason"),
    pytest.param({"reason": 5}, "json", None, id="int-reason"),
    pytest.param({"reason": None}, "json", None, id="null-reason"),
    pytest.param({"detail": "trunc"}, "json", None, id="no-reason"),
]


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch):
    """Clear the trace switches so a developer's environment cannot turn tracing on (DEC-000-48)."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _openai() -> OpenAIAPIClient:
    """OpenAI Responses client built with an explicit test credential (tests/AGENTS.md)."""
    return OpenAIAPIClient(model=_MODEL, base_url=_OPENAI_BASE_URL, bearer_token=_CREDENTIAL)


def _llamaswap(**kw: Any) -> LlamaSwapAPIClient:
    """llama-swap Responses client on the fixed test address; defaults from ``kw``."""
    return LlamaSwapAPIClient(model=_MODEL, base_url=_LLAMASWAP_BASE_URL, **kw)


def _lookup(city: str | None = None) -> str:
    """A tool the app offers; never expected to run in these tests."""
    return "ok"


def _fake_request(monkeypatch, client: Any, reply: Any) -> list[dict[str, Any]]:
    """Replace the ``_request`` seam (DEC-003-16) with a fake answering ``reply`` (deep copy) to every call;
    returns the captured requests with the JSON body deep-copied against later mutation."""
    captured: list[dict[str, Any]] = []

    async def fake_request(method, path, **kw):
        captured.append({"method": method, "path": path, "json": copy.deepcopy(kw.get("json"))})
        return copy.deepcopy(reply)

    monkeypatch.setattr(client, "_request", fake_request)
    return captured


def _install_sse(monkeypatch, client: Any, payloads: list[str]) -> None:
    """Replace the ``_iter_sse_data`` seam with a fake yielding ``payloads`` as SSE data strings."""

    async def fake_iter_sse_data(_response):
        for payload in payloads:
            yield payload

    monkeypatch.setattr(client, "_iter_sse_data", fake_iter_sse_data)


def _completed_text_reply(text: str) -> dict[str, Any]:
    """A completed non-streamed reply with one message item holding ``text``."""
    return {
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
    }


def _incomplete_reply(details: Any) -> dict[str, Any]:
    """A non-streamed reply with status incomplete and the given details."""
    return {"status": "incomplete", "incomplete_details": copy.deepcopy(details), "output": []}


def _incomplete_event(details: Any) -> str:
    """The SSE payload of a ``response.incomplete`` event carrying ``details``."""
    return json.dumps(
        {"type": "response.incomplete", "response": {"status": "incomplete", "incomplete_details": details}}
    )


def _assert_description(err: LLMError, details: Any, mode: str, reason: str | None) -> None:
    """DEC-003-21: a non-empty string reason is the description; otherwise the details as JSON text."""
    if mode == "text":
        assert err.description == reason
    else:
        assert isinstance(err.description, str)
        assert json.loads(err.description) == details


# ---------------------------------------------------------------------------
# Bug A — nested "type" never overrides json_schema (DEC-003-10)
# ---------------------------------------------------------------------------


def test_llamaswap_text_format_keeps_json_schema_type_over_nested_type__s004_DoD12():
    """The llama-swap translation of a nested JSON-schema format keeps type json_schema and
    drops the nested type key (DEC-003-10)."""
    assert _llamaswap_text_format(copy.deepcopy(_NESTED_WITH_TYPE)) == _EXPECTED_FORMAT


def test_openai_text_format_keeps_json_schema_type_over_nested_type__s002_DoD12():
    """Parity guard: the OpenAI translation gives the same format for the same input."""
    assert _openai_text_format(copy.deepcopy(_NESTED_WITH_TYPE)) == _EXPECTED_FORMAT


@pytest.mark.asyncio
async def test_llamaswap_chat_sends_json_schema_type_despite_nested_type__s004_DoD12(monkeypatch):
    """Non-streamed chat sends text.format with type json_schema, nested type dropped."""
    client = _llamaswap()
    captured = _fake_request(monkeypatch, client, _completed_text_reply("ok"))

    await client.chat(list(_MESSAGES), options={"response_format": copy.deepcopy(_NESTED_WITH_TYPE)})

    assert len(captured) == 1
    assert captured[0]["json"]["text"] == {"format": _EXPECTED_FORMAT}


@pytest.mark.asyncio
async def test_llamaswap_chat_json_sends_json_schema_type_despite_nested_type__s004_DoD13(monkeypatch):
    """Chat_json sends the translated setting, so type stays json_schema."""
    client = _llamaswap(default_options={"response_format": copy.deepcopy(_NESTED_WITH_TYPE)})
    captured = _fake_request(monkeypatch, client, _completed_text_reply('{"answer": "x"}'))

    await client.chat_json(list(_MESSAGES))

    assert len(captured) == 1
    assert captured[0]["json"]["text"] == {"format": _EXPECTED_FORMAT}


@pytest.mark.asyncio
async def test_llamaswap_chat_with_tools_response_format_keeps_json_schema_type__f004_s003_DoD3(monkeypatch):
    """The chat_with_tools response_format parameter is sent translated as
    text.format, type json_schema, nested type dropped (DEC-003-10)."""
    client = _llamaswap()
    captured = _fake_request(monkeypatch, client, _completed_text_reply('{"answer": "x"}'))

    await client.chat_with_tools(
        list(_MESSAGES),
        tools_definitions=copy.deepcopy(_LOOKUP_DEFS),
        tools={"lookup": _lookup},
        response_format=copy.deepcopy(_NESTED_WITH_TYPE),
    )

    assert len(captured) == 1
    assert captured[0]["json"]["text"] == {"format": _EXPECTED_FORMAT}


# ---------------------------------------------------------------------------
# Bug B — incomplete description: non-empty string reason, else details as JSON (DEC-003-21)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("details", "mode", "reason"), _REASON_CASES)
async def test_openai_chat_incomplete_reply_description__s002_DoD5(monkeypatch, details, mode, reason):
    """Non-streamed OpenAI chat on an incomplete reply raises LLMError whose description follows
    DEC-003-21."""
    client = _openai()
    _fake_request(monkeypatch, client, _incomplete_reply(details))

    with pytest.raises(LLMError) as excinfo:
        await client.chat(list(_MESSAGES))

    _assert_description(excinfo.value, details, mode, reason)


@pytest.mark.asyncio
@pytest.mark.parametrize(("details", "mode", "reason"), _REASON_CASES)
async def test_openai_stream_incomplete_event_description__s003_DoD5(monkeypatch, details, mode, reason):
    """The OpenAI chat stream parser raises LLMError on response.incomplete with the DEC-003-21
    description."""
    client = _openai()
    _install_sse(monkeypatch, client, [_incomplete_event(details)])

    with pytest.raises(LLMError) as excinfo:
        await client._read_responses_stream(None, None)

    _assert_description(excinfo.value, details, mode, reason)


@pytest.mark.asyncio
@pytest.mark.parametrize(("details", "mode", "reason"), _REASON_CASES)
async def test_openai_tool_stream_incomplete_event_description__f004_s002_DoD8(monkeypatch, details, mode, reason):
    """The OpenAI tool-round stream parser raises LLMError on response.incomplete
    with the DEC-003-21 description."""
    client = _openai()
    _install_sse(monkeypatch, client, [_incomplete_event(details)])

    with pytest.raises(LLMError) as excinfo:
        await client._read_responses_tool_stream(None, None)

    _assert_description(excinfo.value, details, mode, reason)


@pytest.mark.asyncio
@pytest.mark.parametrize(("details", "mode", "reason"), _REASON_CASES)
async def test_openai_tool_round_incomplete_reply_description__f004_s001_DoD17(monkeypatch, details, mode, reason):
    """A non-streamed OpenAI tool round with status incomplete raises LLMError
    with the DEC-003-21 description."""
    client = _openai()
    _fake_request(monkeypatch, client, _incomplete_reply(details))

    with pytest.raises(LLMError) as excinfo:
        await client.chat_with_tools(
            list(_MESSAGES), tools_definitions=copy.deepcopy(_LOOKUP_DEFS), tools={"lookup": _lookup}
        )

    _assert_description(excinfo.value, details, mode, reason)


@pytest.mark.asyncio
@pytest.mark.parametrize(("details", "mode", "reason"), _REASON_CASES)
async def test_llamaswap_chat_incomplete_reply_description__s004_DoD5(monkeypatch, details, mode, reason):
    """Non-streamed llama-swap chat on an incomplete reply raises LLMError whose description
    follows DEC-003-21."""
    client = _llamaswap()
    _fake_request(monkeypatch, client, _incomplete_reply(details))

    with pytest.raises(LLMError) as excinfo:
        await client.chat(list(_MESSAGES))

    _assert_description(excinfo.value, details, mode, reason)


@pytest.mark.asyncio
@pytest.mark.parametrize(("details", "mode", "reason"), _REASON_CASES)
async def test_llamaswap_stream_incomplete_event_description__s005_DoD6(monkeypatch, details, mode, reason):
    """Regression guard: the llama-swap chat stream parser raises LLMError on
    response.incomplete with the DEC-003-21 description."""
    client = _llamaswap()
    _install_sse(monkeypatch, client, [_incomplete_event(details)])

    with pytest.raises(LLMError) as excinfo:
        await client._stream_responses_response(None, None)

    _assert_description(excinfo.value, details, mode, reason)


@pytest.mark.asyncio
@pytest.mark.parametrize(("details", "mode", "reason"), _REASON_CASES)
async def test_llamaswap_tool_stream_incomplete_event_description__f004_s004_DoD9(monkeypatch, details, mode, reason):
    """Regression guard: the llama-swap tool-round stream parser raises LLMError on
    response.incomplete with the DEC-003-21 description."""
    client = _llamaswap()
    _install_sse(monkeypatch, client, [_incomplete_event(details)])

    with pytest.raises(LLMError) as excinfo:
        await client._stream_responses_tool_round(None, None)

    _assert_description(excinfo.value, details, mode, reason)


@pytest.mark.asyncio
@pytest.mark.parametrize(("details", "mode", "reason"), _REASON_CASES)
async def test_llamaswap_tool_round_incomplete_reply_description__f004_s003_DoD17(monkeypatch, details, mode, reason):
    """A non-streamed llama-swap tool round with status incomplete raises LLMError
    with the DEC-003-21 description."""
    client = _llamaswap()
    _fake_request(monkeypatch, client, _incomplete_reply(details))

    with pytest.raises(LLMError) as excinfo:
        await client.chat_with_tools(
            list(_MESSAGES), tools_definitions=copy.deepcopy(_LOOKUP_DEFS), tools={"lookup": _lookup}
        )

    _assert_description(excinfo.value, details, mode, reason)
