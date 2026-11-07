"""Tool-loop error mapping in ``chat_with_tools`` on the OpenAI and llama-swap clients.

Covers feature 002.edge-fixes, step 001 (``docs/plans/002.edge-fixes/001.tool-loop-errors.md``): unoffered or
unnamed tools, unreadable arguments, unserialisable results and the round-limit error, plus the failure kinds
that must stay unchanged. Every scenario runs on both clients, streamed and non-streamed.
"""

import json
import re
from collections.abc import Callable
from typing import Any

import pytest

from llm.exceptions import LLMError
from llm.llama_swap import LlamaSwapAPIClient
from llm.openai import OpenAIAPIClient

STREAM_MODES = [pytest.param(False, id="non_streamed"), pytest.param(True, id="streamed")]

USER_MESSAGES = [{"role": "user", "content": "Use the tools."}]

# Sentinel meaning "leave the key out of the tool call" (distinct from an empty value).
_MISSING = object()

Round = tuple[str, list[dict[str, Any]]]


@pytest.fixture(autouse=True)
def _no_tracing(monkeypatch):
    """Keep tracing off so a developer's LLM_TRACE setting cannot change what these tests observe."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _openai() -> OpenAIAPIClient:
    """Build an OpenAI client with an explicit test credential."""
    return OpenAIAPIClient(model="gpt-4o-mini", bearer_token="sk-test")


def _llamaswap() -> LlamaSwapAPIClient:
    """Build a llama-swap client with an explicit test credential."""
    return LlamaSwapAPIClient(model="some-model", base_url="http://localhost:8080/v1", bearer_token="sk-test")


def _tool_def(name: str) -> dict[str, Any]:
    """Return a tool definition advertising ``name`` in the chat-completions tool shape."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Test tool {name}.",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _call(name: Any = "add", arguments: Any = _MISSING, call_id: str = "call_1") -> dict[str, Any]:
    """Return one requested tool call; ``arguments`` left as ``_MISSING`` omits the key."""
    function: dict[str, Any] = {"name": name}
    if arguments is not _MISSING:
        function["arguments"] = arguments
    return {"id": call_id, "type": "function", "function": function}


def _install_rounds(
    monkeypatch,
    client: Any,
    *,
    stream: bool,
    rounds: list[Round],
    repeat_last: bool = False,
) -> list[dict[str, Any]]:
    """Fake the server: each request gets the next ``(content, tool_calls)`` round.

    Streamed rounds replace ``_tools_stream_request``; non-streamed rounds replace ``_request`` and wrap the
    round in a chat-completions reply. ``repeat_last`` models a model that keeps asking for tools forever.
    Returns the list of payloads sent, with messages copied so later mutation of the history cannot alias.
    """
    sent: list[dict[str, Any]] = []

    def next_round(payload: dict[str, Any]) -> Round:
        sent.append({**payload, "messages": [dict(m) for m in payload["messages"]]})
        index = len(sent) - 1
        if index >= len(rounds):
            if not repeat_last:
                raise AssertionError(f"Unexpected extra request #{index + 1}")
            index = len(rounds) - 1
        return rounds[index]

    if stream:

        async def fake_stream_request(payload, on_delta=None):
            return next_round(payload)

        monkeypatch.setattr(client, "_tools_stream_request", fake_stream_request)
    else:

        async def fake_request(method, path, **kw):
            assert method == "POST"
            assert path == "/chat/completions"
            content, tool_calls = next_round(kw["json"])
            message: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                message["tool_calls"] = tool_calls
            return {"choices": [{"message": message}]}

        monkeypatch.setattr(client, "_request", fake_request)
    return sent


def _last_tool_message(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the tool message the client sent at the end of a request's history."""
    message = payload["messages"][-1]
    assert message["role"] == "tool"
    return message


def _recording_add(calls: list[dict[str, Any]]) -> Callable[..., int]:
    """Return an ``add(a, b)`` tool that records every invocation's keyword arguments."""

    def add(a: int = 0, b: int = 0) -> int:
        calls.append({"a": a, "b": b})
        return a + b

    return add


# ---------------------------------------------------------------------------
# Scenarios (shared by both clients)
# ---------------------------------------------------------------------------


async def _check_unoffered_tool(monkeypatch, client: Any, stream: bool) -> None:
    """DoD-1: a requested name not in ``tools_definitions`` raises LLMError naming it; its callable never runs."""
    add_calls: list[dict[str, Any]] = []
    secret_calls: list[dict[str, Any]] = []

    def secret_tool() -> str:
        secret_calls.append({})
        return "leaked"

    _install_rounds(
        monkeypatch, client, stream=stream, rounds=[("", [_call("secret_tool", "{}")]), ("Done.", [])]
    )
    with pytest.raises(LLMError) as exc_info:
        await client.chat_with_tools(
            USER_MESSAGES,
            tools_definitions=[_tool_def("add")],
            # The callable exists in ``tools`` but is not advertised: it must still be refused (DEC-002-3).
            tools={"add": _recording_add(add_calls), "secret_tool": secret_tool},
            stream=stream,
        )
    assert "secret_tool" in str(exc_info.value)
    assert secret_calls == []
    assert add_calls == []


_UNNAMED_CALLS = [
    pytest.param({"id": "call_1", "type": "function", "function": {"name": "", "arguments": "{}"}}, id="empty_name"),
    pytest.param({"id": "call_1", "type": "function", "function": {"arguments": "{}"}}, id="missing_name"),
    pytest.param({"id": "call_1", "type": "function"}, id="missing_function"),
]


async def _check_unnamed_tool(monkeypatch, client: Any, stream: bool, call: dict[str, Any]) -> None:
    """DoD-2: a requested call with an empty or missing name raises LLMError and no tool runs."""
    add_calls: list[dict[str, Any]] = []
    _install_rounds(monkeypatch, client, stream=stream, rounds=[("", [call]), ("Done.", [])])
    with pytest.raises(LLMError):
        await client.chat_with_tools(
            USER_MESSAGES,
            tools_definitions=[_tool_def("add")],
            tools={"add": _recording_add(add_calls)},
            stream=stream,
        )
    assert add_calls == []


async def _check_unreadable_arguments(monkeypatch, client: Any, stream: bool, arguments: str) -> None:
    """DoD-3/4: argument text that is not a JSON object raises LLMError naming the tool; the tool never runs."""
    calls: list[dict[str, Any]] = []
    _install_rounds(
        monkeypatch, client, stream=stream, rounds=[("", [_call("add_numbers", arguments)]), ("Done.", [])]
    )
    with pytest.raises(LLMError) as exc_info:
        await client.chat_with_tools(
            USER_MESSAGES,
            tools_definitions=[_tool_def("add_numbers")],
            tools={"add_numbers": _recording_add(calls)},
            stream=stream,
        )
    assert "add_numbers" in str(exc_info.value)
    assert calls == []


_ARGUMENT_FORMS = [
    pytest.param('{"a": 2, "b": 5}', {"a": 2, "b": 5}, id="json_object_text"),
    pytest.param({"a": 2, "b": 5}, {"a": 2, "b": 5}, id="mapping"),
    pytest.param("", {}, id="empty_text"),
    pytest.param(_MISSING, {}, id="missing"),
]


async def _check_arguments_used(
    monkeypatch, client: Any, stream: bool, arguments: Any, expected: dict[str, Any]
) -> None:
    """DoD-5: the tool runs with the model's arguments (object text or mapping), or with none when absent."""
    received: list[dict[str, Any]] = []

    # Named parameters (not **kwargs) so the existing argument-name check accepts ``a``/``b``;
    # the sentinel default tells "not passed" apart from any value the model could send.
    def tool(a: Any = _MISSING, b: Any = _MISSING) -> str:
        received.append({k: v for k, v in (("a", a), ("b", b)) if v is not _MISSING})
        return "ok"

    _install_rounds(
        monkeypatch, client, stream=stream, rounds=[("", [_call("tool", arguments)]), ("All done.", [])]
    )
    result = await client.chat_with_tools(
        USER_MESSAGES,
        tools_definitions=[_tool_def("tool")],
        tools={"tool": tool},
        stream=stream,
    )
    assert result == "All done."
    assert received == [expected]


def _circular_mapping() -> dict[str, Any]:
    """Return a self-referencing mapping, which JSON serialisation rejects (circular reference)."""
    value: dict[str, Any] = {}
    value["self"] = value
    return value


_UNSERIALISABLE_RESULTS = [
    pytest.param(lambda: {"value": object()}, id="mapping_with_object"),
    pytest.param(lambda: [object()], id="list_with_object"),
    pytest.param(lambda: [{1, 2}], id="list_with_set"),
    pytest.param(_circular_mapping, id="circular_mapping"),
]


async def _check_unserialisable_result(monkeypatch, client: Any, stream: bool, make: Callable[[], Any]) -> None:
    """DoD-6: a mapping/list result that cannot become JSON raises LLMError naming the tool."""

    def report_tool() -> Any:
        return make()

    _install_rounds(monkeypatch, client, stream=stream, rounds=[("", [_call("report_tool", "{}")]), ("Done.", [])])
    with pytest.raises(LLMError) as exc_info:
        await client.chat_with_tools(
            USER_MESSAGES,
            tools_definitions=[_tool_def("report_tool")],
            tools={"report_tool": report_tool},
            stream=stream,
        )
    assert "report_tool" in str(exc_info.value)


class _CustomText:
    """A non-text tool result with its own text form."""

    def __str__(self) -> str:
        return "custom-text-form"


async def _check_custom_text_result(monkeypatch, client: Any, stream: bool) -> None:
    """DoD-6: a non-mapping, non-list result is sent to the model as its text form."""
    sent = _install_rounds(
        monkeypatch, client, stream=stream, rounds=[("", [_call("custom", "{}")]), ("Final.", [])]
    )
    result = await client.chat_with_tools(
        USER_MESSAGES,
        tools_definitions=[_tool_def("custom")],
        tools={"custom": lambda: _CustomText()},
        stream=stream,
    )
    assert result == "Final."
    assert len(sent) == 2
    assert _last_tool_message(sent[1])["content"] == "custom-text-form"


_SERIALISABLE_RESULTS = [
    pytest.param({"city": "Zürich", "note": "日本語", "count": 2}, id="mapping"),
    pytest.param(["Zürich", {"greeting": "привет"}, 3], id="list"),
]


async def _check_serialised_result(monkeypatch, client: Any, stream: bool, value: Any) -> None:
    """DoD-7: a serialisable mapping/list is sent next round as JSON text with non-ASCII kept unescaped."""
    sent = _install_rounds(
        monkeypatch, client, stream=stream, rounds=[("", [_call("lookup", "{}")]), ("Answer.", [])]
    )
    result = await client.chat_with_tools(
        USER_MESSAGES,
        tools_definitions=[_tool_def("lookup")],
        tools={"lookup": lambda: value},
        stream=stream,
    )
    assert result == "Answer."
    assert len(sent) == 2
    content = _last_tool_message(sent[1])["content"]
    assert isinstance(content, str)
    assert json.loads(content) == value
    assert "Zürich" in content
    assert "\\u" not in content


async def _check_round_limit_message(monkeypatch, client: Any, stream: bool) -> None:
    """DoD-8: reaching the round limit raises RuntimeError stating the limit, ``max_loops`` and its value."""
    add_calls: list[dict[str, Any]] = []
    # Every round carries partial text plus a tool request, so the model never finishes on its own.
    _install_rounds(
        monkeypatch,
        client,
        stream=stream,
        rounds=[("partial answer", [_call("add", '{"a": 1, "b": 1}')])],
        repeat_last=True,
    )
    with pytest.raises(RuntimeError) as exc_info:
        await client.chat_with_tools(
            USER_MESSAGES,
            tools_definitions=[_tool_def("add")],
            tools={"add": _recording_add(add_calls)},
            stream=stream,
            max_loops=4,
        )
    message = str(exc_info.value)
    assert "max_loops" in message
    # The configured value appears as a standalone number in the message.
    assert re.search(r"(?<!\d)4(?!\d)", message)
    # The message states the round limit as the reason (wording not fixed by the spec beyond this).
    assert re.search(r"limit|reached|exceed", message, re.IGNORECASE)


async def _check_round_limit_count(monkeypatch, client: Any, stream: bool) -> None:
    """DoD-9: with ``max_loops=2`` and a model that keeps asking for tools, exactly 2 requests are sent."""
    sent = _install_rounds(
        monkeypatch,
        client,
        stream=stream,
        rounds=[("", [_call("add", '{"a": 1, "b": 2}')])],
        repeat_last=True,
    )
    with pytest.raises(RuntimeError):
        await client.chat_with_tools(
            USER_MESSAGES,
            tools_definitions=[_tool_def("add")],
            tools={"add": _recording_add([])},
            stream=stream,
            max_loops=2,
        )
    assert len(sent) == 2


async def _check_tool_raises(monkeypatch, client: Any, stream: bool) -> None:
    """DoD-10: a tool that raises ends the call with RuntimeError naming the tool."""

    def failing_tool() -> str:
        raise ZeroDivisionError("boom")

    _install_rounds(monkeypatch, client, stream=stream, rounds=[("", [_call("failing_tool", "{}")]), ("Done.", [])])
    with pytest.raises(RuntimeError) as exc_info:
        await client.chat_with_tools(
            USER_MESSAGES,
            tools_definitions=[_tool_def("failing_tool")],
            tools={"failing_tool": failing_tool},
            stream=stream,
        )
    assert "failing_tool" in str(exc_info.value)


async def _check_bad_argument_name(monkeypatch, client: Any, stream: bool) -> None:
    """DoD-10: an argument name the callable lacks ends with ValueError and the tool does not run."""
    add_calls: list[dict[str, Any]] = []
    _install_rounds(
        monkeypatch, client, stream=stream, rounds=[("", [_call("add", '{"zzz": 1}')]), ("Done.", [])]
    )
    with pytest.raises(ValueError):
        await client.chat_with_tools(
            USER_MESSAGES,
            tools_definitions=[_tool_def("add")],
            tools={"add": _recording_add(add_calls)},
            stream=stream,
        )
    assert add_calls == []


async def _check_missing_callable(monkeypatch, client: Any, stream: bool) -> None:
    """DoD-10: an advertised tool without a callable is refused with ValueError before any request."""
    sent = _install_rounds(monkeypatch, client, stream=stream, rounds=[("Done.", [])])
    with pytest.raises(ValueError):
        await client.chat_with_tools(
            USER_MESSAGES,
            tools_definitions=[_tool_def("add"), _tool_def("orphan")],
            tools={"add": _recording_add([])},
            stream=stream,
        )
    assert sent == []


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_unoffered_tool_raises_llmerror__DoD1(monkeypatch, stream):
    """An unadvertised tool name ends with LLMError naming it; its callable is not run (DoD-1)."""
    await _check_unoffered_tool(monkeypatch, _openai(), stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("call", _UNNAMED_CALLS)
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_unnamed_tool_raises_llmerror__DoD2(monkeypatch, stream, call):
    """An empty or missing tool name ends with LLMError and no tool runs (DoD-2)."""
    await _check_unnamed_tool(monkeypatch, _openai(), stream, call)


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["{not json", '{"a": 1,'])
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_invalid_json_arguments_raise_llmerror__DoD3(monkeypatch, stream, arguments):
    """Argument text that is not valid JSON ends with LLMError naming the tool; it does not run (DoD-3)."""
    await _check_unreadable_arguments(monkeypatch, _openai(), stream, arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["[1, 2]", "42", '"text"', "null"])
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_non_object_json_arguments_raise_llmerror__DoD4(monkeypatch, stream, arguments):
    """Argument text decoding to a list, number, string or null ends with LLMError naming the tool (DoD-4)."""
    await _check_unreadable_arguments(monkeypatch, _openai(), stream, arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize(("arguments", "expected"), _ARGUMENT_FORMS)
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_tool_runs_with_model_arguments__DoD5(monkeypatch, stream, arguments, expected):
    """The tool runs with object-text or mapping arguments, and with none when empty or missing (DoD-5)."""
    await _check_arguments_used(monkeypatch, _openai(), stream, arguments, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("make", _UNSERIALISABLE_RESULTS)
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_unserialisable_result_raises_llmerror__DoD6(monkeypatch, stream, make):
    """An unserialisable mapping/list result ends with LLMError naming the tool (DoD-6)."""
    await _check_unserialisable_result(monkeypatch, _openai(), stream, make)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_custom_text_result_sent_as_text__DoD6(monkeypatch, stream):
    """A result with a custom text form is sent as that text (DoD-6)."""
    await _check_custom_text_result(monkeypatch, _openai(), stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", _SERIALISABLE_RESULTS)
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_serialisable_result_sent_as_unescaped_json__DoD7(monkeypatch, stream, value):
    """A serialisable mapping/list is sent next round as JSON text with non-ASCII unescaped (DoD-7)."""
    await _check_serialised_result(monkeypatch, _openai(), stream, value)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_round_limit_error_names_limit_and_value__DoD8(monkeypatch, stream):
    """The round-limit RuntimeError states the reason, contains ``max_loops`` and the value (DoD-8)."""
    await _check_round_limit_message(monkeypatch, _openai(), stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_round_limit_of_two_sends_two_requests__DoD9(monkeypatch, stream):
    """A per-call limit of 2 sends exactly 2 requests before the error (DoD-9)."""
    await _check_round_limit_count(monkeypatch, _openai(), stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_raising_tool_still_runtimeerror__DoD10(monkeypatch, stream):
    """A tool that raises still ends with RuntimeError naming the tool (DoD-10)."""
    await _check_tool_raises(monkeypatch, _openai(), stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_bad_argument_name_still_valueerror__DoD10(monkeypatch, stream):
    """An argument name the callable lacks still ends with ValueError; the tool does not run (DoD-10)."""
    await _check_bad_argument_name(monkeypatch, _openai(), stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_missing_callable_still_valueerror_before_request__DoD10(monkeypatch, stream):
    """An advertised tool without a callable is still refused with ValueError before any request (DoD-10)."""
    await _check_missing_callable(monkeypatch, _openai(), stream)


# ---------------------------------------------------------------------------
# llama-swap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_llamaswap_unoffered_tool_raises_llmerror__DoD1(monkeypatch, stream):
    """An unadvertised tool name ends with LLMError naming it; its callable is not run (DoD-1)."""
    await _check_unoffered_tool(monkeypatch, _llamaswap(), stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("call", _UNNAMED_CALLS)
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_llamaswap_unnamed_tool_raises_llmerror__DoD2(monkeypatch, stream, call):
    """An empty or missing tool name ends with LLMError and no tool runs (DoD-2)."""
    await _check_unnamed_tool(monkeypatch, _llamaswap(), stream, call)


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["{not json", '{"a": 1,'])
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_llamaswap_invalid_json_arguments_raise_llmerror__DoD3(monkeypatch, stream, arguments):
    """Argument text that is not valid JSON ends with LLMError naming the tool; it does not run (DoD-3)."""
    await _check_unreadable_arguments(monkeypatch, _llamaswap(), stream, arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["[1, 2]", "42", '"text"', "null"])
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_llamaswap_non_object_json_arguments_raise_llmerror__DoD4(monkeypatch, stream, arguments):
    """Argument text decoding to a list, number, string or null ends with LLMError naming the tool (DoD-4)."""
    await _check_unreadable_arguments(monkeypatch, _llamaswap(), stream, arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize(("arguments", "expected"), _ARGUMENT_FORMS)
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_llamaswap_tool_runs_with_model_arguments__DoD5(monkeypatch, stream, arguments, expected):
    """The tool runs with object-text or mapping arguments, and with none when empty or missing (DoD-5)."""
    await _check_arguments_used(monkeypatch, _llamaswap(), stream, arguments, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("make", _UNSERIALISABLE_RESULTS)
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_llamaswap_unserialisable_result_raises_llmerror__DoD6(monkeypatch, stream, make):
    """An unserialisable mapping/list result ends with LLMError naming the tool (DoD-6)."""
    await _check_unserialisable_result(monkeypatch, _llamaswap(), stream, make)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_llamaswap_custom_text_result_sent_as_text__DoD6(monkeypatch, stream):
    """A result with a custom text form is sent as that text (DoD-6)."""
    await _check_custom_text_result(monkeypatch, _llamaswap(), stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", _SERIALISABLE_RESULTS)
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_llamaswap_serialisable_result_sent_as_unescaped_json__DoD7(monkeypatch, stream, value):
    """A serialisable mapping/list is sent next round as JSON text with non-ASCII unescaped (DoD-7)."""
    await _check_serialised_result(monkeypatch, _llamaswap(), stream, value)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_llamaswap_round_limit_error_names_limit_and_value__DoD8(monkeypatch, stream):
    """The round-limit RuntimeError states the reason, contains ``max_loops`` and the value (DoD-8)."""
    await _check_round_limit_message(monkeypatch, _llamaswap(), stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_llamaswap_round_limit_of_two_sends_two_requests__DoD9(monkeypatch, stream):
    """A per-call limit of 2 sends exactly 2 requests before the error (DoD-9)."""
    await _check_round_limit_count(monkeypatch, _llamaswap(), stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_llamaswap_raising_tool_still_runtimeerror__DoD10(monkeypatch, stream):
    """A tool that raises still ends with RuntimeError naming the tool (DoD-10)."""
    await _check_tool_raises(monkeypatch, _llamaswap(), stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_llamaswap_bad_argument_name_still_valueerror__DoD10(monkeypatch, stream):
    """An argument name the callable lacks still ends with ValueError; the tool does not run (DoD-10)."""
    await _check_bad_argument_name(monkeypatch, _llamaswap(), stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_llamaswap_missing_callable_still_valueerror_before_request__DoD10(monkeypatch, stream):
    """An advertised tool without a callable is still refused with ValueError before any request (DoD-10)."""
    await _check_missing_callable(monkeypatch, _llamaswap(), stream)
