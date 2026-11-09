"""Tolerant / strict tool-call failures and the tool-event observer on ``OpenAIChatCompletionsClient``.

Feature 006 (tolerant-tool-loop), step 008: ``chat_with_tools`` gains ``tool_errors`` (``"result"`` default,
``"raise"`` opt-out) and ``on_tool_event``. Covers the step's DoD-1..22: the model-facing texts per failure kind
(DEC-arch-14), the argument-name check and its ``**kwargs`` / unreadable-signature skips (DEC-arch-16), strict
mode as before with ``kind="tool"`` (DEC-arch-17, DEC-arch-19), observer events and their filling
(DEC-arch-18, DEC-006-4), observer failures (DEC-arch-26, DEC-006-8), cancellation, the round limit, and the
trace entries of tolerant failures (DEC-006-7). The server is faked per round through ``_request``
(non-streamed) or ``_tools_stream_request`` (streamed). Every expected value comes from the step's Definition
of done, the cited decisions and docs/architecture/interfaces.md § Tool failures / § Tool event observer.
"""

import asyncio
import json
import re
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import llm.trace
from llm.exceptions import LLMError, LLMToolObserverWarning
from llm.openai_chat_completions import OpenAIChatCompletionsClient

STREAM_MODES = [pytest.param(False, id="non_streamed"), pytest.param(True, id="streamed")]

# Both tool-failure handling modes; ``{}`` passes nothing so the default (tolerant) applies.
MODES = [pytest.param({}, id="default"), pytest.param({"tool_errors": "raise"}, id="strict")]

USER_MESSAGES = [{"role": "user", "content": "Use the tools."}]
ANSWER = "Final answer after the tools."

# Sentinel meaning "leave the key out of the tool call" (distinct from an empty or null value).
_MISSING = object()

Round = tuple[str, list[dict[str, Any]]]


@pytest.fixture(autouse=True)
def _no_tracing(monkeypatch):
    """Keep tracing off unless a test turns it on, so a developer's LLM_TRACE cannot change what is observed."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


# ---------------------------------------------------------------------------
# Builders and fakes
# ---------------------------------------------------------------------------


def _client() -> OpenAIChatCompletionsClient:
    """Build the client under test with an explicit test credential."""
    return OpenAIChatCompletionsClient(model="gpt-4o-mini", bearer_token="sk-test")


def _tool_def(name: str) -> dict[str, Any]:
    """Return a chat-completions tool definition advertising ``name``."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Test tool {name}.",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _call(name: Any, arguments: Any = _MISSING, call_id: str | None = "call_1") -> dict[str, Any]:
    """Return one requested tool call; ``arguments=_MISSING`` omits the key, ``call_id=None`` omits the id."""
    function: dict[str, Any] = {"name": name}
    if arguments is not _MISSING:
        function["arguments"] = arguments
    call: dict[str, Any] = {"type": "function", "function": function}
    if call_id is not None:
        call["id"] = call_id
    return call


def _install_rounds(
    monkeypatch, client: Any, *, stream: bool, rounds: list[Round], repeat_last: bool = False
) -> list[dict[str, Any]]:
    """Fake the server: each request gets the next ``(content, tool_calls)`` round.

    Streamed rounds replace ``_tools_stream_request``; non-streamed rounds replace ``_request`` and wrap the
    round in a chat-completions reply. An extra request fails the test unless ``repeat_last`` models a model
    that keeps asking forever. Returns the payloads sent, messages copied so later history growth cannot alias.
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


def _tool_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the tool messages that follow the last assistant turn of a request's history."""
    messages = payload["messages"]
    last_assistant = max(i for i, m in enumerate(messages) if m.get("role") == "assistant")
    tail = messages[last_assistant + 1 :]
    assert tail, "no tool message follows the assistant tool-call turn"
    assert all(m.get("role") == "tool" for m in tail), tail
    return tail


def _told(payload: dict[str, Any], call_id: str = "call_1") -> str:
    """Return what the model is told for ``call_id``: the content of that call's tool message (exactly one)."""
    matches = [m for m in _tool_messages(payload) if m.get("tool_call_id") == call_id]
    assert len(matches) == 1, _tool_messages(payload)
    return matches[0]["content"]


def _event(
    type_: str,
    name: str,
    call_id: str | None,
    arguments: str,
    *,
    output: str | None = None,
    error_kind: str | None = None,
) -> dict[str, Any]:
    """Return an expected ``ToolEvent``: all six keys present, ``None`` where unused (DEC-006-4)."""
    return {
        "type": type_,
        "name": name,
        "call_id": call_id,
        "arguments": arguments,
        "output": output,
        "error_kind": error_kind,
    }


class _EventLog:
    """Plain-function observer that keeps a copy of every event it receives."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def __call__(self, event: Any) -> None:
        self.events.append(dict(event))


class _ObserverBoom(Exception):
    """The ordinary exception a failing observer raises."""


class _FailingObserver:
    """Observer that records each event, then raises on chosen ``(event type, tool name)`` pairs.

    ``is_async`` makes it a coroutine observer, so the failure comes from awaiting its result. ``raised`` holds
    every exception it raised, so a test can check the very same object reaches the caller.
    """

    def __init__(self, fail_on: set[tuple[str, str]], *, is_async: bool = False) -> None:
        self.fail_on = fail_on
        self.is_async = is_async
        self.events: list[dict[str, Any]] = []
        self.raised: list[_ObserverBoom] = []

    def _handle(self, event: Any) -> None:
        self.events.append(dict(event))
        if (event["type"], event["name"]) in self.fail_on:
            boom = _ObserverBoom("observer exploded")
            self.raised.append(boom)
            raise boom

    def __call__(self, event: Any) -> Any:
        if not self.is_async:
            self._handle(event)
            return None

        async def handle() -> None:
            await asyncio.sleep(0)
            self._handle(event)

        return handle()


def _lookup(key: str = "") -> str:
    """A well-behaved tool declaring one argument, ``key``."""
    return f"record:{key}"


def _circular_mapping() -> dict[str, Any]:
    """Return a self-referencing mapping, which JSON serialisation rejects."""
    value: dict[str, Any] = {}
    value["self"] = value
    return value


_UNSERIALISABLE_RESULTS = [
    pytest.param(lambda: {"value": object()}, id="mapping_with_object"),
    pytest.param(lambda: [{1, 2}], id="list_with_set"),
    pytest.param(_circular_mapping, id="circular_mapping"),
]


def _raising_tool(*args: Any, **kwargs: Any) -> str:
    """A tool that always raises an ordinary exception (failure kind ``tool_raised``)."""
    raise ZeroDivisionError("secret-internal-detail-4711")


def _unserialisable_tool() -> Any:
    """A tool whose mapping result cannot become JSON text (failure kind ``unserialisable_result``)."""
    return {"value": object()}


# One case per failure kind: (requested name, raw arguments, offered tools, kind, model-facing text).
# The offered tools double as the callables; ``ghost`` is never offered.
_FAILURE_CASES = [
    pytest.param("ghost", '{"key": "k1"}', {"lookup": _lookup}, "not_offered", "Tool 'ghost' is not available.",
                 id="not_offered"),
    pytest.param("lookup", "{oops", {"lookup": _lookup}, "bad_arguments", "Invalid arguments for tool 'lookup'.",
                 id="bad_arguments"),
    pytest.param("lookup", '{"zzz": 1}', {"lookup": _lookup}, "unknown_argument",
                 "Tool 'lookup' has no argument 'zzz'.", id="unknown_argument"),
    pytest.param("explode", "{}", {"explode": _raising_tool}, "tool_raised", "Tool 'explode' failed.",
                 id="tool_raised"),
    pytest.param("report", "{}", {"report": _unserialisable_tool}, "unserialisable_result",
                 "Tool 'report' failed.", id="unserialisable_result"),
]


async def _run(
    client: Any, tools: dict[str, Callable[..., Any]], *, stream: bool = False, **kwargs: Any
) -> str:
    """Call ``chat_with_tools`` offering exactly ``tools`` (one definition per callable)."""
    return await client.chat_with_tools(
        USER_MESSAGES,
        tools_definitions=[_tool_def(name) for name in tools],
        tools=tools,
        stream=stream,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# DoD-1 — not offered / empty name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_chat_completions_unoffered_tool_told_not_available__DoD1(monkeypatch, stream):
    """An unoffered tool (callable present but not advertised) runs nothing; the model is told; answer returned."""
    secret_calls: list[str] = []

    def secret_tool() -> str:
        secret_calls.append("ran")
        return "leaked"

    client = _client()
    sent = _install_rounds(
        monkeypatch, client, stream=stream, rounds=[("", [_call("secret_tool", "{}")]), (ANSWER, [])]
    )
    result = await client.chat_with_tools(
        USER_MESSAGES,
        tools_definitions=[_tool_def("lookup")],
        tools={"lookup": _lookup, "secret_tool": secret_tool},
        stream=stream,
    )
    assert result == ANSWER
    assert len(sent) == 2
    assert secret_calls == []
    assert _told(sent[1]) == "Tool 'secret_tool' is not available."


@pytest.mark.asyncio
async def test_openai_chat_completions_empty_tool_name_told_not_available__DoD1(monkeypatch):
    """An empty requested name runs no tool; the model is told ``Tool '' is not available.``; answer returned."""
    ran: list[str] = []

    def lookup(key: str = "") -> str:
        ran.append(key)
        return "x"

    client = _client()
    sent = _install_rounds(monkeypatch, client, stream=False, rounds=[("", [_call("", "{}")]), (ANSWER, [])])
    result = await _run(client, {"lookup": lookup})
    assert result == ANSWER
    assert ran == []
    assert _told(sent[1]) == "Tool '' is not available."


# ---------------------------------------------------------------------------
# DoD-2 — bad arguments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["{not json", '{"a": 1,', "[1, 2]", "42", '"text"', "null"])
async def test_openai_chat_completions_bad_arguments_told_invalid__DoD2(monkeypatch, arguments):
    """Invalid JSON or non-object JSON runs no tool; model told ``Invalid arguments for tool '<n>'.``; continues."""
    ran: list[Any] = []

    def add_numbers(**kwargs: Any) -> str:
        ran.append(kwargs)
        return "x"

    client = _client()
    sent = _install_rounds(
        monkeypatch, client, stream=False, rounds=[("", [_call("add_numbers", arguments)]), (ANSWER, [])]
    )
    result = await _run(client, {"add_numbers": add_numbers})
    assert result == ANSWER
    assert ran == []
    assert _told(sent[1]) == "Invalid arguments for tool 'add_numbers'."


# ---------------------------------------------------------------------------
# DoD-3 — unknown argument name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "first_unknown"),
    [
        pytest.param('{"a": 1, "zzz": 2, "yyy": 3}', "zzz", id="zzz_first"),
        pytest.param('{"yyy": 1, "a": 2, "zzz": 3}', "yyy", id="yyy_first"),
    ],
)
async def test_openai_chat_completions_unknown_argument_told_first_name__DoD3(monkeypatch, arguments, first_unknown):
    """An undeclared argument name runs no tool; the model is told the first undeclared name in sent order."""
    ran: list[Any] = []

    def add(a: int = 0, b: int = 0) -> int:
        ran.append((a, b))
        return a + b

    client = _client()
    sent = _install_rounds(monkeypatch, client, stream=False, rounds=[("", [_call("add", arguments)]), (ANSWER, [])])
    result = await _run(client, {"add": add})
    assert result == ANSWER
    assert ran == []
    assert _told(sent[1]) == f"Tool 'add' has no argument '{first_unknown}'."


# ---------------------------------------------------------------------------
# DoD-4 — tool raised
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [pytest.param(False, id="sync_tool"), pytest.param(True, id="async_tool")])
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_chat_completions_raising_tool_told_failed_without_message__DoD4(monkeypatch, stream, is_async):
    """A tool raising an ordinary exception: model told ``Tool '<n>' failed.``, no exception text; continues."""
    secret = "secret-internal-detail-4711"

    def sync_tool() -> str:
        raise ValueError(secret)

    async def async_tool() -> str:
        raise ValueError(secret)

    client = _client()
    sent = _install_rounds(monkeypatch, client, stream=stream, rounds=[("", [_call("fragile", "{}")]), (ANSWER, [])])
    result = await _run(client, {"fragile": async_tool if is_async else sync_tool}, stream=stream)
    assert result == ANSWER
    told = _told(sent[1])
    assert told == "Tool 'fragile' failed."
    assert secret not in told


# ---------------------------------------------------------------------------
# DoD-5 — unserialisable result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("make", _UNSERIALISABLE_RESULTS)
async def test_openai_chat_completions_unserialisable_result_told_failed__DoD5(monkeypatch, make):
    """A result that cannot become text: the model is told ``Tool '<n>' failed.``; the conversation continues."""
    client = _client()
    sent = _install_rounds(monkeypatch, client, stream=False, rounds=[("", [_call("report", "{}")]), (ANSWER, [])])
    result = await _run(client, {"report": lambda: make()})
    assert result == ANSWER
    assert _told(sent[1]) == "Tool 'report' failed."


# ---------------------------------------------------------------------------
# DoD-6 — the middle of three calls fails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_chat_completions_middle_failure_others_run_in_order__DoD6(monkeypatch):
    """Three calls, middle fails: first and third run in requested order; each call gets its own tool message."""
    order: list[str] = []

    def first() -> str:
        order.append("first")
        return "one"

    def middle() -> str:
        order.append("middle")
        raise RuntimeError("middle broke")

    def third() -> str:
        order.append("third")
        return "three"

    client = _client()
    calls = [_call("first", "{}", "call_a"), _call("middle", "{}", "call_b"), _call("third", "{}", "call_c")]
    sent = _install_rounds(monkeypatch, client, stream=False, rounds=[("", calls), (ANSWER, [])])
    result = await _run(client, {"first": first, "middle": middle, "third": third})
    assert result == ANSWER
    assert order == ["first", "middle", "third"]
    told = [(m.get("tool_call_id"), m["content"]) for m in _tool_messages(sent[1])]
    assert told == [("call_a", "one"), ("call_b", "Tool 'middle' failed."), ("call_c", "three")]


# ---------------------------------------------------------------------------
# DoD-7 — four rounds of mixed failures, streamed and not
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_chat_completions_four_round_recovery_returns_answer__DoD7(monkeypatch, stream):
    """Unoffered tool, non-object arguments, unknown argument, then an answer: round 4's answer, no error."""
    ran: list[Any] = []

    def add(a: int = 0, b: int = 0) -> int:
        ran.append((a, b))
        return a + b

    client = _client()
    sent = _install_rounds(
        monkeypatch,
        client,
        stream=stream,
        rounds=[
            ("", [_call("ghost", "{}", "call_r1")]),
            ("", [_call("add", "[1, 2]", "call_r2")]),
            ("", [_call("add", '{"zzz": 1}', "call_r3")]),
            (ANSWER, []),
        ],
    )
    result = await _run(client, {"add": add}, stream=stream)
    assert result == ANSWER
    assert len(sent) == 4
    assert ran == []
    assert _told(sent[1], "call_r1") == "Tool 'ghost' is not available."
    assert _told(sent[2], "call_r2") == "Invalid arguments for tool 'add'."
    assert _told(sent[3], "call_r3") == "Tool 'add' has no argument 'zzz'."


# ---------------------------------------------------------------------------
# DoD-8 — round limit with failures every round
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_chat_completions_failing_every_round_ends_with_round_limit__DoD8(monkeypatch):
    """A model that requests a failing call in every round ends with the round-limit RuntimeError and its value."""
    client = _client()
    sent = _install_rounds(
        monkeypatch, client, stream=False, rounds=[("", [_call("ghost", "{}")])], repeat_last=True
    )
    with pytest.raises(RuntimeError) as exc_info:
        await _run(client, {"lookup": _lookup}, max_loops=3)
    message = str(exc_info.value)
    assert "max_loops" in message
    assert re.search(r"(?<!\d)3(?!\d)", message)
    assert re.search(r"limit|reached|exceed", message, re.IGNORECASE)
    assert len(sent) == 3


# ---------------------------------------------------------------------------
# DoD-9 — cancellation propagates
# ---------------------------------------------------------------------------


def _sync_cancel() -> str:
    """A plain tool raising cancellation, which is not an ordinary exception."""
    raise asyncio.CancelledError()


async def _async_cancel() -> str:
    """A coroutine tool raising cancellation, which is not an ordinary exception."""
    raise asyncio.CancelledError()


_CANCELLING_TOOLS = [pytest.param(_sync_cancel, id="sync_tool"), pytest.param(_async_cancel, id="async_tool")]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("tool", _CANCELLING_TOOLS)
async def test_openai_chat_completions_cancellation_propagates_without_tool_message__DoD9(monkeypatch, tool, mode):
    """A tool raising cancellation: it propagates, no further round is requested, so no tool message is sent."""
    client = _client()
    sent = _install_rounds(monkeypatch, client, stream=False, rounds=[("", [_call("halt", "{}")]), (ANSWER, [])])
    with pytest.raises(asyncio.CancelledError):
        await _run(client, {"halt": tool}, **mode)
    assert len(sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_openai_chat_completions_cancellation_sends_no_error_event__DoD9(monkeypatch, mode):
    """With an observer, a cancelling tool yields its ``start`` event only: no ``error`` event is sent for it."""
    client = _client()
    log = _EventLog()
    sent = _install_rounds(monkeypatch, client, stream=False, rounds=[("", [_call("halt", "{}")]), (ANSWER, [])])
    with pytest.raises(asyncio.CancelledError):
        await _run(client, {"halt": _async_cancel}, on_tool_event=log, **mode)
    assert len(sent) == 1
    assert [e["type"] for e in log.events] == ["start"]


# ---------------------------------------------------------------------------
# DoD-10 — strict mode ends the call as before, with kind="tool" on LLMError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
@pytest.mark.parametrize(
    ("name", "arguments", "tools"),
    [
        pytest.param("ghost", "{}", {"lookup": _lookup}, id="not_offered"),
        pytest.param("", "{}", {"lookup": _lookup}, id="not_offered_empty_name"),
        pytest.param("lookup", "{oops", {"lookup": _lookup}, id="bad_arguments_invalid_json"),
        pytest.param("lookup", "[1, 2]", {"lookup": _lookup}, id="bad_arguments_not_object"),
        pytest.param("report", "{}", {"report": _unserialisable_tool}, id="unserialisable_result"),
    ],
)
async def test_openai_chat_completions_strict_llmerror_kinds_carry_tool_kind__DoD10(
    monkeypatch, stream, name, arguments, tools
):
    """Strict: not_offered / bad_arguments / unserialisable_result end with LLMError, ``kind == "tool"``."""
    client = _client()
    sent = _install_rounds(monkeypatch, client, stream=stream, rounds=[("", [_call(name, arguments)]), (ANSWER, [])])
    with pytest.raises(LLMError) as exc_info:
        await _run(client, tools, stream=stream, tool_errors="raise")
    assert exc_info.value.kind == "tool"
    if name:
        # Today's text names the tool (DEC-002-3..5).
        assert name in str(exc_info.value)
    assert len(sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_chat_completions_strict_unknown_argument_valueerror__DoD10(monkeypatch, stream):
    """Strict: an undeclared argument name ends with ValueError; the tool does not run; no further round."""
    ran: list[Any] = []

    def add(a: int = 0, b: int = 0) -> int:
        ran.append((a, b))
        return a + b

    client = _client()
    sent = _install_rounds(
        monkeypatch, client, stream=stream, rounds=[("", [_call("add", '{"zzz": 1}')]), (ANSWER, [])]
    )
    with pytest.raises(ValueError):
        await _run(client, {"add": add}, stream=stream, tool_errors="raise")
    assert ran == []
    assert len(sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_chat_completions_strict_tool_raised_runtimeerror_chained__DoD10(monkeypatch, stream):
    """Strict: a raising tool ends with RuntimeError naming the tool, chained from the tool's exception."""
    original = ZeroDivisionError("boom")

    def fragile() -> str:
        raise original

    client = _client()
    sent = _install_rounds(monkeypatch, client, stream=stream, rounds=[("", [_call("fragile", "{}")]), (ANSWER, [])])
    with pytest.raises(RuntimeError) as exc_info:
        await _run(client, {"fragile": fragile}, stream=stream, tool_errors="raise")
    assert "fragile" in str(exc_info.value)
    assert exc_info.value.__cause__ is original
    assert len(sent) == 1


# ---------------------------------------------------------------------------
# DoD-11 — **kwargs callables and unreadable signatures
# ---------------------------------------------------------------------------


class _UnreadableSignatureTool:
    """A callable whose signature cannot be read.

    ``inspect.signature`` raises ``TypeError`` when ``__signature__`` holds a value that is neither a Signature,
    text nor a callable, which is the "signature cannot be read" case of DEC-arch-16.
    """

    __signature__ = 42

    def __init__(self) -> None:
        self.received: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> str:
        self.received.append(kwargs)
        return "unreadable-ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_openai_chat_completions_var_keyword_tool_gets_exact_arguments__DoD11(monkeypatch, mode):
    """A ``**kwargs`` callable runs with exactly the arguments the model sent, in both modes."""
    received: list[dict[str, Any]] = []

    def catch_all(**kwargs: Any) -> str:
        received.append(kwargs)
        return "ok"

    client = _client()
    sent = _install_rounds(
        monkeypatch,
        client,
        stream=False,
        rounds=[("", [_call("catch_all", '{"alpha": 1, "beta": "two", "gamma": [3]}')]), (ANSWER, [])],
    )
    result = await _run(client, {"catch_all": catch_all}, **mode)
    assert result == ANSWER
    assert received == [{"alpha": 1, "beta": "two", "gamma": [3]}]
    assert _told(sent[1]) == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_openai_chat_completions_unreadable_signature_skips_name_check__DoD11(monkeypatch, mode):
    """A callable whose signature cannot be read is called with the model's arguments, without the name check."""
    tool = _UnreadableSignatureTool()
    client = _client()
    sent = _install_rounds(
        monkeypatch, client, stream=False, rounds=[("", [_call("opaque", '{"anything": 5}')]), (ANSWER, [])]
    )
    result = await _run(client, {"opaque": tool}, **mode)
    assert result == ANSWER
    assert tool.received == [{"anything": 5}]
    assert _told(sent[1]) == "unreadable-ok"


# ---------------------------------------------------------------------------
# DoD-12 — start event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_chat_completions_start_event_before_tool_runs__DoD12(monkeypatch, stream):
    """Before the tool runs the observer gets ``start`` with name, call id and the argument text verbatim."""
    raw = '{ "key" :  "k1" }'  # odd spacing: must arrive verbatim, not re-encoded
    log = _EventLog()
    seen_at_run: list[list[dict[str, Any]]] = []

    def lookup(key: str = "") -> str:
        seen_at_run.append(list(log.events))
        return "found"

    client = _client()
    _install_rounds(monkeypatch, client, stream=stream, rounds=[("", [_call("lookup", raw, "call_xyz")]), (ANSWER, [])])
    await _run(client, {"lookup": lookup}, stream=stream, on_tool_event=log)
    assert seen_at_run == [[_event("start", "lookup", "call_xyz", raw)]]
    assert log.events[0] == _event("start", "lookup", "call_xyz", raw)


@pytest.mark.asyncio
async def test_openai_chat_completions_call_without_id_gives_none_call_id__DoD12(monkeypatch):
    """A call without an id gives ``call_id`` none, and its tool message carries no ``tool_call_id`` (DEC-000-29)."""
    log = _EventLog()
    client = _client()
    sent = _install_rounds(
        monkeypatch, client, stream=False, rounds=[("", [_call("lookup", '{"key": "k"}', None)]), (ANSWER, [])]
    )
    await _run(client, {"lookup": _lookup}, on_tool_event=log)
    assert log.events[0] == _event("start", "lookup", None, '{"key": "k"}')
    assert all(e["call_id"] is None for e in log.events)
    (message,) = _tool_messages(sent[1])
    assert "tool_call_id" not in message
    assert message["content"] == "record:k"


# ---------------------------------------------------------------------------
# DoD-13 — result event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_chat_completions_result_event_output_equals_tool_message__DoD13(monkeypatch):
    """After a successful tool the observer gets ``result`` whose output equals the tool message content."""
    value = {"city": "Zürich", "count": 2}
    log = _EventLog()
    client = _client()
    sent = _install_rounds(monkeypatch, client, stream=False, rounds=[("", [_call("weather", "{}")]), (ANSWER, [])])
    await _run(client, {"weather": lambda: value}, on_tool_event=log)
    content = _told(sent[1])
    assert json.loads(content) == value
    assert log.events == [
        _event("start", "weather", "call_1", "{}"),
        _event("result", "weather", "call_1", "{}", output=content),
    ]


# ---------------------------------------------------------------------------
# DoD-14 — start then error for each failure kind
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
@pytest.mark.parametrize(("name", "arguments", "tools", "kind", "text"), _FAILURE_CASES)
async def test_openai_chat_completions_failure_gives_start_then_error_event__DoD14(
    monkeypatch, stream, name, arguments, tools, kind, text
):
    """Each failure kind gives ``start`` then ``error`` with name, call id, arguments as sent and the kind."""
    log = _EventLog()
    client = _client()
    _install_rounds(monkeypatch, client, stream=stream, rounds=[("", [_call(name, arguments, "call_k")]), (ANSWER, [])])
    result = await _run(client, tools, stream=stream, on_tool_event=log)
    assert result == ANSWER
    assert log.events == [
        _event("start", name, "call_k", arguments),
        _event("error", name, "call_k", arguments, error_kind=kind),
    ]


# ---------------------------------------------------------------------------
# DoD-15 — call order and inline coroutine observer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_chat_completions_events_in_call_order_coroutine_awaited_first__DoD15(monkeypatch):
    """Several calls: events per call in order; a coroutine observer's ``start`` work ends before the tool runs."""
    log: list[str] = []

    async def observer(event: Any) -> None:
        # Yield to the loop first: if the library did not await the observer inline, the tool would run before
        # this append.
        await asyncio.sleep(0)
        log.append(f"observed:{event['type']}:{event['name']}")

    def tool_a() -> str:
        log.append("ran:a")
        return "A"

    def tool_b() -> str:
        log.append("ran:b")
        raise RuntimeError("b broke")

    def tool_c() -> str:
        log.append("ran:c")
        return "C"

    client = _client()
    calls = [_call("a", "{}", "call_a"), _call("b", "{}", "call_b"), _call("c", "{}", "call_c")]
    _install_rounds(monkeypatch, client, stream=False, rounds=[("", calls), (ANSWER, [])])
    result = await _run(client, {"a": tool_a, "b": tool_b, "c": tool_c}, on_tool_event=observer)
    assert result == ANSWER
    assert log == [
        "observed:start:a", "ran:a", "observed:result:a",
        "observed:start:b", "ran:b", "observed:error:b",
        "observed:start:c", "ran:c", "observed:result:c",
    ]


# ---------------------------------------------------------------------------
# DoD-16 — strict: error event before the exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments", "tools", "kind", "expected_exc"),
    [
        pytest.param("ghost", "{}", {"lookup": _lookup}, "not_offered", LLMError, id="not_offered"),
        pytest.param("lookup", "{oops", {"lookup": _lookup}, "bad_arguments", LLMError, id="bad_arguments"),
        pytest.param("lookup", '{"zzz": 1}', {"lookup": _lookup}, "unknown_argument", ValueError,
                     id="unknown_argument"),
        pytest.param("explode", "{}", {"explode": _raising_tool}, "tool_raised", RuntimeError, id="tool_raised"),
        pytest.param("report", "{}", {"report": _unserialisable_tool}, "unserialisable_result", LLMError,
                     id="unserialisable_result"),
    ],
)
async def test_openai_chat_completions_strict_error_event_before_exception__DoD16(
    monkeypatch, name, arguments, tools, kind, expected_exc
):
    """Strict with an observer: the failing call's ``error`` event is received before its exception surfaces."""
    timeline: list[str] = []

    def observer(event: Any) -> None:
        timeline.append(f"{event['type']}:{event['error_kind']}")

    client = _client()
    _install_rounds(monkeypatch, client, stream=False, rounds=[("", [_call(name, arguments)]), (ANSWER, [])])
    try:
        await _run(client, tools, tool_errors="raise", on_tool_event=observer)
    except expected_exc:
        timeline.append("caller-caught")
    assert timeline == ["start:None", f"error:{kind}", "caller-caught"]


# ---------------------------------------------------------------------------
# DoD-17 — strict: observer failure propagates unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [pytest.param(False, id="plain"), pytest.param(True, id="coroutine")])
async def test_openai_chat_completions_strict_observer_fails_on_start_propagates__DoD17(monkeypatch, is_async):
    """Strict: an observer raising on ``start`` ends the call with that same exception; the tool did not run."""
    ran: list[str] = []

    def lookup(key: str = "") -> str:
        ran.append(key)
        return "x"

    observer = _FailingObserver({("start", "lookup")}, is_async=is_async)
    client = _client()
    sent = _install_rounds(monkeypatch, client, stream=False, rounds=[("", [_call("lookup", "{}")]), (ANSWER, [])])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(_ObserverBoom) as exc_info:
            await _run(client, {"lookup": lookup}, tool_errors="raise", on_tool_event=observer)
    assert exc_info.value is observer.raised[0]
    assert ran == []
    assert len(sent) == 1
    # Strict mode: the failure propagates and no observer warning is emitted (interfaces.md § Exposed: warnings).
    assert not [w for w in caught if issubclass(w.category, LLMToolObserverWarning)]


@pytest.mark.asyncio
async def test_openai_chat_completions_strict_observer_fails_on_result_propagates__DoD17(monkeypatch):
    """Strict: an observer raising on ``result`` ends the call with that same exception, unchanged."""
    observer = _FailingObserver({("result", "lookup")})
    client = _client()
    sent = _install_rounds(monkeypatch, client, stream=False, rounds=[("", [_call("lookup", "{}")]), (ANSWER, [])])
    with pytest.raises(_ObserverBoom) as exc_info:
        await _run(client, {"lookup": _lookup}, tool_errors="raise", on_tool_event=observer)
    assert exc_info.value is observer.raised[0]
    assert len(sent) == 1


# ---------------------------------------------------------------------------
# DoD-18 — tolerant: observer fails on start
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [pytest.param(False, id="plain"), pytest.param(True, id="coroutine")])
async def test_openai_chat_completions_tolerant_observer_fails_on_start__DoD18(monkeypatch, is_async):
    """Tolerant, observer raises on ``start``: tool not run, model told it failed, no more events for that call."""
    ran: list[str] = []

    def first() -> str:
        ran.append("first")
        return "one"

    def second() -> str:
        ran.append("second")
        return "two"

    observer = _FailingObserver({("start", "first")}, is_async=is_async)
    client = _client()
    calls = [_call("first", "{}", "call_a"), _call("second", "{}", "call_b")]
    sent = _install_rounds(monkeypatch, client, stream=False, rounds=[("", calls), (ANSWER, [])])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = await _run(client, {"first": first, "second": second}, on_tool_event=observer)
    assert result == ANSWER
    assert ran == ["second"]
    assert _told(sent[1], "call_a") == "Tool 'first' failed."
    assert _told(sent[1], "call_b") == "two"
    assert observer.events == [
        _event("start", "first", "call_a", "{}"),
        _event("start", "second", "call_b", "{}"),
        _event("result", "second", "call_b", "{}", output="two"),
    ]


# ---------------------------------------------------------------------------
# DoD-19 — tolerant: observer fails on result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_chat_completions_tolerant_observer_fails_on_result__DoD19(monkeypatch):
    """Tolerant, observer raises on ``result``: model told ``Tool '<n>' failed.`` instead; no more events for it."""
    observer = _FailingObserver({("result", "lookup")})
    client = _client()
    calls = [_call("lookup", '{"key": "k"}', "call_a"), _call("other", "{}", "call_b")]
    sent = _install_rounds(monkeypatch, client, stream=False, rounds=[("", calls), (ANSWER, [])])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = await _run(client, {"lookup": _lookup, "other": lambda: "fine"}, on_tool_event=observer)
    assert result == ANSWER
    assert _told(sent[1], "call_a") == "Tool 'lookup' failed."
    assert _told(sent[1], "call_b") == "fine"
    assert observer.events == [
        _event("start", "lookup", "call_a", '{"key": "k"}'),
        _event("result", "lookup", "call_a", '{"key": "k"}', output="record:k"),
        _event("start", "other", "call_b", "{}"),
        _event("result", "other", "call_b", "{}", output="fine"),
    ]


# ---------------------------------------------------------------------------
# DoD-20 — tolerant: one warning per observer failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fail_on", "expected_warnings"),
    [
        pytest.param({("start", "a")}, 1, id="one_failure"),
        pytest.param({("start", "a"), ("result", "b")}, 2, id="two_failures"),
        pytest.param({("start", "a"), ("result", "b"), ("error", "c")}, 3, id="three_failures"),
    ],
)
async def test_openai_chat_completions_tolerant_observer_failure_warns_once_each__DoD20(
    monkeypatch, fail_on, expected_warnings
):
    """Tolerant: each observer failure emits exactly one LLMToolObserverWarning; the call still returns its answer."""
    observer = _FailingObserver(fail_on)
    client = _client()
    calls = [_call("a", "{}", "call_a"), _call("b", "{}", "call_b"), _call("c", "{}", "call_c")]
    _install_rounds(monkeypatch, client, stream=False, rounds=[("", calls), (ANSWER, [])])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await _run(
            client, {"a": lambda: "A", "b": lambda: "B", "c": _raising_tool}, on_tool_event=observer
        )
    assert result == ANSWER
    observer_warnings = [w for w in caught if issubclass(w.category, LLMToolObserverWarning)]
    assert len(observer_warnings) == expected_warnings


# ---------------------------------------------------------------------------
# DoD-21 — trace entries of tolerant failures
# ---------------------------------------------------------------------------

# One fixed UTC instant for the frozen trace clock, so the record lands at a known path (data-model § Trace record).
_FIXED_NOW = datetime(2026, 3, 4, 5, 6, 7, 890123, tzinfo=UTC)
_RECORD_RELATIVE = Path("2026-03-04") / "llm-trace-20260304-050607890123.log"

# Section titles of the trace record (docs/architecture/data-model.md § Trace record; DEC-006-7 adds TOOL ERROR).
_HEADINGS = frozenset({"SYSTEM PROMPT", "MESSAGE HISTORY", "REQUEST", "RESPONSE", "CHUNK", "ERROR", "END TRACE"})


class _FrozenClock:
    """Stands in for ``llm.trace.datetime`` so the session starts at _FIXED_NOW (DEC-002-16 clock seam)."""

    @staticmethod
    def now(tz: Any = None) -> datetime:
        return _FIXED_NOW


def _enable_tracing(monkeypatch, folder: Path) -> None:
    """Turn tracing on through the environment and freeze the trace clock; run before building the client."""
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(folder))
    monkeypatch.setattr(llm.trace, "datetime", _FrozenClock)


def _sections(text: str) -> list[tuple[str, str]]:
    """Split a record into ``(title, body text)``; tool headings are ``TOOL CALL|RESULT|ERROR: <name>`` lines."""
    sections: list[tuple[str, list[str]]] = [("HEADER", [])]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) <= {"=", "-"}:
            continue
        if stripped in _HEADINGS or stripped.startswith(("TOOL CALL:", "TOOL RESULT:", "TOOL ERROR:")):
            sections.append((stripped, []))
        else:
            sections[-1][1].append(stripped)
    return [(title, "\n".join(body)) for title, body in sections]


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "arguments", "tools", "kind", "text"), _FAILURE_CASES)
async def test_openai_chat_completions_trace_tolerant_failure_result_then_error__DoD21(
    monkeypatch, tmp_path, name, arguments, tools, kind, text
):
    """Tracing on: a tolerant failure writes ``TOOL RESULT`` with the model text, then ``TOOL ERROR`` with its kind."""
    _enable_tracing(monkeypatch, tmp_path)
    client = _client()
    _install_rounds(monkeypatch, client, stream=False, rounds=[("", [_call(name, arguments)]), (ANSWER, [])])
    assert await _run(client, tools) == ANSWER
    sections = _sections((tmp_path / _RECORD_RELATIVE).read_text(encoding="utf-8"))
    pairs = [
        (body, sections[i + 1] if i + 1 < len(sections) else None)
        for i, (title, body) in enumerate(sections)
        if title == f"TOOL RESULT: {name}"
    ]
    assert pairs, [title for title, _ in sections]
    assert any(
        text in body and following is not None and following[0] == f"TOOL ERROR: {name}" and kind in following[1]
        for body, following in pairs
    ), sections


@pytest.mark.asyncio
async def test_openai_chat_completions_trace_tolerant_observer_failure_note__DoD21(monkeypatch, tmp_path):
    """Tracing on: a tolerant observer failure writes ``TOOL ERROR`` carrying the observer-failure note."""
    _enable_tracing(monkeypatch, tmp_path)
    observer = _FailingObserver({("start", "lookup")})
    client = _client()
    _install_rounds(monkeypatch, client, stream=False, rounds=[("", [_call("lookup", "{}")]), (ANSWER, [])])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert await _run(client, {"lookup": _lookup}, on_tool_event=observer) == ANSWER
    sections = _sections((tmp_path / _RECORD_RELATIVE).read_text(encoding="utf-8"))
    notes = [body for title, body in sections if title == "TOOL ERROR: lookup"]
    # DEC-006-7: the note is ``observer failed: <ExceptionType>: <message>``.
    assert any("observer failed: _ObserverBoom: observer exploded" in note for note in notes), sections


# ---------------------------------------------------------------------------
# DoD-22 — arguments field filling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [pytest.param(_MISSING, id="missing"), pytest.param(None, id="null")])
async def test_openai_chat_completions_event_arguments_empty_without_arguments__DoD22(monkeypatch, arguments):
    """A call carrying no arguments gives the observer ``arguments`` as empty text."""
    log = _EventLog()
    client = _client()
    _install_rounds(monkeypatch, client, stream=False, rounds=[("", [_call("ping", arguments)]), (ANSWER, [])])
    await _run(client, {"ping": lambda: "pong"}, on_tool_event=log)
    assert log.events == [
        _event("start", "ping", "call_1", ""),
        _event("result", "ping", "call_1", "", output="pong"),
    ]


@pytest.mark.asyncio
async def test_openai_chat_completions_event_arguments_json_for_mapping__DoD22(monkeypatch):
    """Arguments arriving as a mapping reach the observer as JSON text, non-ASCII kept (DEC-006-4)."""
    mapping = {"key": "Zürich", "n": 2}
    log = _EventLog()
    client = _client()
    _install_rounds(monkeypatch, client, stream=False, rounds=[("", [_call("find", mapping)]), (ANSWER, [])])
    await _run(client, {"find": lambda key="", n=0: f"{key}:{n}"}, on_tool_event=log)
    expected_text = json.dumps(mapping, ensure_ascii=False)
    assert log.events[0] == _event("start", "find", "call_1", expected_text)
    assert json.loads(log.events[0]["arguments"]) == mapping
