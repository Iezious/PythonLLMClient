"""Unit tests for tolerant / strict tool-call failures and the tool-event observer on the llama-swap
Responses class (plan 006.tolerant-tool-loop, step 007).

Covers ``LlamaSwapAPIClient.chat_with_tools``: the five tool failure kinds sent back to the model with the
fixed model-facing texts by default (DEC-arch-14, DEC-arch-28), the strict opt-out ``tool_errors="raise"``
(DEC-arch-17), the argument-name check and its ``**kwargs`` / unreadable-signature skip (DEC-arch-16),
cancellation, the round limit, the ``on_tool_event`` observer (DEC-arch-18, DEC-arch-26, DEC-006-4,
DEC-006-8), the observer warning and the trace of tolerant failures (DEC-006-7).

The server is faked through the DEC-004-14 seams: a per-round ``_request`` fake for non-streamed rounds,
and a fake ``_stream_post`` plus a per-round ``_iter_sse_data`` queue for streamed rounds. "The model is
told X" is checked as the next round's request carrying a ``function_call_output`` item for that call whose
output is exactly X. Every expected value comes from the step's Definition of done and the cited decisions;
each test names the DoD id it covers. Helpers are module-local (tests/AGENTS.md).
"""

import asyncio
import builtins
import copy
import inspect
import json
import re
import warnings
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import llm.trace
from llm.exceptions import LLMError, LLMToolObserverWarning
from llm.llama_swap import LlamaSwapAPIClient

_BASE_URL = "http://localhost:8080/v1"
_MODEL = "some-model"

# Sentinel for "leave this key out of the call item" in the reply builders.
_MISSING = object()

_ASK = "What is the code for Paris?"
_FINAL_TEXT = "The code is QX-48812."


def _tool_def(name: str, *params: str) -> dict[str, Any]:
    """An app tool definition in the nested chat-completions shape with optional text parameters."""
    properties = {p: {"type": "string"} for p in params}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Tool {name}.",
            "parameters": {"type": "object", "properties": properties},
        },
    }


_LOOKUP_DEFS = [_tool_def("lookup", "city")]

# Model-facing texts (DEC-arch-14), with ``<n>`` the requested tool name.


def _not_available(name: str) -> str:
    return f"Tool '{name}' is not available."


def _invalid_arguments(name: str) -> str:
    return f"Invalid arguments for tool '{name}'."


def _no_argument(name: str, arg: str) -> str:
    return f"Tool '{name}' has no argument '{arg}'."


def _failed(name: str) -> str:
    return f"Tool '{name}' failed."


# ---------------------------------------------------------------------------
# Fixtures, reply builders and fakes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch):
    """Start every test with tracing off so a developer's shell env cannot change behaviour (DEC-000-48)."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _client() -> LlamaSwapAPIClient:
    """A llama-swap Responses client on the fixed test address with the test credential."""
    return LlamaSwapAPIClient(model=_MODEL, base_url=_BASE_URL, bearer_token="sk-test")


def _user(text: str = _ASK) -> dict[str, Any]:
    """A caller user turn."""
    return {"role": "user", "content": text}


def _call(name: Any, arguments: Any, call_id: str) -> dict[str, Any]:
    """A function_call output item in the llama.cpp probe shape; ``_MISSING`` drops the name or arguments."""
    item: dict[str, Any] = {"type": "function_call", "id": f"fc_{call_id}", "call_id": call_id}
    if name is not _MISSING:
        item["name"] = name
    if arguments is not _MISSING:
        item["arguments"] = arguments
    item["status"] = "completed"
    return item


def _message(text: str) -> dict[str, Any]:
    """An assistant message output item holding one output_text part."""
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _reply(*items: dict[str, Any]) -> dict[str, Any]:
    """A completed non-streamed Responses reply with the given output items."""
    return {"id": "resp_1", "object": "response", "status": "completed", "output": list(items)}


_FINAL_ROUND = _reply(_message(_FINAL_TEXT))


def _rounds(monkeypatch, client: Any, replies: list[Any]) -> list[dict[str, Any]]:
    """Replace ``_request`` with a per-round fake (DEC-004-14); returns every captured request.

    Replies and captured bodies are deep-copied so neither side can alter what the other sees. A request
    beyond the canned replies fails the test, which is how "no further round is requested" is detected.
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


def _outputs(request: dict[str, Any]) -> list[tuple[Any, Any]]:
    """The ``(call_id, output)`` of every function_call_output input item a round request carries."""
    return [
        (item.get("call_id"), item.get("output"))
        for item in request["json"]["input"]
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]


def _told(request: dict[str, Any], call_id: str) -> list[Any]:
    """Every output sent for ``call_id`` on that round request (a call must get exactly one)."""
    return [output for cid, output in _outputs(request) if cid == call_id]


class _Tool:
    """A recording tool with an explicit signature (default just ``city``), so it is not a catch-all.

    Remembers the keyword arguments of every run, optionally appends ``label`` to a shared ``log`` (to
    check ordering against observer events), then raises ``error`` when given or returns ``result``.
    """

    def __init__(
        self,
        *params: str,
        result: Any = "ok",
        error: BaseException | None = None,
        log: list[Any] | None = None,
        label: str = "tool",
    ) -> None:
        names = params or ("city",)
        self.__signature__ = inspect.Signature(
            [inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, default=None) for n in names]
        )
        self.result = result
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self._log = log
        self._label = label

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._log is not None:
            self._log.append(self._label)
        if self.error is not None:
            raise self.error
        return self.result


class _Events:
    """A plain-function observer recording every event it is handed (a copy, in order)."""

    def __init__(self, log: list[Any] | None = None) -> None:
        self.events: list[dict[str, Any]] = []
        self._log = log

    def __call__(self, event: Any) -> None:
        self.events.append(dict(event))
        if self._log is not None:
            self._log.append(("event", event["type"], event["call_id"]))

    def of(self, call_id: str) -> list[str]:
        """The event types received for one call, in order."""
        return [e["type"] for e in self.events if e["call_id"] == call_id]


def _event(type_: str, name: str, call_id: str | None, arguments: str, output=None, error_kind=None) -> dict:
    """The full ToolEvent a test expects: all six keys present, none where not applicable (DEC-006-4)."""
    return {
        "type": type_,
        "name": name,
        "call_id": call_id,
        "arguments": arguments,
        "output": output,
        "error_kind": error_kind,
    }


def _observer_warnings(records: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    """Only the library's observer warnings out of a ``catch_warnings(record=True)`` list."""
    return [w for w in records if issubclass(w.category, LLMToolObserverWarning)]


# The five failure kinds as one-round scenarios: (requested name, arguments, tools mapping factory).
# Each factory returns the tools mapping and the recording ``lookup`` tool, so a test can check whether it ran.
_SECRET_TOOL_TEXT = "secret-internal-detail-7781"


def _scenario(kind: str) -> tuple[str, str, dict[str, Any], _Tool]:
    """One failing call of ``kind`` (interfaces.md § Tool failures); the tool name it requests and its args."""
    if kind == "not_offered":
        hidden = _Tool()
        return "hidden_tool", "{}", {"lookup": _Tool(), "hidden_tool": hidden}, hidden
    if kind == "bad_arguments":
        tool = _Tool()
        return "lookup", "{not json", {"lookup": tool}, tool
    if kind == "unknown_argument":
        tool = _Tool()
        return "lookup", '{"town": "Paris"}', {"lookup": tool}, tool
    if kind == "tool_raised":
        tool = _Tool(error=ZeroDivisionError(_SECRET_TOOL_TEXT))
        return "lookup", '{"city": "Paris"}', {"lookup": tool}, tool
    if kind == "unserialisable_result":
        tool = _Tool(result={"when": object()})
        return "lookup", '{"city": "Paris"}', {"lookup": tool}, tool
    raise AssertionError(kind)


_KINDS = ["not_offered", "bad_arguments", "unknown_argument", "tool_raised", "unserialisable_result"]


# ---------------------------------------------------------------------------
# DoD-1 — tool not offered
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_unoffered_tool_told_not_available_and_conversation_continues__DoD1(monkeypatch):
    """DoD-1: an unoffered tool runs no callable (even one present under that name); the model is told
    ``Tool '<n>' is not available.``; the next round runs and its answer is returned (US-064.AC-1, -12, -13,
    -20)."""
    hidden = _Tool()
    lookup = _Tool()
    client = _client()
    captured = _rounds(monkeypatch, client, [_reply(_call("hidden_tool", "{}", "call_1")), _FINAL_ROUND])

    result = await client.chat_with_tools(
        [_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": lookup, "hidden_tool": hidden}
    )

    assert result == _FINAL_TEXT
    assert len(captured) == 2
    assert hidden.calls == []
    assert lookup.calls == []
    assert _told(captured[1], "call_1") == [_not_available("hidden_tool")]


@pytest.mark.asyncio
async def test_llamaswap_empty_tool_name_told_not_available_and_conversation_continues__DoD1(monkeypatch):
    """DoD-1: an empty tool name runs no callable; the model is told ``Tool '' is not available.`` (``<n>``
    is the requested, empty, name); the call returns the next round's answer (US-064.AC-1, -12, -13)."""
    lookup = _Tool()
    client = _client()
    captured = _rounds(monkeypatch, client, [_reply(_call("", "{}", "call_1")), _FINAL_ROUND])

    result = await client.chat_with_tools([_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": lookup})

    assert result == _FINAL_TEXT
    assert lookup.calls == []
    assert _told(captured[1], "call_1") == [_not_available("")]


# ---------------------------------------------------------------------------
# DoD-2 — unreadable arguments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param("{not json", id="not-json"),
        pytest.param('["Paris"]', id="json-list"),
        pytest.param('"Paris"', id="json-string"),
        pytest.param("42", id="json-number"),
    ],
)
async def test_llamaswap_unreadable_arguments_told_invalid_and_conversation_continues__DoD2(monkeypatch, arguments):
    """DoD-2: arguments that are not JSON, or JSON that is not an object, run no callable; the model is told
    ``Invalid arguments for tool '<n>'.``; the conversation continues (US-064.AC-2, -14, -15)."""
    lookup = _Tool()
    client = _client()
    captured = _rounds(monkeypatch, client, [_reply(_call("lookup", arguments, "call_1")), _FINAL_ROUND])

    result = await client.chat_with_tools([_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": lookup})

    assert result == _FINAL_TEXT
    assert lookup.calls == []
    assert _told(captured[1], "call_1") == [_invalid_arguments("lookup")]


# ---------------------------------------------------------------------------
# DoD-3 — unknown argument name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_unknown_argument_told_first_undeclared_name_in_sent_order__DoD3(monkeypatch):
    """DoD-3: an undeclared argument name runs no callable; the model is told ``Tool '<n>' has no argument
    '<arg>'.`` naming the first undeclared name in the order the model sent them (not sorted); the
    conversation continues (US-064.AC-3, -16, DEC-arch-16)."""
    lookup = _Tool("city")
    client = _client()
    arguments = '{"city": "Paris", "zeta": 1, "alpha": 2}'
    captured = _rounds(monkeypatch, client, [_reply(_call("lookup", arguments, "call_1")), _FINAL_ROUND])

    result = await client.chat_with_tools([_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": lookup})

    assert result == _FINAL_TEXT
    assert lookup.calls == []
    assert _told(captured[1], "call_1") == [_no_argument("lookup", "zeta")]


# ---------------------------------------------------------------------------
# DoD-4 — the tool raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("flavour", ["sync", "async"])
async def test_llamaswap_raising_tool_told_failed_without_exception_text__DoD4(monkeypatch, flavour):
    """DoD-4: a sync or async tool raising an ordinary exception → the model is told ``Tool '<n>' failed.``,
    nothing of the exception message is sent, and the conversation continues (US-064.AC-4, -17)."""
    runs: list[str] = []

    def sync_lookup(city: str) -> str:
        runs.append(city)
        raise ValueError(_SECRET_TOOL_TEXT)

    async def async_lookup(city: str) -> str:
        runs.append(city)
        raise ValueError(_SECRET_TOOL_TEXT)

    tool = sync_lookup if flavour == "sync" else async_lookup
    client = _client()
    captured = _rounds(
        monkeypatch, client, [_reply(_call("lookup", '{"city": "Paris"}', "call_1")), _FINAL_ROUND]
    )

    result = await client.chat_with_tools([_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": tool})

    assert result == _FINAL_TEXT
    assert runs == ["Paris"]
    assert _told(captured[1], "call_1") == [_failed("lookup")]
    assert _SECRET_TOOL_TEXT not in json.dumps(captured[1]["json"])


# ---------------------------------------------------------------------------
# DoD-5 — the result cannot be turned into text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result_value",
    [pytest.param({"when": object()}, id="mapping"), pytest.param([object()], id="list")],
)
async def test_llamaswap_unserialisable_result_told_failed_and_conversation_continues__DoD5(
    monkeypatch, result_value
):
    """DoD-5: a mapping / list result that cannot be turned into JSON text → the model is told
    ``Tool '<n>' failed.``; the conversation continues (US-064.AC-5, -18)."""
    tool = _Tool(result=result_value)
    client = _client()
    captured = _rounds(
        monkeypatch, client, [_reply(_call("lookup", '{"city": "Paris"}', "call_1")), _FINAL_ROUND]
    )

    result = await client.chat_with_tools([_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": tool})

    assert result == _FINAL_TEXT
    assert len(tool.calls) == 1
    assert _told(captured[1], "call_1") == [_failed("lookup")]


# ---------------------------------------------------------------------------
# DoD-6 — several calls in one round, the middle one fails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_middle_failure_other_calls_run_in_order_each_gets_output__DoD6(monkeypatch):
    """DoD-6: three calls where the middle one fails → the first and third tools run in the order requested,
    and every call gets its own function-call output, in that order (US-064.AC-6)."""
    log: list[Any] = []
    first = _Tool("key", result="first-result", log=log, label="first")
    middle = _Tool("key", error=RuntimeError("middle broke"), log=log, label="middle")
    third = _Tool("key", result="third-result", log=log, label="third")
    defs = [_tool_def("first", "key"), _tool_def("middle", "key"), _tool_def("third", "key")]
    client = _client()
    captured = _rounds(
        monkeypatch,
        client,
        [
            _reply(
                _call("first", '{"key": "a"}', "call_1"),
                _call("middle", '{"key": "b"}', "call_2"),
                _call("third", '{"key": "c"}', "call_3"),
            ),
            _FINAL_ROUND,
        ],
    )

    result = await client.chat_with_tools(
        [_user()], tools_definitions=defs, tools={"first": first, "middle": middle, "third": third}
    )

    assert result == _FINAL_TEXT
    assert log == ["first", "middle", "third"]
    assert first.calls == [{"key": "a"}]
    assert third.calls == [{"key": "c"}]
    assert _outputs(captured[1]) == [
        ("call_1", "first-result"),
        ("call_2", _failed("middle")),
        ("call_3", "third-result"),
    ]


# ---------------------------------------------------------------------------
# DoD-7 — four-round tolerant conversation, streamed and not
# ---------------------------------------------------------------------------

# Streamed round vocabulary (DEC-004-9, DEC-003-13), llama.cpp shapes: each round is a list of SSE payloads.


def _ev(event_type: str, **fields: Any) -> str:
    """One stream payload string: a JSON event object with ``type`` and the given fields."""
    return json.dumps({"type": event_type, **fields})


_CREATED = _ev(
    "response.created", response={"id": "resp_1", "object": "response", "status": "in_progress", "output": []}
)


def _completed(*output: dict[str, Any]) -> str:
    """The completed event carrying the round's full output."""
    return _ev("response.completed", response=_reply(*output))


def _stream_call_round(item_id: str, call_id: str, name: str, arguments: str) -> list[str]:
    """A streamed round with one function call at position 0 (added, one arguments delta, done, completed)."""
    added = {"type": "function_call", "id": item_id, "call_id": call_id, "name": name, "arguments": "",
             "status": "in_progress"}
    done = {"type": "function_call", "id": item_id, "call_id": call_id, "name": name, "arguments": arguments,
            "status": "completed"}
    return [
        _CREATED,
        _ev("response.output_item.added", output_index=0, item=added),
        _ev("response.function_call_arguments.delta", item_id=item_id, output_index=0, delta=arguments),
        _ev("response.output_item.done", output_index=0, item=done),
        _completed(done),
    ]


def _stream_answer_round(text: str) -> list[str]:
    """A streamed final round: one assistant message at position 0 whose text arrives in one delta."""
    added = {"type": "message", "id": "msg_9", "role": "assistant", "status": "in_progress", "content": []}
    done = _message(text) | {"id": "msg_9"}
    return [
        _CREATED,
        _ev("response.output_item.added", output_index=0, item=added),
        _ev("response.output_text.delta", item_id="msg_9", output_index=0, content_index=0, delta=text),
        _ev("response.output_item.done", output_index=0, item=done),
        _completed(done),
    ]


def _stream_rounds(monkeypatch, client: Any, rounds: list[list[str]]) -> list[dict[str, Any]]:
    """Install the streamed seams (DEC-004-14): a fake ``_stream_post`` recording each body under ``json``
    (deep-copied) and a fake ``_iter_sse_data`` yielding one scripted round per read; ``_request`` is
    forbidden so streamed rounds cannot fall back to the non-streamed seam."""
    pending = [list(r) for r in rounds]
    captured: list[dict[str, Any]] = []

    @asynccontextmanager
    async def fake_stream_post(path, payload):
        captured.append({"path": path, "json": copy.deepcopy(payload)})
        yield None

    async def fake_iter_sse_data(_response):
        if not pending:
            pytest.fail("the stream of an unscripted round was read")
        for payload in pending.pop(0):
            yield payload

    async def forbidden_request(method, path, **kw):
        pytest.fail(f"a streamed tool round sent a non-streamed request: {method} {path}")

    monkeypatch.setattr(client, "_stream_post", fake_stream_post)
    monkeypatch.setattr(client, "_iter_sse_data", fake_iter_sse_data)
    monkeypatch.setattr(client, "_request", forbidden_request)
    return captured


# The three failing rounds of DoD-7: (call id, requested name, arguments).
_FOUR_ROUND_CALLS = [
    ("call_1", "secret_tool", "{}"),
    ("call_2", "fetch_record", '["alpha-7"]'),
    ("call_3", "fetch_record", '{"nope": "x"}'),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [pytest.param(False, id="non_streamed"), pytest.param(True, id="streamed")])
async def test_llamaswap_four_round_tolerant_conversation_returns_round_four_answer__DoD7(monkeypatch, stream):
    """DoD-7: round 1 an unoffered tool, round 2 non-object arguments, round 3 an unknown argument, round 4
    the answer → the call returns round 4's answer without an error, streamed and not; each failure is told
    to the model on the next round (US-064.AC-7)."""
    fetch = _Tool("key", result="record")
    client = _client()
    if stream:
        rounds = [_stream_call_round(f"fc_{cid}", cid, name, args) for cid, name, args in _FOUR_ROUND_CALLS]
        captured = _stream_rounds(monkeypatch, client, [*rounds, _stream_answer_round(_FINAL_TEXT)])
    else:
        replies = [_reply(_call(name, args, cid)) for cid, name, args in _FOUR_ROUND_CALLS]
        captured = _rounds(monkeypatch, client, [*replies, _FINAL_ROUND])

    result = await client.chat_with_tools(
        [_user()],
        tools_definitions=[_tool_def("fetch_record", "key")],
        tools={"fetch_record": fetch},
        stream=stream,
    )

    assert result == _FINAL_TEXT
    assert len(captured) == 4
    assert fetch.calls == []
    assert _told(captured[1], "call_1") == [_not_available("secret_tool")]
    assert _told(captured[2], "call_2") == [_invalid_arguments("fetch_record")]
    assert _told(captured[3], "call_3") == [_no_argument("fetch_record", "nope")]


# ---------------------------------------------------------------------------
# DoD-8 — round limit with a failing call every round
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_failing_call_every_round_ends_with_round_limit_error__DoD8(monkeypatch):
    """DoD-8: a model that requests a failing call in every round ends with the round-limit RuntimeError
    stating the limit (max_loops) and its value (US-064.AC-8)."""
    client = _client()
    captured = _rounds(
        monkeypatch,
        client,
        [_reply(_call("hidden_tool", "{}", "call_1")), _reply(_call("hidden_tool", "{}", "call_2"))],
    )

    with pytest.raises(RuntimeError) as excinfo:
        await client.chat_with_tools(
            [_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool()}, max_loops=2
        )

    message = str(excinfo.value)
    assert len(captured) == 2
    assert "max_loops" in message
    assert re.search(r"(?<!\d)2(?!\d)", message)


# ---------------------------------------------------------------------------
# DoD-9 — cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("flavour", ["sync", "async"])
@pytest.mark.parametrize("with_observer", [pytest.param(False, id="no-observer"), pytest.param(True, id="observer")])
async def test_llamaswap_cancellation_propagates_without_output_or_error_event__DoD9(
    monkeypatch, flavour, with_observer
):
    """DoD-9: a tool raising cancellation → the cancellation propagates, no further round is requested (so no
    function-call output is sent for it), and an observer gets no ``error`` event for it (US-064.AC-9, -19)."""

    def sync_lookup(city: str) -> str:
        raise asyncio.CancelledError()

    async def async_lookup(city: str) -> str:
        raise asyncio.CancelledError()

    tool = sync_lookup if flavour == "sync" else async_lookup
    observer = _Events()
    client = _client()
    captured = _rounds(
        monkeypatch, client, [_reply(_call("lookup", '{"city": "Paris"}', "call_1")), _FINAL_ROUND]
    )
    extra = {"on_tool_event": observer} if with_observer else {}

    with pytest.raises(asyncio.CancelledError):
        await client.chat_with_tools([_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": tool}, **extra)

    assert len(captured) == 1
    assert _outputs(captured[0]) == []
    assert all(e["type"] != "error" for e in observer.events)


# ---------------------------------------------------------------------------
# DoD-10 — strict mode keeps today's exceptions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["not_offered", "bad_arguments", "unserialisable_result"])
async def test_llamaswap_strict_llmerror_kinds_raise_llmerror_kind_tool__DoD10(monkeypatch, kind):
    """DoD-10: with ``tool_errors="raise"``, not_offered / bad_arguments / unserialisable_result end the call
    with LLMError naming the tool and ``kind == "tool"``; no further round is requested (US-064.AC-10,
    US-067.AC-2)."""
    name, arguments, tools, _ = _scenario(kind)
    client = _client()
    captured = _rounds(monkeypatch, client, [_reply(_call(name, arguments, "call_1")), _FINAL_ROUND])

    with pytest.raises(LLMError) as excinfo:
        await client.chat_with_tools([_user()], tools_definitions=_LOOKUP_DEFS, tools=tools, tool_errors="raise")

    assert name in str(excinfo.value)
    assert excinfo.value.kind == "tool"
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_llamaswap_strict_empty_name_raises_llmerror_kind_tool__DoD10(monkeypatch):
    """DoD-10: with ``tool_errors="raise"``, an empty tool name (not_offered) ends with LLMError of
    ``kind == "tool"``; no tool runs and no further round is requested (US-064.AC-10, US-067.AC-2)."""
    lookup = _Tool()
    client = _client()
    captured = _rounds(monkeypatch, client, [_reply(_call("", "{}", "call_1")), _FINAL_ROUND])

    with pytest.raises(LLMError) as excinfo:
        await client.chat_with_tools(
            [_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": lookup}, tool_errors="raise"
        )

    assert excinfo.value.kind == "tool"
    assert lookup.calls == []
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_llamaswap_strict_unknown_argument_raises_valueerror__DoD10(monkeypatch):
    """DoD-10: with ``tool_errors="raise"``, an unknown argument name ends with ValueError; the tool does not
    run and no further round is requested (US-064.AC-10)."""
    name, arguments, tools, tool = _scenario("unknown_argument")
    client = _client()
    captured = _rounds(monkeypatch, client, [_reply(_call(name, arguments, "call_1")), _FINAL_ROUND])

    with pytest.raises(ValueError):
        await client.chat_with_tools([_user()], tools_definitions=_LOOKUP_DEFS, tools=tools, tool_errors="raise")

    assert tool.calls == []
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_llamaswap_strict_raising_tool_raises_runtimeerror_chained__DoD10(monkeypatch):
    """DoD-10: with ``tool_errors="raise"``, a raising tool ends with RuntimeError naming the tool, chained
    from the tool's own exception; no further round is requested (US-064.AC-10)."""
    original = ZeroDivisionError("boom inside the tool")
    tool = _Tool(error=original)
    client = _client()
    captured = _rounds(
        monkeypatch, client, [_reply(_call("lookup", '{"city": "Paris"}', "call_1")), _FINAL_ROUND]
    )

    with pytest.raises(RuntimeError) as excinfo:
        await client.chat_with_tools(
            [_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": tool}, tool_errors="raise"
        )

    assert "lookup" in str(excinfo.value)
    assert excinfo.value.__cause__ is original
    assert len(captured) == 1


# ---------------------------------------------------------------------------
# DoD-11 — catch-all and unreadable-signature callables
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [pytest.param(None, id="default"), "result", "raise"])
async def test_llamaswap_kwargs_callable_runs_with_exactly_sent_arguments__DoD11(monkeypatch, mode):
    """DoD-11: a callable declaring ``**kwargs`` runs with exactly the arguments the model sent, tolerant and
    strict, and the call returns its answer (US-066.AC-1, DEC-arch-16)."""
    seen: list[dict[str, Any]] = []

    def catch_all(**kwargs: Any) -> str:
        seen.append(kwargs)
        return "done"

    client = _client()
    arguments = '{"anything": 1, "other": "x"}'
    _rounds(monkeypatch, client, [_reply(_call("catch_all", arguments, "call_1")), _FINAL_ROUND])
    extra = {} if mode is None else {"tool_errors": mode}

    result = await client.chat_with_tools(
        [_user()], tools_definitions=[_tool_def("catch_all")], tools={"catch_all": catch_all}, **extra
    )

    assert result == _FINAL_TEXT
    assert seen == [{"anything": 1, "other": "x"}]


class _Unreadable:
    """A callable whose signature cannot be read: ``inspect.signature`` fails on it (its ``__signature__``
    lookup raises), so the argument-name check must be skipped (DEC-arch-16)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def __signature__(self) -> Any:
        raise ValueError("no signature available")

    def __call__(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "read"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [pytest.param(None, id="default"), "raise"])
async def test_llamaswap_unreadable_signature_callable_called_without_name_check__DoD11(monkeypatch, mode):
    """DoD-11: a callable whose signature cannot be read is called without the name check, with the
    arguments the model sent (US-066.AC-1, DEC-arch-16)."""
    with pytest.raises(ValueError):
        inspect.signature(_Unreadable())  # precondition: the signature really is unreadable
    tool = _Unreadable()
    client = _client()
    captured = _rounds(
        monkeypatch, client, [_reply(_call("opaque", '{"free": 1, "form": "y"}', "call_1")), _FINAL_ROUND]
    )
    extra = {} if mode is None else {"tool_errors": mode}

    result = await client.chat_with_tools(
        [_user()], tools_definitions=[_tool_def("opaque")], tools={"opaque": tool}, **extra
    )

    assert result == _FINAL_TEXT
    assert tool.calls == [{"free": 1, "form": "y"}]
    assert _told(captured[1], "call_1") == ["read"]


# ---------------------------------------------------------------------------
# DoD-12, DoD-13 — start and result events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_observer_gets_start_event_before_tool_runs__DoD12(monkeypatch):
    """DoD-12: before the tool runs the observer gets a ``start`` event with the requested name, the call's
    Responses call id and the argument string verbatim; ``output`` and ``error_kind`` none (US-065.AC-1)."""
    log: list[Any] = []
    tool = _Tool(result="ok", log=log, label="tool")
    observer = _Events(log)
    arguments = '{ "city" :   "Paris-7731" }'
    client = _client()
    _rounds(monkeypatch, client, [_reply(_call("lookup", arguments, "call_aa1")), _FINAL_ROUND])

    await client.chat_with_tools(
        [_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": tool}, on_tool_event=observer
    )

    assert observer.events[0] == _event("start", "lookup", "call_aa1", arguments)
    assert log.index(("event", "start", "call_aa1")) < log.index("tool")


@pytest.mark.asyncio
async def test_llamaswap_observer_result_event_output_equals_text_sent__DoD13(monkeypatch):
    """DoD-13: after a successful tool the observer gets a ``result`` event whose ``output`` equals the
    output text sent to the model (US-065.AC-2)."""
    tool = _Tool(result={"code": "QX-1", "note": "Zürich"})
    observer = _Events()
    arguments = '{"city": "Paris"}'
    client = _client()
    captured = _rounds(monkeypatch, client, [_reply(_call("lookup", arguments, "call_1")), _FINAL_ROUND])

    await client.chat_with_tools(
        [_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": tool}, on_tool_event=observer
    )

    sent = _told(captured[1], "call_1")
    assert len(sent) == 1
    assert [e["type"] for e in observer.events] == ["start", "result"]
    assert observer.events[1] == _event("result", "lookup", "call_1", arguments, output=sent[0])


# ---------------------------------------------------------------------------
# DoD-14 — start then error for each failure kind
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", _KINDS)
async def test_llamaswap_observer_gets_start_then_error_with_kind__DoD14(monkeypatch, kind):
    """DoD-14: for each failure kind the observer gets ``start`` then an ``error`` event with the name, call
    id, arguments as sent and the matching ``error_kind`` — including not_offered and bad_arguments, where no
    tool runs (US-065.AC-3)."""
    name, arguments, tools, _ = _scenario(kind)
    observer = _Events()
    client = _client()
    _rounds(monkeypatch, client, [_reply(_call(name, arguments, "call_1")), _FINAL_ROUND])

    result = await client.chat_with_tools(
        [_user()], tools_definitions=_LOOKUP_DEFS, tools=tools, on_tool_event=observer
    )

    assert result == _FINAL_TEXT
    assert observer.events == [
        _event("start", name, "call_1", arguments),
        _event("error", name, "call_1", arguments, error_kind=kind),
    ]


# ---------------------------------------------------------------------------
# DoD-15 — event order and awaited coroutine observer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_events_arrive_in_call_order__DoD15(monkeypatch):
    """DoD-15: with several calls, events arrive in call order: ``start`` then ``result`` / ``error`` per
    call (US-065.AC-4)."""
    observer = _Events()
    lookup = _Tool(result="ok")
    client = _client()
    _rounds(
        monkeypatch,
        client,
        [
            _reply(
                _call("lookup", '{"city": "A"}', "call_1"),
                _call("hidden_tool", "{}", "call_2"),
                _call("lookup", '{"city": "C"}', "call_3"),
            ),
            _FINAL_ROUND,
        ],
    )

    await client.chat_with_tools(
        [_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": lookup}, on_tool_event=observer
    )

    assert [(e["type"], e["call_id"]) for e in observer.events] == [
        ("start", "call_1"),
        ("result", "call_1"),
        ("start", "call_2"),
        ("error", "call_2"),
        ("start", "call_3"),
        ("result", "call_3"),
    ]


@pytest.mark.asyncio
async def test_llamaswap_coroutine_observer_start_work_finishes_before_tool_runs__DoD15(monkeypatch):
    """DoD-15: a coroutine observer's work for ``start`` has finished before the tool is called — it
    suspends before recording, so recording first proves it was awaited inline (US-065.AC-5, DEC-arch-18)."""
    log: list[Any] = []

    async def observer(event: Any) -> None:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        log.append(("done", event["type"], event["call_id"]))

    lookup = _Tool(result="ok", log=log, label="tool")
    client = _client()
    two_calls = _reply(_call("lookup", '{"city": "A"}', "call_1"), _call("lookup", '{"city": "B"}', "call_2"))
    _rounds(monkeypatch, client, [two_calls, _FINAL_ROUND])

    result = await client.chat_with_tools(
        [_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": lookup}, on_tool_event=observer
    )

    assert result == _FINAL_TEXT
    assert log == [
        ("done", "start", "call_1"),
        "tool",
        ("done", "result", "call_1"),
        ("done", "start", "call_2"),
        "tool",
        ("done", "result", "call_2"),
    ]


# ---------------------------------------------------------------------------
# DoD-16, DoD-17 — strict mode with an observer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", _KINDS)
async def test_llamaswap_strict_error_event_received_before_exception__DoD16(monkeypatch, kind):
    """DoD-16: with ``tool_errors="raise"`` and an observer, the failing call's ``error`` event has been
    received by the time the call's exception reaches the caller (US-065.AC-6, DEC-arch-17)."""
    name, arguments, tools, _ = _scenario(kind)
    observer = _Events()
    client = _client()
    _rounds(monkeypatch, client, [_reply(_call(name, arguments, "call_1")), _FINAL_ROUND])
    seen_at_raise: list[dict[str, Any]] = []

    with pytest.raises((LLMError, ValueError, RuntimeError)):
        try:
            await client.chat_with_tools(
                [_user()], tools_definitions=_LOOKUP_DEFS, tools=tools, tool_errors="raise", on_tool_event=observer
            )
        except Exception:
            seen_at_raise.extend(observer.events)
            raise

    assert seen_at_raise == [
        _event("start", name, "call_1", arguments),
        _event("error", name, "call_1", arguments, error_kind=kind),
    ]


class _ObserverBoom(Exception):
    """The observer's own failure, distinct from anything the library raises."""


@pytest.mark.asyncio
@pytest.mark.parametrize("flavour", ["plain", "coroutine"])
async def test_llamaswap_strict_observer_failing_on_start_propagates_and_tool_not_run__DoD17(monkeypatch, flavour):
    """DoD-17: with ``tool_errors="raise"``, an observer (plain or coroutine) raising on ``start`` ends the
    call with that same exception unchanged; the tool did not run and no further round is requested
    (US-065.AC-7)."""
    boom = _ObserverBoom("observer broke on start")

    def plain(event: Any) -> None:
        if event["type"] == "start":
            raise boom

    async def coroutine(event: Any) -> None:
        if event["type"] == "start":
            raise boom

    tool = _Tool()
    client = _client()
    captured = _rounds(
        monkeypatch, client, [_reply(_call("lookup", '{"city": "Paris"}', "call_1")), _FINAL_ROUND]
    )

    with pytest.raises(_ObserverBoom) as excinfo:
        await client.chat_with_tools(
            [_user()],
            tools_definitions=_LOOKUP_DEFS,
            tools={"lookup": tool},
            tool_errors="raise",
            on_tool_event=plain if flavour == "plain" else coroutine,
        )

    assert excinfo.value is boom
    assert tool.calls == []
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_llamaswap_strict_observer_failing_on_result_propagates_unchanged__DoD17(monkeypatch):
    """DoD-17: with ``tool_errors="raise"``, an observer raising on ``result`` ends the call with that same
    exception unchanged; no further round is requested (US-065.AC-7)."""
    boom = _ObserverBoom("observer broke on result")

    def observer(event: Any) -> None:
        if event["type"] == "result":
            raise boom

    tool = _Tool()
    client = _client()
    captured = _rounds(
        monkeypatch, client, [_reply(_call("lookup", '{"city": "Paris"}', "call_1")), _FINAL_ROUND]
    )

    with pytest.raises(_ObserverBoom) as excinfo:
        await client.chat_with_tools(
            [_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": tool}, tool_errors="raise",
            on_tool_event=observer,
        )

    assert excinfo.value is boom
    assert tool.calls == [{"city": "Paris"}]
    assert len(captured) == 1


# ---------------------------------------------------------------------------
# DoD-18, DoD-19, DoD-20 — tolerant observer failures
# ---------------------------------------------------------------------------


def _failing_observer(fail_type: str, fail_call: str) -> tuple[_Events, Any]:
    """An observer recording every event it receives, raising on the ``fail_type`` event of ``fail_call``."""
    recorder = _Events()

    def observer(event: Any) -> None:
        recorder(event)
        if event["type"] == fail_type and event["call_id"] == fail_call:
            raise _ObserverBoom(f"observer broke on {fail_type}")

    return recorder, observer


_TWO_CALLS = _reply(_call("lookup", '{"city": "A"}', "call_1"), _call("lookup", '{"city": "B"}', "call_2"))


@pytest.mark.asyncio
async def test_llamaswap_tolerant_observer_failing_on_start_tool_not_run_told_failed__DoD18(monkeypatch):
    """DoD-18: tolerant, an observer raising on ``start`` → that tool does not run, the model is told
    ``Tool '<n>' failed.``, the conversation continues, the observer gets no further event for that call,
    and the later call still gets its events (US-065.AC-8, -11, -12, DEC-arch-26)."""
    recorder, observer = _failing_observer("start", "call_1")
    lookup = _Tool(result="lookup-result")
    client = _client()
    captured = _rounds(monkeypatch, client, [_TWO_CALLS, _FINAL_ROUND])

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = await client.chat_with_tools(
            [_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": lookup}, on_tool_event=observer
        )

    assert result == _FINAL_TEXT
    assert lookup.calls == [{"city": "B"}]
    assert _outputs(captured[1]) == [("call_1", _failed("lookup")), ("call_2", "lookup-result")]
    assert recorder.of("call_1") == ["start"]
    assert recorder.of("call_2") == ["start", "result"]


@pytest.mark.asyncio
async def test_llamaswap_tolerant_observer_failing_on_result_told_failed_instead_of_result__DoD19(monkeypatch):
    """DoD-19: tolerant, an observer raising on ``result`` → the model is told ``Tool '<n>' failed.`` instead
    of the tool's result (DEC-006-8), with no further event for that call; the later call is unaffected
    (US-065.AC-11, -12, DEC-arch-26)."""
    recorder, observer = _failing_observer("result", "call_1")
    lookup = _Tool(result="lookup-result")
    client = _client()
    captured = _rounds(monkeypatch, client, [_TWO_CALLS, _FINAL_ROUND])

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = await client.chat_with_tools(
            [_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": lookup}, on_tool_event=observer
        )

    assert result == _FINAL_TEXT
    assert lookup.calls == [{"city": "A"}, {"city": "B"}]
    assert _outputs(captured[1]) == [("call_1", _failed("lookup")), ("call_2", "lookup-result")]
    assert recorder.of("call_1") == ["start", "result"]
    assert recorder.of("call_2") == ["start", "result"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_type", ["start", "result"])
async def test_llamaswap_tolerant_observer_failure_emits_exactly_one_warning__DoD20(monkeypatch, fail_type):
    """DoD-20: tolerant, one observer failure (on ``start`` or on ``result``) emits exactly one
    LLMToolObserverWarning and the call still returns its answer (US-065.AC-9, DEC-006-8)."""
    _, observer = _failing_observer(fail_type, "call_1")
    client = _client()
    _rounds(monkeypatch, client, [_TWO_CALLS, _FINAL_ROUND])

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = await client.chat_with_tools(
            [_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool()}, on_tool_event=observer
        )

    assert result == _FINAL_TEXT
    assert len(_observer_warnings(records)) == 1


@pytest.mark.asyncio
async def test_llamaswap_tolerant_each_observer_failure_emits_its_own_warning__DoD20(monkeypatch):
    """DoD-20: tolerant, two observer failures in one call (one per tool call) emit exactly two
    LLMToolObserverWarnings — one per failure, no once-per-client rule (US-065.AC-9, DEC-006-8)."""

    def observer(event: Any) -> None:
        if event["type"] == "start":
            raise _ObserverBoom("observer broke")

    client = _client()
    _rounds(monkeypatch, client, [_TWO_CALLS, _FINAL_ROUND])

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = await client.chat_with_tools(
            [_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool()}, on_tool_event=observer
        )

    assert result == _FINAL_TEXT
    assert len(_observer_warnings(records)) == 2


@pytest.mark.asyncio
async def test_llamaswap_tolerant_observer_failing_on_error_event_emits_one_warning__DoD20(monkeypatch):
    """DoD-20: tolerant, an observer raising on the ``error`` event of a failing call emits exactly one
    LLMToolObserverWarning and the call still returns its answer (US-065.AC-9, DEC-006-8)."""

    def observer(event: Any) -> None:
        if event["type"] == "error":
            raise _ObserverBoom("observer broke on error")

    client = _client()
    _rounds(monkeypatch, client, [_reply(_call("hidden_tool", "{}", "call_1")), _FINAL_ROUND])

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = await client.chat_with_tools(
            [_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool()}, on_tool_event=observer
        )

    assert result == _FINAL_TEXT
    assert len(_observer_warnings(records)) == 1


# ---------------------------------------------------------------------------
# DoD-21 — trace of tolerant failures
# ---------------------------------------------------------------------------

# One fixed UTC instant for the frozen trace clock (the ``llm.trace.datetime`` seam, DEC-002-16).
_FIXED_NOW = datetime(2026, 3, 4, 5, 6, 7, 890123, tzinfo=UTC)
# Captured before any test patches ``llm.trace.open`` so the pass-through can still create real files.
_REAL_OPEN = builtins.open
# Section titles of the trace record (data-model.md § Trace record) and the tool-entry prefixes (DEC-006-7).
_HEADINGS = frozenset({"SYSTEM PROMPT", "MESSAGE HISTORY", "REQUEST", "RESPONSE", "CHUNK", "ERROR", "END TRACE"})
_TOOL_PREFIXES = ("TOOL CALL:", "TOOL RESULT:", "TOOL ERROR:")


class _FrozenClock:
    """Stands in for ``llm.trace.datetime`` so the record starts at a fixed instant (DEC-002-16)."""

    @staticmethod
    def now(tz: Any = None) -> datetime:
        return _FIXED_NOW


def _enable_tracing(monkeypatch, folder: Path) -> list[str]:
    """Turn tracing on through the environment (read when the client is built, so call this first), freeze
    the trace clock and route the tracer's ``open`` through a pass-through recording each opened path."""
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(folder))
    monkeypatch.setattr(llm.trace, "datetime", _FrozenClock)
    opened: list[str] = []

    def passthrough_open(file, mode="r", *args, **kwargs):
        opened.append(str(file))
        return _REAL_OPEN(file, mode, *args, **kwargs)

    # raising=False: llm.trace has no module-level ``open`` attribute; the patch shadows the builtin there.
    monkeypatch.setattr(llm.trace, "open", passthrough_open, raising=False)
    return opened


def _only_record(root: Path) -> str:
    """The text of the single trace record written under ``root``."""
    records = sorted(root.rglob("*.log"))
    assert len(records) == 1, records
    return records[0].read_text(encoding="utf-8")


def _sections(text: str) -> list[tuple[str, str]]:
    """Split a record into ``(title, body text)``; lines before the first section form ``"HEADER"``.

    A section starts at a line that is a known heading or starts with a tool-entry prefix; blank lines and
    pure ``=`` / ``-`` rule lines are layout and skipped.
    """
    sections: list[tuple[str, list[str]]] = [("HEADER", [])]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) <= {"=", "-"}:
            continue
        if stripped in _HEADINGS or stripped.startswith(_TOOL_PREFIXES):
            sections.append((stripped, []))
        else:
            sections[-1][1].append(stripped)
    return [(title, "\n".join(body)) for title, body in sections]


def _result_then_error(text: str, name: str, model_text: str) -> str:
    """Find the ``TOOL RESULT: <name>`` section carrying ``model_text`` and return the body of the section
    right after it, asserting that section is ``TOOL ERROR: <name>`` (DEC-006-7 order)."""
    sections = _sections(text)
    for index, (title, body) in enumerate(sections):
        if title.startswith(f"TOOL RESULT: {name}") and model_text in body:
            assert index + 1 < len(sections), "no section follows the tool result"
            next_title, next_body = sections[index + 1]
            assert next_title.startswith(f"TOOL ERROR: {name}"), next_title
            return next_body
    raise AssertionError(f"no TOOL RESULT: {name} section carrying {model_text!r}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "model_text"),
    [
        pytest.param("not_offered", _not_available("hidden_tool"), id="not_offered"),
        pytest.param("bad_arguments", _invalid_arguments("lookup"), id="bad_arguments"),
        pytest.param("unknown_argument", _no_argument("lookup", "town"), id="unknown_argument"),
        pytest.param("tool_raised", _failed("lookup"), id="tool_raised"),
        pytest.param("unserialisable_result", _failed("lookup"), id="unserialisable_result"),
    ],
)
async def test_llamaswap_traced_tolerant_failure_writes_result_then_error_kind__DoD21(
    monkeypatch, tmp_path, kind, model_text
):
    """DoD-21: with tracing on, a tolerant tool failure writes ``TOOL RESULT`` with the model-facing text,
    followed by ``TOOL ERROR`` with the failure kind name (US-065.AC-13, DEC-006-7)."""
    opened = _enable_tracing(monkeypatch, tmp_path)
    name, arguments, tools, _ = _scenario(kind)
    client = _client()
    _rounds(monkeypatch, client, [_reply(_call(name, arguments, "call_1")), _FINAL_ROUND])

    result = await client.chat_with_tools([_user()], tools_definitions=_LOOKUP_DEFS, tools=tools)

    assert result == _FINAL_TEXT
    assert opened
    note = _result_then_error(_only_record(tmp_path), name, model_text)
    assert kind in note


@pytest.mark.asyncio
async def test_llamaswap_traced_tolerant_observer_failure_writes_observer_note__DoD21(monkeypatch, tmp_path):
    """DoD-21: with tracing on, a tolerant observer failure writes ``TOOL RESULT`` with the observer-failure
    text, then ``TOOL ERROR`` with the note ``observer failed: <ExceptionType>: <message>`` (US-065.AC-13,
    DEC-006-7)."""
    opened = _enable_tracing(monkeypatch, tmp_path)

    def observer(event: Any) -> None:
        if event["type"] == "start":
            raise _ObserverBoom("observer exploded 5521")

    client = _client()
    _rounds(monkeypatch, client, [_reply(_call("lookup", '{"city": "Paris"}', "call_1")), _FINAL_ROUND])

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = await client.chat_with_tools(
            [_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": _Tool()}, on_tool_event=observer
        )

    assert result == _FINAL_TEXT
    assert opened
    note = _result_then_error(_only_record(tmp_path), "lookup", _failed("lookup"))
    assert "observer failed: _ObserverBoom: observer exploded 5521" in note


# ---------------------------------------------------------------------------
# DoD-22 — observer arguments when missing or a mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_observer_arguments_empty_text_when_call_has_none__DoD22(monkeypatch):
    """DoD-22: a call that carried no arguments gives the observer ``arguments == ""`` (US-065.AC-1,
    DEC-006-4)."""
    observer = _Events()

    def ping() -> str:
        return "pong"

    client = _client()
    _rounds(monkeypatch, client, [_reply(_call("ping", _MISSING, "call_1")), _FINAL_ROUND])

    await client.chat_with_tools(
        [_user()], tools_definitions=[_tool_def("ping")], tools={"ping": ping}, on_tool_event=observer
    )

    assert observer.events[0] == _event("start", "ping", "call_1", "")
    assert all(e["arguments"] == "" for e in observer.events)


@pytest.mark.asyncio
async def test_llamaswap_observer_arguments_json_text_when_mapping__DoD22(monkeypatch):
    """DoD-22: arguments that arrived as a mapping reach the observer as JSON text, non-ASCII kept
    (``json.dumps(value, ensure_ascii=False)``, DEC-006-4) (US-065.AC-1)."""
    observer = _Events()
    mapping = {"city": "Zürich", "days": 2}
    tool = _Tool("city", "days")
    client = _client()
    _rounds(monkeypatch, client, [_reply(_call("lookup", mapping, "call_1")), _FINAL_ROUND])

    await client.chat_with_tools(
        [_user()], tools_definitions=_LOOKUP_DEFS, tools={"lookup": tool}, on_tool_event=observer
    )

    expected = json.dumps(mapping, ensure_ascii=False)
    assert observer.events[0] == _event("start", "lookup", "call_1", expected)
    assert all(e["arguments"] == expected for e in observer.events)
    assert tool.calls == [mapping]
