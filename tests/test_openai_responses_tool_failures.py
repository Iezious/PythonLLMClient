"""Unit tests for tolerant tool failures and the tool-event observer on the OpenAI Responses class.

Feature 006 (tolerant-tool-loop), step 006: ``OpenAIAPIClient.chat_with_tools`` handles tool-call failures
tolerantly by default (the model is told a fixed text and the conversation continues, DEC-arch-14, DEC-arch-28),
raises today's exceptions with ``tool_errors="raise"`` (DEC-arch-17), skips the argument-name check for
``**kwargs`` callables and unreadable signatures (DEC-arch-16), serves the ``on_tool_event`` observer inline
(DEC-arch-18, DEC-arch-26, DEC-006-4, DEC-006-8) and traces tolerant / observer failures (DEC-006-7).

The server is faked through the client seams (DEC-004-14): ``_request`` with one canned reply per round for
non-streamed rounds, ``_stream_post`` + ``_iter_sse_data`` with one event list per round for streamed rounds.
"The model is told X" is checked as: the next round's request carries a ``function_call_output`` input item for
that call whose ``output`` is exactly X. Every expected value comes from the step's Definition of done and the
cited decisions; each test names the DoD id it covers.
"""

import asyncio
import copy
import json
import re
import warnings
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from llm.exceptions import LLMError, LLMToolObserverWarning
from llm.openai import OpenAIAPIClient

BASE_URL = "https://api.example.test/v1"
CREDENTIAL = "sk-test"
MODEL = "gpt-6-luna"

USER_TEXT = "What is stored under key k-9?"
USER_MESSAGES = [{"role": "user", "content": USER_TEXT}]

FINAL_TEXT = "The record under k-9 is unavailable right now."

_LOOKUP_PARAMS = {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}

# Sentinel for "key absent" in item builders, distinct from an explicit None.
_MISSING = object()


# ---------------------------------------------------------------------------
# Model-facing texts (DEC-arch-14), written out from the decision
# ---------------------------------------------------------------------------


def _not_available(name: str) -> str:
    return f"Tool '{name}' is not available."


def _invalid_arguments(name: str) -> str:
    return f"Invalid arguments for tool '{name}'."


def _no_argument(name: str, arg: str) -> str:
    return f"Tool '{name}' has no argument '{arg}'."


def _failed(name: str) -> str:
    return f"Tool '{name}' failed."


# ---------------------------------------------------------------------------
# Fixtures and helpers (module-local, tests/AGENTS.md)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_trace_env(monkeypatch):
    """Tracing is off unless a test switches it on, so no test depends on the developer's environment."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _client() -> OpenAIAPIClient:
    """A Responses client with the app-provided test credential."""
    return OpenAIAPIClient(model=MODEL, base_url=BASE_URL, bearer_token=CREDENTIAL)


def _function_call(call_id: str, name: Any = _MISSING, arguments: Any = _MISSING) -> dict:
    """A ``function_call`` output item; ``name`` / ``arguments`` omitted when ``_MISSING``."""
    item: dict[str, Any] = {"type": "function_call", "id": f"fc_{call_id}", "call_id": call_id, "status": "completed"}
    if name is not _MISSING:
        item["name"] = name
    if arguments is not _MISSING:
        item["arguments"] = arguments
    return item


def _message(text: str) -> dict:
    """An assistant ``message`` output item with one output-text part."""
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _reply(*items: dict) -> dict:
    """A completed non-streamed Responses reply with the given output items."""
    return {"id": "resp_1", "object": "response", "status": "completed", "output": list(items)}


def _answer_reply(text: str = FINAL_TEXT) -> dict:
    """A round that answers (no tool call)."""
    return _reply(_message(text))


def _nested_def(name: str, parameters: Any = _MISSING) -> dict:
    """An app tool definition in the nested shape."""
    function: dict[str, Any] = {"name": name, "description": f"Tool {name}."}
    if parameters is not _MISSING:
        function["parameters"] = parameters
    return {"type": "function", "function": function}


def _lookup_def() -> dict:
    return _nested_def("lookup", _LOOKUP_PARAMS)


def _capture_rounds(monkeypatch, client: OpenAIAPIClient, replies: list[dict]) -> list[dict]:
    """Replace ``_request`` with a fake returning one canned reply per round, in order (DEC-004-14).

    Each record keeps a deep copy of the body sent, so later changes by the client cannot alter what was
    observed. A request beyond the canned replies fails the test (it would be a round the spec forbids).
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


def _outputs(body: dict) -> list[tuple[Any, Any]]:
    """``(call_id, output)`` of every ``function_call_output`` input item of one request body, in order."""
    return [
        (item.get("call_id"), item.get("output"))
        for item in body["input"]
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]


def _event(type_: str, name: str, call_id: Any, arguments: str, *, output: Any = None, error_kind: Any = None):
    """The full ``ToolEvent`` the observer must receive (total shape, DEC-006-4)."""
    return {
        "type": type_,
        "name": name,
        "call_id": call_id,
        "arguments": arguments,
        "output": output,
        "error_kind": error_kind,
    }


class _Events:
    """A plain observer recording a copy of every event it receives, in order."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def __call__(self, event: Any) -> None:
        self.events.append(dict(event))


class _ObserverBoom(Exception):
    """An observer-side failure distinct from any library error (an ordinary exception)."""


class _FailingObserver:
    """Records every event, then raises ``self.boom`` for events matching ``fail_on`` (``(type, call_id)``
    pairs; a ``None`` call id matches any call), so a test chooses exactly which event the observer fails on."""

    def __init__(self, fail_on: set[tuple[str, Any]]) -> None:
        self.fail_on = fail_on
        self.events: list[dict] = []
        self.boom = _ObserverBoom("observer broke 4411")

    def __call__(self, event: Any) -> None:
        self.events.append(dict(event))
        if (event["type"], event["call_id"]) in self.fail_on or (event["type"], None) in self.fail_on:
            raise self.boom


@contextmanager
def _capture_warnings() -> Iterator[list[warnings.WarningMessage]]:
    """Record every warning with filter "always", so Python's de-duplication neither hides nor fakes a
    per-failure warning (tests/AGENTS.md)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def _observer_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in caught if issubclass(w.category, LLMToolObserverWarning)]


# --- Streamed rounds (DEC-004-14): Responses tool-round event vocabulary (DEC-004-9).


def _ev(type_: str, **fields: Any) -> str:
    """One SSE data payload: the JSON text of a Responses stream event."""
    return json.dumps({"type": type_, **fields})


_CREATED = _ev("response.created", response={"id": "resp_s", "status": "in_progress"})
_COMPLETED = _ev("response.completed", response={"id": "resp_s", "status": "completed"})


def _sse_call_round(call_id: str, name: str, arguments: str) -> list[str]:
    """Events of a streamed round requesting one call: item added, argument delta and done, item done."""
    item_id = f"fc_{call_id}"
    added = {"type": "function_call", "id": item_id, "call_id": call_id, "name": name, "arguments": "",
             "status": "in_progress"}
    done = {**added, "arguments": arguments, "status": "completed"}
    return [
        _CREATED,
        _ev("response.output_item.added", output_index=0, item=added),
        _ev("response.function_call_arguments.delta", item_id=item_id, output_index=0, delta=arguments),
        _ev("response.function_call_arguments.done", item_id=item_id, output_index=0, arguments=arguments),
        _ev("response.output_item.done", output_index=0, item=done),
        _COMPLETED,
    ]


def _sse_answer_round(text: str) -> list[str]:
    """Events of a streamed round that answers with ``text``."""
    added = {"type": "message", "id": "msg_s", "role": "assistant", "status": "in_progress", "content": []}
    return [
        _CREATED,
        _ev("response.output_item.added", output_index=0, item=added),
        _ev("response.content_part.added", item_id="msg_s", output_index=0, content_index=0,
            part={"type": "output_text", "text": ""}),
        _ev("response.output_text.delta", item_id="msg_s", output_index=0, content_index=0, delta=text),
        _ev("response.output_text.done", item_id="msg_s", output_index=0, content_index=0, text=text),
        _ev("response.output_item.done", output_index=0, item={**_message(text), "id": "msg_s"}),
        _COMPLETED,
    ]


def _install_stream_rounds(monkeypatch, client: OpenAIAPIClient, rounds: list[list[str]]) -> list[dict]:
    """Replace ``_stream_post`` (recording a deep copy of each body) and ``_iter_sse_data`` (one canned event
    list per round) — DEC-004-14. A non-streamed request fails the test."""
    calls: list[dict] = []
    pending = [list(events) for events in rounds]

    @asynccontextmanager
    async def fake_stream_post(path, payload):
        calls.append({"path": path, "payload": copy.deepcopy(payload)})
        yield SimpleNamespace(round=len(calls))

    async def fake_iter_sse_data(_response):
        if not pending:
            raise AssertionError("more rounds were streamed than canned rounds exist")
        for payload in pending.pop(0):
            yield payload

    async def forbidden_request(method, path, **kw):
        raise AssertionError("a streamed tool round must not use the non-streamed request path")

    monkeypatch.setattr(client, "_stream_post", fake_stream_post)
    monkeypatch.setattr(client, "_iter_sse_data", fake_iter_sse_data)
    monkeypatch.setattr(client, "_request", forbidden_request)
    return calls


# --- Failure-kind scenarios shared by the strict and observer tests.


class _Runs:
    """Collects which tools actually ran."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def lookup(self, key: str) -> str:
        self.calls.append(("lookup", {"key": key}))
        return f"record {key}"


def _scenario(kind: str, runs: _Runs) -> tuple[dict, dict, str, str]:
    """For one failure kind: ``(round-1 function call item, tools, requested name, arguments as sent)``.

    Each scenario is a single call with id ``call_1`` that fails with ``kind``.
    """
    def raising_lookup(key: str) -> str:
        runs.calls.append(("raising_lookup", {"key": key}))
        raise OSError("disk unavailable 77 /secret/path")

    def unserialisable_lookup(key: str) -> Any:
        runs.calls.append(("unserialisable_lookup", {"key": key}))
        return {"values": {1, 2, 3}}

    if kind == "not_offered":
        args = "{}"
        tools = {"lookup": runs.lookup, "secret_tool": lambda: runs.calls.append(("secret_tool", {}))}
        return _function_call("call_1", "secret_tool", args), tools, "secret_tool", args
    if kind == "bad_arguments":
        args = "not json {"
        return _function_call("call_1", "lookup", args), {"lookup": runs.lookup}, "lookup", args
    if kind == "unknown_argument":
        args = '{"key": "k-9", "colour": "red"}'
        return _function_call("call_1", "lookup", args), {"lookup": runs.lookup}, "lookup", args
    if kind == "tool_raised":
        args = '{"key": "k-9"}'
        return _function_call("call_1", "lookup", args), {"lookup": raising_lookup}, "lookup", args
    if kind == "unserialisable_result":
        args = '{"key": "k-9"}'
        return _function_call("call_1", "lookup", args), {"lookup": unserialisable_lookup}, "lookup", args
    raise AssertionError(kind)


FAILURE_KINDS = ["not_offered", "bad_arguments", "unknown_argument", "tool_raised", "unserialisable_result"]


# ---------------------------------------------------------------------------
# Tolerant default: the model is told, the conversation continues (DoD-1 .. DoD-5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["secret_tool", ""], ids=["unoffered", "empty-name"])
async def test_openai_unoffered_tool_told_to_model_and_conversation_continues__DoD1(monkeypatch, name):
    """DoD-1 (US-064.AC-1, AC-12, AC-13, AC-20): by default a requested tool that was not offered (or an empty
    name) runs no callable, the model is told ``Tool '<n>' is not available.``, and the answer is returned."""
    runs = _Runs()
    client = _client()
    records = _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", name, '{"key": "k-9"}')), _answer_reply()]
    )

    result = await client.chat_with_tools(
        list(USER_MESSAGES),
        tools_definitions=[_lookup_def()],
        tools={"lookup": runs.lookup, "secret_tool": lambda **kw: runs.calls.append(("secret_tool", kw))},
    )

    assert result == FINAL_TEXT
    assert runs.calls == []
    assert len(records) == 2
    assert _outputs(records[1]["json"]) == [("call_1", _not_available(name))]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments", ["not json {", "[1, 2]", "3", '"k-9"'], ids=["invalid-json", "list", "number", "string"]
)
async def test_openai_unreadable_arguments_told_to_model_and_conversation_continues__DoD2(monkeypatch, arguments):
    """DoD-2 (US-064.AC-2, AC-14, AC-15): arguments that are not valid JSON, or JSON that is not an object, run
    no callable; the model is told ``Invalid arguments for tool '<n>'.``; the conversation continues."""
    runs = _Runs()
    client = _client()
    records = _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", "lookup", arguments)), _answer_reply()]
    )

    result = await client.chat_with_tools(
        list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": runs.lookup}
    )

    assert result == FINAL_TEXT
    assert runs.calls == []
    assert len(records) == 2
    assert _outputs(records[1]["json"]) == [("call_1", _invalid_arguments("lookup"))]


@pytest.mark.asyncio
async def test_openai_unknown_argument_told_to_model_naming_first_in_sent_order__DoD3(monkeypatch):
    """DoD-3 (US-064.AC-3, AC-16): undeclared argument names run no callable; the model is told
    ``Tool '<n>' has no argument '<arg>'.`` naming the first undeclared name in the order the model sent them
    (not alphabetical); the conversation continues."""
    runs = _Runs()
    client = _client()
    arguments = '{"key": "k-9", "zz_extra": 1, "aa_extra": 2}'
    records = _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", "lookup", arguments)), _answer_reply()]
    )

    result = await client.chat_with_tools(
        list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": runs.lookup}
    )

    assert result == FINAL_TEXT
    assert runs.calls == []
    assert len(records) == 2
    assert _outputs(records[1]["json"]) == [("call_1", _no_argument("lookup", "zz_extra"))]


def _sync_raising(key: str) -> str:
    """A sync tool raising an ordinary exception whose text must never reach the model (DEC-arch-14)."""
    raise OSError("disk unavailable 77 at /srv/secret-store")


async def _async_raising(key: str) -> str:
    """The async twin of ``_sync_raising``; it yields once so the failure happens after a real await."""
    await asyncio.sleep(0)
    raise OSError("disk unavailable 77 at /srv/secret-store")


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", [_sync_raising, _async_raising], ids=["sync", "async"])
async def test_openai_raising_tool_told_failed_without_exception_text__DoD4(monkeypatch, tool):
    """DoD-4 (US-064.AC-4, AC-17): a sync or async tool raising an ordinary exception: the model is told
    ``Tool '<n>' failed.``, nothing of the exception's message is sent, and the conversation continues."""
    client = _client()
    records = _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", "lookup", '{"key": "k-9"}')), _answer_reply()]
    )

    result = await client.chat_with_tools(
        list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": tool}
    )

    assert result == FINAL_TEXT
    assert len(records) == 2
    assert _outputs(records[1]["json"]) == [("call_1", _failed("lookup"))]
    sent = json.dumps(records[1]["json"])
    assert "disk unavailable" not in sent
    assert "secret-store" not in sent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result_value", [{"values": {1, 2, 3}}, [{1, 2}]], ids=["mapping-with-set", "list-with-set"]
)
async def test_openai_unserialisable_result_told_failed_and_conversation_continues__DoD5(monkeypatch, result_value):
    """DoD-5 (US-064.AC-5, AC-18): a tool result that cannot be turned into text: the model is told
    ``Tool '<n>' failed.``; the conversation continues."""
    client = _client()
    records = _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", "lookup", '{"key": "k-9"}')), _answer_reply()]
    )

    result = await client.chat_with_tools(
        list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": lambda key: result_value}
    )

    assert result == FINAL_TEXT
    assert len(records) == 2
    assert _outputs(records[1]["json"]) == [("call_1", _failed("lookup"))]


# ---------------------------------------------------------------------------
# Several calls, mixed rounds, round limit, cancellation (DoD-6 .. DoD-9)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_middle_call_fails_others_run_in_order_each_gets_output__DoD6(monkeypatch):
    """DoD-6 (US-064.AC-6): three calls in one round, the middle one failing: the first and third tools run in
    the order requested, and every call gets its own function-call output in that order."""
    order: list[str] = []

    def lookup(key: str) -> str:
        order.append(f"lookup:{key}")
        return f"record {key}"

    def explode(key: str) -> str:
        order.append(f"explode:{key}")
        raise RuntimeError("boom 12")

    client = _client()
    round_1 = _reply(
        _function_call("call_1", "lookup", '{"key": "k-1"}'),
        _function_call("call_2", "explode", '{"key": "k-2"}'),
        _function_call("call_3", "lookup", '{"key": "k-3"}'),
    )
    records = _capture_rounds(monkeypatch, client, [round_1, _answer_reply()])

    result = await client.chat_with_tools(
        list(USER_MESSAGES),
        tools_definitions=[_lookup_def(), _nested_def("explode", _LOOKUP_PARAMS)],
        tools={"lookup": lookup, "explode": explode},
    )

    assert result == FINAL_TEXT
    assert order == ["lookup:k-1", "explode:k-2", "lookup:k-3"]
    assert _outputs(records[1]["json"]) == [
        ("call_1", "record k-1"),
        ("call_2", _failed("explode")),
        ("call_3", "record k-3"),
    ]


# Rounds 1-3 of the mixed conversation (DoD-7): unoffered tool, non-object arguments, unknown argument.
_MIXED_CALLS = [
    ("call_r1", "secret_tool", '{"key": "k-9"}'),
    ("call_r2", "lookup", "[1, 2]"),
    ("call_r3", "lookup", '{"key": "k-9", "colour": "red"}'),
]
_MIXED_OUTPUTS = [
    ("call_r1", _not_available("secret_tool")),
    ("call_r2", _invalid_arguments("lookup")),
    ("call_r3", _no_argument("lookup", "colour")),
]
_ROUND_4_ANSWER = "Round four answer 8812."


@pytest.mark.asyncio
async def test_openai_mixed_failures_over_rounds_non_streamed_returns_round_4__DoD7(monkeypatch):
    """DoD-7 (US-064.AC-7), non-streamed: unoffered tool, non-object arguments, unknown argument, then an
    answer → the call returns round 4's answer without an error; each failure was told to the model."""
    runs = _Runs()
    client = _client()
    replies = [_reply(_function_call(cid, name, args)) for cid, name, args in _MIXED_CALLS]
    records = _capture_rounds(monkeypatch, client, [*replies, _answer_reply(_ROUND_4_ANSWER)])

    result = await client.chat_with_tools(
        list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": runs.lookup}
    )

    assert result == _ROUND_4_ANSWER
    assert runs.calls == []
    assert len(records) == 4
    assert _outputs(records[3]["json"]) == _MIXED_OUTPUTS


@pytest.mark.asyncio
async def test_openai_mixed_failures_over_rounds_streamed_returns_round_4__DoD7(monkeypatch):
    """DoD-7 (US-064.AC-7), streamed: the same four rounds streamed → the call returns round 4's answer without
    an error; each failure was told to the model."""
    runs = _Runs()
    client = _client()
    rounds = [_sse_call_round(cid, name, args) for cid, name, args in _MIXED_CALLS]
    calls = _install_stream_rounds(monkeypatch, client, [*rounds, _sse_answer_round(_ROUND_4_ANSWER)])

    result = await client.chat_with_tools(
        list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": runs.lookup}, stream=True
    )

    assert result == _ROUND_4_ANSWER
    assert runs.calls == []
    assert len(calls) == 4
    assert _outputs(calls[3]["payload"]) == _MIXED_OUTPUTS


@pytest.mark.asyncio
async def test_openai_failing_call_every_round_ends_with_round_limit_error__DoD8(monkeypatch):
    """DoD-8 (US-064.AC-8): a model requesting a failing call in every round ends with the round-limit
    RuntimeError stating the limit and its value, after exactly ``max_loops`` rounds."""
    client = _client()
    replies = [_reply(_function_call(f"call_{n}", "secret_tool", "{}")) for n in range(1, 5)]
    records = _capture_rounds(monkeypatch, client, replies)

    with pytest.raises(RuntimeError) as excinfo:
        await client.chat_with_tools(
            list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": _Runs().lookup}, max_loops=3
        )

    message = str(excinfo.value)
    assert len(records) == 3
    assert "max_loops" in message
    assert re.search(r"(?<!\d)3(?!\d)", message)


def _sync_cancelling(key: str) -> str:
    """A tool raising cancellation, which is not an ordinary exception and must propagate (US-064.AC-9)."""
    raise asyncio.CancelledError()


async def _async_cancelling(key: str) -> str:
    """The async twin of ``_sync_cancelling``."""
    await asyncio.sleep(0)
    raise asyncio.CancelledError()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", [_sync_cancelling, _async_cancelling], ids=["sync", "async"])
@pytest.mark.parametrize("with_observer", [False, True], ids=["no-observer", "observer"])
async def test_openai_cancellation_in_tool_propagates_no_output_no_error_event__DoD9(
    monkeypatch, tool, with_observer
):
    """DoD-9 (US-064.AC-9, AC-19): a tool raising cancellation: the cancellation propagates out of the call, no
    further round is requested (so no function-call output is sent for it); with an observer, no ``error``
    event is sent for it (only its ``start``)."""
    observer = _Events()
    client = _client()
    records = _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", "lookup", '{"key": "k-9"}')), _answer_reply()]
    )
    extra = {"on_tool_event": observer} if with_observer else {}

    with pytest.raises(asyncio.CancelledError):
        await client.chat_with_tools(
            list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": tool}, **extra
        )

    assert len(records) == 1
    if with_observer:
        assert [e["type"] for e in observer.events] == ["start"]


# ---------------------------------------------------------------------------
# Strict opt-out (DoD-10)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["not_offered", "bad_arguments", "unserialisable_result"])
async def test_openai_strict_llmerror_kinds_raise_with_kind_tool__DoD10(monkeypatch, kind):
    """DoD-10 (US-064.AC-10, US-067.AC-2): with ``tool_errors="raise"``, ``not_offered``, ``bad_arguments`` and
    ``unserialisable_result`` end the call with LLMError naming the tool and carrying ``kind == "tool"``; no
    further round is requested."""
    runs = _Runs()
    call, tools, name, _ = _scenario(kind, runs)
    client = _client()
    records = _capture_rounds(monkeypatch, client, [_reply(call), _answer_reply()])

    with pytest.raises(LLMError) as excinfo:
        await client.chat_with_tools(
            list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools=tools, tool_errors="raise"
        )

    assert excinfo.value.kind == "tool"
    assert name in str(excinfo.value)
    assert len(records) == 1


@pytest.mark.asyncio
async def test_openai_strict_empty_name_raises_llmerror_with_kind_tool__DoD10(monkeypatch):
    """DoD-10 (US-064.AC-10, US-067.AC-2): with ``tool_errors="raise"`` an empty tool name ends the call with
    LLMError (today's text says the name was empty) carrying ``kind == "tool"``; no tool runs."""
    runs = _Runs()
    client = _client()
    records = _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", "", '{"key": "k-9"}')), _answer_reply()]
    )

    with pytest.raises(LLMError) as excinfo:
        await client.chat_with_tools(
            list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": runs.lookup}, tool_errors="raise"
        )

    assert excinfo.value.kind == "tool"
    assert "empty" in str(excinfo.value).lower()
    assert runs.calls == []
    assert len(records) == 1


@pytest.mark.asyncio
async def test_openai_strict_unknown_argument_raises_value_error__DoD10(monkeypatch):
    """DoD-10 (US-064.AC-10): with ``tool_errors="raise"`` an undeclared argument name ends the call with
    ValueError; the tool does not run and no further round is requested."""
    runs = _Runs()
    call, tools, _, _ = _scenario("unknown_argument", runs)
    client = _client()
    records = _capture_rounds(monkeypatch, client, [_reply(call), _answer_reply()])

    with pytest.raises(ValueError):
        await client.chat_with_tools(
            list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools=tools, tool_errors="raise"
        )

    assert runs.calls == []
    assert len(records) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", [_sync_raising, _async_raising], ids=["sync", "async"])
async def test_openai_strict_raising_tool_raises_runtime_error_chained__DoD10(monkeypatch, tool):
    """DoD-10 (US-064.AC-10): with ``tool_errors="raise"`` a tool raising an ordinary exception ends the call
    with RuntimeError naming the tool, chained from the tool's exception; no further round is requested."""
    client = _client()
    records = _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", "lookup", '{"key": "k-9"}')), _answer_reply()]
    )

    with pytest.raises(RuntimeError) as excinfo:
        await client.chat_with_tools(
            list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": tool}, tool_errors="raise"
        )

    assert "lookup" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, OSError)
    assert "disk unavailable 77" in str(excinfo.value.__cause__)
    assert len(records) == 1


# ---------------------------------------------------------------------------
# Any-argument tools and unreadable signatures (DoD-11)
# ---------------------------------------------------------------------------


class _UnreadableSignature:
    """A callable whose signature cannot be read: reading ``__signature__`` raises, so ``inspect.signature``
    fails (DEC-arch-16: the name check is then skipped and the tool itself decides)."""

    def __init__(self) -> None:
        self.received: list[dict] = []

    @property
    def __signature__(self):
        raise ValueError("signature not readable")

    def __call__(self, **kwargs: Any) -> str:
        self.received.append(dict(kwargs))
        return "unreadable-ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [{}, {"tool_errors": "raise"}], ids=["default", "strict"])
async def test_openai_var_keyword_tool_runs_with_exact_arguments__DoD11(monkeypatch, mode):
    """DoD-11 (US-066.AC-1): a callable declaring ``**kwargs`` runs with exactly the arguments the model sent,
    in both modes, and its result is sent to the model."""
    received: list[dict] = []

    def anything(**kwargs: Any) -> str:
        received.append(dict(kwargs))
        return "kwargs-ok"

    client = _client()
    records = _capture_rounds(
        monkeypatch,
        client,
        [_reply(_function_call("call_1", "anything", '{"alpha": 1, "beta": "two"}')), _answer_reply()],
    )

    result = await client.chat_with_tools(
        list(USER_MESSAGES), tools_definitions=[_nested_def("anything")], tools={"anything": anything}, **mode
    )

    assert result == FINAL_TEXT
    assert received == [{"alpha": 1, "beta": "two"}]
    assert _outputs(records[1]["json"]) == [("call_1", "kwargs-ok")]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [{}, {"tool_errors": "raise"}], ids=["default", "strict"])
async def test_openai_unreadable_signature_tool_called_without_name_check__DoD11(monkeypatch, mode):
    """DoD-11 (US-066.AC-1, DEC-arch-16): a callable whose signature cannot be read is called without the name
    check, with the model's arguments, in both modes."""
    tool = _UnreadableSignature()
    client = _client()
    records = _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", "opaque", '{"whatever": 5}')), _answer_reply()]
    )

    result = await client.chat_with_tools(
        list(USER_MESSAGES), tools_definitions=[_nested_def("opaque")], tools={"opaque": tool}, **mode
    )

    assert result == FINAL_TEXT
    assert tool.received == [{"whatever": 5}]
    assert _outputs(records[1]["json"]) == [("call_1", "unreadable-ok")]


# ---------------------------------------------------------------------------
# Observer events (DoD-12 .. DoD-16, DoD-22)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_observer_gets_start_before_tool_with_verbatim_arguments__DoD12(monkeypatch):
    """DoD-12 (US-065.AC-1): before the tool runs the observer receives a ``start`` event with the requested
    name, the call's Responses call id and the argument string verbatim; ``output`` and ``error_kind`` none."""
    log: list[Any] = []
    arguments = '{ "key" :   "k-9" }'

    def observer(event: Any) -> None:
        log.append(("event", dict(event)))

    def lookup(key: str) -> str:
        log.append(("tool", key))
        return "record k-9"

    client = _client()
    _capture_rounds(monkeypatch, client, [_reply(_function_call("call_1", "lookup", arguments)), _answer_reply()])

    await client.chat_with_tools(
        list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": lookup}, on_tool_event=observer
    )

    assert log[0] == ("event", _event("start", "lookup", "call_1", arguments))
    assert log[1] == ("tool", "k-9")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value", ["plain text 55", {"answer": 42, "items": [1, 2]}], ids=["text-result", "mapping-result"]
)
async def test_openai_observer_result_output_equals_text_sent_to_model__DoD13(monkeypatch, value):
    """DoD-13 (US-065.AC-2): after a successful tool the observer receives a ``result`` event whose ``output``
    equals the function-call output text sent to the model."""
    observer = _Events()
    arguments = '{"key": "k-9"}'
    client = _client()
    records = _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", "lookup", arguments)), _answer_reply()]
    )

    await client.chat_with_tools(
        list(USER_MESSAGES),
        tools_definitions=[_lookup_def()],
        tools={"lookup": lambda key: value},
        on_tool_event=observer,
    )

    [(call_id, sent)] = _outputs(records[1]["json"])
    assert call_id == "call_1"
    assert observer.events == [
        _event("start", "lookup", "call_1", arguments),
        _event("result", "lookup", "call_1", arguments, output=sent),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", FAILURE_KINDS)
async def test_openai_observer_gets_start_then_error_with_kind__DoD14(monkeypatch, kind):
    """DoD-14 (US-065.AC-3): for each of the five failure kinds the observer receives ``start`` then an
    ``error`` event with the name, call id, arguments as sent and the matching ``error_kind`` — including
    ``not_offered`` and ``bad_arguments``, where no tool runs."""
    runs = _Runs()
    call, tools, name, arguments = _scenario(kind, runs)
    observer = _Events()
    client = _client()
    _capture_rounds(monkeypatch, client, [_reply(call), _answer_reply()])

    result = await client.chat_with_tools(
        list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools=tools, on_tool_event=observer
    )

    assert result == FINAL_TEXT
    assert observer.events == [
        _event("start", name, "call_1", arguments),
        _event("error", name, "call_1", arguments, error_kind=kind),
    ]


@pytest.mark.asyncio
async def test_openai_observer_events_arrive_in_call_order__DoD15(monkeypatch):
    """DoD-15 (US-065.AC-4): with several calls the events arrive in call order — ``start`` then ``result`` /
    ``error`` per call."""
    observer = _Events()

    def explode(key: str) -> str:
        raise RuntimeError("boom 31")

    client = _client()
    round_1 = _reply(
        _function_call("call_1", "lookup", '{"key": "k-1"}'),
        _function_call("call_2", "explode", '{"key": "k-2"}'),
        _function_call("call_3", "lookup", '{"key": "k-3"}'),
    )
    _capture_rounds(monkeypatch, client, [round_1, _answer_reply()])

    await client.chat_with_tools(
        list(USER_MESSAGES),
        tools_definitions=[_lookup_def(), _nested_def("explode", _LOOKUP_PARAMS)],
        tools={"lookup": lambda key: f"record {key}", "explode": explode},
        on_tool_event=observer,
    )

    assert [(e["type"], e["call_id"]) for e in observer.events] == [
        ("start", "call_1"),
        ("result", "call_1"),
        ("start", "call_2"),
        ("error", "call_2"),
        ("start", "call_3"),
        ("result", "call_3"),
    ]
    assert observer.events[3]["error_kind"] == "tool_raised"


@pytest.mark.asyncio
async def test_openai_coroutine_observer_finishes_start_before_tool_runs__DoD15(monkeypatch):
    """DoD-15 (US-065.AC-5, DEC-arch-18): a coroutine observer is awaited inline — its work for ``start`` has
    finished before the tool is called, and each event's work finishes before the next step."""
    log: list[str] = []

    async def observer(event: Any) -> None:
        # Yield to the loop several times, so unawaited or scheduled work would finish too late.
        for _ in range(3):
            await asyncio.sleep(0)
        log.append(f"observed:{event['type']}:{event['call_id']}")

    def lookup(key: str) -> str:
        log.append(f"tool:{key}")
        return f"record {key}"

    client = _client()
    round_1 = _reply(
        _function_call("call_1", "lookup", '{"key": "k-1"}'),
        _function_call("call_2", "lookup", '{"key": "k-2"}'),
    )
    _capture_rounds(monkeypatch, client, [round_1, _answer_reply()])

    await client.chat_with_tools(
        list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": lookup}, on_tool_event=observer
    )

    assert log == [
        "observed:start:call_1",
        "tool:k-1",
        "observed:result:call_1",
        "observed:start:call_2",
        "tool:k-2",
        "observed:result:call_2",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", FAILURE_KINDS)
async def test_openai_strict_error_event_received_before_exception__DoD16(monkeypatch, kind):
    """DoD-16 (US-065.AC-6, DEC-arch-17): with ``tool_errors="raise"`` and an observer, the failing call's
    ``error`` event has been received by the time the call's exception reaches the caller."""
    runs = _Runs()
    call, tools, name, arguments = _scenario(kind, runs)
    observer = _Events()
    client = _client()
    _capture_rounds(monkeypatch, client, [_reply(call), _answer_reply()])

    with pytest.raises((LLMError, ValueError, RuntimeError)):
        await client.chat_with_tools(
            list(USER_MESSAGES),
            tools_definitions=[_lookup_def()],
            tools=tools,
            tool_errors="raise",
            on_tool_event=observer,
        )

    assert observer.events == [
        _event("start", name, "call_1", arguments),
        _event("error", name, "call_1", arguments, error_kind=kind),
    ]


@pytest.mark.asyncio
async def test_openai_observer_arguments_empty_when_call_has_none__DoD22(monkeypatch):
    """DoD-22 (US-065.AC-1, DEC-006-4): the observer ``arguments`` field is empty text when the call carried no
    arguments."""
    observer = _Events()
    client = _client()
    _capture_rounds(monkeypatch, client, [_reply(_function_call("call_1", "ping")), _answer_reply()])

    await client.chat_with_tools(
        list(USER_MESSAGES), tools_definitions=[_nested_def("ping")], tools={"ping": lambda: "pong"},
        on_tool_event=observer,
    )

    assert observer.events[0] == _event("start", "ping", "call_1", "")
    assert observer.events[1]["arguments"] == ""


@pytest.mark.asyncio
async def test_openai_observer_arguments_json_text_when_mapping__DoD22(monkeypatch):
    """DoD-22 (US-065.AC-1, DEC-006-4): when the arguments arrived as a mapping, the observer ``arguments`` field
    is its JSON text (non-ASCII kept as is)."""
    observer = _Events()
    mapping = {"key": "k-é 9"}
    client = _client()
    _capture_rounds(monkeypatch, client, [_reply(_function_call("call_1", "lookup", mapping)), _answer_reply()])

    await client.chat_with_tools(
        list(USER_MESSAGES),
        tools_definitions=[_lookup_def()],
        tools={"lookup": lambda key: f"record {key}"},
        on_tool_event=observer,
    )

    expected = json.dumps(mapping, ensure_ascii=False)
    assert observer.events[0] == _event("start", "lookup", "call_1", expected)
    assert json.loads(observer.events[0]["arguments"]) == mapping


# ---------------------------------------------------------------------------
# Observer failures (DoD-17 .. DoD-20)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_on", ["start", "result"])
async def test_openai_strict_observer_failure_propagates_unchanged__DoD17(monkeypatch, fail_on):
    """DoD-17 (US-065.AC-7): with ``tool_errors="raise"`` an observer raising on ``start`` or on ``result`` ends
    the call with that same exception, unchanged; when it raised on ``start`` the tool did not run."""
    runs = _Runs()
    observer = _FailingObserver({(fail_on, "call_1")})
    client = _client()
    records = _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", "lookup", '{"key": "k-9"}')), _answer_reply()]
    )

    with pytest.raises(_ObserverBoom) as excinfo:
        await client.chat_with_tools(
            list(USER_MESSAGES),
            tools_definitions=[_lookup_def()],
            tools={"lookup": runs.lookup},
            tool_errors="raise",
            on_tool_event=observer,
        )

    assert excinfo.value is observer.boom
    assert len(records) == 1
    if fail_on == "start":
        assert runs.calls == []
    else:
        assert runs.calls == [("lookup", {"key": "k-9"})]


@pytest.mark.asyncio
async def test_openai_tolerant_observer_failure_on_start_skips_tool_and_tells_failed__DoD18(monkeypatch):
    """DoD-18 (US-065.AC-8, AC-11, AC-12): by default an observer raising on ``start`` → the tool does not run,
    the model is told ``Tool '<n>' failed.``, the conversation continues; the observer receives no further
    event for that call while the later call still gets its events."""
    runs = _Runs()
    observer = _FailingObserver({("start", "call_1")})
    client = _client()
    round_1 = _reply(
        _function_call("call_1", "lookup", '{"key": "k-1"}'),
        _function_call("call_2", "lookup", '{"key": "k-2"}'),
    )
    records = _capture_rounds(monkeypatch, client, [round_1, _answer_reply()])

    with _capture_warnings():
        result = await client.chat_with_tools(
            list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": runs.lookup},
            on_tool_event=observer,
        )

    assert result == FINAL_TEXT
    assert runs.calls == [("lookup", {"key": "k-2"})]
    assert _outputs(records[1]["json"]) == [("call_1", _failed("lookup")), ("call_2", "record k-2")]
    assert [(e["type"], e["call_id"]) for e in observer.events] == [
        ("start", "call_1"),
        ("start", "call_2"),
        ("result", "call_2"),
    ]


@pytest.mark.asyncio
async def test_openai_tolerant_observer_failure_on_result_replaces_output__DoD19(monkeypatch):
    """DoD-19 (US-065.AC-11, AC-12, DEC-006-8): by default an observer raising on ``result`` → the model is told
    ``Tool '<n>' failed.`` instead of the tool's result; no further event for that call; the call goes on."""
    runs = _Runs()
    observer = _FailingObserver({("result", "call_1")})
    client = _client()
    records = _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", "lookup", '{"key": "k-9"}')), _answer_reply()]
    )

    with _capture_warnings():
        result = await client.chat_with_tools(
            list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": runs.lookup},
            on_tool_event=observer,
        )

    assert result == FINAL_TEXT
    assert runs.calls == [("lookup", {"key": "k-9"})]
    assert _outputs(records[1]["json"]) == [("call_1", _failed("lookup"))]
    assert [e["type"] for e in observer.events] == ["start", "result"]


def _explode(key: str) -> str:
    """A tool that always fails, so its call produces an ``error`` event (tool_raised)."""
    raise RuntimeError("boom 64")


# (fail_on set, expected number of observer failures); call_1 succeeds via lookup, call_2 raises via explode.
_OBSERVER_FAILURES = [
    pytest.param({("start", "call_1")}, 1, id="on-start"),
    pytest.param({("result", "call_1")}, 1, id="on-result"),
    pytest.param({("error", "call_2")}, 1, id="on-error"),
    pytest.param({("start", "call_1"), ("start", "call_2")}, 2, id="two-failures"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_on, expected", _OBSERVER_FAILURES)
async def test_openai_tolerant_observer_failure_emits_one_warning_each__DoD20(monkeypatch, fail_on, expected):
    """DoD-20 (US-065.AC-9, DEC-006-8): by default each observer failure emits exactly one
    ``LLMToolObserverWarning`` (naming the tool), and the call still returns its answer."""
    observer = _FailingObserver(fail_on)
    client = _client()
    round_1 = _reply(
        _function_call("call_1", "lookup", '{"key": "k-1"}'),
        _function_call("call_2", "explode", '{"key": "k-2"}'),
    )
    _capture_rounds(monkeypatch, client, [round_1, _answer_reply()])

    with _capture_warnings() as caught:
        result = await client.chat_with_tools(
            list(USER_MESSAGES),
            tools_definitions=[_lookup_def(), _nested_def("explode", _LOOKUP_PARAMS)],
            tools={"lookup": lambda key: f"record {key}", "explode": _explode},
            on_tool_event=observer,
        )

    assert result == FINAL_TEXT
    messages = _observer_warnings(caught)
    assert len(messages) == expected
    for _, call_id in fail_on:
        name = "lookup" if call_id == "call_1" else "explode"
        assert any(name in message for message in messages), messages


# ---------------------------------------------------------------------------
# Trace of tolerant and observer failures (DoD-21)
# ---------------------------------------------------------------------------

# Section headings of a trace record (data-model.md § Trace record; DEC-006-7 adds ``TOOL ERROR: <name>``).
_HEADINGS = frozenset({"SYSTEM PROMPT", "MESSAGE HISTORY", "REQUEST", "RESPONSE", "CHUNK", "ERROR", "END TRACE"})


def _sections(text: str) -> list[tuple[str, list[str]]]:
    """Split a record into ``(title, body lines)``; blank and pure ``=``/``-`` rule lines are layout and skipped."""
    sections: list[tuple[str, list[str]]] = [("HEADER", [])]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) <= {"=", "-"}:
            continue
        if stripped in _HEADINGS or stripped.startswith(("TOOL CALL:", "TOOL RESULT:", "TOOL ERROR:")):
            sections.append((stripped, []))
        else:
            sections[-1][1].append(stripped)
    return sections


def _only_record(root: Path) -> str:
    logs = sorted(root.rglob("*.log"))
    assert len(logs) == 1, logs
    return logs[0].read_text(encoding="utf-8")


def _enable_tracing(monkeypatch, folder: Path) -> None:
    """The tracer reads its switches when the client is built, so they are set before building it."""
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(folder))


def _result_then_error(record: str, name: str) -> tuple[str, str]:
    """The bodies of the ``TOOL RESULT: <name>`` section and the ``TOOL ERROR: <name>`` section that must follow
    it directly (DEC-006-7)."""
    sections = _sections(record)
    titles = [title for title, _ in sections]
    assert f"TOOL RESULT: {name}" in titles, titles
    at = titles.index(f"TOOL RESULT: {name}")
    assert at + 1 < len(sections) and sections[at + 1][0] == f"TOOL ERROR: {name}", titles
    return "\n".join(sections[at][1]), "\n".join(sections[at + 1][1])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind, arguments, tool, told",
    [
        ("tool_raised", '{"key": "k-9"}', _sync_raising, _failed("lookup")),
        ("bad_arguments", "[1, 2]", lambda key: "never", _invalid_arguments("lookup")),
    ],
    ids=["tool_raised", "bad_arguments"],
)
async def test_openai_traced_tolerant_tool_failure_writes_result_then_error__DoD21(
    monkeypatch, tmp_path, kind, arguments, tool, told
):
    """DoD-21 (US-065.AC-13, DEC-006-7): with tracing on, a tolerant tool failure writes ``TOOL RESULT`` with the
    model-facing text followed by ``TOOL ERROR`` with the failure kind name; the call still returns."""
    _enable_tracing(monkeypatch, tmp_path)
    client = _client()
    _capture_rounds(monkeypatch, client, [_reply(_function_call("call_1", "lookup", arguments)), _answer_reply()])

    result = await client.chat_with_tools(
        list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": tool}
    )

    assert result == FINAL_TEXT
    result_body, error_body = _result_then_error(_only_record(tmp_path), "lookup")
    assert told in result_body
    assert kind in error_body


@pytest.mark.asyncio
async def test_openai_traced_tolerant_observer_failure_writes_observer_note__DoD21(monkeypatch, tmp_path):
    """DoD-21 (US-065.AC-13, DEC-006-7): with tracing on, a tolerant observer failure writes ``TOOL RESULT`` with
    the observer-failure text followed by ``TOOL ERROR`` with the note ``observer failed: <type>: <message>``."""
    _enable_tracing(monkeypatch, tmp_path)
    observer = _FailingObserver({("start", "call_1")})
    client = _client()
    _capture_rounds(
        monkeypatch, client, [_reply(_function_call("call_1", "lookup", '{"key": "k-9"}')), _answer_reply()]
    )

    with _capture_warnings():
        result = await client.chat_with_tools(
            list(USER_MESSAGES), tools_definitions=[_lookup_def()], tools={"lookup": _Runs().lookup},
            on_tool_event=observer,
        )

    assert result == FINAL_TEXT
    result_body, error_body = _result_then_error(_only_record(tmp_path), "lookup")
    assert _failed("lookup") in result_body
    assert "observer failed:" in error_body
    assert "_ObserverBoom" in error_body
    assert "observer broke 4411" in error_body
