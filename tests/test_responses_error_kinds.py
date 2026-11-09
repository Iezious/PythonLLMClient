"""Error kinds at the Responses classes' own raise sites (feature 006.tolerant-tool-loop, step 005).

Checks that every ``LLMError`` raised by ``OpenAIAPIClient`` and ``LlamaSwapAPIClient`` themselves, outside tool
execution, carries the ``kind`` DEC-arch-19 gives its raise site, and that the error text keeps its form
(``message: description``, US-067.AC-3): non-streamed replies (DoD-1), streamed ``chat`` (DoD-2), tool rounds
streamed and not, including a round cut at the app's own output limit that requests a tool (DoD-3, DEC-arch-13),
model-list and embeddings replies (DoD-4) and ``chat_json`` parse failure (DoD-5). Every expected value comes
from the step's Definition of done, DEC-arch-19, DEC-arch-13 and docs/architecture/interfaces.md § Exposed:
error contract.
"""

import copy
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

from llm.exceptions import LLMError
from llm.llama_swap import LlamaSwapAPIClient
from llm.openai import OpenAIAPIClient

OPENAI_BASE_URL = "https://api.example.test/v1"
LLAMASWAP_BASE_URL = "http://localhost:8080/v1"
MODEL = "some-model"
CREDENTIAL = "sk-test"

USER_MESSAGES = [{"role": "user", "content": "What is stored under key k-1?"}]

# Server reasons carried by the failure replies / events, distinctive so their presence in the text is observable.
FAILED_REASON = "Engine stalled 64"
ERROR_EVENT_REASON = "Slow down 29"
CUT_REASON = "max_output_tokens"
OTHER_INCOMPLETE_REASON = "content_filter"

# An output limit set by the app (DEC-arch-13: ``max_tokens`` present after the merge, call option or default).
APP_LIMIT = {"max_tokens": 64}

BACKENDS = [pytest.param("openai", id="openai"), pytest.param("llamaswap", id="llamaswap")]
STREAM_MODES = [pytest.param(False, id="non-streamed"), pytest.param(True, id="streamed")]

# An expected shape for chat_json (DoD-5), so the parse check runs.
EXPECTED_SHAPE: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


# ---------------------------------------------------------------------------
# Clients, tools and seams (DEC-003-16, DEC-004-14)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch):
    """Clear the trace switches so a developer's environment cannot turn tracing on (DEC-000-48)."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _client(backend: str, defaults: dict | None = None) -> OpenAIAPIClient | LlamaSwapAPIClient:
    """A Responses client of the given backend, built with an explicit test credential (tests/AGENTS.md)."""
    if backend == "openai":
        return OpenAIAPIClient(
            model=MODEL, base_url=OPENAI_BASE_URL, bearer_token=CREDENTIAL, default_options=defaults
        )
    return LlamaSwapAPIClient(
        model=MODEL, base_url=LLAMASWAP_BASE_URL, bearer_token=CREDENTIAL, default_options=defaults
    )


def _lookup_def() -> dict:
    """The app's nested definition of the single ``lookup`` tool."""
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Fetch a stored record by its key.",
            "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
        },
    }


class _Tools:
    """The ``lookup`` tool, recording every key it is run with, so "no tool runs" is observable."""

    def __init__(self) -> None:
        self.keys: list[str] = []

    def lookup(self, key: str) -> str:
        self.keys.append(key)
        return f"rec-{key}"


class _Recorder:
    """Async receiver that records every piece it is given."""

    def __init__(self) -> None:
        self.pieces: list[str] = []

    async def __call__(self, piece: str) -> None:
        self.pieces.append(piece)


class _Raises:
    """Marks a canned reply as an exception the request seam raises instead of returning a body.

    A factory is stored so each test raises a fresh exception object."""

    def __init__(self, make) -> None:
        self.make = make


def _serve(monkeypatch, client: Any, reply: Any) -> list[dict]:
    """Replace the ``_request`` seam with a fake answering ``reply`` (deep copy) to every call, or raising it for
    a ``_Raises`` reply; returns the recorded requests. The streamed path fails the test if used."""
    calls: list[dict] = []

    async def fake_request(method, path, **kw):
        calls.append({"method": method, "path": path, "json": copy.deepcopy(kw.get("json"))})
        if isinstance(reply, _Raises):
            raise reply.make()
        # Case data is shared across tests; hand out a copy so no call can alter another's reply.
        return copy.deepcopy(reply)

    def no_stream_post(*_args, **_kw):
        raise AssertionError("a non-streamed call must not open a stream")

    monkeypatch.setattr(client, "_request", fake_request)
    monkeypatch.setattr(client, "_stream_post", no_stream_post)
    return calls


def _stream(monkeypatch, client: Any, events: list[str]) -> list[dict]:
    """Replace ``_stream_post`` with a fake async context manager recording each body, and ``_iter_sse_data``
    with a fake yielding ``events`` (DEC-003-16, DEC-004-14); the non-streamed path fails the test if used."""
    calls: list[dict] = []

    @asynccontextmanager
    async def fake_stream_post(path, payload):
        calls.append({"path": path, "json": copy.deepcopy(payload)})
        yield SimpleNamespace(status=200)

    async def fake_iter_sse_data(_response):
        for payload in events:
            yield payload

    async def no_request(*_args, **_kw):
        raise AssertionError("a streamed call must not use the non-streamed request path")

    monkeypatch.setattr(client, "_stream_post", fake_stream_post)
    monkeypatch.setattr(client, "_iter_sse_data", fake_iter_sse_data)
    monkeypatch.setattr(client, "_request", no_request)
    return calls


def _assert_text_form(err: LLMError) -> None:
    """US-067.AC-3 / interfaces.md § error contract: the text form stays ``message: description`` (or
    ``message`` alone), ``kind`` is an attribute only, and ``status`` is none for a failure that is not a
    non-2xx reply."""
    message = err.args[0]
    expected = f"{message}: {err.description}" if err.description else message
    assert str(err) == expected
    assert err.status is None


def _carries(err: LLMError, reason: str) -> bool:
    """Whether the error carries the server's reason, in its text or as its description."""
    return reason in str(err) or err.description == reason


# ---------------------------------------------------------------------------
# Replies and events (Responses wire vocabulary)
# ---------------------------------------------------------------------------


def _message_item(text: str) -> dict:
    """An assistant ``message`` output item holding ``text``."""
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _call_item(arguments: str = '{"key": "k-1"}', status: str = "completed") -> dict:
    """A ``function_call`` output item for the ``lookup`` tool."""
    return {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": arguments,
        "status": status,
    }


def _reply(output: list[dict], **fields: Any) -> dict:
    """A non-streamed Responses reply with the given output items and status fields."""
    return {"id": "resp_1", "object": "response", "output": output, **fields}


def _completed_reply(text: str) -> dict:
    return _reply([_message_item(text)], status="completed")


def _failed_reply(output: list[dict] | None = None) -> dict:
    return _reply(output or [], status="failed", error={"code": "server_error", "message": FAILED_REASON})


def _incomplete_reply(reason: str, output: list[dict] | None = None) -> dict:
    return _reply(output or [], status="incomplete", incomplete_details={"reason": reason})


def _ev(type_: str, **fields: Any) -> str:
    """One SSE data payload: the JSON text of a Responses stream event."""
    return json.dumps({"type": type_, **fields})


CREATED = _ev("response.created", response={"id": "resp_1", "status": "in_progress", "output": []})


def _text_delta(delta: str) -> str:
    return _ev("response.output_text.delta", item_id="msg_1", output_index=0, content_index=0, delta=delta)


FAILED_EVENT = _ev(
    "response.failed",
    response={"id": "resp_1", "status": "failed", "error": {"code": "server_error", "message": FAILED_REASON}},
)
ERROR_EVENT = _ev("error", code="rate_limit_exceeded", message=ERROR_EVENT_REASON)


def _incomplete_event(reason: str) -> str:
    return _ev(
        "response.incomplete",
        response={"id": "resp_1", "status": "incomplete", "incomplete_details": {"reason": reason}, "output": []},
    )


# A trailing completion after a failure ending: reading past the failure would make the call succeed.
LATE_COMPLETED = _ev("response.completed", response={"id": "resp_1", "status": "completed", "output": []})


def _call_events() -> list[str]:
    """The streamed events of one ``lookup`` call at output position 1, without any ending event."""
    return [
        _ev("response.output_item.added", output_index=1, item=_call_item("", status="in_progress")),
        _ev("response.function_call_arguments.delta", item_id="fc_1", output_index=1, delta='{"key": "k-1"}'),
        _ev("response.output_item.done", output_index=1, item=_call_item()),
    ]


async def _chat_with_tools(client: Any, tools: _Tools, *, stream: bool, options: dict | None = None) -> str:
    """Run ``chat_with_tools`` with the ``lookup`` tool over a fresh copy of the caller data."""
    return await client.chat_with_tools(
        copy.deepcopy(USER_MESSAGES),
        tools_definitions=[_lookup_def()],
        tools={"lookup": tools.lookup},
        options=options,
        stream=stream,
        on_delta=_Recorder() if stream else None,
    )


# ---------------------------------------------------------------------------
# DoD-1 — non-streamed reply: not a mapping / failed / incomplete not at the app's limit
# ---------------------------------------------------------------------------

NON_MAPPING_REPLIES = [
    pytest.param([_message_item("hi")], id="list"),
    pytest.param("not an object", id="text"),
    pytest.param(42, id="number"),
    pytest.param(None, id="null"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("reply", NON_MAPPING_REPLIES)
async def test_responses_chat_non_mapping_reply_is_invalid_reply__DoD1(monkeypatch, backend, reply):
    """DoD-1: a non-streamed ``chat`` reply that is not a mapping raises ``LLMError`` with kind
    ``invalid_reply``; text form unchanged (US-067.AC-2, US-067.AC-3, DEC-arch-19)."""
    client = _client(backend)
    _serve(monkeypatch, client, reply)

    with pytest.raises(LLMError) as excinfo:
        await client.chat(copy.deepcopy(USER_MESSAGES))

    assert excinfo.value.kind == "invalid_reply"
    _assert_text_form(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
async def test_responses_chat_failed_reply_is_server_failed__DoD1(monkeypatch, backend):
    """DoD-1: a non-streamed ``chat`` reply reporting failure raises ``LLMError`` with kind ``server_failed``,
    still carrying the server's reason; text form unchanged (US-067.AC-2, US-067.AC-3)."""
    client = _client(backend)
    _serve(monkeypatch, client, _failed_reply())

    with pytest.raises(LLMError) as excinfo:
        await client.chat(copy.deepcopy(USER_MESSAGES))

    assert excinfo.value.kind == "server_failed"
    assert _carries(excinfo.value, FAILED_REASON)
    _assert_text_form(excinfo.value)


# An incomplete reply not cut at the app's own limit: another reason, or the output-limit reason with no app
# limit set (DEC-arch-13); both stay errors.
INCOMPLETE_NOT_APP_CUT = [
    pytest.param(OTHER_INCOMPLETE_REASON, APP_LIMIT, id="other-reason-with-app-limit"),
    pytest.param(OTHER_INCOMPLETE_REASON, None, id="other-reason-no-limit"),
    pytest.param(CUT_REASON, None, id="output-limit-reason-no-app-limit"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("reason", "options"), INCOMPLETE_NOT_APP_CUT)
async def test_responses_chat_incomplete_reply_is_incomplete__DoD1(monkeypatch, backend, reason, options):
    """DoD-1: a non-streamed ``chat`` reply that is incomplete, not cut at the app's own limit, raises
    ``LLMError`` with kind ``incomplete`` whose description is the reason (DEC-003-21); text form unchanged
    (US-067.AC-2, US-067.AC-3, DEC-arch-13)."""
    client = _client(backend)
    _serve(monkeypatch, client, _incomplete_reply(reason))

    with pytest.raises(LLMError) as excinfo:
        await client.chat(copy.deepcopy(USER_MESSAGES), options=options)

    assert excinfo.value.kind == "incomplete"
    assert excinfo.value.description == reason
    _assert_text_form(excinfo.value)


# ---------------------------------------------------------------------------
# DoD-2 — streamed chat: failure / error event, no completion event, incomplete not at the app's limit
# ---------------------------------------------------------------------------

STREAM_FAILURES = [
    pytest.param([CREATED, _text_delta("Partial"), FAILED_EVENT, LATE_COMPLETED], FAILED_REASON, id="failed-event"),
    pytest.param([CREATED, _text_delta("Partial"), ERROR_EVENT, LATE_COMPLETED], ERROR_EVENT_REASON, id="error-event"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("events", "reason"), STREAM_FAILURES)
async def test_responses_stream_chat_failure_event_is_server_failed__DoD2(monkeypatch, backend, events, reason):
    """DoD-2: streamed ``chat`` meeting a failure or an error event raises ``LLMError`` with kind
    ``server_failed``, carrying the server's reason (US-067.AC-2, DEC-arch-19)."""
    client = _client(backend)
    _stream(monkeypatch, client, events)

    with pytest.raises(LLMError) as excinfo:
        await client.chat(copy.deepcopy(USER_MESSAGES), stream=True, on_delta=_Recorder())

    assert excinfo.value.kind == "server_failed"
    assert _carries(excinfo.value, reason)
    _assert_text_form(excinfo.value)


STREAM_INCOMPLETE = [
    pytest.param([CREATED, _text_delta("Cut short")], None, id="ends-without-completion"),
    pytest.param([CREATED], APP_LIMIT, id="ends-without-completion-with-app-limit"),
    pytest.param(
        [CREATED, _text_delta("Cut"), _incomplete_event(OTHER_INCOMPLETE_REASON), LATE_COMPLETED],
        APP_LIMIT,
        id="incomplete-other-reason-with-app-limit",
    ),
    pytest.param(
        [CREATED, _text_delta("Cut"), _incomplete_event(CUT_REASON), LATE_COMPLETED],
        None,
        id="incomplete-output-limit-no-app-limit",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("events", "options"), STREAM_INCOMPLETE)
async def test_responses_stream_chat_incomplete_is_incomplete__DoD2(monkeypatch, backend, events, options):
    """DoD-2: streamed ``chat`` whose stream ends without its completion event, or reports incompleteness not
    at the app's own limit, raises ``LLMError`` with kind ``incomplete`` (US-067.AC-2, DEC-arch-13)."""
    client = _client(backend)
    _stream(monkeypatch, client, events)

    with pytest.raises(LLMError) as excinfo:
        await client.chat(copy.deepcopy(USER_MESSAGES), options=options, stream=True, on_delta=_Recorder())

    assert excinfo.value.kind == "incomplete"
    _assert_text_form(excinfo.value)


# ---------------------------------------------------------------------------
# DoD-3 — chat_with_tools, streamed and not: failed / incomplete round, cut round requesting a tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
async def test_responses_tool_round_failed_reply_is_server_failed__DoD3(monkeypatch, backend):
    """DoD-3: a non-streamed tool round reporting failure raises ``LLMError`` with kind ``server_failed``,
    carrying the server's reason; no tool runs (US-067.AC-2)."""
    client = _client(backend)
    _serve(monkeypatch, client, _failed_reply([_call_item()]))
    tools = _Tools()

    with pytest.raises(LLMError) as excinfo:
        await _chat_with_tools(client, tools, stream=False)

    assert excinfo.value.kind == "server_failed"
    assert _carries(excinfo.value, FAILED_REASON)
    _assert_text_form(excinfo.value)
    assert tools.keys == []


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("ending", "reason"),
    [
        pytest.param(FAILED_EVENT, FAILED_REASON, id="failed-event"),
        pytest.param(ERROR_EVENT, ERROR_EVENT_REASON, id="error-event"),
    ],
)
async def test_responses_stream_tool_round_failure_event_is_server_failed__DoD3(monkeypatch, backend, ending, reason):
    """DoD-3: a streamed tool round ending with a failure or an error event raises ``LLMError`` with kind
    ``server_failed``, carrying the server's reason; no tool runs (US-067.AC-2, DEC-arch-19)."""
    client = _client(backend)
    _stream(monkeypatch, client, [CREATED, *_call_events(), ending, LATE_COMPLETED])
    tools = _Tools()

    with pytest.raises(LLMError) as excinfo:
        await _chat_with_tools(client, tools, stream=True)

    assert excinfo.value.kind == "server_failed"
    assert _carries(excinfo.value, reason)
    _assert_text_form(excinfo.value)
    assert tools.keys == []


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("reason", "options"), INCOMPLETE_NOT_APP_CUT)
async def test_responses_tool_round_incomplete_reply_is_incomplete__DoD3(monkeypatch, backend, reason, options):
    """DoD-3: a non-streamed tool round that is incomplete, not cut at the app's own limit, raises ``LLMError``
    with kind ``incomplete``; no tool runs (US-067.AC-2, DEC-arch-13)."""
    client = _client(backend)
    _serve(monkeypatch, client, _incomplete_reply(reason, [_call_item()]))
    tools = _Tools()

    with pytest.raises(LLMError) as excinfo:
        await _chat_with_tools(client, tools, stream=False, options=options)

    assert excinfo.value.kind == "incomplete"
    assert excinfo.value.description == reason
    _assert_text_form(excinfo.value)
    assert tools.keys == []


STREAM_TOOL_INCOMPLETE = [
    pytest.param([CREATED, *_call_events()], None, id="ends-without-completion"),
    pytest.param(
        [CREATED, *_call_events(), _incomplete_event(OTHER_INCOMPLETE_REASON), LATE_COMPLETED],
        APP_LIMIT,
        id="incomplete-other-reason-with-app-limit",
    ),
    pytest.param(
        [CREATED, *_call_events(), _incomplete_event(CUT_REASON), LATE_COMPLETED],
        None,
        id="incomplete-output-limit-no-app-limit",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("events", "options"), STREAM_TOOL_INCOMPLETE)
async def test_responses_stream_tool_round_incomplete_is_incomplete__DoD3(monkeypatch, backend, events, options):
    """DoD-3: a streamed tool round that ends without its completion event, or reports incompleteness not at
    the app's own limit, raises ``LLMError`` with kind ``incomplete``; no tool runs (US-067.AC-2)."""
    client = _client(backend)
    _stream(monkeypatch, client, events)
    tools = _Tools()

    with pytest.raises(LLMError) as excinfo:
        await _chat_with_tools(client, tools, stream=True, options=options)

    assert excinfo.value.kind == "incomplete"
    _assert_text_form(excinfo.value)
    assert tools.keys == []


# The app's output limit as a call option or as a client default (DEC-arch-13: either counts).
LIMIT_SOURCES = [
    pytest.param(None, APP_LIMIT, id="call-option"),
    pytest.param(APP_LIMIT, None, id="client-default"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("stream", STREAM_MODES)
@pytest.mark.parametrize(("defaults", "options"), LIMIT_SOURCES)
async def test_responses_tool_round_cut_at_app_limit_requesting_tool_is_incomplete__DoD3(
    monkeypatch, backend, stream, defaults, options
):
    """DoD-3: a tool round cut at the app's own output limit that requests a tool raises ``LLMError`` with kind
    ``incomplete``, streamed and not, and the requested tool does not run; one round only (US-067.AC-2,
    DEC-arch-13, DEC-arch-19)."""
    client = _client(backend, defaults)
    if stream:
        events = [CREATED, *_call_events(), _incomplete_event(CUT_REASON), LATE_COMPLETED]
        calls = _stream(monkeypatch, client, events)
    else:
        calls = _serve(monkeypatch, client, _incomplete_reply(CUT_REASON, [_call_item()]))
    tools = _Tools()

    with pytest.raises(LLMError) as excinfo:
        await _chat_with_tools(client, tools, stream=stream, options=options)

    assert excinfo.value.kind == "incomplete"
    _assert_text_form(excinfo.value)
    assert tools.keys == []
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# DoD-4 — model-list and embeddings replies: malformed or unreadable
# ---------------------------------------------------------------------------


def _content_type_error() -> aiohttp.ContentTypeError:
    """The decode error the response's JSON reader raises for a 2xx body it cannot read as JSON."""
    url = f"{OPENAI_BASE_URL}/models"
    info = SimpleNamespace(url=url, real_url=url, method="GET", headers={})
    return aiohttp.ContentTypeError(info, (), message="Attempt to decode JSON with unexpected mimetype: text/html")


# A 2xx body that cannot be read as JSON (DEC-002-10).
UNREADABLE_BODIES = [
    pytest.param(b"<html><body>Bad gateway</body></html>", id="bytes-body"),
    pytest.param(_Raises(lambda: json.JSONDecodeError("Expecting value", "<html>", 0)), id="json-decode-error"),
    pytest.param(_Raises(_content_type_error), id="content-type-error"),
]

# A malformed model list or an entry without a usable id (DEC-002-9).
MALFORMED_MODEL_LISTS = [
    pytest.param([{"id": "a"}], id="reply-is-list"),
    pytest.param({"object": "list"}, id="data-missing"),
    pytest.param({"object": "list", "data": "a,b"}, id="data-is-text"),
    pytest.param({"object": "list", "data": [{"id": "a"}, {"object": "model"}]}, id="entry-without-id"),
    pytest.param({"object": "list", "data": [{"id": "a"}, "b"]}, id="entry-is-text"),
]

EMBED_TEXTS = ["alpha", "beta"]

# A malformed embeddings reply, an entry without a vector, or a count not matching the input (DEC-002-8).
MALFORMED_EMBEDDINGS = [
    pytest.param([[0.1, 0.2], [0.3, 0.4]], id="reply-is-list"),
    pytest.param({"object": "list"}, id="data-missing"),
    pytest.param(
        {"object": "list", "data": [{"embedding": [0.1]}, {"object": "embedding"}]}, id="entry-without-vector"
    ),
    pytest.param({"object": "list", "data": [{"embedding": [0.1, 0.2]}]}, id="count-mismatch"),
    pytest.param({"object": "list", "data": []}, id="empty-result"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", ["list_models", "get_models"])
@pytest.mark.parametrize("reply", [*MALFORMED_MODEL_LISTS, *UNREADABLE_BODIES])
async def test_responses_models_bad_reply_is_invalid_reply__DoD4(monkeypatch, backend, method, reply):
    """DoD-4: a malformed or unreadable model-list reply raises ``LLMError`` with kind ``invalid_reply`` from
    ``list_models`` and ``get_models`` on both Responses classes (US-067.AC-2, DEC-arch-19)."""
    client = _client(backend)
    _serve(monkeypatch, client, reply)

    with pytest.raises(LLMError) as excinfo:
        await getattr(client, method)()

    assert excinfo.value.kind == "invalid_reply"
    _assert_text_form(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("reply", [*MALFORMED_EMBEDDINGS, *UNREADABLE_BODIES])
async def test_responses_embed_batch_bad_reply_is_invalid_reply__DoD4(monkeypatch, backend, reply):
    """DoD-4: a malformed or unreadable embeddings reply (entry without a vector, count mismatch included)
    raises ``LLMError`` with kind ``invalid_reply`` from ``embed_batch`` (US-067.AC-2, DEC-arch-19)."""
    client = _client(backend)
    _serve(monkeypatch, client, reply)

    with pytest.raises(LLMError) as excinfo:
        await client.embed_batch(list(EMBED_TEXTS))

    assert excinfo.value.kind == "invalid_reply"
    _assert_text_form(excinfo.value)


# ---------------------------------------------------------------------------
# DoD-5 — chat_json with an expected shape: reply that does not parse as JSON
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "text",
    [pytest.param("Sure! It is great.", id="prose"), pytest.param('{"answer": "unterminated', id="truncated-object")],
)
async def test_responses_chat_json_not_json_is_invalid_reply__DoD5(monkeypatch, backend, text):
    """DoD-5: ``chat_json`` with an expected shape whose reply does not parse as JSON raises ``LLMError`` with
    kind ``invalid_reply`` (US-067.AC-2, DEC-arch-19)."""
    client = _client(backend)
    _serve(monkeypatch, client, _completed_reply(text))

    with pytest.raises(LLMError) as excinfo:
        await client.chat_json(copy.deepcopy(USER_MESSAGES), response_schema=EXPECTED_SHAPE)

    assert excinfo.value.kind == "invalid_reply"
    _assert_text_form(excinfo.value)
