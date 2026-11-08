"""Embedding reply checks on the OpenAI-family clients (DEC-002-8, DEC-002-10).

Covers ``embed_batch`` and ``embed`` on ``OpenAIChatCompletionsClient`` and ``LlamaSwapChatCompletionsClient``
(the chat-completions regression suite, DEC-arch-12): one vector per text, ordering by
position when every entry carries an integer position (DEC-002-8), malformed and unreadable replies ending
with ``LLMError`` (DEC-002-8, DEC-002-10), and request-helper refusals passing through unchanged. Every
expected value comes from the step's Definition of done (DoD-1..DoD-10) and its decisions.
"""

import copy
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

from llm.exceptions import LLMError
from llm.llama_swap_chat_completions import LlamaSwapChatCompletionsClient
from llm.openai_chat_completions import OpenAIChatCompletionsClient

EMBED_MODEL = "text-embedding-test"
OPENAI_BASE_URL = "https://api.example.test/v1"
LLAMASWAP_BASE_URL = "http://localhost:8080/v1"

TEXTS = ["alpha", "beta", "gamma"]
# One distinct vector per text, so a wrong order or a dropped entry is visible in the result.
VEC = {"alpha": [0.1, 0.2], "beta": [0.3, 0.4], "gamma": [0.5, 0.6]}

Client = OpenAIChatCompletionsClient | LlamaSwapChatCompletionsClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch):
    """Start every test with tracing off so a developer's shell env cannot change behaviour (DEC-000-4)."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _openai() -> OpenAIChatCompletionsClient:
    """An OpenAI client with an explicit test credential."""
    return OpenAIChatCompletionsClient(model=EMBED_MODEL, base_url=OPENAI_BASE_URL, bearer_token="sk-test")


def _llamaswap() -> LlamaSwapChatCompletionsClient:
    """A llama-swap client on a local test address with an explicit test credential."""
    return LlamaSwapChatCompletionsClient(model=EMBED_MODEL, base_url=LLAMASWAP_BASE_URL, bearer_token="sk-test")


class _Raises:
    """Marks a fake reply as an exception the request helper raises instead of returning a body.

    A factory is stored (not an instance) so each test raises a fresh exception object.
    """

    def __init__(self, make: Callable[[], BaseException]) -> None:
        self.make = make


def _entries(*names: str, positions: list[Any] | None = None) -> list[dict]:
    """Well-formed reply entries for the named texts, in the given (server) order.

    With ``positions`` each entry carries ``index`` set to the matching value; without, no entry has one.
    """
    entries: list[dict] = []
    for i, name in enumerate(names):
        entry: dict[str, Any] = {"object": "embedding", "embedding": list(VEC[name])}
        if positions is not None:
            entry["index"] = positions[i]
        entries.append(entry)
    return entries


def _reply(data: object) -> dict:
    """An embeddings reply body in the server's shape, with ``data`` as the entry list."""
    return {"object": "list", "data": data, "model": EMBED_MODEL, "usage": {"prompt_tokens": 3, "total_tokens": 3}}


def _serve(monkeypatch, client: Client, reply: object, texts: list[str]) -> list[dict]:
    """Replace the ``_request`` seam with a fake embeddings server.

    The fake checks it receives ``POST /embeddings`` for the given texts and model, then returns ``reply``
    (or raises it, for a ``_Raises`` reply). Returns the list of recorded requests.
    """
    calls: list[dict] = []

    async def fake_request(method: str, path: str, **kw: Any) -> object:
        calls.append({"method": method, "path": path, "json": dict(kw.get("json") or {})})
        assert method == "POST"
        assert path == "/embeddings"
        assert list(kw["json"]["input"]) == texts
        assert kw["json"]["model"] == EMBED_MODEL
        if isinstance(reply, _Raises):
            raise reply.make()
        # Case data is shared across tests and backends; hand out a copy so no call can alter another's reply.
        return copy.deepcopy(reply)

    monkeypatch.setattr(client, "_request", fake_request)
    return calls


async def _embed_ok(monkeypatch, client: Client, texts: list[str], reply: object) -> list[list[float]]:
    """Run ``embed_batch`` against a canned reply and return its result; the request must be sent once."""
    calls = _serve(monkeypatch, client, reply, texts)
    result = await client.embed_batch(texts)
    assert len(calls) == 1
    return result


async def _embed_fails(monkeypatch, client: Client, texts: list[str], reply: object) -> LLMError:
    """Run ``embed_batch`` against a canned reply, require ``LLMError`` and return it."""
    _serve(monkeypatch, client, reply, texts)
    with pytest.raises(LLMError) as excinfo:
        await client.embed_batch(texts)
    return excinfo.value


def _content_type_error() -> aiohttp.ContentTypeError:
    """The decode error the response's JSON reader raises for a body it cannot read as JSON."""
    url = f"{OPENAI_BASE_URL}/embeddings"
    info = SimpleNamespace(url=url, real_url=url, method="POST", headers={})
    return aiohttp.ContentTypeError(info, (), message="Attempt to decode JSON with unexpected mimetype: text/html")


# ---------------------------------------------------------------------------
# Cases (shared by both backends)
# ---------------------------------------------------------------------------

# DoD-3: replies whose entries do not all carry an integer position keep the server's order.
# "Integer position" excludes booleans and text (DEC-002-8); partial positions keep server order.
SERVER_ORDER_CASES = [
    pytest.param(
        TEXTS,
        _entries("gamma", "alpha", "beta"),
        [VEC["gamma"], VEC["alpha"], VEC["beta"]],
        id="no-positions",
    ),
    pytest.param(
        TEXTS,
        [
            {"index": 2, "embedding": VEC["gamma"]},
            {"embedding": VEC["alpha"]},
            {"index": 1, "embedding": VEC["beta"]},
        ],
        [VEC["gamma"], VEC["alpha"], VEC["beta"]],
        id="only-some-positions",
    ),
    pytest.param(
        ["alpha", "beta"],
        _entries("beta", "alpha", positions=[True, False]),
        [VEC["beta"], VEC["alpha"]],
        id="boolean-positions",
    ),
    pytest.param(
        ["alpha", "beta"],
        _entries("beta", "alpha", positions=["1", "0"]),
        [VEC["beta"], VEC["alpha"]],
        id="text-positions",
    ),
]

# DoD-4: entry count differs from the number of texts.
COUNT_MISMATCH_CASES = [
    pytest.param(_entries("alpha", "beta", positions=[0, 1]), id="fewer-with-positions"),
    pytest.param(_entries("alpha", "beta"), id="fewer-without-positions"),
    pytest.param(
        _entries("alpha", "beta", "gamma") + [{"object": "embedding", "embedding": [0.7, 0.8]}], id="more-no-positions"
    ),
    pytest.param(
        _entries("alpha", "beta", "gamma", positions=[0, 1, 2]) + [{"index": 3, "embedding": [0.7, 0.8]}],
        id="more-with-positions",
    ),
]

# DoD-5: every entry carries an integer position, but the set is not exactly 0..n-1 (n = 3).
BAD_POSITION_CASES = [
    pytest.param([0, 1, 1], id="duplicate"),
    pytest.param([0, 1, 3], id="gap"),
    pytest.param([-1, 0, 1], id="negative-out-of-range"),
    pytest.param([1, 2, 3], id="out-of-range"),
]

# DoD-6: one malformed entry among well-formed ones.
BAD_ENTRY_CASES = [
    pytest.param([0.3, 0.4], id="entry-is-list"),
    pytest.param("beta", id="entry-is-text"),
    pytest.param(None, id="entry-is-null"),
    pytest.param({"object": "embedding", "index": 1}, id="no-vector"),
    pytest.param({"index": 1, "embedding": "0.3,0.4"}, id="vector-is-text"),
    pytest.param({"index": 1, "embedding": None}, id="vector-is-null"),
    pytest.param({"index": 1, "embedding": {"0": 0.3, "1": 0.4}}, id="vector-is-mapping"),
    pytest.param({"index": 1, "embedding": []}, id="empty-vector"),
]

# DoD-7: the reply itself is not a mapping, or its entry list is missing or not a list.
BAD_REPLY_CASES = [
    pytest.param(_entries("alpha", "beta", "gamma"), id="reply-is-list"),
    pytest.param(None, id="reply-is-null"),
    pytest.param("ok", id="reply-is-text"),
    pytest.param(42, id="reply-is-number"),
    pytest.param({"object": "list", "model": EMBED_MODEL}, id="data-missing"),
    pytest.param(_reply({"0": {"embedding": VEC["alpha"]}}), id="data-is-mapping"),
    pytest.param(_reply("alpha,beta,gamma"), id="data-is-text"),
    pytest.param(_reply(None), id="data-is-null"),
]

# DoD-8: a successful reply that cannot be read as JSON.
UNREADABLE_CASES = [
    pytest.param(b"<html><body>Bad gateway</body></html>", id="bytes-body"),
    pytest.param(b"", id="empty-bytes-body"),
    pytest.param(_Raises(lambda: json.JSONDecodeError("Expecting value", "<html>", 0)), id="json-decode-error"),
    pytest.param(_Raises(_content_type_error), id="content-type-error"),
]

# DoD-9: refusals the request helper raises as the library's error (non-2xx and transport).
REFUSAL_CASES = [
    pytest.param(
        lambda: LLMError("Request to /embeddings failed with status 503", description="upstream overloaded"),
        id="non-2xx",
    ),
    pytest.param(
        lambda: LLMError("Request to /embeddings failed: connection failure", description="Cannot connect to host"),
        id="transport",
    ),
]

# DoD-10: a one-text reply that holds no vector.
NO_VECTOR_CASES = [
    pytest.param(_reply([]), id="no-entries"),
    pytest.param(_reply([{"object": "embedding", "index": 0}]), id="entry-without-vector"),
]

BACKENDS = [pytest.param(_openai, id="openai"), pytest.param(_llamaswap, id="llamaswap")]


# ---------------------------------------------------------------------------
# DoD-1 — n texts, n well-formed entries, n vectors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_embed_batch_returns_one_vector_per_text__DoD1(monkeypatch):
    """DoD-1: for 3 texts and 3 well-formed entries, OpenAI ``embed_batch`` returns exactly 3 vectors."""
    reply = _reply(_entries("alpha", "beta", "gamma", positions=[0, 1, 2]))
    result = await _embed_ok(monkeypatch, _openai(), TEXTS, reply)
    assert result == [VEC["alpha"], VEC["beta"], VEC["gamma"]]


@pytest.mark.asyncio
async def test_llamaswap_embed_batch_returns_one_vector_per_text__DoD1(monkeypatch):
    """DoD-1: for 3 texts and 3 well-formed entries, llama-swap ``embed_batch`` returns exactly 3 vectors."""
    reply = _reply(_entries("alpha", "beta", "gamma", positions=[0, 1, 2]))
    result = await _embed_ok(monkeypatch, _llamaswap(), TEXTS, reply)
    assert result == [VEC["alpha"], VEC["beta"], VEC["gamma"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("make_client", BACKENDS)
async def test_embed_batch_with_tracing_returns_one_vector_per_text__DoD1(monkeypatch, tmp_path, make_client):
    """DoD-1: with tracing on (checks run inside the trace session), n texts still give exactly n vectors."""
    # Tracing is env-driven and read when the client is built (DEC-000-4), so set it first.
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(tmp_path))
    reply = _reply(_entries("alpha", "beta", "gamma", positions=[0, 1, 2]))
    result = await _embed_ok(monkeypatch, make_client(), TEXTS, reply)
    assert result == [VEC["alpha"], VEC["beta"], VEC["gamma"]]


# ---------------------------------------------------------------------------
# DoD-2 — positions out of order: vectors come back in the texts' order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_embed_batch_orders_vectors_by_position__DoD2(monkeypatch):
    """DoD-2: OpenAI entries sent as positions 2, 0, 1 come back in the texts' order."""
    reply = _reply(_entries("gamma", "alpha", "beta", positions=[2, 0, 1]))
    result = await _embed_ok(monkeypatch, _openai(), TEXTS, reply)
    assert result == [VEC["alpha"], VEC["beta"], VEC["gamma"]]


@pytest.mark.asyncio
async def test_llamaswap_embed_batch_orders_vectors_by_position__DoD2(monkeypatch):
    """DoD-2: llama-swap entries sent as positions 2, 0, 1 come back in the texts' order."""
    reply = _reply(_entries("gamma", "alpha", "beta", positions=[2, 0, 1]))
    result = await _embed_ok(monkeypatch, _llamaswap(), TEXTS, reply)
    assert result == [VEC["alpha"], VEC["beta"], VEC["gamma"]]


# ---------------------------------------------------------------------------
# DoD-3 — no (usable) positions: server order is kept
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("texts", "data", "expected"), SERVER_ORDER_CASES)
async def test_openai_embed_batch_keeps_server_order_without_positions__DoD3(monkeypatch, texts, data, expected):
    """DoD-3: OpenAI entries without integer positions on every entry come back in the reply's order."""
    assert await _embed_ok(monkeypatch, _openai(), texts, _reply(data)) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("texts", "data", "expected"), SERVER_ORDER_CASES)
async def test_llamaswap_embed_batch_keeps_server_order_without_positions__DoD3(monkeypatch, texts, data, expected):
    """DoD-3: llama-swap entries without integer positions on every entry come back in the reply's order."""
    assert await _embed_ok(monkeypatch, _llamaswap(), texts, _reply(data)) == expected


# ---------------------------------------------------------------------------
# DoD-4 — fewer or more entries than texts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("data", COUNT_MISMATCH_CASES)
async def test_openai_embed_batch_count_mismatch_raises_llmerror__DoD4(monkeypatch, data):
    """DoD-4: an OpenAI reply with fewer or more entries than texts ends with ``LLMError``."""
    await _embed_fails(monkeypatch, _openai(), TEXTS, _reply(data))


@pytest.mark.asyncio
@pytest.mark.parametrize("data", COUNT_MISMATCH_CASES)
async def test_llamaswap_embed_batch_count_mismatch_raises_llmerror__DoD4(monkeypatch, data):
    """DoD-4: a llama-swap reply with fewer or more entries than texts ends with ``LLMError``."""
    await _embed_fails(monkeypatch, _llamaswap(), TEXTS, _reply(data))


@pytest.mark.asyncio
@pytest.mark.parametrize("make_client", BACKENDS)
async def test_embed_batch_with_tracing_count_mismatch_raises_llmerror__DoD4(monkeypatch, tmp_path, make_client):
    """DoD-4: with tracing on, a reply with fewer entries than texts still ends with ``LLMError``."""
    # Tracing is env-driven and read when the client is built (DEC-000-4), so set it first.
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(tmp_path))
    await _embed_fails(monkeypatch, make_client(), TEXTS, _reply(_entries("alpha", "beta", positions=[0, 1])))


# ---------------------------------------------------------------------------
# DoD-5 — positions that are not exactly 0..n-1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("positions", BAD_POSITION_CASES)
async def test_openai_embed_batch_bad_positions_raise_llmerror__DoD5(monkeypatch, positions):
    """DoD-5: OpenAI positions with a duplicate, a gap or an out-of-range value end with ``LLMError``."""
    reply = _reply(_entries("alpha", "beta", "gamma", positions=positions))
    await _embed_fails(monkeypatch, _openai(), TEXTS, reply)


@pytest.mark.asyncio
@pytest.mark.parametrize("positions", BAD_POSITION_CASES)
async def test_llamaswap_embed_batch_bad_positions_raise_llmerror__DoD5(monkeypatch, positions):
    """DoD-5: llama-swap positions with a duplicate, a gap or an out-of-range value end with ``LLMError``."""
    reply = _reply(_entries("alpha", "beta", "gamma", positions=positions))
    await _embed_fails(monkeypatch, _llamaswap(), TEXTS, reply)


# ---------------------------------------------------------------------------
# DoD-6 — malformed entries
# ---------------------------------------------------------------------------


def _with_bad_middle_entry(bad_entry: object) -> dict:
    """A 3-entry reply whose middle entry is replaced by ``bad_entry``; the others are well-formed."""
    data: list[object] = list(_entries("alpha", "beta", "gamma", positions=[0, 1, 2]))
    data[1] = bad_entry
    return _reply(data)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_entry", BAD_ENTRY_CASES)
async def test_openai_embed_batch_malformed_entry_raises_llmerror__DoD6(monkeypatch, bad_entry):
    """DoD-6: an OpenAI entry that is no mapping, lacks a vector, or has a non-list/empty vector gives LLMError."""
    await _embed_fails(monkeypatch, _openai(), TEXTS, _with_bad_middle_entry(bad_entry))


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_entry", BAD_ENTRY_CASES)
async def test_llamaswap_embed_batch_malformed_entry_raises_llmerror__DoD6(monkeypatch, bad_entry):
    """DoD-6: a llama-swap entry that is no mapping, lacks a vector, or has a non-list/empty vector gives LLMError."""
    await _embed_fails(monkeypatch, _llamaswap(), TEXTS, _with_bad_middle_entry(bad_entry))


# ---------------------------------------------------------------------------
# DoD-7 — malformed reply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", BAD_REPLY_CASES)
async def test_openai_embed_batch_malformed_reply_raises_llmerror__DoD7(monkeypatch, reply):
    """DoD-7: an OpenAI reply that is no mapping, or whose entry list is missing/not a list, gives LLMError."""
    await _embed_fails(monkeypatch, _openai(), TEXTS, reply)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", BAD_REPLY_CASES)
async def test_llamaswap_embed_batch_malformed_reply_raises_llmerror__DoD7(monkeypatch, reply):
    """DoD-7: a llama-swap reply that is no mapping, or whose entry list is missing/not a list, gives LLMError."""
    await _embed_fails(monkeypatch, _llamaswap(), TEXTS, reply)


# ---------------------------------------------------------------------------
# DoD-8 — unreadable successful body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", UNREADABLE_CASES)
async def test_openai_embed_batch_unreadable_body_raises_llmerror__DoD8(monkeypatch, reply):
    """DoD-8: an OpenAI 2xx body that is bytes, or whose JSON decoding fails, ends with ``LLMError``."""
    await _embed_fails(monkeypatch, _openai(), TEXTS, reply)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", UNREADABLE_CASES)
async def test_llamaswap_embed_batch_unreadable_body_raises_llmerror__DoD8(monkeypatch, reply):
    """DoD-8: a llama-swap 2xx body that is bytes, or whose JSON decoding fails, ends with ``LLMError``."""
    await _embed_fails(monkeypatch, _llamaswap(), TEXTS, reply)


# ---------------------------------------------------------------------------
# DoD-9 — request-helper refusal passes through unchanged
# ---------------------------------------------------------------------------


async def _assert_refusal_unchanged(monkeypatch, client: Client, make_refusal: Callable[[], LLMError]) -> None:
    """The ``LLMError`` raised by the request helper must reach the caller with the same message and description."""
    expected = make_refusal()
    raised = await _embed_fails(monkeypatch, client, TEXTS, _Raises(lambda: expected))
    # The text form is ``message: description`` (error contract), so equal text means both parts are kept.
    assert str(raised) == str(expected)
    assert raised.args == expected.args


@pytest.mark.asyncio
@pytest.mark.parametrize("make_refusal", REFUSAL_CASES)
async def test_openai_embed_batch_request_refusal_passes_through__DoD9(monkeypatch, make_refusal):
    """DoD-9: a refusal the request helper raises as ``LLMError`` reaches the OpenAI caller unchanged."""
    await _assert_refusal_unchanged(monkeypatch, _openai(), make_refusal)


@pytest.mark.asyncio
@pytest.mark.parametrize("make_refusal", REFUSAL_CASES)
async def test_llamaswap_embed_batch_request_refusal_passes_through__DoD9(monkeypatch, make_refusal):
    """DoD-9: a refusal the request helper raises as ``LLMError`` reaches the llama-swap caller unchanged."""
    await _assert_refusal_unchanged(monkeypatch, _llamaswap(), make_refusal)


# ---------------------------------------------------------------------------
# DoD-10 — embed: one text, one vector; no vector is an error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_embed_returns_single_vector__DoD10(monkeypatch):
    """DoD-10: OpenAI ``embed`` returns the single vector for one text."""
    client = _openai()
    _serve(monkeypatch, client, _reply(_entries("beta", positions=[0])), ["beta"])
    assert await client.embed("beta") == VEC["beta"]


@pytest.mark.asyncio
async def test_llamaswap_embed_returns_single_vector__DoD10(monkeypatch):
    """DoD-10: llama-swap ``embed`` returns the single vector for one text."""
    client = _llamaswap()
    _serve(monkeypatch, client, _reply(_entries("beta")), ["beta"])
    assert await client.embed("beta") == VEC["beta"]


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", NO_VECTOR_CASES)
async def test_openai_embed_without_vector_raises_llmerror__DoD10(monkeypatch, reply):
    """DoD-10: OpenAI ``embed`` ends with ``LLMError`` when the reply holds no vector."""
    client = _openai()
    _serve(monkeypatch, client, reply, ["beta"])
    with pytest.raises(LLMError):
        await client.embed("beta")


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", NO_VECTOR_CASES)
async def test_llamaswap_embed_without_vector_raises_llmerror__DoD10(monkeypatch, reply):
    """DoD-10: llama-swap ``embed`` ends with ``LLMError`` when the reply holds no vector."""
    client = _llamaswap()
    _serve(monkeypatch, client, reply, ["beta"])
    with pytest.raises(LLMError):
        await client.embed("beta")
