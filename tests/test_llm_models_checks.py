"""Model-list reply checks on the OpenAI-family clients (plan 002.edge-fixes, step 003).

Covers ``list_models`` and ``get_models`` on ``OpenAIChatCompletionsClient`` and
``LlamaSwapChatCompletionsClient`` (the chat-completions regression suite, feature 003 step 001 DoD-11): ids in the
server's order and entries returned exactly as sent, and malformed or unreadable replies ending with
``LLMError`` from both methods (DEC-002-9, DEC-002-10). Every expected value comes from the step's
Definition of done (DoD-11..DoD-14) and its decisions.
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

MODEL = "some-model"
OPENAI_BASE_URL = "https://api.example.test/v1"
LLAMASWAP_BASE_URL = "http://localhost:8080/v1"

# Deliberately not in alphabetical order, with differing extra fields, so "server order" and
# "entries exactly as sent" are both observable.
SAMPLE_MODELS: list[dict[str, Any]] = [
    {"id": "zeta-chat", "object": "model", "created": 1700000300, "owned_by": "org-z"},
    {"id": "alpha-embed", "object": "model", "created": 1700000100, "owned_by": "org-a", "meta": {"ctx": 8192}},
    {"id": "mid-coder", "object": "model", "created": 1700000200, "owned_by": "org-m", "aliases": ["coder"]},
]

Client = OpenAIChatCompletionsClient | LlamaSwapChatCompletionsClient
METHODS = ["list_models", "get_models"]


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
    return OpenAIChatCompletionsClient(model=MODEL, base_url=OPENAI_BASE_URL, bearer_token="sk-test")


def _llamaswap() -> LlamaSwapChatCompletionsClient:
    """A llama-swap client on a local test address with an explicit test credential."""
    return LlamaSwapChatCompletionsClient(model=MODEL, base_url=LLAMASWAP_BASE_URL, bearer_token="sk-test")


class _Raises:
    """Marks a fake reply as an exception the request helper raises instead of returning a body.

    A factory is stored (not an instance) so each test raises a fresh exception object.
    """

    def __init__(self, make: Callable[[], BaseException]) -> None:
        self.make = make


def _reply(data: object) -> dict:
    """A model-list reply body in the server's shape, with ``data`` as the entry list."""
    return {"object": "list", "data": data}


def _serve(monkeypatch, client: Client, reply: object) -> list[dict]:
    """Replace the ``_request`` seam with a fake model-list server.

    The fake checks it receives ``GET /models`` and returns a copy of ``reply`` (or raises it, for a
    ``_Raises`` reply). Returns the list of recorded requests.
    """
    calls: list[dict] = []

    async def fake_request(method: str, path: str, **kw: Any) -> object:
        calls.append({"method": method, "path": path})
        assert method == "GET"
        assert path == "/models"
        if isinstance(reply, _Raises):
            raise reply.make()
        # Case data is shared across tests and backends; hand out a copy so no call can alter another's reply.
        return copy.deepcopy(reply)

    monkeypatch.setattr(client, "_request", fake_request)
    return calls


async def _call_fails(monkeypatch, client: Client, method: str, reply: object) -> LLMError:
    """Call ``method`` (``list_models`` or ``get_models``) against a canned reply, require ``LLMError``."""
    _serve(monkeypatch, client, reply)
    with pytest.raises(LLMError) as excinfo:
        await getattr(client, method)()
    return excinfo.value


def _content_type_error() -> aiohttp.ContentTypeError:
    """The decode error the response's JSON reader raises for a body it cannot read as JSON."""
    url = f"{OPENAI_BASE_URL}/models"
    info = SimpleNamespace(url=url, real_url=url, method="GET", headers={})
    return aiohttp.ContentTypeError(info, (), message="Attempt to decode JSON with unexpected mimetype: text/html")


def _with_bad_middle_entry(bad_entry: object) -> dict:
    """A 3-entry reply whose middle entry is replaced by ``bad_entry``; the others are well-formed."""
    data: list[object] = copy.deepcopy(SAMPLE_MODELS)
    data[1] = bad_entry
    return _reply(data)


# ---------------------------------------------------------------------------
# Cases (shared by both backends and both methods)
# ---------------------------------------------------------------------------

# DoD-12: one malformed entry among well-formed ones.
BAD_ENTRY_CASES = [
    pytest.param("alpha-embed", id="entry-is-text"),
    pytest.param(["alpha-embed"], id="entry-is-list"),
    pytest.param(None, id="entry-is-null"),
    pytest.param({"object": "model", "owned_by": "org-a"}, id="no-id"),
    pytest.param({"id": 7, "object": "model"}, id="id-is-number"),
    pytest.param({"id": None, "object": "model"}, id="id-is-null"),
    pytest.param({"id": ["alpha-embed"], "object": "model"}, id="id-is-list"),
    pytest.param({"id": "", "object": "model"}, id="empty-id"),
]

# DoD-13: the reply itself is not a mapping, or its entry list is missing or not a list.
BAD_REPLY_CASES = [
    pytest.param(copy.deepcopy(SAMPLE_MODELS), id="reply-is-list"),
    pytest.param(None, id="reply-is-null"),
    pytest.param("ok", id="reply-is-text"),
    pytest.param(42, id="reply-is-number"),
    pytest.param({"object": "list"}, id="data-missing"),
    pytest.param({"error": "boom"}, id="error-without-data"),
    pytest.param(_reply({"zeta-chat": {"object": "model"}}), id="data-is-mapping"),
    pytest.param(_reply("zeta-chat,alpha-embed"), id="data-is-text"),
    pytest.param(_reply(None), id="data-is-null"),
]

# DoD-14: a successful reply that cannot be read as JSON.
UNREADABLE_CASES = [
    pytest.param(b"<html><body>Bad gateway</body></html>", id="bytes-body"),
    pytest.param(b"", id="empty-bytes-body"),
    pytest.param(_Raises(lambda: json.JSONDecodeError("Expecting value", "<html>", 0)), id="json-decode-error"),
    pytest.param(_Raises(_content_type_error), id="content-type-error"),
]


# ---------------------------------------------------------------------------
# DoD-11 — ids in server order; entries exactly as sent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_list_models_returns_ids_in_server_order__DoD11(monkeypatch):
    """DoD-11: OpenAI ``list_models`` returns the ids in the order the server sent them."""
    client = _openai()
    _serve(monkeypatch, client, _reply(SAMPLE_MODELS))
    assert await client.list_models() == ["zeta-chat", "alpha-embed", "mid-coder"]


@pytest.mark.asyncio
async def test_llamaswap_list_models_returns_ids_in_server_order__DoD11(monkeypatch):
    """DoD-11: llama-swap ``list_models`` returns the ids in the order the server sent them."""
    client = _llamaswap()
    _serve(monkeypatch, client, _reply(SAMPLE_MODELS))
    assert await client.list_models() == ["zeta-chat", "alpha-embed", "mid-coder"]


@pytest.mark.asyncio
async def test_openai_get_models_returns_entries_as_sent__DoD11(monkeypatch):
    """DoD-11: OpenAI ``get_models`` returns the entries exactly as sent, extra fields and order included."""
    client = _openai()
    _serve(monkeypatch, client, _reply(SAMPLE_MODELS))
    assert await client.get_models() == SAMPLE_MODELS


@pytest.mark.asyncio
async def test_llamaswap_get_models_returns_entries_as_sent__DoD11(monkeypatch):
    """DoD-11: llama-swap ``get_models`` returns the entries exactly as sent, extra fields and order included."""
    client = _llamaswap()
    _serve(monkeypatch, client, _reply(SAMPLE_MODELS))
    assert await client.get_models() == SAMPLE_MODELS


# ---------------------------------------------------------------------------
# DoD-12 — malformed entries, from both methods
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bad_entry", BAD_ENTRY_CASES)
async def test_openai_models_malformed_entry_raises_llmerror__DoD12(monkeypatch, method, bad_entry):
    """DoD-12: an OpenAI entry that is no mapping, lacks an id, or has a non-text/empty id gives LLMError."""
    await _call_fails(monkeypatch, _openai(), method, _with_bad_middle_entry(bad_entry))


@pytest.mark.asyncio
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bad_entry", BAD_ENTRY_CASES)
async def test_llamaswap_models_malformed_entry_raises_llmerror__DoD12(monkeypatch, method, bad_entry):
    """DoD-12: a llama-swap entry that is no mapping, lacks an id, or has a non-text/empty id gives LLMError."""
    await _call_fails(monkeypatch, _llamaswap(), method, _with_bad_middle_entry(bad_entry))


# ---------------------------------------------------------------------------
# DoD-13 — malformed reply, from both methods
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("reply", BAD_REPLY_CASES)
async def test_openai_models_malformed_reply_raises_llmerror__DoD13(monkeypatch, method, reply):
    """DoD-13: an OpenAI reply that is no mapping, or whose entry list is missing/not a list, gives LLMError."""
    await _call_fails(monkeypatch, _openai(), method, reply)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("reply", BAD_REPLY_CASES)
async def test_llamaswap_models_malformed_reply_raises_llmerror__DoD13(monkeypatch, method, reply):
    """DoD-13: a llama-swap reply that is no mapping, or whose entry list is missing/not a list, gives LLMError."""
    await _call_fails(monkeypatch, _llamaswap(), method, reply)


# ---------------------------------------------------------------------------
# DoD-14 — unreadable successful body, from both methods
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("reply", UNREADABLE_CASES)
async def test_openai_models_unreadable_body_raises_llmerror__DoD14(monkeypatch, method, reply):
    """DoD-14: an OpenAI 2xx body that is bytes, or whose JSON decoding fails, gives ``LLMError``."""
    await _call_fails(monkeypatch, _openai(), method, reply)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("reply", UNREADABLE_CASES)
async def test_llamaswap_models_unreadable_body_raises_llmerror__DoD14(monkeypatch, method, reply):
    """DoD-14: a llama-swap 2xx body that is bytes, or whose JSON decoding fails, gives ``LLMError``."""
    await _call_fails(monkeypatch, _llamaswap(), method, reply)
