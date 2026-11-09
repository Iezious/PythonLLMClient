"""Error kinds at the chat-completions classes' own raise sites (feature 006.tolerant-tool-loop, step 005).

Checks that the ``LLMError`` raised by ``OpenAIChatCompletionsClient`` and ``LlamaSwapChatCompletionsClient``
for a malformed or unreadable model-list or embeddings reply (DoD-4) and for a ``chat_json`` reply that does not
parse as JSON under an expected shape (DoD-5) carries kind ``invalid_reply`` (DEC-arch-19), with the error text
keeping its form (``message: description``, US-067.AC-3). Every expected value comes from the step's Definition
of done, DEC-arch-19 and docs/architecture/interfaces.md § Exposed: error contract.
"""

import copy
import json
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

from llm.exceptions import LLMError
from llm.llama_swap_chat_completions import LlamaSwapChatCompletionsClient
from llm.openai_chat_completions import OpenAIChatCompletionsClient

OPENAI_BASE_URL = "https://api.example.test/v1"
LLAMASWAP_BASE_URL = "http://localhost:8080/v1"
MODEL = "some-model"
CREDENTIAL = "sk-test"

USER_MESSAGES = [{"role": "user", "content": "Rate the answer and explain."}]
EMBED_TEXTS = ["alpha", "beta"]

BACKENDS = [pytest.param("openai", id="openai"), pytest.param("llamaswap", id="llamaswap")]

# An expected shape for chat_json (DoD-5), so the parse check runs.
EXPECTED_SHAPE: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


# ---------------------------------------------------------------------------
# Clients and seams
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch):
    """Clear the trace switches so a developer's environment cannot turn tracing on (DEC-000-48)."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _client(backend: str) -> OpenAIChatCompletionsClient | LlamaSwapChatCompletionsClient:
    """A chat-completions client of the given backend, built with an explicit test credential (tests/AGENTS.md)."""
    if backend == "openai":
        return OpenAIChatCompletionsClient(model=MODEL, base_url=OPENAI_BASE_URL, bearer_token=CREDENTIAL)
    return LlamaSwapChatCompletionsClient(model=MODEL, base_url=LLAMASWAP_BASE_URL, bearer_token=CREDENTIAL)


class _Raises:
    """Marks a canned reply as an exception the request seam raises instead of returning a body.

    A factory is stored so each test raises a fresh exception object."""

    def __init__(self, make) -> None:
        self.make = make


def _serve(monkeypatch, client: Any, method: str, path: str, reply: Any) -> list[dict]:
    """Replace the ``_request`` seam with a fake that checks it receives ``method`` / ``path`` and answers
    ``reply`` (deep copy), or raises it for a ``_Raises`` reply; returns the recorded requests."""
    calls: list[dict] = []

    async def fake_request(m, p, **kw):
        calls.append({"method": m, "path": p, "json": copy.deepcopy(kw.get("json"))})
        assert m == method
        assert p == path
        if isinstance(reply, _Raises):
            raise reply.make()
        # Case data is shared across tests; hand out a copy so no call can alter another's reply.
        return copy.deepcopy(reply)

    monkeypatch.setattr(client, "_request", fake_request)
    return calls


def _assert_text_form(err: LLMError) -> None:
    """US-067.AC-3 / interfaces.md § error contract: the text form stays ``message: description`` (or
    ``message`` alone), ``kind`` is an attribute only, and ``status`` is none for a failure that is not a
    non-2xx reply."""
    message = err.args[0]
    expected = f"{message}: {err.description}" if err.description else message
    assert str(err) == expected
    assert err.status is None


def _content_type_error() -> aiohttp.ContentTypeError:
    """The decode error the response's JSON reader raises for a 2xx body it cannot read as JSON."""
    url = f"{OPENAI_BASE_URL}/models"
    info = SimpleNamespace(url=url, real_url=url, method="GET", headers={})
    return aiohttp.ContentTypeError(info, (), message="Attempt to decode JSON with unexpected mimetype: text/html")


# A 2xx body that cannot be read as JSON (DEC-002-10).
UNREADABLE_BODIES = [
    pytest.param(b"<html><body>Bad gateway</body></html>", id="bytes-body"),
    pytest.param(b"", id="empty-bytes-body"),
    pytest.param(_Raises(lambda: json.JSONDecodeError("Expecting value", "<html>", 0)), id="json-decode-error"),
    pytest.param(_Raises(_content_type_error), id="content-type-error"),
]


# ---------------------------------------------------------------------------
# DoD-4 — model-list and embeddings replies: malformed or unreadable
# ---------------------------------------------------------------------------

# A malformed model list or an entry without a usable id (DEC-002-9).
MALFORMED_MODEL_LISTS = [
    pytest.param([{"id": "a"}], id="reply-is-list"),
    pytest.param(None, id="reply-is-null"),
    pytest.param({"object": "list"}, id="data-missing"),
    pytest.param({"object": "list", "data": "a,b"}, id="data-is-text"),
    pytest.param({"object": "list", "data": [{"id": "a"}, {"object": "model"}]}, id="entry-without-id"),
    pytest.param({"object": "list", "data": [{"id": "a"}, {"id": ""}]}, id="entry-with-empty-id"),
    pytest.param({"object": "list", "data": [{"id": "a"}, "b"]}, id="entry-is-text"),
]

# A malformed embeddings reply, an entry without a vector, a count not matching the input, or an empty
# embedding result (DEC-002-8).
MALFORMED_EMBEDDINGS = [
    pytest.param([[0.1, 0.2], [0.3, 0.4]], id="reply-is-list"),
    pytest.param({"object": "list"}, id="data-missing"),
    pytest.param({"object": "list", "data": "x"}, id="data-is-text"),
    pytest.param(
        {"object": "list", "data": [{"embedding": [0.1]}, {"object": "embedding"}]}, id="entry-without-vector"
    ),
    pytest.param({"object": "list", "data": [{"embedding": [0.1, 0.2]}]}, id="count-fewer"),
    pytest.param(
        {"object": "list", "data": [{"embedding": [0.1]}, {"embedding": [0.2]}, {"embedding": [0.3]}]},
        id="count-more",
    ),
    pytest.param({"object": "list", "data": []}, id="empty-result"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", ["list_models", "get_models"])
@pytest.mark.parametrize("reply", [*MALFORMED_MODEL_LISTS, *UNREADABLE_BODIES])
async def test_chat_completions_models_bad_reply_is_invalid_reply__DoD4(monkeypatch, backend, method, reply):
    """DoD-4: a malformed or unreadable model-list reply raises ``LLMError`` with kind ``invalid_reply`` from
    ``list_models`` and ``get_models`` on both chat-completions classes (US-067.AC-2, DEC-arch-19)."""
    client = _client(backend)
    _serve(monkeypatch, client, "GET", "/models", reply)

    with pytest.raises(LLMError) as excinfo:
        await getattr(client, method)()

    assert excinfo.value.kind == "invalid_reply"
    _assert_text_form(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("reply", [*MALFORMED_EMBEDDINGS, *UNREADABLE_BODIES])
async def test_chat_completions_embed_batch_bad_reply_is_invalid_reply__DoD4(monkeypatch, backend, reply):
    """DoD-4: a malformed or unreadable embeddings reply (entry without a vector, count mismatch included)
    raises ``LLMError`` with kind ``invalid_reply`` from ``embed_batch`` (US-067.AC-2, DEC-arch-19)."""
    client = _client(backend)
    _serve(monkeypatch, client, "POST", "/embeddings", reply)

    with pytest.raises(LLMError) as excinfo:
        await client.embed_batch(list(EMBED_TEXTS))

    assert excinfo.value.kind == "invalid_reply"
    _assert_text_form(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
async def test_chat_completions_embed_entry_without_vector_is_invalid_reply__DoD4(monkeypatch, backend):
    """DoD-4: the single-text ``embed`` (which goes through ``embed_batch``) on an entry without a vector raises
    ``LLMError`` with kind ``invalid_reply`` (US-067.AC-2)."""
    client = _client(backend)
    _serve(monkeypatch, client, "POST", "/embeddings", {"object": "list", "data": [{"object": "embedding"}]})

    with pytest.raises(LLMError) as excinfo:
        await client.embed("alpha")

    assert excinfo.value.kind == "invalid_reply"
    _assert_text_form(excinfo.value)


# ---------------------------------------------------------------------------
# DoD-5 — chat_json with an expected shape: reply that does not parse as JSON
# ---------------------------------------------------------------------------


def _chat_reply(content: str) -> dict:
    """A chat-completions reply whose single choice carries ``content``."""
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "content",
    [pytest.param("Sure! It is great.", id="prose"), pytest.param('{"answer": "unterminated', id="truncated-object")],
)
async def test_chat_completions_chat_json_not_json_is_invalid_reply__DoD5(monkeypatch, backend, content):
    """DoD-5: ``chat_json`` with an expected shape whose reply does not parse as JSON raises ``LLMError`` with
    kind ``invalid_reply`` (US-067.AC-2, DEC-arch-19)."""
    client = _client(backend)
    _serve(monkeypatch, client, "POST", "/chat/completions", _chat_reply(content))

    with pytest.raises(LLMError) as excinfo:
        await client.chat_json(copy.deepcopy(USER_MESSAGES), response_schema=EXPECTED_SHAPE)

    assert excinfo.value.kind == "invalid_reply"
    _assert_text_form(excinfo.value)
