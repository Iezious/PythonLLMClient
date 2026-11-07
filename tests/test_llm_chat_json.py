"""JSON-only answers (``chat_json``) on the OpenAI-family clients (plan 002.edge-fixes, step 003).

Covers the parse-only check (DEC-002-7): with an expected shape whose fields have no defaults, any valid
single JSON value is returned unchanged on ``OpenAIAPIClient`` and ``LlamaSwapAPIClient``; a reply that is
not JSON ends with ``LLMError`` when a shape is given and is returned as is without one. Every expected
value comes from the step's Definition of done (DoD-15, DoD-16).
"""

from typing import Any

import pytest

from llm.exceptions import LLMError
from llm.llama_swap import LlamaSwapAPIClient
from llm.openai import OpenAIAPIClient

MODEL = "some-model"
OPENAI_BASE_URL = "https://api.example.test/v1"
LLAMASWAP_BASE_URL = "http://localhost:8080/v1"

USER_MESSAGES = [{"role": "user", "content": "Rate the answer and explain."}]

# An expected shape whose fields carry no defaults (DoD-15), so a field-presence check, if any, would fire.
EXPECTED_SHAPE: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "score": {"type": "number"}},
    "required": ["answer", "score"],
}

# DoD-15: valid single JSON values that are not objects with the shape's fields.
SINGLE_JSON_VALUES = [
    pytest.param("42", id="integer"),
    pytest.param("3.5", id="decimal"),
    pytest.param("true", id="true"),
    pytest.param("null", id="null"),
    pytest.param('"just a string"', id="string"),
    pytest.param("[1, 2, 3]", id="list"),
]

# DoD-16: replies that are not JSON at all.
NOT_JSON_REPLIES = [
    pytest.param("Sure! Here is my answer: it is great.", id="prose"),
    pytest.param('{"answer": "unterminated', id="truncated-object"),
]

Client = OpenAIAPIClient | LlamaSwapAPIClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch):
    """Start every test with tracing off so a developer's shell env cannot change behaviour (DEC-000-4)."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _openai() -> OpenAIAPIClient:
    """An OpenAI client with an explicit test credential."""
    return OpenAIAPIClient(model=MODEL, base_url=OPENAI_BASE_URL, bearer_token="sk-test")


def _llamaswap() -> LlamaSwapAPIClient:
    """A llama-swap client on a local test address with an explicit test credential."""
    return LlamaSwapAPIClient(model=MODEL, base_url=LLAMASWAP_BASE_URL, bearer_token="sk-test")


def _serve(monkeypatch, client: Client, content: str) -> list[dict]:
    """Replace the ``_request`` seam with a fake chat-completions server answering with ``content``.

    The fake checks it receives ``POST /chat/completions`` carrying the user's messages. Returns the list of
    recorded requests.
    """
    calls: list[dict] = []

    async def fake_request(method: str, path: str, **kw: Any) -> dict:
        body = dict(kw.get("json") or {})
        calls.append({"method": method, "path": path, "json": body})
        assert method == "POST"
        assert path == "/chat/completions"
        assert [dict(m) for m in body["messages"]][-1] == USER_MESSAGES[-1]
        message = {"role": "assistant", "content": content}
        return {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}

    monkeypatch.setattr(client, "_request", fake_request)
    return calls


# ---------------------------------------------------------------------------
# DoD-15 — a valid single JSON value is returned unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("content", SINGLE_JSON_VALUES)
async def test_openai_chat_json_returns_single_json_value_unchanged__DoD15(monkeypatch, content):
    """DoD-15: OpenAI ``chat_json`` with a no-default shape returns a number/true/null/string/list reply unchanged."""
    client = _openai()
    calls = _serve(monkeypatch, client, content)
    assert await client.chat_json(USER_MESSAGES, response_schema=EXPECTED_SHAPE) == content
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("content", SINGLE_JSON_VALUES)
async def test_llamaswap_chat_json_returns_single_json_value_unchanged__DoD15(monkeypatch, content):
    """DoD-15: llama-swap ``chat_json`` with a no-default shape returns a single JSON value reply unchanged."""
    client = _llamaswap()
    calls = _serve(monkeypatch, client, content)
    assert await client.chat_json(USER_MESSAGES, response_schema=EXPECTED_SHAPE) == content
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# DoD-16 — not JSON: error with a shape, returned as is without one
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("content", NOT_JSON_REPLIES)
async def test_openai_chat_json_not_json_with_shape_raises_llmerror__DoD16(monkeypatch, content):
    """DoD-16: OpenAI ``chat_json`` with an expected shape ends with ``LLMError`` for a reply that is not JSON."""
    client = _openai()
    _serve(monkeypatch, client, content)
    with pytest.raises(LLMError):
        await client.chat_json(USER_MESSAGES, response_schema=EXPECTED_SHAPE)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", NOT_JSON_REPLIES)
async def test_llamaswap_chat_json_not_json_with_shape_raises_llmerror__DoD16(monkeypatch, content):
    """DoD-16: llama-swap ``chat_json`` with an expected shape ends with ``LLMError`` for a reply that is not JSON."""
    client = _llamaswap()
    _serve(monkeypatch, client, content)
    with pytest.raises(LLMError):
        await client.chat_json(USER_MESSAGES, response_schema=EXPECTED_SHAPE)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", NOT_JSON_REPLIES)
async def test_openai_chat_json_not_json_without_shape_is_returned__DoD16(monkeypatch, content):
    """DoD-16: OpenAI ``chat_json`` without an expected shape returns a non-JSON reply as is."""
    client = _openai()
    _serve(monkeypatch, client, content)
    assert await client.chat_json(USER_MESSAGES) == content


@pytest.mark.asyncio
@pytest.mark.parametrize("content", NOT_JSON_REPLIES)
async def test_llamaswap_chat_json_not_json_without_shape_is_returned__DoD16(monkeypatch, content):
    """DoD-16: llama-swap ``chat_json`` without an expected shape returns a non-JSON reply as is."""
    client = _llamaswap()
    _serve(monkeypatch, client, content)
    assert await client.chat_json(USER_MESSAGES) == content
