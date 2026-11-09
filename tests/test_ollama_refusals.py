"""The deprecated Ollama client refuses the 0.2.1 tool-loop parameters.

Feature 006 (tolerant-tool-loop), step 009, DoD-23: ``OllamaAPIClient.chat_with_tools`` does not declare
``tool_errors`` or ``on_tool_event`` (DEC-arch-24), so passing either raises ``TypeError`` naming the
parameter, before the server is contacted. The server seam ``_request`` is replaced by a recorder that fails
the test if it is ever reached.
"""

from typing import Any

import pytest

from llm.ollama import OllamaAPIClient

USER_MESSAGES = [{"role": "user", "content": "Use the tools."}]

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look a record up.",
            "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
        },
    }
]


@pytest.fixture(autouse=True)
def _no_tracing(monkeypatch):
    """Keep tracing off so a developer's LLM_TRACE setting cannot write records during these tests."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _lookup(key: str = "") -> str:
    """A well-behaved tool; it must never run, since the call is refused first."""
    raise AssertionError("the tool ran although the call should have been refused")


def _client_without_server(monkeypatch) -> tuple[OllamaAPIClient, list[tuple[Any, ...]]]:
    """Build an Ollama client whose request seam records any contact, so "no server contact" is checkable."""
    client = OllamaAPIClient(model="llama3")
    contacts: list[tuple[Any, ...]] = []

    async def fake_request(method, path, **kw):
        contacts.append((method, path))
        raise AssertionError("the server was contacted although the call should have been refused")

    monkeypatch.setattr(client, "_request", fake_request)
    return client, contacts


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [pytest.param("raise", id="raise"), pytest.param("result", id="result")])
async def test_ollama_chat_with_tools_refuses_tool_errors__DoD23(monkeypatch, value):
    """``tool_errors`` on Ollama's ``chat_with_tools`` raises TypeError naming it, before any server contact."""
    client, contacts = _client_without_server(monkeypatch)
    with pytest.raises(TypeError) as exc_info:
        await client.chat_with_tools(
            USER_MESSAGES,
            tools_definitions=TOOL_DEFINITIONS,
            tools={"lookup": _lookup},
            tool_errors=value,
        )
    assert "tool_errors" in str(exc_info.value)
    assert contacts == []


@pytest.mark.asyncio
async def test_ollama_chat_with_tools_refuses_on_tool_event__DoD23(monkeypatch):
    """``on_tool_event`` on Ollama's ``chat_with_tools`` raises TypeError naming it, before any server contact."""
    client, contacts = _client_without_server(monkeypatch)
    events: list[Any] = []
    with pytest.raises(TypeError) as exc_info:
        await client.chat_with_tools(
            USER_MESSAGES,
            tools_definitions=TOOL_DEFINITIONS,
            tools={"lookup": _lookup},
            on_tool_event=events.append,
        )
    assert "on_tool_event" in str(exc_info.value)
    assert contacts == []
    assert events == []
