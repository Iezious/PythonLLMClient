"""Unit tests for streaming chat_with_tools in OpenAI and LlamaSwap backends."""
import pytest

from llm.openai import OpenAIAPIClient
from llm.llama_swap import LlamaSwapAPIClient

_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        },
    }
]


def _add(a: int, b: int) -> int:
    return a + b


TOOLS = {"add": _add}


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_streaming_with_tool_call(monkeypatch):
    client = OpenAIAPIClient(model="gpt-4o-mini", bearer_token="sk-test")
    call_count = 0
    captured_payloads = []

    async def fake_stream_request(payload, on_delta=None):
        nonlocal call_count
        call_count += 1
        # Copy messages to avoid aliasing the mutable msgs list
        captured_payloads.append({**payload, "messages": list(payload["messages"])})
        if call_count == 1:
            return ("", [{"id": "call_1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}}])
        return ("The result is 3.", [])

    monkeypatch.setattr(client, "_tools_stream_request", fake_stream_request)

    result = await client.chat_with_tools(
        [{"role": "user", "content": "What is 1+2?"}],
        tools_definitions=_TOOL_DEFS,
        tools=TOOLS,
        stream=True,
    )
    assert result == "The result is 3."
    assert call_count == 2
    tool_msg = captured_payloads[1]["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["content"] == "3"


@pytest.mark.asyncio
async def test_openai_streaming_on_delta_called(monkeypatch):
    client = OpenAIAPIClient(model="gpt-4o-mini", bearer_token="sk-test")
    received = []

    async def on_delta(text: str) -> None:
        received.append(text)

    async def fake_stream_request(payload, on_delta=None):
        if on_delta:
            await on_delta("The result ")
            await on_delta("is 3.")
        return ("The result is 3.", [])

    monkeypatch.setattr(client, "_tools_stream_request", fake_stream_request)

    result = await client.chat_with_tools(
        [{"role": "user", "content": "What is 1+2?"}],
        tools_definitions=_TOOL_DEFS,
        tools=TOOLS,
        stream=True,
        on_delta=on_delta,
    )
    assert result == "The result is 3."
    assert received == ["The result ", "is 3."]


@pytest.mark.asyncio
async def test_openai_no_tool_call_streaming(monkeypatch):
    client = OpenAIAPIClient(model="gpt-4o-mini", bearer_token="sk-test")
    call_count = 0

    async def fake_stream_request(payload, on_delta=None):
        nonlocal call_count
        call_count += 1
        return ("The sky is blue.", [])

    monkeypatch.setattr(client, "_tools_stream_request", fake_stream_request)

    result = await client.chat_with_tools(
        [{"role": "user", "content": "What color is the sky?"}],
        tools_definitions=_TOOL_DEFS,
        tools=TOOLS,
        stream=True,
    )
    assert result == "The sky is blue."
    assert call_count == 1


@pytest.mark.asyncio
async def test_openai_non_streaming_unchanged(monkeypatch):
    client = OpenAIAPIClient(model="gpt-4o-mini", bearer_token="sk-test")
    call_count = 0

    async def fake_request(method, path, **kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"choices": [{"message": {
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "add", "arguments": '{"a": 3, "b": 4}'}}],
            }}]}
        return {"choices": [{"message": {"content": "Result is 7.", "tool_calls": []}}]}

    monkeypatch.setattr(client, "_request", fake_request)

    result = await client.chat_with_tools(
        [{"role": "user", "content": "3+4?"}],
        tools_definitions=_TOOL_DEFS,
        tools=TOOLS,
        stream=False,
    )
    assert result == "Result is 7."
    assert call_count == 2


@pytest.mark.asyncio
async def test_openai_streaming_max_loops_exceeded(monkeypatch):
    client = OpenAIAPIClient(model="gpt-4o-mini", bearer_token="sk-test")

    async def fake_stream_request(payload, on_delta=None):
        return ("", [{"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1, "b": 1}'}}])

    monkeypatch.setattr(client, "_tools_stream_request", fake_stream_request)

    with pytest.raises(RuntimeError, match="max_loops"):
        await client.chat_with_tools(
            [{"role": "user", "content": "loop forever"}],
            tools_definitions=_TOOL_DEFS,
            tools=TOOLS,
            stream=True,
            max_loops=3,
        )


# ---------------------------------------------------------------------------
# LlamaSwap mirrors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_streaming_with_tool_call(monkeypatch):
    client = LlamaSwapAPIClient(model="some-model", base_url="http://localhost:8080/v1")
    call_count = 0
    captured_payloads = []

    async def fake_stream_request(payload, on_delta=None):
        nonlocal call_count
        call_count += 1
        captured_payloads.append({**payload, "messages": list(payload["messages"])})
        if call_count == 1:
            return ("", [{"id": "call_2", "type": "function", "function": {"name": "add", "arguments": '{"a": 5, "b": 6}'}}])
        return ("The result is 11.", [])

    monkeypatch.setattr(client, "_tools_stream_request", fake_stream_request)

    result = await client.chat_with_tools(
        [{"role": "user", "content": "What is 5+6?"}],
        tools_definitions=_TOOL_DEFS,
        tools=TOOLS,
        stream=True,
    )
    assert result == "The result is 11."
    assert call_count == 2
    tool_msg = captured_payloads[1]["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_2"
    assert tool_msg["content"] == "11"


@pytest.mark.asyncio
async def test_llamaswap_no_tool_call_streaming(monkeypatch):
    client = LlamaSwapAPIClient(model="some-model", base_url="http://localhost:8080/v1")

    async def fake_stream_request(payload, on_delta=None):
        return ("Direct answer.", [])

    monkeypatch.setattr(client, "_tools_stream_request", fake_stream_request)

    result = await client.chat_with_tools(
        [{"role": "user", "content": "Hello"}],
        tools_definitions=_TOOL_DEFS,
        tools=TOOLS,
        stream=True,
    )
    assert result == "Direct answer."


@pytest.mark.asyncio
async def test_llamaswap_streaming_max_loops_exceeded(monkeypatch):
    client = LlamaSwapAPIClient(model="some-model", base_url="http://localhost:8080/v1")

    async def fake_stream_request(payload, on_delta=None):
        return ("", [{"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1, "b": 1}'}}])

    monkeypatch.setattr(client, "_tools_stream_request", fake_stream_request)

    with pytest.raises(RuntimeError, match="max_loops"):
        await client.chat_with_tools(
            [{"role": "user", "content": "loop"}],
            tools_definitions=_TOOL_DEFS,
            tools=TOOLS,
            stream=True,
            max_loops=2,
        )
