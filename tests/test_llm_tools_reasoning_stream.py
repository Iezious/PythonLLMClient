"""Parser-level unit tests for _stream_openai_tools_response reasoning handling.

tests/test_llm_tools_streaming.py monkeypatches _tools_stream_request wholesale and so
never exercises the SSE parser. These tests drive the parser directly by feeding canned
`data:` payloads through a stubbed _iter_sse_data.
"""
import json
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

from llm.openai_chat_completions import OpenAIChatCompletionsClient


def assert_think_tags_well_formed(text: str) -> None:
    """If <think> tags are present, verify they are properly matched and ordered."""
    open_count = text.count("<think>")
    close_count = text.count("</think>")
    assert open_count == close_count, f"Mismatched think tags: {open_count} open, {close_count} close"
    if open_count > 0:
        assert text.index("<think>") < text.index("</think>"), "<think> should come before </think>"


def _delta(**fields: Any) -> str:
    return json.dumps({"choices": [{"index": 0, "delta": fields}]})


def _tool_call_delta(
    index: int = 0,
    call_id: Optional[str] = None,
    name: Optional[str] = None,
    arguments: Optional[str] = None,
) -> Dict[str, Any]:
    func: Dict[str, Any] = {}
    if name is not None:
        func["name"] = name
    if arguments is not None:
        func["arguments"] = arguments
    entry: Dict[str, Any] = {"index": index, "function": func}
    if call_id is not None:
        entry["id"] = call_id
    return entry


def _client_with_stream(monkeypatch, frames: List[str]) -> OpenAIChatCompletionsClient:
    client = OpenAIChatCompletionsClient(model="gpt-4o-mini", bearer_token="sk-test")

    async def fake_iter_sse_data(_response: Any) -> AsyncIterator[str]:
        for frame in frames:
            yield frame

    monkeypatch.setattr(client, "_iter_sse_data", fake_iter_sse_data)
    return client


async def _run(monkeypatch, frames: List[str]) -> tuple[str, list[dict], List[str]]:
    client = _client_with_stream(monkeypatch, frames)
    chunks: List[str] = []

    async def on_delta(delta: str) -> None:
        chunks.append(delta)

    content, tool_calls = await client._stream_openai_tools_response(None, on_delta)
    return content, tool_calls, chunks


@pytest.mark.asyncio
async def test_reasoning_then_content(monkeypatch):
    """Reasoning is streamed <think>-wrapped; the returned content carries no tags."""
    content, tool_calls, chunks = await _run(
        monkeypatch,
        [
            _delta(reasoning_content="Let me think. "),
            _delta(content="The answer is 3."),
            "[DONE]",
        ],
    )

    streamed = "".join(chunks)
    assert streamed == "<think>Let me think. </think>The answer is 3."
    assert_think_tags_well_formed(streamed)
    assert content == "The answer is 3."
    assert "<think>" not in content and "</think>" not in content
    assert tool_calls == []


@pytest.mark.asyncio
async def test_reasoning_split_across_deltas(monkeypatch):
    """Multiple reasoning deltas produce exactly one <think> and one </think>."""
    _content, _tool_calls, chunks = await _run(
        monkeypatch,
        [
            _delta(reasoning_content="First "),
            _delta(reasoning_content="second "),
            _delta(reasoning_content="third."),
            _delta(content="Done."),
            "[DONE]",
        ],
    )

    streamed = "".join(chunks)
    assert streamed.count("<think>") == 1
    assert streamed.count("</think>") == 1
    assert streamed == "<think>First second third.</think>Done."
    assert_think_tags_well_formed(streamed)


@pytest.mark.asyncio
async def test_reasoning_then_tool_calls_without_content(monkeypatch):
    """A round ending in tool calls and no content still closes the <think> tag."""
    content, tool_calls, chunks = await _run(
        monkeypatch,
        [
            _delta(reasoning_content="I should call add."),
            _delta(tool_calls=[_tool_call_delta(call_id="call_1", name="add", arguments='{"a": 1,')]),
            _delta(tool_calls=[_tool_call_delta(arguments=' "b": 2}')]),
            "[DONE]",
        ],
    )

    streamed = "".join(chunks)
    assert streamed == "<think>I should call add.</think>"
    assert_think_tags_well_formed(streamed)
    assert content == ""
    assert tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
        }
    ]


@pytest.mark.asyncio
async def test_delta_carrying_both_reasoning_and_tool_calls(monkeypatch):
    """A single delta with both reasoning_content and tool_calls loses neither."""
    content, tool_calls, chunks = await _run(
        monkeypatch,
        [
            _delta(
                reasoning_content="Calling now.",
                tool_calls=[_tool_call_delta(call_id="call_7", name="add", arguments='{"a": 4, "b": 5}')],
            ),
            "[DONE]",
        ],
    )

    streamed = "".join(chunks)
    assert streamed == "<think>Calling now.</think>"
    assert_think_tags_well_formed(streamed)
    assert content == ""
    assert tool_calls == [
        {
            "id": "call_7",
            "type": "function",
            "function": {"name": "add", "arguments": '{"a": 4, "b": 5}'},
        }
    ]


@pytest.mark.asyncio
async def test_no_reasoning_is_unchanged(monkeypatch):
    """Regression: a stream without reasoning_content behaves exactly as before."""
    content, tool_calls, chunks = await _run(
        monkeypatch,
        [
            _delta(content="Hello "),
            _delta(tool_calls=[_tool_call_delta(call_id="call_2", name="add", arguments='{"a": 1,')]),
            _delta(tool_calls=[_tool_call_delta(arguments=' "b": 2}')]),
            _delta(content="world."),
            "[DONE]",
        ],
    )

    assert chunks == ["Hello ", "world."]
    assert content == "Hello world."
    assert "<think>" not in content
    assert tool_calls == [
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
        }
    ]


@pytest.mark.asyncio
async def test_server_inlined_think_tags_are_not_double_wrapped(monkeypatch):
    """A server inlining its own <think> tags into content passes straight through."""
    content, _tool_calls, chunks = await _run(
        monkeypatch,
        [
            _delta(content="<think>inline thinking</think>"),
            _delta(content="The answer is 3."),
            "[DONE]",
        ],
    )

    streamed = "".join(chunks)
    assert streamed == "<think>inline thinking</think>The answer is 3."
    assert streamed.count("<think>") == 1
    assert content == "<think>inline thinking</think>The answer is 3."


@pytest.mark.asyncio
async def test_multiple_tool_calls_by_index(monkeypatch):
    """Parallel tool calls accumulate independently alongside reasoning."""
    content, tool_calls, chunks = await _run(
        monkeypatch,
        [
            _delta(reasoning_content="Two calls needed."),
            _delta(
                tool_calls=[
                    _tool_call_delta(index=0, call_id="call_a", name="add", arguments='{"a": 1, "b": 2}'),
                    _tool_call_delta(index=1, call_id="call_b", name="mul", arguments='{"a": 3,'),
                ]
            ),
            _delta(tool_calls=[_tool_call_delta(index=1, arguments=' "b": 4}')]),
            "[DONE]",
        ],
    )

    assert "".join(chunks) == "<think>Two calls needed.</think>"
    assert content == ""
    assert [tc["id"] for tc in tool_calls] == ["call_a", "call_b"]
    assert tool_calls[1]["function"]["arguments"] == '{"a": 3, "b": 4}'


@pytest.mark.asyncio
async def test_parser_without_on_delta(monkeypatch):
    """Reasoning handling must not blow up when no on_delta callback is supplied."""
    client = _client_with_stream(
        monkeypatch,
        [
            _delta(reasoning_content="quiet thinking"),
            _delta(content="Answer."),
            "[DONE]",
        ],
    )

    content, tool_calls = await client._stream_openai_tools_response(None)
    assert content == "Answer."
    assert tool_calls == []
