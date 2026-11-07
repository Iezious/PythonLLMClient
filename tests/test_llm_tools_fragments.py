"""Streamed tool-call fragment assembly in the shared tool-stream parser.

Feature 002.edge-fixes, step 002 (DEC-002-6): `LLMClient._stream_openai_tools_response` groups
positioned fragments by position (name set, not appended) and positionless fragments by call id and
name, returning positioned calls by position and then positionless calls in order of appearance.
The parser is driven directly by feeding canned `data:` payloads through a stubbed `_iter_sse_data`.
"""

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from llm.llama_swap_chat_completions import LlamaSwapChatCompletionsClient
from llm.llm_client import LLMClient
from llm.openai_chat_completions import OpenAIChatCompletionsClient

# The parser lives on the shared base class; both OpenAI-family clients reach it, so every test runs
# against both to show the grouping is backend-independent.
_CLIENT_FACTORIES: list[Callable[[], LLMClient]] = [
    lambda: OpenAIChatCompletionsClient(model="gpt-4o-mini", bearer_token="sk-test"),
    lambda: LlamaSwapChatCompletionsClient(
        model="some-model", base_url="http://localhost:8080/v1", bearer_token="sk-test"
    ),
]
_CLIENT_IDS = ["openai", "llamaswap"]

parametrize_client = pytest.mark.parametrize("make_client", _CLIENT_FACTORIES, ids=_CLIENT_IDS)


def _delta(**fields: Any) -> str:
    """One SSE payload whose first choice carries the given delta fields."""
    return json.dumps({"choices": [{"index": 0, "delta": fields}]})


def _tools(*fragments: dict[str, Any]) -> str:
    """One SSE payload whose delta carries the given tool-call fragments."""
    return _delta(tool_calls=list(fragments))


def _fragment(
    *,
    index: object = None,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> dict[str, Any]:
    """A tool-call fragment; `index=None` omits the position entirely (unlike the older helper).

    Any non-None `index` is sent as given, so a non-integer value can be used to exercise the
    "non-integer index counts as no position" rule (002.context.md).
    """
    func: dict[str, Any] = {}
    if name is not None:
        func["name"] = name
    if arguments is not None:
        func["arguments"] = arguments
    entry: dict[str, Any] = {"function": func}
    if index is not None:
        entry["index"] = index
    if call_id is not None:
        entry["id"] = call_id
    return entry


def _call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    """The assembled call shape the tool loop reads."""
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


async def _run(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[[], LLMClient], frames: list[str]
) -> tuple[str, list[dict], list[str]]:
    """Feed `frames` to the parser and return (content, tool calls, text sent to the receiver).

    Every fake stream ends with `[DONE]`: since feature 001 a stream with neither a finish reason
    nor `[DONE]` is an early end (DEC-001-1), which is not what these tests exercise.
    """
    client = make_client()

    async def fake_iter_sse_data(_response: Any) -> AsyncIterator[str]:
        for frame in frames:
            yield frame

    monkeypatch.setattr(client, "_iter_sse_data", fake_iter_sse_data)
    received: list[str] = []

    async def on_delta(text: str) -> None:
        received.append(text)

    content, tool_calls = await client._stream_openai_tools_response(None, on_delta)  # type: ignore[arg-type]
    return content, tool_calls, received


@parametrize_client
@pytest.mark.asyncio
async def test_openai_positioned_repeated_name_set_once__DoD1(monkeypatch, make_client):
    """DoD-1: positioned fragments repeating the name give one call named once, arguments concatenated."""
    _content, tool_calls, _received = await _run(
        monkeypatch,
        make_client,
        [
            _tools(_fragment(index=0, call_id="call_1", name="add", arguments='{"a": ')),
            _tools(_fragment(index=0, name="add", arguments="1, ")),
            _tools(_fragment(index=0, name="add", arguments='"b": 2}')),
            "[DONE]",
        ],
    )

    assert tool_calls == [_call("call_1", "add", '{"a": 1, "b": 2}')]


@parametrize_client
@pytest.mark.asyncio
async def test_openai_positioned_repeated_name_and_id_every_fragment__DoD1(monkeypatch, make_client):
    """DoD-1: a positioned call repeating both id and name in every fragment still has the name once."""
    _content, tool_calls, _received = await _run(
        monkeypatch,
        make_client,
        [
            _tools(_fragment(index=0, call_id="call_9", name="lookup", arguments='{"q": ')),
            _tools(_fragment(index=0, call_id="call_9", name="lookup", arguments='"x"}')),
            "[DONE]",
        ],
    )

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "lookup"
    assert tool_calls[0]["function"]["arguments"] == '{"q": "x"}'
    assert tool_calls[0]["id"] == "call_9"


@parametrize_client
@pytest.mark.asyncio
async def test_openai_positionless_distinct_ids_give_separate_calls__DoD2(monkeypatch, make_client):
    """DoD-2: positionless fragments of two calls with different ids give two calls in appearance order."""
    _content, tool_calls, _received = await _run(
        monkeypatch,
        make_client,
        [
            _tools(_fragment(call_id="call_a", name="add", arguments='{"a": 1,')),
            _tools(_fragment(arguments=' "b": 2}')),
            _tools(_fragment(call_id="call_b", name="mul", arguments='{"a": 3,')),
            _tools(_fragment(arguments=' "b": 4}')),
            "[DONE]",
        ],
    )

    assert tool_calls == [
        _call("call_a", "add", '{"a": 1, "b": 2}'),
        _call("call_b", "mul", '{"a": 3, "b": 4}'),
    ]


@parametrize_client
@pytest.mark.asyncio
async def test_openai_positionless_distinct_ids_same_tool_name__DoD2(monkeypatch, make_client):
    """DoD-2: two positionless calls of the same tool are told apart by their different ids."""
    _content, tool_calls, _received = await _run(
        monkeypatch,
        make_client,
        [
            _tools(_fragment(call_id="call_1", name="add", arguments='{"a": 1, "b": 2}')),
            _tools(_fragment(call_id="call_2", name="add", arguments='{"a": 5, "b": 6}')),
            "[DONE]",
        ],
    )

    assert tool_calls == [
        _call("call_1", "add", '{"a": 1, "b": 2}'),
        _call("call_2", "add", '{"a": 5, "b": 6}'),
    ]


@parametrize_client
@pytest.mark.asyncio
async def test_openai_non_integer_index_counts_as_no_position__DoD2(monkeypatch, make_client):
    """DoD-2: a non-integer index is treated as no position, so different ids still give two calls."""
    _content, tool_calls, _received = await _run(
        monkeypatch,
        make_client,
        [
            _tools(_fragment(index="x", call_id="call_a", name="add", arguments='{"a": 1}')),
            _tools(_fragment(index="x", call_id="call_b", name="mul", arguments='{"b": 2}')),
            "[DONE]",
        ],
    )

    assert tool_calls == [
        _call("call_a", "add", '{"a": 1}'),
        _call("call_b", "mul", '{"b": 2}'),
    ]


@parametrize_client
@pytest.mark.asyncio
async def test_openai_positionless_no_ids_split_per_named_fragment__DoD3(monkeypatch, make_client):
    """DoD-3: positionless, id-less fragments give one call per named fragment with its following arguments."""
    _content, tool_calls, _received = await _run(
        monkeypatch,
        make_client,
        [
            _tools(_fragment(name="add", arguments='{"a": ')),
            _tools(_fragment(arguments='1, "b": 2}')),
            _tools(_fragment(name="mul")),
            _tools(_fragment(arguments='{"a": 3, ')),
            _tools(_fragment(arguments='"b": 4}')),
            _tools(_fragment(name="add", arguments='{"a": 7, "b": 8}')),
            "[DONE]",
        ],
    )

    assert tool_calls == [
        _call("", "add", '{"a": 1, "b": 2}'),
        _call("", "mul", '{"a": 3, "b": 4}'),
        _call("", "add", '{"a": 7, "b": 8}'),
    ]


@parametrize_client
@pytest.mark.asyncio
async def test_openai_positionless_repeated_id_and_name_one_call__DoD4(monkeypatch, make_client):
    """DoD-4: positionless fragments repeating the same id and name give one call, name once, args joined."""
    _content, tool_calls, _received = await _run(
        monkeypatch,
        make_client,
        [
            _tools(_fragment(call_id="call_1", name="add", arguments='{"a": ')),
            _tools(_fragment(call_id="call_1", name="add", arguments="1, ")),
            _tools(_fragment(call_id="call_1", name="add", arguments='"b": 2}')),
            "[DONE]",
        ],
    )

    assert tool_calls == [_call("call_1", "add", '{"a": 1, "b": 2}')]


@parametrize_client
@pytest.mark.asyncio
async def test_openai_positioned_interleaved_calls_by_position__DoD5(monkeypatch, make_client):
    """DoD-5: positioned calls interleaved across frames are assembled per position, returned in position order."""
    _content, tool_calls, _received = await _run(
        monkeypatch,
        make_client,
        [
            # Position 1 starts first, so appearance order differs from position order.
            _tools(_fragment(index=1, call_id="call_b", name="mul", arguments='{"a": 3,')),
            _tools(_fragment(index=0, call_id="call_a", name="add", arguments='{"a": 1,')),
            _tools(
                _fragment(index=1, arguments=' "b": 4}'),
                _fragment(index=2, call_id="call_c", name="neg", arguments='{"x": '),
            ),
            _tools(_fragment(index=0, arguments=' "b": 2}')),
            _tools(_fragment(index=2, arguments="9}")),
            "[DONE]",
        ],
    )

    assert tool_calls == [
        _call("call_a", "add", '{"a": 1, "b": 2}'),
        _call("call_b", "mul", '{"a": 3, "b": 4}'),
        _call("call_c", "neg", '{"x": 9}'),
    ]


@parametrize_client
@pytest.mark.asyncio
async def test_openai_positioned_before_positionless_in_order__DoD5(monkeypatch, make_client):
    """DoD-5 (order rule): positioned calls come first by position, then positionless calls by appearance."""
    _content, tool_calls, _received = await _run(
        monkeypatch,
        make_client,
        [
            _tools(_fragment(call_id="call_p1", name="first_free", arguments="{}")),
            _tools(_fragment(index=1, call_id="call_i1", name="second_pos", arguments="{}")),
            _tools(_fragment(call_id="call_p2", name="second_free", arguments="{}")),
            _tools(_fragment(index=0, call_id="call_i0", name="first_pos", arguments="{}")),
            "[DONE]",
        ],
    )

    assert [tc["id"] for tc in tool_calls] == ["call_i0", "call_i1", "call_p1", "call_p2"]
    assert [tc["function"]["name"] for tc in tool_calls] == ["first_pos", "second_pos", "first_free", "second_free"]


@parametrize_client
@pytest.mark.asyncio
async def test_openai_every_call_has_tool_loop_shape__DoD6(monkeypatch, make_client):
    """DoD-6: each returned call has id text ("" when none sent), type "function", function name and arguments."""
    _content, tool_calls, _received = await _run(
        monkeypatch,
        make_client,
        [
            _tools(_fragment(index=0, name="add", arguments='{"a": 1}')),
            _tools(_fragment(index=1, call_id="call_b", name="mul", arguments='{"b": 2}')),
            # Positionless: an id-bearing call, then an id-less named fragment. Per DEC-002-6 the
            # second starts a new call (no id, a name, and the current call already has a name).
            _tools(_fragment(call_id="call_d", name="echo", arguments='{"s": "hi"}')),
            _tools(_fragment(name="neg", arguments='{"x": 3}')),
            "[DONE]",
        ],
    )

    assert len(tool_calls) == 4
    for call in tool_calls:
        assert set(call) == {"id", "type", "function"}
        assert isinstance(call["id"], str)
        assert call["type"] == "function"
        assert set(call["function"]) == {"name", "arguments"}
        assert isinstance(call["function"]["name"], str)
        assert isinstance(call["function"]["arguments"], str)
    assert [tc["id"] for tc in tool_calls] == ["", "call_b", "call_d", ""]
    assert [tc["function"]["name"] for tc in tool_calls] == ["add", "mul", "echo", "neg"]


@parametrize_client
@pytest.mark.asyncio
async def test_openai_positionless_id_after_name_attaches_to_call__DoD6(monkeypatch, make_client):
    """DoD-6 (DEC-002-6): an id arriving after the name of an id-less positionless call joins that call."""
    _content, tool_calls, _received = await _run(
        monkeypatch,
        make_client,
        [
            _tools(_fragment(name="add", arguments='{"a": ')),
            _tools(_fragment(call_id="call_1", arguments='1, "b": 2}')),
            "[DONE]",
        ],
    )

    assert tool_calls == [_call("call_1", "add", '{"a": 1, "b": 2}')]


@parametrize_client
@pytest.mark.asyncio
async def test_openai_positionless_fragments_keep_content_and_reasoning__DoD7(monkeypatch, make_client):
    """DoD-7: with positionless tool fragments, the receiver gets content and reasoning; content excludes it."""
    content, tool_calls, received = await _run(
        monkeypatch,
        make_client,
        [
            _delta(reasoning_content="I should add. "),
            _delta(reasoning_content="Then multiply."),
            _tools(_fragment(call_id="call_a", name="add", arguments='{"a": 1, "b": 2}')),
            _tools(_fragment(call_id="call_b", name="mul", arguments='{"a": 3,')),
            _tools(_fragment(arguments=' "b": 4}')),
            _delta(content="Working on it."),
            "[DONE]",
        ],
    )

    # The reasoning block is wrapped in <think> tags for the receiver only (DEC-000-22).
    assert "".join(received) == "<think>I should add. Then multiply.</think>Working on it."
    assert content == "Working on it."
    assert "I should add." not in content
    assert tool_calls == [
        _call("call_a", "add", '{"a": 1, "b": 2}'),
        _call("call_b", "mul", '{"a": 3, "b": 4}'),
    ]
