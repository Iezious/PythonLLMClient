import json
import os

import pytest

from llm.llama_swap_chat_completions import LlamaSwapChatCompletionsClient
from llm.openai_chat_completions import OpenAIChatCompletionsClient


def _openai_api_key() -> str:
    """Return the OpenAI key from the test's own environment, skipping the test when absent.

    The test reads the key and passes it as ``bearer_token``; the library never
    reads the environment itself (DEC-001-6).
    """
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_AI_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY / OPEN_AI_KEY not set")
    return key


# llama.cpp rejects a json_schema grammar combined with tools ("failed to parse grammar"); a server
# limitation, not a library defect. Non-strict so a server fix shows up as XPASS (DEC-005-9).
_LLAMA_SWAP_JSON_SCHEMA_XFAIL = pytest.mark.xfail(strict=False, reason="llama.cpp json_schema grammar with tools")


TOOLS_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_record",
            "description": "Retrieve a stored record by its key from the database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The record key to look up"},
                },
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_index",
            "description": "Search the index for entries matching a tag and return the count.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tag": {"type": "string", "description": "The tag to search for"},
                },
                "required": ["tag"],
            },
        },
    },
]

RESULT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "lookup_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "record": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["record", "count"],
            "additionalProperties": False,
        },
    },
}


def _make_tools():
    called = {"fetch_record": False, "query_index": False}

    def _fetch_record(key: str) -> str:
        called["fetch_record"] = True
        return "alpha-77"

    def _query_index(tag: str) -> int:
        called["query_index"] = True
        return 42

    return called, {"fetch_record": _fetch_record, "query_index": _query_index}


MESSAGES = [
    {
        "role": "user",
        "content": (
            "Look up the record with key 'usr-991' using fetch_record, "
            "and search the index for tag 'inv-x' using query_index. "
            "Return the results as JSON with keys 'record' and 'count'."
        ),
    }
]


def _assert_valid_result(response: str, called: dict):
    assert isinstance(response, str)
    assert called["fetch_record"] is True
    assert called["query_index"] is True
    data = json.loads(response)
    assert data["record"] == "alpha-77"
    assert data["count"] == 42


# ---------------------------------------------------------------------------
# OpenAI — json_object mode
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_openai_tools_response_format_json_object():
    api_key = _openai_api_key()  # DoD-20

    model = os.environ.get("OPENAI_TOOL_MODEL", "gpt-4o-mini")
    client = OpenAIChatCompletionsClient(model=model, bearer_token=api_key)
    called, tools = _make_tools()

    async with client:
        response = await client.chat_with_tools(
            MESSAGES,
            tools_definitions=TOOLS_DEFINITIONS,
            tools=tools,
            response_format={"type": "json_object"},
        )
    _assert_valid_result(response, called)


# ---------------------------------------------------------------------------
# OpenAI — json_schema mode
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_openai_tools_response_format_json_schema():
    api_key = _openai_api_key()  # DoD-20

    model = os.environ.get("OPENAI_TOOL_MODEL", "gpt-4o-mini")
    client = OpenAIChatCompletionsClient(model=model, bearer_token=api_key)
    called, tools = _make_tools()

    async with client:
        response = await client.chat_with_tools(
            MESSAGES,
            tools_definitions=TOOLS_DEFINITIONS,
            tools=tools,
            response_format=RESULT_SCHEMA,
        )
    _assert_valid_result(response, called)


# ---------------------------------------------------------------------------
# OpenAI — streaming + json_object
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_openai_tools_streaming_response_format():
    api_key = _openai_api_key()  # DoD-20

    model = os.environ.get("OPENAI_TOOL_MODEL", "gpt-4o-mini")
    client = OpenAIChatCompletionsClient(model=model, bearer_token=api_key)
    called, tools = _make_tools()
    chunks: list[str] = []

    async def on_delta(delta: str) -> None:
        chunks.append(delta)

    async with client:
        response = await client.chat_with_tools(
            MESSAGES,
            tools_definitions=TOOLS_DEFINITIONS,
            tools=tools,
            stream=True,
            on_delta=on_delta,
            response_format={"type": "json_object"},
        )
    _assert_valid_result(response, called)
    assert len(chunks) > 1, "Final answer should be streamed in multiple chunks"
    assert "".join(chunks) == response


# ---------------------------------------------------------------------------
# LlamaSwap — json_schema mode
# ---------------------------------------------------------------------------
@_LLAMA_SWAP_JSON_SCHEMA_XFAIL
@pytest.mark.asyncio
async def test_llamaswap_tools_response_format_json_schema():
    base_url = (
        os.environ.get("LLAMA_SWITCH_URL")
        or os.environ.get("LLAMA_SWAP_BASE_URL")
        or "http://india.loc:9292/v1"
    )
    model = os.environ.get("LLAMA_SWAP_TOOL_MODEL") or "Qwen3.8-27B-Uncensored-FP8.i1-Q5_K_M"

    client = LlamaSwapChatCompletionsClient(model=model, base_url=base_url)
    _, tools = _make_tools()

    async with client:
        response = await client.chat_with_tools(
            MESSAGES,
            tools_definitions=TOOLS_DEFINITIONS,
            tools=tools,
            response_format=RESULT_SCHEMA,
        )
    # Model may skip tools when schema makes the answer obvious — only assert JSON validity
    assert isinstance(response, str)
    data = json.loads(response)
    assert "record" in data
    assert "count" in data


# ---------------------------------------------------------------------------
# LlamaSwap — streaming + json_schema
# ---------------------------------------------------------------------------
@_LLAMA_SWAP_JSON_SCHEMA_XFAIL
@pytest.mark.asyncio
async def test_llamaswap_tools_streaming_response_format():
    base_url = (
        os.environ.get("LLAMA_SWITCH_URL")
        or os.environ.get("LLAMA_SWAP_BASE_URL")
        or "http://india.loc:9292/v1"
    )
    model = os.environ.get("LLAMA_SWAP_TOOL_MODEL") or "Qwen3.8-27B-Uncensored-FP8.i1-Q5_K_M"

    client = LlamaSwapChatCompletionsClient(model=model, base_url=base_url)
    _, tools = _make_tools()
    chunks: list[str] = []

    async def on_delta(delta: str) -> None:
        chunks.append(delta)

    async with client:
        response = await client.chat_with_tools(
            MESSAGES,
            tools_definitions=TOOLS_DEFINITIONS,
            tools=tools,
            stream=True,
            on_delta=on_delta,
            response_format=RESULT_SCHEMA,
        )
    assert isinstance(response, str)
    data = json.loads(response)
    assert "record" in data
    assert "count" in data
    assert len(chunks) > 1, "Final answer should be streamed in multiple chunks"
    assert "".join(chunks) == response
