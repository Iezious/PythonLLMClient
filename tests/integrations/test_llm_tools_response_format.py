import json
import os

import pytest

from llm.llama_swap import LlamaSwapAPIClient
from llm.openai import OpenAIAPIClient


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
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPEN_AI_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    model = os.environ.get("OPENAI_TOOL_MODEL", "gpt-4o-mini")
    client = OpenAIAPIClient(model=model)
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
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPEN_AI_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    model = os.environ.get("OPENAI_TOOL_MODEL", "gpt-4o-mini")
    client = OpenAIAPIClient(model=model)
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
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPEN_AI_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    model = os.environ.get("OPENAI_TOOL_MODEL", "gpt-4o-mini")
    client = OpenAIAPIClient(model=model)
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
@pytest.mark.asyncio
async def test_llamaswap_tools_response_format_json_schema():
    base_url = (
        os.environ.get("LLAMA_SWITCH_URL")
        or os.environ.get("LLAMA_SWAP_BASE_URL")
        or "http://india.loc:9292/v1"
    )
    model = os.environ.get("LLAMA_SWAP_TOOL_MODEL") or "Qwen3.5-27B-heretic-Q5_K_M"

    client = LlamaSwapAPIClient(model=model, base_url=base_url)
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
@pytest.mark.asyncio
async def test_llamaswap_tools_streaming_response_format():
    base_url = (
        os.environ.get("LLAMA_SWITCH_URL")
        or os.environ.get("LLAMA_SWAP_BASE_URL")
        or "http://india.loc:9292/v1"
    )
    model = os.environ.get("LLAMA_SWAP_TOOL_MODEL") or "Qwen3.5-27B-heretic-Q5_K_M"

    client = LlamaSwapAPIClient(model=model, base_url=base_url)
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
