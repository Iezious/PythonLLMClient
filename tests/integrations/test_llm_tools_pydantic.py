import asyncio
import os
from typing import Callable

import pytest
from pydantic import BaseModel, Field

from llm.llama_swap import LlamaSwapAPIClient
from llm.openai import OpenAIAPIClient
from llm.tools import pydantic_to_openai_tool


class FetchRecordInput(BaseModel):
    key: str = Field(..., description="The record key to look up")


class QueryIndexInput(BaseModel):
    tag: str = Field(..., description="The tag to search for")


def _build_tools(called: dict) -> tuple[list[dict], dict[str, object]]:
    def fetch_record_impl(key: str) -> str:
        called["fetch_record"] = True
        return "alpha-77"

    def query_index_impl(tag: str) -> int:
        called["query_index"] = True
        return 42

    tool_defs = [
        pydantic_to_openai_tool("fetch_record", "Retrieve a stored record by its key from the database.", FetchRecordInput),
        pydantic_to_openai_tool("query_index", "Search the index for entries matching a tag and return the count.", QueryIndexInput),
    ]
    tool_funcs = {"fetch_record": fetch_record_impl, "query_index": query_index_impl}
    return tool_defs, tool_funcs


def _build_async_tools(called: dict) -> tuple[list[dict], dict[str, Callable]]:
    async def fetch_record_impl(key: str) -> str:
        await asyncio.sleep(0.1)
        called["fetch_record"] = True
        return "alpha-77"

    async def query_index_impl(tag: str) -> int:
        await asyncio.sleep(0.1)
        called["query_index"] = True
        return 42

    tool_defs = [
        pydantic_to_openai_tool("fetch_record", "Retrieve a stored record by its key from the database.", FetchRecordInput),
        pydantic_to_openai_tool("query_index", "Search the index for entries matching a tag and return the count.", QueryIndexInput),
    ]
    tool_funcs = {"fetch_record": fetch_record_impl, "query_index": query_index_impl}
    return tool_defs, tool_funcs


MESSAGES = [
    {
        "role": "user",
        "content": (
            "Look up the record with key 'usr-991' using fetch_record, "
            "and search the index for tag 'inv-x' using query_index. "
            "Reply with 'RECORD=<value> COUNT=<value>'."
        ),
    }
]


@pytest.mark.asyncio
async def test_openai_chat_with_tools_pydantic():
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPEN_AI_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    model = os.environ.get("OPENAI_TOOL_MODEL", "gpt-4o-mini")
    client = OpenAIAPIClient(model=model)
    called = {"fetch_record": False, "query_index": False}
    tool_defs, tool_funcs = _build_tools(called)
    async with client:
        response = await client.chat_with_tools(
            MESSAGES,
            tools_definitions=tool_defs,
            tools=tool_funcs,
        )
        assert isinstance(response, str)
        assert called["fetch_record"] is True
        assert called["query_index"] is True
        assert "alpha-77" in response
        assert "42" in response


@pytest.mark.asyncio
async def test_llamaswap_chat_with_tools_pydantic():
    base_url = (
        os.environ.get("LLAMA_SWITCH_URL")
        or os.environ.get("LLAMA_SWAP_BASE_URL")
        or "http://india.loc:9292/v1"
    )

    model = os.environ.get("LLAMA_SWAP_TOOL_MODEL") or "Qwen3.5-27B-heretic-Q5_K_M"

    client = LlamaSwapAPIClient(model=model, base_url=base_url)
    called = {"fetch_record": False, "query_index": False}
    tool_defs, tool_funcs = _build_tools(called)
    async with client:
        response = await client.chat_with_tools(
            MESSAGES,
            tools_definitions=tool_defs,
            tools=tool_funcs,
        )
        assert isinstance(response, str)
        assert called["fetch_record"] is True
        assert called["query_index"] is True
        assert "alpha-77" in response
        assert "42" in response


@pytest.mark.asyncio
async def test_openai_chat_with_async_tools_pydantic():
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPEN_AI_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    model = os.environ.get("OPENAI_TOOL_MODEL", "gpt-4o-mini")
    client = OpenAIAPIClient(model=model)
    called = {"fetch_record": False, "query_index": False}
    tool_defs, tool_funcs = _build_async_tools(called)
    async with client:
        response = await client.chat_with_tools(
            MESSAGES,
            tools_definitions=tool_defs,
            tools=tool_funcs,
        )
        assert isinstance(response, str)
        assert called["fetch_record"] is True
        assert called["query_index"] is True
        assert "alpha-77" in response
        assert "42" in response


@pytest.mark.asyncio
async def test_llamaswap_chat_with_async_tools_pydantic():
    base_url = (
        os.environ.get("LLAMA_SWITCH_URL")
        or os.environ.get("LLAMA_SWAP_BASE_URL")
        or "http://india.loc:9292/v1"
    )

    model = os.environ.get("LLAMA_SWAP_TOOL_MODEL") or "Qwen3.5-27B-heretic-Q5_K_M"

    client = LlamaSwapAPIClient(model=model, base_url=base_url)
    called = {"fetch_record": False, "query_index": False}
    tool_defs, tool_funcs = _build_async_tools(called)
    async with client:
        response = await client.chat_with_tools(
            MESSAGES,
            tools_definitions=tool_defs,
            tools=tool_funcs,
        )
        assert isinstance(response, str)
        assert called["fetch_record"] is True
        assert called["query_index"] is True
        assert "alpha-77" in response
        assert "42" in response
