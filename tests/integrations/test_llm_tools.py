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


@pytest.mark.asyncio
async def test_openai_chat_with_tools():
    api_key = _openai_api_key()  # DoD-20

    model = os.environ.get("OPENAI_TOOL_MODEL", "gpt-4o-mini")
    client = OpenAIChatCompletionsClient(model=model, bearer_token=api_key)
    called = {"fetch_record": False, "query_index": False}

    def _fetch_record(key: str) -> str:
        called["fetch_record"] = True
        return "alpha-77"

    def _query_index(tag: str) -> int:
        called["query_index"] = True
        return 42

    messages = [
        {
            "role": "user",
            "content": (
                "Look up the record with key 'usr-991' using fetch_record, "
                "and search the index for tag 'inv-x' using query_index. "
                "Reply with 'RECORD=<value> COUNT=<value>'."
            ),
        }
    ]
    async with client:
        response = await client.chat_with_tools(
            messages,
            tools_definitions=TOOLS_DEFINITIONS,
            tools={"fetch_record": _fetch_record, "query_index": _query_index},
        )
        assert isinstance(response, str)
        assert called["fetch_record"] is True
        assert called["query_index"] is True
        assert "alpha-77" in response
        assert "42" in response


@pytest.mark.asyncio
async def test_llamaswap_chat_with_tools():
    base_url = (
        os.environ.get("LLAMA_SWITCH_URL")
        or os.environ.get("LLAMA_SWAP_BASE_URL")
        or "http://india.loc:9292/v1"
    )

    model = os.environ.get("LLAMA_SWAP_TOOL_MODEL") or "Qwen3.8-27B-Uncensored-FP8.i1-Q5_K_M"

    client = LlamaSwapChatCompletionsClient(model=model, base_url=base_url)
    called = {"fetch_record": False, "query_index": False}

    def _fetch_record(key: str) -> str:
        called["fetch_record"] = True
        return "alpha-77"

    def _query_index(tag: str) -> int:
        called["query_index"] = True
        return 42

    messages = [
        {
            "role": "user",
            "content": (
                "Look up the record with key 'usr-991' using fetch_record, "
                "and search the index for tag 'inv-x' using query_index. "
                "Reply with 'RECORD=<value> COUNT=<value>'."
            ),
        }
    ]
    async with client:
        response = await client.chat_with_tools(
            messages,
            tools_definitions=TOOLS_DEFINITIONS,
            tools={"fetch_record": _fetch_record, "query_index": _query_index},
        )
        assert isinstance(response, str)
        assert called["fetch_record"] is True
        assert called["query_index"] is True
        assert "alpha-77" in response
        assert "42" in response
