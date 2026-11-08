import os
import re

import pytest

from llm.llama_swap_chat_completions import LlamaSwapChatCompletionsClient
from llm.openai_chat_completions import OpenAIChatCompletionsClient


def _openai_api_key() -> str:
    """Return the OpenAI key from the test's own environment, skipping the test when absent.

    Step 004 DoD-20: the test reads the key and passes it as ``bearer_token``; the library never
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


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks.

    chat_with_tools streams reasoning through on_delta but deliberately keeps it out of the
    returned content, so the streamed text is the returned content plus think blocks.
    """
    return _THINK_BLOCK.sub("", text)


def assert_think_tags_well_formed(text: str) -> None:
    """If <think> tags are present, verify they are properly matched and ordered."""
    open_count = text.count("<think>")
    close_count = text.count("</think>")
    assert open_count == close_count, f"Mismatched think tags: {open_count} open, {close_count} close"
    if open_count > 0:
        assert text.index("<think>") < text.index("</think>"), "<think> should come before </think>"


@pytest.mark.asyncio
async def test_openai_chat_with_tools_streaming():
    api_key = _openai_api_key()  # DoD-20

    model = os.environ.get("OPENAI_TOOL_MODEL", "gpt-4o-mini")
    client = OpenAIChatCompletionsClient(model=model, bearer_token=api_key)
    called = {"fetch_record": False, "query_index": False}
    chunks = []

    def _fetch_record(key: str) -> str:
        called["fetch_record"] = True
        return "alpha-77"

    def _query_index(tag: str) -> int:
        called["query_index"] = True
        return 42

    async def on_delta(delta: str) -> None:
        chunks.append(delta)

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
            stream=True,
            on_delta=on_delta,
        )

    assert isinstance(response, str)
    assert called["fetch_record"] is True
    assert called["query_index"] is True
    assert "alpha-77" in response
    assert "42" in response
    assert len(chunks) > 1, "Final answer should be streamed in multiple chunks"
    streamed = "".join(chunks)
    assert_think_tags_well_formed(streamed)
    # Reasoning is streamed but intentionally excluded from the returned content
    assert strip_think_blocks(streamed).endswith(response)


@pytest.mark.asyncio
async def test_llamaswap_chat_with_tools_streaming():
    base_url = (
        os.environ.get("LLAMA_SWITCH_URL")
        or os.environ.get("LLAMA_SWAP_BASE_URL")
        or "http://india.loc:9292/v1"
    )

    model = os.environ.get("LLAMA_SWAP_TOOL_MODEL") or "Qwen3.8-27B-Uncensored-FP8.i1-Q5_K_M"

    client = LlamaSwapChatCompletionsClient(model=model, base_url=base_url)
    called = {"fetch_record": False, "query_index": False}
    chunks = []

    def _fetch_record(key: str) -> str:
        called["fetch_record"] = True
        return "alpha-77"

    def _query_index(tag: str) -> int:
        called["query_index"] = True
        return 42

    async def on_delta(delta: str) -> None:
        chunks.append(delta)

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
            stream=True,
            on_delta=on_delta,
        )

    assert isinstance(response, str)
    assert called["fetch_record"] is True
    assert called["query_index"] is True
    assert "alpha-77" in response
    assert "42" in response
    assert len(chunks) > 1, "Final answer should be streamed in multiple chunks"
    streamed = "".join(chunks)
    assert_think_tags_well_formed(streamed)
    # Reasoning is streamed but intentionally excluded from the returned content
    assert strip_think_blocks(streamed).endswith(response)
