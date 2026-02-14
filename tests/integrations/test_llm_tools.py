import os

import pytest

from llm.llama_swap import LlamaSwapAPIClient
from llm.openai import OpenAIAPIClient


TOOLS_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers and return the sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mul",
            "description": "Multiply two integers and return the product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        },
    },
]


@pytest.mark.asyncio
async def test_openai_chat_with_tools():
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPEN_AI_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    model = os.environ.get("OPENAI_TOOL_MODEL", "gpt-4o-mini")
    client = OpenAIAPIClient(model=model)
    called = {"add": False, "mul": False}

    def _add(a: int, b: int) -> int:
        called["add"] = True
        return a + b

    def _mul(a: int, b: int) -> int:
        called["mul"] = True
        return a * b
    messages = [
        {
            "role": "user",
            "content": (
                "Call add with a=2,b=3 and mul with a=4,b=5. "
                "Reply with 'ADD=<sum> MUL=<product>'."
            ),
        }
    ]
    async with client:
        response = await client.chat_with_tools(
            messages,
            tools_definitions=TOOLS_DEFINITIONS,
            tools={"add": _add, "mul": _mul},
        )
        assert isinstance(response, str)
        assert called["add"] is True
        assert called["mul"] is True
        assert "5" in response
        assert "20" in response


@pytest.mark.asyncio
async def test_llamaswap_chat_with_tools():
    base_url = (
        os.environ.get("LLAMA_SWITCH_URL")
        or os.environ.get("LLAMA_SWAP_BASE_URL")
        or "http://india.loc:9292/v1"
    )

    model = os.environ.get("LLAMA_SWAP_TOOL_MODEL") or "Qwen3-Silent-Scream-6B.i1-Q6_K"

    client = LlamaSwapAPIClient(model=model, base_url=base_url)
    called = {"add": False, "mul": False}

    def _add(a: int, b: int) -> int:
        called["add"] = True
        return a + b

    def _mul(a: int, b: int) -> int:
        called["mul"] = True
        return a * b
    messages = [
        {
            "role": "user",
            "content": (
                "Call add with a=2,b=3 and mul with a=4,b=5. "
                "Reply with 'ADD=<sum> MUL=<product>'."
            ),
        }
    ]
    async with client:
        response = await client.chat_with_tools(
            messages,
            tools_definitions=TOOLS_DEFINITIONS,
            tools={"add": _add, "mul": _mul},
        )
        assert isinstance(response, str)
        assert called["add"] is True
        assert called["mul"] is True
        assert "5" in response
        assert "20" in response
