import asyncio
import os
from typing import Callable

import pytest
from pydantic import BaseModel, Field

from llm.llama_swap import LlamaSwapAPIClient
from llm.openai import OpenAIAPIClient
from llm.tools import pydantic_to_openai_tool

class AddInput(BaseModel):
    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


class MulInput(BaseModel):
    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


def _build_tools(called: dict) -> tuple[list[dict], dict[str, object]]:
    def add_impl(a: int, b: int) -> int:
        called["add"] = True
        return a + b

    def mul_impl(a: int, b: int) -> int:
        called["mul"] = True
        return a * b

    tool_defs = [
        pydantic_to_openai_tool("add", "Add two integers and return the sum.", AddInput),
        pydantic_to_openai_tool("mul", "Multiply two integers and return the product.", MulInput),
    ]
    tool_funcs = {"add": add_impl, "mul": mul_impl}
    return tool_defs, tool_funcs


def _build_async_tools(called: dict) -> tuple[list[dict], dict[str, Callable]]:
    async def add_impl(a: int, b: int) -> int:
        await asyncio.sleep(0.1)
        called["add"] = True
        return a + b

    async def mul_impl(a: int, b: int) -> int:
        await asyncio.sleep(0.1)
        called["mul"] = True
        return a * b

    tool_defs = [
        pydantic_to_openai_tool("add", "Add two integers and return the sum.", AddInput),
        pydantic_to_openai_tool("mul", "Multiply two integers and return the product.", MulInput),
    ]
    tool_funcs = {"add": add_impl, "mul": mul_impl}
    return tool_defs, tool_funcs


@pytest.mark.asyncio
async def test_openai_chat_with_tools_pydantic():
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPEN_AI_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    model = os.environ.get("OPENAI_TOOL_MODEL", "gpt-4o-mini")
    client = OpenAIAPIClient(model=model)
    called = {"add": False, "mul": False}
    tool_defs, tool_funcs = _build_tools(called)
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
            tools_definitions=tool_defs,
            tools=tool_funcs,
        )
        assert isinstance(response, str)
        assert called["add"] is True
        assert called["mul"] is True
        assert "5" in response
        assert "20" in response


@pytest.mark.asyncio
async def test_llamaswap_chat_with_tools_pydantic():
    base_url = (
        os.environ.get("LLAMA_SWITCH_URL")
        or os.environ.get("LLAMA_SWAP_BASE_URL")
        or "http://india.loc:9292/v1"
    )

    model = os.environ.get("LLAMA_SWAP_TOOL_MODEL") or "Qwen3.5-9B-heretic.i1-Q5_K_M"

    client = LlamaSwapAPIClient(model=model, base_url=base_url)
    called = {"add": False, "mul": False}
    tool_defs, tool_funcs = _build_tools(called)
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
            tools_definitions=tool_defs,
            tools=tool_funcs,
        )
        assert isinstance(response, str)
        assert called["add"] is True
        assert called["mul"] is True
        assert "5" in response
        assert "20" in response


@pytest.mark.asyncio
async def test_openai_chat_with_async_tools_pydantic():
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPEN_AI_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    model = os.environ.get("OPENAI_TOOL_MODEL", "gpt-4o-mini")
    client = OpenAIAPIClient(model=model)
    called = {"add": False, "mul": False}
    tool_defs, tool_funcs = _build_async_tools(called)
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
            tools_definitions=tool_defs,
            tools=tool_funcs,
        )
        assert isinstance(response, str)
        assert called["add"] is True
        assert called["mul"] is True
        assert "5" in response
        assert "20" in response


@pytest.mark.asyncio
async def test_llamaswap_chat_with_async_tools_pydantic():
    base_url = (
        os.environ.get("LLAMA_SWITCH_URL")
        or os.environ.get("LLAMA_SWAP_BASE_URL")
        or "http://india.loc:9292/v1"
    )

    model = os.environ.get("LLAMA_SWAP_TOOL_MODEL") or "Qwen3.5-9B-heretic.i1-Q5_K_M"

    client = LlamaSwapAPIClient(model=model, base_url=base_url)
    called = {"add": False, "mul": False}
    tool_defs, tool_funcs = _build_async_tools(called)
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
            tools_definitions=tool_defs,
            tools=tool_funcs,
        )
        assert isinstance(response, str)
        assert called["add"] is True
        assert called["mul"] is True
        assert "5" in response
        assert "20" in response
