import os
import pytest
from llm.llama_swap_chat_completions import LlamaSwapChatCompletionsClient
from llm.message import LLMMessage
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


def assert_think_tags_well_formed(text: str) -> None:
    """If <think> tags are present, verify they are properly matched and ordered."""
    open_count = text.count("<think>")
    close_count = text.count("</think>")
    assert open_count == close_count, f"Mismatched think tags: {open_count} open, {close_count} close"
    if open_count > 0:
        assert text.index("<think>") < text.index("</think>"), "<think> should come before </think>"


# ── llama-swap ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llamaswap_thinking_streaming():
    """Test that enable_thinking produces <think>...</think> wrapped reasoning content."""
    base_url = (
        os.environ.get("LLAMA_SWITCH_URL")
        or os.environ.get("LLAMA_SWAP_BASE_URL")
        or "http://india.loc:9292/v1"
    )

    model = os.environ.get("LLAMA_SWAP_THINKING_MODEL") or "Qwen3.8-27B-Uncensored-FP8.i1-Q5_K_M"

    client = LlamaSwapChatCompletionsClient(model=model, base_url=base_url)

    messages: list[LLMMessage] = [{"role": "user", "content": "What is 15 * 37?"}]
    chunks = []

    async def on_delta(delta: str):
        chunks.append(delta)

    async with client:
        response = await client.chat(
            messages,
            stream=True,
            on_delta=on_delta,
            options={"enable_thinking": True},
        )

    full_response = "".join(chunks)
    assert response == full_response
    assert len(chunks) > 1, "Response should be streamed in multiple chunks"
    assert "<think>" in full_response, "Response should contain <think> opening tag"
    assert "</think>" in full_response, "Response should contain </think> closing tag"
    assert_think_tags_well_formed(full_response)
    # The actual answer (555) should appear after thinking
    think_end = full_response.index("</think>")
    assert "555" in full_response[think_end:], "Answer should appear after thinking block"


@pytest.mark.asyncio
async def test_llamaswap_thinking_tags_well_formed():
    """Test that think tags are well-formed across multiple calls."""
    base_url = (
        os.environ.get("LLAMA_SWITCH_URL")
        or os.environ.get("LLAMA_SWAP_BASE_URL")
        or "http://india.loc:9292/v1"
    )

    model = os.environ.get("LLAMA_SWAP_THINKING_MODEL") or "Qwen3.8-27B-Uncensored-FP8.i1-Q5_K_M"

    client = LlamaSwapChatCompletionsClient(model=model, base_url=base_url)

    messages: list[LLMMessage] = [{"role": "user", "content": "What is 15 * 37?"}]
    chunks = []

    async def on_delta(delta: str):
        chunks.append(delta)

    async with client:
        response = await client.chat(
            messages,
            stream=True,
            on_delta=on_delta,
            options={"enable_thinking": True},
        )

    full_response = "".join(chunks)
    assert response == full_response
    assert_think_tags_well_formed(full_response)


@pytest.mark.asyncio
async def test_llamaswap_thinking_streaming_with_tools():
    """chat_with_tools(stream=True) must stream <think>-wrapped reasoning too.

    The reasoning is deliberately absent from the returned content, which is conversation
    history re-sent to the model; it only reaches the caller via on_delta.
    """
    base_url = (
        os.environ.get("LLAMA_SWITCH_URL")
        or os.environ.get("LLAMA_SWAP_BASE_URL")
        or "http://india.loc:9292/v1"
    )

    model = os.environ.get("LLAMA_SWAP_THINKING_MODEL") or "Qwen3.8-27B-Uncensored-FP8.i1-Q5_K_M"

    client = LlamaSwapChatCompletionsClient(model=model, base_url=base_url)

    tools_definitions = [
        {
            "type": "function",
            "function": {
                "name": "multiply",
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
        }
    ]

    called = {"multiply": False}

    def _multiply(a: int, b: int) -> int:
        called["multiply"] = True
        return a * b

    chunks: list[str] = []

    async def on_delta(delta: str):
        chunks.append(delta)

    messages = [
        {
            "role": "user",
            "content": "Use the multiply tool to compute 15 * 37, then state the result.",
        }
    ]

    async with client:
        response = await client.chat_with_tools(
            messages,
            tools_definitions=tools_definitions,
            tools={"multiply": _multiply},
            stream=True,
            on_delta=on_delta,
            options={"enable_thinking": True},
        )

    streamed = "".join(chunks)
    assert called["multiply"] is True
    assert "<think>" in streamed, "Tool-calling stream should contain <think> opening tag"
    assert "</think>" in streamed, "Tool-calling stream should contain </think> closing tag"
    assert_think_tags_well_formed(streamed)
    assert "<think>" not in response, "Returned content must stay free of reasoning tags"
    assert "555" in response


# ── OpenAI ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_openai_o3mini_thinking_streaming():
    """Test that enable_thinking translates to reasoning_effort for the OpenAI reasoning model."""
    api_key = _openai_api_key()  # DoD-20

    model = os.environ.get("OPENAI_THINKING_MODEL", "gpt-6-luna")

    client = OpenAIChatCompletionsClient(model=model, bearer_token=api_key)

    messages: list[LLMMessage] = [{"role": "user", "content": "What is 15 * 37? Show the working steps."}]
    chunks = []

    async def on_delta(delta: str):
        chunks.append(delta)

    # enable_thinking=True should auto-translate to reasoning_effort="medium"
    async with client:
        response = await client.chat(
            messages,
            stream=True,
            on_delta=on_delta,
            options={"enable_thinking": True},
        )

    full_response = "".join(chunks)
    assert response == full_response
    assert len(chunks) > 1, "Response should be streamed in multiple chunks"
    assert "555" in full_response


@pytest.mark.asyncio
async def test_openai_o3mini_explicit_effort():
    """Test that explicit reasoning_effort overrides enable_thinking default."""
    api_key = _openai_api_key()  # DoD-20

    model = os.environ.get("OPENAI_THINKING_MODEL", "gpt-6-luna")

    client = OpenAIChatCompletionsClient(model=model, bearer_token=api_key)

    messages: list[LLMMessage] = [{"role": "user", "content": "What is 15 * 37?"}]
    chunks = []

    async def on_delta(delta: str):
        chunks.append(delta)

    async with client:
        response = await client.chat(
            messages,
            stream=True,
            on_delta=on_delta,
            options={"enable_thinking": True, "reasoning_effort": "low"},
        )

    full_response = "".join(chunks)
    assert response == full_response
    assert "555" in full_response


@pytest.mark.asyncio
async def test_openai_streaming_no_thinking_tags():
    """Test that standard OpenAI streaming (no thinking model) produces no <think> tags."""
    api_key = _openai_api_key()  # DoD-20

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAIChatCompletionsClient(model=model, bearer_token=api_key)

    messages: list[LLMMessage] = [{"role": "user", "content": "What is 15 * 37?"}]
    chunks = []

    async def on_delta(delta: str):
        chunks.append(delta)

    async with client:
        response = await client.chat(
            messages,
            stream=True,
            on_delta=on_delta,
        )

    full_response = "".join(chunks)
    assert response == full_response
    assert "<think>" not in full_response, "Standard OpenAI model should not produce <think> tags"
    assert "555" in full_response
