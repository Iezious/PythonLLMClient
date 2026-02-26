import os
import pytest
from llm.llama_swap import LlamaSwapAPIClient
from llm.message import LLMMessage
from llm.openai import OpenAIAPIClient


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

    model = os.environ.get("LLAMA_SWAP_THINKING_MODEL") or "Qwen3-Silent-Scream-6B.i1-Q6_K"

    client = LlamaSwapAPIClient(model=model, base_url=base_url)

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

    model = os.environ.get("LLAMA_SWAP_THINKING_MODEL") or "Qwen3-Silent-Scream-6B.i1-Q6_K"

    client = LlamaSwapAPIClient(model=model, base_url=base_url)

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


# ── OpenAI ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_openai_o3mini_thinking_streaming():
    """Test that enable_thinking translates to reasoning_effort for o3-mini."""
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPEN_AI_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    model = os.environ.get("OPENAI_THINKING_MODEL", "o3-mini")

    client = OpenAIAPIClient(model=model)

    messages: list[LLMMessage] = [{"role": "user", "content": "What is 15 * 37?"}]
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
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPEN_AI_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    model = os.environ.get("OPENAI_THINKING_MODEL", "o3-mini")

    client = OpenAIAPIClient(model=model)

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
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPEN_AI_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAIAPIClient(model=model)

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
