import os
import pytest
from llm.llama_swap import LlamaSwapAPIClient
from llm.openai import OpenAIAPIClient


def _openai_api_key() -> str:
    """Return the OpenAI key from the test's own environment, skipping the test when absent.

    Step 004 DoD-20: the test reads the key and passes it as ``bearer_token``; the library never
    reads the environment itself (DEC-001-6).
    """
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_AI_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY / OPEN_AI_KEY not set")
    return key


@pytest.mark.asyncio
async def test_openai_chat_streaming():
    api_key = _openai_api_key()  # DoD-20

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAIAPIClient(model=model, bearer_token=api_key)

    messages = [{"role": "user", "content": "Write a short poem about the rust programming language."}]
    chunks = []

    async def on_delta(delta: str):
        chunks.append(delta)

    async with client:
        response = await client.chat(
            messages,
            stream=True,
            on_delta=on_delta
        )

    full_response = "".join(chunks)
    assert response == full_response
    assert len(chunks) > 1, "Response should be streamed in multiple chunks"
    assert "Rust" in full_response or "rust" in full_response

@pytest.mark.asyncio
async def test_llamaswap_chat_streaming():
    base_url = (
        os.environ.get("LLAMA_SWITCH_URL")
        or os.environ.get("LLAMA_SWAP_BASE_URL")
        or "http://india.loc:9292/v1"
    )

    model = os.environ.get("LLAMA_SWAP_MODEL") or "Qwen3.5-27B-heretic-Q5_K_M"

    client = LlamaSwapAPIClient(model=model, base_url=base_url)

    messages = [{"role": "user", "content": "Write a short poem about the rust programming language."}]
    chunks = []

    async def on_delta(delta: str):
        chunks.append(delta)

    async with client:
        response = await client.chat(
            messages,
            stream=True,
            on_delta=on_delta
        )

    full_response = "".join(chunks)
    assert response == full_response
    assert len(chunks) > 1, "Response should be streamed in multiple chunks"
    assert "Rust" in full_response or "rust" in full_response
