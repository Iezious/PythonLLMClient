import pytest
import os
from llm.llama_swap_chat_completions import LlamaSwapChatCompletionsClient
from llm.openai_chat_completions import OpenAIChatCompletionsClient

# LlamaSwap configuration
LLAMA_SWAP_BASE_URL = "http://india.loc:9292/v1"
LLAMA_SWAP_GEN_MODEL = "Qwen3.5-27B-heretic-Q5_K_M"
LLAMA_SWAP_EMBED_MODEL = "bge-large-en-v1.5.i1-Q6_K"

# OpenAI configuration
OPENAI_GEN_MODEL = "gpt-4o-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-small"


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
async def test_llamaswap_generate():
    client = LlamaSwapChatCompletionsClient(model=LLAMA_SWAP_GEN_MODEL, base_url=LLAMA_SWAP_BASE_URL)
    async with client:
        response = await client.generate("Say hello")
        assert isinstance(response, str)
        assert len(response) > 0

@pytest.mark.asyncio
async def test_llamaswap_chat():
    client = LlamaSwapChatCompletionsClient(model=LLAMA_SWAP_GEN_MODEL, base_url=LLAMA_SWAP_BASE_URL)
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    async with client:
        response = await client.chat(messages)
        assert isinstance(response, str)
        assert "Paris" in response

@pytest.mark.asyncio
async def test_llamaswap_embed():
    client = LlamaSwapChatCompletionsClient(model=LLAMA_SWAP_EMBED_MODEL, base_url=LLAMA_SWAP_BASE_URL)
    async with client:
        vector = await client.embed("Test embedding")
        assert isinstance(vector, list)
        assert all(isinstance(x, float) for x in vector)
        assert len(vector) > 0

@pytest.mark.asyncio
async def test_llamaswap_embed_batch():
    client = LlamaSwapChatCompletionsClient(model=LLAMA_SWAP_EMBED_MODEL, base_url=LLAMA_SWAP_BASE_URL)
    texts = ["Batch text 1", "Batch text 2", "Batch text 3"]
    async with client:
        # Get batch vectors
        vectors = await client.embed_batch(texts)
        assert isinstance(vectors, list)
        assert len(vectors) == len(texts)
        for i, v in enumerate(vectors):
            assert isinstance(v, list)
            assert len(v) > 0
            assert all(isinstance(x, float) for x in v)

@pytest.mark.asyncio
async def test_llamaswap_get_models():
    client = LlamaSwapChatCompletionsClient(model=LLAMA_SWAP_GEN_MODEL, base_url=LLAMA_SWAP_BASE_URL)
    async with client:
        models = await client.get_models()
        assert isinstance(models, list)
        assert len(models) > 0
        for model in models:
            assert isinstance(model, dict)
            assert "id" in model


@pytest.mark.asyncio
async def test_openai_get_models():
    api_key = _openai_api_key()  # DoD-20
    client = OpenAIChatCompletionsClient(model=OPENAI_GEN_MODEL, bearer_token=api_key)
    async with client:
        models = await client.get_models()
        assert isinstance(models, list)
        assert len(models) > 0
        for model in models:
            assert isinstance(model, dict)
            assert "id" in model


@pytest.mark.asyncio
async def test_openai_generate():
    api_key = _openai_api_key()  # DoD-20
    client = OpenAIChatCompletionsClient(model=OPENAI_GEN_MODEL, bearer_token=api_key)
    async with client:
        response = await client.generate("Say hello")
        assert isinstance(response, str)
        assert len(response) > 0

@pytest.mark.asyncio
async def test_openai_chat():
    api_key = _openai_api_key()  # DoD-20
    client = OpenAIChatCompletionsClient(model=OPENAI_GEN_MODEL, bearer_token=api_key)
    messages = [{"role": "user", "content": "What is the capital of Japan?"}]
    async with client:
        response = await client.chat(messages)
        assert isinstance(response, str)
        assert "Tokyo" in response

@pytest.mark.asyncio
async def test_openai_embed():
    api_key = _openai_api_key()  # DoD-20
    client = OpenAIChatCompletionsClient(model=OPENAI_EMBED_MODEL, bearer_token=api_key)
    async with client:
        vector = await client.embed("Test embedding")
        assert isinstance(vector, list)
        assert all(isinstance(x, float) for x in vector)
        assert len(vector) > 0

@pytest.mark.asyncio
async def test_openai_embed_batch():
    api_key = _openai_api_key()  # DoD-20
    client = OpenAIChatCompletionsClient(model=OPENAI_EMBED_MODEL, bearer_token=api_key)
    texts = ["Batch text A", "Batch text B"]
    async with client:
        # Get batch vectors
        vectors = await client.embed_batch(texts)
        assert isinstance(vectors, list)
        assert len(vectors) == len(texts)
        for i, v in enumerate(vectors):
            assert isinstance(v, list)
            assert len(v) > 0
            assert all(isinstance(x, float) for x in v)
