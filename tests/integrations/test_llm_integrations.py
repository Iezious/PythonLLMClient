import pytest
import os
from llm.llama_swap import LlamaSwapAPIClient
from llm.openai import OpenAIAPIClient

# LlamaSwap configuration
LLAMA_SWAP_BASE_URL = "http://india.loc:9292/v1"
LLAMA_SWAP_GEN_MODEL = "Qwen3-Silent-Scream-6B.i1-Q6_K"
LLAMA_SWAP_EMBED_MODEL = "bge-large-en-v1.5.i1-Q6_K"

# OpenAI configuration
OPENAI_GEN_MODEL = "gpt-4o-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-small"

@pytest.mark.asyncio
async def test_llamaswap_generate():
    client = LlamaSwapAPIClient(model=LLAMA_SWAP_GEN_MODEL, base_url=LLAMA_SWAP_BASE_URL)
    async with client:
        response = await client.generate("Say hello")
        assert isinstance(response, str)
        assert len(response) > 0

@pytest.mark.asyncio
async def test_llamaswap_chat():
    client = LlamaSwapAPIClient(model=LLAMA_SWAP_GEN_MODEL, base_url=LLAMA_SWAP_BASE_URL)
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    async with client:
        response = await client.chat(messages)
        assert isinstance(response, str)
        assert "Paris" in response

@pytest.mark.asyncio
async def test_llamaswap_embed():
    client = LlamaSwapAPIClient(model=LLAMA_SWAP_EMBED_MODEL, base_url=LLAMA_SWAP_BASE_URL)
    async with client:
        vector = await client.embed("Test embedding")
        assert isinstance(vector, list)
        assert all(isinstance(x, float) for x in vector)
        assert len(vector) > 0

@pytest.mark.asyncio
async def test_llamaswap_embed_batch():
    client = LlamaSwapAPIClient(model=LLAMA_SWAP_EMBED_MODEL, base_url=LLAMA_SWAP_BASE_URL)
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
    client = LlamaSwapAPIClient(model=LLAMA_SWAP_GEN_MODEL, base_url=LLAMA_SWAP_BASE_URL)
    async with client:
        models = await client.get_models()
        assert isinstance(models, list)
        assert len(models) > 0
        for model in models:
            assert isinstance(model, dict)
            assert "id" in model


@pytest.mark.asyncio
async def test_openai_get_models():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    client = OpenAIAPIClient(model=OPENAI_GEN_MODEL)
    async with client:
        models = await client.get_models()
        assert isinstance(models, list)
        assert len(models) > 0
        for model in models:
            assert isinstance(model, dict)
            assert "id" in model


@pytest.mark.asyncio
async def test_openai_generate():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    client = OpenAIAPIClient(model=OPENAI_GEN_MODEL)
    async with client:
        response = await client.generate("Say hello")
        assert isinstance(response, str)
        assert len(response) > 0

@pytest.mark.asyncio
async def test_openai_chat():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    client = OpenAIAPIClient(model=OPENAI_GEN_MODEL)
    messages = [{"role": "user", "content": "What is the capital of Japan?"}]
    async with client:
        response = await client.chat(messages)
        assert isinstance(response, str)
        assert "Tokyo" in response

@pytest.mark.asyncio
async def test_openai_embed():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    client = OpenAIAPIClient(model=OPENAI_EMBED_MODEL)
    async with client:
        vector = await client.embed("Test embedding")
        assert isinstance(vector, list)
        assert all(isinstance(x, float) for x in vector)
        assert len(vector) > 0

@pytest.mark.asyncio
async def test_openai_embed_batch():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    client = OpenAIAPIClient(model=OPENAI_EMBED_MODEL)
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
