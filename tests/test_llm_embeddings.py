import pytest

from llm.ollama import OllamaAPIClient
from llm.openai import OpenAIAPIClient
from llm.llama_swap import LlamaSwapAPIClient
from llm.exceptions import LLMError


@pytest.mark.asyncio
async def test_openai_embed_batch_parsing(monkeypatch):
    client = OpenAIAPIClient(model="text-embedding-3-small", base_url="https://api.openai.com/v1", bearer_token="sk-test")

    async def fake_request(method, path, **kw):
        assert method == "POST"
        assert path == "/embeddings"
        assert kw["json"]["input"] == ["a", "b"]
        assert kw["json"]["model"] == "text-embedding-3-small"
        return {
            "data": [
                {"embedding": [0.1, 0.2]},
                {"embedding": [0.3, 0.4]},
            ]
        }

    monkeypatch.setattr(client, "_request", fake_request)
    vectors = await client.embed_batch(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.asyncio
async def test_llamaswap_embed_batch_parsing(monkeypatch):
    client = LlamaSwapAPIClient(model="some-model", base_url="http://localhost:8080/v1")

    async def fake_request(method, path, **kw):
        assert method == "POST"
        assert path == "/embeddings"
        assert kw["json"]["input"] == ["x"]
        assert kw["json"]["model"] == "some-model"
        return {"data": [{"embedding": [0.9, 0.8, 0.7]}]}

    monkeypatch.setattr(client, "_request", fake_request)
    vectors = await client.embed_batch(["x"])
    assert vectors == [[0.9, 0.8, 0.7]]


@pytest.mark.asyncio
async def test_ollama_embed_batch_fallback(monkeypatch):
    client = OllamaAPIClient(model="nomic-embed-text", base_url="http://localhost:11434")

    calls = {"embed": 0, "embeddings": 0}

    async def fake_request(method, path, **kw):
        if path == "/api/embed":
            calls["embed"] += 1
            assert kw["json"]["model"] == "nomic-embed-text"
            raise LLMError("Request failed with status 404", description="Not Found")
        if path == "/api/embeddings":
            calls["embeddings"] += 1
            assert kw["json"]["model"] == "nomic-embed-text"
            return {"embedding": [float(calls["embeddings"])]}
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr(client, "_request", fake_request)
    vectors = await client.embed_batch(["a", "b"])
    assert calls == {"embed": 1, "embeddings": 2}
    assert vectors == [[1.0], [2.0]]


@pytest.mark.asyncio
async def test_embed_single_wrapper(monkeypatch):
    client = OpenAIAPIClient(model="text-embedding-3-small", base_url="https://api.openai.com/v1", bearer_token="sk-test")

    async def fake_request(method, path, **kw):
        return {"data": [{"embedding": [0.42]}]}

    monkeypatch.setattr(client, "_request", fake_request)
    vector = await client.embed("hello")
    assert vector == [0.42]
