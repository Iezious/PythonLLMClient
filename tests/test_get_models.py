import pytest

from llm.openai import OpenAIAPIClient
from llm.llama_swap import LlamaSwapAPIClient
from llm.exceptions import LLMError


SAMPLE_MODELS = [
    {"id": "gpt-4o", "object": "model", "created": 1715367049, "owned_by": "system"},
    {"id": "gpt-4o-mini", "object": "model", "created": 1721172741, "owned_by": "system"},
]


@pytest.mark.asyncio
async def test_openai_get_models(monkeypatch):
    client = OpenAIAPIClient(model="gpt-4o-mini", base_url="https://api.openai.com/v1", bearer_token="sk-test")

    async def fake_request(method, path, **kw):
        assert method == "GET"
        assert path == "/models"
        return {"object": "list", "data": SAMPLE_MODELS}

    monkeypatch.setattr(client, "_request", fake_request)
    models = await client.get_models()
    assert models == SAMPLE_MODELS
    assert all("id" in m for m in models)


@pytest.mark.asyncio
async def test_llamaswap_get_models(monkeypatch):
    client = LlamaSwapAPIClient(model="some-model", base_url="http://localhost:8080/v1")

    async def fake_request(method, path, **kw):
        assert method == "GET"
        assert path == "/models"
        return {"object": "list", "data": SAMPLE_MODELS}

    monkeypatch.setattr(client, "_request", fake_request)
    models = await client.get_models()
    assert models == SAMPLE_MODELS
    assert len(models) == 2


@pytest.mark.asyncio
async def test_openai_get_models_invalid_response(monkeypatch):
    client = OpenAIAPIClient(model="gpt-4o-mini", base_url="https://api.openai.com/v1", bearer_token="sk-test")

    async def fake_request(method, path, **kw):
        return {"error": "something went wrong"}

    monkeypatch.setattr(client, "_request", fake_request)
    with pytest.raises(LLMError):
        await client.get_models()


@pytest.mark.asyncio
async def test_llamaswap_get_models_invalid_response(monkeypatch):
    client = LlamaSwapAPIClient(model="some-model", base_url="http://localhost:8080/v1")

    async def fake_request(method, path, **kw):
        return {"error": "something went wrong"}

    monkeypatch.setattr(client, "_request", fake_request)
    with pytest.raises(LLMError):
        await client.get_models()
