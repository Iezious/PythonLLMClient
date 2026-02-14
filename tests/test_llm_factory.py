from llm.llm import get_llm_client
from llm.llama_swap import LlamaSwapAPIClient
from llm.ollama import OllamaAPIClient
from llm.openai import OpenAIAPIClient
import os

def test_clients():
    # Test 1: Ollama
    client = get_llm_client("ollama", "http://localhost:11434", model="llama3")
    assert isinstance(client, OllamaAPIClient)
    assert client.model == "llama3"
    print("Ollama client created successfully")

    # Test 2: OpenAI (requires env var mock)
    os.environ["OPENAI_API_KEY"] = "sk-test"
    client = get_llm_client("openai", "https://api.openai.com/v1", model="gpt-4")
    assert isinstance(client, OpenAIAPIClient)
    assert client.model == "gpt-4"
    print("OpenAI client created successfully")

    # Test 3: LlamaSwap
    client = get_llm_client("llamaswap", "http://localhost:8080/v1", model="mistral")
    assert isinstance(client, LlamaSwapAPIClient)
    assert client.model == "mistral"
    # Check internals
    assert client.base_url == "http://localhost:8080/v1"
    assert client.bearer_token is None
    print("LlamaSwap client created successfully")

    # Test 4: LlamaSwap with token
    client = get_llm_client("llamaswap", "http://localhost:8080/v1", model="mistral", bearer_token="some-token")
    assert client.bearer_token == "some-token"
    print("LlamaSwap client with token created successfully")

if __name__ == "__main__":
    try:
        test_clients()
        print("\nAll smoke tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        exit(1)
