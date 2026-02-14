from llm.llm import get_llm_client
from llm.ollama import OllamaAPIClient
from llm.openai import OpenAIAPIClient
from llm.options import OllamaOptions
import os

def test_option_precedence():
    print("Testing option precedence...")

    # Setup OpenAI key for mock
    os.environ["OPENAI_API_KEY"] = "sk-test"

    # 1. Constructor defaults (using typed dict)
    defaults = OllamaOptions(temperature=0.5, top_p=0.8)
    client = get_llm_client("ollama", "http://localhost:11434", model="llama3", default_options=defaults)

    # Check if defaults are stored
    assert client.default_options == defaults, "Constructor defaults not stored"

    # Mock _merge_options check
    # Case A: Use only constructor defaults
    opts = client._merge_options(options=None)
    assert opts["temperature"] == 0.5, f"Expected temp 0.5, got {opts.get('temperature')}"
    assert opts["top_p"] == 0.8, f"Expected top_p 0.8, got {opts.get('top_p')}"

    # Case B: Override with explicit typed options
    call_options = OllamaOptions(temperature=0.9)
    opts = client._merge_options(options=call_options)
    assert opts["temperature"] == 0.9, f"Expected temp 0.9 (override), got {opts.get('temperature')}"
    assert opts["top_p"] == 0.8, f"Expected top_p 0.8 (inherited), got {opts.get('top_p')}"

    # Case C: Aliasing (Ollama: max_tokens -> num_predict is NOT automatically handled by TypedDict keys unless mapped manually)
    # The OllamaOptions TypedDict uses `num_predict` directly.
    # If we want to test that specific key works:
    client2 = get_llm_client("ollama", "http://localhost:11434", model="llama3", default_options=OllamaOptions(num_predict=100))
    opts = client2._merge_options(options=None)
    assert opts["num_predict"] == 100, f"Expected num_predict 100, got {opts.get('num_predict')}"

    print("Option precedence tests passed!")

if __name__ == "__main__":
    try:
        test_option_precedence()
        print("\nAll integration tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        exit(1)
