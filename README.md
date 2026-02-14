# LLMClient

Async Python client library for interacting with LLM APIs. Provides a unified interface across multiple backends with typed options, streaming support, tool calling, embeddings, and structured JSON output.

## Supported Backends

| Backend | Class | Default URL |
|---------|-------|-------------|
| **Ollama** | `OllamaAPIClient` | `http://localhost:11434` |
| **OpenAI** | `OpenAIAPIClient` | `https://api.openai.com/v1` |
| **llama-swap** | `LlamaSwapAPIClient` | `http://localhost:8080/v1` |

## Installation

```bash
pip install git+https://github.com/Iezious/PythonLLMClient.git
```

To add as a dependency in your project's `pyproject.toml`:

```toml
dependencies = [
    "llm-client @ git+https://github.com/Iezious/PythonLLMClient.git",
]
```

## Quick Start

### Using the factory

```python
from llm import get_llm_client

async with get_llm_client("ollama", "http://localhost:11434", "llama3.2") as client:
    reply = await client.chat([
        {"role": "user", "content": "Hello!"}
    ])
    print(reply)
```

### Direct client instantiation

```python
from llm import OllamaAPIClient, OpenAIAPIClient

# Ollama
async with OllamaAPIClient("llama3.2") as client:
    reply = await client.generate("Explain quantum computing in one sentence.")

# OpenAI (reads OPENAI_API_KEY from environment)
async with OpenAIAPIClient("gpt-4o") as client:
    reply = await client.chat([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"}
    ])
```

## Features

### Chat & Generate

```python
# Chat with message history
reply = await client.chat([
    {"role": "user", "content": "What is Python?"}
])

# Single prompt generation
reply = await client.generate("Write a haiku about code.")

# With system message
reply = await client.chat(messages, system="You are a pirate.")
```

### Streaming (OpenAI / llama-swap)

```python
async def handle_delta(delta: str):
    print(delta, end="", flush=True)

reply = await client.chat(messages, stream=True, on_delta=handle_delta)
```

### Structured JSON Output

```python
json_str = await client.chat_json([
    {"role": "user", "content": "List 3 colors as JSON: {\"colors\": [...]}"}
])
```

### Tool Calling

```python
def get_weather(city: str) -> dict:
    return {"city": city, "temp": "22C"}

tools_defs = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
}]

reply = await client.chat_with_tools(
    [{"role": "user", "content": "What's the weather in Paris?"}],
    tools_definitions=tools_defs,
    tools={"get_weather": get_weather},
)
```

### Embeddings

```python
# Single text
vector = await client.embed("Hello world")

# Batch
vectors = await client.embed_batch(["Hello", "World"])
```

### Options

Each backend accepts typed options dictionaries:

```python
from llm.options import OllamaOptions, OpenAIOptions

# Ollama-specific
reply = await client.chat(messages, options=OllamaOptions(
    temperature=0.7, top_k=50, num_predict=1024
))

# OpenAI-specific
reply = await client.chat(messages, options=OpenAIOptions(
    temperature=0.5, max_tokens=500
))
```

## Architecture

```
llm/
  __init__.py       # Public API exports
  exceptions.py     # LLMError exception
  message.py        # LLMMessage TypedDict
  options.py        # BaseLLMOptions, OllamaOptions, OpenAIOptions
  llm_client.py     # Abstract base class (LLMClient)
  ollama.py         # OllamaAPIClient
  openai.py         # OpenAIAPIClient
  llama_swap.py     # LlamaSwapAPIClient
  llm.py            # Factory (get_llm_client) + utilities
```

## Environment Variables

| Variable | Used By | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | `OpenAIAPIClient` | API authentication |
| `OPEN_AI_KEY` | `OpenAIAPIClient` | Alternative API key variable |

## License

MIT
