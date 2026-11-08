# LLMClient

Async Python client library for interacting with LLM APIs. Provides a unified interface across multiple backends with typed options, streaming support, tool calling, embeddings, and structured JSON output.

## Supported Backends

| Backend | Responses protocol (default) | Chat Completions protocol | Default URL |
|---------|------------------------------|---------------------------|-------------|
| **Ollama** | -- | `OllamaAPIClient` (native Ollama API, no protocol choice) | `http://localhost:11434` |
| **OpenAI** | `OpenAIAPIClient` | `OpenAIChatCompletionsClient` | `https://api.openai.com/v1` |
| **llama-swap** | `LlamaSwapAPIClient` | `LlamaSwapChatCompletionsClient` | `http://localhost:8080/v1` |

### Protocols

OpenAI and llama-swap are each offered over two wire protocols, one class per protocol:

- **Responses** (`/responses`) — `OpenAIAPIClient`, `LlamaSwapAPIClient`. This is the default.
- **Chat Completions** (`/chat/completions`) — `OpenAIChatCompletionsClient`, `LlamaSwapChatCompletionsClient`.
  These keep the behaviour the library had before the protocol split under the new names.

The two classes of a backend are independent (neither subclasses the other) and share the same constructor,
methods and settings keys. A class never switches protocol on its own, and the address never decides the protocol.

> **Upgrading from 0.2.x:** `OpenAIAPIClient` and `LlamaSwapAPIClient` now speak the Responses protocol.
> To keep the previous chat-completions behaviour, use `OpenAIChatCompletionsClient` /
> `LlamaSwapChatCompletionsClient` or pass `protocol="chat_completions"` to the factory.

> **Interim limitation:** tool conversations (`chat_with_tools`) are not available yet on the Responses classes.
> `OpenAIAPIClient.chat_with_tools` / `LlamaSwapAPIClient.chat_with_tools` raise `NotImplementedError` before
> sending anything, naming the chat-completions class to use meanwhile; they never fall back to
> `/chat/completions`. Use `OpenAIChatCompletionsClient` / `LlamaSwapChatCompletionsClient` for tool calling
> until Responses tool support lands.

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

# OpenAI over Responses (the default protocol); the app passes the credential
client = get_llm_client("openai", "https://api.openai.com/v1", "gpt-4o", bearer_token=api_key)

# llama-swap over Chat Completions
client = get_llm_client("llamaswap", "http://localhost:8080/v1", "qwen3", protocol="chat_completions")
```

`get_llm_client(server_type, base_url, model, timeout=600.0, bearer_token=None, default_options=None, protocol=None)`
picks the class from `server_type` and `protocol` and forwards the other arguments unchanged:

| `server_type` | `protocol=None` or `"responses"` | `protocol="chat_completions"` |
|---------------|----------------------------------|-------------------------------|
| `"openai"` | `OpenAIAPIClient` | `OpenAIChatCompletionsClient` |
| `"llamaswap"` | `LlamaSwapAPIClient` | `LlamaSwapChatCompletionsClient` |
| `"ollama"` | `OllamaAPIClient` (only with `protocol=None`) | refused |

The factory raises `ValueError` for:

- an unknown `server_type` (the message names it);
- any `protocol` given with `"ollama"` — a protocol choice is not offered on Ollama;
- a `protocol` other than `"responses"` or `"chat_completions"` (the message names the value). Matching is exact:
  `"Responses"` or `" responses"` is refused;
- `"openai"` without a `bearer_token`, whichever protocol.

### Direct client instantiation

```python
from llm import OllamaAPIClient, OpenAIAPIClient, OpenAIChatCompletionsClient

# Ollama
async with OllamaAPIClient("llama3.2") as client:
    reply = await client.generate("Explain quantum computing in one sentence.")

# OpenAI over Responses; the credential comes only from bearer_token, never from the environment
async with OpenAIAPIClient("gpt-4o", bearer_token=api_key) as client:
    reply = await client.chat([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"}
    ])

# OpenAI over Chat Completions (same constructor and methods)
async with OpenAIChatCompletionsClient("gpt-4o", bearer_token=api_key) as client:
    reply = await client.generate("Hello!")
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

On OpenAI and llama-swap, use the chat-completions classes for tool calling for now; the Responses classes
refuse `chat_with_tools` with `NotImplementedError` (see [Protocols](#protocols)).

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

### Tool Calling with JSON Output

Combine tool use with structured JSON output by passing `response_format`. The format is applied to the API payload so the final response (after all tool calls complete) is constrained to valid JSON.

```python
# Basic JSON mode — guarantees valid JSON, no schema enforcement
reply = await client.chat_with_tools(
    messages,
    tools_definitions=tools_defs,
    tools={"get_weather": get_weather},
    response_format={"type": "json_object"},
)

# Structured outputs — guarantees JSON conforming to a schema
reply = await client.chat_with_tools(
    messages,
    tools_definitions=tools_defs,
    tools={"get_weather": get_weather},
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "weather_report",
            "schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "temp": {"type": "string"},
                },
                "required": ["city", "temp"],
            },
        },
    },
)
```

Works with streaming too — `on_delta` fires for content tokens and the final result is valid JSON:

```python
reply = await client.chat_with_tools(
    messages,
    tools_definitions=tools_defs,
    tools={"get_weather": get_weather},
    stream=True,
    on_delta=handle_delta,
    response_format={"type": "json_object"},
)
```

| Backend | `json_object` | `json_schema` | Notes |
| --------- | :-----------: | :-----------: | ------- |
| OpenAI (`OpenAIChatCompletionsClient`) | Yes | Yes | Full support |
| llama-swap (`LlamaSwapChatCompletionsClient`) | Yes | Yes | Depends on underlying engine (vLLM, llama.cpp) |
| Ollama | Yes | -- | Translated to `format="json"` internally |

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

Both protocols accept the same settings keys. The Responses protocol has no counterpart for some of them
(`presence_penalty`, `frequency_penalty`, `seed`, `stop`, `logit_bias`): the Responses classes do not send
such a setting and emit an `LLMSettingIgnoredWarning` naming it, once per setting per client instance.

### Ignored-setting warning

`LLMSettingIgnoredWarning` (exported from `llm`) is a `UserWarning` subclass. It is a warning, never an error;
filter or escalate it by category with Python's `warnings` module:

```python
import warnings
from llm import LLMSettingIgnoredWarning

warnings.filterwarnings("ignore", category=LLMSettingIgnoredWarning)   # silence it
warnings.filterwarnings("error", category=LLMSettingIgnoredWarning)    # or turn it into an exception
```

## Architecture

```
llm/
  __init__.py       # Public API exports
  exceptions.py     # LLMError exception, LLMSettingIgnoredWarning category
  message.py        # LLMMessage TypedDict
  options.py        # BaseLLMOptions, OllamaOptions, OpenAIOptions
  llm_client.py     # Abstract base class (LLMClient)
  ollama.py         # OllamaAPIClient
  openai.py         # OpenAIAPIClient (Responses)
  openai_chat_completions.py      # OpenAIChatCompletionsClient (Chat Completions)
  llama_swap.py     # LlamaSwapAPIClient (Responses)
  llama_swap_chat_completions.py  # LlamaSwapChatCompletionsClient (Chat Completions)
  trace.py          # LLMTracer — debug trace logging
  llm.py            # Factory (get_llm_client) + utilities
```

## Environment Variables

The library never reads a credential from the environment: the app passes the OpenAI key as `bearer_token`.

| Variable | Used By | Purpose |
| -------- | ------- | ------- |
| `LLM_TRACE` | All clients | Enable trace logging (`1`, `true`, or `on`) |
| `LLM_TRACE_FOLDER` | All clients | Directory for trace log files |

### Debug Trace Logging

Set both `LLM_TRACE=1` and `LLM_TRACE_FOLDER=/path/to/traces` to write a human-readable log file for every LLM interaction. Each file captures system prompts, message history, request payloads, streaming chunks, tool calls, tool results, and responses.

Files are written to `<LLM_TRACE_FOLDER>/<YYYY-MM-DD>/llm-trace-<YYYYMMDD-HHMMSSffffff>.log`. When disabled (default), tracing has zero overhead.

## License

MIT
