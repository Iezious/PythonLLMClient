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

Tool conversations (`chat_with_tools`) run on both protocols. On the Responses classes they go over
`/responses`, streamed and not; the reasoning of each round is carried to the later rounds of the same call
(never into the caller's conversation or the answer), and `response_format` is accepted and translated to the
protocol's text-format setting. The chat-completions classes remain the alternative protocol for tool calling.

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

`get_llm_client(server_type, base_url, model, timeout=600.0, bearer_token=None, default_options=None, protocol=None,
*, read_timeout=None, trace=None)` picks the class from `server_type` and `protocol` and forwards the other arguments
unchanged (see [Time limits](#time-limits) and [Debug Trace Logging](#debug-trace-logging)):

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
- `"openai"` without a `bearer_token`, whichever protocol;
- `read_timeout` given with `"ollama"` (Ollama has no streaming);
- `trace=True` when `LLM_TRACE_FOLDER` is not set.

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

On OpenAI and llama-swap, `chat_with_tools` works on both the Responses classes and the chat-completions
classes, streamed and not (see [Protocols](#protocols)).

**A failing tool call does not end the conversation by default.** When a tool call fails, the model is told with a
short fixed text and the conversation goes on, so the model can correct itself or answer without the tool. Pass
`tool_errors="raise"` to get the earlier behaviour back: the call ends with an exception and the model is not told
(see [Tool failures](#tool-failures)).

> **Changed in 0.2.1:** tolerant handling (`tool_errors="result"`) is now the default on the four OpenAI-family
> classes. Code that relied on a tool failure ending the call must pass `tool_errors="raise"`.

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

Tools may be plain functions or coroutine functions. Tools requested in one reply run one after another, in the
order the model asked for them. `max_loops` (default 5) bounds the number of server rounds; a model that keeps asking
for tools past it ends the call with an error, whichever failure handling is chosen.

#### Tool failures

A tool call can fail in five ways (the names are `ToolErrorKind`, exported from `llm`):

| Kind | What happened |
|------|---------------|
| `not_offered` | the model asked for a tool that was not offered, or gave an empty name |
| `bad_arguments` | the model's arguments cannot be read, or are not an object of named values |
| `unknown_argument` | the model passed an argument name the tool function does not declare |
| `tool_raised` | the tool raised an exception |
| `unserialisable_result` | the tool's result cannot be turned into text |

`tool_errors` chooses what happens:

- `"result"` (default, tolerant) — the failing tool is not run (or its exception is caught), the model receives a
  short fixed text as that call's result, and the loop goes on. The texts say only that the tool is not available,
  that its arguments are invalid, which argument is unknown, or that the tool failed; they never carry the
  exception's text, so your application's internals are not sent to the provider. Other tool calls of the same
  round still run, in order.
- `"raise"` (strict) — the call ends with the exception that failure has always raised: `LLMError` for
  `not_offered`, `bad_arguments` and `unserialisable_result`, `ValueError` for `unknown_argument`, `RuntimeError`
  for `tool_raised`. The model is not told.

```python
# Strict: a broken tool ends the call
reply = await client.chat_with_tools(
    messages,
    tools_definitions=tools_defs,
    tools={"get_weather": get_weather},
    tool_errors="raise",
)
```

In both modes only ordinary exceptions are handled: cancelling the call (for example `asyncio.CancelledError`)
stops it and is never reported to the model as a tool result. The round limit applies unchanged in both modes.

#### Tool event observer

Pass `on_tool_event` to see every tool call as it happens. The observer receives one `ToolEvent` (a typed
dictionary exported from `llm`) per event:

| Field | Meaning |
|-------|---------|
| `type` | `"start"`, `"result"` or `"error"` |
| `name` | the tool name the model asked for |
| `call_id` | the server's id for the call, or `None` when the server gave none |
| `arguments` | the arguments exactly as the model sent them, as text (empty text when there were none) |
| `output` | on a `"result"` event, the result text the model receives; otherwise `None` |
| `error_kind` | on an `"error"` event, the `ToolErrorKind`; otherwise `None` |

For each tool call the observer gets a `"start"` event before the tool runs, then exactly one `"result"` or
`"error"` event — also when the tool never runs (`not_offered`, `bad_arguments`, `unknown_argument`). Calls are
reported in the order they run. The observer may be a plain function or a coroutine function; the loop waits for it
to finish with each event before going on.

```python
from llm import ToolEvent

def log_tool(event: ToolEvent) -> None:
    if event["type"] == "error":
        print(f"tool {event['name']} failed: {event['error_kind']}")

reply = await client.chat_with_tools(
    messages,
    tools_definitions=tools_defs,
    tools={"get_weather": get_weather},
    on_tool_event=log_tool,
)
```

If the observer itself raises:

- strict (`tool_errors="raise"`) — the observer sees the `"error"` event before the call ends with its exception;
  an exception from the observer ends the call and reaches you unchanged.
- tolerant (default) — the observer's failure is treated like a tool failure: a tool not yet run is not run, the
  model is told the tool failed, and the loop goes on. The observer receives no further event for that tool call.
  The library emits `LLMToolObserverWarning` (a `UserWarning` subclass exported from `llm`, filterable like any
  warning) and records the failure in the trace.

#### Tools that accept any arguments

The argument names the model sends must exist in the tool function's parameters. A function that accepts
`**kwargs` skips this check and runs with whatever arguments the model sends:

```python
def lookup(**kwargs) -> dict:
    return {"received": kwargs}
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
| OpenAI (`OpenAIAPIClient`, Responses) | Yes | Yes | Same `response_format` shape, translated to the Responses text format |
| OpenAI (`OpenAIChatCompletionsClient`) | Yes | Yes | Full support |
| llama-swap (`LlamaSwapAPIClient`, Responses) | Yes | Yes | Same `response_format` shape, translated to the Responses text format; depends on underlying engine (vLLM, llama.cpp) |
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

### Errors

Failures of a library operation reach you as `LLMError` (exported from `llm`). Its text form is unchanged:
`message: description`, or just `message`. Two attributes let you tell failures apart without reading the text:

- `status` — the HTTP status when the server refused the request (a non-2xx reply), otherwise `None`.
- `kind` — one `LLMErrorKind` name, or `None`:

| `kind` | Meaning |
|--------|---------|
| `"status"` | the server refused the request; `status` holds the HTTP status |
| `"timeout"` | the total time limit or the silence limit was exceeded |
| `"connection"` | the server could not be reached |
| `"interrupted"` | the transfer was cut mid-reply |
| `"invalid_reply"` | the reply could not be read (malformed model list or embeddings, a JSON answer that does not parse) |
| `"server_failed"` | the server reported a failure, in a stream or in its reply |
| `"incomplete"` | the reply was incomplete (a stream that ended early, a reply the server marked incomplete) |
| `"tool"` | a tool conversation failed in strict mode (`not_offered`, `bad_arguments`, `unserialisable_result`) |

```python
from llm import LLMError

try:
    reply = await client.chat(messages)
except LLMError as exc:
    if exc.kind == "status" and exc.status == 429:
        ...  # rate limited
    elif exc.kind == "timeout":
        ...
```

`ValueError` and `RuntimeError` raised by the library (bad arguments, the round limit, strict tool failures) are
unchanged and carry no kind.

### Time limits

- `timeout` (constructor and factory, default 600 seconds) — one total limit per request, streamed or not.
- `read_timeout` (constructor and factory, optional; OpenAI and llama-swap classes) — a limit on silence between
  two reads of a streamed reply. A stream that sends nothing for longer fails with `LLMError` (`kind="timeout"`);
  a stream that keeps producing pieces runs to the end. It does not change `timeout` and does not affect
  non-streamed requests. Not passed means no silence limit. The factory refuses it with `ValueError` for Ollama.

```python
client = OpenAIAPIClient("gpt-4o", bearer_token=api_key, timeout=300.0, read_timeout=30.0)
```

### Ollama (deprecated)

`OllamaAPIClient` is deprecated and kept for observation only. It does not accept `tool_errors`, `on_tool_event`
or `read_timeout` (passing them raises `TypeError`; the factory refuses `read_timeout` with `ValueError`), and a
failing tool call still ends its tool conversation with an exception. It does accept `trace`.

## Architecture

```
llm/
  __init__.py       # Public API exports
  exceptions.py     # LLMError, LLMErrorKind, LLMSettingIgnoredWarning, LLMToolObserverWarning
  tool_events.py    # ToolEvent, ToolErrorKind
  message.py        # LLMMessage TypedDict
  options.py        # BaseLLMOptions, OllamaOptions, OpenAIOptions
  llm_client.py     # Shared base class (LLMClient): HTTP/SSE transport plumbing, options merge; no abc
  ollama.py         # OllamaAPIClient
  openai.py         # OpenAIAPIClient (Responses)
  openai_chat_completions.py      # OpenAIChatCompletionsClient (Chat Completions)
  llama_swap.py     # LlamaSwapAPIClient (Responses)
  llama_swap_chat_completions.py  # LlamaSwapChatCompletionsClient (Chat Completions)
  tools.py          # pydantic_to_openai_tool helper
  trace.py          # LLMTracer — debug trace logging
  llm.py            # Factory (get_llm_client) + utilities
  py.typed          # PEP 561 typing marker
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

The `trace` argument (every client constructor and the factory) decides per client, read once when the client is
created:

- not passed (`None`) — the environment decides, as described above;
- `False` — this client never traces, whatever the environment says;
- `True` — this client traces into `LLM_TRACE_FOLDER` even when `LLM_TRACE` is off; without `LLM_TRACE_FOLDER`
  the client is refused with `ValueError`.

```python
client = get_llm_client("llamaswap", "http://localhost:8080/v1", "qwen3", trace=False)
```

In a tool conversation with tolerant handling, a failing call's `TOOL RESULT` entry shows the text the model was
told, followed by a note naming the failure kind; a failing tool-event observer is recorded as well.

## License

MIT
