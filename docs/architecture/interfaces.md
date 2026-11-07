# Interfaces

## Purpose

Contracts the library exposes to callers and the backend APIs it consumes. Data shapes are in `data-model.md`; component structure in `subsystems.md`.

## Exposed: client method contract

All methods are async and defined on `LLMClient`; each backend client implements them. `OllamaAPIClient` is **deprecated**.

| Method | Contract |
|---|---|
| `list_models()` | Returns model ids as a list of strings. |
| `get_models()` | Returns the raw model records. OpenAI and llama-swap only; on the deprecated Ollama client it raises `NotImplementedError`. |
| `generate(prompt, *, system, options, stream, on_delta)` | Returns reply text. A `system` instruction is passed along with the prompt. |
| `chat(messages, *, system, options, stream, on_delta)` | A truthy `system` is prepended as a system message to a copy of `messages`; the caller's list is not changed. Returns the reply text only. |
| `chat_json(messages, *, response_schema, system, options)` | Requests JSON mode and returns the reply as JSON text. `response_schema` is not sent to the backend. |
| `chat_with_tools(messages, *, tools_definitions, tools, system, max_loops, options, stream, on_delta, response_format)` | Runs the tool loop and returns the final answer text. No transcript is returned and the caller's `messages` is not changed. |
| `embed(text, *, options)` | Returns one embedding vector; defined on top of `embed_batch`. |
| `embed_batch(texts, *, options)` | Returns one vector per input text. |
| `async with client:` | Session lifecycle (see `overview.md`). |

Per-method behaviour that varies by backend is in the capability matrix below.

### Streaming callback

`on_delta` is optional. It receives one string per streamed piece and may be a plain function or a coroutine function. With `stream=True` and a callback the caller sees pieces as they arrive; the method still returns the full text.

### Tool loop contract

- `tools_definitions` is a list of tool definitions (see `data-model.md`); `tools` maps tool name to a callable (sync or async).
- Every advertised tool name must have a callable, checked before any request is sent.
- Each round sends the conversation and tool definitions; if the reply has no tool calls it is the answer; otherwise each requested tool is executed and its result is added to the conversation, then the next round runs.
- Tool arguments are accepted as a JSON string or a mapping; every argument name must exist in the callable's parameters. Required-ness and types are not checked.
- A tool result that is a mapping or list is serialised to JSON text; anything else is converted to text.
- A tool exception aborts the whole call; it is not fed back to the model.
- `max_loops` bounds the number of rounds; exceeding it is an error.
- `response_format`, when given, is sent on every round.

## Exposed: capability matrix

| Capability | Ollama (**deprecated**) | OpenAI | llama-swap |
|---|---|---|---|
| Auth | optional bearer | bearer required (argument, else `OPEN_AI_KEY`, else `OPENAI_API_KEY`) | optional bearer |
| Streaming in `chat` | not supported | yes | yes |
| Streaming in `chat_with_tools` | not supported | yes | yes |
| `on_delta` parameter | not accepted | yes | yes |
| `generate` route | native generate route | through chat | through chat |
| `get_models` | not supported | yes | yes |
| `chat_json` request | native JSON format flag | `response_format` of type `json_object` | same as OpenAI |
| `chat_json` validation | always checks the reply parses | only when `response_schema` is given | same as OpenAI |
| `response_format` in tool chat | any value collapses to the JSON format flag | passed through | passed through |
| Thinking control | none | `enable_thinking` is translated to `reasoning_effort` | `enable_thinking` is sent; `reasoning_effort` is dropped |
| Reasoning tagged with `<think>` | no | streaming only | streaming only |
| Embeddings route | embed route, with legacy per-text fallback | embeddings route | embeddings route |
| Sampling options in embedding request | yes | no | no |
| Assistant tool-call turn kept in history | no | yes | yes |
| Tool-result message identity | tool `name` | `tool_call_id` | `tool_call_id` |
| Tool-call argument encoding | mapping | JSON string | JSON string |

OpenAI and llama-swap differ only in: auth requirement, thinking control, and default address.

On the deprecated Ollama client, `stream=True` raises `ValueError` and passing `on_delta` raises `TypeError`.

## Exposed: factory

`get_llm_client(server_type, base_url, model, timeout, bearer_token, default_options)` returns an `LLMClient`.

- `server_type` is one of `ollama` (deprecated), `openai`, `llamaswap`; any other value raises `ValueError`.
- `base_url` is required at the factory; the client constructors have their own default address.
- For `openai`, the `bearer_token` argument is not forwarded; the credential is resolved from `OPEN_AI_KEY`, then `OPENAI_API_KEY`, and `ValueError` is raised if neither is set.

## Exposed: utilities

- `pydantic_to_openai_tool(name, description, schema)` turns a pydantic model into an OpenAI function-tool definition, used for all backends. Because the package imports it at import time, `pydantic` is needed to import `llm` at all.
- `substitute_placeholders(template, placeholders)` replaces `{{KEY}}` with the text of the value. It is not recursive; unknown keys are left as written.
- `LLMTracer` is exported for the trace facility (see `subsystems.md`).

## Exposed: error contract

| Error | Raised for |
|---|---|
| `LLMError` (`message`, optional `description`; text form is `message: description`) | Non-2xx on non-streaming requests (description is the response body); malformed model-list or embeddings responses; invalid JSON from `chat_json`; empty embedding result. |
| `aiohttp.ClientResponseError` | Non-2xx on streaming chat and on streaming tool rounds. |
| `ValueError` | Missing tool callables; bad tool arguments; unsupported `server_type`; missing OpenAI credential; `stream=True` on Ollama. |
| `RuntimeError` | A tool raised; tool loop exceeded `max_loops`. |
| `KeyError` / `IndexError` | Non-streaming reply missing the expected choice or message. |
| aiohttp transport and timeout exceptions | Passed through unwrapped. |

A reply whose content is explicitly null is returned as `None`. There are no retries.

## Consumed: backend APIs

Authentication on every request, when a token is present: `Authorization: Bearer <token>`. No other scheme.

**Ollama (deprecated)** — base address has no `/v1`:

| Route | Used for |
|---|---|
| `GET /api/tags` | `list_models` (model `name` per entry) |
| `POST /api/generate` | `generate` (payload: model, prompt, `stream` false, nested `options`, optional `system`) |
| `POST /api/chat` | `chat`, `chat_json`, `chat_with_tools` (optional `format` JSON flag, optional `tools`) |
| `POST /api/embed` | `embed_batch` (array input) |
| `POST /api/embeddings` | per-text fallback when the embed route answers 404 |

**OpenAI and llama-swap** — base address includes `/v1`:

| Route | Used for |
|---|---|
| `GET /models` | `list_models`, `get_models` (records under `data`) |
| `POST /chat/completions` | all chat-style operations; options are top-level fields; optional `response_format` and `tools`; `stream` selects SSE |
| `POST /embeddings` | `embed_batch` (array input, optional `user`; no sampling options) |

**SSE stream format** (OpenAI and llama-swap): `data: {json}` frames terminated by `data: [DONE]`. Frames without `choices` and non-JSON frames are skipped. Only the first choice's `delta` is read, with `content`, `reasoning_content` and `tool_calls`.
