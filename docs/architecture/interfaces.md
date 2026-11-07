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
| `chat_json(messages, *, response_schema, system, options)` | Requests JSON mode and returns the reply as JSON text. On OpenAI and llama-swap `response_schema` gates the parse check (always run on Ollama) and is not sent to the backend today (see error contract). |
| `chat_with_tools(messages, *, tools_definitions, tools, system, max_loops, options, stream, on_delta, response_format)` | Runs the tool loop and returns the final answer text. No transcript is returned and the caller's `messages` is not changed. |
| `embed(text, *, options)` | Returns one embedding vector; defined on top of `embed_batch`. |
| `embed_batch(texts, *, options)` | Returns one vector per input text, in input order; an empty batch gives an empty result (not met today, see error contract). |
| `async with client:` | Session lifecycle (see `overview.md`). |

Per-method behaviour that varies by backend is in the capability matrix below.

### Streaming callback

`on_delta` is optional. It receives one string per streamed piece and may be a plain function or a coroutine function. With `stream=True` and a callback the caller sees pieces as they arrive; the method still returns the full text.

- The receiver gets pieces in order, and the next piece is not read until the receiver has finished with the current one.
- A receiver failure ends the call and propagates to the caller.
- On the tool path the receiver sees every round's pieces, but only the final answer is returned.

### Tool loop contract

- `tools_definitions` is a list of tool definitions (see `data-model.md`); `tools` maps tool name to a callable (sync or async).
- Every advertised tool name must have a callable, checked before any request is sent.
- Each round sends the conversation and tool definitions; if the reply has no tool calls it is the answer; otherwise each requested tool is executed and its result is added to the conversation, then the next round runs.
- Tool arguments are accepted as a JSON string or a mapping; every argument name must exist in the callable's parameters. Required-ness and types are not checked.
- Tools requested in one reply run sequentially, in the order requested. Extra callables beyond the advertised tools are allowed.
- A tool result that is a mapping or list is serialised to JSON text; anything else is converted to text.
- A tool failure, or an argument name the callable does not have, ends the call; the model is not told.
- `max_loops` bounds the number of server rounds, and the round that produces the answer counts. Tools requested in the last permitted round still run; if the model asks for more, the call fails (see error contract).
- `response_format`, when given, is sent on every round and is not validated against the reply.

## Exposed: capability matrix

| Capability | Ollama (**deprecated**) | OpenAI | llama-swap |
|---|---|---|---|
| Auth | optional caller-provided bearer | caller-provided bearer required | optional caller-provided bearer |
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
- The factory forwards the caller's credential (`bearer_token`) to every backend. The library never reads a credential from the environment. The OpenAI client refuses to be created without a credential (`ValueError`); llama-swap and Ollama accept none.
- Known defect: today the OpenAI branch drops the caller's credential and reads the environment instead; see `quick-reference.md`, Open.

## Exposed: utilities

The supported surface is the package top level.

- `pydantic_to_openai_tool(name, description, schema)` turns a pydantic model into an OpenAI function-tool definition, used for all backends. The schema library must not be required to load the package; today it is (known defect, see `quick-reference.md`, Open).
- `substitute_placeholders(template, placeholders)` is the single-fill helper: it replaces `{{KEY}}` with the text of the value, values chaining in mapping order; unknown keys are left as written.
- Repeated-fill template helper: supported; top-level export pending (see `quick-reference.md`, Open).
- Typed settings shapes and the message shape are public; top-level export is pending (see `data-model.md`, Settings).
- `LLMTracer` is a supported surface: an application may open a trace session with any operation label and log to it (see `subsystems.md`).

## Exposed: error contract

Rule: unless the table below names another error, a failure of a library operation reaches the caller as `LLMError` (`message`, optional `description`; text form is `message: description`). Error text carries the request address and the full server reply, and is recorded in the trace (accepted). There are no retries.

| Failure kind | Intended | Today |
|---|---|---|
| Non-2xx reply, non-streaming | `LLMError` | as intended |
| Non-2xx reply, streaming chat and streaming tool rounds | `aiohttp.ClientResponseError`, without the server reply (accepted) | as intended |
| Transport failure (server unreachable, timeout) | `LLMError` | aiohttp / asyncio exceptions pass through (known defect, see Open) |
| 2xx reply that is not JSON | `LLMError` on model listing and embeddings; on chat-style calls a generic error (accepted) | not mapped on listing and embeddings (known defect, see Open) |
| Reply with null content, non-streaming (streaming yields empty text) | empty text | `None` is returned (known defect, see Open) |
| Reply missing the expected choice or message | generic error (accepted) | `KeyError` / `IndexError` / `TypeError` |
| In-stream error piece, or stream that ends early | `LLMError` | skipped silently; the call returns what arrived (known defect, see Open) |
| Tool offered without a callable | `ValueError`, before any request | as intended |
| Tool requested that was not offered | `LLMError` | raw lookup error (known defect, see Open) |
| Tool arguments unreadable | `LLMError` | raw decode error (known defect, see Open) |
| Tool argument name not in the callable's parameters | `ValueError` | as intended |
| Tool raised | `RuntimeError` naming the tool | as intended |
| Tool result not serialisable | `LLMError` | raw serialisation error (known defect, see Open) |
| Round limit exhausted | error that states the round limit as the reason | `RuntimeError` whose text does not clearly say so (known defect, see Open) |
| Malformed model list | `LLMError` | `LLMError` on OpenAI and llama-swap; empty list on Ollama (accepted); entry without a name or id not mapped (known defect, see Open) |
| Malformed embeddings reply, or entry without a vector | `LLMError` | `LLMError` for a missing result list; entry without a vector not mapped (known defect, see Open) |
| Operation not supported by a backend (`get_models` on the deprecated Ollama client) | refused before the server is contacted | `NotImplementedError` |
| `stream=True` or `on_delta` on the deprecated Ollama client | refused before the server is contacted | `ValueError` / `TypeError` |
| Unsupported `server_type` | `ValueError` | as intended |
| Missing OpenAI credential (none from the caller) | `ValueError` at creation | raised only when the environment also holds no key (known defect, see Open) |
| Receiver failure | ends the call and propagates unchanged (DEC-000-23) | as intended |

Other rules:

- JSON-only answer (`chat_json`): the parse check runs only when an expected shape is given; on the deprecated Ollama client it always runs. The expected shape is not sent to the backend today (open, see Open). A failed parse check is `LLMError`.
- Embeddings: one vector per input text, in input order; an empty batch gives an empty result without a request. Neither is met today (known defects, see Open).

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

**SSE stream format** (OpenAI and llama-swap): `data: {json}` frames terminated by `data: [DONE]`. Only the first choice's `delta` is read, with `content`, `reasoning_content` and `tool_calls`; reasoning is read only from the dedicated `reasoning_content` field. Frames without `choices` and non-JSON frames are skipped today (known defect for error frames, see error contract).

- Text is decoded incrementally. Known defect: a multi-byte character split between two reads is lost.
- Line terminators and multi-line events are to be honoured. Known defect: a bare carriage-return terminator and multi-line events lose text.
- Tool-call fragments are matched by their position and returned in order. Known defect: fragments without a position, or repeating the name in every fragment, merge into one broken call.
- One total time limit bounds a whole stream, per server round.
