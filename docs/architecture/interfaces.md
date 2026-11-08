# Interfaces

## Purpose

Contracts the library exposes to callers and the backend APIs it consumes. Data shapes are in `data-model.md`; component structure in `subsystems.md`.

## Exposed: client method contract

All methods are async and defined on `LLMClient`; each backend client implements them. `OllamaAPIClient` is **deprecated**. The contract below is the same on both OpenAI-family protocols, tool conversations included: Responses and chat completions differ only in the rows of the protocol matrix further down.

| Method | Contract |
|---|---|
| `list_models()` | Returns model ids as a list of strings. |
| `get_models()` | Returns the raw model records. OpenAI and llama-swap only; on the deprecated Ollama client it raises `NotImplementedError`. |
| `generate(prompt, *, system, options, stream, on_delta)` | Returns reply text. A `system` instruction is passed along with the prompt. |
| `chat(messages, *, system, options, stream, on_delta)` | A truthy `system` is prepended as a system message to a copy of `messages`; the caller's list is not changed. Returns the reply text only. |
| `chat_json(messages, *, response_schema, system, options)` | Requests JSON mode and returns the reply as JSON text. On OpenAI and llama-swap `response_schema` gates the parse check (always run on Ollama) and is not sent to the backend today (see error contract). |
| `chat_with_tools(messages, *, tools_definitions, tools, system, max_loops, options, stream, on_delta, response_format)` | Runs the tool loop and returns the final answer text. No transcript is returned and the caller's `messages` is not changed. |
| `embed(text, *, options)` | Returns one embedding vector; defined on top of `embed_batch`. |
| `embed_batch(texts, *, options)` | Returns one vector per input text, in input order; an empty batch gives an empty result (the empty batch is not met today, see error contract). |
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

These rules hold on both OpenAI-family protocols.

## Exposed: capability matrix

| Capability | Ollama (**deprecated**) | OpenAI | llama-swap |
|---|---|---|---|
| Auth | optional caller-provided bearer | caller-provided bearer required | optional caller-provided bearer |
| Streaming in `chat` | not supported | yes | yes |
| Streaming in `chat_with_tools` | not supported | yes | yes |
| `on_delta` parameter | not accepted | yes | yes |
| `generate` route | native generate route | through chat | through chat |
| `get_models` | not supported | yes | yes |
| `chat_json` request | native JSON format flag | the app's `response_format` setting when set, otherwise `response_format` of type `json_object` | same as OpenAI |
| `chat_json` validation | always checks the reply parses | only when `response_schema` is given | same as OpenAI |
| `response_format` in tool chat | any value collapses to the JSON format flag | passed through | passed through |
| Thinking control | none | `enable_thinking` is translated to `reasoning_effort` | `enable_thinking` is sent; `reasoning_effort` is dropped |
| Reasoning tagged with `<think>` | no | streaming only | streaming only |
| Embeddings route | embed route, with legacy per-text fallback | embeddings route | embeddings route |
| Sampling options in embedding request | yes | no | no |
| Assistant tool-call turn kept in history | no | yes | yes |
| Tool-result message identity | tool `name` | `tool_call_id` | `tool_call_id` |
| Tool-call argument encoding | mapping | JSON string | JSON string |

OpenAI and llama-swap differ only in: auth requirement, thinking control, and default address. The OpenAI and llama-swap columns above describe the chat-completions classes; the Responses classes differ from them only in the rows of the protocol axis below.

### Protocol axis, OpenAI family

Rows that differ between the two protocols. All other rows of the matrix above hold for both.

| Capability | Responses (`OpenAIAPIClient`, `LlamaSwapAPIClient`) | Chat completions (`OpenAIChatCompletionsClient`, `LlamaSwapChatCompletionsClient`) |
|---|---|---|
| Chat route | `POST /responses` | `POST /chat/completions` |
| `chat_json` request | the protocol's JSON text-format setting; the caller's `response_format` shape is accepted unchanged and translated | the app's `response_format` setting when set, otherwise `response_format` of type `json_object` |
| `response_format` in tool chat | translated to the protocol's text-format setting | passed through |
| Thinking control | on both backends, thinking on/off and an explicit effort map to the protocol's reasoning setting, and an explicit effort wins; only OpenAI also asks for the reasoning summary. A model that refuses the requested effort fails the call with the server's reason; there is no retry (DEC-arch-8; values in DEC-003-1, DEC-003-2) | as in the matrix above |
| Settings sent | only those the app set; a setting with no counterpart is ignored with the warning | only those the app set |
| Reasoning carried between tool rounds | yes, within one call, never into the caller's conversation or the answer | no |
| Reasoning and content in one streamed piece | not applicable (they arrive separately) | both delivered, reasoning first |
| Streamed non-2xx reply | `LLMError` with address and server reply | `LLMError` with address and server reply |
| Failure reported inside the stream | `LLMError` with the server's reason | `LLMError` with the server's reason |
| Stream that ends without its completion event | `LLMError` | `LLMError` when neither completion signal arrived (see Consumed: SSE stream format) |

On the deprecated Ollama client, `stream=True` raises `ValueError` and passing `on_delta` raises `TypeError`.

## Exposed: factory

`get_llm_client(server_type, base_url, model, timeout, bearer_token, default_options, protocol)` returns an `LLMClient`.

- `server_type` is one of `ollama` (deprecated), `openai`, `llamaswap`; any other value raises `ValueError`.
- `protocol` is optional and selects the class for `openai` and `llamaswap`:
  - not passed, or `responses` → `OpenAIAPIClient` / `LlamaSwapAPIClient`
  - `chat_completions` → `OpenAIChatCompletionsClient` / `LlamaSwapChatCompletionsClient`
  - Any other value raises `ValueError`.
  - Passing any protocol with `ollama` is refused with `ValueError`, because Ollama offers no protocol choice.
  - The base address never influences the choice. Details are in DEC-arch-4 and DEC-003-8.
- `base_url` is required at the factory; the client constructors have their own default address.
- The factory forwards the caller's credential (`bearer_token`) to every backend. The library never reads a credential from the environment. The OpenAI client refuses to be created without a credential (`ValueError`); llama-swap and Ollama accept none.
- The earlier defect (the OpenAI branch dropped the caller's credential and the client read the environment) is fixed (DEC-001-5, DEC-001-6).

## Exposed: utilities

The supported surface is the package top level.

- `pydantic_to_openai_tool(name, description, schema)` turns a pydantic model into an OpenAI function-tool definition, used for all backends. The schema library it relies on is a declared runtime dependency, so the package loads after a normal install (DEC-001-3).
- `substitute_placeholders(template, placeholders)` is the single-fill helper: it replaces `{{KEY}}` with the text of the value, values chaining in mapping order; unknown keys are left as written.
- Repeated-fill template helper: supported; top-level export pending (see `quick-reference.md`, Open).
- Typed settings shapes and the message shape are public; top-level export is pending (see `data-model.md`, Settings).
- `LLMTracer` is a supported surface: an application may open a trace session with any operation label and log to it (see `subsystems.md`).

## Exposed: warnings

The library exports its own warning category, `LLMSettingIgnoredWarning`, a kind of user warning that applications can filter or escalate.

- A Responses class emits it when the app set a setting that has no counterpart on the Responses protocol, in any chat operation including tool conversations. The setting is not sent, and the call goes ahead.
- The library chose a warning rather than logging, so the signal shows up without any logging set up by the app. The mechanism and the once-per-setting-per-client rule are in DEC-arch-6; the name and emission rule in DEC-003-5, DEC-003-6.

## Exposed: error contract

Rule: unless the table below names another error, a failure of a library operation reaches the caller as `LLMError` (`message`, optional `description`; text form is `message: description`). Error text carries the request address and the full server reply, and is recorded in the trace (accepted). There are no retries.

| Failure kind | Intended | Today |
|---|---|---|
| Non-2xx reply, non-streaming | `LLMError` | as intended |
| Non-2xx reply, streaming chat and streaming tool rounds (both protocols) | `LLMError` carrying the request address and the server's reply, never the credential (replaces the earlier accepted `aiohttp.ClientResponseError`, DEC-arch-5) | as intended (DEC-001-8) |
| Failure reported inside a Responses stream, chat or tool round | `LLMError` carrying the server's reason | as intended |
| Responses reply, streamed or not, chat or tool round, that reports failure or incompleteness | `LLMError` carrying the server's reason. One exception: a reply the server stops at the output limit the app set is not a failure. In chat-style calls it returns its text. In a tool conversation such a round is the answer when it requests no tool; when it requests a tool the call ends with `LLMError` and that round's tools are not run, because the call's arguments may be cut (DEC-arch-13) | as intended. On llama-swap a reply cut at a limit the app did not set is returned as text, because the server reports it complete (accepted gap, see `quick-reference.md`, Open) |
| Responses stream that ends without its completion event | `LLMError` | as intended |
| Transport failure (server unreachable, timeout, transfer cut mid-reply) | `LLMError` carrying the request address and the failure kind | as intended on every backend, streamed or not (DEC-001-7) |
| 2xx reply that is not JSON | `LLMError` on model listing and embeddings; on chat-style calls a generic error (accepted) | as intended on OpenAI and llama-swap (DEC-002-10); the deprecated Ollama client is unchanged |
| Reply with null content, non-streaming (streaming yields empty text) | empty text | as intended on OpenAI and llama-swap, tool conversations included (DEC-001-12); the deprecated Ollama client is unchanged |
| Reply missing the expected choice or message | generic error (accepted) | `KeyError` / `IndexError` / `TypeError` |
| In-stream error piece, or stream that ends without a completion signal (chat completions) | `LLMError`; an error piece carries the server's reason | as intended (DEC-001-1, DEC-001-9) |
| Tool offered without a callable | `ValueError`, before any request | as intended |
| Tool requested that was not offered | `LLMError` naming the tool; the tool is not run | as intended on OpenAI and llama-swap (DEC-002-3); the deprecated Ollama client is unchanged |
| Tool arguments unreadable | `LLMError` naming the tool; the tool is not run | as intended on OpenAI and llama-swap (DEC-002-4); the deprecated Ollama client is unchanged |
| Tool argument name not in the callable's parameters | `ValueError` | as intended |
| Tool raised | `RuntimeError` naming the tool | as intended |
| Tool result not serialisable | `LLMError` naming the tool | as intended on OpenAI and llama-swap (DEC-002-5); the deprecated Ollama client is unchanged |
| Round limit exhausted | error that states the round limit as the reason | as intended on OpenAI and llama-swap: `RuntimeError` stating the limit and its value (DEC-002-2); the deprecated Ollama client keeps its earlier text |
| Malformed model list, or entry without an id | `LLMError` | as intended on OpenAI and llama-swap (DEC-002-9); empty list on the deprecated Ollama client (accepted) |
| Malformed embeddings reply, entry without a vector, or count not matching the input | `LLMError` | as intended on OpenAI and llama-swap (DEC-002-8); the deprecated Ollama client is unchanged |
| Operation not supported by a backend (`get_models` on the deprecated Ollama client) | refused before the server is contacted | `NotImplementedError` |
| `stream=True` or `on_delta` on the deprecated Ollama client | refused before the server is contacted | `ValueError` / `TypeError` |
| Unsupported `server_type` | `ValueError` | as intended |
| Missing OpenAI credential (none from the caller) | `ValueError` at creation | as intended (DEC-001-6) |
| Receiver failure | ends the call and propagates unchanged (DEC-000-23) | as intended |

The tool rows hold on both OpenAI-family protocols.

Other rules:

- JSON-only answer (`chat_json`): the parse check runs only when an expected shape is given; on the deprecated Ollama client it always runs. The expected shape is not sent to the backend today (open, see Open). A failed parse check is `LLMError`. A JSON answer cut at the app's output limit keeps the check: if it does not parse, it fails like any invalid JSON answer.
- Embeddings: one vector per input text, in input order (met on OpenAI and llama-swap); an empty batch gives an empty result without a request. Only the empty batch remains a known defect: it is still sent to the server (see Open).

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
| `GET /models` | `list_models`, `get_models` (records under `data`); both protocols |
| `POST /responses` | all chat-style operations on the Responses classes, tool conversations included; the conversation is sent as input items; tool definitions are sent in the protocol's tool shape; tool calls and their results travel as items; `stream` selects SSE with typed events |
| `POST /chat/completions` | all chat-style operations on the chat-completions classes; options are top-level fields; optional `response_format` and `tools`; `stream` selects SSE |
| `POST /embeddings` | `embed_batch` (array input, optional `user`; no sampling options); both protocols |

**Responses stream format**: `data: {json}` frames, each a typed event. The type is read from the payload, so event-name lines need not be read. The events carry output text, reasoning, tool-call items and argument fragments, reasoning items to carry over, completion, and failure. Tool-call items and fragments are assembled into the round's tool calls; they are not delivered to the receiver. A failure event ends the call with `LLMError`, and so does a stream that ends without its completion event; the one exception is the cut at the app's own output limit (see error contract). The list of events and how each is handled are in DEC-arch-11. Fixed request fields (no storage, reasoning carry, reasoning summary) are in DEC-arch-7.

**SSE stream format, chat completions** (OpenAI and llama-swap): `data: {json}` frames terminated by `data: [DONE]`. Only the first choice's `delta` is read, with `content`, `reasoning_content` and `tool_calls`; reasoning is read only from the dedicated `reasoning_content` field. A piece carrying both reasoning and content delivers both, reasoning first.

- A frame that carries an error ends the call with `LLMError` carrying the server's reason.
- A stream is complete when the server signals completion in either of its usual ways: a finish reason on a choice, or the `[DONE]` frame. A stream that ends with neither ends the call with `LLMError`.
- Other unreadable frames (not JSON, or without choices and without an error) are skipped.
- Text is decoded incrementally, so a multi-byte character split between two reads is kept.
- Line terminators (line feed, carriage return plus line feed, bare carriage return) and multi-line events are honoured.
- Tool-call fragments are matched by their position. Fragments without a position are told apart by call identity and name, and a name repeated in every fragment is not duplicated. Calls are returned in order: positioned calls by position, then the others in order of appearance.
- One total time limit bounds a whole stream, per server round.

The decoding, terminator and time-limit rules above apply to both protocols, because they share the stream reader.

## Decision history

- 2026-11-07 — The schema library behind the tool-description helper becomes a declared runtime dependency, so the package loads after a normal install. See DEC-001-3, superseding DEC-000-43.
- 2026-11-07 — On chat completions, `chat_json` sends the app's response-format setting when set, otherwise the JSON-object format. See DEC-001-2, superseding DEC-000-36.
- 2026-11-07 — A non-2xx reply on a streamed request becomes `LLMError` with the server's reply on both protocols. This replaces the accepted `aiohttp.ClientResponseError` row. See DEC-arch-5, superseding DEC-000-10.
- 2026-11-07 — Factory `protocol` argument, protocol axis in the capability matrix, and the ignored-setting warning category added. See DEC-arch-1, DEC-arch-4, DEC-arch-6.
- 2026-11-08 — A Responses reply cut at the output limit the app set is returned, not an error; a cut tool round that requests a tool still ends with `LLMError` and its tools are not run. See DEC-arch-13.
- 2026-11-08 — The Responses protocol is built for every chat operation, tool conversations included; the interim refusal of tool conversations on the Responses classes is lifted. See DEC-004-19, superseding DEC-003-4.
