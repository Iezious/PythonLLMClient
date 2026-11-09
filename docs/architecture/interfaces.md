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
| `chat_with_tools(messages, *, tools_definitions, tools, system, max_loops, options, stream, on_delta, response_format, tool_errors, on_tool_event)` | Runs the tool loop and returns the final answer text. No transcript is returned and the caller's `messages` is not changed. `tool_errors` (`"result"`, the default, or `"raise"`) and `on_tool_event` exist on the four OpenAI-family classes only (see Tool failures, Tool event observer). |
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
- Tool arguments are accepted as a JSON string or a mapping; every argument name must exist in the callable's parameters, except that a callable which accepts arbitrary keyword arguments skips this check. Required-ness and types are not checked.
- Tools requested in one reply run sequentially, in the order requested. Extra callables beyond the advertised tools are allowed.
- A tool result that is a mapping or list is serialised to JSON text; anything else is converted to text.
- A failing tool call is handled per `tool_errors` (see Tool failures below).
- `max_loops` bounds the number of server rounds, and the round that produces the answer counts. Tools requested in the last permitted round still run; if the model asks for more, the call fails (see error contract). This holds in both failure modes.
- `response_format`, when given, is sent on every round and is not validated against the reply.

These rules hold on both OpenAI-family protocols.

### Tool failures

A tool call fails in one of five kinds. Each has a public name (`ToolErrorKind`, see `data-model.md`, Tool events):

| Kind | Meaning |
|---|---|
| `not_offered` | the model requested a tool that was not offered, or gave an empty name |
| `bad_arguments` | the arguments cannot be read, or are not an object |
| `unknown_argument` | an argument name the callable does not declare (not checked for a callable that accepts arbitrary keyword arguments) |
| `tool_raised` | the tool raised |
| `unserialisable_result` | the tool's result cannot be turned into text |

`tool_errors` selects the handling, on all four OpenAI-family classes and both protocols. Its public signature is `tool_errors: Literal["raise", "result"] = "result"`.

- **`"result"` (default, tolerant)** — the failing call's tool is not run, or its failure is caught. A tool-result for that call (the protocol's function-call output, or a tool message) carrying a short fixed model-facing text is added to the conversation, and the loop continues. Other calls in the same round still run, in order. The text never carries exception text or a server body (texts in DEC-arch-14).
- **`"raise"` (strict, opt-out)** — the earlier behaviour, unchanged: the call ends with the exception that kind has always raised (see error contract). The model is not told.

In both modes only ordinary exceptions are handled: cancellation, and any other failure that is not an ordinary exception, propagates unchanged.

Tolerant is the default because the user wants tolerant behaviour. The cost is accepted: an existing caller of the four OpenAI-family classes that relied on a tool failure ending the call now has the failure sent to the model instead, unless it passes `tool_errors="raise"`. Strict stays available for callers that need the call to end. The deprecated Ollama client does not take the parameter and keeps its raising behaviour (DEC-arch-28).

### Tool event observer

`on_tool_event` is optional. It receives one `ToolEvent` (see `data-model.md`, Tool events) per event, and may be a plain function or a coroutine function. It is called inline, like the stream receiver: the loop goes on only after the observer has finished with the event (DEC-arch-18).

- Events fire in call order. For each tool call: a `start` event before the tool runs, then exactly one `result` event or `error` event. This holds for every failure kind, including `not_offered` and `bad_arguments`, where the tool never runs. The one exception is a tolerant-mode observer failure (below).
- In strict mode (`tool_errors="raise"`) the `error` event fires before the call ends with its exception (DEC-arch-17).
- An observer failure in strict mode ends the call and propagates unchanged, like a receiver failure.
- An observer failure in tolerant mode (the default) is handled like a tool failure. The model is told the tool failed, a tool not yet run is not run, and the loop continues. The library emits `LLMToolObserverWarning` (see Exposed: warnings) and records the failure in the trace.
- After an observer failure in tolerant mode, no further event of that tool call is sent to the observer. The failure is reported only through the warning and the trace, and `ToolErrorKind` keeps its five tool failure kinds. An extra `error` event with an observer kind was rejected: it would go to the observer that has just failed, and the failure kinds stay about tools (DEC-arch-26).

## Exposed: capability matrix

| Capability | Ollama (**deprecated**) | OpenAI | llama-swap |
|---|---|---|---|
| Auth | optional caller-provided bearer | caller-provided bearer required | optional caller-provided bearer |
| Streaming in `chat` | not supported | yes | yes |
| Streaming in `chat_with_tools` | not supported | yes | yes |
| `on_delta` parameter | not accepted | yes | yes |
| `tool_errors` parameter (tool chat) | not accepted | yes | yes |
| `on_tool_event` parameter (tool chat) | not accepted | yes | yes |
| `read_timeout` constructor argument (silence limit on streamed replies) | not accepted | yes | yes |
| `trace` constructor argument | accepted, passed to the base only | yes | yes |
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

On the deprecated Ollama client, `stream=True` raises `ValueError`. Passing `on_delta`, `tool_errors` or `on_tool_event` raises `TypeError`, because the client does not declare them, and so does `read_timeout` at its constructor (DEC-arch-24). The client is not extended.

## Exposed: constructor arguments

Every client class takes `model`, `base_url`, `timeout`, `bearer_token`, `default_options` and `trace`. The four OpenAI-family classes also take `read_timeout`.

- `timeout` — one total time limit per request, streamed or not (unchanged).
- `read_timeout` — optional. A limit on silence between reads of a streamed reply; when it is exceeded the call fails as a transport timeout. Not passed means no silence limit. It does not change the total limit, and non-streamed requests are not affected (DEC-arch-20).
- `trace` — optional; three states:
  - not passed (`None`): the environment decides, as today (`LLM_TRACE` and `LLM_TRACE_FOLDER`, read once at creation).
  - `False`: never trace, whatever the environment says.
  - `True`: trace, into the folder named by `LLM_TRACE_FOLDER`. `True` without that folder raises `ValueError` at creation.

## Exposed: factory

`get_llm_client(server_type, base_url, model, timeout, bearer_token, default_options, protocol, read_timeout, trace)` returns an `LLMClient[Any]` (see Exposed: typing contract).

- `read_timeout` and `trace` are forwarded to the class the factory picks. `read_timeout` with `ollama` raises `ValueError`, because Ollama has no streaming (DEC-arch-24). `trace` is forwarded to every backend.

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

A second category, `LLMToolObserverWarning`, is also a kind of user warning and is exported at the top level.

- An OpenAI-family class emits it when the tool-event observer fails during a tool conversation in tolerant mode (`tool_errors="result"`, the default). The call goes ahead (see Tool event observer).
- In strict mode (`tool_errors="raise"`) the observer's failure propagates, and no warning is emitted.

## Exposed: typing contract

The public surface type-checks under a strict static checker. The package already ships its typing marker. The checker and how it is configured are in DEC-arch-22.

- The base is generic over its settings shape: `LLMClient[OptionsT]`. Each client class binds its own settings shape, so `options` and `default_options` are typed per class.
- The factory returns `LLMClient[Any]`, because the class it picks is chosen at run time.
- `chat_with_tools` takes `messages` as a read-only sequence of mappings from string keys to any value (`Sequence[Mapping[str, Any]]`), so turns carrying tool calls type-check.
- Entering a usage scope returns the client's own type (`Self`), and leaving it is fully annotated.
- The deprecated Ollama client is outside the strict check.

## Exposed: error contract

Rule: unless the table below names another error, a failure of a library operation reaches the caller as `LLMError`. Error text carries the request address and the full server reply, and is recorded in the trace (accepted). There are no retries.

`LLMError` carries:

- `message` and optional `description`; the text form is `message: description`, unchanged.
- `status` (keyword-only, optional) — the HTTP status when the failure is a non-2xx reply, otherwise none.
- `kind` (keyword-only, optional) — one value of `LLMErrorKind` (see `data-model.md`, Error kinds), saying which failure this is. It is given in the Kind column below; the site-by-site mapping is DEC-arch-19.

Kinds for non-2xx and transport failures come from the base's request plumbing, so they hold on every backend. The other kinds are set by the OpenAI-family classes; the deprecated Ollama client's own errors are not extended. Existing `RuntimeError` and `ValueError` raises are unchanged and carry no kind. **Status (2026-11-09):** `status`, `kind`, the tolerant tool mode, the observer, the silence limit and the `trace` argument are agreed and not yet built; "Today" describes the code before them.

| Failure kind | Intended | Kind | Today |
|---|---|---|---|
| Non-2xx reply, non-streaming | `LLMError` with `status` | `status` | as intended, without `status` / `kind` |
| Non-2xx reply, streaming chat and streaming tool rounds (both protocols) | `LLMError` carrying the request address and the server's reply, never the credential (replaces the earlier accepted `aiohttp.ClientResponseError`, DEC-arch-5), with `status` | `status` | as intended (DEC-001-8), without `status` / `kind` |
| Failure reported inside a Responses stream, chat or tool round | `LLMError` carrying the server's reason | `server_failed` | as intended, without `kind` |
| Responses reply, streamed or not, chat or tool round, that reports failure or incompleteness | `LLMError` carrying the server's reason. One exception: a reply the server stops at the output limit the app set is not a failure. In chat-style calls it returns its text. In a tool conversation such a round is the answer when it requests no tool; when it requests a tool the call ends with `LLMError` and that round's tools are not run, because the call's arguments may be cut (DEC-arch-13) | `server_failed` for a failed reply; `incomplete` for an incomplete one, the cut tool round included | as intended, without `kind`. On llama-swap a reply cut at a limit the app did not set is returned as text, because the server reports it complete (accepted gap, see `quick-reference.md`, Open) |
| Responses stream that ends without its completion event | `LLMError` | `incomplete` | as intended, without `kind` |
| Responses reply that is not a mapping | `LLMError` | `invalid_reply` | as intended, without `kind` |
| Transport failure (server unreachable, total time limit or silence limit exceeded, transfer cut mid-reply) | `LLMError` carrying the request address and the failure kind | `connection` / `timeout` / `interrupted` | as intended on every backend, streamed or not (DEC-001-7), without `kind`; no silence limit yet |
| 2xx reply that is not JSON | `LLMError` on model listing and embeddings; on chat-style calls a generic error (accepted) | `invalid_reply` (listing, embeddings) | as intended on OpenAI and llama-swap (DEC-002-10), without `kind`; the deprecated Ollama client is unchanged |
| Reply with null content, non-streaming (streaming yields empty text) | empty text | — | as intended on OpenAI and llama-swap, tool conversations included (DEC-001-12); the deprecated Ollama client is unchanged |
| Reply missing the expected choice or message (chat completions) | generic error (accepted) | — | `KeyError` / `IndexError` / `TypeError` |
| In-stream error piece, or stream that ends without a completion signal (chat completions) | `LLMError`; an error piece carries the server's reason | `server_failed` (error piece) / `incomplete` (no completion signal) | as intended (DEC-001-1, DEC-001-9), without `kind` |
| Tool offered without a callable | `ValueError`, before any request | — | as intended |
| Tool requested that was not offered, or empty name (`not_offered`) | tolerant (default): the model is told, the loop continues. Strict: `LLMError` naming the tool; the tool is not run | `tool` (strict) | every call is strict today, as intended for strict on OpenAI and llama-swap (DEC-002-3), without `kind`; the deprecated Ollama client is unchanged; the tolerant default is not built |
| Tool arguments unreadable or not an object (`bad_arguments`) | tolerant (default): the model is told, the loop continues. Strict: `LLMError` naming the tool; the tool is not run | `tool` (strict) | every call is strict today, as intended for strict on OpenAI and llama-swap (DEC-002-4), without `kind`; the deprecated Ollama client is unchanged; the tolerant default is not built |
| Tool argument name not declared by the callable (`unknown_argument`; not checked when the callable accepts arbitrary keyword arguments) | tolerant (default): the tool is not run, the model is told which argument, the loop continues. Strict: `ValueError` | — | every call is strict today, as intended for strict, except that a callable accepting arbitrary keyword arguments is rejected today; the tolerant default is not built |
| Tool raised an ordinary exception (`tool_raised`) | tolerant (default): the failure is caught, the model is told, the loop continues. Strict: `RuntimeError` naming the tool. Cancellation propagates in both modes | — | every call is strict today, as intended for strict; the tolerant default is not built |
| Tool result not serialisable (`unserialisable_result`) | tolerant (default): the model is told, the loop continues. Strict: `LLMError` naming the tool | `tool` (strict) | every call is strict today, as intended for strict on OpenAI and llama-swap (DEC-002-5), without `kind`; the deprecated Ollama client is unchanged; the tolerant default is not built |
| Tool-event observer failure | tolerant (default): the model is told the tool failed, a tool not yet run is not run, the loop continues, `LLMToolObserverWarning` is emitted and the failure is traced; the observer receives no further event of that tool call. Strict: ends the call and propagates unchanged | — | not built |
| Round limit exhausted | error that states the round limit as the reason, in both tool-failure modes | — | as intended on OpenAI and llama-swap: `RuntimeError` stating the limit and its value (DEC-002-2); the deprecated Ollama client keeps its earlier text |
| Malformed model list, or entry without an id | `LLMError` | `invalid_reply` | as intended on OpenAI and llama-swap (DEC-002-9), without `kind`; empty list on the deprecated Ollama client (accepted) |
| Malformed embeddings reply, entry without a vector, count not matching the input, or empty embedding result | `LLMError` | `invalid_reply` | as intended on OpenAI and llama-swap (DEC-002-8), without `kind`; the deprecated Ollama client is unchanged |
| Operation not supported by a backend (`get_models` on the deprecated Ollama client) | refused before the server is contacted | — | `NotImplementedError` |
| `stream=True`, `on_delta`, `tool_errors` or `on_tool_event` on the deprecated Ollama client | refused before the server is contacted | — | `ValueError` / `TypeError`; the two tool parameters are refused as `TypeError` once they exist on the other classes |
| `read_timeout` with `server_type="ollama"` at the factory | `ValueError` | — | not built |
| `trace=True` with no trace folder in the environment | `ValueError` at creation | — | not built |
| Unsupported `server_type` | `ValueError` | — | as intended |
| Missing OpenAI credential (none from the caller) | `ValueError` at creation | — | as intended (DEC-001-6) |
| Receiver failure | ends the call and propagates unchanged (DEC-000-23) | — | as intended |

The tool rows hold on both OpenAI-family protocols. "Tolerant" is `tool_errors="result"` (the default); "strict" is `tool_errors="raise"`, which the caller passes to opt out (see Tool failures). The deprecated Ollama client has no `tool_errors` and keeps raising.

Other rules:

- JSON-only answer (`chat_json`): the parse check runs only when an expected shape is given; on the deprecated Ollama client it always runs. The expected shape is not sent to the backend today (open, see Open). A failed parse check is `LLMError` of kind `invalid_reply`. A JSON answer cut at the app's output limit keeps the check: if it does not parse, it fails like any invalid JSON answer.
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
- One total time limit bounds a whole stream, per server round. When the app sets `read_timeout`, a silence limit also bounds the gap between two reads of the stream. Either one exceeded fails the call as a transport timeout.

The decoding, terminator and time-limit rules above apply to both protocols, because they share the stream reader and the streamed-request plumbing.

## Decision history

- 2026-11-07 — The schema library behind the tool-description helper becomes a declared runtime dependency, so the package loads after a normal install. See DEC-001-3, superseding DEC-000-43.
- 2026-11-07 — On chat completions, `chat_json` sends the app's response-format setting when set, otherwise the JSON-object format. See DEC-001-2, superseding DEC-000-36.
- 2026-11-07 — A non-2xx reply on a streamed request becomes `LLMError` with the server's reply on both protocols. This replaces the accepted `aiohttp.ClientResponseError` row. See DEC-arch-5, superseding DEC-000-10.
- 2026-11-07 — Factory `protocol` argument, protocol axis in the capability matrix, and the ignored-setting warning category added. See DEC-arch-1, DEC-arch-4, DEC-arch-6.
- 2026-11-08 — A Responses reply cut at the output limit the app set is returned, not an error; a cut tool round that requests a tool still ends with `LLMError` and its tools are not run. See DEC-arch-13.
- 2026-11-08 — The Responses protocol is built for every chat operation, tool conversations included; the interim refusal of tool conversations on the Responses classes is lifted. See DEC-004-19, superseding DEC-003-4.
- 2026-11-09 (release 0.2.1) — The tool loop gains a strict/tolerant failure mode (`tool_errors`) and a tool-event observer (`on_tool_event`) on the four OpenAI-family classes. The five tool failure kinds are named, a callable that accepts arbitrary keyword arguments skips the argument-name check, and the observer warning category is added. The deprecated Ollama client refuses both parameters. See DEC-arch-14, DEC-arch-16, DEC-arch-17, DEC-arch-18, DEC-arch-24.
- 2026-11-09 (release 0.2.1) — `LLMError` gains `status` and `kind`, with the text form unchanged. See DEC-arch-19.
- 2026-11-09 (release 0.2.1) — Constructor `read_timeout` (silence limit on streamed replies, OpenAI family; refused with Ollama at the factory) and `trace` (explicit on/off over the environment, all classes). See DEC-arch-20, DEC-arch-24, DEC-arch-27 (amending DEC-000-4).
- 2026-11-09 (release 0.2.1) — A tolerant-mode observer failure sends no further event of that tool call to the observer; `ToolErrorKind` stays at five kinds. See DEC-arch-26.
- 2026-11-09 (release 0.2.1) — Typing contract: generic base per settings shape, factory returns `LLMClient[Any]`, tool-chat messages as a sequence of mappings, typed usage scope; checked strictly. See DEC-arch-22. Version: DEC-arch-23, superseding DEC-001-4.
- 2026-11-09 (release 0.2.1) — The default of `tool_errors` on the four OpenAI-family classes flips from `"raise"` to `"result"`: tolerant is the default, strict is the opt-out. Existing callers now have tool-call failures sent to the model unless they pass `"raise"`; the Ollama client is unchanged. See DEC-arch-28 (amending DEC-arch-14).
