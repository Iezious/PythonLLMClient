# Subsystems

## Purpose

Components, their responsibilities and boundaries, and the call structure between them. Contracts are in `interfaces.md`; shapes in `data-model.md`.

## Base client (`LLMClient`)

- Holds the state shared by every backend client: model, base address (trailing `/` stripped), timeout, bearer token, default options, the HTTP session and the tracer.
- Provides the shared HTTP request contract: prefixes the base address, adds the bearer header only if a token exists, applies one total time limit per request, maps non-2xx to `LLMError`, returns decoded JSON for JSON responses.
- Provides SSE parsing and the option-merge routine used by every backend.
- Declares the full method set; backends override it. The only concrete operation is `embed`, built on `embed_batch`.

## Backend clients

- Each client class owns: the table of settings it supports, the alias table, backend-specific post-processing of settings, and the wire mapping for every operation.
- Today: `LlamaSwapAPIClient`, `OpenAIAPIClient` (both chat completions) and `OllamaAPIClient` (**deprecated**).
- Target shape (planned): five classes, each extending the base directly:
  - `OpenAIAPIClient` and `LlamaSwapAPIClient` on Responses.
  - `OpenAIChatCompletionsClient` and `LlamaSwapChatCompletionsClient` on chat completions (today's behaviour, renamed).
  - `OllamaAPIClient` (deprecated, unchanged).
- The four OpenAI-family classes are independent implementations. They have no shared Responses base and no protocol component, so the protocol logic exists twice and every fix lands in both copies. This is accepted because the user prefers self-contained classes (DEC-arch-2; it continues DEC-000-35).
- Ollama nests settings under a payload `options` object; the OpenAI family places them at the top level of the request.
- Within one protocol, OpenAI and llama-swap differ only in auth requirement, thinking control and default address.

## Chat protocols (planned)

The OpenAI family offers two chat protocols:

- **Responses** — the default.
- **Chat completions** — for servers that do not offer Responses.

A client's protocol is fixed by its class, and the factory picks the class (see `interfaces.md`, Exposed: factory).

- Every chat operation follows the class's protocol, streamed or not: `generate`, `chat`, `chat_json` and `chat_with_tools`.
- Model listing and embeddings are the same on both protocols and do not change.
- App-facing shapes are the same on both protocols: the message shape, the tool-definition shape, the `response_format` shape, the `on_delta` receiver contract, the `<think>` convention in streamed text and the returned values. A Responses class translates these to and from the wire, so an app can switch protocol without changing its calls.
- A Responses class keeps no server-side conversation state. Each call sends the conversation it needs, and the request asks the provider not to store it.

## Options layer

Each client merges settings at call time in this order, lowest to highest: backend built-in values, constructor `default_options`, call-time `options`.

- Keys are resolved through the class's alias table.
- Keys the class does not support are silently dropped.
- Empty (`None`) values are removed last.

Post-processing per class:

- OpenAI translates `enable_thinking` to `reasoning_effort`.
- llama-swap drops `reasoning_effort`.
- Ollama omits the mirostat tuning keys while mirostat is off.

Key lists are in `data-model.md`.

Target shape (planned):

- Every class has its own supported-settings table. The chat-completions classes keep today's tables and post-processing.
- The Responses classes have no built-in values. They send only the settings the app set, and translate each one to its Responses counterpart.
- A setting that is supported but has no Responses counterpart is not sent, and the class emits the library's ignored-setting warning (see `interfaces.md`, Exposed: warnings). An unsupported key is still dropped silently, as today.

Mapping details are in DEC-arch-8.

## Call structure of an operation

1. Merge settings.
2. Prepend the system instruction (Ollama `generate` sends it as a separate field).
3. Build the payload for the backend.
4. Open a trace session and record system prompt, messages and request.
5. Send the request; a streaming reply is read as SSE.
6. Parse the reply and record the response.
7. Return the text.

`chat_json` validates that the reply parses as JSON after the trace session has closed, so a validation failure is not recorded in the trace.

## Tool loop

Per `chat_with_tools` call:

1. Pre-flight: every advertised tool has a callable, else `ValueError` before any request.
2. Round: build payload (model, conversation, tool definitions, stream flag, settings, optional `response_format`) and send.
3. No tool calls in the reply: return the assistant content.
4. Tool calls present: record the assistant turn (not on Ollama), run each tool, add one tool-result message per call, start the next round.
5. Round limit reached: the call fails (rules in `interfaces.md`, Tool loop contract).

Reasoning carry on Responses (planned): when a round produces reasoning, it is sent back to the server on the next round of the same call, ahead of that round's tool calls. That way a reasoning model keeps its line of thought across tool results. The carried reasoning never enters the caller's conversation (which is never modified) or the returned answer, and it is discarded when the call ends. Chat completions carries no reasoning between rounds, as today. Details are in DEC-arch-9.

Arguments are validated against the tool's parameter names only. Ordering, failure and round-limit rules are in `interfaces.md`, Tool loop contract; failure kinds and their mapping are in `interfaces.md`, error contract.

## Streaming and reasoning

- Streaming reads SSE deltas: content, reasoning and tool-call fragments. This is the chat-completions stream.
- Responses stream (planned): typed events feed the same receiver contract. Text events become text, reasoning events become reasoning inside think markers, and tool-call events become tool-call fragments. Reasoning arrives separately from text on this protocol, so the chat-completions loss of content when one piece carries both does not occur. A failure event inside the stream ends the call with `LLMError` carrying the server's reason. Event handling is in DEC-arch-11.
- Reasoning tagging: only the dedicated reasoning field is wrapped in think markers. The first reasoning piece of a stream is prefixed with `<think>`, the first content piece after reasoning is prefixed with `</think>`, and a block still open at the end is closed. Inline reasoning written by the server into content passes through as text, without wrapping (accepted). The tool path returns no reasoning, and produces one closed block per round in what the receiver sees.
- The receiver contract (order, completion before the next read, failure propagation) is in `interfaces.md`, Streaming callback.
- Two parse modes with deliberately different returns:
  - Plain streaming chat returns the reasoning (tagged) together with the answer; the returned text equals what the callback received.
  - Streaming tool chat sends reasoning to the callback but keeps it out of the returned and history content. Reason stated in code: the content is added to the history that is re-sent to the model, and re-sending reasoning wastes context or is rejected by some servers.
- Non-streaming replies return only the message content; reasoning is ignored, so no `<think>` tagging off-stream.
- The deprecated Ollama client has no streaming.

## Tracer (`LLMTracer`)

- A debugging aid, not an audit record: plain text, not redacted, best-effort.
- Opt-in by environment: enabled when `LLM_TRACE` is true-like and `LLM_TRACE_FOLDER` is non-empty. Read once when a client is constructed.
- One trace record per traced operation (`generate`, `chat`, `chat_json`, `chat_with_tools`, `embed_batch`), including concurrent calls. Known defect: concurrent calls can collide on one record (see `quick-reference.md`, Open).
- Tracing must never change a call's outcome. Known defect: an unwritable trace folder fails the call (see Open).
- Covered: plain streamed chat pieces (not tool-path streams); tool requests and results; errors raised inside the traced part, which are recorded and re-raised. Not covered: round-limit exhaustion, the JSON-check failure, model listing.
- The credential header is never written. Prompts, completions, tool arguments and results are recorded.
- The tracer is a supported public surface: an application may open a trace session with any operation label and log to it.
- Record layout is in `data-model.md`.

## Utilities

`substitute_placeholders`, the repeated-fill template helper and `pydantic_to_openai_tool` are stateless helpers with no dependency on the clients; the factory `get_llm_client` selects and constructs a backend client. Contracts are in `interfaces.md`.
