# Subsystems

## Purpose

Components, their responsibilities and boundaries, and the call structure between them. Contracts are in `interfaces.md`; shapes in `data-model.md`.

## Base client (`LLMClient`)

**Rule: no shared behaviour logic between client classes.** The base holds transport plumbing only: requests, streams, transport-error mapping, time limits and the tracer. Each client class holds its own copy of its behaviour, the tool loop included. The specification in these documents is what is maintained, so a class's code may be rewritten on its own without touching another class. The existing shared chat-completions stream parsers stay in the base, but new behaviour is not added to them. Behaviour that a feature adds goes into each class that needs it, as its own copy.

We chose this because the user wants the classes to stay independent and flexible: "NO MIXED logic. We need to stay flexible. We are not supporting code. We do support specifications and rewrite code if needed." The cost is accepted: a behaviour change lands in every class that carries it. This rule restates the earlier per-class choices as architecture (DEC-arch-15; earlier DEC-000-35, DEC-arch-2).

The option-merge routine also sits in the base and is behaviour, not transport. It is treated like the shared stream parsers: it stays in the base with its current behaviour and is not extended. A class whose merging must diverge takes its own copy and changes that, leaving the base routine as it is. We chose this over moving the merge into every class now, because that move would rewrite every class without changing any behaviour, while copying on divergence already keeps each class free to change on its own (DEC-arch-25).

- Holds the state shared by every backend client: model, base address (trailing `/` stripped), timeout, silence limit (OpenAI family), bearer token, default options, the HTTP session and the tracer.
- Provides the shared HTTP request contract, for plain and streamed requests alike: prefixes the base address, and adds the bearer header only if a token exists. It applies one total time limit per request and, on streamed requests, the optional silence limit between reads. It maps non-2xx to `LLMError` with the HTTP status (a refused stream carries the server's reply), and maps transport failures (unreachable server, timeout, interrupted transfer) to `LLMError` carrying the address and the matching error kind. For a plain request it returns decoded JSON for JSON responses; for a streamed request it opens the stream and hands the open reply to the caller, and read failures while the stream is consumed are mapped the same way. Every protocol class opens its streams through this contract.
- Provides the SSE reader, the existing chat-completions stream parsers and the option-merge routine (shared as they are, not extended; see the rule above).
- Builds the tracer once per client from the constructor's `trace` argument, or from the environment when it is not passed (see Tracer).
- Declares the full method set; backends override it. The only concrete operation is `embed`, built on `embed_batch`.
- Is generic over the settings shape; each class binds its own (see `interfaces.md`, Exposed: typing contract).

## Backend clients

- Each client class owns: the table of settings it supports, the alias table, backend-specific post-processing of settings, and the wire mapping for every operation.
- Five classes, each extending the base directly:
  - `OpenAIAPIClient` and `LlamaSwapAPIClient` on Responses.
  - `OpenAIChatCompletionsClient` and `LlamaSwapChatCompletionsClient` on chat completions (the earlier behaviour of the two names above, renamed).
  - `OllamaAPIClient` (deprecated, unchanged).
- The four OpenAI-family classes are independent implementations, per the rule under Base client. They have no shared Responses base, no protocol component and no shared tool execution step, so the protocol logic and the tool loop exist once per class and every fix lands in each copy.
- Ollama nests settings under a payload `options` object; the OpenAI family places them at the top level of the request.
- Within one protocol, OpenAI and llama-swap differ only in auth requirement, thinking control and default address.

## Chat protocols

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

Every class has its own supported-settings table. Key lists are in `data-model.md`.

The OpenAI-family classes, on both protocols, have no built-in values: only settings the app set are sent. Their tables include stop sequences, token bias, the end-user identifier and the response format; on chat completions these reach the server like any other setting, and on Responses a key with no counterpart is handled as below. The deprecated Ollama client keeps two known defects, not fixed because it is observation-only: it still sends its built-in values, and it still drops some keys declared in its typed settings shape (see `quick-reference.md`, Open).

Post-processing on the chat-completions classes and Ollama:

- OpenAI translates `enable_thinking` to `reasoning_effort`.
- llama-swap drops `reasoning_effort`.
- Ollama omits the mirostat tuning keys while mirostat is off.

Post-processing on the Responses classes:

- They send only the settings the app set, and translate each one to its Responses counterpart.
- Thinking on/off and an explicit effort reach both backends through the protocol's reasoning setting; only OpenAI also asks for the reasoning summary.
- A setting that is supported but has no Responses counterpart is not sent, and the class emits the library's ignored-setting warning (see `interfaces.md`, Exposed: warnings). An unsupported key is still dropped silently.

Mapping details are in DEC-arch-8 and DEC-003-9.

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
4. Tool calls present: record the assistant turn (not on Ollama). Then, for each call in order: tell the observer the call starts, check and run the tool, tell the observer the result or the error, and add one tool-result message for the call. A failing call either gets a fixed model-facing text as its result (tolerant, the default) or ends the call (strict, when the caller asks for it). Start the next round.
5. Round limit reached: the call fails (rules in `interfaces.md`, Tool loop contract).

The same loop runs on both OpenAI-family protocols. Each class holds its own copy, the failure handling and the observer included (rule under Base client; DEC-arch-15). The deprecated Ollama loop has neither.

Reasoning carry on Responses: when a round produces reasoning, it is sent back to the server on the next round of the same call, ahead of that round's tool calls. That way a reasoning model keeps its line of thought across tool results. The carried reasoning never enters the caller's conversation (which is never modified) or the returned answer, and it is discarded when the call ends. Chat completions carries no reasoning between rounds. Details are in DEC-arch-9.

Arguments are validated against the tool's parameter names only, and not at all for a callable that accepts arbitrary keyword arguments. Ordering, failure and round-limit rules are in `interfaces.md`, Tool loop contract, Tool failures and Tool event observer; failure kinds and their mapping, including a round cut at the app's output limit, are in `interfaces.md`, error contract.

## Streaming and reasoning

- Streaming reads SSE deltas: content, reasoning and tool-call fragments. This is the chat-completions stream. A piece carrying both reasoning and content delivers both to the receiver, reasoning first. A failure the server reports inside the stream ends the call with `LLMError` carrying the server's reason, and so does a stream that ends without any completion signal.
- Responses stream: typed events feed the same receiver contract. Text events become text, reasoning events become reasoning inside think markers, and tool-call events are assembled into the round's tool calls. Reasoning arrives separately from text on this protocol. A failure event inside the stream, or a stream that ends without its completion event, ends the call with `LLMError` carrying the server's reason (exception: the cut at the app's own output limit, see `interfaces.md`, error contract). Event handling is in DEC-arch-11.
- Tool-call fragments are assembled into calls and are never delivered to the receiver, on either protocol.
- Reasoning tagging: only the dedicated reasoning field is wrapped in think markers. The first reasoning piece of a stream is prefixed with `<think>`, the first content piece after reasoning is prefixed with `</think>`, and a block still open at the end is closed. Inline reasoning written by the server into content passes through as text, without wrapping (accepted). The tool path returns no reasoning, and produces one closed block per round in what the receiver sees.
- The receiver contract (order, completion before the next read, failure propagation) is in `interfaces.md`, Streaming callback.
- Two parse modes with deliberately different returns:
  - Plain streaming chat returns the reasoning (tagged) together with the answer; the returned text equals what the callback received.
  - Streaming tool chat sends reasoning to the callback but keeps it out of the returned and history content. Reason stated in code: the content is added to the history that is re-sent to the model, and re-sending reasoning wastes context or is rejected by some servers.
- Non-streaming replies return only the message content; reasoning is ignored, so no `<think>` tagging off-stream.
- The deprecated Ollama client has no streaming.

## Tracer (`LLMTracer`)

- A debugging aid, not an audit record: plain text, not redacted, best-effort.
- Opt-in, decided once when a client is constructed. The constructor's `trace` argument decides when given: `False` never traces, and `True` traces into `LLM_TRACE_FOLDER`, refusing creation when that folder is not set. When `trace` is not passed, the environment decides as before: enabled when `LLM_TRACE` is true-like and `LLM_TRACE_FOLDER` is non-empty (see `interfaces.md`, Exposed: constructor arguments).
- One trace record per traced operation (`generate`, `chat`, `chat_json`, `chat_with_tools`, `embed_batch`), including concurrent calls that start at the same instant: each gets its own record (naming in `data-model.md`, Trace record).
- Tracing must never change a call's outcome:
  - A record that cannot be created (for example, an unwritable folder) leaves the call untraced; nothing is reported.
  - A failure while writing a record ends that record, never the call; the call's own result or error reaches the caller unchanged, and the record file is always released.
- Covered: plain streamed chat pieces (not tool-path streams); tool requests and results; tool failures handled in tolerant mode, recorded as what the model was told plus the failure kind; observer failures in tolerant mode; errors raised inside the traced part, which are recorded and re-raised. Not covered: round-limit exhaustion, the JSON-check failure, model listing.
- The credential header is never written. Prompts, completions, tool arguments and results are recorded.
- The tracer is a supported public surface: an application may open a trace session with any operation label and log to it.
- Record layout is in `data-model.md`.

## Utilities

`substitute_placeholders`, the repeated-fill template helper and `pydantic_to_openai_tool` are stateless helpers with no dependency on the clients; the factory `get_llm_client` selects and constructs a backend client. Contracts are in `interfaces.md`.

## Decision history

- 2026-11-09 (release 0.2.1) — "No shared behaviour logic between client classes" becomes an architecture rule. The base holds transport plumbing only, and each class holds its own copy of its behaviour, the tool loop included. See DEC-arch-15 (restating DEC-000-35, DEC-arch-2).
- 2026-11-09 (release 0.2.1) — The tool loop gains tolerant failure handling and a tool-event observer, in each OpenAI-family class's own copy; the base gains the streamed-reply silence limit and the error kinds in its transport mapping; the tracer can be switched on or off by the constructor. See DEC-arch-14, DEC-arch-19, DEC-arch-20, DEC-arch-21, DEC-arch-27.
- 2026-11-09 (release 0.2.1) — The option-merge routine stays in the base like the shared stream parsers: not extended, copied into a class that needs to diverge. See DEC-arch-25.
