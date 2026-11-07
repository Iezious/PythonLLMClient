# Quick reference

Index only. Details live in the linked files.

## Public exports (`llm`)

`__version__`, `LLMError`, `LLMClient`, `OllamaAPIClient` (**deprecated**), `OpenAIAPIClient`, `LlamaSwapAPIClient`, `get_llm_client`, `substitute_placeholders`, `pydantic_to_openai_tool`, `LLMTracer`. See `interfaces.md`, Exposed: utilities. The supported surface is the package top level. Pending top-level export: typed settings shapes and the message shape (DEC-000-18), the repeated-fill template helper (DEC-000-51).

Planned exports:

- `OpenAIChatCompletionsClient` and `LlamaSwapChatCompletionsClient` (chat completions).
- The ignored-setting warning category. _TBD: its name is settled in planning._

`OpenAIAPIClient` and `LlamaSwapAPIClient` change meaning to Responses. See `overview.md` What it is, `interfaces.md` Exposed: warnings.

## Client classes and protocols (planned)

| Backend | Responses (default) | Chat completions (compatibility) | Factory `protocol` |
|---|---|---|---|
| OpenAI | `OpenAIAPIClient` | `OpenAIChatCompletionsClient` | `responses` / `chat_completions` |
| llama-swap | `LlamaSwapAPIClient` | `LlamaSwapChatCompletionsClient` | `responses` / `chat_completions` |
| Ollama (**deprecated**) | — | — | refused |

See `interfaces.md` Exposed: factory, Protocol axis.

## Environment variables

| Variable | Meaning | See |
|---|---|---|
| `LLM_TRACE` | true-like value (`1`, `true`, `on`) enables tracing; needs `LLM_TRACE_FOLDER` too | `subsystems.md` Tracer |
| `LLM_TRACE_FOLDER` | root folder for trace files | `data-model.md` Trace record |

The library reads no credential from the environment. All other configuration is constructor arguments: `model`, `base_url`, `timeout`, `bearer_token` (the caller-provided credential), `default_options`. The factory also takes `protocol` (planned). An application may read its own key from its environment and pass it in.

## Backend routes

| Backend | Routes | See |
|---|---|---|
| Ollama (**deprecated**) | `/api/tags`, `/api/generate`, `/api/chat`, `/api/embed`, `/api/embeddings` (fallback) | `interfaces.md` Consumed |
| OpenAI, llama-swap | `/models`, `/chat/completions`, `/embeddings` (under `/v1` base); planned: `/responses` for the Responses classes, `/chat/completions` for the chat-completions classes only | `interfaces.md` Consumed |

## Settings keys

| Backend | Keys | See |
|---|---|---|
| OpenAI, llama-swap | `temperature`, `top_p`, `max_tokens`, `presence_penalty`, `frequency_penalty`, `seed`, `enable_thinking`, `reasoning_effort` (planned: per-class tables; Responses sends only what the app set) | `data-model.md` Settings |
| Ollama (**deprecated**) | `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`, `num_predict`, `num_ctx`, `num_keep`, `mirostat`, `mirostat_tau`, `mirostat_eta`, `seed` | `data-model.md` Settings |

## Component index

| Component | See |
|---|---|
| Base client, backend clients, chat protocols, options layer | `subsystems.md` |
| Tool loop, streaming and reasoning, tracer | `subsystems.md` |
| Method contract, capability matrix (with protocol axis), factory, warnings, errors | `interfaces.md` |
| Message, tool definition, trace record | `data-model.md` |

## Open (architecture questions)

- Errors are not fully normalised. Planned: a streamed non-2xx reply raises `LLMError` with the server's reply on both protocols (DEC-arch-5; today it is `aiohttp.ClientResponseError`), and a failure inside a Responses stream raises `LLMError`. Still open: error pieces inside a chat-completions stream (DEC-000-13).
- Planned, open in planning: the thinking-off effort value per model on Responses, and how thinking on/off reaches llama-swap over Responses (DEC-arch-8).
- The deprecated Ollama tool loop does not keep the assistant tool-call turn in the conversation.
- The `context_length` alias means context size on Ollama but an output-token cap on OpenAI and llama-swap.
- The llama-swap client accepts the same settings as the OpenAI client; top-k, min-p and repeat-penalty are not accepted (dropped).
- Known defect: the OpenAI client reads the credential from the environment, and the factory drops the caller's credential (DEC-000-1, DEC-000-2).
- Known defect: built-in setting values are always sent (DEC-000-6).
- Known defect: settings declared in the typed shapes are dropped by the merge (DEC-000-7).
- No llama-swap settings shape; typed shapes are not exported at the top level (DEC-000-18).
- Known defect: a reply with null content is returned as `None`, and tracing fails on it (DEC-000-9).
- Known defect: unreachable server and timeout are not mapped to the library error (DEC-000-12).
- Known defect: in-stream error pieces and early stream end are skipped silently instead of raising (DEC-000-13).
- Known defect: a delta carrying both reasoning and content loses the content (DEC-000-21). This is chat completions only; it does not apply on Responses (planned).
- Known defect: a requested tool that was not offered escapes as a raw lookup error (DEC-000-28).
- Known defect: unreadable tool arguments escape as a raw decode error (DEC-000-27).
- Known defect: an unserialisable tool result escapes as a raw error (DEC-000-29).
- Known defect: the round-limit error does not clearly state the limit as its reason (DEC-000-25).
- Known defect: streamed tool-call fragments without a position, or repeating the name, merge into one broken call (DEC-000-32).
- Known defect: the stream reader loses a multi-byte character split between reads, and mishandles bare carriage-return terminators and multi-line events (DEC-000-34).
- The expected shape of a JSON-only answer is not sent to the backend nor checked against the reply, and a scalar reply can be falsely reported as invalid JSON (DEC-000-38).
- The repeated-fill template helper is supported but not exported at the top level (gap; DEC-000-51).
- Known defect: embeddings do not guarantee one vector per input in input order; an empty batch is sent instead of returning an empty result; an entry without a vector is not mapped; an unreadable reply is not mapped (DEC-000-40).
- Known defect: a model-list entry without an id or name, or an unreadable list, is not mapped to the library error (DEC-000-41).
- Known defect: the package cannot be loaded after a normal install; the tool helper's schema library is required at load but is not a declared runtime dependency (DEC-000-43).
- Known defect: concurrent calls can share and overwrite one trace record (DEC-000-45).
- Known defect: an unwritable trace folder fails the call; tracing must not change the outcome (DEC-000-47).
- Without a usage scope the client's connection is never released (open gap; DEC-000-24).
