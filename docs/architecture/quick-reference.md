# Quick reference

Index only. Details live in the linked files.

## Public exports (`llm`)

`__version__`, `LLMError`, `LLMErrorKind`, `LLMSettingIgnoredWarning`, `LLMToolObserverWarning`, `ToolEvent`, `ToolErrorKind`, `LLMClient`, `OllamaAPIClient` (**deprecated**), `OpenAIAPIClient`, `LlamaSwapAPIClient`, `OpenAIChatCompletionsClient`, `LlamaSwapChatCompletionsClient`, `get_llm_client`, `substitute_placeholders`, `pydantic_to_openai_tool`, `LLMTracer`. See `interfaces.md`, Exposed: utilities, Exposed: warnings, Exposed: error contract; `data-model.md`, Tool events, Error kinds. `LLMErrorKind`, `LLMToolObserverWarning`, `ToolEvent` and `ToolErrorKind` are agreed for release 0.2.1 and not yet built. The supported surface is the package top level. Pending top-level export: typed settings shapes and the message shape (DEC-000-18), the repeated-fill template helper (DEC-000-51).

`OpenAIAPIClient` and `LlamaSwapAPIClient` mean Responses; the chat-completions behaviour is under the two `*ChatCompletionsClient` names. See `overview.md` What it is.

## Client classes and protocols

| Backend | Responses (default) | Chat completions (compatibility) | Factory `protocol` |
|---|---|---|---|
| OpenAI | `OpenAIAPIClient` | `OpenAIChatCompletionsClient` | omitted or `responses` / `chat_completions` |
| llama-swap | `LlamaSwapAPIClient` | `LlamaSwapChatCompletionsClient` | omitted or `responses` / `chat_completions` |
| Ollama (**deprecated**) | — | — | refused |

Every chat operation, tool conversations included, is available on both protocols. See `interfaces.md` Exposed: factory, Protocol axis.

## Environment variables

| Variable | Meaning | See |
|---|---|---|
| `LLM_TRACE` | true-like value (`1`, `true`, `on`) enables tracing; needs `LLM_TRACE_FOLDER` too. Read only when the constructor's `trace` is not passed | `subsystems.md` Tracer |
| `LLM_TRACE_FOLDER` | root folder for trace files; also the folder for `trace=True` (which refuses creation without it) | `data-model.md` Trace record |

The library reads no credential from the environment. An application may read its own key from its environment and pass it in.

## Constructor and call arguments

| Argument | Where | Meaning | See |
|---|---|---|---|
| `model`, `base_url`, `timeout`, `bearer_token`, `default_options` | every class, factory | as before; `timeout` is the total limit per request | `interfaces.md` Exposed: constructor arguments |
| `trace` | every class (Ollama: passed to the base only), factory | not passed = environment decides; `False` = never; `True` = on, folder from `LLM_TRACE_FOLDER` | `interfaces.md` Exposed: constructor arguments |
| `read_timeout` | OpenAI-family classes, factory (refused with `ollama`) | silence limit between reads of a streamed reply | `interfaces.md` Exposed: constructor arguments |
| `protocol` | factory | `responses` (default) or `chat_completions` | `interfaces.md` Exposed: factory |
| `tool_errors` | `chat_with_tools`, OpenAI family | `"result"` (default, tolerant) or `"raise"` (strict, opt-out) | `interfaces.md` Tool failures |
| `on_tool_event` | `chat_with_tools`, OpenAI family | observer receiving `ToolEvent` | `interfaces.md` Tool event observer |

`trace`, `read_timeout`, `tool_errors` and `on_tool_event` are agreed for release 0.2.1 and not yet built.

## Backend routes

| Backend | Routes | See |
|---|---|---|
| Ollama (**deprecated**) | `/api/tags`, `/api/generate`, `/api/chat`, `/api/embed`, `/api/embeddings` (fallback) | `interfaces.md` Consumed |
| OpenAI, llama-swap | `/models`, `/embeddings` (both protocols); `/responses` (Responses classes); `/chat/completions` (chat-completions classes); all under the `/v1` base | `interfaces.md` Consumed |

## Settings keys

| Backend | Keys | See |
|---|---|---|
| OpenAI, llama-swap | `temperature`, `top_p`, `max_tokens`, `presence_penalty`, `frequency_penalty`, `seed`, `enable_thinking`, `reasoning_effort`, `stop`, `logit_bias`, `user`, `response_format` (per-class tables; no built-in values on either protocol; Responses sends only what the app set, translated, and warns on keys with no counterpart) | `data-model.md` Settings |
| Ollama (**deprecated**) | `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`, `num_predict`, `num_ctx`, `num_keep`, `mirostat`, `mirostat_tau`, `mirostat_eta`, `seed` | `data-model.md` Settings |

## Component index

| Component | See |
|---|---|
| Base client (the no-shared-behaviour rule), backend clients, chat protocols, options layer | `subsystems.md` |
| Tool loop, streaming and reasoning, tracer | `subsystems.md` |
| Method contract, tool failures, tool event observer, capability matrix (with protocol axis), constructor arguments, factory, warnings, typing contract, errors | `interfaces.md` |
| Message, tool definition, tool events, error kinds, trace record | `data-model.md` |

## Open (architecture questions)

- Errors are not fully normalised: a chat-style reply that is malformed or not JSON surfaces as a generic error (accepted). Transport failures, streamed non-2xx replies, in-stream failures and streams that end without completing are `LLMError` on both protocols (DEC-001-7, DEC-arch-5, DEC-001-1, DEC-001-9).
- The deprecated Ollama tool loop does not keep the assistant tool-call turn in the conversation.
- The `context_length` alias means context size on Ollama but an output-token cap on OpenAI and llama-swap.
- The llama-swap client accepts the same settings as the OpenAI client; top-k, min-p and repeat-penalty are not accepted (dropped).
- Known defects, deprecated Ollama client only (kept, observation-only): it still sends its built-in setting values, and still drops some keys declared in its typed shape (DEC-001-13, DEC-001-14).
- No llama-swap settings shape; typed shapes are not exported at the top level (DEC-000-18).
- The expected shape of a JSON-only answer is not sent to the backend (gap; DEC-000-38, open part).
- The repeated-fill template helper is supported but not exported at the top level (gap; DEC-000-51).
- Known defect: an empty embeddings batch is sent to the server instead of returning an empty result without a request (DEC-000-40, open part).
- Without a usage scope the client's connection is never released (open gap; DEC-000-24).
- Accepted gap: on llama-swap over Responses, a reply cut at a limit the app did not set (server cap, full context) cannot be told from a complete reply and is returned as text, because the server reports it as complete (DEC-arch-13).
