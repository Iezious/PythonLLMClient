# Quick reference

Index only. Details live in the linked files.

## Public exports (`llm`)

`__version__`, `LLMError`, `LLMClient`, `OllamaAPIClient` (**deprecated**), `OpenAIAPIClient`, `LlamaSwapAPIClient`, `get_llm_client`, `substitute_placeholders`, `pydantic_to_openai_tool`, `LLMTracer`. See `interfaces.md`.

## Environment variables

| Variable | Meaning | See |
|---|---|---|
| `LLM_TRACE` | true-like value (`1`, `true`, `on`) enables tracing; needs `LLM_TRACE_FOLDER` too | `subsystems.md` Tracer |
| `LLM_TRACE_FOLDER` | root folder for trace files | `data-model.md` Trace record |
| `OPEN_AI_KEY` | OpenAI credential, checked before `OPENAI_API_KEY` | `interfaces.md` Factory |
| `OPENAI_API_KEY` | OpenAI credential, fallback | `interfaces.md` Factory |

All other configuration is constructor arguments: `model`, `base_url`, `timeout`, `bearer_token`, `default_options`.

## Backend routes

| Backend | Routes | See |
|---|---|---|
| Ollama (**deprecated**) | `/api/tags`, `/api/generate`, `/api/chat`, `/api/embed`, `/api/embeddings` (fallback) | `interfaces.md` Consumed |
| OpenAI, llama-swap | `/models`, `/chat/completions`, `/embeddings` (under `/v1` base) | `interfaces.md` Consumed |

## Settings keys

| Backend | Keys | See |
|---|---|---|
| OpenAI, llama-swap | `temperature`, `top_p`, `max_tokens`, `presence_penalty`, `frequency_penalty`, `seed`, `enable_thinking`, `reasoning_effort` | `data-model.md` Settings |
| Ollama (**deprecated**) | `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`, `num_predict`, `num_ctx`, `num_keep`, `mirostat`, `mirostat_tau`, `mirostat_eta`, `seed` | `data-model.md` Settings |

## Component index

| Component | See |
|---|---|
| Base client, backend clients, options layer | `subsystems.md` |
| Tool loop, streaming and reasoning, tracer | `subsystems.md` |
| Method contract, capability matrix, factory, errors | `interfaces.md` |
| Message, tool definition, trace record | `data-model.md` |

## Open (architecture questions)

- Errors are not normalised: non-streaming failures raise `LLMError`, streaming failures raise `aiohttp.ClientResponseError`.
- `pydantic` is an import-time dependency of the package but is not a declared runtime dependency.
- The deprecated Ollama tool loop does not keep the assistant tool-call turn in the conversation.
- The `context_length` alias means context size on Ollama but an output-token cap on OpenAI and llama-swap.
- The llama-swap client accepts the same settings as the OpenAI client; top-k, min-p and repeat-penalty are not accepted (dropped).
