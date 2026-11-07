# Data model

## Purpose

Data shapes that callers pass in or rely on: messages, tool definitions, settings keys, the reasoning tag convention and the trace file. There is no persistent store.

## Message

A message is a dictionary with required `role` and `content` strings, used by `chat`, `chat_json` and `generate`. Content is text only; multimodal content is not supported.

`chat_with_tools` accepts a list of string-valued dictionaries; during the loop, richer turns are appended to an internal copy:

| Backend family | Assistant tool-call turn | Tool-result turn |
|---|---|---|
| OpenAI, llama-swap | `role: assistant`, `content`, `tool_calls` | `role: tool`, `content`, `tool_call_id` (when the backend supplied an id) |
| Ollama (**deprecated**) | not appended | `role: tool`, `name`, `content` |

## Tool definition

All backends take the OpenAI function-tool shape:

```json
{"type": "function",
 "function": {"name": "...", "description": "...", "parameters": { /* JSON Schema */ }}}
```

`tools` passed to `chat_with_tools` is a mapping from tool name to callable.

## Settings

Settings are passed as `options` (per call) or `default_options` (per client). Defaults are not documented here.

**Merge precedence**, lowest to highest: backend built-in values, client `default_options`, call `options`. Aliases are resolved first, unsupported keys are dropped, empty values are removed last. An explicit "nothing" (`None`) clears a value at every layer.

- The typed settings shapes are hints. The per-backend supported tables are authoritative; plain dictionaries are accepted.
- The shapes are public; top-level export is pending. There is no llama-swap shape yet (it uses the OpenAI keys).
- Known defects (see `quick-reference.md`, Open): built-in values are always sent, not only the settings the application set; some settings declared in the typed shapes are dropped by the merge.
- The end-user identifier is sent with embedding requests; the call's value wins and a call cannot clear it (accepted). Other settings are not sent with embeddings on OpenAI and llama-swap.

**OpenAI and llama-swap keys:** `temperature`, `top_p`, `max_tokens`, `presence_penalty`, `frequency_penalty`, `seed`, `enable_thinking`, `reasoning_effort`. Alias: `context_length` -> `max_tokens`.

Key meanings that need a note: `enable_thinking` toggles reasoning (OpenAI: translated to `reasoning_effort`; llama-swap: sent as-is); `reasoning_effort` is used by OpenAI and dropped by llama-swap. `max_tokens` is sent only when set.

**Ollama keys (deprecated):** `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`, `num_predict`, `num_ctx`, `num_keep`, `mirostat`, `mirostat_tau`, `mirostat_eta`, `seed`.

| Ollama alias | Resolves to |
|---|---|
| `max_tokens`, `length` | `num_predict` |
| `ctx_len`, `ctx_length`, `context_length` | `num_ctx` |
| `mirostat_mode` | `mirostat` |

`context_length` means context size on Ollama but an output-token cap on OpenAI and llama-swap (see `quick-reference.md`, Open).

## Reasoning tag convention

Reasoning text is delimited in returned or streamed text as `<think>` ... `</think>`. Applies to streaming only (see `subsystems.md`).

## Trace record

Written only when tracing is enabled.

- Layout: `<LLM_TRACE_FOLDER>/<date>/llm-trace-<timestamp>.log`, with UTC date folders and a microsecond-resolution timestamp in the file name; one file per traced operation; folders are created on demand.
- Sections, in emission order:
  1. Header: operation, model and ISO timestamp.
  2. `SYSTEM PROMPT` (when a system instruction is present).
  3. `MESSAGE HISTORY` (`[role] content`, tool calls shown as `-> tool_call: name(args)`).
  4. `REQUEST` (method and path, then the payload as indented JSON; in tool loops tagged with the loop number).
  5. `CHUNK`, one per streamed piece (plain streaming chat only, not tool-path streams).
  6. `TOOL CALL: <name>` and `TOOL RESULT: <name>`.
  7. `RESPONSE` (final text, or the count of embeddings returned).
  8. `ERROR` (`<ExceptionType>: <message>`), followed by the exception being re-raised.
  9. Footer `END TRACE`.
- Request headers are not recorded; the credential header is never written.
- Recording scope and the tracing rules are in `subsystems.md`, Tracer.
