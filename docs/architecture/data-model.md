# Data model

## Purpose

Data shapes that callers pass in or rely on: messages, tool definitions, tool events, error kinds, settings keys, the reasoning tag convention and the trace file. There is no persistent store.

## Message

A message is a dictionary with required `role` and `content` strings, used by `chat`, `chat_json` and `generate`. Content is text only; multimodal content is not supported.

`chat_with_tools` accepts a read-only sequence of mappings with string keys and any values (`Sequence[Mapping[str, Any]]`, see `interfaces.md`, Exposed: typing contract), so the caller may pass turns that carry tool calls. During the loop, richer turns are appended to an internal copy:

| Backend family | Assistant tool-call turn | Tool-result turn |
|---|---|---|
| OpenAI, llama-swap | `role: assistant`, `content`, `tool_calls` | `role: tool`, `content`, `tool_call_id` (when the backend supplied an id) |
| Ollama (**deprecated**) | not appended | `role: tool`, `name`, `content` |

The message shape and the internal turn shapes above are the same on both OpenAI-family protocols. A Responses class converts them to and from the protocol's input and output items on the wire, tool calls and tool results included. Reasoning carried between tool rounds is not a message and is never added to the conversation (see `subsystems.md`, Tool loop).

## Tool definition

All backends take the OpenAI function-tool shape:

```json
{"type": "function",
 "function": {"name": "...", "description": "...", "parameters": { /* JSON Schema */ }}}
```

`tools` passed to `chat_with_tools` is a mapping from tool name to callable.

This shape is accepted on both OpenAI-family protocols. A Responses class converts it to the protocol's tool shape. The same applies to the `response_format` shape (`json_object` / `json_schema`). Mapping details are in DEC-arch-10.

## Tool events

`ToolEvent` is the typed dictionary the tool-event observer receives (`on_tool_event`, see `interfaces.md`, Tool event observer). It is exported at the top level and is the same on both OpenAI-family protocols.

| Field | Type | Meaning |
|---|---|---|
| `type` | `"start"` \| `"result"` \| `"error"` | `start` before the tool runs; then `result` or `error` — except when the observer itself fails under tolerant handling, after which it receives no further event for that tool call (see `interfaces.md` § Tool event observer; DEC-arch-26) |
| `name` | text | the tool name the model requested |
| `call_id` | text or none | the call identity the server gave the call (the Responses call id, or the chat-completions tool-call id); none when the server gave none |
| `arguments` | text | the raw arguments as the model sent them; a mapping is JSON-encoded, missing arguments are empty text |
| `output` | text or none | the result text the model receives, on a `result` event; none otherwise |
| `error_kind` | `ToolErrorKind` or none | the failure kind, on an `error` event; none otherwise |

`ToolErrorKind` is an exported closed set of names: `"not_offered"`, `"bad_arguments"`, `"unknown_argument"`, `"tool_raised"`, `"unserialisable_result"`. Their meanings are in `interfaces.md`, Tool failures.

## Error kinds

`LLMErrorKind` is an exported closed set of names, carried by `LLMError.kind`: `"status"`, `"timeout"`, `"connection"`, `"interrupted"`, `"invalid_reply"`, `"server_failed"`, `"incomplete"`, `"tool"`. Which failure carries which kind is in `interfaces.md`, Exposed: error contract. `server_failed` covers a failure the server reports, streamed or not, which is why it is not named after streams (DEC-arch-19).

## Settings

Settings are passed as `options` (per call) or `default_options` (per client). Defaults are not documented here.

**Merge precedence**, lowest to highest: backend built-in values (only the deprecated Ollama client has any), client `default_options`, call `options`. Aliases are resolved first, unsupported keys are dropped, empty values are removed last. An explicit "nothing" (`None`) clears a value at every layer.

- The typed settings shapes are hints. The per-backend supported tables are authoritative; plain dictionaries are accepted.
- The shapes are public; top-level export is pending. There is no llama-swap shape yet (it uses the OpenAI keys).
- OpenAI and llama-swap have no built-in values: only the settings the application set are sent, and every key declared in their typed shape is in their supported table.
- Known defects, deprecated Ollama client only (see `quick-reference.md`, Open): its built-in values are always sent, not only the settings the application set; some settings declared in its typed shape are dropped by the merge.
- The end-user identifier is sent with embedding requests; the call's value wins and a call cannot clear it (accepted). Other settings are not sent with embeddings on OpenAI and llama-swap.

**OpenAI and llama-swap keys:** `temperature`, `top_p`, `max_tokens`, `presence_penalty`, `frequency_penalty`, `seed`, `enable_thinking`, `reasoning_effort`, `stop`, `logit_bias`, `user`, `response_format`. Alias: `context_length` -> `max_tokens`.

Key meanings that need a note: `enable_thinking` toggles reasoning (OpenAI: translated to `reasoning_effort`; llama-swap: sent as-is); `reasoning_effort` is used by OpenAI and dropped by llama-swap. `max_tokens` is sent only when set. `stop` holds stop sequences, `logit_bias` the token bias, `user` the end-user identifier (also sent with embeddings, see above). `response_format` is the answer format: `chat_json` sends it when set, and in a tool chat the call's own `response_format` argument replaces it (see `interfaces.md`, capability matrix and Tool loop contract).

**Ollama keys (deprecated):** `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`, `num_predict`, `num_ctx`, `num_keep`, `mirostat`, `mirostat_tau`, `mirostat_eta`, `seed`.

| Ollama alias | Resolves to |
|---|---|
| `max_tokens`, `length` | `num_predict` |
| `ctx_len`, `ctx_length`, `context_length` | `num_ctx` |
| `mirostat_mode` | `mirostat` |

`context_length` means context size on Ollama but an output-token cap on OpenAI and llama-swap (see `quick-reference.md`, Open).

The OpenAI and llama-swap keys and notes above are those of the chat-completions classes.

**Responses classes.** They accept the same app-facing keys as the chat-completions classes, and each class has its own supported table. Differences:

- No built-in values: only keys the app set are sent.
- Each key is translated to its Responses counterpart, for example the output-token cap and the thinking switch. On both backends the thinking switch and an explicit effort go through the protocol's reasoning setting.
- A key with no Responses counterpart is not sent, and the ignored-setting warning is emitted (see `interfaces.md`, Exposed: warnings).
- The app sets the output limit when the output-token cap (or its alias) is present after the merge, as a call option or a client default; this decides how a reply cut at that limit is treated (see `interfaces.md`, error contract).

The key-by-key mapping is in DEC-arch-8 and DEC-003-9; the thinking-off value and how llama-swap receives thinking are settled in DEC-003-1 and DEC-003-2.

## Reasoning tag convention

Reasoning text is delimited in returned or streamed text as `<think>` ... `</think>`. Applies to streaming only (see `subsystems.md`). The same convention applies on both OpenAI-family protocols. On Responses, the reasoning text and the reasoning summary the server streams both become marked reasoning.

## Trace record

Written only when tracing is enabled.

- Layout: `<LLM_TRACE_FOLDER>/<date>/llm-trace-<timestamp>.log`, with UTC date folders and a microsecond-resolution timestamp in the file name; one file per traced operation; folders are created on demand. When a record with that name already exists, a disambiguating suffix follows the timestamp, so records that start at the same instant never overwrite each other.
- Sections, in emission order:
  1. Header: operation, model and ISO timestamp.
  2. `SYSTEM PROMPT` (when a system instruction is present). The instruction is shown here, in its own section.
  3. `MESSAGE HISTORY` (`[role] content`, tool calls shown as `-> tool_call: name(args)`). The system message the library inserted from the instruction is not repeated here; system messages the caller supplied are.
  4. `REQUEST` (method and path, then the payload as indented JSON; in tool loops tagged with the loop number). The request is shown as sent, so it still contains the instruction.
  5. `CHUNK`, one per streamed piece (plain streaming chat only, not tool-path streams).
  6. `TOOL CALL: <name>` and `TOOL RESULT: <name>`. In tolerant mode a failing call's result is the text the model was told, followed by a note naming the failure kind; an observer failure is recorded too (DEC-arch-21).
  7. `RESPONSE` (final text, or the count of embeddings returned).
  8. `ERROR` (`<ExceptionType>: <message>`), followed by the exception being re-raised.
  9. Footer `END TRACE`.
- Request headers are not recorded; the credential header is never written.
- Recording scope and the tracing rules are in `subsystems.md`, Tracer.

## Decision history

- 2026-11-09 (release 0.2.1) — `ToolEvent`, `ToolErrorKind` and `LLMErrorKind` added as exported shapes. Tool-chat messages are accepted as a sequence of mappings with any values. The trace records tolerant tool failures. See DEC-arch-14, DEC-arch-19, DEC-arch-21, DEC-arch-22.
