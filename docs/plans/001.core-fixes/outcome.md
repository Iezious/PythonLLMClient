## Product

- US-003 — AC-1, AC-2 and AC-4 met; AC-3 kept. The FEAT-001 credential defect gap closes.
- US-047.AC-1 — met. The FEAT-001 load-after-install defect gap closes.
- US-011.AC-3 — met on every backend, the deprecated Ollama client included. The FEAT-002 unreachable-server / time-limit defect gap and the matching cross-feature line close.
- US-009 — AC-1 and AC-3 met on OpenAI and llama-swap, including the final answer of a tool conversation; the deprecated Ollama client is not changed. The FEAT-002 empty-content defect gap closes for OpenAI and llama-swap.
- US-024 — AC-1 and AC-2 met on chat completions. The FEAT-004 annotation "open for chat completions" closes, and the cross-feature rule's chat-completions exception no longer covers a failure the server reports inside a stream or a stream that ends early. A stream is complete when the server signals completion in either of its usual ways.
- US-036 — AC-1..AC-3 met. The FEAT-004 complete-text defect gap closes.
- US-026.AC-5 — met. The FEAT-005 defect gap for a piece carrying both reasoning and answer text closes.
- US-018.AC-1 — delivered with modification: met on OpenAI and llama-swap; the deprecated Ollama client still sends its own values (out of scope per the brief, Ollama is observation-only).
- US-019.AC-1 — delivered with modification: stop sequences, token bias, end-user identifier and also response format reach the server on OpenAI and llama-swap (the FEAT-010 gap names response format; the AC does not yet); the deprecated Ollama client still ignores its extra offered settings.
- New behaviour (settled with the user): when the app sets a response format as a setting, a JSON-only answer requests that format; otherwise it requests plain JSON. No story covers this yet (FEAT-006).

## Architecture

- `interfaces.md` § Exposed: error contract — the "Today" column becomes "as intended" for: transport failure; streamed non-2xx on chat completions; null content, non-streamed (OpenAI and llama-swap); in-stream error piece or early end on chat completions; missing OpenAI credential.
- `interfaces.md` § Exposed: capability matrix, protocol axis — chat-completions column: settings sent are only those the app set; a piece with both reasoning and content delivers both; a failure reported inside the stream ends the call with the library error carrying the server's reason. The `chat_json` request row for chat completions: the app's response-format setting is sent when set, otherwise the JSON-object format.
- `interfaces.md` § Exposed: factory — drop the known-defect bullet on the OpenAI credential.
- `interfaces.md` § Exposed: utilities — the schema library becomes a declared runtime dependency, so the package loads after a normal install; replace the "must not be required to load the package" wording.
- `interfaces.md` § Consumed: backend APIs, SSE stream format (chat completions) — drop the known defects on split multi-byte characters, bare carriage-return terminators and multi-line events; error frames end the call with the library error; a stream that ends without any completion signal ends the call with the library error; other unreadable frames are still skipped.
- `subsystems.md` § Base client — the shared request contract also covers streamed requests: open the stream, map a refused stream (with the server's reply) and transport failures to the library error; transport failures on non-streamed requests are mapped too.
- `subsystems.md` § Options layer and `data-model.md` § Settings — OpenAI and llama-swap have no built-in values (only settings the app set are sent); their key list gains stop sequences, token bias, end-user identifier and response format; the two known defects stay for the deprecated Ollama client only.
- `subsystems.md` § Streaming and reasoning — a piece carrying both reasoning and content delivers both, reasoning first; a failure reported inside a chat-completions stream ends the call with the library error.
- `overview.md` § Stack choices — runtime dependencies include the schema library. § Out of scope — error pieces inside a chat-completions stream are no longer a known defect.
- `quick-reference.md` § Open — remove the resolved items (credential, built-in values and dropped keys for the OpenAI family, null content, transport mapping, in-stream errors and early end, reasoning-and-content piece, stream reader, load after install); narrow the first item to its remaining open part. § Settings keys — add the four keys for OpenAI and llama-swap.

## Decisions

- DEC-001-1 — Early stream end raises unless a finish reason or a done frame arrived
- DEC-001-2 — chat_json uses the response format setting when the app set one
- DEC-001-3 — Pydantic is a declared runtime dependency
- DEC-001-4 — Package version is 0.2.0 in both declarations
- DEC-001-5 — The factory forwards the credential to the OpenAI client
- DEC-001-6 — The OpenAI client refuses a missing credential and never reads the environment
- DEC-001-7 — Transport failures map to LLMError carrying the address
- DEC-001-8 — Streamed requests open through one base helper
- DEC-001-9 — In-stream error frames raise LLMError and malformed frames are skipped
- DEC-001-10 — Stream reader decodes incrementally and follows SSE line and event rules
- DEC-001-11 — A delta with reasoning and content delivers both
- DEC-001-12 — Null reply content becomes empty text where it is extracted
- DEC-001-13 — OpenAI and llama-swap tables carry no built-in values
- DEC-001-14 — Offered settings keys join the OpenAI and llama-swap tables
- DEC-001-15 — In a tool chat the response format parameter wins over the setting
- DEC-001-16 — The embedding end-user id keeps its direct read

## Observations

_appended by coder / fixer / reviser; each bullet tagged [product] | [architecture] | [decision]_
