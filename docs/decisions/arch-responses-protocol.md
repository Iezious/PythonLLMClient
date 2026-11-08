# Decisions — arch-responses-protocol

**Stage:** — · **Feature:** — (architect session `responses-protocol`, 2026-11-07; no roadmap yet)

Decisions taken with the user while adding the Responses protocol to the OpenAI-family clients. They sit below the architecture recorded in `docs/architecture/` (§ references in each entry). The design is built: features 003.responses-chat and 004.responses-tools (stage 002.responses) delivered it on 2026-11-08, and their decisions (DEC-003-*, DEC-004-*) realize these entries inside the classes; "Lands in" names the code locations as planned at the time. Ids use the `arch` namespace (`DEC-arch-<n>`); the counter is shared by every `arch-<slug>.md` file so ids stay unique.

### DEC-arch-1 — Protocol chosen by client class; existing names mean Responses
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-responses-protocol) · **Step:** —
- **Date:** 2026-11-07 · **Decided by:** user
- **Question:** How does an app choose between the Responses protocol and chat completions, and which names carry which protocol?
- **Decision:** One class per protocol per backend. `OpenAIAPIClient` and `LlamaSwapAPIClient` use Responses. Today's chat-completions behaviour moves, unchanged, to `OpenAIChatCompletionsClient` and `LlamaSwapChatCompletionsClient`. Apps that construct the existing classes directly move to Responses on upgrade; this is intended.
- **Reason:** The user prefers separate classes. Keeping the existing names on the new default makes Responses the default for direct construction as well as for the factory.
- **Rejected:** One OpenAI-family base that delegates to a protocol component (it would remove the duplication, but the user prefers separate classes). A protocol component used alongside duplicated methods (pays for both). New names for Responses (`OpenAIResponsesClient` …), because direct construction of the old names would stay on chat completions, against the "default is Responses" requirement.
- **Lands in:** `llm/openai.py`, `llm/llama_swap.py`, new chat-completions class modules in `llm/`, `llm/__init__.py`
- **Cites:** docs/architecture/overview.md § What it is; docs/architecture/subsystems.md § Chat protocols; docs/architecture/interfaces.md § Exposed: capability matrix

### DEC-arch-2 — The two Responses classes are fully independent
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-responses-protocol) · **Step:** —
- **Date:** 2026-11-07 · **Decided by:** user
- **Question:** Do the OpenAI and llama-swap Responses classes share a Responses base?
- **Decision:** No. Each of the four OpenAI-family classes extends `LLMClient` directly and holds its own copy of the protocol logic. This continues DEC-000-35 (separate copies, free to diverge). Accepted consequence: the Responses logic is written twice and every fix lands in two files.
- **Reason:** The user chose it. It keeps each class readable on its own and consistent with the existing per-backend copies.
- **Rejected:** A shared Responses base with thin OpenAI and llama-swap subclasses.
- **Lands in:** `llm/openai.py`, `llm/llama_swap.py`
- **Cites:** docs/architecture/subsystems.md § Backend clients; DEC-000-35

### DEC-arch-3 — System instruction on Responses is a system-role input item
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-responses-protocol) · **Step:** —
- **Date:** 2026-11-07 · **Decided by:** user
- **Question:** How does the `system` argument reach the server on the Responses protocol?
- **Decision:** As a system-role item at the head of the input list, the same rule as on chat completions.
- **Reason:** One conversion rule for both protocols. Local engines support system-role items more reliably.
- **Rejected:** The top-level `instructions` field: support on local engines is less certain, and it would mean two conversion rules.
- **Lands in:** `llm/openai.py`, `llm/llama_swap.py`
- **Cites:** docs/architecture/subsystems.md § Call structure of an operation

### DEC-arch-4 — Factory `protocol` values and refusals
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-responses-protocol) · **Step:** —
- **Date:** 2026-11-07 · **Decided by:** user
- **Question:** Which values does the factory's `protocol` argument take, and what happens with bad or inapplicable values?
- **Decision:** `"responses"` (the default) and `"chat_completions"`. Any other value raises `ValueError`. A protocol passed together with `server_type="ollama"` raises `ValueError` saying the choice is not offered on Ollama. Matching is exact, like `server_type` (DEC-000-3).
- **Reason:** Mirrors the existing factory style. Refusing Ollama makes it visible that the choice has no effect there.
- **Lands in:** `llm/llm.py`
- **Cites:** docs/architecture/interfaces.md § Exposed: factory

### DEC-arch-5 — Streamed non-2xx replies raise LLMError with the server reply
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-responses-protocol) · **Step:** —
- **Date:** 2026-11-07 · **Decided by:** user
- **Question:** What does the app see when a streamed request (either protocol) gets a non-2xx reply?
- **Decision:** The client reads the reply body and raises `LLMError` carrying the request address and the server's reply, never the credential. This matches the non-streamed rule (DEC-000-11). It applies to all four OpenAI-family classes. Supersedes DEC-000-10.
- **Reason:** The library's error has to carry the server's reason. The accepted exception for streamed calls is no longer needed.
- **Rejected:** Keeping the raw `aiohttp.ClientResponseError` (DEC-000-10), because it drops the reason.
- **Lands in:** `llm/openai.py`, `llm/llama_swap.py`, new chat-completions class modules, `llm/llm_client.py`
- **Cites:** docs/architecture/interfaces.md § Exposed: error contract; DEC-000-10, DEC-000-11

### DEC-arch-6 — Ignored-setting signal: a library `UserWarning` subclass, once per setting per client
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-responses-protocol) · **Step:** —
- **Date:** 2026-11-07 · **Decided by:** user
- **Question:** How does the library tell the app that a setting was ignored because Responses has no counterpart for it?
- **Decision:** It uses Python's `warnings` mechanism with a library-specific subclass of `UserWarning`, emitted once per setting per client instance. The class is `LLMSettingIgnoredWarning` (name settled in planning, DEC-003-5; emission rule DEC-003-6, tool conversations DEC-004-12).
- **Reason:** Warnings are shown by default and can be filtered by category without any app logging setup.
- **Rejected:** The `logging` module: it depends on the app's logging configuration and is quiet by default.
- **Lands in:** `llm/exceptions.py`, `llm/openai.py`, `llm/llama_swap.py`, `llm/__init__.py`
- **Cites:** docs/architecture/interfaces.md § Exposed: warnings; DEC-003-5; DEC-003-6

### DEC-arch-7 — Responses request: not stored, encrypted reasoning included, summary on OpenAI
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-responses-protocol) · **Step:** —
- **Date:** 2026-11-07 · **Decided by:** user
- **Question:** Which fixed request fields does every Responses request carry?
- **Decision:** `store: false` asks the provider not to store the conversation; this is a preference, not a guarantee. `include: ["reasoning.encrypted_content"]` makes reasoning available to carry between tool rounds (DEC-arch-9). When thinking is on, OpenAI also gets `reasoning.summary: "auto"`.
- **Reason:** The library holds no server-side state, so it should not ask the provider to keep any. Carrying reasoning needs the encrypted content. The summary is what OpenAI streams as visible reasoning.
- **Lands in:** `llm/openai.py`, `llm/llama_swap.py`
- **Cites:** docs/architecture/subsystems.md § Chat protocols; docs/architecture/interfaces.md § Consumed: backend APIs

### DEC-arch-8 — Responses settings mapping
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-responses-protocol) · **Step:** —
- **Date:** 2026-11-07 (sub-points settled 2026-11-07, flipped at finalization 2026-11-08) · **Decided by:** user
- **Question:** Which settings does a Responses request send, and how are they named on the wire?
- **Decision:**
  - Only settings the app set are sent. The Responses classes have no always-sent built-in values, so DEC-000-6 does not apply to them; it still applies to the chat-completions classes.
  - `max_tokens` is sent as `max_output_tokens`.
  - Thinking on → `reasoning.effort`; an explicit effort set by the app wins.
  - Thinking off → the lowest effort the model allows: `none`, unless the app gives an effort itself (settled by DEC-003-1, checked by live tests). There is no retry with another value, because retries are a non-goal.
  - A setting with no Responses counterpart is ignored with the warning in DEC-arch-6.
  - On the llama-swap Responses class, thinking on/off and an explicit effort reach the engine through `reasoning.effort`, as on OpenAI, without the reasoning summary (settled by DEC-003-2, with a live check).
  - The full key-by-key table is DEC-003-9.
- **Reason:** The user wants only what the app set to be sent (the fix asked for in DEC-000-6). The field names follow the Responses wire format.
- **Rejected:** Retrying with another effort value when the model refuses one, because retries are out of scope.
- **Amended:** 2026-11-08, at finalization — the two open sub-points (thinking-off value; how llama-swap receives thinking) are closed by DEC-003-1 and DEC-003-2; status was "accepted (two sub-points open, settled in planning)".
- **Lands in:** the supported-settings tables and merge post-processing in `llm/openai.py`, `llm/llama_swap.py`
- **Cites:** docs/architecture/subsystems.md § Options layer; docs/architecture/data-model.md § Settings; DEC-000-6, DEC-000-19; DEC-003-1; DEC-003-2; DEC-003-9

### DEC-arch-9 — Reasoning carry-over between tool rounds
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-responses-protocol) · **Step:** —
- **Date:** 2026-11-07 · **Decided by:** user
- **Question:** How is a round's reasoning carried to the next round of a Responses tool loop?
- **Decision:** A round's `reasoning` output items are kept verbatim and sent again on the next round of the same call, ahead of that round's `function_call` items. They are never stored on caller messages and are dropped when the call ends. This works for OpenAI (encrypted content) and for local servers (plain content, empty encrypted content).
- **Reason:** Reasoning models need their earlier reasoning to keep going after a tool result. Keeping it inside the call leaves the caller's conversation and the returned answer as they are.
- **Lands in:** `llm/openai.py`, `llm/llama_swap.py`
- **Cites:** docs/architecture/subsystems.md § Tool loop; DEC-000-22

### DEC-arch-10 — Translating app-facing shapes to Responses
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-responses-protocol) · **Step:** —
- **Date:** 2026-11-07 · **Decided by:** user
- **Question:** How do today's tool definitions, `response_format`, tool calls and tool results map onto the Responses wire?
- **Decision:**
  - Tool definitions are accepted in today's nested `function` shape and flattened for Responses.
  - `response_format` (`json_object` / `json_schema`) is accepted in today's shape and mapped to the Responses text-format setting. `chat_json` uses the JSON text format.
  - `function_call` items (`call_id`, `name`, `arguments`) map to and from today's assistant tool-call turn. Tool results are sent as `function_call_output` items.
  - Tool-call indices are dense and 0-based within each round.
- **Reason:** App-facing shapes stay the same on both protocols, so the class does the translation.
- **Lands in:** `llm/openai.py`, `llm/llama_swap.py`
- **Cites:** docs/architecture/data-model.md § Tool definition, § Message; docs/architecture/interfaces.md § Exposed: capability matrix; DEC-000-36

### DEC-arch-11 — Parsing the Responses stream by event type
- **Status:** accepted — amended by DEC-arch-13 (2026-11-08)
- **Stage:** — · **Feature:** — (arch-responses-protocol) · **Step:** —
- **Date:** 2026-11-07 · **Decided by:** user
- **Question:** How are Responses stream events turned into receiver pieces, carry items and failures?
- **Decision:** The parser dispatches on the JSON `type` field of each data payload; `event:` lines may be ignored.
  - Output text delta → text.
  - Reasoning text delta and reasoning summary text delta → reasoning, inside think markers.
  - Output item added (`function_call`) plus function-call arguments delta → tool-call fragments.
  - Output item done (`reasoning`) → carry item (DEC-arch-9).
  - `response.completed` → end of stream.
  - `response.failed`, `response.incomplete` and `error` → `LLMError` carrying the server's reason.
- **Reason:** The JSON payload carries the type itself, so the existing data-only SSE reader can be reused. Reporting failure events closes the in-stream gap (DEC-000-13) on this protocol.
- **Amended:** 2026-11-08, user — the `response.incomplete` clause: when the reason is `max_output_tokens` and the app set the output limit, `response.incomplete` ends the stream like `response.completed` instead of raising; with no app limit, and for every other reason, it stays `LLMError` (DEC-arch-13).
- **Lands in:** `llm/openai.py`, `llm/llama_swap.py`, `llm/llm_client.py` (SSE reader reuse)
- **Cites:** docs/architecture/subsystems.md § Streaming and reasoning; docs/architecture/interfaces.md § Exposed: error contract; DEC-000-13, DEC-000-20

### DEC-arch-12 — Test strategy for the two protocols
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-responses-protocol) · **Step:** —
- **Date:** 2026-11-07 · **Decided by:** user
- **Question:** How are the two protocols covered by tests?
- **Decision:**
  - The existing unit tests switch to the chat-completions class names and become the compatibility regression suite.
  - New unit tests cover: the Responses request shape for each message kind, tool flattening, the mapping from events to the receiver (including interleaved reasoning, text and tool fragments, and dense indices), reasoning carry-over, failure events, and non-2xx bodies.
  - Live tests on OpenAI and llama-swap cover every chat operation.
  - A further live test runs a two-round tool conversation on an OpenAI reasoning model. The model is set by an environment variable (e.g. `OPENAI_REASONING_MODEL`, default `gpt-6-luna`). Only the tests read this variable, never the library.
- **Reason:** Keeps today's behaviour pinned while the default changes, and checks the wire assumptions against real servers.
- **Lands in:** `tests/`, `tests/integrations/`
- **Cites:** docs/architecture/subsystems.md § Chat protocols

### DEC-arch-13 — A reply cut at the app's own output limit is returned, not an error
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-responses-protocol) · **Step:** —
- **Date:** 2026-11-08 · **Decided by:** user
- **Question:** What does the library do when a Responses reply stops because of the output token limit?
- **Decision:** It depends on who set the limit. "The app set the limit" means `max_tokens` (or one of its aliases) is present in the settings after the three-layer merge, whether as a call-time option or as a client default.
  - **App set the limit.**
    - *Non-streamed.* A reply with status `incomplete` and reason `max_output_tokens` returns the extracted text exactly as a completed reply does (extraction per DEC-003-12).
    - *Streamed.* `response.incomplete` with that reason ends the stream as `response.completed` does. A reasoning block still open is closed by a `</think>` piece, and the return equals what the receiver got (DEC-003-13).
    - *Tool conversation.* A cut round with no tool calls is the answer and is returned. A cut round that contains tool calls ends with `LLMError`, because the calls' arguments may be truncated, so they are not run.
    - *JSON answers.* `chat_json` and the JSON answer formats keep their check: a cut answer that is not valid JSON fails as any invalid JSON does.
    - No warning or separate notice is given.
  - **App set no limit.** `incomplete` / `response.incomplete` with reason `max_output_tokens` stays `LLMError` carrying the server's reason.
  - **Unchanged.** Every other incomplete reason (e.g. content filter), `failed` / `response.failed`, the `error` event, and a stream that ends without a completion event stay `LLMError` as before.
  - **llama-swap (accepted gap).** llama.cpp reports a capped reply as `completed` with no incomplete details (`incomplete_details: null`, `usage.output_tokens` equal to the cap, the stream ending `response.completed`; feature 005 live run). A cut at the app's limit therefore already returns text. A cut at a limit the app did not set (the server's own cap, a full context) cannot be told apart from a complete reply and is returned as text. US-062.AC-6 cannot be met on llama-swap until the server reports the reason.
- **Reason:** The caller chose the limit and must expect a cut answer (user, 2026-11-08). Other early stops are breaks the app did not cause, so they stay errors (US-024 narrowed).
- **Rejected:** An error on every cut (the previous rule, DEC-003-12 / DEC-003-13 / DEC-arch-11). Returning the text with a warning (user: no signal). Returning text for every length stop regardless of who set the limit (user kept the error for cuts the app did not ask for). For now, inferring a llama-swap cut from `usage.output_tokens`: it only identifies the case that must return text anyway, and it would need the cap value the server applied.
- **Lands in:** `llm/openai.py`, `llm/llama_swap.py` (Responses reply extraction, Responses stream parser, tool-round reading streamed and not)
- **Cites:** docs/architecture/interfaces.md § Exposed: error contract; DEC-arch-11; DEC-003-12; DEC-003-13; DEC-004-8; DEC-004-9; DEC-004-13; DEC-005-6; DEC-005-8; US-062; US-024.AC-2
