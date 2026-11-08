# Feature 004 — responses-tools

| Step | File                                     | Status  | Verifier | Date |
|------|------------------------------------------|---------|----------|------|
| 001  | `001.openai-responses-tools.md`          | done    | PASS     | 2026-11-08 |
| 002  | `002.openai-responses-tools-stream.md`   | done    | PASS     | 2026-11-08 |
| 003  | `003.llamaswap-responses-tools.md`       | done    | PASS     | 2026-11-08 |
| 004  | `004.llamaswap-responses-tools-stream.md`| done    | PASS     | 2026-11-08 |

## Skeleton


### Step 001 — frozen interface (2026-11-08)
All in `llm/openai.py`; every new body raises `NotImplementedError`. Import line extended with `NamedTuple`.

- `class _ResponsesToolRound(NamedTuple): text: str; tool_calls: list[dict[str, Any]]; reasoning: list[dict[str, Any]]` — new. Result of one non-streamed round: round text (DEC-003-12 rules, no reasoning/markers); tool calls dense and in output order in the internal assistant-turn shape `{"id": <call_id>, "type": "function", "function": {"name": <name>, "arguments": <arguments>}}` (values as sent, missing → None); reasoning output items verbatim, in order (carry items, DEC-004-7).
- `def _advertised_tool_names(tools_definitions: list[dict[str, Any]]) -> set[str]` — new. Advertised names, `function.name` for nested entries and `name` for flat ones (DEC-004-5); feeds the pre-flight `ValueError` and offered checks; input unmodified.
- `def _flatten_responses_tools(tools_definitions: list[dict[str, Any]]) -> list[dict[str, Any]]` — new. Nested entry → `{type: "function", name, description?, parameters?, strict: <function.strict or False>}`; flat entry unchanged; order kept; new list, input unmodified (DEC-004-5, DEC-004-1).
- `def _read_responses_tool_round(response: Any) -> _ResponsesToolRound` — new. Status checked first as DEC-003-12 (non-mapping / failed / incomplete → `LLMError` with the server's reason); then output visited in order: function_call → calls, message → text, reasoning → carry items, others ignored (DEC-004-8).
- `def _offered_tool_name(call: Any, offered: set[str]) -> str` — new (module copy of DEC-002-3 rule). Reads `call["function"]["name"]` of the internal tool-call shape; empty/missing or not offered → `LLMError` naming the tool / saying the name was empty.
- `def _read_tool_arguments(name: str, raw: Any) -> dict[str, Any]` — new (module copy of DEC-002-4). None/"" → {}; mapping → dict copy; JSON text → decoded object; invalid JSON / non-object / other type → `LLMError` naming the tool.
- `def _tool_result_text(name: str, result: Any) -> str` — new (module copy of DEC-002-5). dict/list → `json.dumps(..., ensure_ascii=False)`; else `str(result)`; unserialisable → `LLMError` naming the tool, chained.
- `OpenAIAPIClient._responses_round_input(self, turns: list[dict[str, Any]], reasoning: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]` — new. Internal copy → one round's `input`: assistant tool-call turn at index i → `reasoning[i]` verbatim, then typed assistant message only for non-empty text, then `{type: function_call, call_id, name, arguments-as-text}` per call; `role: tool` → `{type: function_call_output, call_id, output}`; other turns → exactly the item `_build_responses_payload` makes (DEC-004-6, DEC-004-7, DEC-004-4). Inputs unmodified.
- `OpenAIAPIClient.chat_with_tools(self, messages: List[Dict[str, str]], *, tools_definitions: List[Dict[str, Any]], tools: Dict[str, Callable[..., Any]], system: Optional[str] = None, max_loops: int = 5, options: Optional[OpenAIOptions] = None, stream: bool = False, on_delta: Optional[Callable[[str], Awaitable[None]]] = None, response_format: Optional[Dict[str, Any]] = None) -> str` — changed body only (signature unchanged; was the DEC-003-4 refusal). Non-streamed Responses tool loop per DEC-004-10/-11/-7/-12; returns the final round text; docstring states the step-001 contract.
- `OpenAIAPIClient._build_responses_payload(...)` — unchanged (no extension frozen; the coder adds `tools` / replaces `input` on the returned body, or factors an item helper internally, without changing what `chat` / `generate` / `chat_json` send).
- Caller-compile edits (out of Source-files scope): None.
- Compile gate: `.venv/bin/python -m compileall -q llm && .venv/bin/python -c "import llm"` — clean.


### Step 002 — frozen interface (2026-11-08)
All in `llm/openai.py`; the new body raises `NotImplementedError("step 004.002: _read_responses_tool_stream")`.

- `OpenAIAPIClient._read_responses_tool_stream(self, response: aiohttp.ClientResponse, on_delta: Callable[[str], Awaitable[None]] | None = None) -> _ResponsesToolRound` — new (placed right after `_read_responses_stream`). One streamed tool round per DEC-004-9: reads only via `_iter_sse_data`, delivers text/reasoning pieces with fresh per-round markers (open block closed by `</think>` at `response.completed`), never delivers tool-call fragments; returns step 001's `_ResponsesToolRound` with text deltas only, dense calls in first-appearance order in the internal `{"id", "type": "function", "function": {"name", "arguments"}}` shape, reasoning carry items verbatim ordered by `output_index`; failed/incomplete/error events or stream end without `response.completed` → `LLMError`; receiver failures propagate unchanged.
- `OpenAIAPIClient.chat_with_tools(...)` — signature unchanged (step 001 freeze); its streamed branch is the coder's.
- `OpenAIAPIClient._read_responses_stream(...)` — unchanged.
- Caller-compile edits (out of Source-files scope): None.
- Compile gate: `.venv/bin/python -m compileall -q llm && .venv/bin/python -c "import llm"` — clean.


### Step 003 — frozen interface (2026-11-08)

All in `llm/llama_swap.py` (module-private, imported from nowhere else; DEC-arch-2). Every new body raises `NotImplementedError`.

- `def _tool_definition_name(definition: Any) -> Any:` — new — advertised name of one app definition: `function.name` (nested) or `name` (flat), as given; used for pre-flight and the offered set (DEC-004-5).
- `def _responses_tools(tools_definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:` — new — flatten nested entries to `{type: function, name, description?, parameters?, strict (copied, else False)}`; flat entries unchanged; order kept; input untouched (DEC-004-5, DEC-004-1).
- `def _responses_tool_round(reply: Any) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:` — new — status check as DEC-003-12 (failed/incomplete → `LLMError`), then `(round text, calls as {"call_id","name","arguments"} in order of appearance (dense, no item id, missing → None), reasoning items verbatim in order)` (DEC-004-8).
- `def _offered_tool_name(call: Any, offered: set[str]) -> str:` — new — reads top-level `name` of a round call; empty/missing → `LLMError("The model requested a tool call with an empty tool name")`; non-str/not offered → `LLMError("The model requested tool '<name>', which was not offered")` (DEC-002-3).
- `def _read_tool_arguments(name: str, raw: Any) -> dict[str, Any]:` — new — copy of chat-completions helper: None/"" → {}; mapping → dict; JSON text object → dict; else `LLMError` naming the tool (DEC-002-4).
- `def _tool_result_text(name: str, result: Any) -> str:` — new — copy: dict/list → `json.dumps(..., ensure_ascii=False)` (failure → `LLMError` naming the tool, chained); else `str(result)` (DEC-002-5).
- `LlamaSwapAPIClient._responses_tool_input(self, history: list[dict[str, Any]], carried: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]` — new — convert internal copy (chat-completions turn shapes) to input items by shape: assistant+tool_calls → `carried[pos]` verbatim, typed assistant message only for non-empty text, `function_call` items (mapping args → JSON text, missing → ""); `role: tool` → `function_call_output` (output as text); others → `_responses_payload`'s item rule (DEC-004-6, -7, -4).
- `LlamaSwapAPIClient._responses_tools_payload(self, items: list[dict[str, Any]], *, settings: dict[str, Any], tools: list[dict[str, Any]], stream: bool) -> dict[str, Any]` — new — tool-round body: exactly `model, input=items, stream, store: False, include: ["reasoning.encrypted_content"], tools, **settings` (DEC-004-10, DEC-003-11). `_responses_payload` is NOT changed (chat/generate/chat_json bodies untouched).
- `LlamaSwapAPIClient.chat_with_tools(self, messages: List[Dict[str, str]], *, tools_definitions: List[Dict[str, Any]], tools: Dict[str, Callable[..., Any]], system: Optional[str] = None, max_loops: int = 5, options: Optional[OpenAIOptions] = None, stream: bool = False, on_delta: Optional[Callable[[str], Awaitable[None]]] = None, response_format: Optional[Dict[str, Any]] = None) -> str` — changed body only (signature unchanged); was the DEC-003-4 refusal; now stub `NotImplementedError("step 004.003: chat_with_tools")`; docstring documents the step-003 loop (pre-flight → merge → copy → trace → rounds via `_request`, `log_request("POST", "/responses (loop <n>)", payload)` n from 1, carry record keyed by assistant-turn position, 002 rules, round-limit `RuntimeError` after the trace session).
- Docstring-only: class `LlamaSwapAPIClient` docstring no longer says `chat_with_tools` is refused.
- Caller-compile edits (out of Source-files scope): None.
- Compile gate: `.venv/bin/python -m compileall -q llm && .venv/bin/python -c "import llm"` — clean. `ruff format --check` clean; no new ruff findings from the new signatures (builtin generics used).


### Step 004 — frozen interface (2026-11-08)

All in `llm/llama_swap.py` (DEC-arch-2). New body raises `NotImplementedError("step 004.004: _stream_responses_tool_round")`.

- `LlamaSwapAPIClient._stream_responses_tool_round(self, response: aiohttp.ClientResponse, on_delta: Callable[[str], Awaitable[None]] | None = None) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]` — new — DEC-004-9 tool-round parser over `_iter_sse_data`, separate from `_stream_responses_response`: delivers text/reasoning pieces (fresh markers per round, open block closed by a `</think>` piece at completed; no tool-call text delivered), returns `(round text without reasoning/markers, calls {"call_id","name","arguments"} dense in first-appearance order, reasoning carry items verbatim ordered by output_index)`; handles the llama.cpp sequence (out-of-order item completion, no arguments-done, no `[DONE]`); failed/incomplete/error/early end → `LLMError`.
- `LlamaSwapAPIClient.chat_with_tools` — unchanged by this skeleton (step 003 frozen signature; streamed branch filled by the coder).
- `LlamaSwapAPIClient._stream_responses_response` — unchanged.
- Caller-compile edits (out of Source-files scope): None.
- Compile gate: `.venv/bin/python -m compileall -q llm && .venv/bin/python -c "import llm"` — clean. `ruff format --check` clean; no new ruff findings in the added lines.

## Tests

### Step 001 — tests (2026-11-08)
- `tests/test_openai_responses_tools.py` — new; covers DoD-1..DoD-20 — non-streamed `OpenAIAPIClient.chat_with_tools` on Responses through a per-round `_request` fake (deep-copied bodies, one canned reply per round, probe-shaped llama.cpp and OpenAI reasoning items) and a fake session for non-2xx.
  - DoD-1: `test_openai_nested_tool_definitions_sent_flattened__DoD1`, `test_openai_flat_tool_definition_sent_unchanged_in_order__DoD1`, `test_openai_tool_definitions_unchanged_after_call__DoD1`, `test_openai_pydantic_tool_definition_accepted_and_flattened__DoD1`
  - DoD-2: `test_openai_every_round_posts_fixed_body_plus_tools__DoD2`, `test_openai_every_round_carries_settings_and_thinking__DoD2`
  - DoD-3: `test_openai_response_format_parameter_sent_on_every_round__DoD3`, `test_openai_response_format_parameter_wins_over_setting__DoD3`, `test_openai_response_format_setting_used_without_parameter__DoD3`, `test_openai_no_response_format_sends_no_text_field__DoD3`
  - DoD-4: `test_openai_first_round_without_calls_returns_its_text__DoD4`
  - DoD-5: `test_openai_function_call_runs_tool_and_next_input_carries_items__DoD5`, `test_openai_system_instruction_heads_every_round_input__DoD5`
  - DoD-6: `test_openai_tool_round_text_sent_as_typed_assistant_item__DoD6`, `test_openai_tool_round_without_text_sends_no_assistant_item__DoD6`
  - DoD-7: `test_openai_two_calls_run_in_order_sequentially__DoD7`
  - DoD-8: `test_openai_reasoning_carried_over_three_rounds__DoD8`
  - DoD-9: `test_openai_caller_messages_unchanged_and_answer_clean__DoD9`
  - DoD-10: `test_openai_caller_tool_history_sent_as_call_items_without_reasoning__DoD10`, `test_openai_caller_tool_history_with_text_and_mapping_arguments__DoD10`
  - DoD-11: `test_openai_awaitable_tool_awaited_and_result_sent__DoD11`, `test_openai_mapping_or_list_result_sent_as_json_text__DoD11`, `test_openai_other_result_sent_as_text_form__DoD11`
  - DoD-12: `test_openai_advertised_tool_without_callable_refused_before_request__DoD12`
  - DoD-13: `test_openai_call_to_unadvertised_tool_raises_and_does_not_run__DoD13`, `test_openai_call_with_empty_or_missing_name_raises__DoD13`
  - DoD-14: `test_openai_unreadable_arguments_raise_and_tool_does_not_run__DoD14`, `test_openai_empty_or_missing_arguments_run_tool_without_arguments__DoD14`
  - DoD-15: `test_openai_unknown_argument_name_raises_value_error__DoD15`, `test_openai_raising_tool_ends_with_runtime_error__DoD15`, `test_openai_unserialisable_mapping_result_raises__DoD15`
  - DoD-16: `test_openai_round_limit_reached_raises_runtime_error__DoD16`
  - DoD-17: `test_openai_failed_or_incomplete_round_raises_with_reason__DoD17`, `test_openai_tool_round_non_2xx_raises_without_credential__DoD17`
  - DoD-18: `test_openai_no_counterpart_setting_absent_and_warned_once__DoD18`, `test_openai_second_tool_call_and_later_chat_emit_no_warning__DoD18`, `test_openai_tool_call_after_chat_emits_no_warning__DoD18`
  - DoD-19: `test_openai_traced_tool_call_records_rounds_calls_and_answer__DoD19`
  - DoD-20: `test_openai_chat_with_tools_no_longer_refused__DoD20`
- `tests/test_openai_responses_request.py` — covers DoD-20 — removed 003's `test_openai_chat_with_tools_refused_before_any_request__DoD14`, its section header and its only-used helpers `_TOOL_DEFS` / `_fetch_record`; module docstring no longer mentions the refusal; no import became unused; no other test touched.
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 ✓, DoD-9 ✓, DoD-10 ✓, DoD-11 ✓, DoD-12 ✓, DoD-13 ✓, DoD-14 ✓, DoD-15 ✓, DoD-16 ✓, DoD-17 ✓, DoD-18 ✓, DoD-19 ✓, DoD-20 ✓

### Step 002 — tests (2026-11-08)
- `tests/test_openai_responses_tools_stream.py` — covers DoD-1..DoD-12 — streamed `chat_with_tools` and `_read_responses_tool_stream` on `OpenAIAPIClient`, driven by per-round queues for fake `_stream_post` (deep-copied bodies) and `_iter_sse_data` (DEC-004-14), over both the OpenAI event sequence (summary deltas, arguments-done, encrypted reasoning) and the llama.cpp sequence (reasoning-text deltas, no arguments-done, reasoning finishing last): streamed body = non-streamed body with `stream` true per round (non-streamed run via a `_request` queue); every round's text to the receiver, final text returned; one closed reasoning block per round, no markers in the answer; call assembly from added + deltas, replacement by arguments-done / item-done, re-sent `function_call` item; two calls behind reasoning dense and in order with own `call_id` outputs; carry items verbatim, by output position, same layout as non-streamed; no call fragments delivered; failure events / early end (incl. `[DONE]`) in round 1 and 2 with no tool of that round run; non-2xx on round 2 via fake session (address, server reply, no credential); receiver failure identity and await ordering; JSON format pieces join to the answer; no-receiver and traced runs equal.
- `tests/integrations/test_openai_responses_tools.py` — covers DoD-14, DoD-15, DoD-16 — live per DEC-004-13: two-round conversation on `OPENAI_REASONING_MODEL` (effort low, max_tokens 2048, thinking on) streamed and not; factory parity on both protocols with `OPENAI_MODEL`, streamed answer contained in receiver text; JSON-object format parses. Skips without `OPENAI_API_KEY` / `OPEN_AI_KEY`.
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 ✓, DoD-9 ✓, DoD-10 ✓, DoD-11 ✓, DoD-12 ✓, DoD-13 [manual/live, no test], DoD-14 ✓ (live), DoD-15 ✓ (live), DoD-16 ✓ (live)

### Step 003 — tests (2026-11-08)
- `tests/test_llamaswap_responses_tools.py` — covers DoD-1..DoD-20 — non-streamed `LlamaSwapAPIClient.chat_with_tools` through a per-round `_request` fake (DEC-004-14, llama.cpp probe shapes) or a sequential fake session (credential-less header, 404): flattening + strict, per-round body keys / thinking effort without summary / no bearer header, response format on every round, round input order (caller items, reasoning verbatim, typed assistant text, function_call, function_call_output), reasoning carry over 3 rounds, caller data untouched, caller tool-turn history, result conversion, pre-flight / offered / argument / tool failures, round limit, failed / incomplete / non-2xx, ignored-setting warning once per client, trace record (`/responses (loop n)` from 1), refusal gone.
- `tests/test_llamaswap_responses_request.py` — covers DoD-20 — removed only `test_llamaswap_chat_with_tools_refused_naming_chat_completions_class__DoD14`, its section header and its only-used helpers `_TOOL_DEFS`, `_add`; module docstring no longer lists the refusal; no imports became unused.
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 ✓, DoD-9 ✓, DoD-10 ✓, DoD-11 ✓, DoD-12 ✓, DoD-13 ✓, DoD-14 ✓, DoD-15 ✓, DoD-16 ✓, DoD-17 ✓, DoD-18 ✓, DoD-19 ✓, DoD-20 ✓

### Step 004 — tests (2026-11-08)
- `tests/test_llamaswap_responses_tools_stream.py` — covers DoD-1..DoD-13 — streamed `chat_with_tools` on `LlamaSwapAPIClient` and `_stream_responses_tool_round` driven directly: per-round body = non-streamed body with `stream` true (DoD-1); every round's text to receiver, final text returned (DoD-2); one closed reasoning block per round incl. summary deltas, clean answer (DoD-3); exact llama.cpp sequence runs the tool with full args and re-sends the done reasoning item ahead of the call (DoD-4); arguments-done / output-item-done replace assembled args (DoD-5); two calls at positions 1 and 2 behind reasoning (DoD-6); carry to every later round ordered by output position, same layout as non-streamed (DoD-7); no call text to receiver (DoD-8); failed / incomplete / error / no completed / `[DONE]` in first and later rounds (DoD-9); later-round 404 through the real `_stream_post` (DoD-10); receiver failure and async receiver (DoD-11); format answer = final round pieces joined (DoD-12); no receiver and tracing parity (DoD-13). Fakes: per-round `_stream_post` + `_iter_sse_data` queues, per-round `_request` for parity (DEC-004-14).
- `tests/integrations/test_llamaswap_responses_tools.py` — covers DoD-15, DoD-16, DoD-17 — live: two-round tool conversation with `LLAMA_SWAP_THINKING_MODEL` and `enable_thinking: true`, streamed and not (tool ran, non-empty, no marker; streamed ≥1 well-formed block); parity via the factory on both protocols with `LLAMA_SWAP_MODEL`, streamed and not (answer contained in receiver text when streamed); JSON-object format answer parses. One `fetch_record` tool with a random code and an invocation count; never skipped.
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 ✓, DoD-9 ✓, DoD-10 ✓, DoD-11 ✓, DoD-12 ✓, DoD-13 ✓, DoD-14 [manual/live, no test], DoD-15 ✓ (live), DoD-16 ✓ (live), DoD-17 ✓ (live)

## Files Changed

_populated by the coder as steps complete_

### Step 001 — OpenAI Responses class: non-streamed tool conversation
- `llm/openai.py` — filled tool flattening, advertised names, non-streamed round reading, offered-name / argument / result helpers (copies of 002's rules), round-input conversion and the non-streamed `chat_with_tools` loop; streamed rounds routed to the step-002 parser stub; module header and class docstring updated

### Step 002 — OpenAI Responses class: streamed tool rounds
- `llm/openai.py` — filled `_read_responses_tool_stream` (DEC-004-9 tool-round parser: per-round markers, call assembly by item id / output index, carry items ordered by output index, failure and early-end errors); streamed branch of `chat_with_tools` documented (same body with `stream` true via `_stream_post`, raw receiver, shared loop); module header updated

### Step 003 — llama-swap Responses class: non-streamed tool conversation
- `llm/llama_swap.py` — filled tool-name and flattening helpers (DEC-004-5/-1), non-streamed round reading on top of `_responses_reply_text` (DEC-004-8), offered-name / argument / result helpers (copies of 002's rules), round-input conversion (DEC-004-6/-7/-4; caller turns delegated to `_responses_payload`'s item rule), tool-round body builder and the `chat_with_tools` loop (DEC-004-10/-11/-12); `stream=True` routed to `_stream_post` + the step-004 parser stub with the raw receiver; module header and `chat_with_tools` docstring updated

### Step 004 — llama-swap Responses class: streamed tool rounds
- `llm/llama_swap.py` — filled `_stream_responses_tool_round` (DEC-004-9 tool-round parser: fresh per-round markers with an open block closed at completed, text-only round text, calls assembled by item id with output-index fallback and replaced by full done arguments, dense first-appearance order, reasoning carry items from their done events ordered by output index, failed/incomplete/error and early-end `LLMError`); step 003's streamed `chat_with_tools` wiring checked and kept unchanged; module header updated

## Notes & Issues

_populated by the coder when worth saying_

- Step 002 tests: DoD-1 and DoD-6 (layout clause) compare against the non-streamed path, so they also depend on step 001's loop being correct. DoD-14's well-formed check accepts zero reasoning blocks (DEC-004-13: OpenAI summary not reliable at low effort); the first-block-before-answer check applies only when the returned answer is found in the receiver text. The DoD-9 test runs the real `_stream_post` against a fake session (round 1 answered 200, round 2 404) while `_iter_sse_data` stays faked.

## Ultra phase

- orient: done — waves: W1 001,003 · W2 002,004 · mode: serial (0 pairable steps: 001/002 share llm/openai.py, 003/004 share llm/llama_swap.py) · typecheck isolated
- STOPPED before harvest (plan-level): context.md requires 003's builder to send assistant turns typed `message`; built 003 (bdb35d3) follows DEC-003-11 as written — bare `{role, content}` items for every turn incl. assistant (DEC-003-11 was never corrected). Also DEC-004-3 (stay 0.2.0) conflicts with shipped DEC-003-14 (0.3.0). Awaiting user decision.
- stop resolved 2026-11-08 (user): 003 corrected first (assistant items typed, version back to 0.2.0) — committed as a 003 fix; resume at harvest
- harvest: done — docs/.cache/ultra/004.responses-tools/harvest.md (A source touch sites, B test seams); build profile light (Python); ruff baseline docs/.cache/ultra/004.responses-tools/ruff-baseline.txt
- mode: serial — 0 pairable steps · profile: light (Python, single package)
- skeleton W1 (001, 003): done
- skeleton W2 (002, 004): done
- tests: done (001-004, one wave)
- red-gate: PASS (run 1; 214 failed / 142 passed — all failures NotImplementedError from stubs or assertions on stub behaviour; live files collect 15)
- code: done — steps 001-004 (serial)
- verify: PASS (run 1; unit 1089 passed; live 77/81 — 4 known environmental (2 unserved hard-coded model, 2 llama.cpp json_schema grammar); all 15 new live tool tests pass; 003 streamed-chat files unchanged and green)
- advisory for /finalize: `ruff format` would rewrap tests/test_openai_responses_tools.py, tests/test_openai_responses_tools_stream.py, tests/test_llamaswap_responses_tools.py (format command is scoped to llm/); copy differences allowed by DEC-arch-2 (flat tool defs copied vs passed; non-mapping args json.dumps vs str; backend-specific error texts; llama-swap records empty carry lists); stale lines from 003 correction: DEC-004-4 "before 003 is built", 004 context.md "None of 001, 002 or 003 is built yet"
- note: working tree also holds another session's 005 planning (docs/decisions/005.responses-live-tests.md, docs/plans/005.responses-live-tests/, docs/decisions/index.md DEC-005 rows, Conventions lines in tests/AGENTS.md + tests/integrations/AGENTS.md) — excluded from the 004 commit
- glossary: done — llm/, tests/, tests/integrations/ catalogues (inside markers only); out-of-marker staleness listed for /finalize: llm/AGENTS.md Conventions (inline session.post rule, _tools_stream_request seam now chat-completions only, reasoning-in-tool-content wording), tests/AGENTS.md Conventions seam rule, DEC-003-4 still accepted in index, root AGENTS.md Repository Structure
