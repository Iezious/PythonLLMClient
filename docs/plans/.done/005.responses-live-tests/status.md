# Feature 005 — responses-live-tests

| Step | File                                  | Status  | Verifier | Date |
|------|---------------------------------------|---------|----------|------|
| 001  | `001.conversation.md`                 | done    | PASS     | 2026-11-08 |
| 002  | `002.settings-formats.md`             | done    | PASS     | 2026-11-08 |
| 003  | `003.errors-session-endpoints.md`     | done    | PASS     | 2026-11-08 |
| 004  | `004.tools-extra.md`                  | done    | PASS     | 2026-11-08 |
| 005  | `005.output-limit-tools-json.md`      | done    | PASS     | 2026-11-08 |

## Skeleton

### Step 001 — frozen interface (2026-11-08)
- none: test-only (DEC-005-1). No Source files; no library signature added or changed; no stubs written. `tests/integrations/live_support.py` is test code owned by the test-coder (DEC-005-2) and is not frozen here.
- Existing library signatures the step's tests call (existing, unchanged; cited from harvest):
  - `llm/llm.py` — `get_llm_client(server_type: str, base_url: str, model: str, timeout: float = 600.0, bearer_token: Optional[str] = None, default_options: Optional[Dict[str, Any]] = None, protocol: str | None = None) -> LLMClient` — existing, unchanged (`protocol` omitted → `OpenAIAPIClient` for `"openai"`, `LlamaSwapAPIClient` for `"llamaswap"`)
  - `llm/openai.py` `OpenAIAPIClient` / `llm/llama_swap.py` `LlamaSwapAPIClient` — `async chat(messages: List[LLMMessage], *, system: Optional[str] = None, options: Optional[OpenAIOptions] = None, stream: bool = False, on_delta: Optional[Callable[[str], Awaitable[None]]] = None) -> str` — existing, unchanged
  - same classes — `async generate(prompt: str, *, system=None, options=None, stream=False, on_delta=None) -> str` — existing, unchanged
  - same classes — `async with client:` session lifecycle (`__aenter__` / `__aexit__`) — existing, unchanged
  - `llm/exceptions.py` — `LLMError(Exception).__init__(self, message: str, description: str | None = None)`; attributes `.message`, `.description` — existing, unchanged (used by the helper's secret-free error check)
- Caller-compile edits (out of Source-files scope): None.
- Compile gate: none (no code written).

### Step 002 — frozen interface (2026-11-08)
- **Stale (replan 2026-11-08):** this entry was frozen for the blocked first run of step 002. The step was replanned (DoD-1 / DoD-2 rewritten, DoD-8 / DoD-9 added); the resumed run rewrites this entry.
- none: test-only (DEC-005-1). No Source files; no library signature added or changed; no stubs written.
- Existing library signatures the step's tests call (existing, unchanged; cited from harvest):
  - `llm/llm.py` — `get_llm_client(server_type: str, base_url: str, model: str, timeout: float = 600.0, bearer_token: Optional[str] = None, default_options: Optional[Dict[str, Any]] = None, protocol: str | None = None) -> LLMClient` — existing, unchanged (`protocol` omitted -> Responses class `OpenAIAPIClient` / `LlamaSwapAPIClient`)
  - `llm/openai.py` `OpenAIAPIClient`, `llm/llama_swap.py` `LlamaSwapAPIClient` — `async chat(messages: List[LLMMessage], *, system: Optional[str] = None, options: Optional[OpenAIOptions] = None, stream: bool = False, on_delta: Optional[Callable[[str], Awaitable[None]]] = None) -> str` — existing, unchanged
  - same classes — `async chat_json(messages: List[LLMMessage], *, response_schema: Optional[Dict[str, Any]] = None, system: Optional[str] = None, options: Optional[OpenAIOptions] = None) -> str` — existing, unchanged
  - same classes — async context manager (`async with client:`) — existing, unchanged
  - `llm/exceptions.py` — `LLMError(Exception).__init__(self, message: str, description: str | None = None)`; attributes `.message`, `.description` — existing, unchanged
  - `llm/exceptions.py` — `LLMSettingIgnoredWarning(UserWarning)` (emitted via `warnings`, never raised) — existing, unchanged
- Test-side dependency (not a library signature): `tests.integrations.live_support` from step 001 (clients, keys, models, recorder, marker checks).
- Caller-compile edits (out of Source-files scope): None.

### Step 003 — frozen interface (2026-11-08)
- none: test-only (DEC-005-1). No Source files; no library signature added or changed; no stubs written.
- Existing library signatures the step's tests call (existing, unchanged; cited from harvest):
  - `llm/llm.py` — `get_llm_client(server_type: str, base_url: str, model: str, timeout: float = 600.0, bearer_token: Optional[str] = None, default_options: Optional[Dict[str, Any]] = None, protocol: str | None = None) -> LLMClient` — existing, unchanged (`protocol=None` → Responses class `OpenAIAPIClient` / `LlamaSwapAPIClient`)
  - `llm/exceptions.py` — `LLMError(Exception).__init__(self, message: str, description: str | None = None)`; attributes `.message`, `.description`; `str(err)` = `"message: description"` when description truthy, else `message` — existing, unchanged
  - `llm/openai.py` (`OpenAIAPIClient`), `llm/llama_swap.py` (`LlamaSwapAPIClient`) — `async chat(messages: List[LLMMessage], *, system: Optional[str] = None, options: Optional[OpenAIOptions] = None, stream: bool = False, on_delta: Optional[Callable[[str], Awaitable[None]]] = None) -> str` — existing, unchanged
  - `llm/llm_client.py` (inherited by both Responses classes) — `async embed(text: str, *, options: Optional[Dict[str, Any]] = None) -> List[float]` — existing, unchanged
  - `llm/openai.py`, `llm/llama_swap.py` — `async embed_batch(texts: List[str], *, options: Optional[OpenAIOptions] = None) -> List[List[float]]` — existing, unchanged
  - `llm/openai.py`, `llm/llama_swap.py` — `async get_models() -> List[Dict[str, Any]]` — existing, unchanged
  - Both Responses classes — async context manager (`async with client:`; session created lazily, closed at scope exit, recreated by a later call, DEC-000-24) — existing, unchanged
- Test-side dependencies (not library interface): shared helper `tests/integrations/live_support.py` from step 001 (builder with credential/address/time-limit overrides, secret-free error check, embedding-model readers, recorder); module-local constants for dummy key and closed-port address (DEC-005-8) and cosine-similarity helper (DEC-005-7) live in the step's Test files.
- Caller-compile edits (out of Source-files scope): None.

### Step 004 — frozen interface (2026-11-08)
- Library interface: none: test-only (DEC-005-1). No Source files; no stubs written; no library signature added or changed.
- Existing, unchanged signatures the step's tests call (from harvest § 1):
  - `llm/llm.py` — `get_llm_client(server_type: str, base_url: str, model: str, timeout: float = 600.0, bearer_token: Optional[str] = None, default_options: Optional[Dict[str, Any]] = None, protocol: str | None = None) -> LLMClient` — existing, unchanged (`protocol` omitted → Responses class: `OpenAIAPIClient` in `llm/openai.py`, `LlamaSwapAPIClient` in `llm/llama_swap.py`)
  - `llm/openai.py`, `llm/llama_swap.py` — `async chat_with_tools(messages: List[Dict[str, str]], *, tools_definitions: List[Dict[str, Any]], tools: Dict[str, Callable[..., Any]], system: Optional[str] = None, max_loops: int = 5, options: Optional[OpenAIOptions] = None, stream: bool = False, on_delta: Optional[Callable[[str], Awaitable[None]]] = None, response_format: Optional[Dict[str, Any]] = None) -> str` — existing, unchanged (delivered by feature 004.responses-tools; `tools` values sync or async)
  - `llm/tools.py` — `pydantic_to_openai_tool(name: str, description: str, schema: type[BaseModel]) -> dict[str, Any]` — existing, unchanged
  - `LLMClient` async context manager (`async with client:`) — existing, unchanged
  - `tests/integrations/live_support.py` (step 001) — clients, recorder, marker checks, well-formed-block check — consumed as frozen by step 001's `## Skeleton` entry, unchanged here
- Caller-compile edits (out of Source-files scope): None.
- Compile gate: none needed (no code touched).

## Tests

### Step 001 — tests (2026-11-08)
- `tests/integrations/live_support.py` — shared helper (DEC-005-2), no tests; consumed read-only by steps 002–004
- `tests/integrations/test_responses_conversation.py` — covers DoD-1..7 — builder class, multi-turn history token + unchanged caller list, streamed equality + well-formed blocks, uppercase system instruction on chat/generate, Cyrillic + Japanese + no U+FFFD streaming, streamed generate piece count
- Traceability:
  - DoD-1: `test_openai_builder_gives_responses_class__DoD1`, `test_llamaswap_builder_gives_responses_class__DoD1`
  - DoD-2: `test_openai_multi_turn_history_not_streamed__DoD2`, `test_llamaswap_multi_turn_history_not_streamed__DoD2`
  - DoD-3: `test_openai_multi_turn_history_streamed__DoD3`, `test_llamaswap_multi_turn_history_streamed__DoD3`
  - DoD-4: `test_openai_chat_follows_uppercase_system__DoD4`, `test_llamaswap_chat_follows_uppercase_system__DoD4`
  - DoD-5: `test_openai_generate_follows_uppercase_system__DoD5`, `test_llamaswap_generate_follows_uppercase_system__DoD5`
  - DoD-6: `test_openai_multilingual_chat_streamed__DoD6`, `test_llamaswap_multilingual_chat_streamed__DoD6`
  - DoD-7: `test_llamaswap_generate_streamed_pieces__DoD7`
- Helper public API (`from tests.integrations.live_support import ...`):
  - Constants: `BACKENDS = ("openai", "llamaswap")`, `OPENAI_URL = "https://api.openai.com/v1"`, `LLAMA_SWAP_THINKING_OFF = {"enable_thinking": False}`, `THINK_OPEN`, `THINK_CLOSE`
  - Readers (DEC-005-3): `llama_swap_base_url()`, `llama_swap_model()`, `llama_swap_embed_model()`, `openai_model()`, `openai_reasoning_model()`, `openai_embed_model()`, `openai_base_url()`, `openai_api_key()` (skips when `OPENAI_API_KEY` / `OPEN_AI_KEY` empty)
  - `build_client(backend, model=None, *, default_options=None, timeout=None, base_url=None, bearer_token=None, thinking_off=True) -> LLMClient` — factory with `protocol` omitted; `model=None` = backend's plain chat model; OpenAI uses `openai_api_key()` unless `bearer_token` given (override never skips); llama-swap `default_options` merged over thinking-off (caller keys win), `thinking_off=False` drops it; `timeout=None` keeps factory default; unknown backend -> `ValueError`
  - `PieceRecorder` — pass as `on_delta`; `.pieces` (list, in order), `.text` (join)
  - `assert_well_formed_reasoning(text)` — strict DEC-005-6 check; no markers passes; else balanced, unnested, closed, only whitespace before/between blocks, non-empty answer after the last block
  - `answer_text(text) -> str` — blocks and stray markers removed, stripped
  - `assert_no_markers(text)`
  - `assert_secret_free_error(err, secret)` — secret absent from `str`, `repr`, `.message`, `.description`, `repr(args)`; secret must be non-empty
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓

### Step 002 — tests (2026-11-08)
- **Stale (replan 2026-11-08):** this entry describes the blocked first run (DoD-1 / DoD-2 asserted `LLMError`). The replanned step rewrites DoD-1 / DoD-2 and adds DoD-8 / DoD-9; the resumed run's test-coder rewrites this entry and replaces the old DoD-1 / DoD-2 test functions.
- `tests/integrations/test_responses_settings.py` — covers DoD-1..7 — capped reply as LLMError (OpenAI description names `max_output_tokens`, llama-swap non-empty) streamed and not; `seed: 7` warns once per client and names the setting, second client adds one; nested `json_schema` answers in `chat` / `chat_json` parse as an object holding `title` and `pages`; `temperature` + `top_p` answer with no ignored-setting warning; OpenAI reasoning model with `reasoning_effort: "medium"` answers with no marker
- Traceability:
  - DoD-1: `test_openai_token_cap_not_streamed_raises__DoD1`, `test_llamaswap_token_cap_not_streamed_raises__DoD1`
  - DoD-2: `test_openai_token_cap_streamed_raises__DoD2`, `test_llamaswap_token_cap_streamed_raises__DoD2`
  - DoD-3: `test_openai_ignored_setting_warns_once_per_client__DoD3`, `test_llamaswap_ignored_setting_warns_once_per_client__DoD3`
  - DoD-4: `test_openai_chat_json_schema_answer__DoD4`, `test_llamaswap_chat_json_schema_answer__DoD4` (xfail strict=False, DEC-005-9)
  - DoD-5: `test_openai_chat_json_schema_format__DoD5`, `test_llamaswap_chat_json_schema_format__DoD5` (xfail strict=False, DEC-005-9)
  - DoD-6: `test_openai_sampling_settings_accepted__DoD6`, `test_llamaswap_sampling_settings_accepted__DoD6`
  - DoD-7: `test_openai_reasoning_effort_medium_accepted__DoD7`
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓

### Step 003 — tests (2026-11-08)
- `tests/integrations/test_responses_errors.py` — covers DoD-1, DoD-2, DoD-3, DoD-4 — dummy-key OpenAI `chat` (non-streamed / streamed with recorder) ends with `LLMError` holding `401`, a non-empty description and no key (helper's secret-free check); closed port `http://127.0.0.1:1/v1` with 10 s limit ends with `LLMError` whose text contains `127.0.0.1:1` on both classes (OpenAI gets the dummy key); none skips
- `tests/integrations/test_responses_session.py` — covers DoD-5, DoD-6, DoD-7, DoD-8, DoD-9 — four concurrent calls (two streamed chats with own recorders, one chat, one generate) in one scope with non-empty text and per-recorder equality; same client reused in a new scope after the DoD-5 scope; `embed` non-empty list of numbers; `embed_batch` of three unrelated texts with equal non-zero length and cosine-argmax order vs single embeds; `get_models` non-empty list of mappings
- Traceability:
  - DoD-1: `test_openai_bad_credential_not_streamed__DoD1`
  - DoD-2: `test_openai_bad_credential_streamed__DoD2`
  - DoD-3: `test_openai_unreachable_server_not_streamed__DoD3`, `test_llamaswap_unreachable_server_not_streamed__DoD3`
  - DoD-4: `test_openai_unreachable_server_streamed__DoD4`, `test_llamaswap_unreachable_server_streamed__DoD4`
  - DoD-5: `test_openai_concurrent_calls_one_scope__DoD5`, `test_llamaswap_concurrent_calls_one_scope__DoD5`
  - DoD-6: `test_openai_reuse_after_scope__DoD6`, `test_llamaswap_reuse_after_scope__DoD6` (each first runs the DoD-5 scope on the same client object)
  - DoD-7: `test_openai_embed__DoD7`, `test_llamaswap_embed__DoD7`
  - DoD-8: `test_openai_embed_batch_order__DoD8`, `test_llamaswap_embed_batch_order__DoD8`
  - DoD-9: `test_openai_get_models__DoD9`, `test_llamaswap_get_models__DoD9`
- Module-local helpers: `_DUMMY_KEY`, `_CLOSED_PORT_URL` / `_CLOSED_PORT_ADDRESS` / `_CLOSED_PORT_TIMEOUT` (DEC-005-8); `_cosine` (DEC-005-7), `_embed_model`, `_is_number`; `live_support.py` unchanged
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 ✓, DoD-9 ✓

### Step 004 — tests (2026-11-08)
- `tests/integrations/test_responses_tools_extra.py` — covers DoD-1..5 — two-record tool calls (non-streamed: both ids + non-empty + no marker; streamed: both ids + containment in recorder + well-formed blocks), caller history with prior tool turns (non-empty + caller list unchanged), pydantic-built definition with coroutine tool (ran + non-empty), streamed json_schema answer (tool ran + JSON object with `record_id`/`value` + containment); llama-swap DoD-5 carries `xfail(strict=False, reason="llama.cpp json_schema grammar")` per DEC-005-9, no other xfail
- Traceability:
  - DoD-1: `test_openai_tools_two_records_not_streamed__DoD1`, `test_llamaswap_tools_two_records_not_streamed__DoD1`
  - DoD-2: `test_openai_tools_two_records_streamed__DoD2`, `test_llamaswap_tools_two_records_streamed__DoD2`
  - DoD-3: `test_openai_tools_history_with_tool_turns__DoD3`, `test_llamaswap_tools_history_with_tool_turns__DoD3`
  - DoD-4: `test_openai_tools_pydantic_async_tool__DoD4`, `test_llamaswap_tools_pydantic_async_tool__DoD4`
  - DoD-5: `test_openai_tools_schema_answer_streamed__DoD5`, `test_llamaswap_tools_schema_answer_streamed__DoD5` (xfail non-strict, DEC-005-9)
- Module-local helpers (DEC-005-11): `_RecordTool` (sync `fetch_record` / async `fetch_record_async`, records normalised ids), `_TOOL_DEFINITIONS` (nested), `_RecordQuery` (pydantic), `_history_with_tool_turns()`, `_SCHEMA_FORMAT`; imports `PieceRecorder`, `answer_text`, `assert_no_markers`, `assert_well_formed_reasoning`, `build_client` from `tests.integrations.live_support` (unchanged)
- Tolerances: tool ids recorded stripped + upper-cased; streamed containment checks `reply.strip()` in the recorder text
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓

## Ultra phase

- orient: done 2026-11-08 — typecheck: isolated (ruff) · pairable: 0 (test-only, no Source files)
- mode: serial — tree not clean and 0 pairable steps · profile: light (Python, uv, single package)
- waves: skeleton W1 001,002,003,004 (no stubs, test-only) · tests W1 001 (shared helper) · W2 002,003,004 (import helper)
- harvest: done — docs/.cache/ultra/005.responses-live-tests/harvest.md (1 report)
- skeleton: W1 done — steps 001, 002, 003, 004 (none: test-only)
- tests: done — steps 001 (W1), 002, 003, 004 (W2)
- red-gate: PASS (run 2; run 1 FAIL 001 TEST — RUF022 in live_support.py, fixed)
- code: n/a — test-only, no Source files (DEC-005-1); no coder dispatched
- verify: FAIL (run 1) — 002 SPEC (finding: llama-swap token cap not reported as LLMError); 001, 003, 004 PASS · live: 47 passed, 2 failed, 2 xfailed, 1 xpassed, 0 skipped · unit: 1089 passed
- blocked: 002 — requirement change decided by user (token-cap cut-off is not an error); awaiting /product-spec → /architect → /planner

## Files Changed

none: test-only (DEC-005-1) — no Source files in any step; the new test files are listed under `## Tests`.

## Notes & Issues

- **Step 002 — blocked, Fault SPEC (finding, verify run 1, 2026-11-08).** `test_llamaswap_token_cap_not_streamed_raises__DoD1` and `test_llamaswap_token_cap_streamed_raises__DoD2` fail deterministically (DID NOT RAISE `LLMError`; re-run once, same result). The llama-swap Responses class sends the cap as `max_output_tokens` and gets output truncated at the cap, but llama.cpp marks the reply `status: "completed"` with `incomplete_details: null` (streamed: ends with `response.completed`, no `response.incomplete`), so the cut-off is returned as a normal answer. US-024.AC-2, US-060.AC-1/AC-2 unmet on llama-swap; OpenAI variants pass. Per DEC-005-1 the test is not weakened; routed for a user decision (see `outcome.md` `## Observations`).
- **Step 002 — user decision 2026-11-08 (requirement change, not a library fix).** The user holds that a reply cut off by the caller's own token cap is correct behaviour and must not be an error: "if we set 16 tokens limit and task 'write 500 word essay', server did the right job with cut-off. It should not be a error, because calling code set the limit and should suppose that result can be cutted." This reverses US-024.AC-2 / US-060.AC-1/AC-2 (as DEC-003-12/13 apply them to a `max_output_tokens` cut-off) on both backends — so the llama-swap behaviour is the wanted one and today's OpenAI `LLMError` on a token-cap cut-off becomes the deviation. Route: `/product-spec` (amend the stories) → `/architect` (amend DEC-003-12/13 for the token-cap reason) → a planned library change for the OpenAI/llama-swap Responses classes → `/planner` revises step 002 DoD-1/2 → resume `/ultra-feature 005` (re-verify, glossary). Row stays `blocked` until then.
- Advisory: `ruff format --check` would reformat `test_responses_conversation.py` (`_has_japanese`) and `test_responses_tools_extra.py` (`_TWO_RECORDS_PROMPT`); `ruff check` passes; no format check is declared for `tests/`.
- **2026-11-08 — Step 002 block resolved; step 002 replanned, step 005 added (planner).** The requirement is now US-062 (product) and DEC-arch-13 (decision). Step 002: DoD-1 / DoD-2 rewritten (a reply cut at the app's cap returns non-empty text, no error, no notice, streamed and not), DoD-8 (`generate`) and DoD-9 (cap as a client default) added, DoD-3..7 unchanged and re-verified in the resumed run. New step 005 (`005.output-limit-tools-json.md`, new file `tests/integrations/test_responses_output_limit.py`): cut tool-conversation answer and cut checked JSON answer. US-062.AC-6 is not checked live (DEC-005-15). Row 002 set from `blocked` to `pending`; rows 002 and 005 must not reach the live verify run before feature 003 steps 006 / 007 (both steps) and feature 004 steps 005 / 006 (step 005) are `done` with PASS (DEC-005-16). The step 002 `## Skeleton` and `## Tests` entries above are stale and are rewritten by the resumed run; the `## Ultra phase` "blocked: 002" line is history for the orchestrator to supersede.

## Ultra phase — resume 002 (replanned) + 005 (new)

- orient: done 2026-11-08 — step 002 replanned by /planner (DoD-1/2 rewritten to expect text, DoD-8/9 new; row reset to pending), step 005 new · dependencies 003/006, 003/007, 004/005, 004/006 all done PASS · test-only (no Source files) · pairable: 0
- mode: serial — tree not clean, 0 pairable · profile: light
- waves: skeleton none (test-only) · tests W1 002, 005 (disjoint test files; both import live_support from 001)
- harvest: reused — docs/.cache/ultra/005.responses-live-tests/harvest.md
- tests: done — steps 002 (rebind), 005 (new)
- red-gate: PASS (run 1; collect 25 tests, ruff clean, traceability 002 DoD-1..9, 005 DoD-1..3; R3 n/a test-only)
- code: n/a — test-only (DEC-005-1)
- verify: PASS (run 1) — 002, 005 PASS; regression 001, 003, 004 PASS · unit 1390 passed · live 005 files: 64 passed, 1 xfailed, 2 xpassed (llama-swap json_schema DoD-4/5, DEC-005-9 non-strict) · whole live dir: 127 passed, 15 failed (out of scope: older test_llm_*.py — removed llama-swap default model 404, OpenAI o3-mini deprecated 404)
- glossary: done — 0 AGENTS.md created, 1 updated-in-markers (tests/integrations/; also listed 5 earlier-005 files that were missing); tests/AGENTS.md unchanged

## Tests

### Step 002 — tests (rebind 2026-11-08)
- `tests/integrations/test_responses_settings.py` — covers DoD-1, DoD-2, DoD-8, DoD-9 — capped (`max_tokens: 16`, long-answer prompt) `chat` / `generate`, streamed and not, cap per call or as `default_options`, on both backends: no `LLMError`, non-empty answer text, no marker (non-streamed), return equals joined recorder pieces + well-formed blocks (streamed), no `UserWarning`-family warning recorded (module helper `_assert_no_library_notice`, DEC-005-12). The four old `*_token_cap_*_raises__DoD1/2` tests (LLMError expectation) are replaced. No xfail on cut tests.
- `tests/integrations/test_responses_settings.py` — covers DoD-3..DoD-7 — unchanged from the first run (ignored setting once per client, json_schema in chat / chat_json with llama-swap xfail per DEC-005-9, sampling accepted, OpenAI reasoning effort "medium").
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 ✓, DoD-9 ✓

### Step 005 — tests (2026-11-08)
- `tests/integrations/test_responses_output_limit.py` — covers DoD-1, DoD-2, DoD-3 — on OpenAI (`OPENAI_MODEL`, skips without key) and llama-swap (thinking off): non-streamed `chat_with_tools` with neutral tool `query_index`, no-tool system instruction, long-answer prompt and `max_tokens: 16` call option returns non-empty answer text with no `LLMError`, no think marker, no `UserWarning`-family notice (DoD-1); same request streamed with a `PieceRecorder`: non-empty answer, stripped return contained in recorder text, recorder text well-formed, no notice (DoD-2); `chat_json` with an expected shape, no `response_format`, long-JSON prompt naming "JSON" and the cap raises `LLMError`, no xfail mark (DoD-3). Whether the tool ran is not asserted (DEC-005-13).
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓
