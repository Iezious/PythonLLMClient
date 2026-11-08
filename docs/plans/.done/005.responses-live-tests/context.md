# 005.responses-live-tests — feature-wide context

## Goal and boundary

This feature adds live tests against real OpenAI and llama-swap servers for the Responses classes `OpenAIAPIClient` and `LlamaSwapAPIClient`. It is test-only. User's request: "Plan feature 005 with all suggested tests — main focus on the new implementation." Revised 2026-11-08 ("update 005 according new test requirement changes"): a reply cut at the token cap the app set is now a returned result, not an error (US-062, DEC-arch-13). The feature is not roadmapped: there is no brief and the stage is `—`.

- **In scope:**
  - multi-turn history, the system instruction, non-English streaming and streamed generate (step 001);
  - a reply cut at the app's token cap in `chat` and `generate`, an ignored setting, `json_schema` answers, and sampling and effort settings (step 002);
  - a bad credential, an unreachable server, concurrent calls, reuse after the usage scope, embeddings and model records (step 003);
  - extra tool conversations (step 004, after feature 004);
  - a tool conversation whose final answer is cut at the app's token cap, and a cut checked JSON answer (step 005).
- **Out of scope:**
  - any change to `llm/`;
  - any edit to an existing test file, including the files of done steps 001, 003 and 004;
  - unit tests;
  - a live check of a length stop with no app limit (US-062.AC-6, DEC-005-15) and of a cut inside a tool-call round (DEC-005-13);
  - the chat-completions classes;
  - Ollama;
  - anything already covered live. That is: chat / generate / chat_json non-streamed; streamed chat equal to the receiver; OpenAI streamed generate; thinking on / off / explicit low / no marker when not streamed; the OpenAI non-reasoning model refusing thinking off; unknown model; list_models. These live in `test_openai_protocols.py` and `test_llamaswap_protocols.py`. 004's own live tests are also out: the two-round reasoning test, protocol parity and the json_object answer (DEC-004-13).

## Fault routing — test-only feature (DEC-005-1)

- **Library defect.** A new live test that fails because the library misbehaves is a **finding**. The verifier gives Fault SPEC (out of scope for 005), not CODE. The finding is recorded as an observation in `outcome.md` and routed to `/bug-fixer`. It is never fixed here, and the test is never weakened to pass.
- **Library step not yet landed.** The cut checks of steps 002 and 005 depend on library steps of features 003 and 004 (DEC-005-16, each step's Dependencies). A cut check that fails before those steps are `done` is **not a finding**: it is the expected state, and the live verify run of that step must not be started until they are. The orchestrator holds the step at `pending`.
- **Test defect.** A failure caused by the test itself is Fault TEST. This includes the model calling the offered tool in step 005 in a way the prompt was meant to prevent (DEC-005-13) and a cut JSON answer that happens to parse (DEC-005-14).
- **Unreachable server.** The test is reported as requires-live-run.
- **Pipeline.**
  - The skeleton records "none: test-only" under `## Skeleton`.
  - The shared helper module is test code, owned by the test-coder.
  - The red gate checks that the tests are collected and that every `[test]` item is covered. A failing test is not expected there.
  - The coder has nothing to implement. The verify run is the live run.
- **LoC budget.** Every step has 0 LoC of source change. The 50–200 budget does not apply to a test-only step, and each step is sized as one or two coherent test files.

## Stories verified live (DoD raw material)

| Story / AC | Step |
|---|---|
| US-006.AC-1, AC-2, AC-3; US-008.AC-1; US-012.AC-2; US-013.AC-1, AC-3; US-036.AC-1, AC-3 | 001 |
| US-062.AC-1, AC-2, AC-3; US-019 (story); US-061.AC-1, AC-2; US-038.AC-1; US-025.AC-4 | 002 |
| US-060.AC-1, AC-4; US-011.AC-1, AC-3; US-029.AC-1; US-004.AC-2; US-041.AC-1; US-042.AC-2, AC-5; US-045.AC-1 | 003 |
| US-030.AC-1, AC-2; US-031.AC-1; US-023.AC-1, AC-2; US-037.AC-1, AC-4 | 004 |
| US-062.AC-4, AC-5 | 005 |
| US-057.AC-1 (each chat operation on Responses), US-057.AC-2 (embeddings, model records) | 001–005 |

Not verified live: US-062.AC-6 and the early stops of US-024.AC-2 other than the app's own limit. They are covered by the unit tests of feature 003 steps 006 / 007 and feature 004 steps 005 / 006; on llama-swap US-062.AC-6 is an accepted gap (DEC-arch-13, DEC-005-15).

## Architecture these tests exercise (cite, never restate)

- `docs/architecture/interfaces.md`:
  - § Exposed: client method contract;
  - § Streaming callback;
  - § Tool loop contract;
  - § Protocol axis, OpenAI family;
  - § Exposed: factory;
  - § Exposed: warnings;
  - § Exposed: error contract;
  - § Consumed: backend APIs.
- Design: `docs/decisions/arch-responses-protocol.md` DEC-arch-3, -5, -6, -7, -8, -10, -11, -12, -13.

## Decisions

- **This feature's decisions:** `docs/decisions/005.responses-live-tests.md`, DEC-005-1 .. DEC-005-16. Steps cite them by id.
- **Still binding (expected values come from these):**
  - DEC-003-6 (warning once per setting per client);
  - DEC-003-9 (settings mapping; no-counterpart keys);
  - DEC-003-10 (answer format);
  - DEC-003-11 (input items; assistant turns typed `message`, amended 2026-11-08);
  - DEC-003-12 (non-streamed incomplete / failed reply, amended by DEC-arch-13);
  - DEC-003-13 (stream completion and failure events, amended by DEC-arch-13);
  - DEC-003-15 (live models and variables);
  - DEC-003-17, DEC-003-19 (the app set the limit; a cut at it reads like a completed reply);
  - DEC-004-16, DEC-004-17 (a cut tool round: the answer without calls, an error with them);
  - DEC-001-7 (transport failure names the address);
  - DEC-000-11 and DEC-arch-5 (non-2xx carries the address and server reply, never the credential);
  - DEC-000-24 (session per client, reusable after the scope);
  - DEC-000-37 (JSON parse check gated on the expected shape);
  - DEC-002-8 (embedding order);
  - DEC-004-6 (caller tool turns converted by shape);
  - DEC-004-13 (004's live tests, which are not repeated).

## Files touched across steps

| File | Steps |
|---|---|
| `tests/integrations/live_support.py` (new, shared helper) | written in 001; imported read-only by 002, 003, 004, 005 |
| `tests/integrations/test_responses_conversation.py` (new) | 001 |
| `tests/integrations/test_responses_settings.py` (new) | 002 |
| `tests/integrations/test_responses_errors.py`, `tests/integrations/test_responses_session.py` (new) | 003 |
| `tests/integrations/test_responses_tools_extra.py` (new) | 004 |
| `tests/integrations/test_responses_output_limit.py` (new) | 005 |
| `llm/**`, every existing file under `tests/` | none — must not be edited |

Dependency waves: {001} → {002, 003} → {004, 005}. Step 004 also needs feature `004.responses-tools` steps 001–004 delivered. Steps 002 and 005 also need the library steps named in their Dependencies (DEC-005-16):

| Step | Needs `done` with verifier PASS |
|---|---|
| 002 | feature 003 steps 006, 007 |
| 005 | feature 003 steps 006, 007; feature 004 steps 005, 006 |

## Shared helper module

`tests/integrations/live_support.py` (DEC-005-2). Its responsibilities are listed in `001.conversation.md` § Interface intent. Steps 002–005 import it as `tests.integrations.live_support` and do not change it. A helper that a later step finds missing is written module-local in that step's file, never added to the shared module.

## Library surface the tests call (facts, not expected values)

- **Factory.** `get_llm_client(server_type, base_url, model, timeout=600.0, bearer_token=None, default_options=None, protocol=None)` in `llm/llm.py`. `server_type` is `"openai"` or `"llamaswap"`. With `protocol` omitted it gives the Responses class.
- **Credential.** `OpenAIAPIClient` refuses an empty `bearer_token` with `ValueError`. llama-swap takes no token.
- **Exports.** `LLMError` (`message`, `description`; text form `message: description`) and `LLMSettingIgnoredWarning` (a `UserWarning`) are exported from `llm`.
- **Methods.** `chat(messages, *, system, options, stream, on_delta)`, `generate(prompt, *, system, options, stream, on_delta)`, `chat_json(messages, *, response_schema, system, options)` (never streams), `chat_with_tools(...)` (the signature in `docs/plans/004.responses-tools/context.md` § Cross-cutting constraints), `embed`, `embed_batch`, `get_models`.
- **chat_json expected shape.** `response_schema` is never sent; it only turns on the JSON parse check (DEC-000-37, DEC-002-7).
- **Settings.** `max_tokens`, `temperature`, `top_p`, `response_format`, `enable_thinking`, `reasoning_effort` and `seed` are accepted setting keys (DEC-003-9). The app set the output limit when `max_tokens` is in the merged settings, from the call options or the client defaults (DEC-arch-13).

## Test conventions (all steps)

- **Files.** New files only, under `tests/integrations/`. They run in the default `uv run pytest` run. There is no marker and no `conftest.py` (DEC-005-2).
- **Naming and environment.** Naming, skips, `async with` and warning counting follow DEC-005-5. Servers, models and variables follow DEC-005-3. Thinking settings follow DEC-005-4.
- **Assertions.** Assertions are structural only (DEC-005-6), with the per-test checks in DEC-005-7 and, for the cut checks, DEC-005-12. Expected values come from the stories and DECs, never from the library source.
- **Coverage.** Every `[test]` DoD item carries `(live)`. It cites the story or DEC it verifies, and each test cites its DoD id in its name.
- **Neutral tokens.** Tool names, record values and history tokens are neutral and cannot be guessed (tests/integrations/AGENTS.md § Conventions).

## Commands

- `uv run pytest tests/integrations/test_responses_conversation.py tests/integrations/test_responses_settings.py tests/integrations/test_responses_errors.py tests/integrations/test_responses_session.py tests/integrations/test_responses_tools_extra.py tests/integrations/test_responses_output_limit.py` (the new live files)
- `uv run pytest` (full run, unit + live)
- `uv run ruff check tests/integrations/`

## Vocabulary

- **Responses class**: `OpenAIAPIClient` / `LlamaSwapAPIClient`, built through the factory with `protocol` omitted.
- **Recorder**: the helper's piece receiver passed as `on_delta`. It keeps every piece in order.
- **Answer text**: a reply with any reasoning block removed (helper).
- **App's token cap**: `max_tokens: 16` (DEC-005-8) set by the test as a call option or a client default; a reply cut there is returned as text (DEC-arch-13).
- **Finding**: a live failure caused by the library, routed per § Fault routing.
