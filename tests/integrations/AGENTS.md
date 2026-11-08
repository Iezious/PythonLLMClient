# integrations

<!-- agent-ready:start -->

Live tests against real llama-swap and OpenAI servers; they are part of the default `uv run pytest` run. The pre-existing `test_llm_*.py` files run the chat-completions classes; the `*_protocols.py`, `*_responses_tools.py` and `test_responses_*.py` files run the Responses classes (and both protocols through the factory).

## Glossary

**Subfolders:** none.

**Files:**
- `__init__.py` — package marker.
- `live_support.py` — shared helpers for the feature-005 live Responses files: server, model and key readers, the Responses client builder, a streamed-piece recorder, reasoning-marker and secret-free error checks.
- `test_llamaswap_protocols.py` — live llama-swap chat, generate, chat_json and streaming on both protocols through the factory, plus thinking on / off / explicit effort and failure cases on Responses; never skipped.
- `test_llamaswap_responses_tools.py` — live llama-swap tool conversation on Responses: two rounds with the thinking model, streamed and not, both-protocol parity through the factory, JSON-object answer after a tool round; never skipped.
- `test_llm_integrations.py` — basic live calls (generate, chat, embed, embed_batch, get_models) against llama-swap and OpenAI.
- `test_llm_stream_robustness.py` — live streamed chat in non-English scripts, checking multi-byte text arrives complete on llama-swap and OpenAI.
- `test_llm_streaming.py` — live streaming chat with a delta callback.
- `test_llm_thinking.py` — live reasoning output and think-tag well-formedness.
- `test_llm_tools.py` — live tool-calling chat.
- `test_llm_tools_pydantic.py` — live tool calling with tool definitions built from pydantic models.
- `test_llm_tools_response_format.py` — live tool calling combined with a JSON answer format.
- `test_llm_tools_streaming.py` — live streaming tool-calling chat.
- `test_openai_protocols.py` — live OpenAI chat operations except tool chat on both protocols through the factory, plus thinking on / off and server refusals on Responses; skips without the key.
- `test_openai_responses_tools.py` — live OpenAI tool conversation on Responses: two rounds on the reasoning model, streamed and not, both-protocol parity through the factory, JSON-object answer format; skips without the key.
- `test_responses_conversation.py` — live Responses conversations on both backends: builder class, multi-turn history, system instruction on chat and generate, non-English and streamed generate.
- `test_responses_errors.py` — live Responses failures: a bad OpenAI credential and an unreachable server end with a library error that never leaks the credential.
- `test_responses_output_limit.py` — live Responses checks on both backends that a tool-conversation answer cut at the app's own output limit is returned (streamed and not), and a checked JSON answer cut there ends with a library error.
- `test_responses_session.py` — live Responses session use on both backends: concurrent calls in one usage scope, reuse after the scope, embeddings and model records through the factory.
- `test_responses_settings.py` — live Responses settings and answer formats on both backends: a reply cut at the app's own output limit returns text in chat and generate (per call or client default), ignored-setting warning, json_schema answers, sampling and reasoning-effort settings.
- `test_responses_tools_extra.py` — extra live Responses tool conversations on both backends: two tool calls, history with tool turns, pydantic-built async tool, streamed json_schema answer.

See also: `docs/architecture/interfaces.md#consumed-backend-apis`; DEC-003-15 (live models and environment variables), DEC-004-13 (live tool tests: models, tools, settings), DEC-004-2 (OpenAI reasoning-model tool test in the default run), DEC-arch-12 (test strategy), DEC-arch-13 (reply cut at the app's output limit), DEC-005-1 (test-only feature; failures are findings), DEC-005-2 (shared live helper module), DEC-005-12 (chat / generate cut checks), DEC-005-13 (tool-conversation cut checks), DEC-005-14 (cut checked JSON ends with the invalid-JSON error), DEC-005-15 (no live length-stop check without an app limit), DEC-005-16 (cut tests run after the library steps)

<!-- agent-ready:end -->

## Conventions

- Live tests against real servers are the primary verification; they stay in the default `uv run pytest` run and more live coverage is wanted — [confirmed: user]
- llama-swap server: `LLAMA_SWITCH_URL` → `LLAMA_SWAP_BASE_URL` → `http://india.loc:9292/v1`; `LLAMA_SWITCH_URL` stays first; llama-swap tests are never skipped — [confirmed: user]
- Models come from the environment with fallbacks. Files from feature 003 on use the variables and defaults of DEC-003-15 (chat, reasoning, non-reasoning) and DEC-005-3 (embedding models, OpenAI address). The pre-existing `test_llm_*.py` files keep their own older defaults (llama-swap `Qwen3.8-27B-Uncensored-FP8.i1-Q5_K_M`; OpenAI `gpt-4o-mini` for plain calls, `gpt-6-luna` for reasoning) — [planner: 005.responses-live-tests]
- OpenAI live tests read the key from the environment (`OPENAI_API_KEY` / `OPEN_AI_KEY`) in the test, skip if absent, and pass it to the client as `bearer_token`; the library itself never reads it (DEC-001-6) — [confirmed: user]
- Always use `async with client:` — [inferred]
- Give test tools neutral names whose results cannot be guessed (`fetch_record`, `query_index`) — [sourced: commit 2efbfd3]
- Assert only what a live model reliably produces: type, non-empty output, JSON validity, well-formed think tags, streamed text equal to returned text; never exact text (content exceptions: DEC-005-6) — [inferred]
- Pydantic tool tests need the `dev` extra — [inferred]
- New live files share `live_support.py` (readers, Responses client builder, piece recorder, reasoning-marker and secret-free error checks) and import nothing from other test files. Older files keep their module-local helpers (DEC-005-2) — [planner: 005.responses-live-tests]
- A live failure caused by the library is a finding routed to `/bug-fixer`, never fixed or weakened in a test-only feature (DEC-005-1) — [planner: 005.responses-live-tests]
