# integrations

<!-- agent-ready:start -->

Live tests against real llama-swap and OpenAI servers; they are part of the default `uv run pytest` run. OpenAI tests take the key from the test environment, pass it as `bearer_token`, and skip without it.

## Contents

**Files:**
- `__init__.py` — package marker.
- `test_llm_integrations.py` — basic live calls (generate, chat, embed, embed_batch, get_models) against llama-swap and OpenAI.
- `test_llm_stream_robustness.py` — live streamed chat in non-English scripts, checking multi-byte text arrives complete on llama-swap and OpenAI.
- `test_llm_streaming.py` — live streaming chat with a delta callback.
- `test_llm_thinking.py` — live reasoning output and think-tag well-formedness.
- `test_llm_tools.py` — live tool-calling chat.
- `test_llm_tools_pydantic.py` — live tool calling with tool definitions built from pydantic models.
- `test_llm_tools_response_format.py` — live tool calling combined with a JSON answer format.
- `test_llm_tools_streaming.py` — live streaming tool-calling chat.

See also: `docs/architecture/interfaces.md#consumed-backend-apis`

<!-- agent-ready:end -->

## Conventions

- Live tests against real servers are the primary verification; they stay in the default `uv run pytest` run and more live coverage is wanted — [confirmed: user]
- llama-swap server: `LLAMA_SWITCH_URL` → `LLAMA_SWAP_BASE_URL` → `http://india.loc:9292/v1`; `LLAMA_SWITCH_URL` stays first; llama-swap tests are never skipped — [confirmed: user]
- Models come from the environment with fallbacks: llama-swap `LLAMA_SWAP_MODEL` / `_TOOL_MODEL` / `_THINKING_MODEL` (default `Qwen3.5-27B-heretic-Q5_K_M`); OpenAI `OPENAI_MODEL` / `OPENAI_TOOL_MODEL` (default `gpt-4o-mini`) / `OPENAI_THINKING_MODEL` (default `o3-mini`) — [inferred]
- OpenAI live tests read the key from the environment (`OPENAI_API_KEY` / `OPEN_AI_KEY`) in the test, skip if absent, and pass it to the client as `bearer_token`; the library itself never reads it (today the tests rely on the library reading it, a defect tied to DEC-000-1/2) — [confirmed: user]
- Always use `async with client:` — [inferred]
- Give test tools neutral names whose results cannot be guessed (`fetch_record`, `query_index`) — [sourced: commit 2efbfd3]
- Assert only what a live model reliably produces: type, non-empty output, JSON validity, well-formed think tags, streamed text equal to returned text; never exact text — [inferred]
- Pydantic tool tests need the `dev` extra — [inferred]
