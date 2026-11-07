# tests

<!-- agent-ready:start -->

Unit tests that fake the server by monkeypatching client seams or a fake HTTP session, so client behaviour is checked without a live backend.

## Contents

**Subfolders:**
- `integrations/` — live tests against real llama-swap / OpenAI servers.

**Files:**
- `__init__.py` — package marker.
- `test_get_models.py` — `get_models` on OpenAI and llama-swap with `_request` stubbed: model records and invalid-response handling.
- `test_llamaswap_client_fixes.py` — `LlamaSwapAPIClient` request behaviour: streamed requests through the base helper (credential header, failures as `LLMError`), null reply content, and which generation settings are sent.
- `test_llm_chat_json.py` — `chat_json` on OpenAI and llama-swap with `_request` stubbed: parse-only check, non-JSON replies with and without an expected shape.
- `test_llm_embeddings.py` — `embed` / `embed_batch` request and response parsing.
- `test_llm_embeddings_checks.py` — `embed` / `embed_batch` reply checks on OpenAI and llama-swap: one vector per text, ordering by position, malformed and unreadable replies as `LLMError`.
- `test_llm_factory.py` — `get_llm_client`: per-kind client construction, credential forwarding, refusal of unknown kinds; builds clients with explicit arguments only.
- `test_llm_models_checks.py` — `list_models` / `get_models` reply checks on OpenAI and llama-swap: server order kept, malformed and unreadable replies as `LLMError`.
- `test_llm_options.py` — option precedence across defaults, constructor and call (older script style).
- `test_llm_request_errors.py` — base-client request errors: transport failures and non-2xx replies of `LLMClient._request` and `_stream_post` mapped to `LLMError`, and the effect on the Ollama embed fallback.
- `test_llm_sse_stream.py` — the shared chat-completions SSE reader and the plain / tool stream parsers: line framing, multi-byte text, error and malformed frames, early stream end, reasoning blocks.
- `test_llm_templates.py` — `recursive_format_placeholders`: stop after a pass that changes nothing, 10-pass cap, slots inside values, computed values and their errors.
- `test_llm_tools_errors.py` — `chat_with_tools` error mapping on OpenAI and llama-swap, streamed and not: unoffered tools, unreadable arguments, unserialisable results, round-limit error.
- `test_llm_tools_fragments.py` — tool-stream parser fragment grouping by position, call id and name, driven through a stubbed `_iter_sse_data`.
- `test_llm_tools_reasoning_stream.py` — tool-stream parser reasoning handling, driven through a stubbed `_iter_sse_data`.
- `test_llm_tools_streaming.py` — streaming `chat_with_tools` on OpenAI and llama-swap with `_tools_stream_request` stubbed.
- `test_llm_trace.py` — `LLMTracer` robustness and record layout: untraced on setup failure, write failures never change the call, collision suffixes, system instruction shown once; patches `llm.trace.open` / `llm.trace.datetime`.
- `test_openai_client_fixes.py` — `OpenAIAPIClient` request behaviour: app-provided credential and its refusal, streamed requests through the base helper, null reply content, which settings are sent, response-format precedence.
- `test_package_surface.py` — packaging contract: declared runtime dependencies and the version agreeing between `pyproject.toml` and `llm.__version__`.

See also: `docs/architecture/subsystems.md#base-client-llmclient`, `#tool-loop`, `#tracer-llmtracer`, `#utilities`; `docs/decisions/002.edge-fixes.md` — DEC-002-2..5 (tool errors), 6 (fragments), 7..10 (reply checks), 11..14, 16 (trace), 15 (templates)

<!-- agent-ready:end -->

## Conventions

- Fake the server by swapping one client method: `monkeypatch.setattr(client, "<seam>", fake)`; use the lowest seam needed: `_request` for non-streaming calls, `_tools_stream_request` for streamed tool rounds, `_iter_sse_data` for the SSE parser — [confirmed: user]
- Fakes assert the request they receive (method, path, payload) as well as returning canned data; copy `messages` to avoid aliasing — [inferred]
- Build clients directly with an explicit `bearer_token="sk-test"`; never set `os.environ` (two older tests do, a known defect) — [inferred]
- Write async tests as `async def test_<backend>_<behaviour>(monkeypatch)` and keep `@pytest.mark.asyncio` even under `asyncio_mode=auto`; backend tokens: `openai`, `llamaswap`, `ollama` — [confirmed: user]
- There is no `conftest.py` yet; helpers are module-local — [inferred]
- `test_llm_factory.py` and `test_llm_options.py` are in an older script style; new tests do not copy it — [inferred]
- Tests of chat-completions behaviour construct `OpenAIChatCompletionsClient` / `LlamaSwapChatCompletionsClient` (the regression suite); `OpenAIAPIClient` / `LlamaSwapAPIClient` mean the Responses protocol — [planner: 003.responses-chat] (DEC-arch-1, DEC-arch-12)
- Count library warnings inside `warnings.catch_warnings(record=True)` with `warnings.simplefilter("always")`, so Python's per-location de-duplication neither hides nor fakes the once-per-client rule — [planner: 003.responses-chat] (DEC-003-6)
