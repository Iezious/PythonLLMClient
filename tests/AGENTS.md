# tests

<!-- agent-ready:start -->

Unit tests that fake the server by monkeypatching client seams.

## Contents

**Subfolders:**
- `integrations/` — live tests against real llama-swap / OpenAI servers.

**Files:**
- `__init__.py` — package marker.
- `test_get_models.py` — `get_models` on OpenAI and llama-swap with `_request` stubbed: full model records returned; an invalid response raises `LLMError`.
- `test_llm_embeddings.py` — `embed` / `embed_batch` request and response parsing.
- `test_llm_factory.py` — `get_llm_client` returns the right client per kind (older script style).
- `test_llm_options.py` — option precedence across defaults, constructor and call (older script style).
- `test_llm_tools_reasoning_stream.py` — tool-stream parser reasoning handling, driven through a stubbed `_iter_sse_data`.
- `test_llm_tools_streaming.py` — streaming `chat_with_tools` on OpenAI and llama-swap with `_tools_stream_request` stubbed.

See also: `docs/architecture/subsystems.md#base-client-llmclient`

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
