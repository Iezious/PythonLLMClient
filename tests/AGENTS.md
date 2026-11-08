# tests

<!-- agent-ready:start -->

Unit tests that fake the server by monkeypatching client seams or a fake HTTP session, so client behaviour is checked without a live backend; the pre-existing suite exercises the chat-completions classes, the `*_responses_*` files the Responses classes.

## Glossary

**Subfolders:**
- `integrations/` — live tests against real llama-swap / OpenAI servers.

**Files:**
- `__init__.py` — package marker.
- `test_get_models.py` — `get_models` on the OpenAI and llama-swap chat-completions classes with `_request` stubbed: model records and invalid-response handling.
- `test_llamaswap_client_fixes.py` — `LlamaSwapChatCompletionsClient` request behaviour: streamed requests through the base helper (credential header, failures as `LLMError`), null reply content, and which generation settings are sent.
- `test_llamaswap_responses_output_limit.py` — `LlamaSwapAPIClient` (Responses) chat-style calls, streamed and not, plus its reply reader and stream parser, when a reply is cut at the output limit the app set versus every other early end (DEC-arch-13).
- `test_llamaswap_responses_request.py` — `LlamaSwapAPIClient` (Responses) non-streamed `chat` / `generate` / `chat_json`: exact request body, settings mapping and ignored-setting warning, thinking as effort without summary, text format, reply extraction and failed / incomplete replies, non-2xx, unchanged model list / embeddings, tracing.
- `test_llamaswap_responses_stream.py` — `LlamaSwapAPIClient` (Responses) streamed `chat`: request to `/responses`, text and reasoning pieces with per-run think markers, completion and early-end rules, failure events, skipped payloads, receiver rules, tracing.
- `test_llamaswap_responses_tools.py` — `LlamaSwapAPIClient` (Responses) non-streamed `chat_with_tools` via a per-round `_request` fake or a fake session: tool flattening and strict flag, per-round body (thinking as effort, no credential header), response format on every round, typed round input items, reasoning carried over rounds, caller data untouched, 002's loop rules and failures, round limit, failed / incomplete / non-2xx rounds, warning once per client, tracing.
- `test_llamaswap_responses_tools_output_limit.py` — `LlamaSwapAPIClient` (Responses) `chat_with_tools`, streamed and not, plus its tool-round reader and tool-round stream parser, when a tool-conversation round is cut at the output limit the app set (with and without tool calls) versus no app limit and every other early end (DEC-arch-13).
- `test_llamaswap_responses_tools_stream.py` — `LlamaSwapAPIClient` (Responses) streamed `chat_with_tools` and `_stream_responses_tool_round` via per-round `_stream_post` / `_iter_sse_data` queues: streamed body parity, per-round text and think markers, call assembly incl. the llama.cpp event order, reasoning carry, failure events and early end, later-round non-2xx, receiver rules, format answers, tracing parity.
- `test_llm_chat_json.py` — `chat_json` on the OpenAI and llama-swap chat-completions classes with `_request` stubbed: parse-only check, non-JSON replies with and without an expected shape.
- `test_llm_embeddings.py` — `embed` / `embed_batch` request and response parsing.
- `test_llm_embeddings_checks.py` — `embed` / `embed_batch` reply checks on the OpenAI and llama-swap chat-completions classes: one vector per text, ordering by position, malformed and unreadable replies as `LLMError`.
- `test_llm_factory.py` — `get_llm_client`: per-kind and per-`protocol` class selection (omitted means Responses), credential forwarding, refusal of unknown kinds, unknown protocols and any protocol on Ollama; builds clients with explicit arguments only.
- `test_llm_models_checks.py` — `list_models` / `get_models` reply checks on the OpenAI and llama-swap chat-completions classes: server order kept, malformed and unreadable replies as `LLMError`.
- `test_llm_options.py` — option precedence across defaults, constructor and call (older script style).
- `test_llm_request_errors.py` — base-client request errors: transport failures and non-2xx replies of `LLMClient._request` and `_stream_post` mapped to `LLMError`, and the effect on the Ollama embed fallback.
- `test_llm_sse_stream.py` — the shared chat-completions SSE reader and the plain / tool stream parsers: line framing, multi-byte text, error and malformed frames, early stream end, reasoning blocks.
- `test_llm_templates.py` — `recursive_format_placeholders`: stop after a pass that changes nothing, 10-pass cap, slots inside values, computed values and their errors.
- `test_llm_tools_errors.py` — `chat_with_tools` error mapping on the OpenAI and llama-swap chat-completions classes, streamed and not: unoffered tools, unreadable arguments, unserialisable results, round-limit error.
- `test_llm_tools_fragments.py` — tool-stream parser fragment grouping by position, call id and name, driven through a stubbed `_iter_sse_data`.
- `test_llm_tools_reasoning_stream.py` — tool-stream parser reasoning handling, driven through a stubbed `_iter_sse_data`.
- `test_llm_tools_streaming.py` — streaming `chat_with_tools` on the OpenAI and llama-swap chat-completions classes with `_tools_stream_request` stubbed.
- `test_llm_trace.py` — `LLMTracer` robustness and record layout: untraced on setup failure, write failures never change the call, collision suffixes, system instruction shown once; patches `llm.trace.open` / `llm.trace.datetime`.
- `test_openai_client_fixes.py` — `OpenAIChatCompletionsClient` request behaviour: app-provided credential and its refusal, streamed requests through the base helper, null reply content, which settings are sent, response-format precedence.
- `test_openai_responses_output_limit.py` — `OpenAIAPIClient` (Responses) chat-style calls, streamed and not, plus its reply reader, when a reply is cut at the output limit the app set versus every other early end (DEC-arch-13).
- `test_openai_responses_request.py` — `OpenAIAPIClient` (Responses) non-streamed path: request body, reply extraction and failed / incomplete replies, settings mapping and ignored-setting warning, response-format translation, non-2xx and credential, unchanged model list / embeddings, tracing.
- `test_openai_responses_stream.py` — `OpenAIAPIClient` (Responses) streamed `chat` / `generate`: one request to `/responses`, text and reasoning pieces with per-run think markers, failure events and early end as `LLMError`, skipped payloads, receiver failures, tracing.
- `test_openai_responses_tools.py` — `OpenAIAPIClient` (Responses) non-streamed `chat_with_tools` via a per-round `_request` fake or a fake session: tool flattening, per-round body and response format, typed round input items, reasoning carried over rounds (llama.cpp and OpenAI reasoning shapes), caller tool history, result conversion, 002's loop rules and failures, round limit, failed / incomplete / non-2xx rounds, warning once per client, tracing.
- `test_openai_responses_tools_output_limit.py` — `OpenAIAPIClient` (Responses) `chat_with_tools`, streamed and not, plus its tool-round reader and tool-round stream parser, when a tool-conversation round is cut at the output limit the app set (with and without tool calls) versus no app limit and every other early end (DEC-arch-13).
- `test_openai_responses_tools_stream.py` — `OpenAIAPIClient` (Responses) streamed `chat_with_tools` and `_read_responses_tool_stream` via per-round `_stream_post` / `_iter_sse_data` queues, over OpenAI and llama.cpp event sequences: streamed body parity, per-round text and think markers, call assembly, carry items by output position, failures and early end, later-round non-2xx, receiver rules, format answers, tracing.
- `test_package_surface.py` — packaging contract: declared runtime dependencies, version 0.2.0 agreeing between `pyproject.toml` and `llm.__version__`, both protocol class pairs exported and independent, `LLMSettingIgnoredWarning` a filterable `UserWarning`.
- `test_responses_bugfix_reason_format.py` — bug-fix reproductions on both Responses classes (`OpenAIAPIClient`, `LlamaSwapAPIClient`): the JSON-schema response format sent as the request's text format on llama-swap, and the description carried by the `LLMError` for an incomplete reply or round, streamed and not, chat and tool rounds (DEC-003-10, DEC-003-21).

See also: `docs/architecture/subsystems.md#base-client-llmclient`, `#tool-loop`, `#tracer-llmtracer`, `#utilities`; `docs/decisions/002.edge-fixes.md` — DEC-002-2..5 (tool errors), 6 (fragments), 7..10 (reply checks), 11..14, 16 (trace), 15 (templates); `docs/decisions/003.responses-chat.md` — DEC-003-6 (warning once per client), 8 (factory protocol), 9..13 (Responses settings, request, reply, stream), 14 (version), 16 (test seams), 17..20 (reply cut at the app's output limit), 21 (incomplete-reply error description); `docs/decisions/004.responses-tools.md` — DEC-004-5, 1 (flattening), 6, 7, 4 (round items, reasoning carry), 8, 9 (round reading, tool-round stream parser), 10..12 (round body, loop rules, warning), 14 (tool-round test seams), 15..18 (tool round cut at the app's output limit); DEC-arch-12 (test strategy), DEC-arch-13 (app-limited cut reply)

<!-- agent-ready:end -->

## Conventions

- Fake the server by swapping one client method: `monkeypatch.setattr(client, "<seam>", fake)`; use the lowest seam needed: `_request` for non-streaming calls (one canned reply per round for tool loops), `_iter_sse_data` for the SSE parser; for streamed tool rounds, on the Responses classes a fake `_stream_post` (async context manager recording each body) plus a fake `_iter_sse_data` with one event list per round, on the chat-completions classes `_tools_stream_request` — [confirmed: user] (DEC-004-14)
- Fake-session seam: assign a fake `client.session` whose `request` / `post` return async-context-manager fake responses or raise transport errors (`_ensure_session` reuses it); a fake `content.iter_chunked` drives the stream reader (DEC-001-7, DEC-001-10)
- Fakes assert the request they receive (method, path, payload) as well as returning canned data; copy `messages` to avoid aliasing — [inferred]
- Build clients directly with an explicit `bearer_token="sk-test"`; never write `os.environ`; use `monkeypatch.setenv` / `delenv` when a test must control the environment (only the older `test_llm_options.py` writes `os.environ` directly, a known defect) — [inferred]
- Write async tests as `async def test_<backend>_<behaviour>(monkeypatch)` and keep `@pytest.mark.asyncio` even under `asyncio_mode=auto`; backend tokens: `openai`, `llamaswap`, `ollama` — [confirmed: user]
- There is no `conftest.py`. Unit-test helpers are module-local. The only shared test module is `integrations/live_support.py`, used by the live files from feature 005 on (DEC-005-2) — [planner: 005.responses-live-tests]
- `test_llm_options.py` is in an older script style (a `__main__` block, direct `os.environ` write); new tests do not copy it — [inferred]
- Tests of chat-completions behaviour construct `OpenAIChatCompletionsClient` / `LlamaSwapChatCompletionsClient` (the regression suite); `OpenAIAPIClient` / `LlamaSwapAPIClient` mean the Responses protocol — [planner: 003.responses-chat] (DEC-arch-1, DEC-arch-12)
- Count library warnings inside `warnings.catch_warnings(record=True)` with `warnings.simplefilter("always")`, so Python's per-location de-duplication neither hides nor fakes the once-per-client rule — [planner: 003.responses-chat] (DEC-003-6)
