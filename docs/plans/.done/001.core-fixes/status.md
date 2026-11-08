# Feature 001 — core-fixes

| Step | File                       | Status  | Verifier | Date |
|------|----------------------------|---------|----------|------|
| 001  | `001.package-surface.md` | done    | PASS     | 2026-11-07 |
| 002  | `002.request-errors.md` | done    | PASS     | 2026-11-07 |
| 003  | `003.stream-parsing.md` | done    | PASS     | 2026-11-07 |
| 004  | `004.openai-client.md` | done    | PASS     | 2026-11-07 |
| 005  | `005.llamaswap-client.md` | done    | PASS     | 2026-11-07 |

## Skeleton

### Step 001 — frozen interface (2026-11-07)
- `pyproject.toml` — `[project] version = "0.2.0"` — changed (was `"0.1.5"`)
- `pyproject.toml` — `[project] dependencies = ["aiohttp>=3.9", "typing_extensions>=4.0", "pydantic>=2.0"]` — changed (pydantic moved from `[project.optional-dependencies] dev` to runtime; removed from `dev`)
- `llm/__init__.py` — `__version__ = "0.2.0"` — changed (was `"0.1.4"`)
- `llm/options.py` — `OpenAIOptions.response_format: Optional[Dict[str, Any]]` — changed (was `Optional[Dict[str, str]]`); `Any` added to the `typing` import; no key added/removed in any shape
- `llm/llm.py` — `get_llm_client(server_type: str, base_url: str, model: str, timeout: float = 600.0, bearer_token: Optional[str] = None, default_options: Optional[Dict[str, Any]] = None) -> LLMClient` — unchanged signature; the `openai` branch forwarding `bearer_token` and replacing the stale env comment is behaviour left to the coder (not touched)
- `uv.lock` — NOT regenerated at skeleton time; the coder must run `uv lock` after the pyproject change
- Caller-compile edits (out of Source-files scope): None.
- Compile gate: `compileall -q llm` + `import llm` OK (`llm.__version__ == "0.2.0"`); `ruff check` on touched llm files adds no findings beyond baseline.

### Step 002 — frozen interface (2026-11-07)
- `llm/llm_client.py` — `async def _request(self, method: str, path: str, **kw) -> Any` — changed (signature unchanged; docstring added stating the DEC-001-7 transport mapping; behaviour change left to the coder, existing behaviour preserved)
- `llm/llm_client.py` — `@asynccontextmanager async def _stream_post(self, path: str, payload: dict[str, Any]) -> AsyncIterator[aiohttp.ClientResponse]` — new (stub: raises `NotImplementedError` on enter; trailing unreachable `yield` keeps it an async generator so `async with` raises `NotImplementedError` rather than `TypeError`). The coder replaces the body, keeping decorator and signature.
- Import added: `from contextlib import asynccontextmanager` in `llm/llm_client.py`.
- Caller-compile edits (out of Source-files scope): None.
- Compile gate: `compileall -q llm` + `import llm` clean; `ruff check llm/llm_client.py` findings identical to the pre-edit set by rule and message (line numbers after line 3 shift by the inserted lines, so the baseline file's positions for this file are offset).

### Step 003 — frozen interface (2026-11-07)
- `llm/llm_client.py` — `async def _iter_sse_data(self, response: aiohttp.ClientResponse) -> AsyncIterator[str]` — changed (signature unchanged; docstring added stating the DEC-001-10 reader contract and the DEC-001-7/DEC-001-1 transport mapping; body untouched, new behaviour left to the coder)
- `llm/llm_client.py` — `async def _stream_openai_chat_response(self, response: aiohttp.ClientResponse, on_delta: Optional[Callable[[str], Awaitable[None]]] = None) -> str` — changed (signature unchanged; docstring added: error frame / early end raise `LLMError` (DEC-001-9, DEC-001-1), both reasoning and content from one piece (DEC-001-11), malformed-piece skip rule, receiver failure propagates (DEC-000-23); body untouched)
- `llm/llm_client.py` — `async def _stream_openai_tools_response(self, response: aiohttp.ClientResponse, on_delta: Optional[Callable[[str], Awaitable[None]]] = None) -> tuple[str, list[dict]]` — changed (signature unchanged; existing docstring kept and extended with the same error-frame / early-end / malformed-piece rules; DEC-000-22 / DEC-000-32 handling unchanged; body untouched)
- Optional shared private helper (error-frame / completion checks): not created; coder's choice.
- `_request` / `_stream_post` (step 002): not touched.
- Caller-compile edits (out of Source-files scope): None.
- Compile gate: `compileall -q llm` + `import llm` clean; `ruff check llm/llm_client.py` findings identical to baseline by rule+message (60 = 60).

### Step 004 — frozen interface (2026-11-07)
- `llm/openai.py` — `_OPENAI_SUPPORTED_PARAMS: Dict[str, Any] = {"temperature": None, "top_p": None, "max_tokens": None, "presence_penalty": None, "frequency_penalty": None, "seed": None, "enable_thinking": None, "reasoning_effort": None, "stop": None, "logit_bias": None, "user": None, "response_format": None}` — changed (was temperature 1.0, top_p 1.0, presence/frequency_penalty 0.0, no stop/logit_bias/user/response_format). Table data is the contract and is final (DEC-001-13, DEC-001-14); this already changes what is sent.
- `llm/openai.py` — `_OPENAI_ALIAS_MAP = {"context_length": "max_tokens"}` — unchanged.
- `llm/openai.py` — `OpenAIAPIClient.__init__(self, model: str, base_url: str = "https://api.openai.com/v1", timeout: float = 600.0, bearer_token: Optional[str] = None, default_options: Optional[OpenAIOptions] = None) -> None` — signature unchanged; class docstring no longer promises env loading; method docstring states the DEC-001-6 contract (`ValueError` on missing/empty credential, no env var named, no credential value). Env-read removal (and dropping `import os`) left to the coder.
- `llm/openai.py` — `_merge_options(self, *, options: Optional[OpenAIOptions]) -> Dict[str, Any]` — unchanged; docstring added.
- `llm/openai.py` — `async def chat(self, messages: List[LLMMessage], *, system: Optional[str] = None, options: Optional[OpenAIOptions] = None, stream: bool = False, on_delta: Optional[Callable[[str], Awaitable[None]]] = None) -> str` — signature unchanged; docstring states `_stream_post` use (DEC-001-8), trace-wrapper rule (DEC-000-48), null content -> `""` (DEC-001-12). Behaviour left to the coder.
- `llm/openai.py` — `async def chat_json(self, messages: List[LLMMessage], *, response_schema: Optional[Dict[str, Any]] = None, system: Optional[str] = None, options: Optional[OpenAIOptions] = None) -> str` — signature unchanged; docstring states setting-over-JSON-object (DEC-001-2) and null -> `""`. Note: the existing `**opt` spread after the fixed format already gives DEC-001-2 precedence now that the key is in the table.
- `llm/openai.py` — `async def _tools_stream_request(self, payload: Dict[str, Any], on_delta: Optional[Callable[[str], Awaitable[None]]] = None) -> tuple[str, list[dict]]` — signature unchanged (seam kept); docstring states `_stream_post` use. Behaviour left to the coder.
- `llm/openai.py` — `async def chat_with_tools(self, messages: List[Dict[str, str]], *, tools_definitions: List[Dict[str, Any]], tools: Dict[str, Callable[..., Any]], system: Optional[str] = None, max_loops: int = 5, options: Optional[OpenAIOptions] = None, stream: bool = False, on_delta: Optional[Callable[[str], Awaitable[None]]] = None, response_format: Optional[Dict[str, Any]] = None) -> str` — signature unchanged; docstring states parameter-over-setting (DEC-001-15, existing order) and null -> `""` for final answer and tool-call turn (DEC-001-12).
- `llm/openai.py` — `embed_batch` — unchanged (DEC-001-16).
- Caller-compile edits (out of Source-files scope): None.
- Compile gate: `.venv/bin/python -m compileall -q llm` + `import llm` clean; `ruff check llm/openai.py` findings identical by rule+message to baseline.

### Step 005 — frozen interface (2026-11-07)
- `llm/llama_swap.py` — `_LLAMA_SWAP_SUPPORTED_PARAMS: Dict[str, Any] = {"temperature": None, "top_p": None, "max_tokens": None, "presence_penalty": None, "frequency_penalty": None, "seed": None, "enable_thinking": None, "reasoning_effort": None, "stop": None, "logit_bias": None, "user": None, "response_format": None}` — changed (was temperature 1.0, top_p 1.0, presence/frequency_penalty 0.0, no `stop`/`logit_bias`/`user`/`response_format`). Frozen as data per the orchestrator brief (DEC-001-13, DEC-001-14); this makes the settings-table half of DoD 9/10 observable before the coder runs.
- `llm/llama_swap.py` — `_LLAMA_SWAP_ALIAS_MAP = {"context_length": "max_tokens"}` — unchanged (comment added).
- `llm/llama_swap.py` — `def _merge_options(self, *, options: Optional[OpenAIOptions]) -> Dict[str, Any]` — unchanged signature and logic; docstring added (DEC-000-5, DEC-000-19).
- `llm/llama_swap.py` — `async def chat(self, messages: List[LLMMessage], *, system: Optional[str] = None, options: Optional[OpenAIOptions] = None, stream: bool = False, on_delta: Optional[Callable[[str], Awaitable[None]]] = None) -> str` — changed contract only (signature unchanged); docstring states streamed branch via `self._stream_post("/chat/completions", payload)` keeping the DEC-000-48 trace wrapper, null content -> `""` (DEC-001-12). Body unchanged (inline `session.post` still present) — coder implements.
- `llm/llama_swap.py` — `async def chat_json(self, messages: List[LLMMessage], *, response_schema: Optional[Dict[str, Any]] = None, system: Optional[str] = None, options: Optional[OpenAIOptions] = None) -> str` — changed contract only; docstring states merged `response_format` setting sent instead of `{"type": "json_object"}` when set (DEC-001-2), null content -> `""`. Body unchanged — coder implements (note: today `**opt` after the literal already lets a set setting override; coder must still guarantee the precedence and null coercion).
- `llm/llama_swap.py` — `async def _tools_stream_request(self, payload: Dict[str, Any], on_delta: Optional[Callable[[str], Awaitable[None]]] = None) -> tuple[str, list[dict]]` — seam kept, signature unchanged; docstring states it opens via `_stream_post` (DEC-001-8). Body unchanged — coder implements.
- `llm/llama_swap.py` — `async def chat_with_tools(self, messages: List[Dict[str, str]], *, tools_definitions: List[Dict[str, Any]], tools: Dict[str, Callable[..., Any]], system: Optional[str] = None, max_loops: int = 5, options: Optional[OpenAIOptions] = None, stream: bool = False, on_delta: Optional[Callable[[str], Awaitable[None]]] = None, response_format: Optional[Dict[str, Any]] = None) -> str` — changed contract only; docstring states parameter wins over setting every round (DEC-001-15), null content -> `""` for final answer and assistant tool-call turn (DEC-001-12). Body unchanged — coder implements.
- `__init__`, `embed_batch`, `generate`, `list_models`, `get_models` — unchanged (no edit).
- No `NotImplementedError` stubs: no new symbols; all changes are behaviour inside existing bodies, left to the coder with existing behaviour preserved.
- Caller-compile edits (out of Source-files scope): None.
- Compile gate (isolated): `.venv/bin/python -m compileall -q llm` + `.venv/bin/python -c "import llm"` clean; `ruff check llm/llama_swap.py` findings identical to the pre-edit set by rule and message (47; positions shift by inserted docstring lines).

## Tests

### Step 001 — tests (2026-11-07)
- `tests/test_package_surface.py` — covers DoD-1, DoD-3 — pydantic is in `[project] dependencies` (read via `tomllib`); `pyproject.toml` version and `llm.__version__` are both `0.2.0` and equal
- `tests/test_llm_factory.py` — covers DoD-4, DoD-5, DoD-6 — rewritten as pytest (old script style and `os.environ` write removed): `openai` kind returns `OpenAIAPIClient` with `bearer_token` equal to the caller's token; `llamaswap` kind forwards a token and builds with `bearer_token is None` without one; unknown kind raises `ValueError` naming it; an AST self-check asserts the file imports no `os`, uses no `environ`/`putenv`/`setenv`, and every factory call passes kind/base_url/model (and `bearer_token` for `openai`)
- Coverage: DoD-1 ✓, DoD-2 [manual/live, no test], DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 [manual/live, no test]

### Step 002 — tests (2026-11-07)
- `tests/test_llm_request_errors.py` — covers DoD-1, DoD-2, DoD-3 — `_request` maps connection failure / disconnect, timeout (on open and on body read) and 2xx body-read payload error / disconnect to `LLMError`; address `base_url + path` appears in the error text (DoD-1, DoD-2)
- `tests/test_llm_request_errors.py` — covers DoD-4 — with `bearer_token` set, transport `LLMError` text (`str`, message, description) from `_request` (open and body read) and from opening `_stream_post` contains neither the credential nor "status <digit>"
- `tests/test_llm_request_errors.py` — covers DoD-5 — `_request` non-2xx (404, 500): message exactly `Request to <url> failed with status <n>`, description = reply text
- `tests/test_llm_request_errors.py` — covers DoD-6 — `_stream_post` non-2xx (400, 401, 503): error text carries address, status and reply body (read while the response is still open), never the credential; caller block does not run
- `tests/test_llm_request_errors.py` — covers DoD-7 — `_stream_post` 2xx yields the open response; one POST to `base_url + path` with `json=payload`, `Authorization: Bearer <token>` with a token and no Authorization header without one, timeout total == client timeout
- `tests/test_llm_request_errors.py` — covers DoD-8 — connection failure / disconnect / timeout opening `_stream_post` raises `LLMError` carrying the address; block does not run
- `tests/test_llm_request_errors.py` — covers DoD-9 — exceptions raised in the `_stream_post` block (custom, ValueError, TimeoutError, aiohttp.ClientConnectionError) reach the caller as the same object and type
- `tests/test_llm_request_errors.py` — covers DoD-10 — Ollama `embed_batch` through the real `_request`: transport failure raises `LLMError` with only the `/api/embed` request made; a 404 still falls back to per-text `/api/embeddings` and returns one vector per input
- Seam: fake `client.session` (request/post/get return an async CM that is also awaitable; raises on enter or yields a fake response whose body reads fail after release); tracer env vars cleared per test.
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 ✓, DoD-9 ✓, DoD-10 ✓, DoD-11 ✓ (covered by the existing unit files `tests/test_llm_embeddings.py`, `tests/test_get_models.py`, `tests/test_llm_tools_streaming.py`, `tests/test_llm_tools_reasoning_stream.py`; no new test)

### Step 003 — tests (2026-11-07)
- `tests/test_llm_sse_stream.py` — covers DoD-1..DoD-13 — real `_iter_sse_data` driven by a fake response (`content.iter_chunked` yields chosen byte reads; `url` set) for split multi-byte chars, LF/CRLF/CR and split CRLF, multi-line data join, ignored comment/event/id/retry lines, final unterminated event, transport errors (payload error, disconnect, timeout) -> `LLMError` chained, naming the URL, no "status <n>"; stubbed `_iter_sse_data` for both parsers: error frames (object message, non-object value, frame with choices) raise `LLMError` with the reason and stop delivery, early-end rule (neither signal raises; `[DONE]` only or finish reason only returns; empty / only-skipped streams raise), reasoning+content in one piece (two pieces, markers), malformed payload skipping, kept reasoning rules, receiver failures propagate unchanged (including transport-like exception types through the real reader, not wrapped).
- `tests/integrations/test_llm_stream_robustness.py` — covers DoD-15, DoD-16 — live streamed chat asking for Russian + Japanese text on llama-swap (never skipped) and OpenAI (key read by the test, passed as `bearer_token`, skip when absent): non-empty, returned == joined pieces, Cyrillic and Japanese present, no U+FFFD.
- DoD-14 — covered by the existing `tests/test_llm_tools_reasoning_stream.py`, left unchanged (not in this step's write set).
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 ✓, DoD-9 ✓, DoD-10 ✓, DoD-11 ✓, DoD-12 ✓, DoD-13 ✓, DoD-14 ✓ (existing file, unchanged), DoD-15 ✓ (live), DoD-16 ✓ (live)

### Step 004 — tests (2026-11-07)
- `tests/test_openai_client_fixes.py` — covers DoD-1 .. DoD-19 — credential header (fake `client.session`, streamed and non-streamed), `ValueError` refusal naming `bearer_token` with no env var name and no credential value, env keys ignored, factory forwarding; streamed `chat` / `_tools_stream_request` non-2xx -> `LLMError` with address, status, body, no credential; unreachable / timeout -> `LLMError`; streamed pieces in order + concatenation; null content -> `""` (chat, generate, traced chat, chat_json, tools final answer, tool-call turn in next round); no unset setting sent; stop / logit_bias / user / response_format sent from defaults or call; chat_json setting vs `{"type": "json_object"}`; tools parameter over setting on every round; merge rules (None clears, alias, alias precedence, unknown dropped, call wins); thinking -> `medium` / explicit effort / none; embed_batch user precedence and no generation settings.
- `tests/integrations/test_llm_integrations.py`, `test_llm_streaming.py`, `test_llm_thinking.py`, `test_llm_tools.py`, `test_llm_tools_pydantic.py`, `test_llm_tools_response_format.py`, `test_llm_tools_streaming.py` — covers DoD-20 — every OpenAI live test reads `OPENAI_API_KEY` / `OPEN_AI_KEY` via a module-local `_openai_api_key()` (skips when absent) and passes it as `bearer_token` (16 construction sites); llama-swap tests unchanged.
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 ✓, DoD-9 ✓, DoD-10 ✓, DoD-11 ✓, DoD-12 ✓, DoD-13 ✓, DoD-14 ✓, DoD-15 ✓, DoD-16 ✓, DoD-17 ✓, DoD-18 ✓, DoD-19 ✓, DoD-20 ✓ (live; requires-live-run without a key)

### Step 005 — tests (2026-11-07)
- `tests/test_llamaswap_client_fixes.py` — covers DoD-1, DoD-2, DoD-3, DoD-4, DoD-5 — streamed `chat` / `_tools_stream_request` against a fake `client.session` (records `post`/`request` calls, fake SSE response with `iter_chunked`): bearer header only with a credential; non-2xx -> `LLMError` with address, status, reply text and no credential; connection failure on open -> `LLMError` naming the address; pieces delivered in order and concatenated (also with tracing on).
- `tests/test_llamaswap_client_fixes.py` — covers DoD-6, DoD-7, DoD-8 — null reply content returns `""` from `chat`, `generate`, traced `chat`, `chat_json` (no shape), `chat_with_tools` final answer; a null tool-call turn enters the next round as `""` (via `_request` fake).
- `tests/test_llamaswap_client_fixes.py` — covers DoD-9, DoD-10, DoD-11, DoD-12 — request bodies (via `_request` / `_tools_stream_request` / fake session): no generation setting without app settings; stop / logit_bias / user / response_format sent as defaults or per call in `chat` (plain and streamed), `generate`, `chat_with_tools` (every round, both modes); `chat_json` setting vs `{"type": "json_object"}`; `chat_with_tools` parameter over setting on every round.
- `tests/test_llamaswap_client_fixes.py` — covers DoD-13, DoD-14 — per-call None clears a default, `context_length` -> `max_tokens` (both alias directions), unknown keys dropped, call wins over default; thinking on/off sent as-is, `reasoning_effort` never sent, unset sends neither.
- `tests/test_llamaswap_client_fixes.py` — covers DoD-15 — `embed_batch` sends call `user` over client's, client's otherwise, and no generation setting.
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 ✓, DoD-9 ✓, DoD-10 ✓, DoD-11 ✓, DoD-12 ✓, DoD-13 ✓, DoD-14 ✓, DoD-15 ✓, DoD-16 ✓ (no new test: regression run of the existing unchanged files `tests/test_llm_tools_streaming.py`, `tests/test_llm_embeddings.py`, `tests/test_get_models.py`)

## Files Changed

### Step 001 — Package surface
- `pyproject.toml` — version 0.2.0; pydantic moved to runtime dependencies (skeleton-frozen values)
- `uv.lock` — regenerated with `uv lock` (llm-client 0.2.0, pydantic as runtime dependency)
- `llm/__init__.py` — `__version__ = "0.2.0"` (skeleton-frozen value)
- `llm/options.py` — `OpenAIOptions.response_format` hinted `Dict[str, Any]` (skeleton-frozen)
- `llm/llm.py` — `get_llm_client` forwards `bearer_token` in the `openai` branch; stale env comment replaced; docstring expanded (fix-1: module headers)

### Step 002 — Request errors
- `llm/llm_client.py` — `_request` maps transport failures (connection, disconnect/payload, timeout) to `LLMError` per DEC-001-7 while keeping the DEC-000-11 status mapping; `_stream_post` body filled (session, optional bearer, total timeout, non-2xx body read + `LLMError`, transport mapping, response released via `AsyncExitStack`, caller exceptions untouched); `AsyncExitStack` added to the `contextlib` import (fix-1: module header, docstring wording)

### Step 003 — Stream parsing
- `llm/llm_client.py` — `_iter_sse_data` rewritten per DEC-001-10 (one incremental UTF-8 decoder with `replace`, LF/CRLF/bare-CR lines with CR-LF split across reads kept as one terminator, SSE comment/field rules, multi-line data joined by `\n`, final event dispatched) with the DEC-001-7 read-failure mapping around the read only; new private helper `_read_stream_frame` (error frame -> `LLMError`, malformed-frame skip, finish-reason detection); both parsers track `[DONE]`/finish reason and raise `LLMError` on early end (DEC-001-1); plain parser delivers reasoning then content from one delta (DEC-001-11); `codecs`, `re` imported (fix-1: module header, docstring wording)

### Step 004 — OpenAI client
- `llm/openai.py` — constructor uses `bearer_token` only and raises `ValueError` naming the argument (DEC-001-6), `os`/`aiohttp` imports dropped; `chat` streamed branch and `_tools_stream_request` open via `_stream_post` (trace wrapper only when enabled); null content -> `""` in `chat`, `chat_json`, `chat_with_tools` final answer and tool-call turn (DEC-001-12); response-format precedence commented as deliberate (DEC-001-2, DEC-001-15); module header added

### Step 005 — llama-swap client
- `llm/llama_swap.py` — `chat` streamed branch and `_tools_stream_request` open via `_stream_post` (trace wrapper only when enabled, DEC-000-48), `aiohttp` import dropped; null content -> `""` in `chat`, `chat_json`, `chat_with_tools` final answer and tool-call turn (DEC-001-12); response-format precedence commented as deliberate (DEC-001-2, DEC-001-15); module header added; constructor, `embed_batch` and tables unchanged

## Notes & Issues

- Step 001 tests: DoD-1, DoD-3 and DoD-6 are expected green at the red gate (values frozen by the skeleton / property of the test file itself); DoD-4 is the red one. The former Ollama factory check was dropped (no DoD clause covers it; Ollama is observation-only).
- Step 004 tests: streamed-path tests drive the real `_stream_post` / `_request` through a module-local fake `client.session` (supports `request`/`post`, `async with` and `await`, transport failure raised on open), so DoD-1/6/7/8/9 also depend on step 002's `_stream_post` and step 003's reader/parser (DoD-9 stream ends with `finish_reason` and `[DONE]`). Tests bind to `OpenAIAPIClient` per the step 004 skeleton; the `tests/AGENTS.md` line naming `OpenAIChatCompletionsClient` belongs to feature 003 and does not apply yet.
- Step 005 tests: DoD 1-5 depend on step 002's `_stream_post` and step 003's reader/parser being implemented (fake session + well-formed stream ending with `finish_reason` and `[DONE]`); at red gate they fail through the unchanged inline `session.post` path or `NotImplementedError`, not a binding error. Tests construct `LlamaSwapAPIClient` (the skeleton's class); `tests/AGENTS.md`'s `LlamaSwapChatCompletionsClient` convention belongs to feature 003 and does not exist yet. Tracing tests use `monkeypatch.setenv` for `LLM_TRACE`/`LLM_TRACE_FOLDER` per `005.context.md`; an autouse fixture clears them for every other test.


## Ultra phase

- orient: done 2026-11-07 — typecheck: isolated (ruff check; `.venv/bin/python`) · pairable: 3 (001, 004, 005)
- mode: serial — user asked to build 001-004 one after another with a commit per feature on `feature/new-api`; worktree mode's per-feature ultra branches would need merges this run may not do · profile: light (Python, one package, uv `.venv`)
- waves: W1 001,002 · W2 003,004,005
- harvest: done — docs/.cache/ultra/001.core-fixes/harvest.md (2 reports + build profile; dev extra installed; ruff baseline 272 pre-existing findings)
- skeleton: W1 done · W2 done — steps 001, 002, 003, 004, 005
- tests: done — steps 001, 002, 003, 004, 005
- red-gate: PASS (run 2; run 1 FAIL 003 DoD-1 test setup, fixed)
- code: done — steps 001, 002, 003, 004, 005
- verify: FAIL (run 1) — 001 CODE, 002 CODE, 003 CODE (missing module headers only; behaviour green; live failures environmental: default llama-swap model not served, o3-mini deprecated)
- verify: PASS (run 2)
- glossary: done — 0 AGENTS.md created, 3 updated in markers (llm/, tests/, tests/integrations/); stale out-of-marker conventions left for /finalize: llm/AGENTS.md (inline session.post streaming line; "env read is a known defect"), tests/AGENTS.md (os.environ/older-script-style lines now only true of test_llm_options.py; fake-session technique missing from the seam line), tests/integrations/AGENTS.md (key-from-library parenthetical), root AGENTS.md ("today's env read is a known defect")
- live-test environment (advisory): default llama-swap model `Qwen3.5-27B-heretic-Q5_K_M` not served on india.loc (pass with `LLAMA_SWAP_*_MODEL=Qwen3.8-27B-Uncensored-FP8.i1-Q5_K_M`); OpenAI `o3-mini` deprecated (pass with `OPENAI_THINKING_MODEL=o3`); 2 llama-swap json_schema tests fail with a server grammar error, same on pre-feature code
