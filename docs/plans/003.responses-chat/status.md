# Feature 003 — responses-chat

| Step | File                                 | Status  | Verifier | Date |
|------|--------------------------------------|---------|----------|------|
| 001  | `001.chat-completions-names.md`      | done    | PASS     | 2026-11-08 |
| 002  | `002.openai-responses-request.md`    | done    | PASS     | 2026-11-08 |
| 003  | `003.openai-responses-stream.md`     | done    | PASS     | 2026-11-08 |
| 004  | `004.llamaswap-responses-request.md` | done    | PASS     | 2026-11-08 |
| 005  | `005.llamaswap-responses-stream.md`  | done    | PASS     | 2026-11-08 |

## Skeleton

### Step 001 — frozen interface (2026-11-08)
- `llm/openai_chat_completions.py` (new module) — `class OpenAIChatCompletionsClient(LLMClient)` — new. Methods, signatures copied character-for-character from `llm/openai.py` `OpenAIAPIClient`:
  - `__init__(self, model: str, base_url: str = "https://api.openai.com/v1", timeout: float = 600.0, bearer_token: Optional[str] = None, default_options: Optional[OpenAIOptions] = None) -> None`
  - `_merge_options(self, *, options: Optional[OpenAIOptions]) -> Dict[str, Any]`
  - `async list_models(self) -> List[str]`; `async get_models(self) -> List[Dict[str, Any]]`
  - `async chat(self, messages: List[LLMMessage], *, system: Optional[str] = None, options: Optional[OpenAIOptions] = None, stream: bool = False, on_delta: Optional[Callable[[str], Awaitable[None]]] = None) -> str`
  - `async generate(self, prompt: str, *, system: Optional[str] = None, options: Optional[OpenAIOptions] = None, stream: bool = False, on_delta: Optional[Callable[[str], Awaitable[None]]] = None) -> str`
  - `async chat_json(self, messages: List[LLMMessage], *, response_schema: Optional[Dict[str, Any]] = None, system: Optional[str] = None, options: Optional[OpenAIOptions] = None) -> str`
  - `async _tools_stream_request(self, payload: Dict[str, Any], on_delta: Optional[Callable[[str], Awaitable[None]]] = None) -> tuple[str, list[dict]]` (test seam)
  - `async chat_with_tools(self, messages: List[Dict[str, str]], *, tools_definitions: List[Dict[str, Any]], tools: Dict[str, Callable[..., Any]], system: Optional[str] = None, max_loops: int = 5, options: Optional[OpenAIOptions] = None, stream: bool = False, on_delta: Optional[Callable[[str], Awaitable[None]]] = None, response_format: Optional[Dict[str, Any]] = None) -> str`
  - `async embed_batch(self, texts: List[str], *, options: Optional[OpenAIOptions] = None) -> List[List[float]]`
- `llm/llama_swap_chat_completions.py` (new module) — `class LlamaSwapChatCompletionsClient(LLMClient)` — new. Same method set and signatures as `llm/llama_swap.py` `LlamaSwapAPIClient`; differences from the OpenAI list: `__init__(self, model: str, base_url: str = "http://localhost:8080/v1", timeout: float = 600.0, bearer_token: Optional[str] = None, default_options: Optional[Dict[str, Any]] = None) -> None` (no credential check).
- **Stub choice for the two modules:** bodies raise `NotImplementedError` (not the verbatim copy). Reason: the skeleton freezes signatures only; raising keeps the moved regression suite and DoD-9 true-red against the new names until the coder lands the copy, and keeps DEC-003-7's "copy of openai.py / llama_swap.py with only names, docstrings, header changed" (DoD-14 diff check) a single coder action. The stubs import only what the signatures use; the coder replaces each module wholesale with the renamed copy of its source module (tables `_OPENAI_*` / `_LLAMA_SWAP_*` and module-private helpers keep their names). Class and method docstrings are copied from the sources; the stub module headers are provisional and the coder writes the final header. `llm/openai.py` / `llm/llama_swap.py` untouched.
- `llm/exceptions.py` — `class LLMSettingIgnoredWarning(UserWarning)` — new; complete as declared (a category with no behaviour of its own, nothing to stub).
- `llm/llm.py` — `get_llm_client(server_type: str, base_url: str, model: str, timeout: float = 600.0, bearer_token: Optional[str] = None, default_options: Optional[Dict[str, Any]] = None, protocol: str | None = None) -> LLMClient` — changed (was the same without `protocol`). Annotated `str | None` (same type as `Optional[str]`) so ruff adds no UP045 finding vs baseline. Stub behaviour: `protocol=None` keeps today's behaviour exactly; any non-`None` protocol raises `NotImplementedError` (coder implements DEC-003-8 ordering, including passing `protocol=None` explicitly = not passed). Docstring states the full contract. `__all__` gains `"OpenAIChatCompletionsClient"`, `"LlamaSwapChatCompletionsClient"`; imports added from the two new modules.
- `llm/__init__.py` — imports + `__all__` gain `LLMSettingIgnoredWarning`, `OpenAIChatCompletionsClient`, `LlamaSwapChatCompletionsClient`. `__version__` left at `"0.2.0"`: the bump to `0.3.0` (with `pyproject.toml`, `uv.lock`) is behaviour for the coder (DoD-10, DEC-003-14).
- Not touched (coder): `pyproject.toml`, `uv.lock`, `README.md`; tests not touched.
- Gate: `.venv/bin/python -m compileall -q llm && .venv/bin/python -c "import llm"` clean; ruff: no new finding in `llm/__init__.py`, `llm/llm.py`, `llm/exceptions.py` vs baseline; new modules report only UP006/UP035/UP045, each count at or below its source module's baseline count (inherited from the copied signatures, as DEC-003-7 requires).
- Caller-compile edits (out of Source-files scope): None.


### Step 002 — frozen interface (2026-11-08)
- `llm/openai.py` — `OpenAIAPIClient.__init__(self, model: str, base_url: str = "https://api.openai.com/v1", timeout: float = 600.0, bearer_token: Optional[str] = None, default_options: Optional[OpenAIOptions] = None) -> None` — unchanged signature; now also sets `self._warned_settings: set[str] = set()` (per-instance warned-settings record, DEC-003-6; implemented, it is a plain empty-set declaration).
- `llm/openai.py` — `_OPENAI_NO_COUNTERPART_PARAMS: frozenset[str] = frozenset({"presence_penalty", "frequency_penalty", "seed", "stop", "logit_bias"})` — new module constant (DEC-003-9). `_OPENAI_SUPPORTED_PARAMS` / `_OPENAI_ALIAS_MAP` unchanged (already match DEC-003-9).
- `llm/openai.py` — `_responses_text_format(response_format: Any) -> Any` — new module-level helper (DEC-003-10); body raises `NotImplementedError`.
- `llm/openai.py` — `_responses_reply_text(response: Any) -> str` — new module-level helper (DEC-003-12; raises `LLMError` for failed/incomplete); body raises `NotImplementedError`.
- `llm/openai.py` — `OpenAIAPIClient._build_responses_payload(self, messages: list[LLMMessage], *, system: str | None, settings: dict[str, Any], stream: bool) -> dict[str, Any]` — new method, also used by step 003 (DEC-003-11; `settings` = `_merge_options` output; builds `input` incl. the system item); body raises `NotImplementedError`.
- `llm/openai.py` — `_merge_options(self, *, options: Optional[OpenAIOptions]) -> Dict[str, Any]`, `chat(...)`, `generate(...)`, `chat_json(...)`, `chat_with_tools(...)` — signatures unchanged; docstrings rewritten to the Responses contracts (DEC-003-9/10/1/6, DEC-003-11/12/16, DEC-003-4). Class docstring and module header rewritten for Responses.
- **Bodies left as chat-completions code (coder replaces):** `_merge_options`, `chat`, `generate`, `chat_json`, `chat_with_tools`, `_tools_stream_request` and the tool-loop helpers `_offered_tool_name`, `_read_tool_arguments`, `_tool_result_text` are still the old chat-completions code, so new tests are true-red against them. Coder removes `_tools_stream_request`, the tool loop and its helpers, and the `/chat/completions` streaming call (step 002 "Removed from this module"); `chat_with_tools` becomes the DEC-003-4 refusal. The `LLMSettingIgnoredWarning` / `warnings` imports are not added yet (would be unused, F401) — coder adds them.
- **Untouched:** `list_models`, `get_models`, `embed_batch`, `_fetch_reply`, `_reply_text`, `_reply_entries`, `_model_entries`, `_embedding_vectors`.
- **Test seams (DEC-003-16):** non-streamed `chat` / `generate` / `chat_json` → patch `OpenAIAPIClient._request`; expected call `_request("POST", "/responses", json=payload)`. Streamed path (`_stream_post("/responses", payload)` + `_iter_sse_data`) is step 003. Warning tests: `warnings.catch_warnings(record=True)` + `simplefilter("always")`.
- Gate: `.venv/bin/python -m compileall -q llm && .venv/bin/python -c "import llm"` clean; `ruff check llm/openai.py` = 43 findings (I001 1, UP006 21, UP035 3, UP045 18), identical per-rule counts to the baseline (new signatures use builtin generics / `X | None`).
- Caller-compile edits (out of Source-files scope): None.


### Step 003 — frozen interface (2026-11-08)
- `llm/openai.py` — `async OpenAIAPIClient._read_responses_stream(self, response: aiohttp.ClientResponse, on_delta: Callable[[str], Awaitable[None]] | None = None) -> str` — new private method (the Responses stream parser, DEC-003-13 / DEC-arch-11 minus tool events; this class's own copy, DEC-arch-2). Body raises `NotImplementedError("_read_responses_stream")`. Docstring states the full contract: reads payloads only via `self._iter_sse_data(response)`; JSON-decodes each; skips non-JSON (incl. `[DONE]`), non-object, missing/non-string `type`, unknown types, empty/missing deltas (no receiver call); `response.output_text.delta` → text; `response.reasoning_text.delta` / `response.reasoning_summary_text.delta` → reasoning; per-run `<think>` / `</think>` markers (DEC-000-20), open block closed by a separate `</think>` piece at `response.completed`; `response.completed` returns at once; `response.failed` / `response.incomplete` / `error` → `LLMError` with the server's reason, nothing delivered after; end without `response.completed` (even after `[DONE]`) → `LLMError`; receiver awaited when awaitable, its exceptions propagate unchanged; return == concatenation of delivered pieces.
- `llm/openai.py` — `OpenAIAPIClient.chat(...)` — signature unchanged; docstring updated for the streamed path (`_build_responses_payload(..., stream=True)` → `self._stream_post("/responses", payload)` → `_read_responses_stream` inside the `"chat"` trace session, `_tracing_delta` wrapper only when `self._tracer.enabled`, log + return the text; streamed failure modes added to Raises). Body still the old chat-completions code (step 002/003 coder replaces it). `generate` unchanged (goes through `chat`).
- Step 002 stubs/helpers (`_build_responses_payload`, `_responses_text_format`, `_responses_reply_text`, `_merge_options`) — untouched.
- **Test seam (DEC-003-16), exactly:**
  - Parser unit tests (DoD 2-7, 9): `monkeypatch.setattr(client, "_iter_sse_data", fake)` where `async def fake(_response): for p in payloads: yield p` (payload strings, i.e. SSE data without `data:`), then `await client._read_responses_stream(None, recorder)` (or `None` receiver).
  - `chat(stream=True)` / `generate(stream=True)` tests (DoD 1, 10, 11): replace `client._stream_post` with a fake `@asynccontextmanager` `(path, payload)` that records `path` and a copy of `payload` and yields a placeholder (e.g. `None`), together with the `_iter_sse_data` fake above. Expected: exactly one call, `path == "/responses"`, payload equals the non-streamed body with `stream` true (compare against the `json=` captured from a patched `_request` non-streamed call).
  - Non-2xx (DoD 8): give the client a fake session (`client.session = _FakeSession(...)`, per `tests/test_openai_client_fixes.py` `_error_session`) so the real `_stream_post` runs; do not patch `_stream_post` there.
  - Tracing (DoD 11): `monkeypatch.setenv("LLM_TRACE", "1")`, `monkeypatch.setenv("LLM_TRACE_FOLDER", str(tmp_path))` before constructing the client.
- Gate: `.venv/bin/python -m compileall -q llm && .venv/bin/python -c "import llm"` clean; `ruff check llm/openai.py` = 43 findings (I001 1, UP006 21, UP035 3, UP045 18), per-rule counts identical to baseline.
- Caller-compile edits (out of Source-files scope): None.

### Step 004 — frozen interface (2026-11-08)
- `llm/llama_swap.py` — module header rewritten (Responses client; provisional, coder may refine).
- `llm/llama_swap.py` — `_LLAMA_SWAP_SUPPORTED_PARAMS` / `_LLAMA_SWAP_ALIAS_MAP` — unchanged keys/values (already match DEC-003-9); comment updated.
- `llm/llama_swap.py` — `_LLAMA_SWAP_IGNORED_SETTINGS: tuple[str, ...] = ("presence_penalty", "frequency_penalty", "seed", "stop", "logit_bias")` — new (data table, complete).
- `llm/llama_swap.py` — `def _responses_text_format(response_format: Any) -> Any` — new, module-level, stub (DEC-003-10; returns the value placed under `text.format`).
- `llm/llama_swap.py` — `def _responses_reply_text(reply: Any) -> str` — new, module-level, stub (DEC-003-12; raises `LLMError` for failed/incomplete).
- `llm/llama_swap.py` — `LlamaSwapAPIClient.__init__(self, model: str, base_url: str = "http://localhost:8080/v1", timeout: float = 600.0, bearer_token: Optional[str] = None, default_options: Optional[Dict[str, Any]] = None) -> None` — signature unchanged; now also sets `self._warned_settings: set[str] = set()` (DEC-003-6; complete).
- `llm/llama_swap.py` — `LlamaSwapAPIClient._merge_options(self, *, options: Optional[OpenAIOptions]) -> Dict[str, Any]` — signature unchanged, contract changed to Responses wire fields (DEC-003-9/10/1/2, warning per DEC-003-6); stub.
- `llm/llama_swap.py` — `LlamaSwapAPIClient._responses_payload(self, messages: list[LLMMessage], *, system: str | None, settings: dict[str, Any], stream: bool) -> dict[str, Any]` — new method, stub (DEC-003-11; step 005 reuses it with `stream=True`).
- `llm/llama_swap.py` — `async chat(self, messages: List[LLMMessage], *, system: Optional[str] = None, options: Optional[OpenAIOptions] = None, stream: bool = False, on_delta: Optional[Callable[[str], Awaitable[None]]] = None) -> str` — signature unchanged; stub (whole body raises, incl. streamed path, which is step 005's).
- `llm/llama_swap.py` — `async generate(self, prompt: str, *, system: Optional[str] = None, options: Optional[OpenAIOptions] = None, stream: bool = False, on_delta: Optional[Callable[[str], Awaitable[None]]] = None) -> str` — unchanged; body kept (already a one-message `chat`, DEC-000-8); docstring added.
- `llm/llama_swap.py` — `async chat_json(self, messages: List[LLMMessage], *, response_schema: Optional[Dict[str, Any]] = None, system: Optional[str] = None, options: Optional[OpenAIOptions] = None) -> str` — signature unchanged; stub.
- `llm/llama_swap.py` — `async chat_with_tools(self, messages: List[Dict[str, str]], *, tools_definitions: List[Dict[str, Any]], tools: Dict[str, Callable[..., Any]], system: Optional[str] = None, max_loops: int = 5, options: Optional[OpenAIOptions] = None, stream: bool = False, on_delta: Optional[Callable[[str], Awaitable[None]]] = None, response_format: Optional[Dict[str, Any]] = None) -> str` — signature unchanged; stub raises a generic `NotImplementedError("step 004: ...")` that does NOT name `LlamaSwapChatCompletionsClient` (coder writes the DEC-003-4 message; DoD-14 message match stays red).
- `list_models`, `get_models`, `embed_batch`, `_fetch_reply`, `_reply_text`, `_reply_entries`, `_model_entries`, `_embedding_vectors` — untouched.
- Left in place for the coder to remove (chat-completions leftovers, now unused): `_offered_tool_name`, `_read_tool_arguments`, `_tool_result_text`, method `_tools_stream_request`. `import inspect` removed (unused after stubbing; step 005 re-adds it for `_tracing_delta`). Coder must add `import warnings` and `from llm.exceptions import LLMSettingIgnoredWarning` (not added now: they would be unused imports, F401).
- Test seams (DEC-003-16): non-streamed `chat` / `generate` / `chat_json` patch `client._request` and expect exactly `_request("POST", "/responses", json=payload)`; non-2xx / credential-header cases (DoD 15, 16) use a fake session or `_request` per the 001 context seams; `list_models` / `get_models` / `embed_batch` use `_request` (`GET /models`, `POST /embeddings`). Streamed path (`_stream_post` + `_iter_sse_data`) is step 005's. Helpers `_responses_text_format`, `_responses_reply_text` are module-level and callable directly.
- Gate: `.venv/bin/python -m compileall -q llm && .venv/bin/python -c "import llm"` clean; ruff `llm/llama_swap.py`: 42 findings (I001 1, UP035 3, UP006 20, UP045 18) vs baseline 44 (I001 1, UP035 3, UP006 22, UP045 18) — no new code; new signatures use `dict` / `X | None`.
- Caller-compile edits (out of Source-files scope): None.

### Step 005 — frozen interface (2026-11-08)
- `llm/llama_swap.py` — `async LlamaSwapAPIClient._stream_responses_response(self, response: aiohttp.ClientResponse, on_delta: Callable[[str], Awaitable[None]] | None = None) -> str` — new private method, stub (raises `NotImplementedError("step 005: ...")`); docstring states the DEC-003-13 contract (event dispatch, per-run `<think>`/`</think>` markers, closing piece at completion, failed/incomplete/error → `LLMError` with server reason, missing `response.completed` → `LLMError` even after `[DONE]`, return == concatenation of delivered pieces). Placed directly after `_responses_payload`. Own copy (DEC-arch-2); does not import from `llm/openai.py`.
- `llm/llama_swap.py` — `async chat(...)` — signature unchanged (step 004 freeze); docstring only: streamed path = `_responses_payload(..., stream=True)` → `async with self._stream_post("/responses", payload)` → `_stream_responses_response(response, _tracing_delta if self._tracer.enabled else on_delta)` inside the trace session → `log_response` → return; Raises extended with the stream failures. Body still the step 004 stub.
- `generate` — untouched (already a one-message `chat`, DEC-000-8).
- Coder notes: re-add `import inspect` for `_tracing_delta` (removed at step 004 as unused). Step 004 stubs/helpers untouched.
- Test seams (DEC-003-16), exactly:
  - Parser unit tests (DoD 2-8, 10): `monkeypatch.setattr(client, "_iter_sse_data", fake)` where `async def fake(_response): for p in payloads: yield p`, then `await client._stream_responses_response(None, recorder)` (or `None` receiver) — `response=None` is fine because the parser only hands it to `_iter_sse_data`.
  - `chat(stream=True)` / `generate(stream=True)` / tracing tests (DoD 1, 11, 12): replace `client._stream_post` with a fake `@asynccontextmanager` capturing `(path, payload)` and yielding a placeholder (e.g. `None`), together with the `_iter_sse_data` fake above. Expected path `"/responses"`; payload equals the non-streamed body (from `_request` capture) except `stream: True`.
  - Non-2xx streamed (DoD 9): leave `_stream_post` / `_iter_sse_data` real and install a fake session (`client.session = _FakeSession(...)`, the `_error_session(404, body)` builder pattern from `tests/test_openai_client_fixes.py`).
- Gate: `.venv/bin/python -m compileall -q llm && .venv/bin/python -c "import llm"` clean; ruff `llm/llama_swap.py`: 42 findings (I001 1, UP035 3, UP006 20, UP045 18) vs baseline 44 — no new finding (new signature uses `X | None`).
- Caller-compile edits (out of Source-files scope): None.

## Tests

### Step 001 — tests (2026-11-08)
- `tests/test_package_surface.py` — covers DoD-1, DoD-2, DoD-10 — the two chat-completions classes and `LLMSettingIgnoredWarning` are importable from `llm` and in `llm.__all__` (old names stay exported); the classes are `LLMClient` subclasses distinct from `OpenAIAPIClient` / `LlamaSwapAPIClient`, with no subclassing in either direction; the warning is a `UserWarning` and a category-only filter silences it or raises it as an error; `pyproject.toml` version == `llm.__version__` == `0.3.0` (the earlier pydantic runtime-dependency test is kept)
- `tests/test_llm_factory.py` — covers DoD-3..DoD-8 (+ DoD-11 for the moved tests) — no protocol, explicit `None` and `"responses"` give exactly `OpenAIAPIClient` / `LlamaSwapAPIClient`; `"chat_completions"` gives the chat-completions classes carrying model, base_url, timeout, bearer_token, default_options; the class is the same on an OpenAI-looking and a local address for every accepted protocol; unknown protocols (`"Responses"`, `"CHAT_COMPLETIONS"`, `"chat-completions"`, `"completions"`, `" responses"`) raise `ValueError` naming the value; Ollama with any protocol raises `ValueError` mentioning protocol + Ollama, and builds `OllamaAPIClient` without one / with `None`; an unknown server type is refused (naming it) whatever the protocol; OpenAI without a credential (`None` / `""`) is refused on every protocol; llama-swap builds with and without a credential on every protocol. The existing four tests gained `protocol="chat_completions"` and the new names. AST meta-test kept (no `os`, positional kind/base_url/model, `bearer_token=` on every `"openai"` call)
- `tests/test_openai_client_fixes.py` — covers DoD-9, DoD-11 — moved to `OpenAIChatCompletionsClient` (factory calls gain `protocol="chat_completions"`); new `__s001_DoD9` tests: non-streamed `chat` sends exactly one `POST /chat/completions` (`_request` seam), and streamed `chat` opens exactly one `POST <base>/chat/completions` (fake session)
- `tests/test_llamaswap_client_fixes.py` — covers DoD-9, DoD-11 — moved to `LlamaSwapChatCompletionsClient`; new `__s001_DoD9` tests, the same two checks for llama-swap
- Moved (rename only, DoD-11; no assertion change): `tests/test_get_models.py`, `tests/test_llm_embeddings.py`, `tests/test_llm_embeddings_checks.py`, `tests/test_llm_models_checks.py`, `tests/test_llm_chat_json.py`, `tests/test_llm_sse_stream.py`, `tests/test_llm_tools_errors.py`, `tests/test_llm_tools_fragments.py`, `tests/test_llm_tools_reasoning_stream.py`, `tests/test_llm_tools_streaming.py`, `tests/test_llm_trace.py`
- Moved live (rename only, DoD-12; no model / env var change): `tests/integrations/test_llm_integrations.py`, `test_llm_streaming.py`, `test_llm_thinking.py`, `test_llm_tools.py`, `test_llm_tools_pydantic.py`, `test_llm_tools_response_format.py`, `test_llm_tools_streaming.py`, `test_llm_stream_robustness.py` (no live test calls the factory, so all use direct construction)
- No edit needed: `tests/test_llm_request_errors.py` (Ollama and base client only; names neither class and never calls the factory for openai/llamaswap)
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 ✓, DoD-9 ✓, DoD-10 ✓, DoD-11 ✓ (moved unit suite), DoD-12 ✓ (live, moved integration suite), DoD-13 [manual/live, no test], DoD-14 [manual/live, no test]

#### Notes
- DoD-9 tests carry the suffix `__s001_DoD9` because the two client-fixes files already use bare `__DoD<n>` ids from feature 001 steps 004/005.
- DoD-4 reads the time limit as `client.timeout` (the base client's documented attribute, used as `ClientTimeout(total=self.timeout)` in 001.core-fixes context).
- Long lines created by the rename were wrapped to stay at or under 120 characters; reflowed module docstrings in `test_llm_embeddings_checks.py`, `test_llm_models_checks.py`, `test_llm_chat_json.py`. Import order kept (the `_chat_completions` modules sort where the old modules sat). Ruff not run (test-coder runs no commands).

### Step 002 — tests (2026-11-08)
- `tests/test_openai_responses_request.py` — covers DoD-1, DoD-2, DoD-3 — exact fixed body keys of `POST /responses`; role/content input items in order, system item at head only when truthy, caller list unchanged; generate's system + single user item.
- `tests/test_openai_responses_request.py` — covers DoD-4, DoD-5 — concatenation of output_text parts of message items (reasoning/refusal ignored), empty text for replies without text; failed → LLMError with server message, incomplete → LLMError with reason.
- `tests/test_openai_responses_request.py` — covers DoD-6, DoD-7, DoD-8 — temperature/top_p/user by name, max_tokens/context_length → max_output_tokens, None removes default, call wins, unknown key dropped silently; no-counterpart keys absent; LLMSettingIgnoredWarning once per setting per client (defaults and call alike, second instance warns again, None not warned, generate/chat_json warn too).
- `tests/test_openai_responses_request.py` — covers DoD-9, DoD-10, DoD-11 — exact `reasoning` object per thinking on/off/unset with and without explicit effort; no top-level enable_thinking / reasoning_effort.
- `tests/test_openai_responses_request.py` — covers DoD-12, DoD-13 — `text: {"format": …}` translation (json_object, nested json_schema flattened, others unchanged); chat_json JSON-object default or translated setting, stream false, parse check with response_schema.
- `tests/test_openai_responses_request.py` — covers DoD-14 — chat_with_tools raises NotImplementedError naming OpenAIChatCompletionsClient, no `_request` / `_stream_post` / session call, no warning (streamed and not).
- `tests/test_openai_responses_request.py` — covers DoD-15, DoD-16 — non-2xx via fake session: LLMError with `<base>/responses` and server message, no credential, one call to /responses only; ValueError without credential; bearer header sent.
- `tests/test_openai_responses_request.py` — covers DoD-17, DoD-18 — list_models/get_models/embed_batch requests and results equal to OpenAIChatCompletionsClient's for the same canned replies, no warning; traced chat returns the text and the `.log` record contains `/responses` and the text.
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 ✓, DoD-9 ✓, DoD-10 ✓, DoD-11 ✓, DoD-12 ✓, DoD-13 ✓, DoD-14 ✓, DoD-15 ✓, DoD-16 ✓, DoD-17 ✓, DoD-18 ✓

#### Notes
- "No warning" assertions (DoD-6, DoD-14, DoD-17) count only `LLMSettingIgnoredWarning`, so unrelated third-party warnings cannot cause false failures.
- DoD-17 is red until step 001's `OpenAIChatCompletionsClient` copy lands (its stub raises NotImplementedError).
- Ruff not run by the test coder (no command execution); written to py311 style, sorted imports, ≤120 columns — verifier should confirm with `.venv/bin/python -m ruff check tests/test_openai_responses_request.py`.

### Step 003 — tests (2026-11-08)
- `tests/test_openai_responses_stream.py` — covers DoD-1..DoD-11 — streamed `chat` posts once to `/responses` with the non-streamed body plus `stream: true`; parser `_read_responses_stream` (via `_iter_sse_data` fake): text pieces in order, reasoning-text / summary pieces with per-run `<think>` / `</think>` markers, closing piece at completion, no markers without reasoning, failed / incomplete / error events raise `LLMError` with the server's reason (string or JSON-text fallback) and deliver nothing after, missing `response.completed` (incl. after `[DONE]`, empty stream) raises, nothing after completion delivered, junk / unknown / `*.done` / empty-delta payloads skipped; non-2xx streamed reply via fake session carries address + reply, not the credential; receiver exception propagates as the same object, async receiver awaited before the next read; no-receiver return, streamed `generate` equals one-message `chat` (pieces, return, request); tracing on gives the same pieces and return.
- `tests/integrations/test_openai_protocols.py` — covers DoD-12..DoD-15 (live, skip without `OPENAI_API_KEY` / `OPEN_AI_KEY`) — factory-built clients on both protocols: `chat`, `generate`, `chat_json` (parses as JSON), streamed `chat` / `generate` equal to receiver text; Responses with `OPENAI_REASONING_MODEL`: thinking on streams a well-formed reasoning block before the answer, thinking off has no block, thinking off + effort `low` succeeds, non-streamed thinking on has no marker; `OPENAI_NON_REASONING_MODEL` thinking off → `LLMError` with reason, no key; unknown model → `LLMError` with reason, no key, streamed and not; `list_models` non-empty list of strings.
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 ✓, DoD-9 ✓, DoD-10 ✓, DoD-11 ✓, DoD-12 ✓ (live), DoD-13 ✓ (live), DoD-14 ✓ (live), DoD-15 ✓ (live)

#### Notes
- Unit file has an autouse fixture clearing `LLM_TRACE` / `LLM_TRACE_FOLDER` so a developer's environment cannot turn tracing on; the DoD-11 tests set them with `monkeypatch.setenv`.
- Exact piece lists assume DEC-003-13's "prefixed" means the marker is concatenated onto the same piece (e.g. `"<think>R"`, `"</think>T"`), while the completion close is a separate `"</think>"` piece.
- Ruff not run (no command execution in this role); files written to py311 style, sorted imports, lines at most 120.

### Step 004 — tests (2026-11-08)
- `tests/test_llamaswap_responses_request.py` — covers DoD-1 .. DoD-18 — `LlamaSwapAPIClient` on Responses with `_request` patched (DoD-1..14, 17, 18) or a fake `client.session` (DoD-14 extra guard, 15, 16): exact `/responses` body keys and input items, `generate` items, output-text extraction and empty replies, failed / incomplete `LLMError`, settings mapping (names, `max_output_tokens`, alias, None removal, per-call win, unknown key silent), no-counterpart settings dropped and warned once per setting per instance (`catch_warnings(record=True)` + `simplefilter("always")`), reasoning effort medium / none / explicit and never a summary or top-level key, `text.format` translation on `chat` and `chat_json`, `chat_json` shape check, `chat_with_tools` refusal naming `LlamaSwapChatCompletionsClient` with no request / warning, 404 error text (address + server reply, no credential, one call to `/responses`), bearer header with / without credential, `list_models` / `get_models` / `embed_batch` equal to `LlamaSwapChatCompletionsClient` for the same canned replies with no warning, traced chat record showing `/responses` and the text.
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 ✓, DoD-9 ✓, DoD-10 ✓, DoD-11 ✓, DoD-12 ✓, DoD-13 ✓, DoD-14 ✓, DoD-15 ✓, DoD-16 ✓, DoD-17 ✓, DoD-18 ✓

#### Notes
- DoD-8 also asserts two DEC-003-6 points beyond the DoD wording: a no-counterpart default cleared per call with `None` is not warned, and `generate` / `chat_json` warn too.
- DoD-5 also asserts DEC-003-12's fallback: a failed reply whose `error` is not a mapping still carries that value in the error text.
- DoD-17 sets the same no-counterpart defaults (`seed`, `stop`) on both clients so a stray warning from these operations would be caught; the comparison is full `(method, path, kwargs)` equality of the `_request` calls.
- DoD-18 reads the record as the single `*.log` under the `tmp_path` trace folder (layout per `tests/test_llm_trace.py`).

### Step 005 — tests (2026-11-08)
- `tests/test_llamaswap_responses_stream.py` — covers DoD-1..DoD-12 — streamed request to `/responses` (body = non-streamed body with `stream` true); text / reasoning / summary pieces with per-run `<think>`/`</think>` markers and the closing piece at completion; the observed llama-swap sequence; failed / incomplete / error events (reason incl. JSON-text fallbacks, nothing delivered after); completion vs early end and `[DONE]`; skipped payloads; 404 streamed reply (address + reply, no credential, real `_stream_post` via fake session); receiver failure / async / plain receivers; no receiver; generate == one-message chat; tracing parity.
- `tests/integrations/test_llamaswap_protocols.py` — covers DoD-13, DoD-14, DoD-15 (live, never skipped) — factory-built clients on both protocols: chat / generate / chat_json non-empty, chat_json parses, streamed == returned; Responses thinking on (well-formed block first), off (no block), off + effort `low` succeeds, non-streamed thinking on has no marker; unknown model raises `LLMError` with a reason (streamed and not); `list_models` non-empty list of str.
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 ✓, DoD-9 ✓, DoD-10 ✓, DoD-11 ✓, DoD-12 ✓, DoD-13 ✓ (live), DoD-14 ✓ (live), DoD-15 ✓ (live)

#### Notes
- Piece-level assertions take DEC-003-13 literally: the opening marker is a prefix of the first reasoning piece and the closing marker a prefix of the first text piece after it (not separate pieces); only the block closed at completion gets `</think>` as its own piece.
- DoD-10 also checks a plain (non-async) receiver, per DEC-000-23 / DEC-003-13 "awaited if awaitable".
- DoD-15 asserts the unknown-model `LLMError` has a non-empty `description` (server reply per DEC-000-11); exact text is not asserted.
- Not run (verifier's job); ruff not run either (no command execution in this role) — the verifier should run `.venv/bin/python -m ruff check` on both files.

## Files Changed

### Step 001 — Chat-completions names, factory protocol, warning category, version
- `llm/openai_chat_completions.py` — `OpenAIChatCompletionsClient`: renamed copy of the snapshot `openai.py` (only module header, class line, class docstring differ)
- `llm/llama_swap_chat_completions.py` — `LlamaSwapChatCompletionsClient`: renamed copy of the snapshot `llama_swap.py` (same three differences)
- `llm/llm.py` — `get_llm_client` protocol selection in DEC-003-8 order; module header mentions protocol
- `llm/__init__.py` — `__version__` 0.3.0
- `pyproject.toml` — version 0.3.0
- `uv.lock` — refreshed (`llm-client` 0.2.0 -> 0.3.0)
- `README.md` — both protocols and four class names, factory `protocol` argument and refusals, `LLMSettingIgnoredWarning`, interim tool-conversation refusal on Responses classes; dropped the stale env-credential lines

### Step 002 — OpenAI Responses class: settings, request and non-streamed reply
- `llm/openai.py` — Responses `OpenAIAPIClient`: `_merge_options` wire mapping + once-per-instance `LLMSettingIgnoredWarning`, `_responses_text_format`, `_build_responses_payload`, `_responses_reply_text`, non-streamed `chat`/`generate`/`chat_json` over `POST /responses`, `chat_with_tools` refusal; tool-loop helpers, `_tools_stream_request` and `/chat/completions` paths removed; streamed `chat` wired to `_stream_post("/responses")` + the step-003 `_read_responses_stream` stub

### Step 003 — OpenAI Responses class: streamed chat
- `llm/openai.py` — `_read_responses_stream` filled per DEC-003-13 (typed-event dispatch, per-run `<think>`/`</think>` markers, failure events and missing `response.completed` → `LLMError`); streamed `chat` wiring from step 002 kept (comment and module header updated)

### Step 004 — llama-swap Responses class: settings, request and non-streamed reply
- `llm/llama_swap.py` — Responses `LlamaSwapAPIClient`: `_merge_options` wire mapping (reasoning effort, never a summary) + once-per-instance `LLMSettingIgnoredWarning`, `_responses_text_format`, `_responses_payload`, `_responses_reply_text`, non-streamed `chat`/`generate`/`chat_json` over `POST /responses`, `chat_with_tools` refusal naming `LlamaSwapChatCompletionsClient`; tool-loop helpers, `_tools_stream_request` and `/chat/completions` paths removed; streamed `chat` wired to `_stream_post("/responses")` + the step-005 `_stream_responses_response` stub

### Step 005 — llama-swap Responses class: streamed chat
- `llm/llama_swap.py` — `_stream_responses_response` filled per DEC-003-13 (typed-event dispatch, per-run `<think>`/`</think>` markers, closing piece at `response.completed`, failed/incomplete/error events and missing completion → `LLMError`); streamed `chat` wiring from step 004 kept unchanged; module header mentions step 005

### 2026-11-08 correction — code
- `llm/openai.py` — `_build_responses_payload`: assistant turns sent as `{"type": "message", "role": "assistant", "content"}`, system/user and head system items stay bare (DEC-003-11 amended); docstring + why-comment updated
- `llm/llama_swap.py` — `_responses_payload`: same typed assistant item rule, own copy (DEC-arch-2); docstring + why-comment updated
- `llm/__init__.py` — `__version__` back to 0.2.0 (DEC-004-3)
- `pyproject.toml` — version back to 0.2.0 (DEC-004-3)
- `uv.lock` — refreshed (`llm-client` 0.3.0 -> 0.2.0)
- `README.md` — the one "0.3.0" mention changed to 0.2.0

## Notes & Issues

- Step 001: README describes the feature's end state (Responses classes speak `/responses`, refuse `chat_with_tools`, emit `LLMSettingIgnoredWarning`); until steps 002/004 land, `OpenAIAPIClient`/`LlamaSwapAPIClient` still behave as chat completions.
- Step 001: `llm/exceptions.py` needed no change (skeleton's `LLMSettingIgnoredWarning` is complete).
- Step 002: streamed `chat` already opens `_stream_post("/responses", payload)` and calls `_read_responses_stream` (still the step-003 stub, so `stream=True` raises `NotImplementedError` until step 003); step 003 needs only the parser body.
- 2026-11-08 — plan correction after delivery (user decisions via the orchestrator; plan side only, code and tests not yet changed):
  - Typed assistant items: llama.cpp's Responses route refuses an assistant input item without `"type": "message"` (400 "Cannot determine type of 'item'"; probe in `docs/plans/004.responses-tools/context.md`). DEC-003-11 amended: caller assistant turns are sent as `{"type": "message", "role": "assistant", "content": …}`, system and user turns stay bare `{"role", "content"}`, in both Responses classes and every operation that builds input items (`chat` streamed and not, `chat_json`). Reworded in place: step 002 DoD-2 and Interface intent (payload builder), step 004 DoD-2 and Interface intent (payload builder). Steps 003 / 005 need no change (their DoD-1 compares against the non-streamed body).
  - Version: DEC-003-14 (bump to 0.3.0) superseded by DEC-004-3 — no release till the end of the roadmap; `pyproject.toml`, `llm/__init__.py` and `uv.lock` go back to 0.2.0. Reworded in place: step 001 Goal, Interface intent (package exports, project metadata) and DoD-10; `001.context.md` version and test-expectation lines; `context.md` Commands line.
  - Code / test work still to do (out of this plan's air gap, for the orchestrator to route): `llm/openai.py` and `llm/llama_swap.py` payload builders; `pyproject.toml`, `llm/__init__.py`, `uv.lock` back to 0.2.0; tests covering step 001 DoD-10 (`tests/test_package_surface.py`) and steps 002 / 004 DoD-2 (`tests/test_openai_responses_request.py`, `tests/test_llamaswap_responses_request.py`) to the amended clauses. The `## Skeleton`, `## Tests` and `## Files Changed` records above describe the delivery as built and are not rewritten.

## Ultra phase

- orient: done — waves: W1 001 (alone: changes existing symbol `get_llm_client`, copies modules) · W2 002, 004 · W3 003, 005
- mode: serial — pairable: 1 (002/003 share llm/openai.py, 004/005 share llm/llama_swap.py) · profile: light (Python, single package, .venv)
- typecheck: isolated (compileall + import + ruff)
- harvest: done — docs/.cache/ultra/003.responses-chat/harvest.md (A source, B tests/README/ruff)
- skeleton W1: done — step 001 (stubs raise NotImplementedError; coder replaces both new modules with renamed copies)
- source snapshot: docs/.cache/ultra/003.responses-chat/source-snapshot/{openai.py,llama_swap.py} = HEAD a9eeaf9 — step 001 coder copies from these (W2 skeletons edit the live modules first)
- skeleton W2: done — steps 002, 004
- skeleton W3: done — steps 003, 005 (all steps frozen)
- tests: done — steps 001-005
- lint exception: UP006/UP035/UP045 findings in llm/openai_chat_completions.py and llm/llama_swap_chat_completions.py are inherited from the verbatim copy (DEC-003-7; same findings as the source modules in the baseline) — not new faults. Test files: 0 new findings.
- red-gate: PASS (run 1; 784 failed / 90 passed, all right-reason; live files collect 66)
- code: done — steps 001-005 (serial)
- verify: PASS (run 1; unit 874 passed; live 62/66 — 4 known environmental: 2 hard-coded unserved model in test_llm_integrations.py, 2 llama.cpp json_schema grammar; all 33 new protocol live tests pass)
- lint exception also covers one inherited I001 per chat-completions copy (same import block as source modules, baseline has it)
- advisory for /finalize: DoD-14 "ruff check llm/ is clean" contradicts verbatim copy (baseline never clean); llama_swap `_responses_text_format` lets a nested `type` key override `json_schema` (OpenAI copy filters it); llama_swap `_responses_reply_text` non-mapping reply → AttributeError (DEC-003-12 allows generic error; OpenAI raises LLMError); llama_swap streamed `response.incomplete` falls back to JSON of incomplete_details when reason is non-string
- glossary: done — llm/, tests/, tests/integrations/ catalogues
- stale out-of-marker text for /finalize: tests/integrations/AGENTS.md Conventions (model env vars + defaults per DEC-003-15; "library reads key" remark stale after DEC-001-5); root AGENTS.md Repository Structure lacks the two *_chat_completions.py modules / protocol split / LLMSettingIgnoredWarning, credential "known defect" stale; llm/AGENTS.md Conventions (inline session.post bullet superseded by DEC-arch-5 `_stream_post`, Adding-a-backend lacks `protocol`, credential defect sentence); tests/AGENTS.md os.environ count (only test_llm_options.py, which still builds OpenAIAPIClient); docs/architecture/interfaces.md protocol axis + warnings sections still "(planned)"
- commit: feature 003 delivered
- 2026-11-08 correction: tests/AGENTS.md catalogue says test_package_surface.py checks 'version 0.3.0' and llm/AGENTS.md says __version__ 0.3.0 — refresh in the correction's glossary pass

### 2026-11-08 correction — tests

## Tests

### Step 001 — tests amendment, DoD-10 (2026-11-08)
- `tests/test_package_surface.py` — covers DoD-10 — expected version rebound from `0.3.0` to `0.2.0` (DEC-004-3, superseding DEC-003-14): `_EXPECTED_VERSION = "0.2.0"`; `test_pyproject_version_is_0_3_0__DoD10` renamed to `test_pyproject_version_is_0_2_0__DoD10`, `test_package_version_is_0_3_0__DoD10` renamed to `test_package_version_is_0_2_0__DoD10` (docstrings updated); module docstring version updated. `test_package_version_matches_pyproject__DoD10` is unchanged. No other test in the file was touched.
- Coverage: DoD-10 ✓ (amended); all other DoD items unchanged from the original Step 001 tests entry.

## Notes & Issues
- 2026-11-08 — `tests/AGENTS.md` still describes `test_package_surface.py` as checking "version 0.3.0". That file was outside my write set, so I left it; whoever owns the docs should change it to 0.2.0.

## Tests

### Step 002 — tests, DoD-2 amendment (2026-11-08)
- `tests/test_openai_responses_request.py` — covers DoD-2 (amended DEC-003-11) — `test_openai_chat_messages_become_input_items_in_order__DoD2` and `test_openai_chat_system_instruction_heads_input__DoD2` now expect assistant turns as `{"type": "message", "role": "assistant", "content": …}`; new `test_openai_chat_items_typed_only_for_assistant__DoD2` (chat) and `test_openai_chat_json_builds_same_items__DoD2` (chat_json) send a caller system / user / assistant / user conversation with a head system instruction and assert the exact item list, exact key sets (assistant: type/role/content; system, user, head system: role/content only) and the caller's list unchanged.
- No other test changed.
- Coverage: DoD-1 ✓, DoD-2 ✓ (amended), DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 ✓, DoD-9 ✓, DoD-10 ✓, DoD-11 ✓, DoD-12 ✓, DoD-13 ✓, DoD-14 ✓, DoD-15 ✓, DoD-16 ✓, DoD-17 ✓, DoD-18 ✓

#### Notes
- The chat_json test passes `system=` as a keyword (the frozen `chat_json` signature is recorded as unchanged; DEC-003-11 names chat_json among the item-building operations). If the verifier reports a binding error on `system`, that is a TEST fault to route back.
- Not run; ruff not run (no command execution in this role).

## Tests

### Step 004 — tests, DoD-2 amendment (2026-11-08)
- `tests/test_llamaswap_responses_request.py` — covers DoD-2 (amended DEC-003-11) — `test_llamaswap_chat_messages_become_items_with_system_first__DoD2` now expects the caller assistant turn as `{"type": "message", "role": "assistant", "content": …}` and system / user / head system items bare; new `test_llamaswap_chat_item_key_sets_per_role__DoD2` pins exact key sets per role (extra caller keys dropped; assistant exactly type/role/content); new `test_llamaswap_chat_json_builds_the_same_input_items__DoD2` asserts `chat_json` sends the same items. Other tests unchanged (one docstring narrowed to "a user input item").
- Coverage: DoD-1 ✓, DoD-2 ✓ (amended), DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 ✓, DoD-9 ✓, DoD-10 ✓, DoD-11 ✓, DoD-12 ✓, DoD-13 ✓, DoD-14 ✓, DoD-15 ✓, DoD-16 ✓, DoD-17 ✓, DoD-18 ✓
- 2026-11-08 correction: README line 19 finally reads "…before the protocol split under the new names" (supersedes the 0.2.0 wording noted in Files Changed)
- 2026-11-08 correction verify: PASS (unit 878; live 62/66, 4 known env; live probe llama-swap [user, assistant, user] OK). Glossary: llm/AGENTS.md + tests/AGENTS.md version lines → 0.2.0. Advisory: no live OpenAI test sends an assistant turn over Responses.
