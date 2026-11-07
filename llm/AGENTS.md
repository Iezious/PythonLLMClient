# llm

<!-- agent-ready:start -->

The package: base client with shared HTTP/SSE/option-merge plumbing, backend clients (llama-swap, OpenAI, deprecated Ollama), typed option and message shapes, tool helper, tracer, factory and template helpers, public exports.

## Glossary

**Files:**
- `__init__.py` — public exports and `__version__` (kept in step with `pyproject.toml`).
- `exceptions.py` — `LLMError`, raised for non-2xx responses, transport failures, in-stream error frames, early stream end and other client-side errors.
- `message.py` — `LLMMessage` TypedDict for one conversation message.
- `options.py` — typed option dicts (`BaseLLMOptions`, `OllamaOptions`, `OpenAIOptions`).
- `llm_client.py` — `LLMClient` base: session lifecycle, `_request` (non-streaming) and `_stream_post` (the one way a streamed request is opened), the SSE reader `_iter_sse_data`, the shared chat/tool stream parsers with error-frame and early-end checks (the tool parser groups streamed tool-call fragments by position, call id and name), tool loop, option-merge base; unsupported methods raise `NotImplementedError`.
- `openai.py` — `OpenAIAPIClient` for OpenAI and OpenAI-compatible servers; takes its credential only from the `bearer_token` argument and never from the environment; tool-loop calls, embedding and model-list replies are checked and failures raised as `LLMError`.
- `llama_swap.py` — `LlamaSwapAPIClient` for llama-swap (primary backend); its own copy of the OpenAI client's tool-loop and reply checks.
- `ollama.py` — `OllamaAPIClient`, deprecated, observation-only.
- `tools.py` — `pydantic_to_openai_tool` helper that builds a tool definition from a pydantic model.
- `trace.py` — `LLMTracer`, env-driven trace logging of LLM calls; a record that cannot be set up or written never changes the call's outcome, and record names never collide.
- `llm.py` — `get_llm_client` factory (forwards the app's credential to the OpenAI client), placeholder-template helpers (repeated filling stops after a pass that changes nothing), retired CLI smoke test.
- `py.typed` — PEP 561 typing marker.

See also: `docs/architecture/subsystems.md`, `docs/architecture/interfaces.md`, `docs/architecture/data-model.md`; decisions by file (`docs/decisions/000.legacy.md` for DEC-000, `docs/decisions/001.core-fixes.md` for DEC-001, `docs/decisions/002.edge-fixes.md` for DEC-002):
- `llm_client.py` — DEC-000-2, 4, 5, 9, 11, 12, 13, 14, 15, 17, 20, 21, 22, 23, 24, 25, 32, 33, 34, 40, 41, 50; DEC-000-16 (withdrawn); DEC-001-1, 7, 8, 9, 10, 11; DEC-002-6
- `openai.py`, `llama_swap.py` — DEC-000-2 (openai), 5, 6, 7, 8, 9, 10, 12, 14, 15, 18 (llama-swap), 19, 23, 25, 26, 27, 28, 29, 30, 33, 35, 36, 37, 38, 39, 40, 41, 48, 49; DEC-001-2, 6 (openai), 8, 12, 13, 14, 15, 16; DEC-002-1, 2, 3, 4, 5, 7, 8, 9, 10
- `ollama.py` — DEC-000-6, 7, 9, 14, 15, 25, 26, 27, 28, 29, 30, 36, 37, 38, 41, 42, 49
- `options.py` — DEC-000-7, 17, 18; DEC-001-14
- `tools.py` — DEC-000-31, 43
- `trace.py` — DEC-000-4, 9, 11, 44, 45, 46, 47, 48, 49, 50; DEC-002-11, 12, 13, 14, 16
- `llm.py` — DEC-000-1, 3, 14, 15, 51, 52; DEC-001-5; DEC-002-15
- `__init__.py` — DEC-000-18, 31, 43, 50, 51, 53; DEC-001-4

<!-- agent-ready:end -->

## Conventions

- Every backend module defines `_<BACKEND>_SUPPORTED_PARAMS` and `_<BACKEND>_ALIAS_MAP`; its `_merge_options` delegates to `self._merge_options_base(...)` and then post-processes — [inferred] (DEC-000-5/6/7)
- Option merge order: supported table → constructor defaults → call options; aliases mapped first, unknown keys dropped, `None` values stripped last — [inferred]
- Public operation shape: `opt = self._merge_options(...)`; copy messages and insert `system` at index 0 only if non-empty; build payload; open `with self._tracer.trace("<op>", self.model) as t:` and log `log_system` → `log_messages` → `log_request` → send → `log_response` inside it — [inferred]
- Non-streaming HTTP goes through `self._request`; streaming uses `await self._ensure_session()`, an inline `session.post`, `raise_for_status()` and the shared parser; a backend never creates its own `aiohttp.ClientSession` — [inferred] (DEC-000-10)
- When streaming with tracing, wrap `on_delta` in a local `_tracing_delta`, only if `self._tracer.enabled` — [inferred] (DEC-000-48)
- An app-supplied callable (receiver, tools) is called and its result awaited only if `inspect.isawaitable(result)` — [inferred]
- Keep `_tools_stream_request(payload, on_delta)` as the HTTP seam for one streamed tool round — [sourced: commit 53d1067]
- Content returned by the tool path never includes the streamed reasoning; do not "fix" this — [sourced: commit 7c27d6b]
- Import siblings absolutely (`from llm.<module> import X`); only `llm/__init__.py` uses relative re-exports — [inferred]
- Option and message shapes are `TypedDict`, not dataclasses or Pydantic; each client types its options with its backend's shape from `options.py` (llama-swap's own shape is pending, DEC-000-18) — [confirmed: user]
- Constructor shape: `(model, base_url=<backend default>, timeout=600.0, bearer_token=None, default_options=None)`, forwarded in full to `LLMClient.__init__` — [inferred]
- The library never reads the OpenAI credential from the environment; the app provides it and the factory forwards it. Today's env read is a known defect (DEC-000-1/2) — [confirmed: user]
- `aiohttp` only for HTTP. Ruff: py311, line-length 120, default rules — [inferred]
- llama-swap is free to diverge from OpenAI; this is not a lockstep rule — [confirmed: user]
- The deprecated Ollama client is maintained for observation only; do not extend it — [confirmed: user]
- Adding a backend: module with a class inheriting `LLMClient`; prefixed params and alias tables; override every `NotImplementedError` method the backend supports, including `get_models` (`embed` is inherited); register an exact kind string in `get_llm_client`; export from `llm/__init__.py` and `__all__`; unit tests plus live tests — [inferred]
- Version: bump `pyproject.toml` version and `llm/__init__.py` `__version__` together in the same change — [confirmed: user]
- `openai_chat_completions.py` / `llama_swap_chat_completions.py` hold the chat-completions classes and change only for chat-completions fixes; Responses logic lives only in `openai.py` / `llama_swap.py`, each its own copy, and never in `llm_client.py` (only the protocol-neutral `_request`, `_stream_post`, `_iter_sse_data` are shared) — [planner: 003.responses-chat] (DEC-arch-2, DEC-003-7)
