# llm

<!-- agent-ready:start -->

The package: a generic transport-only base client (requests, streams, error mapping, time limits, tracer), backend clients that each hold their own behaviour and tolerant tool loop (llama-swap and OpenAI, each on the Responses protocol and on chat completions; deprecated Ollama), typed option, message, error and tool-event shapes, tool helper, tracer, factory and template helpers, public exports.

## Glossary

**Files:**
- `__init__.py` — public exports (both protocol class pairs, `LLMError` / `LLMErrorKind`, `LLMSettingIgnoredWarning`, `LLMToolObserverWarning`, `ToolEvent` / `ToolErrorKind`) and `__version__` 0.2.1 (kept in step with `pyproject.toml`).
- `exceptions.py` — `LLMError`, raised for non-2xx responses, transport failures, in-stream error frames, early stream end and other client-side errors, carrying an optional machine-readable `kind` (`LLMErrorKind`, a closed set of failure kinds) and `status` (HTTP status) for apps to branch on; `LLMSettingIgnoredWarning`, the `UserWarning` subclass for a setting the Responses clients accept but cannot apply; `LLMToolObserverWarning`, the `UserWarning` subclass for a failing tool-event observer under tolerant tool handling.
- `tool_events.py` — data shapes for the tool-event observer, declared once so the four tool-loop classes share them: `ToolEvent` (one start / result / error event per tool call) and `ToolErrorKind` (closed set of tool failure kinds); no behaviour.
- `message.py` — `LLMMessage` TypedDict for one conversation message.
- `options.py` — typed option dicts (`BaseLLMOptions`, `OllamaOptions`, `OpenAIOptions`).
- `llm_client.py` — `LLMClient[OptionsT]` base, generic over its options shape: transport plumbing only — session lifecycle, `_request` (non-streaming), `_stream_post` (the one way a streamed request is opened, with the optional silence limit `read_timeout`), the SSE reader `_iter_sse_data`, failures mapped to `LLMError` with a kind, the tracer built from the three-state `trace` switch; keeps the existing shared chat-completions stream parsers and option-merge base unextended; holds no tool loop; behaviour methods (`chat_with_tools` included) raise `NotImplementedError`.
- `openai.py` — `OpenAIAPIClient`, OpenAI on the Responses protocol: `chat` / `generate` / `chat_json` over `/responses` with its own stream parser, settings mapped to the wire with a once-per-client `LLMSettingIgnoredWarning`; a reply cut at the output limit the app itself set is returned as text, any other incomplete reply is an error; `chat_with_tools` runs its own tool loop over `/responses`, streamed and not, tolerant of tool-call failures by default (the failure is returned to the model as text; `tool_errors="raise"` makes it strict) and reporting each call to an optional `on_tool_event` observer; streamed rounds read by its own tool-round parser; credential only from `bearer_token`; model-list and embedding replies checked.
- `openai_chat_completions.py` — `OpenAIChatCompletionsClient`, OpenAI on chat completions: its own copy of the chat-completions behaviour and of the tolerant tool loop (`tool_errors`, `on_tool_event`), streamed tool rounds through its `_tools_stream_request` seam.
- `llama_swap.py` — `LlamaSwapAPIClient`, llama-swap (primary backend) on the Responses protocol: its own copy of the Responses request, reply and stream handling (thinking sent as the reasoning effort, never a summary; a reply cut at the app's own output limit returned as text) and its own copy of the tolerant Responses tool loop (`tool_errors`, `on_tool_event`), streamed rounds read by its own tool-round parser.
- `llama_swap_chat_completions.py` — `LlamaSwapChatCompletionsClient`, llama-swap on chat completions: its own copy of the chat-completions behaviour and of the tolerant tool loop (`tool_errors`, `on_tool_event`), streamed tool rounds through its `_tools_stream_request` seam.
- `ollama.py` — `OllamaAPIClient`, deprecated, observation-only (takes only the `trace` switch from this release; no new tool parameters).
- `tools.py` — `pydantic_to_openai_tool` helper that builds a tool definition from a pydantic model.
- `trace.py` — `LLMTracer`, trace logging of LLM calls, switched by the environment or a per-client three-state `trace` argument (on / off / environment decides); records tool calls, results and tool-error notes; a record that cannot be set up or written never changes the call's outcome, and record names never collide.
- `llm.py` — `get_llm_client` factory (server kind plus `protocol`: omitted / `"responses"` / `"chat_completions"`, refused on Ollama; forwards the app's credential and the keyword-only `read_timeout` and `trace`, `read_timeout` refused on Ollama), placeholder-template helpers (repeated filling stops after a pass that changes nothing).
- `py.typed` — PEP 561 typing marker.

See also: `docs/architecture/subsystems.md`, `docs/architecture/interfaces.md`, `docs/architecture/data-model.md`; decisions by file (`docs/decisions/000.legacy.md` for DEC-000, `docs/decisions/001.core-fixes.md` for DEC-001, `docs/decisions/002.edge-fixes.md` for DEC-002, `docs/decisions/003.responses-chat.md` for DEC-003, `docs/decisions/004.responses-tools.md` for DEC-004, `docs/decisions/006.tolerant-tool-loop.md` for DEC-006, `docs/decisions/arch-responses-protocol.md` for DEC-arch-1..13, `docs/decisions/arch-tool-loop-0-2-1.md` for DEC-arch-14..28):
- `llm_client.py` — DEC-000-2, 4, 5, 9, 11, 12, 13, 14, 15, 17, 20, 21, 22, 23, 24, 25, 32, 33, 34, 40, 41, 50; DEC-000-16 (withdrawn); DEC-001-1, 7, 8, 9, 10, 11; DEC-002-6; DEC-006-5, 6, 10; DEC-arch-19, 20, 25, 27
- `openai.py`, `llama_swap.py` (Responses) — DEC-003-1, 2 (llama-swap), 4 (tool-chat refusal, lifted by feature 004), 5, 6, 9, 10, 11, 12, 13 (both amended by DEC-arch-13; incomplete-reply description amended by DEC-003-21), 17, 18, 19, 20, 21; DEC-004-1, 4, 5, 6, 7, 8, 9 (both amended by DEC-arch-13), 10, 11, 12, 15, 16, 17, 18; DEC-arch-1, 2, 3, 5, 6, 7, 8, 10, 11, 13
- all four OpenAI-family modules (`openai.py`, `llama_swap.py`, `openai_chat_completions.py`, `llama_swap_chat_completions.py`) — tolerant tool loop: DEC-006-4, 5, 7, 8, 9; DEC-arch-14, 15, 16, 17, 18, 19, 20, 21, 26, 27, 28
- `openai_chat_completions.py`, `llama_swap_chat_completions.py` — DEC-003-3, 7; DEC-arch-1; carried over from the previous clients: DEC-000-2 (openai), 5, 6, 7, 8, 9, 10, 12, 14, 15, 18 (llama-swap), 19, 23, 25, 26, 27, 28, 29, 30, 33, 35, 36, 37, 38, 39, 40, 41, 48, 49; DEC-001-2, 6 (openai), 8, 12, 13, 14, 15, 16; DEC-002-1, 2, 3, 4, 5, 7, 8, 9, 10
- `ollama.py` — DEC-000-6, 7, 9, 14, 15, 25, 26, 27, 28, 29, 30, 36, 37, 38, 41, 42, 49; DEC-006-5 (`trace` only), 13; DEC-arch-24, 27
- `options.py` — DEC-000-7, 17, 18; DEC-001-14
- `tools.py` — DEC-000-31, 43 (superseded by DEC-001-3)
- `trace.py` — DEC-000-4, 9, 11, 44, 45, 46, 47, 48, 49, 50; DEC-002-11, 12, 13, 14, 16; DEC-006-6, 7; DEC-arch-21, 27
- `llm.py` — DEC-000-1, 3, 14, 15, 51, 52; DEC-001-5; DEC-002-15; DEC-003-8; DEC-arch-4; DEC-006-5; DEC-arch-20, 24, 27
- `tool_events.py` — DEC-006-3, 4
- `__init__.py` — DEC-000-18, 31, 43 (superseded by DEC-001-3), 50, 51, 53 (superseded by DEC-001-4); DEC-001-4 and DEC-004-3 (both superseded by DEC-arch-23); DEC-003-14 (superseded); DEC-006-3; DEC-arch-19, 23 (version 0.2.1)
- `exceptions.py` — DEC-003-5; DEC-arch-6; DEC-006-3; DEC-arch-19

<!-- agent-ready:end -->

## Conventions

- Every backend module defines `_<BACKEND>_SUPPORTED_PARAMS` and `_<BACKEND>_ALIAS_MAP`; its `_merge_options` delegates to `self._merge_options_base(...)` and then post-processes — [inferred] (DEC-000-5/6/7)
- Option merge order: supported table → constructor defaults → call options; aliases mapped first, unknown keys dropped, `None` values stripped last — [inferred]
- Public operation shape: `opt = self._merge_options(...)`; copy messages and insert `system` at index 0 only if non-empty; build payload; open `with self._tracer.trace("<op>", self.model) as t:` and log `log_system` → `log_messages` → `log_request` → send → `log_response` inside it — [inferred]
- Non-streaming HTTP goes through `self._request`; a streamed request opens only through `async with self._stream_post(path, payload) as response:` (session, credential header, timeout, non-2xx and transport failures as `LLMError`) and is read by the shared or the backend's own parser; never an inline `session.post`; a backend never creates its own `aiohttp.ClientSession` — [inferred] (DEC-000-10, DEC-001-8)
- When streaming with tracing, wrap `on_delta` in a local `_tracing_delta`, only if `self._tracer.enabled` — [inferred] (DEC-000-48)
- An app-supplied callable (receiver, tools) is called and its result awaited only if `inspect.isawaitable(result)` — [inferred]
- On the chat-completions copies (`openai_chat_completions.py`, `llama_swap_chat_completions.py`), keep `_tools_stream_request(payload, on_delta)` as the HTTP seam for one streamed tool round; the Responses classes have no such method — each streamed tool round opens `_stream_post("/responses", payload)` and is read by the class's own tool-round parser — [sourced: commit 53d1067] (DEC-004-14)
- Content returned by the tool path never includes the streamed reasoning; do not "fix" this — [sourced: commit 7c27d6b]
- Import siblings absolutely (`from llm.<module> import X`); only `llm/__init__.py` uses relative re-exports — [inferred]
- Option and message shapes are `TypedDict`, not dataclasses or Pydantic; each client types its options with its backend's shape from `options.py` (llama-swap's own shape is pending, DEC-000-18) — [confirmed: user]
- Constructor shape: `(model, base_url=<backend default>, timeout=600.0, bearer_token=None, default_options=None, *, read_timeout=None, trace=None)`, forwarded in full to `LLMClient.__init__`; the deprecated Ollama constructor adds only `trace` — [planner: 006.tolerant-tool-loop] (DEC-006-5, DEC-arch-24)
- The library never reads the OpenAI credential from the environment; the app provides it as `bearer_token` and the factory forwards it; a missing or empty one is refused with `ValueError` at construction (the earlier env read is fixed, DEC-001-6, superseding DEC-000-2) — [confirmed: user]
- `aiohttp` only for HTTP. Ruff: py311, line-length 120, default rules — [inferred]
- llama-swap is free to diverge from OpenAI; this is not a lockstep rule — [confirmed: user]
- The deprecated Ollama client is maintained for observation only; do not extend it — [confirmed: user]
- Adding a backend: module with a class inheriting `LLMClient`; prefixed params and alias tables; override every `NotImplementedError` method the backend supports, including `get_models` (`embed` is inherited); register an exact kind string in `get_llm_client`; export from `llm/__init__.py` and `__all__`; unit tests plus live tests. A backend on the OpenAI family may need two classes, one per protocol (a Responses class and a chat-completions class), selected by the factory's `protocol` argument (`docs/architecture/interfaces.md` § Protocol axis) — [inferred]
- Version: bump `pyproject.toml` version and `llm/__init__.py` `__version__` together in the same change — [confirmed: user]
- `openai_chat_completions.py` / `llama_swap_chat_completions.py` hold the chat-completions classes and change only for chat-completions fixes; Responses logic lives only in `openai.py` / `llama_swap.py`, each its own copy, and never in `llm_client.py` (only the protocol-neutral `_request`, `_stream_post`, `_iter_sse_data` are shared) — [planner: 003.responses-chat] (DEC-arch-2, DEC-003-7)
- Tool-call failure handling, the tool-event observer, the argument-name check, the model-facing failure texts and `LLMToolObserverWarning` emission live in each OpenAI-family module's own `chat_with_tools` and private helpers; never in `llm_client.py`, never imported from another client module — [planner: 006.tolerant-tool-loop] (DEC-arch-15)
- Every `LLMError` raise in `llm_client.py` and the four OpenAI-family modules passes `kind=` (and `status=` for a non-2xx reply) per the error-kind mapping; the message text stays as it was; `ollama.py`'s own raises are not extended — [planner: 006.tolerant-tool-loop] (DEC-arch-19)
- Code stays clean under `uv run mypy --strict llm`; `ollama.py` is silenced by a per-module override, not fixed — [planner: 006.tolerant-tool-loop] (DEC-006-1)
