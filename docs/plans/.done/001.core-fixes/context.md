# 001.core-fixes — feature-wide context

## Goal

Fix the defects in the shared request, stream and settings paths every chat-completions call goes through, on the OpenAI and llama-swap clients (and, where a shared base fix reaches it unchanged, the deprecated Ollama client). The agreed boundary is `brief.md` (Definition, Scope In/Out) — this plan does not re-derive it.

Stage `001.fixes` (`docs/plans/roadmap.md`). Delivers FEAT-001, FEAT-002, FEAT-004, FEAT-005, FEAT-010.

## Stories this feature meets (DoD raw material)

| Story | ACs to meet | ACs to keep |
|---|---|---|
| US-003 (OpenAI credential from the app) | AC-1, AC-2, AC-4 | AC-3 |
| US-009 (no content is empty text) | AC-1, AC-3 | AC-2 |
| US-011 (server failures reported) | AC-3 | AC-1, AC-2, AC-4 |
| US-018 (only the app's settings sent) | AC-1 | — |
| US-019 (offered settings reach the server) | AC-1 | — |
| US-024 (failures inside a stream) | AC-1, AC-2 | — |
| US-026 (reasoning while streaming) | AC-5 | AC-1..AC-4 |
| US-036 (streamed text complete) | AC-1..AC-3 | — |
| US-047 (loads after a normal install) | AC-1 | — |

Regression guards that must stay met: US-005 (llama-swap optional credential), US-008, US-015, US-016, US-017, US-021, US-022, US-025.AC-1/3/4, US-027, US-028, US-041.AC-3/4.

## Out of scope (from `brief.md`)

Responses protocol / protocol choice; tool-loop defects; JSON, embedding, model-list, trace and template defects (feature 002, including non-JSON 2xx replies on listing and embeddings); llama-swap top-k / min-p / repeat-penalty (US-020); empty-input refusal (US-010); connection release without a usage scope; any change to `llm/ollama.py`.

## Architecture this builds against

- `docs/architecture/interfaces.md` § Exposed: error contract, § Exposed: factory, § Consumed: backend APIs (SSE stream format, chat completions), § Streaming callback.
- `docs/architecture/subsystems.md` § Base client (`LLMClient`), § Backend clients, § Options layer, § Streaming and reasoning.
- `docs/architecture/data-model.md` § Settings.
- `docs/architecture/overview.md` § Stack choices.
- `docs/decisions/arch-responses-protocol.md` DEC-arch-5 — this feature delivers its chat-completions half (DEC-001-8).

## Decisions

All planning-time decisions are in `docs/decisions/001.core-fixes.md` (DEC-001-1 .. DEC-001-16). Steps cite them by id and never restate them. Legacy entries this feature supersedes are listed there; still-binding legacy entries a coder must respect: DEC-000-5 (merge algorithm, unchanged), DEC-000-11 (non-2xx message format, unchanged), DEC-000-19 (thinking mapping, unchanged), DEC-000-20 / DEC-000-22 (think-marker and tool-path return rules), DEC-000-23 (receiver called inline; its failure propagates unchanged), DEC-000-35 (OpenAI and llama-swap are separate copies), DEC-000-39 (embedding request shape), DEC-000-42 (Ollama embed fallback matches "status 404"), DEC-000-48 (trace wrapper only when tracing is enabled).

## Files touched across steps

| File | Steps |
|---|---|
| `pyproject.toml`, `uv.lock`, `llm/__init__.py`, `llm/llm.py`, `llm/options.py` | 001 |
| `llm/llm_client.py` | 002, 003 (different functions; 003 runs after 002) |
| `llm/openai.py` | 004 |
| `llm/llama_swap.py` | 005 |
| `llm/ollama.py`, `llm/trace.py`, `llm/tools.py` | none — must not be edited |

Dependency waves: {001, 002} then {003, 004, 005}.

## Cross-cutting constraints

- **Credential.** The library never reads a credential from the environment. Error text, trace text and refusal messages never contain a credential value and are never built from request headers (US-003.AC-3, US-011.AC-4). Tests may read their own env vars (the app may; the library may not).
- **Both copies.** Every client-level fix lands in both `llm/openai.py` and `llm/llama_swap.py` (DEC-000-35). The two copies may differ in detail; the observable behaviour in the DoD is the same.
- **Ollama is observation-only.** No edit to `llm/ollama.py`. A base change (`_request`, reader) reaches it; its requests and its embed 404 fallback must behave as before (DEC-000-42, DEC-001-7).
- **Receiver contract.** A receiver failure ends the call and propagates unchanged; nothing added by this feature may catch, wrap or swallow it (DEC-000-23, US-022).
- **Library error.** Every failure this feature maps raises `LLMError` (`llm/exceptions.py`: `LLMError(message, description=None)`, attrs `.message`, `.description`, `str()` is `message: description`), chained from the original exception where there is one.
- **Superseded folder conventions.** `llm/AGENTS.md` `## Conventions` still says streaming uses "an inline `session.post`, `raise_for_status()`" (superseded by DEC-001-8) and calls the env credential read a known defect (removed by DEC-001-6). Follow the DEC entries, not those lines.

## Test seams (all usable without code changes)

- `monkeypatch.setattr(client, "_request", fake)` — non-streamed calls (payload capture).
- `monkeypatch.setattr(client, "_tools_stream_request", fake)` — one streamed tool round, wholesale.
- `monkeypatch.setattr(client, "_iter_sse_data", fake)` — feed canned data payload strings to either parser; parsers accept `response=None` then.
- A fake `client.session`: `_ensure_session()` returns an existing `self.session` as is, so a test may assign an object whose `request(...)` / `post(...)` return async-context-managed fake responses (with `.status`, `.text()`, `.json()`, `.read()`, `.headers`, `.content`, `.raise_for_status()`), or raise transport exceptions.
- A fake response whose `.content.iter_chunked(n)` is an async generator of `bytes` drives the stream reader directly.
- Existing parser tests (`tests/test_llm_tools_reasoning_stream.py`, 8 tests) all end with a `[DONE]` frame and must keep passing unchanged.

## Live tests

`tests/integrations/` runs against real servers in the default `uv run pytest` run (`tests/integrations/AGENTS.md`). A DoD item tagged `[test] (live)` is covered by a test in `tests/integrations/`; it needs the server and is reported as requires-live-run when none is reachable. OpenAI live tests read `OPENAI_API_KEY` / `OPEN_AI_KEY` in the test, skip when absent, and pass the value as `bearer_token`.

## Commands

`uv run pytest` (unit + live), `uv run pytest tests/ --ignore=tests/integrations` (unit only), `uv run ruff check llm/`, `uv run ruff format llm/`, `uv lock` (after a dependency change).
