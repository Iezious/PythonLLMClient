# 003.responses-chat — feature-wide context

## Goal and boundary

Give the OpenAI and llama-swap clients the Responses protocol for chat, generate and JSON-only answers (streamed and not), keep today's chat-completions behaviour under new class names, let the factory choose the protocol, and warn the developer about settings Responses cannot take. The agreed Definition and Scope In/Out are in `brief.md` (authoritative, not restated). Out, in short: the Responses tool loop and tool-call events (feature 004), chat-completions behaviour changes beyond the rename, Ollama, removing chat completions.

Stage `002.responses`. Delivers FEAT-014, FEAT-005 (on Responses); realizes UC-013 (main 1–3 minus tool conversation, A1, A3, A4, A5, E0–E4).

**Tool conversation deferred.** US-057.AC-1's tool-conversation part and UC-013 A2 are delivered by feature 004. Until then `chat_with_tools` on the Responses classes is refused (DEC-003-4); tool conversations stay available on the chat-completions classes.

## Builds on features 001 and 002 — re-read the code

003 is built after `002.edge-fixes` (DEC-003-3; the roadmap still lists only 001 as a dependency — a `/roadmap` correction, not this plan's). Both earlier features edit `llm/openai.py`, `llm/llama_swap.py` and `llm/llm_client.py`:

- 001 adds `LLMClient._stream_post` (DEC-001-8), maps transport failures (DEC-001-7), rewrites the SSE reader (DEC-001-10), requires the OpenAI credential and never reads the environment (DEC-001-6), forwards the factory credential (DEC-001-5), empties the built-in setting values (DEC-001-13), adds `stop`, `logit_bias`, `user`, `response_format` to the tables (DEC-001-14), coerces null content (DEC-001-12), sets version 0.2.0 (DEC-001-4).
- 002 changes the tool loop errors, embedding / model-list reply checks (DEC-002-8..10), `chat_json`'s check (DEC-002-7) and the tracer (DEC-002-11..14).

Every name or line number quoted in this plan from the harvest describes the code **before** 001. The skeleton and the coder re-read the current code and locate each touch site by name.

## Stories this feature meets (DoD raw material)

| Story | ACs to meet (on Responses unless noted) | Notes |
|---|---|---|
| US-056 (choose the protocol) | AC-1..AC-5 | factory + class names |
| US-057 (same result on both protocols) | AC-1 (chat, generate, JSON-only; streamed and not), AC-2 | tool part → 004 |
| US-059 (reasoning on Responses) | AC-1..AC-3 | |
| US-060 (failures carry the server's reason) | AC-1..AC-4 | |
| US-061 (settings without a counterpart) | AC-1, AC-2 | |
| US-025 (thinking switch) | AC-5, AC-6 met on Responses; AC-1..AC-4 kept on Responses | AC-2 verified live on llama-swap Responses |

Regression guards that must stay met on both protocols: US-006, US-007, US-008, US-009.AC-1, US-014..US-017, US-018.AC-1, US-021, US-022, US-026.AC-1..AC-4, US-028.AC-1, US-003.AC-1..AC-4, US-005.

## Architecture this builds against

- `docs/architecture/interfaces.md` § Exposed: client method contract, § Protocol axis, OpenAI family, § Exposed: factory, § Exposed: warnings, § Exposed: error contract, § Consumed: backend APIs (Responses stream format), § Streaming callback.
- `docs/architecture/subsystems.md` § Backend clients, § Chat protocols, § Options layer, § Call structure of an operation, § Streaming and reasoning.
- `docs/architecture/data-model.md` § Message, § Settings, § Reasoning tag convention.
- `docs/architecture/quick-reference.md` § Client classes and protocols.
- Design: `docs/decisions/arch-responses-protocol.md` DEC-arch-1..12 (steps cite, never restate).

## Decisions

Planning-time decisions: `docs/decisions/003.responses-chat.md`, DEC-003-1 .. DEC-003-16. Steps cite them by id. Amended 2026-11-08: DEC-003-11 (assistant items typed `message`) and DEC-003-14 (superseded by DEC-004-3, version stays 0.2.0).

Still binding from earlier features: DEC-000-5 (merge algorithm), DEC-000-8 (generate is a one-message chat), DEC-000-11 (non-2xx error format), DEC-000-20 (think markers), DEC-000-23 (receiver inline, failures propagate), DEC-000-35 (separate copies), DEC-000-37 + DEC-002-7 (`chat_json` parse check), DEC-000-39 + DEC-001-16 (embedding request), DEC-000-48 (trace wrapper only when enabled), DEC-001-2, DEC-001-6, DEC-001-7, DEC-001-8, DEC-001-12, DEC-002-8, DEC-002-9, DEC-002-10.

## Files touched across steps

| File | Steps |
|---|---|
| `llm/openai_chat_completions.py` (new), `llm/llama_swap_chat_completions.py` (new), `llm/__init__.py`, `llm/llm.py`, `llm/exceptions.py`, `pyproject.toml`, `uv.lock`, `README.md` | 001 |
| `llm/openai.py` | 002, then 003 |
| `llm/llama_swap.py` | 004, then 005 |
| `llm/llm_client.py`, `llm/ollama.py`, `llm/options.py`, `llm/trace.py`, `llm/tools.py`, `llm/message.py` | none — must not be edited |

Dependency waves: {001} → {002, 004} → {003, 005}. Step 001 must land first: it copies today's `llm/openai.py` / `llm/llama_swap.py` into the chat-completions modules before 002 / 004 rewrite them, and it adds the warning category both rewrites use.

## Cross-cutting constraints

- **Independent copies.** The OpenAI and llama-swap Responses classes each hold their own settings mapping, payload building, reply extraction and stream parser; nothing Responses-specific is added to `llm/llm_client.py` (DEC-arch-2). The base's `_request`, `_stream_post` and `_iter_sse_data` are reused as they are.
- **Chat-completions modules are frozen copies.** After step 001 they change only for chat-completions fixes; no Responses logic goes there (DEC-003-7).
- **Ollama untouched.** `llm/ollama.py` is in no step's scope; the factory's Ollama branch only gains the protocol refusal.
- **Credential.** The library never reads a credential from the environment; error and trace text never contains it (US-060.AC-4, DEC-001-6). Tests read their own env vars.
- **Library error.** `LLMError(message, description=None)` from `llm/exceptions.py`; text form `message: description`; chained with `from` where there is an underlying exception. Errors raised inside `with self._tracer.trace(...)` are recorded and re-raised unchanged.
- **Receiver contract.** Pieces in order, each handled before the next is read; a receiver failure propagates unchanged (DEC-000-23). The trace wrapper is applied only when tracing is enabled (DEC-000-48).
- **Unchanged operations.** `list_models`, `get_models`, `embed_batch` (and so `embed`) on the Responses classes behave exactly as on the chat-completions classes (US-057.AC-2); the rewrite keeps them as they are.
- **Operation shape.** Merge settings → copy messages and put a truthy `system` first → build the payload → open the trace session (`log_system`, `log_messages`, `log_request("POST", "/responses", payload)`) → send → extract → `log_response` → return (subsystems § Call structure).

## Responses wire vocabulary (shared by steps 002–005)

- Request: `POST /responses`, body per DEC-003-11 with settings per DEC-003-9 / DEC-003-10.
- Non-streamed reply: a mapping with `status`, `output` (items: `message` with `content` parts `output_text`; `reasoning` with `summary` / `content` / `encrypted_content`), `error`, `incomplete_details`. Handling: DEC-003-12.
- Stream: `data:` payloads, each a JSON event with `type`. Observed on llama-swap: `response.created`, `response.in_progress`, `response.output_item.added`, `response.reasoning_text.delta`, `response.content_part.added`, `response.output_text.delta`, `response.output_item.done`, `response.output_text.done`, `response.content_part.done`, `response.completed`; no `[DONE]`. OpenAI with a summary also sends `response.reasoning_summary_text.delta`. Handling: DEC-003-13.
- llama-swap unknown model → HTTP 404 with a JSON `{"error": {…}}` body.

## Test conventions

- Unit: per `tests/AGENTS.md` § Conventions — `async def test_<backend>_<behaviour>(monkeypatch)` with `@pytest.mark.asyncio`, backend tokens `openai`, `llamaswap`; clients built directly (OpenAI with `bearer_token="sk-test"`); no `os.environ` writes (tracing tests use `monkeypatch.setenv`); helpers module-local (no `conftest.py`).
- Seams for the Responses classes: DEC-003-16. Seams for the chat-completions classes: as in `docs/plans/001.core-fixes/context.md` § Test seams.
- Each step's new test files are disjoint from every other step's. Every `[test]` DoD item cites the `US-###.AC-#` or DEC it verifies; each test cites its DoD id.
- Live: `tests/integrations/`, in the default run. A `[test] (live)` item is covered by a test in `tests/integrations/`, reported requires-live-run when the server is unreachable. Models, env vars and prompt guidance: DEC-003-15. Always `async with client:`. Protocol-parity tests build clients through the factory (`protocol` omitted for Responses, `"chat_completions"` for the other).
- Assert only what a live model reliably produces (type, non-empty, JSON validity, well-formed reasoning blocks, streamed equals returned) — never exact text. A **well-formed reasoning block**: opening and closing markers balanced, every block closed, the first block before the answer text.

## Commands

`uv run pytest` (unit + live), `uv run pytest tests/ --ignore=tests/integrations` (unit only), `uv run ruff check llm/`, `uv run ruff format llm/`, `uv lock` (after any change to the project version).

## Vocabulary

- **Responses class**: `OpenAIAPIClient` / `LlamaSwapAPIClient` after this feature.
- **Chat-completions class**: `OpenAIChatCompletionsClient` / `LlamaSwapChatCompletionsClient`.
- **No-counterpart setting**: a supported key with no Responses field (DEC-003-9); dropped with `LLMSettingIgnoredWarning`.
- **Thinking on / off / unset**: `enable_thinking` truthy / set and falsy / not in the merged settings.

## Extension 2026-11-08 — DEC-arch-13

User request (2026-11-08): change the code to follow DEC-arch-13 as additional steps of the current features. This folder gets the chat (non-tool) part as steps 006 (OpenAI, `llm/openai.py`) and 007 (llama-swap, `llm/llama_swap.py`); feature 004 gets the tool-round part in its own planning pass. Everything above still binds these steps; this section adds only what is new.

- **Stories.** US-062.AC-1, AC-2, AC-3, AC-5, AC-6 and US-024.AC-2 (narrowed: early ends other than a cut at the app's limit stay errors). US-062.AC-4 (tool conversation) is feature 004's.
- **Architecture.** `docs/architecture/interfaces.md` § Exposed: error contract and § Streaming callback, unchanged. Design: DEC-arch-13, which amends DEC-arch-11, DEC-003-12 and DEC-003-13.
- **Decisions.** DEC-003-17 (the flag and how the caller computes it), DEC-003-18 (which reason matches), DEC-003-19 (what a matched cut does, streamed and not), DEC-003-20 (the flag name is shared with feature 004). Steps cite them by id.
- **Code baseline.** Unlike the "before 001" note above, the line numbers in `006.context.md` / `007.context.md` come from the 2026-11-08 harvest of the current code (after feature 004 was delivered). The skeleton and the coder still locate each site by name.
- **Scope.** One source file per step; no other library file changes. Each step writes one new test file; no existing test file is edited. The two steps share no file and do not depend on each other.
- **Unchanged neighbours.** Any caller of the reply reader or the stream parser that passes no flag — including feature 004's tool-round code — keeps today's behaviour; the flag's default is off. The `chat_json` parse check (DEC-000-37, DEC-002-7) runs after the trace block only when an expected shape is given and needs no change to meet US-062.AC-5. A returned cut reply is logged as the call's response like any completed reply. No warning or notice is given (DEC-arch-13).
- **Test vocabulary.** A **cut at the app's limit**: a non-streamed reply with status `incomplete` and `incomplete_details.reason` `"max_output_tokens"`, or a streamed `response.incomplete` event whose `response.incomplete_details.reason` is `"max_output_tokens"`, on a call where the app set the limit. The app **sets the limit** through `max_tokens` or the `context_length` alias, as a call option or a client default; it **sets no limit** when neither is set, or when a default is cleared per call with `None` (DEC-000-5).
- **Live.** The live check lands in feature 005 step 002's token-cap live tests, revised in a separate pass; no live file is in these steps' scope.
- **Folder guides.** `llm/AGENTS.md` lines describing the old rule are refreshed by the glossary pass after verify; the coder does not edit them.
