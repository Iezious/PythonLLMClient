# Feature 006 — tolerant-tool-loop (release 0.2.1) — feature-wide context

## Goal

Deliver release 0.2.1 of the library: tool-call failures in a tool conversation go back to the model by default (tolerant), with a strict opt-out; a tool-event observer; tools that accept any keyword arguments; `LLMError` carrying `status` and `kind`; a silence limit on streamed replies (`read_timeout`); a per-client `trace` switch; a strict typing contract checked with mypy (Ollama excluded); a corrected reasoning docstring; version 0.2.1, README and agent-doc updates; one live test.

Not roadmapped (no `brief.md`). Scope is bounded by the stories and architecture below; nothing outside them is built.

## Product (delivered ids — cite in `[test]` items as `US-###.AC-#`)

- `docs/product/stories.md`: US-064 (AC-1..AC-20, tolerant default / strict opt-out), US-065 (AC-1..AC-13, observer), US-066 (AC-1, any-argument tools), US-067 (AC-1..AC-3, error kinds), US-068 (AC-1..AC-3, silence limit), US-069 (AC-1..AC-4, trace switch). Strict behaviour unchanged as in US-033 / US-035 (US-064.AC-10).
- `docs/product/features.md`: FEAT-001, FEAT-007, FEAT-011; Cross-feature rules (failure kinds; "an app checked by a strict type checker uses the library without type errors").
- `docs/product/use-cases.md`: UC-001 (A5, E3, E4), UC-007 (A8–A10, E6, E7), UC-011 (A4, E5).

## Architecture (build against; never edit)

- `docs/architecture/interfaces.md` — § Exposed: client method contract; § Tool loop contract; § Tool failures; § Tool event observer; § Exposed: capability matrix (+ Protocol axis); § Exposed: constructor arguments; § Exposed: factory; § Exposed: warnings; § Exposed: typing contract; § Exposed: error contract (Kind column); § Consumed: SSE stream format (silence limit).
- `docs/architecture/subsystems.md` — § Base client (rule: **no shared behaviour logic between client classes**); § Backend clients; § Tool loop; § Streaming and reasoning; § Tracer.
- `docs/architecture/data-model.md` — § Tool events (`ToolEvent`, `ToolErrorKind`); § Error kinds (`LLMErrorKind`); § Trace record (item 6).
- `docs/architecture/quick-reference.md` — public exports and arguments index.

## Binding decisions (cite by id; do not restate in code beyond a citation)

- Architect session: DEC-arch-14 (model-facing texts), -15 (failure handling copied into each class), -16 (argument-name check, `**kwargs` skip), -17 (strict: error event, then the existing exception), -18 (observer awaited inline), -19 (`kind` / `status` per raise site), -20 (silence limit = per-read timeout on streamed requests only), -21 (trace of tolerant / observer failures), -22 (mypy --strict, Ollama excluded), -23 (version 0.2.1, tag v0.2.1), -24 (Ollama refusals; factory refuses `read_timeout` with ollama), -25 (option merge stays in the base, not extended), -26 (tolerant observer failure ends that call's events), -27 (trace switch states), -28 (tolerant default, `"raise"` opt-out).
- Earlier: DEC-arch-2, DEC-000-35 (independent class copies), DEC-arch-13 (a cut tool round that requests a tool still errors — not a tool failure, no events), DEC-000-23 (callbacks awaited only if awaitable), DEC-002-2..5 (strict texts), DEC-002-12 (trace never changes the outcome), DEC-001-7 (transport error text), DEC-004-14 (Responses tool-round test seams).
- This feature: `docs/decisions/006.tolerant-tool-loop.md`, DEC-006-1..13.

## Cross-cutting constraints

- **Placement rule (DEC-arch-15).** Tool-failure handling, the observer, the argument-name check, the model-facing texts and the observer warning are written into **each** of the four OpenAI-family classes' own `chat_with_tools` loop (and that class's own private helpers). No class imports tool-loop code from another class's module. Nothing tool-loop-related goes into `llm/llm_client.py` except the `chat_with_tools` declaration (DEC-006-10). Data definitions (`ToolEvent`, `ToolErrorKind`, `LLMErrorKind`, `LLMToolObserverWarning`) live once (DEC-006-3). Transport plumbing in the base: `status` / `kind` at base raise sites, `read_timeout`, tracer construction from `trace`.
- **Deprecated Ollama** (`llm/ollama.py`): only the edits in DEC-006-13. Its `chat_with_tools` keeps refusing `tool_errors` / `on_tool_event` by not declaring them; its constructor does not declare `read_timeout`.
- **Typing gate.** From step 002 on, `uv run mypy --strict llm` stays clean after every step (DEC-006-1); `tests/test_typing_contract.py` enforces it in the default test run (DEC-006-2). Every step's code is fully annotated.
- **Option merge** (`_merge_options_base`) is not changed (DEC-arch-25).
- **Default flip.** Once a class's loop step lands, its tool conversations are tolerant unless `tool_errors="raise"` is passed. Existing unit tests that expect a raise from that class's tool loop are migrated in that step by passing `tool_errors="raise"` (tests folder convention), not by weakening them.
- **Code documentation** per `docs/plans/AGENTS.md` § Code documentation; decision comments cite the DEC / DoD id.
- **Error text unchanged.** `str(LLMError)` stays `message: description` (or `message`); `kind` / `status` are additional attributes only (US-067.AC-3).

## Shared vocabulary

- **Tolerant** = `tool_errors="result"` (the default). **Strict** = `tool_errors="raise"`. Only `"raise"` selects strict (DEC-006-9).
- **Failure kinds** (`ToolErrorKind`): `not_offered`, `bad_arguments`, `unknown_argument`, `tool_raised`, `unserialisable_result` — meanings in interfaces.md § Tool failures.
- **Model-facing text** — the fixed tool-result text of DEC-arch-14 (per kind; observer failure → `Tool '<n>' failed.`).
- **Observer** — the app's `on_tool_event` callable; receives `ToolEvent` (shape and filling: data-model.md § Tool events, DEC-006-4).

## Tool execution today (v0.2.0, each of the four classes has its own copy)

Per requested call: offered check (`_offered_tool_name`, `LLMError` if not offered or empty name) → `tools[name]` → read arguments (`_read_tool_arguments`, `LLMError` if unreadable or not an object) → `t.log_tool_call(name, kwargs)` → name check `all(k in inspect.signature(func).parameters for k in kwargs)` else `ValueError("Bad args for tool ...")` → `func(**kwargs)`, awaited if awaitable, `except Exception` → `RuntimeError("Tool '<n>' failed: ...") from exc` → `_tool_result_text` (`LLMError` if a dict / list is not JSON-serialisable) → `t.log_tool_result` → append the tool turn. Round limit → `RuntimeError` after the loop; advertised tool without callable → `ValueError` pre-flight. Streamed and non-streamed rounds share the execution block within each class. Cancellation is not caught anywhere (only `except Exception`). Raw arguments (string, mapping or none) are available at execution and discarded after parsing. Strict mode keeps all of these exceptions and texts.

## Per-call procedure (target; steps 006–009, each class writes its own copy)

For every tool call of a round, in the order requested:

1. Build the event identity: `name` (as requested, may be empty), `call_id`, raw `arguments` (DEC-006-4).
2. Observer: `start` event (awaited inline if awaitable, DEC-arch-18).
3. Offered check → failure `not_offered`.
4. Read arguments (JSON text or mapping; must be an object) → failure `bad_arguments`.
5. Argument-name check per DEC-arch-16 (skipped for a `**kwargs` callable or an unreadable signature) → failure `unknown_argument`, naming the first undeclared name in the model's order.
6. Trace `TOOL CALL` entry (DEC-006-7).
7. Run the tool (await if awaitable). An ordinary exception → failure `tool_raised`. Cancellation and other non-ordinary exceptions propagate unchanged in both modes, with no event and no tool result.
8. Turn the result into text (mapping / list → JSON text, else `str`) → failure `unserialisable_result`.
9. Success: observer `result` event with `output` = that text; trace `TOOL RESULT`; append the tool-result for the call.

On a failure of kind K:
- **Strict:** observer `error` event (K), then raise exactly today's exception: `LLMError` (`kind="tool"`) for `not_offered`, `bad_arguments`, `unserialisable_result`; `ValueError` for `unknown_argument`; `RuntimeError` (chained from the tool's exception) for `tool_raised`. Texts unchanged (DEC-002-3..5).
- **Tolerant:** observer `error` event (K); tool-result carrying the model-facing text for K; trace `TOOL RESULT` + `TOOL ERROR` note (DEC-006-7); go on with the next call.

On an observer failure (an ordinary exception from the observer or from awaiting its result):
- **Strict:** propagates unchanged; nothing further for this call (if it failed on `start`, the tool does not run).
- **Tolerant:** no further event for this call (DEC-arch-26); if on `start`, the tool does not run; tool-result carries the observer-failure text (DEC-006-8); `LLMToolObserverWarning` emitted (DEC-006-8); trace per DEC-006-7; go on with the next call.

Unchanged around the procedure: pre-flight `ValueError` for an advertised tool without a callable; sequential execution in requested order; `max_loops` and its `RuntimeError` in both modes; a Responses round cut at the app's output limit that requests a tool raises `LLMError` (`kind="incomplete"`, step 005) before any tool runs — no events; streamed and non-streamed rounds share the procedure within each class.

## Files touched across steps

| Path | Steps |
|---|---|
| `llm/exceptions.py` | 001, 002 |
| `llm/tool_events.py` (new) | 001 |
| `llm/__init__.py` | 001, 010 |
| `llm/llm_client.py` | 002, 003, 004, 009 |
| `llm/openai.py` | 002, 003, 004, 005, 006 |
| `llm/llama_swap.py` | 002, 003, 004, 005, 007 |
| `llm/openai_chat_completions.py` | 002, 003, 004, 005, 008 |
| `llm/llama_swap_chat_completions.py` | 002, 003, 004, 005, 009 |
| `llm/ollama.py` | 002, 004 |
| `llm/trace.py` | 002, 004 |
| `llm/llm.py` | 002, 003, 004 |
| `pyproject.toml`, `uv.lock` | 002, 010 |
| `README.md`, root `AGENTS.md` | 010 |
| `tests/test_package_surface.py` | 001, 010 |
| `tests/test_typing_contract.py` (new) | 002, 009 |
| `tests/test_llm_factory.py` | 003, 004 |
| `tests/test_llm_tools_errors.py`, `tests/test_llm_tools_streaming.py`, `tests/test_llm_tools_fragments.py`, `tests/test_llm_tools_reasoning_stream.py` | 008, 009 |

## Test conventions (from `tests/AGENTS.md`, binding)

Fake the server through client seams (`_request`, `_stream_post` + `_iter_sse_data` on Responses, `_tools_stream_request` on chat completions, fake session); build clients with `bearer_token="sk-test"`; environment only via `monkeypatch.setenv` / `delenv`; `async def test_<backend>_<behaviour>(monkeypatch)` with `@pytest.mark.asyncio`; count warnings with `catch_warnings(record=True)` + `simplefilter("always")`; no `conftest.py`, helpers module-local. Live tests: `tests/integrations/AGENTS.md`.

## Commands

`uv run pytest tests/ --ignore=tests/integrations` (unit) · `uv run pytest` (unit + live) · `uv run mypy --strict llm` · `uv run ruff check llm/` · `uv lock` after dependency or version changes.
