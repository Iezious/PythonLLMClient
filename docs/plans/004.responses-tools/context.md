# 004.responses-tools — feature-wide context

## Goal and boundary

Tool conversations (`chat_with_tools`) work on the Responses protocol on `OpenAIAPIClient` and `LlamaSwapAPIClient`, streamed and not. This replaces 003's interim refusal (DEC-003-4). Tools are offered in today's shape. Calls and results travel as Responses items. Each round's reasoning is carried to later rounds of the same call and never reaches the app's conversation or the answer. The agreed Definition and Scope In/Out are in `brief.md` (authoritative, not restated). Out, in short: chat / generate / JSON on Responses (003), observing tool calls as they happen, storing reasoning on caller messages, Ollama.

Stage `002.responses`. Delivers FEAT-014, FEAT-005, FEAT-007 (on Responses); US-057 (AC-1 tool-conversation part), US-058. Realizes UC-013 A2 and UC-007 on Responses.

**Receiver contract on the tool path.** The brief says the receiver gets each round's "tool-call fragments under the same contract as chat completions". That contract assembles fragments into calls and does not deliver them to the receiver (DEC-004-9). The receiver gets text and reasoning only.

## Builds on features 001, 002 and 003 — re-read the code

None of 001, 002 or 003 is built yet. The steps describe the code as 003 leaves it. The skeleton and the coder re-read the built code and find each touch site by name. 003 names no private helper, so the names here are responsibilities, not identifiers.

After 003, each Responses class (`llm/openai.py`, `llm/llama_swap.py`) holds its own copy of:

- settings post-processing in `_merge_options` (DEC-003-9), which also emits the ignored-setting warning (DEC-003-6);
- a response-format translation helper (DEC-003-10);
- a payload builder (DEC-003-11);
- reply extraction (DEC-003-12);
- a private chat stream parser (DEC-003-13).

`chat_with_tools` raises `NotImplementedError` (DEC-003-4). Today's tool loop, with 002's fixes, lives only in the frozen chat-completions copies `llm/openai_chat_completions.py` and `llm/llama_swap_chat_completions.py` (DEC-003-7). It is the reference for the loop's order and failure rules. It may be read and copied, never imported from and never edited.

**Upstream corrections in 003 (2026-11-08).** Both conflicts this plan found in 003 are corrected there: DEC-003-11 is amended so 003's payload builder sends caller assistant turns as `{"type": "message", "role": "assistant", "content": …}` (system and user turns stay bare), and DEC-003-14 is superseded by DEC-004-3, so the version is back to 0.2.0. 004's own round items always carry their `type` (DEC-004-4); 004 bumps nothing.

## Stories this feature meets (DoD raw material)

| Story | ACs to meet on Responses | Notes |
|---|---|---|
| US-058 (tool conversation with a reasoning model) | AC-1..AC-5 | AC-1, AC-2 live |
| US-057 (same result on both protocols) | AC-1, tool-conversation part, streamed and not | live, through the factory |
| US-030 (tools run, model answers) | AC-1..AC-3 | |
| US-031 (tool from a typed data model) | AC-1 | flattening of `pydantic_to_openai_tool` output |
| US-032, US-033, US-034, US-035 | US-032.AC-1; US-033.AC-1..AC-4; US-034.AC-1..AC-4; US-035.AC-1..AC-3 (AC-4 for separate calls in a stream) | 002's rules reproduced (DEC-004-11) |
| US-023 (stream a tool conversation) | AC-1, AC-2 | |
| US-027 (reasoning in a streamed tool conversation) | AC-1, AC-2 | AC-3 is chat-completions-only; the Responses counterpart is US-058.AC-3 |
| US-037 (structured final answer) | AC-1..AC-4 | AC-1 live |
| US-061 (settings without a counterpart) | AC-1, AC-2 on tool chat | DEC-004-12 |
| US-060, US-024, US-022 | US-060.AC-1, AC-2, AC-4; US-024.AC-2; US-022.AC-1, AC-2 | failures inside tool rounds |
| US-049, US-050 | US-049.AC-4, AC-5, AC-12; US-050.AC-4 | trace of tool rounds |

## Architecture this builds against

- `docs/architecture/subsystems.md` § Tool loop (reasoning carry on Responses), § Chat protocols, § Streaming and reasoning (Responses stream, tool path returns no reasoning).
- `docs/architecture/interfaces.md` § Streaming callback, § Tool loop contract, § Protocol axis, OpenAI family (reasoning carried between tool rounds; `response_format` in tool chat), § Exposed: error contract, § Consumed: backend APIs (tool calls and results travel as items).
- `docs/architecture/data-model.md` § Message (internal turn shapes are the same on both protocols), § Tool definition (accepted in today's shape, converted).
- Design: `docs/decisions/arch-responses-protocol.md` DEC-arch-2, -5, -7, -9, -10, -11, -12 (steps cite, never restate).

## Decisions

Planning-time decisions: `docs/decisions/004.responses-tools.md`, DEC-004-1 .. DEC-004-14. Steps cite them by id.

Still binding from earlier features: DEC-003-6, -9, -10, -11 (as amended 2026-11-08, see above), -12, -13, -15, -16; DEC-002-1..-5; DEC-001-8, DEC-001-15; DEC-000-20, -22, -23, -26, -30, -33, -48.

## Files touched across steps

| File | Steps |
|---|---|
| `llm/openai.py` | 001, then 002 |
| `llm/llama_swap.py` | 003, then 004 |
| `llm/llm_client.py`, `llm/openai_chat_completions.py`, `llm/llama_swap_chat_completions.py`, `llm/ollama.py`, `llm/tools.py`, `llm/trace.py`, `llm/options.py`, `llm/message.py`, `llm/__init__.py`, `llm/llm.py`, `pyproject.toml` | none — must not be edited |

Dependency waves: {001, 003} → {002, 004}. The OpenAI and llama-swap steps touch different files and may run in parallel. Each streamed step builds on the same file's non-streamed step.

Size: each non-streamed step holds the tool flattening, round-input conversion, round reading and the loop itself, about 170–200 LoC. Each streamed step holds the tool-round parser and the streamed branch of the loop, about 100–130 LoC.

## Cross-cutting constraints

- **Independent copies.** Each Responses class holds its own copy of the tool loop, flattening, item conversion, round reading and tool-round parser (DEC-arch-2, DEC-002-1). Nothing is added to `llm/llm_client.py`. Nothing is imported from the other Responses module or from the chat-completions modules. The base's `_request`, `_stream_post` (DEC-001-8) and `_iter_sse_data` are reused unchanged.
- **003's chat path unchanged.** `chat`, `generate`, `chat_json`, the chat stream parser, `list_models`, `get_models` and `embed_batch` keep their 003 behaviour. 003's tests (other than the two refusal tests named in steps 001 and 003) must keep passing.
- **Signature unchanged.** `chat_with_tools(messages, *, tools_definitions, tools, system=None, max_loops=5, options=None, stream=False, on_delta=None, response_format=None) -> str`, as on the chat-completions classes.
- **Caller data untouched.** The caller's `messages` list and the app's `tools_definitions` are never modified. Reasoning is never stored on caller messages or internal turns (DEC-004-7; US-058.AC-4).
- **Loop rules.** DEC-004-11 (002's rules, same order). Request per round: DEC-004-10. Item conversion: DEC-004-6. Reasoning carry: DEC-004-7.
- **Library error.** `LLMError(message, description=None)` from `llm/exceptions.py`. Its text form is `message: description`. It is chained with `from` where there is an underlying exception. Errors raised inside `with self._tracer.trace(...)` are recorded and re-raised unchanged. The round-limit `RuntimeError` is raised after the trace session (DEC-002-2).
- **Credential.** Never in error or trace text (US-060.AC-4). Tests read their own environment variables.
- **Receiver contract.** Pieces arrive in order, and each is handled before the next is read. A receiver failure propagates unchanged (DEC-000-23). No trace wrapper on the tool path (DEC-004-10).

## Responses wire vocabulary for tools (shared by steps 001–004)

From a llama.cpp probe behind llama-swap (2026-11-07) and OpenAI's documented API:

- **Tool definition sent:** `{type: "function", name, description?, parameters?, strict}` (DEC-004-5, DEC-004-1).
- **Non-streamed reply `output`:** for example `[reasoning, function_call]` in a tool round and `[reasoning, message]` in the answer round.
  - Reasoning item keys: `type`, `id`, `status`, `content` (`[{type: "reasoning_text", text}]`), `summary` (`[]` on llama.cpp; filled on OpenAI), `encrypted_content` (`""` on llama.cpp; filled on OpenAI).
  - function_call item keys: `type`, `id` (`fc_…`), `call_id` (`call_…`), `name`, `arguments` (JSON text), `status`.
- **Input items built by 004:**
  - `{type: "function_call", call_id, name, arguments}`;
  - `{type: "function_call_output", call_id, output}`, where `output` must be text (a raw JSON object gets 400);
  - `{type: "message", role: "assistant", content}`;
  - carried reasoning items, verbatim.

  OpenAI matches outputs to calls by `call_id`. It needs reasoning items re-sent ahead of their function calls, with encrypted content, because `store` is false.
- **Probe, round-2 input variants accepted by llama.cpp:** reasoning verbatim; reasoning omitted; `encrypted_content` removed; function_call reduced to `{type, call_id, name, arguments}`. Refused: an assistant item without `type: "message"`.
- **Stream, as observed on llama.cpp** (`event:` and `data:` lines, no `[DONE]`):
  1. `response.created`, `response.in_progress`
  2. `response.output_item.added` (reasoning, `output_index` 0), then `response.reasoning_text.delta` ×N
  3. `response.output_item.added` (function_call, `output_index` 1). The item already has `name` and `call_id`, with `arguments: ""`.
  4. `response.function_call_arguments.delta` ×N (`delta`, `item_id`, `output_index`)
  5. `response.output_item.done` for each item. The function_call item has its full `arguments`; the reasoning item has no `status`.
  6. `response.completed` (with the full `response.output`)

  More stream facts:
  - llama.cpp sends no `function_call_arguments.done`; OpenAI does.
  - Items can finish out of order: reasoning's `output_item.done` can arrive after the call's deltas or after the text deltas.
  - Two calls are separate items (`output_index` 1 and 2) and are not interleaved.
  - `output_index` counts every item. The library's call numbering is dense and 0-based per round (DEC-arch-10).

## Test conventions

- Unit tests follow `tests/AGENTS.md` § Conventions:
  - `async def test_<backend>_<behaviour>(monkeypatch)` with `@pytest.mark.asyncio`; backend tokens `openai`, `llamaswap`;
  - clients built directly (OpenAI with `bearer_token="sk-test"`);
  - no `os.environ` writes; tracing tests use `monkeypatch.setenv` and `tmp_path`;
  - helpers are module-local (no `conftest.py`);
  - warning counts use `warnings.catch_warnings(record=True)` with `simplefilter("always")`.
- Seams: DEC-004-14. Fake replies and streams reproduce the probe shapes above (wire vocabulary).
- Each step's test files are disjoint from every other step's and from 003's. Every `[test]` DoD item cites the `US-###.AC-#` or DEC it verifies, and each test cites its DoD id.
- Live tests go in `tests/integrations/` and are part of the default run. A `[test] (live)` item is covered by a test there. The verifier reports it as requires-live-run when the server is unreachable. Settings, tool and assertions: DEC-004-13. Always `async with client:`. A **well-formed reasoning block** is defined in `docs/plans/003.responses-chat/context.md` § Test conventions.
- Assert only what a live model reliably produces, never exact text.

## Commands

- `uv run pytest` (unit + live)
- `uv run pytest tests/ --ignore=tests/integrations` (unit only)
- `uv run ruff check llm/`
- `uv run ruff format llm/`

## Vocabulary

- **Responses class**: `OpenAIAPIClient` / `LlamaSwapAPIClient`.
- **Round**: one request/reply to `/responses` inside one `chat_with_tools` call.
- **Round text**: the reply's answer text for a round, without reasoning or markers.
- **Carry item**: a reasoning output item kept verbatim to be re-sent on later rounds of the same call (DEC-004-7).
- **Internal copy**: the loop's private copy of the conversation in chat-completions turn shapes (DEC-004-6).
