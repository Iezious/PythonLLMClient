# Feature 002 — edge-fixes

| Step | File                           | Status  | Verifier | Date |
|------|--------------------------------|---------|----------|------|
| 001  | `001.tool-loop-errors.md`      | done    | PASS     | 2026-11-07 |
| 002  | `002.tool-call-fragments.md`   | done    | PASS     | 2026-11-07 |
| 003  | `003.reply-checks.md`          | done    | PASS     | 2026-11-07 |
| 004  | `004.trace-robustness.md`      | done    | PASS     | 2026-11-07 |
| 005  | `005.repeated-fill-stop.md`    | done    | PASS     | 2026-11-07 |

## Skeleton

### Step 001 — frozen interface (2026-11-07)
- `llm/openai.py` — `OpenAIAPIClient.chat_with_tools(self, messages: List[Dict[str, str]], *, tools_definitions: List[Dict[str, Any]], tools: Dict[str, Callable[..., Any]], system: Optional[str] = None, max_loops: int = 5, options: Optional[OpenAIOptions] = None, stream: bool = False, on_delta: Optional[Callable[[str], Awaitable[None]]] = None, response_format: Optional[Dict[str, Any]] = None) -> str` — signature unchanged; docstring updated with the new contract (per-call order; offered-tool check -> LLMError, DEC-002-3; argument reading -> LLMError naming the tool, DEC-002-4; result conversion -> LLMError naming the tool, DEC-002-5; round-limit RuntimeError message with `max_loops` and value, DEC-002-2; Raises section updated). Behaviour not implemented.
- `llm/llama_swap.py` — `LlamaSwapAPIClient.chat_with_tools(...)` — same signature as above, unchanged; docstring updated identically (Args section kept). Behaviour not implemented.
- No private helpers created (coder's choice per Interface intent).
- Caller-compile edits (out of Source-files scope): None.
- Compile gate: `compileall -q llm` + `import llm` OK; ruff on both files identical to baseline by rule+message.

### Step 002 — frozen interface (2026-11-07)
- `llm/llm_client.py` — `async def _stream_openai_tools_response(self, response: aiohttp.ClientResponse, on_delta: Optional[Callable[[str], Awaitable[None]]] = None) -> tuple[str, list[dict]]` — unchanged signature; docstring updated to the DEC-002-6 grouping contract (positioned fragments: id set when truthy, name set not appended, arguments appended; positionless fragments: new call on a different non-empty id, or on no id + name while the current call has a name, same id continues; order: positioned by position, then positionless by appearance; call shape `{"id": str ("" when none), "type": "function", "function": {"name": str, "arguments": str}}`). Body untouched (old index-keyed `+=` grouping remains for the coder to replace).
- Caller-compile edits (out of Source-files scope): None.
- Compile gate: `compileall -q llm` + `import llm` OK; ruff on `llm/llm_client.py` 60 findings, identical by rule+message to the baseline.

### Step 003 — frozen interface (2026-11-07)
- `llm/openai.py` — `OpenAIAPIClient.embed_batch(self, texts: List[str], *, options: Optional[OpenAIOptions] = None) -> List[List[float]]` — signature unchanged; docstring added with the DEC-002-8/10 contract (mapping with list `data`; each entry a mapping with non-empty list `embedding`; count == len(texts); all-integer non-bool `index` → must be exactly 0..n-1 and used for order, else server order; checks inside the trace session before `log_response`, validated count logged; decode errors `json.JSONDecodeError`/`aiohttp.ContentTypeError` → `LLMError` from the error; bytes/non-mapping body → `LLMError`; `LLMError` from `_request` passes through; existing "Invalid response from OpenAI API when embedding" message kept for missing `data`).
- `llm/openai.py` — `OpenAIAPIClient.list_models(self) -> List[str]` — signature unchanged; docstring replaced with the DEC-002-9/10 contract (mapping, list `data`, each entry a mapping with non-empty str `id`; ids in server order; untraced; same unreadable-body mapping; existing "when listing models" message kept).
- `llm/openai.py` — `OpenAIAPIClient.get_models(self) -> List[Dict[str, Any]]` — signature unchanged; docstring added: same checks as `list_models`, entries returned unchanged; existing "when fetching models" message kept.
- `llm/openai.py` — `OpenAIAPIClient.chat_json(self, messages: List[LLMMessage], *, response_schema: Optional[Dict[str, Any]] = None, system: Optional[str] = None, options: Optional[OpenAIOptions] = None) -> str` — signature unchanged; docstring "Note" replaced: only the JSON parse is checked (DEC-002-7), any valid JSON value returned unchanged, schema not sent. The no-op `properties` loop is still in the body for the coder to remove.
- `llm/llama_swap.py` — `LlamaSwapAPIClient.embed_batch`, `.list_models`, `.get_models` — same signatures as the OpenAI copies, unchanged; docstrings identical except the backend name ("llama-swap").
- `llm/llama_swap.py` — `LlamaSwapAPIClient.chat_json(...)` — signature unchanged; docstring states the parse-only check (DEC-002-7). Confirmed: the body has only `json.loads` in the try, no field loop.
- `chat_with_tools` untouched in both files. No private helpers created (coder's choice).
- Behaviour: none implemented. The new checks cannot be stubbed with a throw without breaking the preserved happy path (`tests/test_llm_embeddings.py`, `tests/test_get_models.py`), so the bodies are untouched. Expected red against the current bodies: DoD 2, 4-8, 12-14, and 15 (OpenAI scalars). DoD 1, 3, 9-11, 16 and llama-swap 15 may already pass (regression guards).
- Caller-compile edits (out of Source-files scope): None.
- Compile gate: `compileall -q llm` + `import llm` OK; `ruff check llm/openai.py llm/llama_swap.py` has 87 findings, identical to the baseline by rule+message.

### Step 004 — frozen interface (2026-11-07)
- `llm/trace.py` — `class TraceSession` — changed (docstring only: broken-on-first-write-failure invariant, remembered instruction, session never closes the file)
- `llm/trace.py` — `TraceSession.__init__(self, f, model: str, operation: str, timestamp: datetime) -> None` — unchanged signature; docstring: header write failure marks session broken, never raises
- `llm/trace.py` — `TraceSession._write(self, text: str) -> None` — unchanged signature; docstring: first `Exception` from write/flush marks broken and is swallowed, later writes skipped (DEC-002-12)
- `llm/trace.py` — `TraceSession.log_system(self, system: Optional[str]) -> None` — unchanged signature; docstring: remembers instruction; `None` writes nothing; `""` writes empty section but is not used for dropping
- `llm/trace.py` — `TraceSession.log_messages(self, messages: List[Dict[str, Any]]) -> None` — unchanged signature; docstring: skip first message only if role `"system"` and content equals the non-empty logged instruction exactly (DEC-002-14)
- `llm/trace.py` — `TraceSession.log_error(self, error: Exception) -> None`, `TraceSession.close(self) -> None` — unchanged signatures; docstrings: never raise on write failure; `close` writes footer only, does not close file
- `llm/trace.py` — `LLMTracer.trace(self, operation: str, model: str) -> Generator[TraceSession, None, None]` (`@contextmanager`) — unchanged signature; docstring freezes DEC-002-11/12/13 contract (exclusive create, `-1`..`-N` suffix, OSError setup -> no-op session yielded once, body exceptions never setup failures, guarded log_error/footer/close, CancelledError path unchanged)
- `llm/trace.py` — `_MAX_COLLISION_SUFFIX = 1000` (module constant) — new; the suffix bound of DEC-002-13 (tests may monkeypatch it lower to exercise exhaustion)
- Module docstring updated with suffix layout, never-changes-outcome rule, and the DEC-002-16 seams.
- Behaviour: none implemented; current code paths unchanged (still `mkdir` unguarded, `open(..., "w")`, unguarded teardown) for the coder to fill.

#### Test seams (DEC-002-16) — how tests patch them
- **Clock**: `llm.trace` binds `datetime` at module level (`from datetime import datetime, timezone`); `trace` calls `datetime.now(timezone.utc)`. Patch: `monkeypatch.setattr(llm.trace, "datetime", FrozenClock)` where `FrozenClock.now(tz)` returns a fixed aware `datetime` (real `datetime` instance, since `.strftime` and `.isoformat` are called on it). The type annotation is unaffected (`from __future__ import annotations`).
- **Open**: the record is opened with the builtin `open` resolved through module globals; there is NO `llm.trace.open` attribute. Patch: `monkeypatch.setattr(llm.trace, "open", fake_open, raising=False)` (or `monkeypatch.setattr("llm.trace.open", fake_open, raising=False)`). Frozen call shape after the coder's change: `open(filepath, "x", encoding="utf-8")` — `filepath` a `pathlib.Path`, mode positional `"x"`, `encoding` keyword. `fake_open` should accept `(file, mode="r", *args, **kwargs)`; it may raise `FileExistsError` (collision -> suffix retry), another `OSError` (setup failure -> untraced), or return an object with `write(str)`, `flush()`, `close()` whose `write` raises after N calls. Delegate to `builtins.open` for real files.
- Folder failure needs no seam: set `LLM_TRACE_FOLDER` to a path whose child date folder cannot be made (e.g. a `tmp_path` subfolder containing a regular file named as the UTC date, or the folder itself being a file).

- Caller-compile edits (out of Source-files scope): None.
- Compile gate: `compileall -q llm` + `import llm` OK; `ruff check llm/trace.py` — same 9 findings as baseline by rule+message (line numbers shifted only).

### Step 005 — frozen interface (2026-11-07)
- `llm/llm.py` — `recursive_format_placeholders(text: str, data: Dict[str, str | bytes | int | float | None | object]) -> str` — changed (contract only; signature unchanged, still module-only in `llm/llm.py` `__all__`, not exported from `llm/__init__.py`). Docstring rewritten to the DEC-002-15 contract: stops on no slot found, on a pass that leaves the text unchanged, or after the 10-pass cap (silent); callables evaluated per occurrence per pass and only when their slot appears; values inserted via `str()`; unknown/self-referring slots returned as written; callable errors pass through. Body untouched: the new behaviour (early stop on an unchanged pass) is an absence of extra passes and cannot be stubbed with a throw without breaking preserved behaviour, so the DoD 1 test (lookup counting) and the "computed once" part of DoD 2 are expected red against the current body.
- Caller-compile edits (out of Source-files scope): None.
- Compile gate: `compileall -q llm` + `import llm` OK; `ruff check llm/llm.py` findings identical to baseline (8, rule+message).

## Tests

### Step 001 — tests (2026-11-07)
- `tests/test_llm_tools_errors.py` — covers DoD-1..DoD-10 — `chat_with_tools` on `OpenAIAPIClient` and `LlamaSwapAPIClient`, each parametrized streamed / non-streamed (seams `_tools_stream_request` / `_request`): unoffered name -> LLMError naming it, callable in `tools` not run (DoD-1); empty name / missing name / missing `function` -> LLMError, no tool runs (DoD-2); invalid JSON text and JSON list/number/string/null -> LLMError naming the tool, not run (DoD-3, DoD-4); object text, mapping, empty text, missing arguments -> tool runs with given / no kwargs (DoD-5); unserialisable mapping/list (object, set, circular) -> LLMError naming the tool, custom `__str__` result sent as its text (DoD-6); serialisable mapping/list sent next round as JSON text, decodes back equal, non-ASCII unescaped (DoD-7); `max_loops=4` -> RuntimeError whose text contains `max_loops`, the value 4 and a limit/reached/exceed reason word (DoD-8); `max_loops=2` with an endless model -> exactly 2 requests (DoD-9); raising tool -> RuntimeError naming it, unknown argument name -> ValueError and not run, advertised tool without callable -> ValueError with no request (DoD-10).
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 ✓, DoD-9 ✓, DoD-10 ✓, DoD-11 [manual/live, no test]

### Step 002 — tests (2026-11-07)
- `tests/test_llm_tools_fragments.py` — covers DoD-1 — positioned fragments repeating the name (and id) give one call named once, arguments concatenated
- `tests/test_llm_tools_fragments.py` — covers DoD-2 — positionless fragments with different ids (different or same tool name; also non-integer index) give separate calls in appearance order
- `tests/test_llm_tools_fragments.py` — covers DoD-3 — positionless id-less fragments split per named fragment, each taking the following argument fragments, id ""
- `tests/test_llm_tools_fragments.py` — covers DoD-4 — positionless fragments repeating id and name give one call
- `tests/test_llm_tools_fragments.py` — covers DoD-5 — interleaved positioned calls assembled per position in position order; positioned before positionless
- `tests/test_llm_tools_fragments.py` — covers DoD-6 — every call has exactly id (str, "" when none) / type "function" / function{name, arguments}
- `tests/test_llm_tools_fragments.py` — covers DoD-7 — reasoning block and content reach the receiver with positionless fragments; returned content excludes reasoning
- All tests parametrized over `OpenAIAPIClient` and `LlamaSwapAPIClient`, parser driven via stubbed `_iter_sse_data`, streams end with `[DONE]`.
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 [manual/live, no test]

### Step 003 — tests (2026-11-07)
- `tests/test_llm_embeddings_checks.py` — covers DoD-1, DoD-2, DoD-3, DoD-4, DoD-5, DoD-6, DoD-7, DoD-8, DoD-9, DoD-10 — `embed_batch`/`embed` on OpenAI and llama-swap (fake `_request`): one vector per text (also with tracing on), position ordering, server order when not every entry has an integer non-bool position, count/position/entry/reply violations and bytes / `JSONDecodeError` / `aiohttp.ContentTypeError` bodies give `LLMError`, helper `LLMError` passes through with the same text, `embed` single vector / no-vector error.
- `tests/test_llm_models_checks.py` — covers DoD-11, DoD-12, DoD-13, DoD-14 — `list_models` ids in server order, `get_models` entries exactly as sent; malformed entries, malformed replies and unreadable bodies give `LLMError` from both methods on both clients.
- `tests/test_llm_chat_json.py` — covers DoD-15, DoD-16 — with a no-default expected shape a number/decimal/true/null/string/list reply is returned unchanged on both clients; a non-JSON reply gives `LLMError` with a shape and is returned as is without one.
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 ✓, DoD-9 ✓, DoD-10 ✓, DoD-11 ✓, DoD-12 ✓, DoD-13 ✓, DoD-14 ✓, DoD-15 ✓, DoD-16 ✓, DoD-17 [manual/live, no test]

### Step 004 — tests (2026-11-07)
- `tests/test_llm_trace.py` — covers DoD-1 — unusable folder (folder is a file; date folder is a file) and refused record open leave a client call's result unchanged with no record; after a setup failure the call's own error still reaches the app; exhausting the suffix bound (`_MAX_COLLISION_SUFFIX` lowered to 2) leaves the scope untraced and existing records untouched
- `tests/test_llm_trace.py` — covers DoD-2 — writes failing from the header / SYSTEM PROMPT / MESSAGE HISTORY / REQUEST / RESPONSE never break a successful call and the file is closed; after the first failure no further write is attempted
- `tests/test_llm_trace.py` — covers DoD-3 — write failures while the error is recorded (or from the first write) keep the call's own `LLMError` (type and text); file closed
- `tests/test_llm_trace.py` — covers DoD-4 — failing footer write keeps the call's result on success and its error on failure; file closed
- `tests/test_llm_trace.py` — covers DoD-5 — failing call record ends ERROR (`LLMError: <text>`) then END TRACE; error identical to the untraced call's; a body `OSError` is recorded and re-raised as the same object (not taken as setup failure)
- `tests/test_llm_trace.py` — covers DoD-6 — cancellation re-raised as the same object; record has no ERROR, ends with END TRACE
- `tests/test_llm_trace.py` — covers DoD-7 — frozen clock: nested (2 and 3 deep) and concurrent sessions get distinct complete records named `<ts>.log`, `<ts>-1.log`, `<ts>-2.log`; a pre-existing record is never overwritten; `-N` at the lowered bound is still used
- `tests/test_llm_trace.py` — covers DoD-8 — no collision: single record `<folder>/2026-03-04/llm-trace-20260304-050607890123.log`; real clock: today's UTC date folder with a matching timestamp name
- `tests/test_llm_trace.py` — covers DoD-9 — LLM_TRACE unset/`0`/`off` with folder, or on with folder unset/empty: tracer disabled, no record from a direct scope or a client call
- `tests/test_llm_trace.py` — covers DoD-10, DoD-11, DoD-12, DoD-13 — OpenAI and llama-swap `chat` / `chat_json` / `chat_with_tools`: instruction in SYSTEM PROMPT, not in MESSAGE HISTORY; caller's own system message (incl. one identical to the instruction) kept; no / empty instruction keeps a leading system message; REQUEST JSON equals the payload as sent with the instruction message first
- `tests/test_llm_trace.py` — covers DoD-14 — direct `LLMTracer` use with label `nightly-digest`: layout, header (label, model, ISO time), sections, END TRACE
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓, DoD-6 ✓, DoD-7 ✓, DoD-8 ✓, DoD-9 ✓, DoD-10 ✓, DoD-11 ✓, DoD-12 ✓, DoD-13 ✓, DoD-14 ✓

### Step 005 — tests (2026-11-07)
- `tests/test_llm_templates.py` — covers DoD-1, DoD-2, DoD-3, DoD-4, DoD-5 — repeated filling: unknown slots kept and filling stops after the first unchanged pass (observed via a mapping counting membership tests per key), self-referring computed value computed once and kept, nested slots filled across passes (incl. text form of non-str values), 10-pass cap ends silently (ever-changing value computed exactly 10 times; 16-deep chain stops at `{{k10}}`), unused computed value not computed, computed-value error passes through as the same object.
- Coverage: DoD-1 ✓, DoD-2 ✓, DoD-3 ✓, DoD-4 ✓, DoD-5 ✓

## Files Changed

_populated by the coder as steps complete_

### Step 001 — Tool-loop error mapping
- `llm/openai.py` — `chat_with_tools` per-call order: offered check, argument reading, result conversion via new private module helpers `_offered_tool_name`, `_read_tool_arguments`, `_tool_result_text` (LLMError, DEC-002-3..5); round-limit `RuntimeError` names the limit and `max_loops=<value>` (DEC-002-2); module header updated
- `llm/llama_swap.py` — same changes in its own copy (DEC-002-1)

### Step 002 — Streamed tool-call fragment assembly
- `llm/llm_client.py` — `_stream_openai_tools_response` fragment grouping per DEC-002-6: positioned (integer, non-bool `index`) fragments keyed by position with name set not appended; positionless fragments matched against the current positionless call by id/name; result is positioned calls by position then positionless calls by appearance; local `new_tool_call` builds the empty call shape

### Step 003 — Embedding, model-list and JSON-only reply checks
- `llm/openai.py` — `embed_batch` / `list_models` / `get_models` go through new private module helpers `_fetch_reply` (maps `json.JSONDecodeError` / `aiohttp.ContentTypeError` to `LLMError` from the error; `LLMError` passes through), `_reply_text`, `_reply_entries`, `_model_entries`, `_embedding_vectors` (DEC-002-8..10; checks inside the trace session before `log_response`, validated count logged); `chat_json` no-op `properties` loop removed (DEC-002-7); `import aiohttp` added; module header updated
- `llm/llama_swap.py` — same reply checks in its own copy with "llama-swap" wording (DEC-002-1); `chat_json` confirmed parse-only, unchanged; module header updated

### Step 004 — Trace robustness
- `llm/trace.py` — `LLMTracer.trace`: guarded setup (date folder + exclusive `open(..., "x")` with `-1`..`-_MAX_COLLISION_SUFFIX` suffix; any `OSError` or exhaustion yields the no-op session once, DEC-002-11/13), guarded `log_error` / footer / file close with the caller's exception re-raised unchanged (DEC-002-12); `TraceSession._write` marks the session broken on the first failure and skips later writes; `log_system` remembers the instruction and `log_messages` drops the matching leading system copy (DEC-002-14)

### Step 005 — Repeated filling stops when nothing more can be filled
- `llm/llm.py` — `recursive_format_placeholders`: each pass is one `re.sub` over the pass-start text (only slots present at pass start are filled; per-occurrence membership test then subscription; unknown slots kept as written); stops on no slot, on a pass that leaves the text unchanged, or at the 10-pass cap (DEC-002-15)

## Notes & Issues

- Step 001 tests: an autouse fixture `delenv`s `LLM_TRACE` / `LLM_TRACE_FOLDER` (no env writes). The DoD-8 "reason" check is deliberately loose (`limit|reached|exceed`, case-insensitive) since the spec fixes no wording; the red signal comes from the required limit value. DoD-9 and the DoD-10 cases are expected to pass already against the stubs (regression guards).
- Step 002 tests: DoD-2 includes one case with a non-integer `index` ("x") treated as no position, per `002.context.md` Rules; DoD-5 includes the mixed positioned/positionless order rule from Interface intent.
- Step 003 tests: clients are `OpenAIAPIClient` / `LlamaSwapAPIClient` (the `*ChatCompletionsClient` names in `tests/AGENTS.md` are forward statements from feature 003 and do not exist yet). DoD-3 cases include "only some entries positioned", boolean and text positions (all keep server order, per `003.context.md` § Positions). DoD-8/14 include `aiohttp.ContentTypeError` alongside `json.JSONDecodeError`, per `003.context.md` and the frozen docstring.
- Step 004 tests: the record is parsed by section titles from data-model § Trace record; blank and pure `=`/`-` rule lines are treated as layout, so the tests do not depend on rule-line widths. Seams used exactly as frozen in `## Skeleton` § Step 004 (`llm.trace.datetime`, `llm.trace.open` with `raising=False`, `_MAX_COLLISION_SUFFIX`).
- Step 005 tests: DoD-1 pass counting relies on the step context's statement that the helper looks up the data with a membership test (`in`) per slot occurrence per pass; the counting mapping overrides `__contains__` only. Exact-output assertions for DoD-4 assume each pass replaces only slots present at the start of that pass (one nesting level per pass), per the spec's "filled across passes".

## Ultra phase

- orient: done 2026-11-07 — typecheck: isolated (ruff check + compileall; `.venv/bin/python`) · pairable: 3 (002, 004, 005)
- mode: serial — user asked for 001-004 one after another with a commit per feature on `feature/new-api` · profile: light (Python, one package, uv `.venv`)
- waves: W1 001,002,004,005 · W2 003 (shares openai.py/llama_swap.py with 001)
- harvest: done — docs/.cache/ultra/002.edge-fixes/harvest.md (2 reports + build profile)
- skeleton: W1 done · W2 done — steps 001, 002, 003, 004, 005
- tests: done — steps 001, 002, 003, 004, 005
- red-gate: PASS (run 2; run 1 FAIL 004 lint SIM117 only, fixed)
- note: step 005 coder read status.md Notes & Issues (test-coder note on lookup counting); content restates 005.context.md, no test file read — air gap judged intact
- code: done — steps 001, 002, 004, 005, 003 (serial)
- verify: run 1 FAIL — 001, 003, 004, 005 PASS; 002 Fault SPEC (DoD-6 positionless case: id-carrying fragment after a named id-less call — DEC-002-6 "non-empty id" vs Interface intent "a different call id starts a new call"). 588 passed / 2 failed (002 DoD-6 x2 clients). Lint 0 new. Live 29/4 (2 known json_schema grammar, 2 env: LLAMA_SWAP_GEN_MODEL unset). Awaiting user decision.
- user decision (2026-11-07): step 002 positionless case — keep DEC-002-6 as written (an id arriving for a named, id-less current call attaches to it). Re-route TEST (DoD-6). Step 002 Interface intent wording ("a different call id starts a new call") is looser than DEC-002-6; DEC-002-6 governs — note for /finalize.
- verify: PASS (run 2; 592 passed, ruff 0 new)
- glossary: done — llm/AGENTS.md, tests/AGENTS.md (inside markers)
- for /finalize (out-of-marker staleness): llm/AGENTS.md Conventions streaming rule still names `_ensure_session`/inline `session.post` (now `_stream_post`, since 001); credential rules still call env read a known defect (fixed by DEC-001-5); tests/AGENTS.md seam rule omits DEC-002-16 module seams (`llm.trace.open`, `llm.trace.datetime`); "two older tests set os.environ" now one (`test_llm_options.py`); step 002 Interface intent wording vs DEC-002-6.
