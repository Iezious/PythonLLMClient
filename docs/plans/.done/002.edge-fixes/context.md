# Feature 002 — edge-fixes: feature-wide context

## Goal and boundary

Fix the defects at the edges of single operations on the chat-completions clients. The agreed Definition and Scope In/Out are in `brief.md` (authoritative, not restated here). In short:

- Tool loop: round-limit message, unoffered tool, unreadable arguments, unserialisable result, and streamed fragments without a position or with a repeated name.
- `chat_json`: a single JSON value is no longer misreported.
- Embeddings and model list: reply checks.
- Tracing: never changes the outcome, concurrent records, instruction once.
- Templates: repeated filling stops early.

Out (from `brief.md`): sending the expected JSON shape (US-040), the empty-batch short-circuit (US-042.AC-4), the top-level export of repeated filling (US-054.AC-4), observing tool calls live, the Ollama tool-loop turn defect, and any Responses work.

## Builds on feature 001 — re-read the code

`001.core-fixes` is delivered before this feature and edits the same files: `llm/openai.py`, `llm/llama_swap.py` and `llm/llm_client.py`. It adds the streamed-request helper `_stream_post` (DEC-001-8), changes the parsers (DEC-001-1, DEC-001-9, DEC-001-11), maps transport failures in `_request` (DEC-001-7), and coerces null content (DEC-001-12). It also changes the settings tables and `chat_json`'s request (DEC-001-2, DEC-001-13, DEC-001-14).

- Every line number in this plan describes the code **before** feature 001. The skeleton and the coder re-read the current code and locate each touch site by function name.
- None of feature 001's work belongs here. **`_request` is not edited** in this feature (DEC-002-10).

## Cross-cutting rules

- **Both OpenAI-family copies, no Ollama edits** — DEC-002-1. Each per-client fix is written once in `llm/openai.py` and once in `llm/llama_swap.py`. `llm/ollama.py` is never in a step's scope.
- **The library's error** is `LLMError(message, description=None)` from `llm/exceptions.py`. Its text form is `message: description`. When a mapped failure has an underlying exception, raise the new error with `from` that exception. No new exception class is added.
- **Errors inside a trace session are recorded.** An exception raised inside `with self._tracer.trace(...) as t:` is written as an ERROR section and then re-raised unchanged. A fix that raises inside the session needs no extra logging.
- **Keep already-met behaviour green.** Existing unit tests are regression guards and must pass unchanged: `tests/test_llm_tools_streaming.py` (it matches `max_loops` in the round-limit error), `tests/test_llm_tools_reasoning_stream.py`, `tests/test_llm_embeddings.py` and `tests/test_get_models.py`. No step lists them as Test files: new behaviour goes in new test files.
- **Public surface unchanged.** No public method signature changes, and nothing is added to or removed from `llm/__init__.py`.

## Files touched across steps

| File | Steps |
|---|---|
| `llm/openai.py` | 001 (tool loop), 003 (`embed_batch`, `list_models`, `get_models`, `chat_json`) |
| `llm/llama_swap.py` | 001 (tool loop), 003 (`embed_batch`, `list_models`, `get_models`; `chat_json` verified only) |

Step 003 depends on 001 because they share these files. All other source files belong to exactly one step.

## Test conventions (from `tests/AGENTS.md`)

- Fake the server by swapping one client seam with `monkeypatch.setattr(client, "<seam>", fake)`:
  - `_request` for non-streamed calls.
  - `_tools_stream_request` for a streamed tool round.
  - `_iter_sse_data` for the tool-stream parser.
- Fakes assert the request they receive and return canned data.
- Build clients directly with `bearer_token="sk-test"`. Never write `os.environ`.
- Test names: `async def test_<backend>_<behaviour>(monkeypatch)`, keeping `@pytest.mark.asyncio`. Backend tokens: `openai`, `llamaswap`.
- There is no `conftest.py`, so helpers are local to each module. Each step's test files are new and disjoint from every other step's.
- Every `[test]` DoD item names the `US-###.AC-#` it verifies, and the test cites that DoD id.

## Architecture pointers

- `docs/architecture/interfaces.md`: § Tool loop contract, § Exposed: error contract, § Exposed: client method contract, § Exposed: utilities, § Consumed: backend APIs (SSE stream format, chat completions).
- `docs/architecture/subsystems.md`: § Tool loop, § Tracer (`LLMTracer`), § Utilities, § Call structure of an operation.
- `docs/architecture/data-model.md`: § Tool definition, § Trace record.

## Vocabulary

- **Offered tool**: a tool whose name is advertised in `tools_definitions`. A callable in `tools` without a definition is not offered (DEC-002-3).
- **Position**: the integer `index` on a streamed tool-call fragment.
- **Unreadable reply**: a 2xx body that is not JSON, or JSON that is not the expected mapping.
