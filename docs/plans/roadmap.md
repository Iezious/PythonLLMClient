# Roadmap
<!-- roadmap:start -->
All features are multi-step so they can be planned by `/planner` and built with `/ultra-feature`.

## Stage 001 — fixes
**Goal:** every defect the spec marks "must be fixed" is gone from the current (chat-completions) code before any rename or rewrite, so the Responses classes start from correct shared behaviour.
**Exit criteria:** every defect gap in FEAT-001..FEAT-012 closed (stories report their defect ACs met); unit and live suites green.
**Not in it:** all "not yet supported" gaps (stop-early stream, observing tool calls, reachability check, Ollama deprecation warning, empty-input refusal, llama-swap top-k/min-p/repeat-penalty, expected JSON shape sent, repeated-fill top-level export, empty embedding batch, connection release without usage scope); any protocol change.

| Feature | Track | Size | Delivers | Depends on | Definition |
|---|---|---|---|---|---|
| `001.core-fixes` | multi-step | M | FEAT-001, FEAT-002, FEAT-004, FEAT-005, FEAT-010 | — | Fix the shared request, stream and settings defects every chat call goes through. |
| `002.edge-fixes` | multi-step | M | FEAT-006, FEAT-007, FEAT-008, FEAT-009, FEAT-011, FEAT-012 | `001.core-fixes` | Fix defects at the edges of single operations: tool loop, JSON, embeddings, model list, tracing, templates. |

## Stage 002 — responses
**Goal:** the library speaks the Responses protocol by default on OpenAI and llama-swap, with chat completions kept as a named compatibility choice (FEAT-014).
**Exit criteria:** US-056..US-061 met; live tests for every chat operation on both protocols on OpenAI and llama-swap, including the two-round tool conversation on an OpenAI reasoning model; chat-completions regression suite green.
**Not in it:** removing chat completions; Ollama protocol choice; any stage 001 "not yet supported" gap; server-side conversation state.

| Feature | Track | Size | Delivers | Depends on | Definition |
|---|---|---|---|---|---|
| `003.responses-chat` | multi-step | L | FEAT-014, FEAT-005 | `001.core-fixes` | Protocol choice at client build, Responses by default; chat, generate and JSON-only on Responses. |
| `004.responses-tools` | multi-step | M | FEAT-014, FEAT-005, FEAT-007 | `003.responses-chat`, `002.edge-fixes` | Tool conversations on Responses, with per-call reasoning carry-over. |
<!-- roadmap:end -->
