# 001.core-fixes — Core request, stream and settings fixes
<!-- roadmap:start -->
- **Stage:** 001.fixes · **Track:** multi-step · **Size:** M
- **Delivers:** FEAT-001, FEAT-002, FEAT-004, FEAT-005, FEAT-010; US-003, US-009, US-011, US-018, US-019, US-024, US-026, US-036, US-047
- **Architecture:** `docs/architecture/interfaces.md` § Exposed: error contract, § Exposed: factory, § Consumed: backend APIs, § Streaming callback; `docs/architecture/subsystems.md` § Base client (`LLMClient`), § Backend clients, § Options layer, § Streaming and reasoning; `docs/architecture/data-model.md` § Settings; `docs/architecture/overview.md` § Stack choices
- **Depends on:** none

## Definition
Fix the defects in the shared request, stream and settings paths every chat call goes through. The package loads after a normal install; the app's OpenAI credential is the one used and the library never reads one from its environment. An unreachable server, an exceeded time limit, a refused streamed request, an error the server sends inside a stream and a stream that ends early all end with the library's error carrying the server's reason (never the credential). Streamed text arrives complete (multi-byte characters, other line endings, multi-line events) and a piece carrying both reasoning and answer text keeps both. A reply with no content is empty text, streamed or not, traced or not. Only settings the app set are sent, and every offered setting (stop sequences, token bias, end-user id, response format) reaches the server.

## Scope
**In:**
- DEC-000-43 load after install
- DEC-000-1/2 credential
- DEC-000-12 connection/timeout mapping
- DEC-arch-5 streamed non-2xx → library error on chat completions (supersedes DEC-000-10)
- DEC-000-13 in-stream error pieces and early end on chat completions
- DEC-000-34 stream reader
- DEC-000-21 reasoning+content piece
- DEC-000-9 null content
- DEC-000-6 built-ins always sent
- DEC-000-7 typed keys dropped
- Both OpenAI and llama-swap clients (DEC-000-35 copies)

**Out:**
- Protocol choice / Responses
- Tool-loop defects (002)
- JSON, embedding, model-list, trace, template defects (002)
- llama-swap extra sampling controls (US-020)
- Empty-input refusal (US-010)
- Connection release without usage scope
- Deprecated Ollama client changes, except where a shared base fix reaches it unchanged

## Open questions for the planner
- Which superseded DEC-000 entries (1, 2, 6, 7, 9, 12, 13, 21, 34, 43) get new DEC-001-n entries, and how DEC-arch-5's chat-completions half is recorded as delivered.
- How "only settings the app set" interacts with the three-layer merge and the "nothing removes the setting" rule (DEC-000-5).
- Whether the deprecated Ollama client is touched by shared-base fixes, and how its behaviour stays unchanged.
- Live-test coverage for credential use and stream robustness against llama-swap / OpenAI.
<!-- roadmap:end -->
