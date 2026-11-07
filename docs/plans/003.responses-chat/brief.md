# 003.responses-chat — Responses protocol for chat, generate and JSON
<!-- roadmap:start -->
- **Stage:** 002.responses · **Track:** multi-step · **Size:** L (split already applied; tools moved to 004)
- **Delivers:** FEAT-014, FEAT-005; US-025, US-056, US-057, US-059, US-060, US-061
- **Architecture:** `docs/architecture/overview.md` § What it is, § Component map; `docs/architecture/subsystems.md` § Backend clients, § Chat protocols (planned), § Options layer, § Streaming and reasoning; `docs/architecture/interfaces.md` § Exposed: client method contract, § Protocol axis, OpenAI family (planned), § Exposed: factory, § Exposed: warnings (planned), § Exposed: error contract, § Consumed: backend APIs; `docs/architecture/data-model.md` § Message, § Settings, § Reasoning tag convention; `docs/architecture/quick-reference.md` § Client classes and protocols (planned)
- **Depends on:** `001.core-fixes`

## Definition
The app chooses the chat protocol when building an OpenAI or llama-swap client; without a choice it gets Responses, and chat completions stays available by name for servers that do not offer Responses. Today's classes keep working under chat-completions names; the existing names now speak Responses. Asking for a protocol on the deprecated Ollama backend is refused with the reason. On Responses, chat, generate and JSON-only answers, streamed and not, give the app the same kind of result as on chat completions; streamed reasoning arrives marked and separate from the answer text; a failure the server reports inside the stream ends with the library's error carrying its reason. Only settings the app set are sent; a setting with no Responses counterpart is ignored and the developer is warned; thinking on asks for reasoning (OpenAI: its summary), thinking off asks for the lowest reasoning the model allows. The library asks the provider not to keep the conversation.

## Scope
**In:**
- Class rename to chat-completions names, plus exports
- Factory protocol choice and refusals (DEC-arch-1, 4)
- System instruction placement (DEC-arch-3)
- Warning category (DEC-arch-6)
- Request flags (DEC-arch-7)
- Settings mapping and thinking (DEC-arch-8)
- Shape translation for messages and response format, minus tools (DEC-arch-10)
- Stream event parsing for text, reasoning, failure, minus tool events (DEC-arch-11)
- Existing unit tests moved to chat-completions names (DEC-arch-12)
- Live tests for chat, generate and JSON-only on both protocols
- US-025.AC-5 / AC-6 met on Responses

**Out:**
- Responses tool loop and tool-call events (004)
- Chat-completions behaviour changes beyond the rename
- Ollama
- Removing chat completions

## Open questions for the planner
- The warning category's public name (quick-reference `_TBD`).
- Thinking-off effort value per OpenAI model, verified by live test (DEC-arch-8 open sub-point).
- How thinking on/off reaches llama-swap over Responses, with a live check (DEC-arch-8 open sub-point).
- Which reasoning model / env var the live tests use (DEC-arch-12 names `OPENAI_REASONING_MODEL`).
- US-057.AC-1's tool-conversation part is deferred to 004; the plan should say so.
<!-- roadmap:end -->
