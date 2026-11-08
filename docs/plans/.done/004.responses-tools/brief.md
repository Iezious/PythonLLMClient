# 004.responses-tools — Tool conversations on Responses
<!-- roadmap:start -->
- **Stage:** 002.responses · **Track:** multi-step · **Size:** M
- **Delivers:** FEAT-014, FEAT-005, FEAT-007; US-057, US-058
- **Architecture:** `docs/architecture/subsystems.md` § Tool loop, § Chat protocols (planned), § Streaming and reasoning; `docs/architecture/interfaces.md` § Tool loop contract, § Streaming callback, § Protocol axis, OpenAI family (planned), § Exposed: error contract; `docs/architecture/data-model.md` § Tool definition, § Message
- **Depends on:** `003.responses-chat`, `002.edge-fixes`

## Definition
Tool conversations work on the Responses protocol, streamed and not, on OpenAI and llama-swap. Tools are offered in today's shape; the model's requests and the app's results travel in the protocol's own form. Each round's reasoning is sent back to the server within the same call where the protocol needs it, never enters the app's conversation, and is dropped when the call ends; the call returns the final answer without reasoning. Streamed tool conversations give the receiver every round's text, reasoning and tool-call fragments under the same contract as chat completions. A two-round tool conversation completes on an OpenAI reasoning model and on llama-swap.

## Scope
**In:**
- Tool definition flattening and call/result mapping (DEC-arch-10 tools part)
- Reasoning carry-over (DEC-arch-9)
- Tool-call stream events with dense indices (DEC-arch-11 tool part)
- Tool-loop unit tests and live two-round tests (DEC-arch-12)
- Stage 001 tool-loop defect behaviour reproduced on Responses

**Out:**
- Chat, generate, JSON on Responses (003)
- Observing tool calls as they happen
- Storing reasoning on caller messages
- Ollama

## Open questions for the planner
- Whether reasoning items without encrypted content (local servers) need special handling when re-sent.
- Live-test prerequisites for the two-round reasoning-model run (cost, model availability).
<!-- roadmap:end -->
