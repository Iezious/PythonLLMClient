# 002.edge-fixes — Per-operation edge fixes
<!-- roadmap:start -->
- **Stage:** 001.fixes · **Track:** multi-step · **Size:** M
- **Delivers:** FEAT-006, FEAT-007, FEAT-008, FEAT-009, FEAT-011, FEAT-012; US-034, US-035, US-039, US-042, US-043, US-046, US-049, US-050, US-051, US-054
- **Architecture:** `docs/architecture/subsystems.md` § Tool loop, § Tracer (`LLMTracer`), § Utilities, § Call structure of an operation; `docs/architecture/interfaces.md` § Tool loop contract, § Exposed: error contract, § Exposed: utilities, § Exposed: capability matrix; `docs/architecture/data-model.md` § Tool definition, § Trace record
- **Depends on:** `001.core-fixes`

## Definition
Fix the defects at the edges of individual operations. In a tool conversation, reaching the round limit says the limit was the reason; a request for a tool that was not offered, argument text that cannot be read, or a result that cannot be turned into text ends with the library's error; streamed tool requests without position numbers or repeating a name run as separate, correct calls. A valid single-value JSON reply is not misreported as invalid. An embedding batch gives exactly one vector per text in the texts' order, or the library's error; a malformed embedding reply or model list ends with the library's error. Tracing never changes a call's outcome: an unwritable record lets the call go ahead untraced, a record-write failure never hides the call's own error, concurrent calls get their own records, and the system instruction appears once. Repeated template filling stops once nothing more can be filled.

## Scope
**In:**
- DEC-000-25, 27, 28, 29, 32 tool loop (both clients)
- DEC-000-38 single-value misreport
- DEC-000-40 embedding correctness (not the empty-batch gap)
- DEC-000-41 model list
- DEC-000-45, 47, 49 tracing
- DEC-000-52 repeated-fill stop

**Out:**
- Sending the expected JSON shape to the server (US-040)
- Empty embedding batch short-circuit (US-042.AC-4)
- Repeated-fill top-level export (US-054.AC-4)
- Observing tool calls as they happen
- Deprecated Ollama tool-loop turn defect
- Any Responses work

## Open questions for the planner
- Exact behaviour for tool-call fragments with no position: how separate calls are told apart (DEC-000-32).
- Whether embedding / model-list fixes also apply to the deprecated Ollama client or only OpenAI / llama-swap.
- How concurrent-record separation is tested deterministically.
<!-- roadmap:end -->
