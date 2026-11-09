## Product

## Architecture

- `docs/architecture/interfaces.md` § Exposed: error contract — once delivered, drop the "agreed and not yet built" status note and bring the Today column in line with the Intended column for the error kind and status, the tolerant tool mode (now the default) and strict opt-out, the observer, the silence limit and the trace switch. Reason: the plan realizes the agreed contract unchanged.
- `docs/architecture/quick-reference.md` § Public exports and § Constructor and call arguments — remove the "agreed for release 0.2.1 and not yet built" notes. Reason: as above.

## Decisions

- DEC-006-1 — mypy configuration: strict in pyproject, Ollama silenced per module, mypy in the dev extra
- DEC-006-2 — The typing contract is checked by a unit test that runs mypy on the package and on a sample app
- DEC-006-3 — Tool-event types live in a new module; the error kind and the observer warning live in exceptions
- DEC-006-4 — ToolEvent is a total typed dictionary; how arguments and the call id are filled
- DEC-006-5 — New constructor and factory arguments are keyword-only and follow the existing ones (refines the "last keyword" position of DEC-003-8)
- DEC-006-6 — The trace switch is resolved inside trace.py through a private construction path
- DEC-006-7 — Tool calls in the trace: one call entry per call, the model text as result, then a tool-error note
- DEC-006-8 — Tolerant observer failure: the observer-failure text replaces the call's output; one warning per failure
- DEC-006-9 — tool_errors is not validated at run time; only "raise" selects strict handling
- DEC-006-10 — The base chat_with_tools stub gains tool_errors and on_tool_event in the last tool-loop step
- DEC-006-11 — Silence-limit tests run against an in-process aiohttp server, with fixed timings
- DEC-006-12 — Live tolerant-failure test: file, tool, prompt, loop bound and assertions
- DEC-006-13 — Ollama: bound to its settings shape in the class header and passes trace through; nothing else changes

## Observations

_appended by coder / fixer / reviser; each bullet tagged [product] | [architecture] | [decision]_

- [decision] Step 002: all four `chat_with_tools` loops copy the caller's turns as `[dict(m) for m in messages]` into their own `list[dict[str, Any]]` (one shallow copy per turn) instead of `list(messages)`, so a tuple or any read-only mapping sequence is accepted and the caller's sequence and its mappings are never aliased by the loop's list; rejected keeping `list(messages)` because it does not type-check against the loop's dict list and would forward non-dict mappings to the JSON body. Possible impact: DEC entry / interfaces.md § Exposed: typing contract note that the caller's turns are copied, not aliased.
- [decision] Step 006: the `TOOL CALL` trace entry is written once per call in both modes, also for `not_offered` / `bad_arguments` (raw argument text) and before a tolerant observer-failure record; after the arguments are read it carries the parsed mapping and comes before the name check (today's position). In strict mode this adds a `TOOL CALL` entry ahead of the `ERROR` section for the two early kinds, where v0.2.0 wrote none; rejected skipping it in strict mode because DEC-006-7 says "once per call before ... its failure is recorded". Possible impact: steps 007–009 should do the same so the four classes trace alike; DEC-006-7 clarification.
- [decision] Step 006: when the observer fails on the `error` event in tolerant mode, the trace holds only the observer-failure pair (`TOOL RESULT` `Tool '<n>' failed.` + `TOOL ERROR` `observer failed: ...`), not the tool failure-kind note as well; `unserialisable_result` covers any ordinary exception while turning the result into text (also a failing `__str__`), and strict re-raises that exception unchanged. Rejected writing both notes (the record would show two reasons for one text). Possible impact: steps 007–009 consistency; DEC-006-7 / DEC-006-8 clarification.
- [decision] Step 006: `LLMToolObserverWarning` text is `Tool-event observer failed on the '<event type>' event of tool '<n>': <ExceptionType>: <message>`, emitted with `stacklevel=4` (points at the app's `chat_with_tools` call site). Possible impact: steps 007–009 use the same wording; DEC-006-8 detail.
