## Product

- US-034.AC-3: met. The round-limit error states the limit as the reason. US-034 is then fully delivered.
- US-035.AC-1..AC-4: met on the OpenAI and llama-swap clients. US-035 is then delivered.
  - The deprecated Ollama client keeps its existing behaviour for unoffered tools, unreadable arguments and unserialisable results.
  - "Offered" means described to the model. A function the app supplied without describing it to the model counts as not offered.
- US-039.AC-3: met. US-039 is then fully delivered.
- US-042.AC-2, AC-3, AC-5: met. US-042.AC-4 (empty batch) remains not delivered.
- US-043.AC-3, AC-4: met. US-043 is then fully delivered.
- US-046.AC-2, AC-3: met for both model names and full model descriptions. US-046 is then fully delivered.
- US-049.AC-6: delivered with modification. The instruction appears once among the record's instruction and conversation sections. The request shown in the record, recorded as it was sent, still contains it. US-049 is then fully delivered.
- US-050.AC-2, AC-3: met. US-050 is then fully delivered.
- US-051.AC-1: met. US-051 is then delivered.
- US-054.AC-3: met. US-054.AC-4 (top-level export) remains not delivered.

## Architecture

- `docs/architecture/interfaces.md` § Exposed: error contract, "Today" column. These rows become "as intended" on the OpenAI and llama-swap clients, with the deprecated client noted as unchanged:
  - tool requested that was not offered
  - tool arguments unreadable
  - tool result not serialisable
  - round limit exhausted
  - 2xx reply that is not JSON on listing and embeddings
  - malformed model list
  - malformed embeddings reply
- `docs/architecture/interfaces.md` § Exposed: error contract, Other rules (Embeddings): one vector per input text in input order is met. Only the empty batch remains a known defect.
- `docs/architecture/interfaces.md` § Exposed: client method contract, `embed_batch` row: the "not met today" note narrows to the empty batch.
- `docs/architecture/interfaces.md` § Consumed: backend APIs (SSE stream format, chat completions): remove the known defect on tool-call fragments. Fragments without a position are told apart by call identity and name, and a repeated name is not duplicated.
- `docs/architecture/subsystems.md` § Tracer: remove the two known-defect notes (concurrent calls colliding, unwritable folder failing the call). State the realized rules:
  - A record that cannot be created leaves the call untraced.
  - A record-write failure ends the record but never the call.
- `docs/architecture/data-model.md` § Trace record:
  - Layout: a disambiguating suffix may follow the timestamp when a record with the same name already exists.
  - Sections: the system instruction is shown in its own section, and the message history no longer repeats it. The request is still shown as sent.
- `docs/architecture/quick-reference.md` § Open: remove the resolved known-defect lines:
  - unoffered tool
  - unreadable arguments
  - unserialisable result
  - round-limit wording
  - fragment assembly
  - model list
  - trace collision
  - unwritable trace folder

  Narrow two lines:
  - JSON-only: only "shape not sent" remains.
  - Embeddings: only the empty batch remains.

## Decisions

- DEC-002-1 — Edge fixes land in both OpenAI-family clients; Ollama is not changed
- DEC-002-2 — The round-limit error names the limit and its value
- DEC-002-3 — A requested tool that was not offered raises LLMError
- DEC-002-4 — Unreadable tool arguments raise LLMError
- DEC-002-5 — An unserialisable tool result raises LLMError
- DEC-002-6 — Streamed tool-call fragments are told apart by position, call id and name
- DEC-002-7 — chat_json drops the no-op field loop
- DEC-002-8 — Embedding replies are checked and ordered by index
- DEC-002-9 — Model-list replies are checked entry by entry
- DEC-002-10 — Unreadable 2xx bodies on listing and embeddings are mapped at the call site
- DEC-002-11 — A trace record that cannot be set up leaves the call untraced
- DEC-002-12 — A trace write failure stops the record, never the call
- DEC-002-13 — Record files are created exclusively with a counter suffix on collision
- DEC-002-14 — The tracer drops the inserted system message from the history
- DEC-002-15 — Repeated filling stops after a pass that changes nothing
- DEC-002-16 — Trace test seams: module-level open and clock

## Observations

_appended by coder / fixer / reviser; each bullet tagged [product] | [architecture] | [decision]_
