## Product

_none_

## Architecture

_none_

## Decisions

- DEC-005-1 — Feature 005 changes no library code; live failures are findings
- DEC-005-2 — New live files share one helper module; existing files are not migrated
- DEC-005-3 — Live servers, models and environment variables for feature 005
- DEC-005-4 — llama-swap tests turn thinking off; the OpenAI plain model gets no thinking setting
- DEC-005-5 — One test function per backend; skip only for a missing OpenAI key
- DEC-005-6 — Assertions are structural; two content checks are allowed
- DEC-005-7 — Structural checks for system instruction, scripts, streaming and embeddings
- DEC-005-8 — Inputs for the failure and ignored-setting tests
- DEC-005-9 — The llama-swap json_schema checks are expected failures, not strict
- DEC-005-10 — Concurrent calls and reuse after the usage scope
- DEC-005-11 — Extra live tool tests: tools, history and answer format
- DEC-005-12 — A reply cut at the app's limit: chat and generate checks on both backends
- DEC-005-13 — A tool conversation whose final answer is cut: prompt and checks
- DEC-005-14 — A cut checked JSON answer ends with the invalid-JSON error, no expected failure
- DEC-005-15 — No live check of a length stop without an app limit
- DEC-005-16 — Cut tests run only after the library steps; the tool and JSON cuts get their own step

## Observations

_appended by coder / fixer / reviser; each bullet tagged [product] | [architecture] | [decision]_

- [product] (step 002, verify run 1, finding per DEC-005-1) On llama-swap, a reply cut off by the token cap is returned as a normal, complete answer instead of being reported as an error, both non-streamed and streamed. US-024.AC-2 and US-060.AC-1/AC-2 are unmet on the llama-swap Responses class; OpenAI meets them. The server marks the capped reply completed, with output tokens equal to the cap and no incomplete reason.
- [decision] (step 002, finding) DEC-003-12 / DEC-003-13 detect a cut-off only from a non-streamed incomplete status or a streamed incomplete event / missing completion; llama.cpp sends neither on a cap. Open options: detect it on llama-swap from usage (output tokens reaching the requested cap); accept and document the server limit (story amendment or DEC, possibly a non-strict xfail like DEC-005-9); or treat it as an upstream llama.cpp defect.
- [decision] (steps 002/004, DEC-005-9) The llama-swap `chat_json` json_schema check XPASSed in verify run 1 while the `chat` and streamed-tool json_schema variants still xfailed — the llama.cpp schema-grammar limit looks prompt- or path-dependent rather than total.
- [product] (step 002, user decision 2026-11-08) A reply cut off by the token cap the caller set is a valid result, returned as text, not an error — the caller set the limit and must expect a cut answer. Supersedes the cut-off-is-an-error reading of US-024.AC-2 / US-060.AC-1/AC-2 for the token-cap case, on both backends; today's OpenAI behaviour (LLMError on `max_output_tokens`) then deviates. For /product-spec.
- [decision] (replan 2026-11-08, planner) The token-cap observations above are resolved: the requirement is US-062 (product) and DEC-arch-13 (decision, including the accepted llama-swap gap that answers the open options of the second bullet). Step 002 was replanned against them and step 005 added (DEC-005-12 .. DEC-005-16); no further action is needed for these bullets beyond confirming them at finalization.
