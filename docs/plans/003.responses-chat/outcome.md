## Product

- US-056 — AC-1..AC-5 met: no protocol named gives Responses; chat completions can be chosen; the choice never comes from the address; no switch on refusal; an explicit protocol on Ollama is refused with the reason.
- US-057.AC-1 — delivered with modification: chat, generate and JSON-only answers, streamed and not, give the same kind of result on both protocols on OpenAI and llama-swap (live). The tool-conversation part is deferred to feature 004; until then a tool conversation on the default protocol is refused with a message pointing to chat completions.
- US-057.AC-1 — delivered with modification: on Responses a reply the server marks as incomplete ends with the library's error carrying the server's reason, except a reply cut short at the output length limit the app set, which returns the text produced (US-062, DEC-arch-13). Needs confirmation with the user at finalization.
- US-062 — AC-1, AC-2, AC-3, AC-5 and AC-6 delivered for chat, generate and JSON-only answers, streamed and not (steps 006–007); the tool-conversation part (AC-4) is delivered by feature 004. US-062.AC-6 is not met on llama-swap: the server does not report why a reply stopped, so a reply cut at a limit the app did not set is returned as text (accepted gap, DEC-arch-13).
- US-024.AC-2 — narrowed: an early end other than a cut at the app's own output length limit still ends with the library's error.
- US-057.AC-2 — met: embeddings and model listing are unchanged on both protocols.
- US-059 — AC-1 and AC-2 met. AC-3 delivered with modification: thinking off asks for no reasoning; a model that does not allow that refuses the call with the server's reason unless the app gives an effort itself (no automatic lower value).
- US-060 — AC-1..AC-4 met on Responses.
- US-061 — AC-1 and AC-2 met; the developer is told once per setting per client.
- US-025.AC-5 — met on Responses for OpenAI (with the US-059.AC-3 modification); not changed on chat completions.
- US-025.AC-6 — met on Responses for llama-swap; not changed on chat completions.
- US-025.AC-2 — verified live on llama-swap Responses.
- FEAT-005 — the thinking-off and explicit-effort gaps close on the default protocol; they stay open on chat completions.

## Architecture

- `interfaces.md` § Protocol axis, OpenAI family — planned → built for every row except reasoning carried between tool rounds (tool feature). Thinking-control row settled: both backends receive thinking on/off and an explicit effort through the protocol's own reasoning setting; only OpenAI also asks for the reasoning summary.
- `interfaces.md` § Exposed: factory — the protocol argument is built; it is optional, its absence means Responses, and any explicit value with Ollama is refused.
- `interfaces.md` § Exposed: warnings — built; replace the `_TBD` with the category's exported name.
- `interfaces.md` § Exposed: error contract — "failure reported inside a Responses stream" built; add that a Responses reply reporting failure or incompleteness, streamed or not, ends with the library error carrying the server's reason, except that a reply that stops at the output limit the app set is not a failure and returns its text; add the interim row: a tool conversation on a Responses class is refused before the server is contacted until the tool feature is built.
- `interfaces.md` § Exposed: client method contract / capability matrix — note the interim: tool conversations on the Responses classes are unavailable until the tool feature; the chat-completions classes keep them.
- `interfaces.md` § Consumed: backend APIs — the Responses route is in use for chat, generate and JSON-only operations (tool operations follow with the tool feature); a Responses stream that ends without its completion event ends the call with the library error.
- `subsystems.md` § Backend clients, § Chat protocols, § Options layer, § Streaming and reasoning — planned → built for the five-class shape, the protocol choice, the Responses settings translation with the ignored-setting warning, and Responses stream parsing without tool events.
- `data-model.md` § Settings, § Reasoning tag convention — the Responses classes are built; drop "planned" and the open thinking-off / llama-swap points.
- `quick-reference.md` § Public exports, § Client classes and protocols, § Settings keys, § Open — the two chat-completions classes and the warning category are exported (name settled); the protocol table is built; remove the open item on the thinking-off value and llama-swap thinking on Responses.

## Decisions

- DEC-003-1 — Thinking off on Responses asks for effort none; an explicit effort wins
- DEC-003-2 — llama-swap receives thinking on Responses as the reasoning effort, like OpenAI
- DEC-003-3 — Feature 003 is built after feature 002; the rename copies the fixed classes
- DEC-003-4 — Tool chat on the Responses classes is refused until feature 004
- DEC-003-5 — The ignored-setting warning is LLMSettingIgnoredWarning
- DEC-003-6 — The warning is emitted at call time, once per setting per client instance
- DEC-003-7 — Chat-completions classes live in their own modules as renamed copies
- DEC-003-8 — Factory protocol argument: last keyword, None means not passed
- DEC-003-9 — Responses settings table and wire mapping
- DEC-003-10 — Response format is translated to the Responses text format
- DEC-003-11 — Responses request body: exact keys and input items (amended 2026-11-08: assistant items typed `message`)
- DEC-003-12 — Non-streamed Responses reply: text extraction and failed or incomplete status
- DEC-003-13 — Responses stream handling: events, completion, markers and failures
- DEC-003-14 — Package version is 0.3.0 in both declarations (superseded by DEC-004-3, 2026-11-08: version stays 0.2.0)
- DEC-003-15 — Live tests: models and environment variables
- DEC-003-16 — Test seams for the Responses classes
- DEC-003-17 — A keyword flag tells the Responses readers that the app set the output limit (extension 2026-11-08, steps 006/007)
- DEC-003-18 — Only the exact reason max_output_tokens counts as a cut at the app's limit (extension 2026-11-08)
- DEC-003-19 — A cut at the app's limit reads like a completed reply, streamed and not (extension 2026-11-08)
- DEC-003-20 — The output-limit flag name is shared with feature 004's tool-round readers (extension 2026-11-08)

## Observations

_appended by coder / fixer / reviser; each bullet tagged [product] | [architecture] | [decision]_
