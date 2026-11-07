## Product

- US-057.AC-1 — tool-conversation part delivered: a tool conversation, streamed and not, gives the same kind of result on both protocols on OpenAI and llama-swap (live). The interim refusal of tool conversations on the default protocol (003's outcome) is withdrawn.
- US-058 — AC-1..AC-5 met: a two-round tool conversation completes on an OpenAI reasoning model and on llama-swap (live). Each round's reasoning goes back to the server within the call. The app's conversation and the returned answer hold no reasoning.
- UC-013 A2 — delivered.
- US-023.AC-1, AC-2; US-027.AC-1, AC-2; US-030; US-031; US-032; US-033; US-034; US-035.AC-1..AC-3; US-037 — hold on the default protocol as on chat completions.
- US-057.AC-1 — the modification recorded by 003 for incomplete replies also applies to tool rounds: a round the server marks as incomplete ends the tool conversation with the library's error carrying the server's reason. Confirm with the user together with 003's item.
- US-061 — the developer is also told about ignored settings when the first call to drop them is a tool conversation.

## Architecture

- `interfaces.md` § Protocol axis, OpenAI family — the "reasoning carried between tool rounds" and "`response_format` in tool chat" rows go from planned to built.
- `interfaces.md` § Exposed: error contract — remove the interim row added by 003's outcome (a tool conversation on a Responses class is refused before the server is contacted). The tool-loop failure rows hold on both protocols. A failed or incomplete reply in a tool round, or a failure inside a tool-round stream, ends the call with the library error carrying the server's reason.
- `interfaces.md` § Exposed: client method contract / capability matrix — drop the interim note that tool conversations are unavailable on the Responses classes. Streaming in tool chat is offered on both protocols.
- `interfaces.md` § Consumed: backend APIs — on the Responses route, tool definitions are sent in the protocol's tool shape, and tool calls and their results travel as items. In the Responses stream format, tool-call items, argument fragments and reasoning items to carry over are built.
- `subsystems.md` § Tool loop — reasoning carry on Responses goes from planned to built.
- `subsystems.md` § Streaming and reasoning — Responses tool-call events are built. Clarify that tool-call fragments are assembled into calls and not delivered to the receiver, on both protocols.
- `data-model.md` § Message, § Tool definition — conversion of internal turns and tool definitions to Responses items goes from planned to built.
- `quick-reference.md` § Client classes and protocols — tool conversations on the Responses classes are built.

## Decisions

- DEC-004-1 — Flattened tools copy the app's strict flag, else send strict false
- DEC-004-2 — The two-round live test on the OpenAI reasoning model runs in the default run
- DEC-004-3 — Feature 004 keeps the package version at 0.2.0
- DEC-004-4 — Every input item 004 builds carries its type; built on the corrected 003 message items
- DEC-004-5 — Tool definitions are flattened per entry; flat entries pass through
- DEC-004-6 — The loop history keeps today's turn shapes; input items are rebuilt each round
- DEC-004-7 — Reasoning items are carried in a per-call record attached to their round's turn
- DEC-004-8 — Reading a non-streamed tool round
- DEC-004-9 — A separate stream parser per class reads tool rounds
- DEC-004-10 — Tool round request body, response format and trace
- DEC-004-11 — Stage 001 tool-loop rules are reproduced unchanged on both Responses classes
- DEC-004-12 — The ignored-setting warning also covers tool conversations
- DEC-004-13 — Live tool tests: files, models, tools and settings
- DEC-004-14 — Test seams for Responses tool rounds

## Observations

_appended by coder / fixer / reviser; each bullet tagged [product] | [architecture] | [decision]_
