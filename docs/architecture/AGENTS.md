# Architecture layer contract — doc root `system` (system root; single-service: also the service root)

## Question this layer answers

HOW the system is shaped. Altitude: services, components, contracts between them.

## What it holds

The system's parts and their responsibilities; boundaries between them; **contracts between parts** — public APIs, DTOs and messages, configuration document shape (keys and meaning, not defaults), protocols, call structure and ordering between components; cross-cutting models (security, failure, identity, observability — which signals exist, not their ids); stack choices at the level of "this library is built on Kestrel"; deployment topology; parity and invariants; what is deliberately out of scope. Every choice with its reasoning.

**Architecture is what changes when a new service, contract, DTO, call structure or cross-component rule is introduced.** If a feature needs one of those and the architecture does not have it, the architecture is incomplete — planning stops until `/architect` adds it.

## Never in architecture

- a type, class, method or field internal to one component
- a framework mechanism used to realize a contract (`SocketsHttpHandler`, `IOptionsMonitor`, `undici` dispatcher internals)
- numeric defaults, limits, timeouts, retry formulas
- log event ids and metric attribute values
- test ids
- step, DoD or review-item ids
- the history of how a feature's plan evolved

**No product ids.** Architecture is not a map of who realizes which requirement: no `Realizes:` headers, no `FEAT-` / `UC-` / `US-` citations, no requirement restated in prose. The what → where mapping lives in the plans layer. A text change in this layer never forces an edit in product.

## Classification test

Apply to any sentence about to be written into documentation, in order:

1. **Would an actor or the business notice if this changed?** → product. (If the only reason to notice is that a test would fail, it is not product.)
2. **Does it add, remove or change a service, a component, a contract between components (API, DTO, message, config key, protocol, call order), or a rule that spans components?** → architecture.
3. **Would it still be true if the component were re-implemented with a different library or framework?** Yes → architecture (if it passed 2) · No → decision.
4. **Does it name a type, method, field, default value, timeout, event id, test id or step id?** → decision, unless the name is itself a public contract (then architecture names the contract, a decision names the mechanism).
5. Otherwise it is below the decisions layer: a folder convention or a code comment.

Steps 2–4 decide architecture vs decision.

## `quick-reference.md` and decision history

`quick-reference.md` is an index of architecture — contracts, keys, ids, open architecture questions — never a second place for mechanisms or a running log. `## Decision history` sections record only revisions of **architecture-level** decisions, in one line each, pointing at the `DEC` entry that holds the detail.

## Files at this doc root

Single-service case: this root has no network layer (no `services.md` / `cross-service.md`); it uses the service-root set:

- `overview.md` — always
- `subsystems.md` — components and boundaries
- `interfaces.md` — contracts the service exposes and consumes
- `data-model.md` — persisted / exchanged data **contracts** only
- `quick-reference.md`

Sized to complexity; not every file is mandatory.

## Writing rules

- State facts with evidence (`path:line`, a code_index result, or a user statement), kept to contract level.
- Unknowns: `_TBD: <reason>_`.
- No marketing prose.
- No code recommendations.

## Writers

- `docs-section-writer` (legacy-docs, deep-docs gap patches).
- `architect` agent (`/architect`, `/finalize`).
- Reviewer: `docs-reviewer`.

_Generated from ~/.claude/docs-layers.md by legacy-docs-bootstrapper, 2026-11-07._
