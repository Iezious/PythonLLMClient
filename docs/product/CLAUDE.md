# Product layer contract — doc root `system` (system root; single-service: also the service root)

## Question this layer answers

WHAT the product does, for WHOM, and WHY. Altitude: observable behaviour, business rules.

## What it holds

Vision, actors, features, use cases, stories with acceptance criteria, glossary, product-level constraints (legal, platform, business), and — when the user specified it as a requirement during the spec discussion — an acceptance-test catalog (test ids, stage matrix, required fixtures). The catalog is a requirement the user stated, not test planning; it is written and amended only through `/product-spec`.

## Acceptance criteria

An acceptance criterion or catalog entry states an **observable outcome** an actor can see. Numbers appear only when the number itself is a business requirement the user stated as such ("a session expires after 30 minutes of inactivity" because that is the policy). A number that exists to make a behaviour testable or to bound an implementation is a decision, not a requirement — the catalog entry says "the old connection closes after its last in-flight call completes"; the `DEC` entry says "within 5 s, measured by T51c".

## Never in product

- library or framework names
- protocol internals
- configuration keys
- type or field names
- tolerances chosen for testability
- delivery history ("amended by delta 16", "step 013")

**No links to architecture.** Product docs carry nothing about architecture — no "resolved, see `docs/architecture/x.md`", no mirrored summary of a design, no status that changes when a design changes. No open questions "for the architect": open requirement questions stay `_TBD:` here until the user answers them; open design questions live in `docs/decisions/` with `Status: open` or in `quick-reference.md` `## Open`, never in product. A text change in this layer never forces an edit in architecture.

**Finalization feeds product only product items**: a requirement delivered with modification, a new behaviour, a withdrawal — by id, in requirement language. Mechanism, numbers and test tolerances learned during delivery go to `docs/decisions/`, never into the catalog or the stories.

## Classification test

Apply to any sentence about to be written into documentation, in order:

1. **Would an actor or the business notice if this changed?** → product. (If the only reason to notice is that a test would fail, it is not product.) — **this layer's test.**
2. **Does it add, remove or change a service, a component, a contract between components (API, DTO, message, config key, protocol, call order), or a rule that spans components?** → architecture.
3. **Would it still be true if the component were re-implemented with a different library or framework?** Yes → architecture (if it passed 2) · No → decision.
4. **Does it name a type, method, field, default value, timeout, event id, test id or step id?** → decision, unless the name is itself a public contract (then architecture names the contract, a decision names the mechanism).
5. Otherwise it is below the decisions layer: a folder convention or a code comment.

## Ids

- `ACT-###` actors · `FEAT-###` features · `UC-###` use cases · `US-###` stories · `US-###.AC-#` acceptance criteria.
- Zero-padded, permanent, never reused. Withdrawal = tombstone (the id stays, marked withdrawn).

## Doc-root id space

Ids are unique within this doc root only. A citation into another doc root is qualified with its key: `system/ACT-002`, `billing/FEAT-004`. Actors live only at the system root.

**Single-service case (this repository):** the root is both the system and the only service doc root, so actors live here (`actors.md`) and are cited unqualified (`ACT-001`).

## Files

- `vision.md`
- `actors.md`
- `features.md`
- `use-cases.md` (or `use-cases/` when it grows)
- `stories.md` (or `stories/` when it grows)
- `glossary.md`

Generated content sits between the merge markers `<!-- product-spec:start -->` and `<!-- product-spec:end -->`; text outside the markers is preserved by re-runs.

## Writers

- Writer: `product-spec-writer` (via `/product-spec`, `legacy-docs`, `deep-docs`, `/finalize`).
- Reviewer: `product-spec-reviewer`.
- Orchestrators never write here directly.

_Generated from ~/.claude/docs-layers.md by legacy-docs-bootstrapper, 2026-11-07._
