# Planning files — layout, lifecycle, and contracts

Every planning and coding agent reads this file first and treats it as authoritative. Two tracks live here: multi-step features and fast features. Bug fixes run against an already-completed plan in either track.

## Tracks — when to use fast vs multi-step

- **Multi-step** (`<NNN>.<feature>/`) — work spanning more than one coherent change: cross-layer coordination, dependencies between sub-parts, design ambiguity needing mid-flight decisions, or more than ~300 LoC of source change. Decomposed into ordered steps by `/planner`. Built either step by step with `/coder` (each step verified before the next) or in one phased run with `/ultra-feature` (same step files and agents; harvest, red gate, test run and verification happen once for the whole feature; per-step phases run strictly in step order).
- **Fast** (`fast/<NNN>.<name>/`) — a single coder pass: one logical change, one or two test files, roughly 50–300 LoC, no internal step boundaries. Planned by `/fast-feature`. If you catch yourself writing "Phase 1 / Phase 2" or "first the schema, then the UI", it is multi-step — promote it.
- **Bug fix** — `/bug-fixer`, run against a `done` plan in either track. Adds no new scope; repairs the existing contract.

The two counters are independent: `fast/001` and `001.<feature>/` are unrelated.

## Layout

```
docs/plans/
  roadmap.md                    # optional; stage index (/roadmap)
  roadmap.<NNN>.md              # optional; one stage's feature table when roadmap.md is split
  <NNN>.<feature>/              # multi-step
    brief.md                    # optional; short definition (/roadmap)
    context.md                  # feature-wide context
    <SSS>.<name>.md             # one step file per step
    <SSS>.context.md            # one per step (never skipped)
    outcome.md                  # what the feature learned, by layer; emptied into the layers at finalization
    status.md                   # step table + lifecycle sections
  fast/
    <NNN>.<name>/               # fast feature
      brief.md                  # optional; short definition (/roadmap)
      context.md
      plan.md                   # the single plan (the step file's equivalent)
      outcome.md
      status.md
  backlog/                      # rough idea seeds; unsequenced, no commitment
  retired.md                    # ledger of finalized features (/finalize → roadmap-writer): cross-reference + number registry
  .done/                        # plan folders archived by /finalize — agent-ignored (Read/Edit deny rules): never globbed, harvested, cited or planned from
```

`NNN` / `SSS` are 3-digit zero-padded. The next free `NNN` is taken over existing folders **and** `retired.md` rows of the same track — a retired number is never reused. Letter suffixes (`001b`) are for splits/rework after the fact, not initial planning.

## Documentation layers above the plans

| Layer | Folder | Holds | Written by |
|---|---|---|---|
| Product | `docs/product/` | what / for whom / why; observable behaviour; the user's acceptance-test catalog | `/product-spec` |
| Architecture | `docs/architecture/` | services, components, contracts between them (APIs, DTOs, config shape, protocols, call structure), cross-cutting rules | `/architect` |
| Decisions | `docs/decisions/` | technical decisions below architecture and above code: mechanisms inside a component, defaults and bounds, field shapes, test tolerances, rejected options, mid-flight user calls; one file per feature, `DEC-<NNN>-<n>` ids, stage-linked | planner (planning time), `architect` agent (finalization) |

Plans cite all three and write none of them directly. A feature that needs a contract, service, DTO or call structure the architecture lacks is **not plannable** — planning stops and `/architect` runs first. A behaviour no story covers stops planning for `/product-spec`. A decision never fills either gap.

**Below decisions — where a small technical choice is recorded** is decided by who needs to know: only a reader of this block or file → a **comment in the code** (coder / fixer / reviser, the `why`, citing the step / `DEC` / architecture § it serves) and nothing else; anyone working in this folder → the folder's **`AGENTS.md` `## Conventions`** section (outside the `agent-ready` markers; planner at planning time, or the glossary writer from a coder's `[decision] Scope: folder <path>` observation); anyone outside the folder → a **`DEC` entry**. Nothing is recorded twice: a comment cites the `DEC` id, never restates it.

## outcome.md — the collection point

The planner writes four headed sections, each possibly empty, nothing else:

```
## Product
_for /product-spec finalization: requirement delivered-with-modification, new behaviour, withdrawal — by id, in requirement language_

## Architecture
_for /architect finalization: contract / component / cross-component rule changes — target doc + section, framework-neutral wording, no type names, numbers or ids_

## Decisions
_DEC entries the planner already wrote to docs/decisions/, listed by id. Mid-flight decisions (escape valve, Fault SPEC, ambiguity settled with the user) appended here by the orchestrator in DEC entry format_

## Observations
_appended by coder / fixer / reviser; each bullet tagged [product] | [architecture] | [decision]_
```

Finalization confirms each item with the user under its category and the category's owner applies it: `/product-spec` the `## Product` items and `[product]` observations; `/architect` the `## Architecture` items into `docs/architecture/` and the `## Decisions` items plus `[decision]` observations into `docs/decisions/`. An item in the wrong section is re-categorized, never applied where it sits. `/finalize` runs all owners in one session, sweeps the whole folder, and retires it once an audit proves nothing would be lost. Once applied, `outcome.md` carries only the footer; nothing in a plan folder is a second home for a decision.

## Roadmap and briefs — optional

A project may use `/roadmap` to sequence features before planning them. It is optional; a project that never runs it has no `roadmap.md` and no `brief.md`, and nothing below applies.

Where it is used:

- **`roadmap.md`** — the stage index: one block per stage (goal + exit criteria) and a table of its features with track, size, delivered product ids, and feature-level dependencies. **No status column** — status is derived from the folder, never duplicated.
- **`brief.md`** — a feature's short definition, written *before* planning: stage, track, size, `Delivers:` (the `docs/product/` ids it realizes, omitted when there's no product layer), `Architecture:` (the `docs/architecture/` docs / sections it builds against, omitted when there's no architecture layer), `Depends on:` (feature-level), a 3–6 sentence behavioural Definition, Scope In/Out, and Open questions for the planner. It is a definition, never a plan: no steps, no Definition of done, no signatures, no file lists. **`Delivers:` + `Architecture:` are the project's spec ↔ architecture ↔ feature cross-reference** — architecture docs carry no product ids and product docs carry no architecture pointers; this is where "what is realized where" is answered.
- **Splitting the roadmap.** `roadmap.md` stays the stage index. A stage whose feature table outgrows it moves to `roadmap.<NNN>.md` (same fences, same columns); `roadmap.md` keeps the stage header, goal, exit criteria and a pointer to the part file. Numbers and stage ids are unaffected.
- **A folder containing only `brief.md` is roadmapped, not planned.** The absence of `status.md` is the signal. `/roadmap` never creates a planner-owned file.

Derived feature state, for any agent that needs it:

| Folder contents | State |
|---|---|
| `brief.md` only | roadmapped — `/planner` or `/fast-feature` has not run |
| `status.md` present | planned — the pipeline owns it from here |
| `status.md` rows all `done` with verifier `PASS` | delivered |

`/planner` and `/fast-feature` **read `brief.md` when present** and treat its Definition and Scope as the feature's agreed boundary rather than re-deriving it — the Out list bounds the plan, and the Open questions are what the harvest must resolve. Neither ever edits it. A brief that turns out wrong is a `/roadmap` re-shape, not a plan-time correction.

`backlog/` and the roadmap are different states, not competitors: backlog is an unsequenced idea with no commitment; a brief is a feature committed to a stage. `/roadmap` promotes a backlog item into a stage.

## Step file / plan.md structure

A multi-step `<SSS>.<name>.md` and a fast `plan.md` carry the same behavioral contract:

- **Goal** — one or two sentences.
- **Source files** — explicit source paths, one per line. This list *is* the coder's scope; the coder touches nothing else.
- **Test files** — explicit test paths, one per line. This list *is* the test-coder's scope. **Must be disjoint from Source files** — neither role crosses into the other's list.
- **Interface intent** — each function / class / type / endpoint the work adds or changes, named with its responsibility and inputs/outputs **in prose**. No exact type signatures — the skeleton agent freezes those.
- **Definition of done** — a numbered checklist (DoD-1, DoD-2, …) of verifiable criteria. Tag each item `[test]` (the test-coder must cover it with a test citing the id) or `[manual/live]` (no automated test; the verifier records it as requires-live-run). The set of `[test]` items is the coverage contract.
- **Dependencies** (multi-step) — earlier steps relied on, or "none".
- **Out of scope** (fast) — short list of deliberate exclusions to bound creep.

Multi-step size budget: 50–200 LoC of source change per step (test volume isn't counted). Fast: 50–300 LoC total.

## Pipeline and ownership — the air gap

Plans run through a chain of single-purpose agents with a strict separation between who writes tests and who writes code:

1. **planner / fast-planner** — writes the plan (sections above). Owns the plan file and the seeded `status.md`.
2. **skeleton / fast-skeleton** — reads Interface intent + code, writes compilable stubs (unimplemented bodies), and **freezes the exact signatures** into a `## Skeleton` record in `status.md`. Owns `## Skeleton`.
3. **test-coder / fast-test-coder** — writes tests from the spec ALONE, bound to the `## Skeleton` signatures, each tagged to a `[test]` DoD id. **Never reads source.** Owns `## Tests` and the Test files.
4. **verifier (red-gate run)** — runs the new tests against the stubs: confirms they compile, cover every `[test]` item, and fail for the right reason.
5. **coder / fast-coder** — fills the stub bodies, **blind to the tests**, runs build/typecheck only. Owns `## Files Changed` and the Source files. May not change a frozen signature (re-route to skeleton if it must).
6. **verifier (verify run)** — the **only** role that runs the tests as a gate. Emits PASS/FAIL plus a **Fault**: `CODE` (back to coder), `TEST` (back to test-coder, then re-red-gate), or `SPEC` (back to the user — the plan itself is wrong). Failure summaries are by-DoD-clause and never quote test internals, so the air gap holds.

The invariant: **expected values come from the spec, never from code; the interface tells you only how to call, never what to expect.** The test-coder never sees the implementation; the coder never sees the tests; the verifier is the sole bridge and relays results, never code or test source.

## status.md

Seeded by the planner. Multi-step:

```
# Feature <NNN> — <name>

| Step | File            | Status  | Verifier | Date |
|------|-----------------|---------|----------|------|
| 001  | `001.<name>.md` | pending | —        | —    |

## Files Changed

_populated by the coder as steps complete_

## Notes & Issues

_populated by the coder when worth saying_
```

Fast:

```
# Fast feature <NNN> — <name>

| Status  | Verifier | Date |
|---------|----------|------|
| pending | —        | —    |

## Files Changed

_populated by the coder when implementation lands_

## Notes & Issues

_populated by the coder when worth saying_
```

As the pipeline runs, agents append their own sections (below `## Files Changed`): `## Skeleton` (frozen signatures, skeleton), `## Tests` (test inventory, each tagged to a DoD id, test-coder), and `## Bug Fixes` (one entry per bug fix against a `done` plan, coder/fixer). A multi-step feature built by `/ultra-feature` also carries `## Ultra phase` (orchestrator-owned resume record: which phase completed for which steps).

### Status lifecycle

`pending` → `wip` (coder may set mid-step) → `done` (orchestrator, on verifier PASS) or `blocked` (skeleton/coder escape valve, or Fault: SPEC). Only these four values. A bug fix never un-completes a `done` plan — its record is the `## Bug Fixes` and `## Tests` entries.

## Code documentation — required on all code

Every file, class and method the pipeline writes or changes is documented in the language's idiomatic doc format (this project: Python docstrings; a convention in the root `AGENTS.md` wins):

- **File header** — 1–3 lines: what the file is for and which problem it solves.
- **Classes / types** — responsibility, why it exists, key invariants.
- **Methods / functions** — contract (not mechanics), parameters, return, errors raised, side effects.
- **Decision comments** — every non-obvious block (workaround, ordering constraint, invariant, concurrency reason, trade-off, magic number, deliberate deviation) gets a comment explaining **why**, citing the step/DoD id or architecture doc it came from.
- No narration of obvious code, no commented-out code, no stale comments.

Ownership: the skeleton documents the frozen signatures and new files from the spec; the coder/fixer/reviser add decision comments and refine (never contradict) the contract docs; the test-coder documents test files, tests and non-obvious fixtures. Verifiers enforce it as a Layer-1 convention: missing documentation on touched code fails (CODE for source, TEST for test files).

## Folder glossaries — `AGENTS.md` in every code folder

Every code folder (holds source or test files; build output, dependencies, caches, dotfolders and `docs/` excluded) carries an `AGENTS.md`: 1–2 sentences on the folder's purpose, then a glossary with one line per file and immediate subfolder stating the reason it exists and the problem it solves. Purpose, not implementation — so every agent, on either side of the air gap, may read it. Auto-generated content sits inside `<!-- agent-ready:start -->` / `<!-- agent-ready:end -->`; content outside the markers is user-owned and never edited by the glossary pass — except one section: **`## Conventions`**, outside the markers, holding folder-scoped technical conventions one line each (planner writes them at planning time; the glossary writer appends the coder's `[decision] Scope: folder <path>` observations). It is the record for choices every file in the folder follows; block-level choices stay in code comments and feature-level ones in `docs/decisions/`.

**Every coding session updates them.** After the verify run PASSes, the orchestrator (`/coder`, `/fast-feature`, `/ultra-feature`, `/bug-fixer`, `/rewrite`) dispatches a glossary writer for every folder the session touched — creating missing files, adding/removing entries, and updating the parent folder's line when a folder's purpose changed. Coders, fixers, skeletons, test-coders and the reviser never write them; verifiers never count them as scope deviations. Harvesters read them first to locate code.

## Ownership summary

| File / section               | Written by                    | Everyone else          |
|------------------------------|-------------------------------|------------------------|
| roadmap.md, roadmap.<NNN>.md, brief.md | roadmap (optional layer) | read-only          |
| plan / step file             | planner / fast-planner        | read-only              |
| context.md, `<SSS>.context`  | planner / fast-planner        | read-only              |
| outcome.md `## Product` / `## Architecture` | planner; applied by /product-spec, /architect | read-only |
| outcome.md `## Decisions`    | planner; orchestrator appends mid-flight | applied by architect agent |
| outcome.md `## Observations` | coder / fixer / reviser (tagged) | read-only           |
| `docs/decisions/<feature>.md` | planner (planning), architect agent (finalization) | read-only |
| status.md row                | orchestrator                  | —                      |
| `## Ultra phase`             | ultra-feature orchestrator    | read-only              |
| `## Skeleton`                | skeleton                      | read-only ground truth |
| `## Tests`                   | test-coder                    | coder never reads      |
| `## Files Changed`           | coder / fixer                 | —                      |
| `## Bug Fixes`               | coder / fixer                 | —                      |
| folder `AGENTS.md` glossary  | orchestrator's glossary writer | read-only, read first  |

`docs/architecture/` and `docs/decisions/` are the architect's domain — plans reference them; only the planner writes planning-time `DEC` entries.
