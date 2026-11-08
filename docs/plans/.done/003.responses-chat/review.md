# Rewrite review — 003.responses-chat (with 004.responses-tools, 005.responses-live-tests)

Hosted in 003 for the finalize batch 2026-11-08 (docs/.cache/finalize/batch-2026-11-08/plan.md, user answer 3 "fix first, then finalize"). Items R2–R3 also touch code delivered by 004 and 005. No behaviour change in any item.

## Session 2026-11-08 — Finalize code gaps (docs in code, README)

### R1 — README describes the delivered Responses classes   [target: README.md]
- Concern: README still describes behaviour that 004 removed and an outdated file tree.
- Files: README.md
- Desired change: (a) Protocols section (~L28-32) and Tool Calling (~L142-143): drop the "Interim limitation" — `OpenAIAPIClient` / `LlamaSwapAPIClient` support `chat_with_tools` (non-streamed and streamed, reasoning carried between rounds, `response_format` accepted); keep the chat-completions classes as the alternative protocol, not as the only tool path. (b) JSON-format table (~L217-221): add the Responses classes. (c) Architecture tree (~L276-283): match the package as listed in root AGENTS.md "Repository Structure" (add `tools.py`, `py.typed`, the `*_chat_completions.py` modules, `trace.py` if missing); `llm_client.py` described as the shared base class (no `abc`). Facts only from docs/architecture/interfaces.md (capability matrix, error contract) and the code.
- Constraints: no code change; no new claims beyond interfaces.md; keep README's existing structure and tone.
- Out of scope: rewriting other README sections; version text.
- STATUS: DONE
- Changes: README.md — interim limitation removed; Tool Calling intro; JSON-format table + Responses rows; tree adds tools.py, py.typed, llm_client.py shared base (no abc). Verified PASS 2026-11-08 (1575 passed, 3 xfailed, 2 xpassed).

### R2 — Test docstrings cite decision ids, not plan files   [target: tests/, tests/integrations/]
- Concern: docstrings and comments point at plan files (`docs/plans/…`, bare `NNN.context.md`), which leave the agent-visible tree when the folders are archived.
- Files: tests/test_llamaswap_responses_tools.py (~L10), tests/test_llm_tools_errors.py, tests/integrations/test_responses_settings.py (~L8, 94, 129, 242, 430), tests/integrations/test_responses_output_limit.py (~L3), tests/integrations/test_responses_tools_extra.py (~L133, 209); plus any other `docs/plans/` or `NNN.context.md` / step-file citation found by grep in tests/**/*.py and llm/**/*.py.
- Desired change: replace each citation with the DEC id(s) or architecture § that records the same fact (read the cited plan section to find it: docs/decisions/*.md, docs/architecture/*.md); where the cited fact has no DEC/architecture home, state the reason inline in one short clause instead of linking. Keep "DoD-n" tags in test names.
- Constraints: comments/docstrings only; no assertion, fixture or test-name change; every test still collected and passing.
- Out of scope: other comment rewording.
- STATUS: DONE
- Changes: 52 files (26 tests/, 19 tests/integrations/, 6 llm/) — plan-path citations replaced by DEC ids / architecture §; unmapped facts stated inline. Comments/docstrings only (AST-equal). Verified PASS 2026-11-08 (1575 passed, 3 xfailed, 2 xpassed).

### R3 — Long docstring lines in openai.py   [target: llm/openai.py]
- Concern: two docstring lines edited by the 2026-11-08 bug fix exceed the 120-char line length (≈L261 in `_responses_reply_text`, ≈L816 in `_read_responses_stream`).
- Files: llm/openai.py
- Desired change: re-wrap them to ≤120 chars, same wording.
- Constraints: docstrings only; ruff findings unchanged versus HEAD.
- Out of scope: reformatting the rest of the file.
- STATUS: DONE
- Changes: llm/openai.py — two docstring lines re-wrapped to <=120 chars. Verified PASS 2026-11-08 (1575 passed, 3 xfailed, 2 xpassed).
