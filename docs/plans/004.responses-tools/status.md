# Feature 004 — responses-tools

| Step | File                                     | Status  | Verifier | Date |
|------|------------------------------------------|---------|----------|------|
| 001  | `001.openai-responses-tools.md`          | pending | —        | —    |
| 002  | `002.openai-responses-tools-stream.md`   | pending | —        | —    |
| 003  | `003.llamaswap-responses-tools.md`       | pending | —        | —    |
| 004  | `004.llamaswap-responses-tools-stream.md`| pending | —        | —    |

## Files Changed

_populated by the coder as steps complete_

## Notes & Issues

_populated by the coder when worth saying_

## Ultra phase

- orient: done — waves: W1 001,003 · W2 002,004 · mode: serial (0 pairable steps: 001/002 share llm/openai.py, 003/004 share llm/llama_swap.py) · typecheck isolated
- STOPPED before harvest (plan-level): context.md requires 003's builder to send assistant turns typed `message`; built 003 (bdb35d3) follows DEC-003-11 as written — bare `{role, content}` items for every turn incl. assistant (DEC-003-11 was never corrected). Also DEC-004-3 (stay 0.2.0) conflicts with shipped DEC-003-14 (0.3.0). Awaiting user decision.
- stop resolved 2026-11-08 (user): 003 corrected first (assistant items typed, version back to 0.2.0) — committed as a 003 fix; resume at harvest
