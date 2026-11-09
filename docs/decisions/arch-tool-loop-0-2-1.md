# Decisions — arch-tool-loop-0-2-1

**Stage:** — · **Feature:** — (architect session `tool-loop-0-2-1`, 2026-11-09; not yet planned)

Decisions taken with the user while extending the tool loop, the library error, the time limits, the trace switch and the typing contract of the OpenAI-family clients. They sit below the architecture recorded in `docs/architecture/` (§ references in each entry). The design is agreed and not yet built; "Lands in" names the code locations as expected at design time, and the planner that builds it may refine them. Ids continue the shared `arch` counter (`DEC-arch-<n>`, after DEC-arch-13 in `arch-responses-protocol.md`).

### DEC-arch-14 — Tolerant-mode model-facing texts are fixed per failure kind
- **Status:** accepted — amended by DEC-arch-28
- **Stage:** — · **Feature:** — (arch-tool-loop-0-2-1) · **Step:** —
- **Date:** 2026-11-09 · **Decided by:** user
- **Question:** In the tolerant tool-failure mode, what text does the model receive as the failing call's output?
- **Decision:** A short fixed text per failure kind, with `<n>` the requested tool name:
  - `not_offered` → `Tool '<n>' is not available.`
  - `bad_arguments` → `Invalid arguments for tool '<n>'.`
  - `unknown_argument` → `Tool '<n>' has no argument '<arg>'.`, naming the first argument the model sent that the callable does not declare.
  - `tool_raised` and `unserialisable_result` → `Tool '<n>' failed.`
  - observer failure → `Tool '<n>' failed.`
  The text never contains exception text or a server body.
- **Reason:** The model needs to know the call did not work and, for argument mistakes, what to correct; anything more would carry application internals to the provider.
- **Rejected:** Including the tool's own failure message in the text (leaks application internals to the provider).
- **Lands in:** the tool loop of `chat_with_tools` in `llm/openai.py`, `llm/llama_swap.py`, `llm/openai_chat_completions.py`, `llm/llama_swap_chat_completions.py`
- **Cites:** docs/architecture/interfaces.md § Tool failures; docs/architecture/data-model.md § Tool events; DEC-000-26 (amended: feeding the failure back to the model is now the opt-in tolerant mode; strict stays the default). Amended 2026-11-09 by DEC-arch-28: tolerant is now the default and strict the opt-out; the texts above are unchanged.

### DEC-arch-15 — Tool failure handling is written into the tool loop of each class
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-tool-loop-0-2-1) · **Step:** —
- **Date:** 2026-11-09 · **Decided by:** user
- **Question:** Where do the new tool-failure handling and the tool-event observer live: in one shared execution step or in each class?
- **Decision:** In each of the four OpenAI-family classes, inside its own copy of the tool loop. No shared tool execution step is added to the base. This restates DEC-000-35 and DEC-arch-2 as the architecture rule "no shared behaviour logic between client classes" (docs/architecture/subsystems.md § Base client); it supersedes nothing else. DEC-000-35 is marked superseded in part: its rule now lives in the architecture, and its observed facts stay.
- **Reason:** The user keeps the classes independent and treats the specification, not the code, as what is maintained: "NO MIXED logic. We need to stay flexible. We are not supporting code. We do support specifications and rewrite code if needed."
- **Rejected:** One shared tool execution step in the base (couples the classes; a class could no longer be rewritten on its own).
- **Lands in:** `llm/openai.py`, `llm/llama_swap.py`, `llm/openai_chat_completions.py`, `llm/llama_swap_chat_completions.py` (each `chat_with_tools` and its own tool helpers); nothing in `llm/llm_client.py`
- **Cites:** docs/architecture/subsystems.md § Base client, § Backend clients, § Tool loop; DEC-000-35; DEC-arch-2

### DEC-arch-16 — The argument-name check runs once per call and is skipped for VAR_KEYWORD callables
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-tool-loop-0-2-1) · **Step:** —
- **Date:** 2026-11-09 · **Decided by:** user
- **Question:** How is "an argument name the callable does not declare" decided, and which callables are exempt?
- **Decision:** The callable's signature is read once per tool call. The check is skipped when the signature has a `VAR_KEYWORD` parameter (`**kwargs`), and when the signature cannot be read (`inspect.signature` fails). Otherwise every argument name the model sent must be a declared parameter. This narrows the name check of DEC-000-27, which today rejects every name on a `**kwargs` callable.
- **Reason:** A catch-all callable accepts any name, so the check would reject valid calls. A callable whose signature cannot be read cannot be checked, so the tool itself decides.
- **Rejected:** — (none weighed)
- **Lands in:** the tool loop of `chat_with_tools` in the four OpenAI-family modules (`llm/openai.py`, `llm/llama_swap.py`, `llm/openai_chat_completions.py`, `llm/llama_swap_chat_completions.py`)
- **Cites:** docs/architecture/interfaces.md § Tool loop contract, § Tool failures; DEC-000-27

### DEC-arch-17 — Strict mode fires the error event, then raises the existing exception
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-tool-loop-0-2-1) · **Step:** —
- **Date:** 2026-11-09 · **Decided by:** user
- **Question:** With `tool_errors="raise"` and an observer present, in which order do the error event and the call's exception happen?
- **Decision:** The observer receives the `error` event first. Then the call raises exactly the exception it raises today for that failure kind (`LLMError`, `ValueError` or `RuntimeError`, see the error contract). If the observer itself fails on that event, its failure propagates instead (strict-mode observer rule).
- **Reason:** The observer is told about every failing call, including the one that ends the call.
- **Rejected:** — (none weighed)
- **Lands in:** the tool loop of `chat_with_tools` in the four OpenAI-family modules
- **Cites:** docs/architecture/interfaces.md § Tool event observer, § Exposed: error contract

### DEC-arch-18 — The tool-event observer is awaited inline when it returns an awaitable
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-tool-loop-0-2-1) · **Step:** —
- **Date:** 2026-11-09 · **Decided by:** user
- **Question:** How is `on_tool_event` called: scheduled, or inline?
- **Decision:** It is called inline for each event. When the return value is awaitable, it is awaited before the loop goes on. This is the same mechanism as the stream receiver (DEC-000-23).
- **Reason:** One calling rule for every application callback. Event order, and "the tool does not run before its start event is handled", follow from it.
- **Rejected:** — (none weighed)
- **Lands in:** the tool loop of `chat_with_tools` in the four OpenAI-family modules
- **Cites:** docs/architecture/interfaces.md § Tool event observer; DEC-000-23

### DEC-arch-19 — LLMError kind and status mapping
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-tool-loop-0-2-1) · **Step:** —
- **Date:** 2026-11-09 · **Decided by:** user
- **Question:** Which `kind` (and `status`) does each existing `LLMError` raise site carry?
- **Decision:**
  - Non-2xx reply, on the plain and the streamed request paths → `status`, with `status` set to the HTTP status.
  - Transport timeout (total limit or silence limit) → `timeout`; interrupted transfer → `interrupted`; any other connection failure → `connection`. The text is unchanged (`Request to <url> failed: <kind>`, DEC-001-7).
  - Unreadable or malformed model-list or embeddings reply, empty embedding result, a Responses reply that is not a mapping, `chat_json` parse failure → `invalid_reply`.
  - In-stream error event or frame, and a failed reply, streamed or not → `server_failed`.
  - Incomplete reply (including a cut tool round that requests a tool, DEC-arch-13) and a stream that ends without completion → `incomplete`.
  - Tool not offered, tool arguments unreadable, tool result not serialisable → `tool`.
  - A raw `KeyError` / `IndexError` on a malformed chat-completions reply stays as accepted. Existing `RuntimeError` and `ValueError` raises are unchanged.
- **Reason:** One closed set of kinds lets an application branch on the failure without parsing text, and the text form stays as it is for existing readers. `server_failed` replaces the requested `stream_failed` because it also covers a failed non-streamed reply.
- **Rejected:** `stream_failed` as the kind name (too narrow, see Reason).
- **Lands in:** `llm/exceptions.py` (`LLMError`, `LLMErrorKind`); `llm/llm_client.py` (`_request`, `_stream_post`, `_iter_sse_data`, shared chat-completions stream parsers); the raise sites in `llm/openai.py`, `llm/llama_swap.py`, `llm/openai_chat_completions.py`, `llm/llama_swap_chat_completions.py`; `llm/__init__.py`
- **Cites:** docs/architecture/interfaces.md § Exposed: error contract; docs/architecture/data-model.md § Error kinds; DEC-000-11; DEC-001-1; DEC-001-7; DEC-001-9; DEC-002-8; DEC-002-9; DEC-002-10; DEC-arch-13

### DEC-arch-20 — The silence limit is the per-read timeout on streamed requests only
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-tool-loop-0-2-1) · **Step:** —
- **Date:** 2026-11-09 · **Decided by:** user
- **Question:** How is `read_timeout` realized, and on which requests?
- **Decision:** It sets the HTTP library's per-read timeout (aiohttp `ClientTimeout` `sock_read`) on streamed requests only, next to the unchanged total (`ClientTimeout(total=self.timeout)`). Non-streamed requests keep the total only. `None` sets no silence limit. Expiry is mapped like any transport timeout (kind `timeout`, DEC-arch-19).
- **Reason:** A non-streamed reply arrives as one body, so the total limit already covers it. The silence limit exists to catch a stream that stalls inside a long total.
- **Rejected:** A separate connection limit: considered and deferred, because no requirement asks for it.
- **Lands in:** `llm/llm_client.py` (`_stream_post`, constructor state); constructors of the four OpenAI-family classes; `llm/llm.py` (`get_llm_client`)
- **Cites:** docs/architecture/interfaces.md § Exposed: factory, § Consumed: backend APIs (SSE stream format); docs/architecture/subsystems.md § Base client; DEC-000-15; DEC-001-8

### DEC-arch-21 — Tolerant tool failures and observer failures in the trace
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-tool-loop-0-2-1) · **Step:** —
- **Date:** 2026-11-09 · **Decided by:** user
- **Question:** What does the trace record when a tool failure is handled in tolerant mode, or when an observer fails?
- **Decision:** The model-facing text (DEC-arch-14) is recorded as that call's tool result, followed by an error note naming the failure kind. An observer failure is recorded too. Trace failures still never change the call's outcome (DEC-002-11, DEC-002-12).
- **Reason:** The trace must show what the model was actually told, and why.
- **Rejected:** — (none weighed)
- **Lands in:** the tool loop of `chat_with_tools` in the four OpenAI-family modules; `llm/trace.py` only if a new record entry is needed
- **Cites:** docs/architecture/subsystems.md § Tracer; docs/architecture/data-model.md § Trace record; DEC-002-11; DEC-002-12

### DEC-arch-22 — Strict type check with mypy, Ollama module excluded
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-tool-loop-0-2-1) · **Step:** —
- **Date:** 2026-11-09 · **Decided by:** user
- **Question:** Which checker proves the typing contract, and how is the deprecated module handled?
- **Decision:** `mypy --strict llm` is the gate. mypy joins the `dev` extra, and a `[tool.mypy]` configuration goes into `pyproject.toml`. The deprecated `llm/ollama.py` is excluded through a per-module override. The planner/coder that builds it adds the command to root `AGENTS.md` Common Commands at delivery.
- **Reason:** The public typing contract (docs/architecture/interfaces.md § Exposed: typing contract) needs a checker that enforces it. The deprecated module is not extended, so fixing its errors is out of scope.
- **Rejected:** — (none weighed)
- **Lands in:** `pyproject.toml` (`dev` extra, `[tool.mypy]`, per-module override), `uv.lock`, root `AGENTS.md` (Common Commands, at delivery)
- **Cites:** docs/architecture/interfaces.md § Exposed: typing contract

### DEC-arch-23 — Package version is 0.2.1 in both declarations, tagged v0.2.1
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-tool-loop-0-2-1) · **Step:** —
- **Date:** 2026-11-09 · **Decided by:** user
- **Question:** Which version ships the tool-loop, error, time-limit, trace and typing additions?
- **Decision:** `pyproject.toml` `version` and `llm/__init__.py` `__version__` both become `0.2.1` and are bumped together. The release commit is tagged `v0.2.1`. Supersedes DEC-001-4 and DEC-004-3.
- **Reason:** The user's call (2026-11-09). All additions are opt-in except one default: `tool_errors` on the four OpenAI-family classes defaults to tolerant (DEC-arch-28), so existing callers now have tool-call failures sent to the model instead of an exception unless they pass `tool_errors="raise"`. The user keeps 0.2.1 despite this behaviour change. (Reason amended 2026-11-09 with DEC-arch-28; it earlier said every default keeps today's behaviour.)
- **Rejected:** 0.3.0 (first proposed; corrected by the user).
- **Lands in:** `pyproject.toml`, `llm/__init__.py`; git tag `v0.2.1`
- **Cites:** DEC-001-4; DEC-004-3; DEC-000-53

### DEC-arch-24 — Ollama refuses the new tool parameters by not having them; the factory refuses read_timeout with ollama
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-tool-loop-0-2-1) · **Step:** —
- **Date:** 2026-11-09 · **Decided by:** user
- **Question:** How does the deprecated Ollama client refuse `tool_errors`, `on_tool_event` and `read_timeout`?
- **Decision:** `OllamaAPIClient.chat_with_tools` does not declare `tool_errors` or `on_tool_event`, so passing them raises `TypeError`, the same precedent as `on_delta`. Its constructor does not declare `read_timeout`. The factory raises `ValueError` when `read_timeout` is given with `server_type="ollama"`. Ollama's constructor accepts `trace` and only passes it on to the base.
- **Reason:** The client is deprecated and not extended. The absence of a parameter is the existing refusal precedent. The factory refuses explicitly because a silence limit on a client without streaming has no effect.
- **Rejected:** — (none weighed)
- **Lands in:** `llm/ollama.py` (constructor: `trace` pass-through only), `llm/llm.py` (`get_llm_client`)
- **Cites:** docs/architecture/interfaces.md § Exposed: capability matrix, § Exposed: factory; DEC-000-42

### DEC-arch-25 — The settings merge in the base is treated like the shared stream parsers
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-tool-loop-0-2-1) · **Step:** —
- **Date:** 2026-11-09 · **Decided by:** user
- **Question:** The option-merge routine sits in the base but is behaviour, not transport. Under the rule "no shared behaviour logic between client classes" (DEC-arch-15), does it stay in the base or move into each class?
- **Decision:** It is handled like the existing shared chat-completions stream parsers. The routine (`_merge_options_base`) stays in the base with its current behaviour and is not extended with new behaviour. A class whose merging needs to diverge takes its own copy of the routine into its own module and changes the copy; the base routine is not changed for it. Each class's own post-processing (`_merge_options`) stays in the class, as today.
- **Reason:** One rule for all existing shared behaviour code: it is not churned, and it does not grow. A class can still change its merging on its own by taking a copy, which is what the no-shared-behaviour rule protects.
- **Rejected:** Moving the merge into each class now (a rewrite of every class with no behaviour change; the copy-on-divergence rule already keeps each class free to change on its own).
- **Lands in:** `llm/llm_client.py` (`_merge_options_base`, unchanged); a class's own module only when that class takes a copy
- **Cites:** docs/architecture/subsystems.md § Base client, § Options layer; DEC-000-5; DEC-arch-15

### DEC-arch-26 — In tolerant mode an observer failure ends that tool call's events; no observer error kind
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-tool-loop-0-2-1) · **Step:** —
- **Date:** 2026-11-09 · **Decided by:** user
- **Question:** With `tool_errors="result"`, when the tool-event observer fails, does the observer receive a further event for the same tool call (for example an `error` event), and with which kind?
- **Decision:** No further event of that tool call is sent to the observer. The observer failure is reported only through `LLMToolObserverWarning` and the trace (DEC-arch-21). `ToolErrorKind` stays at its five tool failure kinds. The model-facing text for the call is the one in DEC-arch-14 (observer failure). Every other tool call keeps the rule "a `start` event, then exactly one `result` or `error` event"; this includes `not_offered` and `bad_arguments`, where the tool never runs.
- **Reason:** An event reporting the observer's own failure would go to the observer that has just failed. The warning and the trace already carry the failure, and the tool failure kinds stay about tools.
- **Rejected:** An extra `error` event with kind `"observer_failed"` (adds a non-tool kind to `ToolErrorKind` and sends the event to the failing observer).
- **Lands in:** the tool loop of `chat_with_tools` in the four OpenAI-family modules (`llm/openai.py`, `llm/llama_swap.py`, `llm/openai_chat_completions.py`, `llm/llama_swap_chat_completions.py`)
- **Cites:** docs/architecture/interfaces.md § Tool event observer, § Exposed: warnings; docs/architecture/data-model.md § Tool events; DEC-arch-14; DEC-arch-21

### DEC-arch-27 — The per-client trace argument: three states over the environment
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-tool-loop-0-2-1) · **Step:** —
- **Date:** 2026-11-09 · **Decided by:** user
- **Question:** How does the constructor `trace` argument decide tracing, and what happens when tracing is forced on without a folder?
- **Decision:** `trace: bool | None = None` on the constructors of all five client classes and on `get_llm_client`, which forwards it to the class it picks (Ollama passes it to the base only, DEC-arch-24). The tracer is still built once, at creation:
  - `None` (not passed) → the environment decides, as in DEC-000-4 (`LLM_TRACE` true-like and `LLM_TRACE_FOLDER` non-empty).
  - `False` → never traces, whatever the environment says.
  - `True` → traces into the folder named by `LLM_TRACE_FOLDER`; `LLM_TRACE` is not consulted. When that folder is not set, creation raises `ValueError`.
  This amends DEC-000-4: the environment read is now the `None` path only.
- **Reason:** An application can switch tracing per client without changing the process environment. Refusing `True` without a folder makes the missing location visible at creation instead of silently not tracing.
- **Rejected:** A folder argument next to the switch (the user chose refusal at creation instead).
- **Lands in:** `llm/llm_client.py` (constructor, tracer construction), `llm/trace.py` (if the switch is resolved there), the constructors in `llm/openai.py`, `llm/llama_swap.py`, `llm/openai_chat_completions.py`, `llm/llama_swap_chat_completions.py`, `llm/ollama.py`; `llm/llm.py` (`get_llm_client`)
- **Cites:** docs/architecture/interfaces.md § Exposed: constructor arguments, § Exposed: factory; docs/architecture/subsystems.md § Tracer; DEC-000-4; DEC-arch-24

### DEC-arch-28 — Tool-call failures are tolerant by default; strict is the opt-out
- **Status:** accepted
- **Stage:** — · **Feature:** — (arch-tool-loop-0-2-1) · **Step:** —
- **Date:** 2026-11-09 · **Decided by:** user
- **Question:** What is the default tool-call failure handling?
- **Decision:** `tool_errors: Literal["raise", "result"] = "result"` on `chat_with_tools` of all four OpenAI-family classes, both protocols. `"result"` (tolerant) is the default; `"raise"` (strict) stays available as an opt-out. Unchanged: the tolerant and strict semantics, the observer rules (DEC-arch-17, DEC-arch-26), cancellation, the round limit, and the deprecated Ollama client, which does not accept the parameter and keeps its raising behaviour (DEC-arch-24). Accepted consequence: existing callers of the four classes now have tool-call failures sent to the model instead of an exception unless they pass `tool_errors="raise"`. The version stays 0.2.1 (DEC-arch-23, Reason amended). Amends the default stated with DEC-arch-14 (and, through it, the "strict stays the default" note on DEC-000-26).
- **Reason:** The user: "I want tolerant behaviour".
- **Rejected:** Strict default (keeps existing callers unchanged; given up by the user). Tolerant only, with strict removed (the user kept the opt-out).
- **Lands in:** the `chat_with_tools` signature and tool loop in `llm/openai.py`, `llm/llama_swap.py`, `llm/openai_chat_completions.py`, `llm/llama_swap_chat_completions.py`
- **Cites:** docs/architecture/interfaces.md § Exposed: client method contract, § Tool failures, § Tool event observer, § Exposed: warnings, § Exposed: error contract; DEC-arch-14; DEC-arch-23; DEC-arch-24
