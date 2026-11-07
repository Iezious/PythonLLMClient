# Overview

## Purpose

What the `llm` library is, how it is composed, and what it deliberately leaves out. Contracts are in `interfaces.md`, components in `subsystems.md`, data shapes in `data-model.md`, indexes in `quick-reference.md`.

## What it is

An async Python library (package `llm`) that gives one client contract over three kinds of LLM backend. For the OpenAI family (OpenAI and llama-swap) it also covers two chat protocols.

**Target shape (planned, not yet built).** The OpenAI family speaks two chat protocols. **Responses** is the default. **Chat completions** is kept for compatibility with servers that do not offer Responses. The protocol is fixed by the client class:

| Backend | Protocol | Client | Status |
|---|---|---|---|
| llama-swap (llama.cpp orchestrator, OpenAI-compatible HTTP) | Responses | `LlamaSwapAPIClient` | primary backend (planned) |
| llama-swap | chat completions | `LlamaSwapChatCompletionsClient` | compatibility (planned rename of today's `LlamaSwapAPIClient`) |
| OpenAI or OpenAI-compatible | Responses | `OpenAIAPIClient` | supported (planned) |
| OpenAI or OpenAI-compatible | chat completions | `OpenAIChatCompletionsClient` | compatibility (planned rename of today's `OpenAIAPIClient`) |
| Ollama (native API) | native | `OllamaAPIClient` | **deprecated**, unchanged |

Today `OpenAIAPIClient` and `LlamaSwapAPIClient` speak chat completions, and the two compatibility class names do not exist yet. After the change, the existing public names mean Responses. An app that constructs those classes directly moves to Responses when it upgrades; this is intended. We chose a class per protocol because the user prefers separate classes and the existing names should follow the new default (DEC-arch-1). The two protocols carry the same app-facing contract, described in `interfaces.md`.

The contract covers text generation, chat, JSON-mode chat, tool-calling chat, streaming with a per-fragment callback, reasoning output, embeddings and model listing. It is built on `aiohttp`; there is no other runtime HTTP stack.

It is a standalone library. It has no in-tree callers besides tests, no database, queue or service of its own. Outward it depends only on the HTTP API of the configured backend and on the trace switch environment variables `LLM_TRACE` and `LLM_TRACE_FOLDER`. The library never reads a backend credential from the environment; the caller provides it.

## Component map

Target shape (planned):

```
public surface (package exports)
        |
   get_llm_client (factory; server_type + protocol select the class)
        |
 +-----------------+-------------------------+-----------------------------+
 | OllamaAPIClient | Responses:              | chat completions:           |
 | (deprecated)    |  OpenAIAPIClient        |  OpenAIChatCompletionsClient|
 |                 |  LlamaSwapAPIClient     |  LlamaSwapChatCompletionsClient
 +-----------------+-------------------------+-----------------------------+
        |  all extend directly (no protocol layer in between)
   LLMClient (abstract base)
        |
   shared: options merge, HTTP request contract, SSE parsing, tracer

 utilities: substitute_placeholders, pydantic_to_openai_tool
 signal:    ignored-setting warning category
```

Today there are three classes under the base: `OllamaAPIClient`, `OpenAIAPIClient` and `LlamaSwapAPIClient`, the last two on chat completions.

Responsibilities of each box are in `subsystems.md`.

## Session lifecycle

One HTTP session per client. It is opened lazily on first use or on entering `async with client:`, and closed on leaving the context. After closing, the client can be reused and opens a fresh session.

- Creating a client does no I/O. Address and model problems, and a credential the server refuses, surface on the first call; a missing OpenAI credential is refused at creation.
- Settings that come from the environment (the trace switch) are captured once, at creation; later environment changes do not affect an existing client.
- Concurrent calls on one client within one async run are supported.
- The session is bound to the async run in which it was opened.
- Usage scopes are not counted: any close releases the session for every caller, and calls in flight at that moment fail.
- Without a usage scope the connection is never released. Known gap, see `quick-reference.md`, Open.

## Stack choices

- HTTP: `aiohttp` for all backend traffic, because the library is async and needs an async HTTP client.
- Options are typed dictionaries rather than dataclasses or schema-validated models: plain dicts, no dependencies.
- One total time limit bounds each request, because local models can be slow.
- Streaming wire format: OpenAI-style server-sent events. On chat completions each event carries a choice delta; on Responses (planned) each event is typed. Reasoning reaches callers through the `<think>` tag convention on both protocols (see `data-model.md`).
- Chat protocol for the OpenAI family (planned): Responses by default, chat completions for compatibility. Responses is the current OpenAI API and lets reasoning models carry their reasoning across tool rounds. Chat completions stays because not every OpenAI-compatible server offers Responses.
- Runtime dependencies: `aiohttp` and `typing_extensions`. `pydantic` is imported when the package is imported but is not a declared runtime dependency (see `quick-reference.md`, Open).

## Out of scope / deliberately absent

- Retries, backoff and circuit breaking. The only fallback is the Ollama embeddings route fallback.
- Full normalisation of errors across streaming and non-streaming paths. Planned: a streamed non-2xx reply and a Responses in-stream failure become `LLMError`. Error pieces inside a chat-completions stream stay a known defect (see `interfaces.md`, error contract).
- Multimodal message content.
- Server-side conversation state on the Responses protocol. Every call sends the full conversation, and the library asks the provider not to store it.
- Configuration files; all settings are constructor arguments, apart from the trace switch environment variables. An application may read its own credential from its environment and pass it in.
- Logging or metrics. There are two observability signals: the opt-in trace files, and (planned) a library-specific warning raised when a setting is ignored because the Responses protocol has no counterpart for it (see `interfaces.md`, Exposed: warnings).

## Decision history

- 2026-11-07 — The OpenAI family gains a second chat protocol, Responses, as the default (planned). The protocol is chosen by client class, the existing class names mean Responses, and chat completions moves to new compatibility classes. See DEC-arch-1, DEC-arch-2.
