# Overview

## Purpose

What the `llm` library is, how it is composed, and what it deliberately leaves out. Contracts are in `interfaces.md`, components in `subsystems.md`, data shapes in `data-model.md`, indexes in `quick-reference.md`.

## What it is

An async Python library (package `llm`) that gives one client contract over three kinds of LLM backend:

| Backend | Client | Status |
|---|---|---|
| llama-swap (llama.cpp orchestrator, OpenAI-compatible HTTP) | `LlamaSwapAPIClient` | primary backend |
| OpenAI or OpenAI-compatible (Chat Completions) | `OpenAIAPIClient` | supported |
| Ollama (native API) | `OllamaAPIClient` | **deprecated** |

The contract covers text generation, chat, JSON-mode chat, tool-calling chat, streaming with a per-fragment callback, reasoning output, embeddings and model listing. It is built on `aiohttp`; there is no other runtime HTTP stack.

It is a standalone library. It has no in-tree callers besides tests, no database, queue or service of its own. Outward it depends only on the HTTP API of the configured backend and on the environment variables `LLM_TRACE`, `LLM_TRACE_FOLDER`, `OPEN_AI_KEY` and `OPENAI_API_KEY`.

## Component map

```
public surface (package exports)
        |
   get_llm_client (factory)
        |
 +------+---------------+-----------------+
 | OllamaAPIClient      | OpenAIAPIClient | LlamaSwapAPIClient
 | (deprecated)         |                 |
 +------+---------------+-----------------+
        |  all extend
   LLMClient (abstract base)
        |
   shared: options merge, HTTP request contract, SSE parsing, tracer
        
 utilities: substitute_placeholders, pydantic_to_openai_tool
```

Responsibilities of each box are in `subsystems.md`.

## Session lifecycle

One HTTP session per client. It is opened lazily on first use or on entering `async with client:`, and closed on leaving the context. After closing, the client can be reused and opens a fresh session.

## Stack choices

- HTTP: `aiohttp` for all backend traffic. _TBD: reason for aiohttp over other HTTP libraries — deep-docs_
- Options are typed dictionaries rather than dataclasses or schema-validated models. _TBD: reason — deep-docs_
- Streaming wire format: OpenAI-style server-sent events. Reasoning is surfaced to callers with a `<think>` tag convention (see `data-model.md`).
- Runtime dependencies: `aiohttp` and `typing_extensions`. `pydantic` is imported when the package is imported but is not a declared runtime dependency (see `quick-reference.md`, Open).

## Out of scope / deliberately absent

- Retries, backoff and circuit breaking. The only fallback is the Ollama embeddings route fallback.
- Normalisation of errors across streaming and non-streaming paths.
- Multimodal message content.
- Configuration files; all settings are constructor arguments or the environment variables above.
- Logging or metrics. The only observability signal is the opt-in trace files.
