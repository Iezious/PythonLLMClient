# LLMClient - Agent Guidelines

## Project Overview

Async Python client library providing a unified interface to multiple LLM backends (Ollama, OpenAI, llama-swap). Built on `aiohttp` with typed options and no heavy dependencies (pydantic is imported at package load, see DEC-000-43).

## Repository Structure

```text
llm/                  # Main package (all source lives here)
  __init__.py         # Public exports
  exceptions.py       # LLMError
  message.py          # LLMMessage TypedDict
  options.py          # Typed option dicts (BaseLLMOptions, OllamaOptions, OpenAIOptions)
  llm_client.py       # Base class with shared HTTP/SSE logic
  ollama.py           # OllamaAPIClient
  openai.py           # OpenAIAPIClient
  llama_swap.py       # LlamaSwapAPIClient
  tools.py            # pydantic_to_openai_tool helper
  trace.py            # LLMTracer — env-driven trace logging for LLM interactions
  llm.py              # Factory function + placeholder utilities + retired CLI smoke test
  py.typed            # PEP 561 typing marker
tests/                # Test directory (pytest + pytest-asyncio)
  integrations/       # Live tests against real servers (in the default run)
pyproject.toml        # Build config, dependencies, tool settings
```

## Key Patterns

- **Base class**: `LLMClient` in `llm_client.py` holds shared HTTP plumbing (`_request`, `_iter_sse_data`, `_stream_openai_chat_response`, `_merge_options_base`). Subclasses override `_merge_options`, `chat`, `generate`, `chat_json`, `chat_with_tools`, `embed_batch`, `list_models`, `get_models` (the base raises `NotImplementedError`; there is no `abc`).
- **Async context manager**: All clients support `async with client:` for session lifecycle.
- **Options merging**: Three-layer merge (defaults -> constructor defaults -> call-time) via `_merge_options_base` with alias support.
- **No models dependency**: This package is self-contained. It does NOT depend on any external `models` package (pydantic is imported at package load but declared only in the `dev` extra, DEC-000-43).

## Conventions

- Python 3.11+
- Type hints everywhere; use `TypedDict` for options, not dataclasses/Pydantic
- `aiohttp` for all HTTP; no `requests` or `httpx`
- Imports use `from llm.xxx import Yyy` (absolute; only `llm/__init__.py` uses relative imports)
- Ruff for linting (line-length 120)
- The app provides the OpenAI credential; the library must never read it from the environment (today's env read is a known defect, DEC-000-1/2)
- Tests with pytest-asyncio, fake the server by monkeypatching client seams (`_request`, `_tools_stream_request`, `_iter_sse_data`); aioresponses is unused

## Adding a New Backend

1. Create `llm/new_backend.py` with a class inheriting `LLMClient`
2. Define `_<BACKEND>_SUPPORTED_PARAMS` and `_<BACKEND>_ALIAS_MAP` dicts (backend-prefixed names)
3. Override every `NotImplementedError` method the backend supports (including `get_models`)
4. Register in `get_llm_client()` factory in `llm/llm.py`
5. Export from `llm/__init__.py`
6. Add tests in `tests/`

## Common Commands

```bash
uv venv                        # Create virtual environment (first time)
uv pip install -e ".[dev]"     # Install with dev dependencies
uv run pytest                  # Run unit AND live tests (live tests need servers)
uv run pytest tests/integrations/  # Run integration tests (needs live servers)
uv run ruff check llm/        # Lint
uv run ruff format llm/       # Format
```

<!-- legacy-docs-bootstrapper:start -->
## Documentation layout

This repository is a single service: the repository root is both the system doc root and the service doc root. One set of layers lives under `docs/`.

- `docs/product/` — spec: vision, actors, features, use cases, stories, glossary
- `docs/architecture/` — how the library is shaped: components, contracts, boundaries
- `docs/decisions/` — technical decisions below architecture (`index.md` is the registry)
- `docs/plans/` — created and owned by `/roadmap` / `/planner` on first use
- `docs/.cache/` — orchestrator working artifacts (generated, gitignored)
- `docs/.helpers/` — code_index data (generated; index files gitignored). This repo uses `code_index` (https://github.com/Iezious/code_index); run `code_index usage` for the manual.

Layer rules: each layer folder's `AGENTS.md`.
<!-- legacy-docs-bootstrapper:end -->

<!-- agent-ready:start -->
## Agent-driven development

This repository is one self-sufficient doc root; the root `AGENTS.md` is the service entry point for a planner or coder opened in any IDE. Stack: Python 3.11+, `aiohttp`, `uv`, pytest + pytest-asyncio, ruff. Build and test commands are in "Common Commands" above.

## Documentation layers

Each layer's `AGENTS.md` holds its rules.
- `docs/product/` — vision, actors, features, use cases, stories, glossary.
- `docs/architecture/` — `overview.md`, `subsystems.md`, `interfaces.md`, `data-model.md`, `quick-reference.md`.
- `docs/decisions/` — `index.md` (registry) and `000.legacy.md` (`DEC-000-<n>`).
- `docs/plans/` — created by `/roadmap` / `/planner` on first use.

## Code index

This repo uses `code_index` (config `docs/.helpers/config.toml`); query it via `code_index search` / `symbols` / `graph`, run `code_index usage` for syntax. Do not initialise another index.

## Agent guidance

- `docs/architecture/interfaces.md` is the public contract surface (method contract, capability matrix, factory, error contract); change it carefully.
- Known defects are recorded as DEC entries and listed in `docs/architecture/quick-reference.md` ("Open"); do not fix them in passing outside a plan.
- The Ollama client is deprecated and kept for observation only; do not extend it.
- Version is declared in `pyproject.toml` and `llm/__init__.py`; bump both together.

## Folder guides

- `llm/AGENTS.md` — the package.
- `tests/AGENTS.md` — unit tests.
- `tests/integrations/AGENTS.md` — live tests.

See also: `docs/architecture/overview.md`; `DEC-000-43` (pydantic imported at package load vs runtime deps), `DEC-000-53` (version declared in two places) in `docs/decisions/000.legacy.md`.
<!-- agent-ready:end -->
