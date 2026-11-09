# LLMClient - Agent Guidelines

## Project Overview

Async Python client library providing a unified interface to multiple LLM backends (Ollama, OpenAI, llama-swap). Built on `aiohttp` with typed options and no heavy dependencies (pydantic is a declared runtime dependency, see DEC-001-3).

## Repository Structure

```text
llm/                  # Main package (all source lives here)
  __init__.py         # Public exports
  exceptions.py       # LLMError, LLMSettingIgnoredWarning
  message.py          # LLMMessage TypedDict
  options.py          # Typed option dicts (BaseLLMOptions, OllamaOptions, OpenAIOptions)
  llm_client.py       # Base class with shared HTTP/SSE logic
  ollama.py           # OllamaAPIClient
  openai.py           # OpenAIAPIClient (Responses protocol)
  openai_chat_completions.py      # OpenAIChatCompletionsClient (chat completions)
  llama_swap.py       # LlamaSwapAPIClient (Responses protocol)
  llama_swap_chat_completions.py # LlamaSwapChatCompletionsClient (chat completions)
  tools.py            # pydantic_to_openai_tool helper
  trace.py            # LLMTracer — env-driven trace logging for LLM interactions
  llm.py              # Factory function + placeholder utilities
  py.typed            # PEP 561 typing marker
tests/                # Test directory (pytest + pytest-asyncio)
  integrations/       # Live tests against real servers (in the default run)
pyproject.toml        # Build config, dependencies, tool settings
```

## Key Patterns

- **Base class**: `LLMClient` in `llm_client.py` holds transport plumbing only (`_request`, `_stream_post`, `_iter_sse_data`: requests, streams, transport-error mapping, time limits, the tracer). No behaviour logic is shared between client classes: each class keeps its own copy of its behaviour, the tool loop included (`docs/architecture/subsystems.md` § Base client). The existing shared chat-completions stream parsers (`_stream_openai_chat_response`, `_stream_openai_tools_response`) and the option merge (`_merge_options_base`) stay in the base as they are and are not extended (DEC-arch-25). `chat_with_tools` takes `messages` as a read-only sequence of mappings (`Sequence[Mapping[str, Any]]`). Subclasses override `_merge_options`, `chat`, `generate`, `chat_json`, `chat_with_tools`, `embed_batch`, `list_models`, `get_models` (the base raises `NotImplementedError`; there is no `abc`).
- **Async context manager**: All clients support `async with client:` for session lifecycle.
- **Options merging**: Three-layer merge (defaults -> constructor defaults -> call-time) via `_merge_options_base` with alias support.
- **No models dependency**: This package is self-contained. It does NOT depend on any external `models` package (pydantic is a declared runtime dependency in `dependencies`; the `dev` extra holds only pytest, pytest-asyncio, aioresponses, ruff — DEC-001-3).

## Conventions

- Python 3.11+
- Type hints everywhere; use `TypedDict` for options, not dataclasses/Pydantic; the base is generic over its options shape (`LLMClient[OptionsT]`, each class binds its own), and `uv run mypy --strict llm` stays clean (deprecated `ollama.py` silenced per module, DEC-arch-22, DEC-006-1)
- `aiohttp` for all HTTP; no `requests` or `httpx`
- Imports use `from llm.xxx import Yyy` (absolute; only `llm/__init__.py` uses relative imports)
- Ruff for linting (line-length 120)
- The app provides the OpenAI credential; the library never reads it from the environment (the earlier env read is fixed, DEC-001-6)
- Tests with pytest-asyncio, fake the server by monkeypatching client seams (`_request`, `_stream_post`, `_iter_sse_data`; `_tools_stream_request` on the chat-completions classes); aioresponses is unused

## Adding a New Backend

1. Create `llm/new_backend.py` with a class inheriting `LLMClient`; choose the protocol — an OpenAI-family backend may need a Responses class and a chat-completions class (`llm/new_backend_chat_completions.py`), picked by the factory's `protocol` argument (`docs/architecture/interfaces.md` § Protocol axis)
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
uv run pytest tests/ --ignore=tests/integrations  # Unit tests only (no servers)
uv run pytest tests/integrations/  # Run integration tests (needs live servers)
uv run ruff check llm/        # Lint
uv run mypy --strict llm      # Type check (strict; ollama.py silenced per module)
uv run ruff format llm/       # Format
```

After changing dependencies in `pyproject.toml`, run `uv lock` to refresh `uv.lock`.

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

See also: `docs/architecture/overview.md`; `DEC-001-3` (pydantic is a runtime dependency; supersedes DEC-000-43), `DEC-arch-23` (version 0.2.1, declared in two places — `pyproject.toml` and `llm/__init__.py`; supersedes DEC-001-4, which superseded DEC-000-53).
<!-- agent-ready:end -->
