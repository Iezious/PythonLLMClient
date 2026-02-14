# LLMClient - Agent Guidelines

## Project Overview

Async Python client library providing a unified interface to multiple LLM backends (Ollama, OpenAI, llama-swap). Built on `aiohttp` with typed options and no heavy dependencies.

## Repository Structure

```
llm/                  # Main package (all source lives here)
  __init__.py         # Public exports
  exceptions.py       # LLMError
  message.py          # LLMMessage TypedDict
  options.py          # Typed option dicts (BaseLLMOptions, OllamaOptions, OpenAIOptions)
  llm_client.py       # Abstract base class with shared HTTP/SSE logic
  ollama.py           # OllamaAPIClient
  openai.py           # OpenAIAPIClient
  llama_swap.py       # LlamaSwapAPIClient
  llm.py              # Factory function + placeholder utilities
tests/                # Test directory (pytest + pytest-asyncio)
pyproject.toml        # Build config, dependencies, tool settings
```

## Key Patterns

- **Abstract base**: `LLMClient` in `llm_client.py` holds shared HTTP plumbing (`_request`, `_iter_sse_data`, `_stream_openai_chat_response`, `_merge_options_base`). Subclasses implement `_merge_options`, `chat`, `generate`, `chat_json`, `chat_with_tools`, `embed_batch`, `list_models`.
- **Async context manager**: All clients support `async with client:` for session lifecycle.
- **Options merging**: Three-layer merge (defaults -> constructor defaults -> call-time) via `_merge_options_base` with alias support.
- **No models dependency**: This package is fully self-contained. It does NOT depend on any external `models` package.

## Conventions

- Python 3.11+
- Type hints everywhere; use `TypedDict` for options, not dataclasses/Pydantic
- `aiohttp` for all HTTP; no `requests` or `httpx`
- Imports use `from llm.xxx import Yyy` (package-relative)
- Ruff for linting (line-length 120)
- Tests with pytest-asyncio, mock HTTP with aioresponses

## Adding a New Backend

1. Create `llm/new_backend.py` with a class inheriting `LLMClient`
2. Define `_SUPPORTED_PARAMS` and `_ALIAS_MAP` dicts
3. Implement all abstract methods
4. Register in `get_llm_client()` factory in `llm/llm.py`
5. Export from `llm/__init__.py`
6. Add tests in `tests/`

## Common Commands

```bash
uv venv                        # Create virtual environment (first time)
uv pip install -e ".[dev]"     # Install with dev dependencies
uv run pytest                  # Run all unit tests
uv run pytest tests/integrations/  # Run integration tests (needs live servers)
uv run ruff check llm/        # Lint
uv run ruff format llm/       # Format
```
