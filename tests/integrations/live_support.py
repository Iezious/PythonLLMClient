"""Shared helpers for the live Responses tests of feature 005.responses-live-tests (step 001, DEC-005-2).

Holds the server / model / key readers (DEC-005-3), the Responses client builder (DEC-005-4), the piece
recorder, the reasoning-marker checks (DEC-005-6) and the secret-free error check. Only the live files
added by feature 005 import it, as ``tests.integrations.live_support``; it imports nothing from other test
files. Steps 002-004 consume it read-only: a helper they find missing is written module-local there.
"""

import os
import re
from typing import Any

import pytest

from llm import LLMClient, LLMError, get_llm_client

__all__ = [
    "BACKENDS",
    "LLAMA_SWAP_THINKING_OFF",
    "OPENAI_URL",
    "THINK_CLOSE",
    "THINK_OPEN",
    "PieceRecorder",
    "answer_text",
    "assert_no_markers",
    "assert_secret_free_error",
    "assert_well_formed_reasoning",
    "build_client",
    "llama_swap_base_url",
    "llama_swap_embed_model",
    "llama_swap_model",
    "openai_api_key",
    "openai_base_url",
    "openai_embed_model",
    "openai_model",
    "openai_reasoning_model",
]

# ---------------------------------------------------------------------------
# Servers, models and keys (DEC-005-3, building on DEC-003-15)
# ---------------------------------------------------------------------------

#: Backend tokens accepted by :func:`build_client`; they are also the ``test_<backend>_`` name prefixes.
BACKENDS = ("openai", "llamaswap")

#: The fixed OpenAI address. DEC-005-3 rejects an address variable for OpenAI, so none is read.
OPENAI_URL = "https://api.openai.com/v1"

_DEFAULT_LLAMA_SWAP_URL = "http://india.loc:9292/v1"
_DEFAULT_LLAMA_SWAP_MODEL = "Qwen3.8-27B-Uncensored-FP8.i1-Q5_K_M"
_DEFAULT_LLAMA_SWAP_EMBED_MODEL = "bge-large-en-v1.5.i1-Q6_K"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_OPENAI_REASONING_MODEL = "gpt-6-luna"
_DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"

#: The llama-swap client default the builder applies (DEC-005-4): thinking off, so runs stay short and
#: replies answer-only. A fresh copy is merged into each client's defaults; never mutate this dict.
LLAMA_SWAP_THINKING_OFF: dict[str, Any] = {"enable_thinking": False}


def _env(*names: str) -> str | None:
    """Return the first non-empty value among the environment variables ``names``, or ``None``.

    An empty variable counts as unset, matching the ``or`` chains of the existing live files.
    """
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def llama_swap_base_url() -> str:
    """llama-swap address: ``LLAMA_SWITCH_URL``, then ``LLAMA_SWAP_BASE_URL``, then ``http://india.loc:9292/v1``.

    ``LLAMA_SWITCH_URL`` stays first (tests/integrations/AGENTS.md § Conventions).
    """
    return _env("LLAMA_SWITCH_URL", "LLAMA_SWAP_BASE_URL") or _DEFAULT_LLAMA_SWAP_URL


def llama_swap_model() -> str:
    """llama-swap chat model: ``LLAMA_SWAP_MODEL``, default ``Qwen3.8-27B-Uncensored-FP8.i1-Q5_K_M``."""
    return _env("LLAMA_SWAP_MODEL") or _DEFAULT_LLAMA_SWAP_MODEL


def llama_swap_embed_model() -> str:
    """llama-swap embedding model: ``LLAMA_SWAP_EMBED_MODEL``, default ``bge-large-en-v1.5.i1-Q6_K``."""
    return _env("LLAMA_SWAP_EMBED_MODEL") or _DEFAULT_LLAMA_SWAP_EMBED_MODEL


def openai_model() -> str:
    """OpenAI plain (non-reasoning) model: ``OPENAI_MODEL``, default ``gpt-4o-mini``."""
    return _env("OPENAI_MODEL") or _DEFAULT_OPENAI_MODEL


def openai_reasoning_model() -> str:
    """OpenAI reasoning model: ``OPENAI_REASONING_MODEL``, default ``gpt-6-luna``."""
    return _env("OPENAI_REASONING_MODEL") or _DEFAULT_OPENAI_REASONING_MODEL


def openai_embed_model() -> str:
    """OpenAI embedding model: ``OPENAI_EMBED_MODEL``, default ``text-embedding-3-small``."""
    return _env("OPENAI_EMBED_MODEL") or _DEFAULT_OPENAI_EMBED_MODEL


def openai_base_url() -> str:
    """The fixed OpenAI address, :data:`OPENAI_URL` (no variable, DEC-005-3)."""
    return OPENAI_URL


def openai_api_key() -> str:
    """Return the OpenAI key from ``OPENAI_API_KEY``, then ``OPEN_AI_KEY``.

    Skips the calling test (``pytest.skip``) when both are empty (DEC-005-5). The key is read here, in the
    test, and handed to the client as ``bearer_token``; the library never reads it (DEC-001-6).
    """
    key = _env("OPENAI_API_KEY", "OPEN_AI_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY / OPEN_AI_KEY not set")
    return key


def _default_model(backend: str) -> str:
    """The plain chat model of ``backend``."""
    return openai_model() if backend == "openai" else llama_swap_model()


# ---------------------------------------------------------------------------
# Responses client builder (DEC-005-3, DEC-005-4)
# ---------------------------------------------------------------------------


def build_client(
    backend: str,
    model: str | None = None,
    *,
    default_options: dict[str, Any] | None = None,
    timeout: float | None = None,
    base_url: str | None = None,
    bearer_token: str | None = None,
    thinking_off: bool = True,
) -> LLMClient:
    """Build a Responses client through ``get_llm_client`` with ``protocol`` omitted.

    Args:
        backend: ``"openai"`` or ``"llamaswap"`` (see :data:`BACKENDS`).
        model: the model; ``None`` means the backend's plain chat model (:func:`openai_model` /
            :func:`llama_swap_model`). Pass :func:`openai_reasoning_model`, :func:`openai_embed_model` or
            :func:`llama_swap_embed_model` explicitly when needed.
        default_options: client default options. On llama-swap they are merged over
            :data:`LLAMA_SWAP_THINKING_OFF`, so a caller key (e.g. ``enable_thinking``) wins. On OpenAI they
            are passed as given; OpenAI gets no thinking setting by default (DEC-005-4).
        timeout: client time limit in seconds; ``None`` keeps the factory default.
        base_url: address override; ``None`` means :func:`llama_swap_base_url` / :data:`OPENAI_URL`.
        bearer_token: credential override. On OpenAI it replaces :func:`openai_api_key` and therefore never
            triggers the missing-key skip (used by the bad-credential / unreachable tests, DEC-005-5).
            On llama-swap it is forwarded only when given (llama-swap needs no token).
        thinking_off: llama-swap only; ``False`` drops the thinking-off default entirely, leaving only
            ``default_options``. Ignored on OpenAI.

    Returns:
        The client (``OpenAIAPIClient`` / ``LlamaSwapAPIClient``), not yet entered; use ``async with``.

    Raises:
        ValueError: for an unknown backend token.
        pytest.skip.Exception: on OpenAI with no ``bearer_token`` override and no key in the environment.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend token {backend!r}; expected one of {BACKENDS}")

    kwargs: dict[str, Any] = {}
    if timeout is not None:
        kwargs["timeout"] = timeout

    if backend == "openai":
        # The override is checked first so a test that supplies its own credential never reads the env key.
        token = bearer_token if bearer_token is not None else openai_api_key()
        options = dict(default_options) if default_options is not None else None
        return get_llm_client(
            "openai",
            base_url or OPENAI_URL,
            model or _default_model(backend),
            bearer_token=token,
            default_options=options,
            **kwargs,
        )

    merged: dict[str, Any] = dict(LLAMA_SWAP_THINKING_OFF) if thinking_off else {}
    if default_options:
        merged.update(default_options)
    if bearer_token is not None:
        kwargs["bearer_token"] = bearer_token
    return get_llm_client(
        "llamaswap",
        base_url or llama_swap_base_url(),
        model or _default_model(backend),
        default_options=merged or None,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Piece recorder
# ---------------------------------------------------------------------------


class PieceRecorder:
    """An ``on_delta`` receiver that keeps every streamed piece in arrival order.

    Pass the instance itself as ``on_delta``. Use one recorder per call, so pieces of concurrent calls
    are never mixed (DEC-005-10).
    """

    def __init__(self) -> None:
        """Start with no pieces."""
        self.pieces: list[str] = []

    async def __call__(self, piece: str) -> None:
        """Record one piece (the receiver contract: an async callable taking one ``str``)."""
        self.pieces.append(piece)

    @property
    def text(self) -> str:
        """All pieces joined in order, with nothing between them."""
        return "".join(self.pieces)


# ---------------------------------------------------------------------------
# Reasoning-marker checks (DEC-005-6)
# ---------------------------------------------------------------------------

#: Reasoning block markers as delivered in text.
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
_MARKER = re.compile(r"</?think>")
# Non-greedy and DOTALL: each block is removed separately, and blocks span lines.
_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def assert_well_formed_reasoning(text: str) -> None:
    """Assert ``text`` holds only well-formed reasoning blocks, with the strict reading of DEC-005-6.

    Text with no markers passes. Otherwise: markers are balanced and never nested, every block is closed,
    nothing but whitespace stands before the first block or between blocks, and answer text follows the
    last block (the strict reading of 003's definition).
    """
    markers = _MARKER.findall(text)
    if not markers:
        return
    depth = 0
    for marker in markers:
        depth += 1 if marker == THINK_OPEN else -1
        assert depth in (0, 1), f"unbalanced or nested reasoning markers: {markers}"
    assert depth == 0, "a reasoning block was left open"
    last_close = text.rindex(THINK_CLOSE)
    # Everything up to the end of the last block must be blocks and whitespace: answer text only after it.
    assert not _BLOCK.sub("", text[: last_close + len(THINK_CLOSE)]).strip(), (
        "answer text appears before or between reasoning blocks"
    )
    assert text[last_close + len(THINK_CLOSE) :].strip(), "expected answer text after the last reasoning block"


def answer_text(text: str) -> str:
    """Return ``text`` with every reasoning block and every stray marker removed, stripped of edge whitespace.

    Used so content checks read only the answer even if a reasoning block appears (DEC-005-4).
    """
    return _MARKER.sub("", _BLOCK.sub("", text)).strip()


def assert_no_markers(text: str) -> None:
    """Assert neither ``<think>`` nor ``</think>`` appears in ``text``."""
    assert THINK_OPEN not in text, "unexpected <think> marker"
    assert THINK_CLOSE not in text, "unexpected </think> marker"


# ---------------------------------------------------------------------------
# Secret-free error check
# ---------------------------------------------------------------------------


def assert_secret_free_error(err: LLMError, secret: str) -> None:
    """Assert ``secret`` appears in none of the error's text, repr, message, description or args.

    The error contract never carries the credential (DEC-000-11, DEC-arch-5). ``secret`` must be non-empty,
    otherwise the check would be vacuous.
    """
    assert secret, "the secret to look for must be non-empty"
    surfaces = {
        "str": str(err),
        "repr": repr(err),
        "message": str(err.message),
        "description": "" if err.description is None else str(err.description),
        "args": repr(err.args),
    }
    leaked = [name for name, value in surfaces.items() if secret in value]
    # The secret itself is never printed in the failure message, so a failure cannot leak it either.
    assert not leaked, f"the secret appears in the error's {', '.join(leaked)}"
