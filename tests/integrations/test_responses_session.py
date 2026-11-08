"""Live session, embedding and model-record tests on the Responses classes (feature 005, step 003, DoD-5..9).

Runs against real OpenAI and llama-swap servers: four concurrent calls on one client inside one usage
scope and reuse of that client after the scope closes (DEC-005-10, DEC-000-24); ``embed``, ``embed_batch``
(order checked by cosine similarity, DEC-005-7) and ``get_models`` on clients built through the factory
(US-057.AC-2). One test function per backend (DEC-005-5); assertions are structural only (DEC-005-6).
OpenAI tests skip without a key; llama-swap tests never skip.
"""

import asyncio
import math
from collections.abc import Mapping, Sequence

import pytest

from llm import LLMClient
from llm.message import LLMMessage
from tests.integrations.live_support import (
    PieceRecorder,
    assert_well_formed_reasoning,
    build_client,
    llama_swap_embed_model,
    openai_embed_model,
)

# ---------------------------------------------------------------------------
# Module-local prompts and helpers
# ---------------------------------------------------------------------------

# Two different prompts for the two streamed calls, so pieces mixed between them could not
# coincidentally still join to each call's own return (DEC-005-10).
_STREAM_PROMPT_A = "Name three fruits in one short sentence."
_STREAM_PROMPT_B = "Name three colours of the rainbow in one short sentence."
_CHAT_PROMPT = "Name one planet of the solar system. Answer in one word."
_GENERATE_PROMPT = "Write one short sentence about mountains."
_REUSE_PROMPT = "Name one ocean. Answer in one word."

# Three unrelated short texts (DEC-005-7): distinct topics so each single vector is clearly closest to
# its own batch vector.
_UNRELATED_TEXTS = [
    "The cat sleeps on the warm windowsill.",
    "Quarterly tax filings are due at the end of March.",
    "The volcano erupted and lava flowed into the sea.",
]


def _user(content: str) -> list[LLMMessage]:
    """A one-turn conversation holding ``content`` as the user turn."""
    return [{"role": "user", "content": content}]


def _embed_model(backend: str) -> str:
    """The embedding model of ``backend`` (DEC-005-3)."""
    return openai_embed_model() if backend == "openai" else llama_swap_embed_model()


def _is_number(value: object) -> bool:
    """True for an int or float that is not a bool (a bool is an int in Python but not a vector component)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors (DEC-005-7); a zero vector fails the test."""
    assert len(a) == len(b), f"vectors differ in length: {len(a)} vs {len(b)}"
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    assert norm_a > 0 and norm_b > 0, "a zero vector has no direction to compare"
    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Shared bodies
# ---------------------------------------------------------------------------


async def _concurrent_in_one_scope(client: LLMClient) -> None:
    """DoD-5 body: inside one ``async with``, four calls at once on ``client`` (DEC-005-10).

    Two streamed chats (each with its own recorder), one non-streamed ``chat`` and one non-streamed
    ``generate``. All four return non-empty text; each streamed return equals its own recorder's join.
    No ignored setting is sent, so the only shared per-client state is the session.
    """
    recorder_a = PieceRecorder()
    recorder_b = PieceRecorder()
    async with client:
        stream_a, stream_b, chat_reply, generate_reply = await asyncio.gather(
            client.chat(_user(_STREAM_PROMPT_A), stream=True, on_delta=recorder_a),
            client.chat(_user(_STREAM_PROMPT_B), stream=True, on_delta=recorder_b),
            client.chat(_user(_CHAT_PROMPT)),
            client.generate(_GENERATE_PROMPT),
        )

    for name, reply in (
        ("streamed chat A", stream_a),
        ("streamed chat B", stream_b),
        ("chat", chat_reply),
        ("generate", generate_reply),
    ):
        assert isinstance(reply, str), f"{name} returned {type(reply).__name__}, expected str"
        assert reply.strip(), f"{name} returned empty text"

    assert stream_a == recorder_a.text, "streamed chat A's return differs from its own recorder's join"
    assert stream_b == recorder_b.text, "streamed chat B's return differs from its own recorder's join"
    # A reasoning block that does appear must not break the structural checks (DEC-005-4).
    assert_well_formed_reasoning(recorder_a.text)
    assert_well_formed_reasoning(recorder_b.text)


async def _concurrent(backend: str) -> None:
    """DoD-5: one client, one scope, four concurrent calls."""
    await _concurrent_in_one_scope(build_client(backend))


async def _reuse_after_scope(backend: str) -> None:
    """DoD-6: after DoD-5's scope closes, the same client object in a new scope still answers.

    The DoD-5 scope runs first on the same object, so the second ``async with`` truly follows a closed
    session (DEC-000-24: the session is recreated by a later call).
    """
    client = build_client(backend)
    await _concurrent_in_one_scope(client)

    async with client:
        reply = await client.chat(_user(_REUSE_PROMPT))

    assert isinstance(reply, str)
    assert reply.strip(), "the reused client returned empty text"


async def _embed_single(backend: str) -> None:
    """DoD-7: ``embed`` returns one non-empty list whose items are all numbers."""
    client = build_client(backend, _embed_model(backend))
    async with client:
        vector = await client.embed("The cat sleeps on the warm windowsill.")

    assert isinstance(vector, list), f"expected a list, got {type(vector).__name__}"
    assert vector, "expected a non-empty vector"
    bad = [i for i, x in enumerate(vector) if not _is_number(x)]
    assert not bad, f"vector items at positions {bad[:5]} are not numbers"


async def _embed_batch_order(backend: str) -> None:
    """DoD-8: three vectors of equal non-zero length, each closest to the single vector of its own text."""
    client = build_client(backend, _embed_model(backend))
    async with client:
        batch = await client.embed_batch(list(_UNRELATED_TEXTS))
        singles = [await client.embed(text) for text in _UNRELATED_TEXTS]

    assert isinstance(batch, list)
    assert len(batch) == len(_UNRELATED_TEXTS), f"expected {len(_UNRELATED_TEXTS)} vectors, got {len(batch)}"
    lengths = {len(vector) for vector in batch}
    assert len(lengths) == 1, f"batch vectors differ in length: {sorted(lengths)}"
    assert lengths.pop() > 0, "batch vectors are empty"

    for i, batch_vector in enumerate(batch):
        similarities = [_cosine(batch_vector, single) for single in singles]
        best = max(range(len(similarities)), key=similarities.__getitem__)
        assert best == i, (
            f"batch vector {i} is most similar to the single vector of text {best}, not its own "
            f"(similarities {similarities})"
        )


async def _models(backend: str) -> None:
    """DoD-9: ``get_models`` returns a non-empty list whose every entry is a mapping."""
    client = build_client(backend)
    async with client:
        models = await client.get_models()

    assert isinstance(models, list), f"expected a list, got {type(models).__name__}"
    assert models, "expected at least one model record"
    bad = [i for i, entry in enumerate(models) if not isinstance(entry, Mapping)]
    assert not bad, f"model records at positions {bad[:5]} are not mappings"


# ---------------------------------------------------------------------------
# DoD-5 — concurrent calls in one scope
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_concurrent_calls_one_scope__DoD5():
    """DoD-5: OpenAI four concurrent calls return non-empty text; streamed returns equal their own joins."""
    await _concurrent("openai")


@pytest.mark.asyncio
async def test_llamaswap_concurrent_calls_one_scope__DoD5():
    """DoD-5: llama-swap four concurrent calls return non-empty text; streamed returns equal their own joins."""
    await _concurrent("llamaswap")


# ---------------------------------------------------------------------------
# DoD-6 — reuse after the scope
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_reuse_after_scope__DoD6():
    """DoD-6: the OpenAI client used again in a new scope after DoD-5's scope returns non-empty text."""
    await _reuse_after_scope("openai")


@pytest.mark.asyncio
async def test_llamaswap_reuse_after_scope__DoD6():
    """DoD-6: the llama-swap client used again in a new scope after DoD-5's scope returns non-empty text."""
    await _reuse_after_scope("llamaswap")


# ---------------------------------------------------------------------------
# DoD-7 — embed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_embed__DoD7():
    """DoD-7: OpenAI ``embed`` on the embedding-model client returns one non-empty list of numbers."""
    await _embed_single("openai")


@pytest.mark.asyncio
async def test_llamaswap_embed__DoD7():
    """DoD-7: llama-swap ``embed`` on the embedding-model client returns one non-empty list of numbers."""
    await _embed_single("llamaswap")


# ---------------------------------------------------------------------------
# DoD-8 — embed_batch order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_embed_batch_order__DoD8():
    """DoD-8: OpenAI ``embed_batch`` gives three equal-length vectors, each closest to its own text's vector."""
    await _embed_batch_order("openai")


@pytest.mark.asyncio
async def test_llamaswap_embed_batch_order__DoD8():
    """DoD-8: llama-swap ``embed_batch`` gives three equal-length vectors, each closest to its own text's vector."""
    await _embed_batch_order("llamaswap")


# ---------------------------------------------------------------------------
# DoD-9 — get_models
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_get_models__DoD9():
    """DoD-9: OpenAI ``get_models`` returns a non-empty list of mappings."""
    await _models("openai")


@pytest.mark.asyncio
async def test_llamaswap_get_models__DoD9():
    """DoD-9: llama-swap ``get_models`` returns a non-empty list of mappings."""
    await _models("llamaswap")
