"""Live conversation tests on the Responses classes (DEC-005-7, DoD-1..7).

Runs against real OpenAI and llama-swap servers: the builder's class, multi-turn history (non-streamed
and streamed), the uppercase-only system instruction on ``chat`` and ``generate``, non-English streaming,
and streamed ``generate`` on llama-swap. One test function per backend (DEC-005-5); assertions are
structural plus the history-token check (DEC-005-6) and the script / case / piece-count checks
(DEC-005-7). OpenAI tests skip without a key; llama-swap tests never skip.
"""

import copy

import pytest

from llm import LlamaSwapAPIClient, OpenAIAPIClient
from llm.message import LLMMessage
from tests.integrations.live_support import (
    PieceRecorder,
    answer_text,
    assert_well_formed_reasoning,
    build_client,
)

# ---------------------------------------------------------------------------
# Module-local prompts and checks
# ---------------------------------------------------------------------------

# Neutral, unguessable code (DEC-005-6). It sits ONLY in the assistant turn, never in a user turn or the
# system instruction, so finding it in the reply proves the earlier assistant turn reached the model.
_HISTORY_TOKEN = "ZV9-3184"

_UPPERCASE_SYSTEM = "Answer only in uppercase letters. Never use lowercase letters."
# A short English answer is asked for (DEC-005-7), the plain prompt already used by the suite.
_PLAIN_PROMPT = "Name one primary colour. Answer in one word."

_MULTILINGUAL_PROMPT = (
    "Write two short sentences about the sea: the first one in Russian (Cyrillic letters), "
    "the second one in Japanese. Do not add any English."
)

_THREE_SENTENCE_PROMPT = "Write three short sentences about the sea."

_REPLACEMENT_CHAR = "�"


def _history() -> list[LLMMessage]:
    """User turn, assistant turn holding the history token, user turn asking to repeat it (DoD-2, DoD-3)."""
    return [
        {"role": "user", "content": "Please give me a reference code that I can ask you for later."},
        {"role": "assistant", "content": f"Here is your reference code: {_HISTORY_TOKEN}"},
        {"role": "user", "content": "What was the reference code you gave me? Repeat it exactly."},
    ]


def _assert_uppercase_answer(reply: str) -> None:
    """The answer text holds at least one letter and every cased letter in it is uppercase (DEC-005-7)."""
    text = answer_text(reply)
    assert any(ch.isalpha() for ch in text), f"expected at least one letter in the answer: {text!r}"
    lowercase = [ch for ch in text if ch.islower()]
    assert not lowercase, f"expected only uppercase letters, found lowercase {lowercase!r} in {text!r}"


def _has_cyrillic(text: str) -> bool:
    """True when ``text`` holds a Cyrillic letter (U+0400-U+04FF, DEC-005-7)."""
    return any("Ѐ" <= ch <= "ӿ" for ch in text)


def _has_japanese(text: str) -> bool:
    """True when ``text`` holds hiragana, katakana or a CJK ideograph (DEC-005-7 ranges)."""
    return any(
        "぀" <= ch <= "ゟ" or "゠" <= ch <= "ヿ" or "一" <= ch <= "鿿" for ch in text
    )


async def _history_not_streamed(backend: str) -> None:
    """DoD-2 body: the reply contains the token (ignoring case) and the caller's list is unchanged."""
    messages = _history()
    before = copy.deepcopy(messages)
    client = build_client(backend)
    async with client:
        reply = await client.chat(messages)

    assert isinstance(reply, str)
    assert _HISTORY_TOKEN.lower() in reply.lower(), f"expected the history token in the reply: {reply!r}"
    assert messages == before, "the caller's message list was changed by the call"


async def _history_streamed(backend: str) -> None:
    """DoD-3 body: the return contains the token, equals the recorder's join, and is well-formed."""
    recorder = PieceRecorder()
    client = build_client(backend)
    async with client:
        reply = await client.chat(_history(), stream=True, on_delta=recorder)

    assert isinstance(reply, str)
    assert _HISTORY_TOKEN.lower() in reply.lower(), f"expected the history token in the reply: {reply!r}"
    assert reply == recorder.text
    assert_well_formed_reasoning(recorder.text)


async def _chat_uppercase(backend: str) -> None:
    """DoD-4 body: non-streamed ``chat`` under the uppercase-only system instruction."""
    client = build_client(backend)
    async with client:
        reply = await client.chat([{"role": "user", "content": _PLAIN_PROMPT}], system=_UPPERCASE_SYSTEM)

    assert isinstance(reply, str)
    _assert_uppercase_answer(reply)


async def _generate_uppercase(backend: str) -> None:
    """DoD-5 body: non-streamed ``generate`` under the uppercase-only system instruction."""
    client = build_client(backend)
    async with client:
        reply = await client.generate(_PLAIN_PROMPT, system=_UPPERCASE_SYSTEM)

    assert isinstance(reply, str)
    _assert_uppercase_answer(reply)


async def _multilingual_streamed(backend: str) -> None:
    """DoD-6 body: streamed multilingual chat arrives complete, in both scripts, with no U+FFFD."""
    recorder = PieceRecorder()
    client = build_client(backend)
    async with client:
        reply = await client.chat([{"role": "user", "content": _MULTILINGUAL_PROMPT}], stream=True, on_delta=recorder)

    assert isinstance(reply, str)
    assert reply == recorder.text
    assert _has_cyrillic(reply), f"expected a Cyrillic letter in the reply: {reply!r}"
    assert _has_japanese(reply), f"expected a Japanese character in the reply: {reply!r}"
    assert _REPLACEMENT_CHAR not in reply, "the return holds U+FFFD"
    bad = [i for i, piece in enumerate(recorder.pieces) if _REPLACEMENT_CHAR in piece]
    assert not bad, f"pieces at positions {bad} hold U+FFFD"


# ---------------------------------------------------------------------------
# DoD-1 — the builder gives the Responses class
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_builder_gives_responses_class__DoD1():
    """DoD-1: with no protocol override the helper builds an ``OpenAIAPIClient`` for ``openai``."""
    client = build_client("openai")
    async with client:
        assert isinstance(client, OpenAIAPIClient)


@pytest.mark.asyncio
async def test_llamaswap_builder_gives_responses_class__DoD1():
    """DoD-1: with no protocol override the helper builds a ``LlamaSwapAPIClient`` for ``llamaswap``."""
    client = build_client("llamaswap")
    async with client:
        assert isinstance(client, LlamaSwapAPIClient)


# ---------------------------------------------------------------------------
# DoD-2 — multi-turn history, non-streamed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_multi_turn_history_not_streamed__DoD2():
    """DoD-2: OpenAI non-streamed reply repeats the assistant-turn token; the caller's list is unchanged."""
    await _history_not_streamed("openai")


@pytest.mark.asyncio
async def test_llamaswap_multi_turn_history_not_streamed__DoD2():
    """DoD-2: llama-swap non-streamed reply repeats the assistant-turn token; the caller's list is unchanged."""
    await _history_not_streamed("llamaswap")


# ---------------------------------------------------------------------------
# DoD-3 — multi-turn history, streamed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_multi_turn_history_streamed__DoD3():
    """DoD-3: OpenAI streamed return holds the token, equals the recorder's join, and is well-formed."""
    await _history_streamed("openai")


@pytest.mark.asyncio
async def test_llamaswap_multi_turn_history_streamed__DoD3():
    """DoD-3: llama-swap streamed return holds the token, equals the recorder's join, and is well-formed."""
    await _history_streamed("llamaswap")


# ---------------------------------------------------------------------------
# DoD-4 — chat follows the system instruction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_chat_follows_uppercase_system__DoD4():
    """DoD-4: OpenAI ``chat`` answer text has a letter and no lowercase letter under the instruction."""
    await _chat_uppercase("openai")


@pytest.mark.asyncio
async def test_llamaswap_chat_follows_uppercase_system__DoD4():
    """DoD-4: llama-swap ``chat`` answer text has a letter and no lowercase letter under the instruction."""
    await _chat_uppercase("llamaswap")


# ---------------------------------------------------------------------------
# DoD-5 — generate follows the system instruction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_generate_follows_uppercase_system__DoD5():
    """DoD-5: OpenAI ``generate`` answer text has a letter and no lowercase letter under the instruction."""
    await _generate_uppercase("openai")


@pytest.mark.asyncio
async def test_llamaswap_generate_follows_uppercase_system__DoD5():
    """DoD-5: llama-swap ``generate`` answer text has a letter and no lowercase letter under the instruction."""
    await _generate_uppercase("llamaswap")


# ---------------------------------------------------------------------------
# DoD-6 — non-English streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_multilingual_chat_streamed__DoD6():
    """DoD-6: OpenAI streamed multilingual chat equals the join, has Cyrillic and Japanese, and no U+FFFD."""
    await _multilingual_streamed("openai")


@pytest.mark.asyncio
async def test_llamaswap_multilingual_chat_streamed__DoD6():
    """DoD-6: llama-swap streamed multilingual chat equals the join, has Cyrillic and Japanese, and no U+FFFD."""
    await _multilingual_streamed("llamaswap")


# ---------------------------------------------------------------------------
# DoD-7 — streamed generate on llama-swap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_generate_streamed_pieces__DoD7():
    """DoD-7: llama-swap streamed ``generate`` gives >= 2 pieces, a non-empty return equal to their join,
    and well-formed recorder output."""
    recorder = PieceRecorder()
    client = build_client("llamaswap")
    async with client:
        reply = await client.generate(_THREE_SENTENCE_PROMPT, stream=True, on_delta=recorder)

    assert len(recorder.pieces) >= 2, f"expected at least two pieces, got {len(recorder.pieces)}"
    assert isinstance(reply, str)
    assert reply.strip()
    assert reply == recorder.text
    assert_well_formed_reasoning(recorder.text)
