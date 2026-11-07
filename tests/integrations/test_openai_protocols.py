"""Live tests for OpenAI on both chat protocols (plan 003.responses-chat, step 003, DoD-12..15).

Every chat operation except tool chat is checked on Responses and on chat completions through the factory
(``protocol`` omitted for Responses, ``"chat_completions"`` for the other), plus thinking on / off and server
refusals on Responses. Models and environment variables follow DEC-003-15; the test reads the OpenAI key itself
and skips without it. Only what a live model reliably produces is asserted: type, non-empty text, JSON validity,
well-formed reasoning blocks, streamed text equal to returned text.
"""

import json
import os
import re

import pytest

from llm.exceptions import LLMError
from llm.llm import get_llm_client
from llm.llm_client import LLMClient

OPENAI_URL = "https://api.openai.com/v1"

# Responses is the default protocol, so it is chosen by omitting ``protocol`` (DEC-003-8).
PROTOCOLS = [
    pytest.param(None, id="responses"),
    pytest.param("chat_completions", id="chat_completions"),
]

# A model id that cannot exist on the server (DEC-003-15).
UNKNOWN_MODEL = "no-such-model-zz-9f3c2a"

# Thinking-on prompts ask for a multi-step answer so reasoning is produced (DEC-003-15).
MULTI_STEP_PROMPT = (
    "A train leaves at 09:40 and travels 168 km at 72 km/h, then waits 25 minutes, then travels 90 km at "
    "60 km/h. At what time does it arrive? Work it out step by step, then give the arrival time."
)
SHORT_PROMPT = "Name one primary colour. Answer in one word."
JSON_PROMPT = 'Reply in JSON only: an object with the keys "city" and "country" for the capital of France.'


def _openai_api_key() -> str:
    """The OpenAI key from the test's own environment (``OPENAI_API_KEY`` then ``OPEN_AI_KEY``); skips the test
    when both are unset. The library never reads it itself (DEC-001-6, DEC-003-15)."""
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_AI_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY / OPEN_AI_KEY not set")
    return key


def _model() -> str:
    """Model for the plain operations (DEC-003-15)."""
    return os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"


def _reasoning_model() -> str:
    """Model for the thinking tests: serves Responses with effort none / low and a summary (DEC-003-15)."""
    return os.environ.get("OPENAI_REASONING_MODEL") or "gpt-6-luna"


def _non_reasoning_model() -> str:
    """Model that refuses any reasoning effort (DEC-003-15)."""
    return os.environ.get("OPENAI_NON_REASONING_MODEL") or "gpt-4o-mini"


def _client(key: str, model: str, protocol: str | None = None) -> LLMClient:
    """Build an OpenAI client through the factory; ``protocol`` is passed only when one is named."""
    if protocol is None:
        return get_llm_client("openai", OPENAI_URL, model, bearer_token=key)
    return get_llm_client("openai", OPENAI_URL, model, bearer_token=key, protocol=protocol)


class _Recorder:
    """Async receiver collecting the streamed pieces in order."""

    def __init__(self) -> None:
        self.pieces: list[str] = []

    async def __call__(self, piece: str) -> None:
        self.pieces.append(piece)

    @property
    def text(self) -> str:
        return "".join(self.pieces)


_MARKER = re.compile(r"(<think>|</think>)")


def _assert_reasoning_well_formed(text: str) -> None:
    """A well-formed reasoning block (context.md § Test conventions).

    Opening and closing markers are balanced and alternate, every block is closed, the first block comes before
    the answer text, and some answer text lies outside the blocks.
    """
    parts = _MARKER.split(text)
    markers = [p for p in parts if p in ("<think>", "</think>")]
    assert markers, "expected at least one reasoning block"
    assert len(markers) % 2 == 0, f"unbalanced markers: {markers}"
    assert markers == ["<think>", "</think>"] * (len(markers) // 2), f"markers do not alternate: {markers}"
    assert parts[0].strip() == "", "the first reasoning block must come before the answer text"
    outside: list[str] = []
    inside = False
    for part in parts:
        if part == "<think>":
            inside = True
        elif part == "</think>":
            inside = False
        elif not inside:
            outside.append(part)
    assert "".join(outside).strip(), "expected answer text outside the reasoning blocks"


def _assert_no_markers(text: str) -> None:
    """No reasoning marker anywhere in ``text`` (US-059.AC-3, US-028.AC-1)."""
    assert "<think>" not in text
    assert "</think>" not in text


def _assert_reason_without_key(err: LLMError, key: str) -> None:
    """The error carries the server's reason (a non-empty description) and never the credential (US-060)."""
    assert err.description, "expected the server's reason in the error"
    assert key not in f"{err} {err.message} {err.description} {err.args!r}"


# ---------------------------------------------------------------------------
# DoD-12 — non-tool chat operations on both protocols
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_openai_protocol_chat_returns_text__DoD12(protocol):
    """Non-streamed ``chat`` returns non-empty text on both protocols (DoD-12, US-057.AC-1, US-056.AC-1/2)."""
    key = _openai_api_key()
    client = _client(key, _model(), protocol)
    async with client:
        result = await client.chat([{"role": "user", "content": SHORT_PROMPT}])
    assert isinstance(result, str)
    assert result.strip()


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_openai_protocol_generate_returns_text__DoD12(protocol):
    """Non-streamed ``generate`` returns non-empty text on both protocols (DoD-12, US-057.AC-1)."""
    key = _openai_api_key()
    client = _client(key, _model(), protocol)
    async with client:
        result = await client.generate(SHORT_PROMPT, system="You are terse.")
    assert isinstance(result, str)
    assert result.strip()


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_openai_protocol_chat_json_returns_json__DoD12(protocol):
    """``chat_json`` returns non-empty text that parses as JSON on both protocols (DoD-12, US-057.AC-1)."""
    key = _openai_api_key()
    client = _client(key, _model(), protocol)
    async with client:
        result = await client.chat_json([{"role": "user", "content": JSON_PROMPT}])
    assert isinstance(result, str)
    assert result.strip()
    json.loads(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_openai_protocol_streamed_chat_returns_what_receiver_got__DoD12(protocol):
    """A streamed ``chat`` returns exactly the text the receiver got, non-empty, on both protocols
    (DoD-12, US-057.AC-1)."""
    key = _openai_api_key()
    client = _client(key, _model(), protocol)
    recorder = _Recorder()
    async with client:
        result = await client.chat([{"role": "user", "content": SHORT_PROMPT}], stream=True, on_delta=recorder)
    assert isinstance(result, str)
    assert result.strip()
    assert result == recorder.text


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_openai_protocol_streamed_generate_returns_what_receiver_got__DoD12(protocol):
    """A streamed ``generate`` returns exactly the text the receiver got, non-empty, on both protocols
    (DoD-12, US-057.AC-1)."""
    key = _openai_api_key()
    client = _client(key, _model(), protocol)
    recorder = _Recorder()
    async with client:
        result = await client.generate(SHORT_PROMPT, stream=True, on_delta=recorder)
    assert isinstance(result, str)
    assert result.strip()
    assert result == recorder.text


# ---------------------------------------------------------------------------
# DoD-13 — thinking on / off on Responses with the reasoning model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_responses_thinking_on_streams_reasoning_block_before_answer__DoD13():
    """With thinking on, a streamed chat gives the receiver a well-formed reasoning block before the answer, and
    returns exactly what the receiver got (DoD-13, US-059.AC-1, US-059.AC-2)."""
    key = _openai_api_key()
    client = _client(key, _reasoning_model())
    recorder = _Recorder()
    async with client:
        result = await client.chat(
            [{"role": "user", "content": MULTI_STEP_PROMPT}],
            stream=True,
            on_delta=recorder,
            options={"enable_thinking": True},
        )
    assert result == recorder.text
    _assert_reasoning_well_formed(recorder.text)


@pytest.mark.asyncio
async def test_openai_responses_thinking_off_streams_no_reasoning_block__DoD13():
    """With thinking off, a streamed chat succeeds with no reasoning block (DoD-13, US-059.AC-3, US-025.AC-5)."""
    key = _openai_api_key()
    client = _client(key, _reasoning_model())
    recorder = _Recorder()
    async with client:
        result = await client.chat(
            [{"role": "user", "content": MULTI_STEP_PROMPT}],
            stream=True,
            on_delta=recorder,
            options={"enable_thinking": False},
        )
    assert isinstance(result, str)
    assert result.strip()
    assert result == recorder.text
    _assert_no_markers(recorder.text)


@pytest.mark.asyncio
async def test_openai_responses_thinking_off_with_explicit_low_effort_succeeds__DoD13():
    """Thinking off with an explicit effort ``low`` succeeds (DoD-13, US-025.AC-4, DEC-003-1)."""
    key = _openai_api_key()
    client = _client(key, _reasoning_model())
    async with client:
        result = await client.chat(
            [{"role": "user", "content": SHORT_PROMPT}],
            options={"enable_thinking": False, "reasoning_effort": "low"},
        )
    assert isinstance(result, str)
    assert result.strip()


@pytest.mark.asyncio
async def test_openai_responses_non_streamed_thinking_on_returns_no_marker__DoD13():
    """A non-streamed chat with thinking on returns the answer with no reasoning marker (DoD-13, US-028.AC-1)."""
    key = _openai_api_key()
    client = _client(key, _reasoning_model())
    async with client:
        result = await client.chat(
            [{"role": "user", "content": MULTI_STEP_PROMPT}],
            options={"enable_thinking": True},
        )
    assert isinstance(result, str)
    assert result.strip()
    _assert_no_markers(result)


# ---------------------------------------------------------------------------
# DoD-14 — thinking off refused by a non-reasoning model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_responses_non_reasoning_model_refuses_thinking_off__DoD14():
    """On a model that refuses any effort, thinking off ends with ``LLMError`` carrying the server's reason
    (DoD-14, DEC-003-1, US-060.AC-1)."""
    key = _openai_api_key()
    client = _client(key, _non_reasoning_model())
    async with client:
        with pytest.raises(LLMError) as excinfo:
            await client.chat([{"role": "user", "content": SHORT_PROMPT}], options={"enable_thinking": False})
    _assert_reason_without_key(excinfo.value, key)


# ---------------------------------------------------------------------------
# DoD-15 — unknown model refused; list_models on the Responses class
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_responses_unknown_model_raises_with_reason_not_key__DoD15():
    """A non-streamed chat on an unknown model ends with ``LLMError`` carrying the server's reason and not the key
    (DoD-15, US-060.AC-1, US-060.AC-4)."""
    key = _openai_api_key()
    client = _client(key, UNKNOWN_MODEL)
    async with client:
        with pytest.raises(LLMError) as excinfo:
            await client.chat([{"role": "user", "content": SHORT_PROMPT}])
    _assert_reason_without_key(excinfo.value, key)


@pytest.mark.asyncio
async def test_openai_responses_unknown_model_streamed_raises_with_reason_not_key__DoD15():
    """A streamed chat on an unknown model ends with ``LLMError`` carrying the server's reason and not the key,
    with nothing delivered to the receiver (DoD-15, US-060.AC-1, US-060.AC-4)."""
    key = _openai_api_key()
    client = _client(key, UNKNOWN_MODEL)
    recorder = _Recorder()
    async with client:
        with pytest.raises(LLMError) as excinfo:
            await client.chat([{"role": "user", "content": SHORT_PROMPT}], stream=True, on_delta=recorder)
    _assert_reason_without_key(excinfo.value, key)
    assert recorder.pieces == []


@pytest.mark.asyncio
async def test_openai_responses_list_models_returns_names__DoD15():
    """``list_models`` on the Responses class returns a non-empty list of strings (DoD-15, US-057.AC-2)."""
    key = _openai_api_key()
    client = _client(key, _model())
    async with client:
        models = await client.list_models()
    assert isinstance(models, list)
    assert models
    assert all(isinstance(name, str) and name for name in models)
