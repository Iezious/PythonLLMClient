"""Live tests of llama-swap on both chat protocols (feature 003.responses-chat, step 005, DoD-13..15).

Every non-tool chat operation runs on the Responses class and on the chat-completions class, both
built through the factory; thinking on / off / explicit effort and failure cases run on Responses.
Models and the address come from the environment per DEC-003-15; llama-swap tests are never
skipped. Only what a live model reliably produces is asserted: type, non-empty text, JSON validity,
well-formed reasoning blocks, streamed text equal to returned text.
"""

import json
import os
import re
from typing import Any

import pytest

from llm.exceptions import LLMError
from llm.llm import get_llm_client
from llm.message import LLMMessage

_OPEN = "<think>"
_CLOSE = "</think>"
_MARKER = re.compile(r"</?think>")

_DEFAULT_MODEL = "Qwen3.8-27B-Uncensored-FP8.i1-Q5_K_M"

# ``None`` = protocol argument omitted (Responses); the other value selects chat completions.
_PROTOCOLS = [pytest.param(None, id="responses"), pytest.param("chat_completions", id="chat_completions")]

_MULTI_STEP_PROMPT = (
    "A train leaves at 09:40 and the trip takes 2 hours 35 minutes, with a 20 minute stop on the way. "
    "Work out step by step when it arrives, then state the arrival time."
)


def _base_url() -> str:
    """llama-swap address: ``LLAMA_SWITCH_URL`` first, then ``LLAMA_SWAP_BASE_URL``, then the default."""
    return os.environ.get("LLAMA_SWITCH_URL") or os.environ.get("LLAMA_SWAP_BASE_URL") or "http://india.loc:9292/v1"


def _model() -> str:
    """Plain-operation model (DEC-003-15)."""
    return os.environ.get("LLAMA_SWAP_MODEL") or _DEFAULT_MODEL


def _thinking_model() -> str:
    """Thinking-test model (DEC-003-15)."""
    return os.environ.get("LLAMA_SWAP_THINKING_MODEL") or _DEFAULT_MODEL


def _client(protocol: str | None, model: str | None = None):
    """Build a llama-swap client through the factory; ``protocol`` is omitted entirely for Responses."""
    kwargs: dict[str, Any] = {} if protocol is None else {"protocol": protocol}
    return get_llm_client("llamaswap", _base_url(), model or _model(), **kwargs)


def _messages(text: str) -> list[LLMMessage]:
    """A one-message user conversation."""
    return [{"role": "user", "content": text}]


async def _streamed_chat(client, messages: list[LLMMessage], **kwargs: Any) -> tuple[str, str]:
    """Run a streamed chat; return (returned text, concatenation of everything the receiver got)."""
    pieces: list[str] = []

    async def on_delta(piece: str) -> None:
        pieces.append(piece)

    async with client:
        result = await client.chat(messages, stream=True, on_delta=on_delta, **kwargs)
    return result, "".join(pieces)


def _assert_well_formed_reasoning_before_answer(text: str) -> None:
    """A well-formed reasoning block (context.md § Test conventions): markers balanced, never nested,
    every block closed, the first block before the answer text, and answer text after the last block."""
    markers = _MARKER.findall(text)
    assert markers, "expected a reasoning block"
    depth = 0
    for marker in markers:
        depth += 1 if marker == _OPEN else -1
        assert depth in (0, 1), f"unbalanced or nested reasoning markers: {markers}"
    assert depth == 0, "a reasoning block was left open"
    assert text.lstrip().startswith(_OPEN), "the first reasoning block must come before the answer text"
    assert text[text.rindex(_CLOSE) + len(_CLOSE) :].strip(), "expected answer text after the reasoning"


# ---------------------------------------------------------------------------
# DoD-13 — every non-tool chat operation on both protocols
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", _PROTOCOLS)
async def test_llamaswap_chat_returns_text_on_both_protocols__DoD13(protocol):
    """DoD-13: a non-streamed ``chat`` returns non-empty text on Responses and on chat completions."""
    client = _client(protocol)
    async with client:
        result = await client.chat(_messages("Name one primary colour in one word."))

    assert isinstance(result, str)
    assert result.strip()


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", _PROTOCOLS)
async def test_llamaswap_generate_returns_text_on_both_protocols__DoD13(protocol):
    """DoD-13: a non-streamed ``generate`` returns non-empty text on both protocols."""
    client = _client(protocol)
    async with client:
        result = await client.generate("Write one short sentence about the sea.", system="Answer briefly.")

    assert isinstance(result, str)
    assert result.strip()


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", _PROTOCOLS)
async def test_llamaswap_chat_json_returns_parseable_json_on_both_protocols__DoD13(protocol):
    """DoD-13: ``chat_json`` returns non-empty text that parses as JSON on both protocols."""
    client = _client(protocol)
    prompt = 'Return a JSON object with a key "colour" whose value is the name of one primary colour.'
    async with client:
        result = await client.chat_json(_messages(prompt), system="Reply with JSON only.")

    assert isinstance(result, str)
    assert result.strip()
    json.loads(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", _PROTOCOLS)
async def test_llamaswap_streamed_chat_returns_exactly_what_receiver_got__DoD13(protocol):
    """DoD-13: a streamed ``chat`` returns exactly the text the receiver got, non-empty, on both protocols."""
    result, streamed = await _streamed_chat(_client(protocol), _messages("Count from one to five in words."))

    assert isinstance(result, str)
    assert result.strip()
    assert result == streamed


# ---------------------------------------------------------------------------
# DoD-14 — thinking on Responses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_responses_thinking_on_streams_reasoning_block_first__DoD14():
    """DoD-14: thinking on, streamed: the receiver gets a well-formed reasoning block before the answer."""
    client = _client(None, _thinking_model())
    result, streamed = await _streamed_chat(
        client, _messages(_MULTI_STEP_PROMPT), options={"enable_thinking": True}
    )

    assert result == streamed
    _assert_well_formed_reasoning_before_answer(streamed)


@pytest.mark.asyncio
async def test_llamaswap_responses_thinking_off_streams_no_reasoning_block__DoD14():
    """DoD-14: thinking off, streamed: the receiver gets no reasoning block, only answer text."""
    client = _client(None, _thinking_model())
    result, streamed = await _streamed_chat(
        client, _messages(_MULTI_STEP_PROMPT), options={"enable_thinking": False}
    )

    assert result == streamed
    assert streamed.strip()
    assert _OPEN not in streamed
    assert _CLOSE not in streamed


@pytest.mark.asyncio
async def test_llamaswap_responses_thinking_off_with_explicit_low_effort_succeeds__DoD14():
    """DoD-14: thinking off with an explicit effort ``low`` is accepted and the call returns text."""
    client = _client(None, _thinking_model())
    async with client:
        result = await client.chat(
            _messages("What is 12 + 30? Answer with the number."),
            options={"enable_thinking": False, "reasoning_effort": "low"},
        )

    assert isinstance(result, str)
    assert result.strip()


@pytest.mark.asyncio
async def test_llamaswap_responses_non_streamed_thinking_on_returns_no_marker__DoD14():
    """DoD-14: a non-streamed chat with thinking on returns text with no reasoning marker."""
    client = _client(None, _thinking_model())
    async with client:
        result = await client.chat(_messages(_MULTI_STEP_PROMPT), options={"enable_thinking": True})

    assert isinstance(result, str)
    assert result.strip()
    assert _OPEN not in result
    assert _CLOSE not in result


# ---------------------------------------------------------------------------
# DoD-15 — unknown model and model list on Responses
# ---------------------------------------------------------------------------

# A model id that cannot exist on any server (DEC-003-15).
_UNKNOWN_MODEL = "no-such-model-3f9c1e7a-does-not-exist"


async def _chat_unknown_model(stream: bool) -> None:
    """Send one chat to the unknown model on the Responses class, streamed or not."""
    client = _client(None, _UNKNOWN_MODEL)
    async with client:
        await client.chat(_messages("Hello"), stream=stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [pytest.param(False, id="non_streamed"), pytest.param(True, id="streamed")])
async def test_llamaswap_responses_unknown_model_raises_llmerror_with_reason__DoD15(stream):
    """DoD-15: an unknown model ends with LLMError carrying the server's reason, streamed and not."""
    with pytest.raises(LLMError) as excinfo:
        await _chat_unknown_model(stream)

    err = excinfo.value
    assert str(err).strip()
    assert (err.description or "").strip(), "the error should carry the server's reason"


@pytest.mark.asyncio
async def test_llamaswap_responses_list_models_returns_names__DoD15():
    """DoD-15: ``list_models`` on the Responses class returns a non-empty list of strings."""
    client = _client(None)
    async with client:
        models = await client.list_models()

    assert isinstance(models, list)
    assert models
    assert all(isinstance(m, str) for m in models)
