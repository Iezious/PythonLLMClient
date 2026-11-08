"""Live tool-conversation tests for OpenAI (DEC-004-13, DoD-14..16).

A two-round tool conversation on the Responses protocol with the OpenAI reasoning model, streamed and not; the
same tool conversation on both protocols through the factory; and a tool conversation with a JSON-object answer
format. Servers, keys, models and settings follow DEC-004-13 and DEC-003-15: the test reads the OpenAI key itself
and skips without it. One neutral tool, ``fetch_record``, with a single text argument; its result cannot be
guessed and it counts its invocations. Only what a live model reliably produces is asserted (DEC-004-13).
"""

import copy
import json
import os
import re
import secrets

import pytest

from llm.llm import get_llm_client
from llm.llm_client import LLMClient

OPENAI_URL = "https://api.openai.com/v1"

# Responses is the default protocol, so it is chosen by omitting ``protocol`` (DEC-003-8, DEC-004-13).
PROTOCOLS = [
    pytest.param(None, id="responses"),
    pytest.param("chat_completions", id="chat_completions"),
]

STREAM_MODES = [
    pytest.param(False, id="non-streamed"),
    pytest.param(True, id="streamed"),
]

# Two-round settings on the reasoning model: low effort and a moderate cap keep the run cheap without cutting
# the reply off (DEC-004-13).
REASONING_OPTIONS = {"enable_thinking": True, "reasoning_effort": "low", "max_tokens": 2048}

RECORD_KEY = "R-118"
TOOL_PROMPT = (
    f"Call the fetch_record tool exactly once with the key '{RECORD_KEY}'. "
    "Then answer in one short sentence using the record it returns."
)
JSON_TOOL_PROMPT = (
    f"Call the fetch_record tool exactly once with the key '{RECORD_KEY}'. "
    'Then reply in JSON only: an object with the keys "key" and "record" holding the key you used and the '
    "text the tool returned."
)

# The one neutral tool in the app's nested definition shape (DEC-004-13, DEC-004-5).
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_record",
            "description": "Fetch a stored record by its key and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string", "description": "The record key."}},
                "required": ["key"],
            },
        },
    }
]


class _RecordTool:
    """``fetch_record`` (DEC-004-13): a single text argument; the result holds a code drawn fresh for each test, so
    the model cannot guess it; ``calls`` counts the invocations."""

    def __init__(self) -> None:
        self.calls = 0
        self.code = secrets.token_hex(6)

    def fetch_record(self, key: str) -> str:
        self.calls += 1
        return f"Record {key}: verification code {self.code}."

    @property
    def tools(self) -> dict:
        return {"fetch_record": self.fetch_record}


def _openai_api_key() -> str:
    """The OpenAI key from the test's own environment (``OPENAI_API_KEY`` then ``OPEN_AI_KEY``); skips the test
    when both are unset. The library never reads it itself (DEC-003-15)."""
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_AI_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY / OPEN_AI_KEY not set")
    return key


def _model() -> str:
    """Model for the protocol-parity and JSON tests (DEC-004-13, DEC-003-15)."""
    return os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"


def _reasoning_model() -> str:
    """Model for the two-round reasoning test (DEC-004-2, DEC-003-15)."""
    return os.environ.get("OPENAI_REASONING_MODEL") or "gpt-6-luna"


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


_MARKER = re.compile(r"<think>|</think>")


def _assert_no_markers(text: str) -> None:
    """No reasoning marker anywhere in ``text`` (US-058.AC-5)."""
    assert "<think>" not in text
    assert "</think>" not in text


def _assert_reasoning_blocks_well_formed(received: str, answer: str) -> None:
    """The receiver's reasoning blocks are well-formed (DEC-005-6): opening and closing
    markers balanced and alternating, every block closed, and the first block before the answer text.

    No block at all is accepted: OpenAI's reasoning summary is not reliably produced at low effort, so its
    presence is not asserted (DEC-004-13). The answer position is checked only when the answer is found in the
    received text, which this test does not assert.
    """
    markers = _MARKER.findall(received)
    assert len(markers) % 2 == 0, f"unbalanced markers: {markers}"
    assert markers == ["<think>", "</think>"] * (len(markers) // 2), f"markers do not alternate: {markers}"
    if markers:
        answer_at = received.rfind(answer)
        if answer_at >= 0:
            assert received.find("<think>") < answer_at, "the first reasoning block must come before the answer"


# ---------------------------------------------------------------------------
# DoD-14 — two-round tool conversation on the reasoning model, streamed and not
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_responses_reasoning_model_two_round_tool_conversation__DoD14(stream):
    """DoD-14: on Responses with ``OPENAI_REASONING_MODEL`` a two-round tool conversation completes; the tool ran,
    the final answer is non-empty and holds no marker; streamed, the receiver's reasoning blocks are well-formed
    (US-058.AC-1, US-058.AC-2, US-058.AC-5, DEC-004-2, DEC-004-13)."""
    key = _openai_api_key()
    tool = _RecordTool()
    recorder = _Recorder() if stream else None
    client = _client(key, _reasoning_model())
    async with client:
        result = await client.chat_with_tools(
            [{"role": "user", "content": TOOL_PROMPT}],
            tools_definitions=copy.deepcopy(TOOL_DEFINITIONS),
            tools=tool.tools,
            options=dict(REASONING_OPTIONS),
            stream=stream,
            on_delta=recorder,
        )
    assert tool.calls >= 1
    assert isinstance(result, str)
    assert result.strip()
    _assert_no_markers(result)
    if stream:
        _assert_reasoning_blocks_well_formed(recorder.text, result)


# ---------------------------------------------------------------------------
# DoD-15 — same kind of result on both protocols through the factory, streamed and not
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_openai_protocol_tool_conversation_same_kind_of_result__DoD15(protocol, stream):
    """DoD-15: with ``OPENAI_MODEL`` built through the factory on either protocol, a tool conversation runs the
    tool and returns non-empty text, streamed and not; streamed, the returned answer is contained in what the
    receiver got (US-057.AC-1, US-023.AC-1, US-023.AC-2, DEC-004-13)."""
    key = _openai_api_key()
    tool = _RecordTool()
    recorder = _Recorder() if stream else None
    client = _client(key, _model(), protocol)
    async with client:
        result = await client.chat_with_tools(
            [{"role": "user", "content": TOOL_PROMPT}],
            tools_definitions=copy.deepcopy(TOOL_DEFINITIONS),
            tools=tool.tools,
            stream=stream,
            on_delta=recorder,
        )
    assert tool.calls >= 1
    assert isinstance(result, str)
    assert result.strip()
    if stream:
        assert result in recorder.text


# ---------------------------------------------------------------------------
# DoD-16 — tool conversation with a JSON-object answer format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", STREAM_MODES)
async def test_openai_responses_tool_conversation_with_json_format_returns_json__DoD16(stream):
    """DoD-16: on Responses a tool conversation with ``response_format`` ``{"type": "json_object"}`` returns a
    final answer that parses as JSON (US-037.AC-1, DEC-004-13)."""
    key = _openai_api_key()
    tool = _RecordTool()
    recorder = _Recorder() if stream else None
    client = _client(key, _model())
    async with client:
        result = await client.chat_with_tools(
            [{"role": "user", "content": JSON_TOOL_PROMPT}],
            tools_definitions=copy.deepcopy(TOOL_DEFINITIONS),
            tools=tool.tools,
            stream=stream,
            on_delta=recorder,
            response_format={"type": "json_object"},
        )
    assert tool.calls >= 1
    assert isinstance(result, str)
    assert result.strip()
    json.loads(result)
