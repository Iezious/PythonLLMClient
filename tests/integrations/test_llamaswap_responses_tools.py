"""Live tool-conversation tests on llama-swap (feature 004.responses-tools, step 004, DoD-15..17).

A two-round tool conversation on the Responses class with the thinking model and thinking on, streamed
and not; protocol parity of a tool conversation on both protocols through the factory; and a JSON-object
answer after a tool round. Address, models, tool and settings follow DEC-004-13 / DEC-003-15; llama-swap
tests are never skipped. Only what a live model reliably produces is asserted: the tool ran, the final
answer is non-empty text, no reasoning marker in the return, well-formed reasoning blocks when streamed,
and JSON validity — never exact text.
"""

import json
import os
import re
import uuid
from typing import Any

import pytest

from llm.llm import get_llm_client

_OPEN = "<think>"
_CLOSE = "</think>"
_MARKER = re.compile(r"</?think>")

_DEFAULT_MODEL = "Qwen3.8-27B-Uncensored-FP8.i1-Q5_K_M"

# ``None`` = protocol argument omitted (Responses); the other value selects chat completions.
_PROTOCOLS = [pytest.param(None, id="responses"), pytest.param("chat_completions", id="chat_completions")]
_STREAM = [pytest.param(False, id="non_streamed"), pytest.param(True, id="streamed")]

_RECORD_KEY = "order-4471"

# The one neutral tool of DEC-004-13, in today's nested definition shape.
_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_record",
            "description": "Fetch the stored record for a key and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string", "description": "The record key."}},
                "required": ["key"],
            },
        },
    }
]

_PROMPT = (
    f"Call the fetch_record tool exactly once with the key '{_RECORD_KEY}'. "
    "After you get its result, answer in one short sentence that states the record's contents."
)
_JSON_PROMPT = (
    f"Call the fetch_record tool exactly once with the key '{_RECORD_KEY}'. "
    'After you get its result, reply with a JSON object with one key "record" whose value is the '
    "record's contents. Reply with the JSON object only."
)


def _base_url() -> str:
    """llama-swap address: ``LLAMA_SWITCH_URL`` first, then ``LLAMA_SWAP_BASE_URL``, then the default."""
    return os.environ.get("LLAMA_SWITCH_URL") or os.environ.get("LLAMA_SWAP_BASE_URL") or "http://india.loc:9292/v1"


def _model() -> str:
    """Parity / plain tool-conversation model (DEC-003-15, DEC-004-13)."""
    return os.environ.get("LLAMA_SWAP_MODEL") or _DEFAULT_MODEL


def _thinking_model() -> str:
    """Two-round thinking model (DEC-003-15, DEC-004-13)."""
    return os.environ.get("LLAMA_SWAP_THINKING_MODEL") or _DEFAULT_MODEL


def _client(protocol: str | None, model: str | None = None):
    """Build a llama-swap client through the factory; ``protocol`` is omitted entirely for Responses."""
    kwargs: dict[str, Any] = {} if protocol is None else {"protocol": protocol}
    return get_llm_client("llamaswap", _base_url(), model or _model(), **kwargs)


class _RecordTool:
    """The neutral ``fetch_record`` tool: returns a fresh random code the model cannot guess and counts its runs."""

    def __init__(self) -> None:
        self.invocations = 0
        self.code = f"rec-{uuid.uuid4().hex[:12]}"

    def fetch_record(self, key: str) -> str:
        self.invocations += 1
        return f"Record {key}: verification code {self.code}."


async def _tool_chat(
    client, tool: _RecordTool, *, stream: bool, prompt: str = _PROMPT, **kwargs: Any
) -> tuple[str, str]:
    """Run one tool conversation (``max_loops`` default); return (returned answer, everything the receiver got)."""
    pieces: list[str] = []

    async def on_delta(piece: str) -> None:
        pieces.append(piece)

    async with client:
        result = await client.chat_with_tools(
            [{"role": "user", "content": prompt}],
            tools_definitions=_TOOL_DEFS,
            tools={"fetch_record": tool.fetch_record},
            stream=stream,
            on_delta=on_delta if stream else None,
            **kwargs,
        )
    return result, "".join(pieces)


def _assert_reasoning_blocks_well_formed(text: str) -> None:
    """Well-formed reasoning blocks (003 context.md § Test conventions): opening and closing markers balanced
    and never nested, every block closed, and the first block before any answer text."""
    markers = _MARKER.findall(text)
    depth = 0
    for marker in markers:
        depth += 1 if marker == _OPEN else -1
        assert depth in (0, 1), f"unbalanced or nested reasoning markers: {markers}"
    assert depth == 0, "a reasoning block was left open"
    if markers:
        assert not text[: text.index(_OPEN)].strip(), "the first reasoning block must come before the answer text"


# ---------------------------------------------------------------------------
# DoD-15 — two-round tool conversation with the thinking model, thinking on
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", _STREAM)
async def test_llamaswap_responses_two_round_tool_conversation_with_thinking__DoD15(stream):
    """DoD-15: on Responses with the thinking model and thinking on, a two-round tool conversation completes:
    the tool ran, the answer is non-empty with no marker; streamed, at least one reasoning block arrived and
    all blocks are well-formed (US-058.AC-2, US-058.AC-5, US-027.AC-1, DEC-004-13)."""
    tool = _RecordTool()
    client = _client(None, _thinking_model())

    result, received = await _tool_chat(client, tool, stream=stream, options={"enable_thinking": True})

    assert tool.invocations >= 1
    assert isinstance(result, str)
    assert result.strip()
    assert _OPEN not in result
    assert _CLOSE not in result
    if stream:
        assert received.count(_OPEN) >= 1, "expected at least one reasoning block on the receiver"
        _assert_reasoning_blocks_well_formed(received)


# ---------------------------------------------------------------------------
# DoD-16 — protocol parity of a tool conversation through the factory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", _STREAM)
@pytest.mark.parametrize("protocol", _PROTOCOLS)
async def test_llamaswap_tool_conversation_same_kind_of_result_on_both_protocols__DoD16(protocol, stream):
    """DoD-16: with the plain model, built through the factory on both protocols, a tool conversation gives
    the same kind of result streamed and not: the tool ran and the answer is non-empty text; streamed, the
    returned answer is contained in what the receiver got (US-057.AC-1, US-023.AC-1, US-023.AC-2, DEC-004-13)."""
    tool = _RecordTool()

    result, received = await _tool_chat(_client(protocol), tool, stream=stream)

    assert tool.invocations >= 1
    assert isinstance(result, str)
    assert result.strip()
    if stream:
        assert result.strip() in received


# ---------------------------------------------------------------------------
# DoD-17 — JSON-object answer after a tool round
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llamaswap_responses_tool_conversation_json_object_answer_parses__DoD17():
    """DoD-17: on Responses, a tool conversation with ``response_format`` ``{"type": "json_object"}`` returns a
    final answer that parses as JSON (US-037.AC-1, DEC-004-13)."""
    tool = _RecordTool()

    result, _received = await _tool_chat(
        _client(None), tool, stream=False, prompt=_JSON_PROMPT, response_format={"type": "json_object"}
    )

    assert isinstance(result, str)
    json.loads(result)
