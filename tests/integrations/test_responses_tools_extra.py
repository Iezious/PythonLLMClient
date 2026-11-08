"""Extra live tool conversations on the Responses classes (feature 005.responses-live-tests, step 004, DoD-1..5).

Runs ``chat_with_tools`` against real OpenAI and llama-swap servers beyond feature 004's own live tests
(DEC-004-13): two tool calls non-streamed and streamed, a caller history that already holds tool turns,
a pydantic-built definition with a coroutine-function tool, and a streamed conversation with a
``json_schema`` final answer. One test function per backend (DEC-005-5); assertions are structural and
read the tool's recorded ids, never the answer's wording (DEC-005-6, DEC-005-11). The llama-swap
schema-answer test is a non-strict expected failure (DEC-005-9). OpenAI tests skip without a key;
llama-swap tests never skip.
"""

import copy
import json
from typing import Any

import pytest
from pydantic import BaseModel

from llm import pydantic_to_openai_tool
from tests.integrations.live_support import (
    PieceRecorder,
    answer_text,
    assert_no_markers,
    assert_well_formed_reasoning,
    build_client,
)

# ---------------------------------------------------------------------------
# Module-local tool, definitions, prompts, history and answer format (DEC-005-11)
# ---------------------------------------------------------------------------

_TOOL_NAME = "fetch_record"

# Each record id maps to a fixed value that cannot be guessed, so the model can only learn it from the tool
# (tests/integrations/AGENTS.md § Conventions). C is the record already fetched in the caller history (DoD-3).
_RECORD_VALUES = {
    "A": "amber-7310",
    "B": "cobalt-2958",
    "C": "violet-6143",
    "D": "saffron-4087",
}
# Returned for an id the table does not hold; the tool still answers so the conversation can finish.
_UNKNOWN_VALUE = "no-such-record"


class _RecordTool:
    """The neutral ``fetch_record`` tool in sync and async forms; records every id it was called with.

    Assertions use :attr:`calls` (DEC-005-11), never the model's wording. Ids are recorded normalised
    (stripped, upper-cased) because the model may echo the letter in either case; the id itself is what
    the check is about, not its spelling.
    """

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.calls: list[str] = []

    def _lookup(self, record_id: str) -> str:
        """Record ``record_id`` and return its fixed value."""
        key = str(record_id).strip().upper()
        self.calls.append(key)
        return _RECORD_VALUES.get(key, _UNKNOWN_VALUE)

    def fetch_record(self, record_id: str) -> str:
        """Sync form: return the value of record ``record_id``."""
        return self._lookup(record_id)

    async def fetch_record_async(self, record_id: str) -> str:
        """Coroutine-function form (DoD-4, DEC-004-1): return the value of record ``record_id``."""
        return self._lookup(record_id)

    def called_ids(self) -> set[str]:
        """The distinct ids the tool was called with."""
        return set(self.calls)


# The nested tool-definition shape of data-model § Message / the tool loop contract.
_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": "Fetch the stored value of one record by its id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "record_id": {"type": "string", "description": "The record id, e.g. A."},
                },
                "required": ["record_id"],
            },
        },
    }
]


class _RecordQuery(BaseModel):
    """Arguments of ``fetch_record`` for the typed definition built by ``pydantic_to_openai_tool`` (DoD-4)."""

    record_id: str


_TWO_RECORDS_PROMPT = (
    "Use the fetch_record tool to fetch record A and record B, then answer in one sentence that states "
    "both values."
)
_ONE_RECORD_PROMPT = "Use the fetch_record tool to fetch record A, then tell me its value in one sentence."
_SCHEMA_PROMPT = (
    "Use the fetch_record tool to fetch record A, then answer with a JSON object holding the record id "
    "as record_id and the fetched value as value."
)

# The json_schema answer format requiring record_id and value (DEC-005-11), in the app-facing shape
# that DEC-003-10 translates for the Responses protocol.
_SCHEMA_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "record_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "record_id": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["record_id", "value"],
            "additionalProperties": False,
        },
    },
}


def _history_with_tool_turns() -> list[dict[str, Any]]:
    """Caller history with a prior tool round (DoD-3; DEC-005-11, 004.context.md shape).

    User turn, assistant turn with empty content and one tool call (``call_prev_c`` for record C), the
    matching tool-result turn, then a user turn asking about C's value and to fetch record D.
    """
    return [
        {"role": "user", "content": "Use the fetch_record tool to fetch record C."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_prev_c",
                    "type": "function",
                    "function": {"name": _TOOL_NAME, "arguments": json.dumps({"record_id": "C"})},
                }
            ],
        },
        {"role": "tool", "content": _RECORD_VALUES["C"], "tool_call_id": "call_prev_c"},
        {
            "role": "user",
            "content": "What was the value of record C? Also fetch record D with the tool and tell me its value.",
        },
    ]


# ---------------------------------------------------------------------------
# Shared test bodies (one test function per backend calls these, DEC-005-5)
# ---------------------------------------------------------------------------


async def _two_records_not_streamed(backend: str) -> None:
    """DoD-1 body: the tool was called for A and B; the answer is non-empty and holds no marker."""
    tool = _RecordTool()
    client = build_client(backend)
    async with client:
        reply = await client.chat_with_tools(
            [{"role": "user", "content": _TWO_RECORDS_PROMPT}],
            tools_definitions=copy.deepcopy(_TOOL_DEFINITIONS),
            tools={_TOOL_NAME: tool.fetch_record},
        )

    assert isinstance(reply, str)
    # One round or two is the model's choice and is not asserted (DEC-005-11).
    assert {"A", "B"} <= tool.called_ids(), f"expected calls for A and B, got {tool.calls!r}"
    assert reply.strip(), "expected a non-empty answer"
    assert_no_markers(reply)


async def _two_records_streamed(backend: str) -> None:
    """DoD-2 body: both ids called; non-empty answer contained in the recorder; well-formed blocks."""
    tool = _RecordTool()
    recorder = PieceRecorder()
    client = build_client(backend)
    async with client:
        reply = await client.chat_with_tools(
            [{"role": "user", "content": _TWO_RECORDS_PROMPT}],
            tools_definitions=copy.deepcopy(_TOOL_DEFINITIONS),
            tools={_TOOL_NAME: tool.fetch_record},
            stream=True,
            on_delta=recorder,
        )

    assert isinstance(reply, str)
    assert {"A", "B"} <= tool.called_ids(), f"expected calls for A and B, got {tool.calls!r}"
    assert answer_text(reply), "expected a non-empty answer"
    # The receiver gets every round's pieces but only the final answer is returned (interfaces § Streaming
    # callback), so the check is containment, not equality.
    assert reply.strip() in recorder.text, "the returned answer is not contained in the receiver's output"
    assert_well_formed_reasoning(recorder.text)


async def _history_with_tool_turns_not_streamed(backend: str) -> None:
    """DoD-3 body: a call over caller-supplied tool turns completes non-empty; the caller's list is unchanged."""
    tool = _RecordTool()
    messages = _history_with_tool_turns()
    # A deep copy taken before the call is what the list must still equal afterwards (004.context.md).
    before = copy.deepcopy(messages)
    client = build_client(backend)
    async with client:
        reply = await client.chat_with_tools(
            messages,
            tools_definitions=copy.deepcopy(_TOOL_DEFINITIONS),
            tools={_TOOL_NAME: tool.fetch_record},
        )

    assert isinstance(reply, str)
    assert reply.strip(), "expected a non-empty answer"
    assert messages == before, "the caller's message list was changed by the call"


async def _pydantic_async_tool_not_streamed(backend: str) -> None:
    """DoD-4 body: a pydantic-built definition with a coroutine-function tool; the tool ran, answer non-empty."""
    tool = _RecordTool()
    definition = pydantic_to_openai_tool(_TOOL_NAME, "Fetch the stored value of one record by its id.", _RecordQuery)
    client = build_client(backend)
    async with client:
        reply = await client.chat_with_tools(
            [{"role": "user", "content": _ONE_RECORD_PROMPT}],
            tools_definitions=[definition],
            tools={_TOOL_NAME: tool.fetch_record_async},
        )

    assert isinstance(reply, str)
    assert tool.calls, "expected the async tool to run at least once"
    assert reply.strip(), "expected a non-empty answer"


async def _schema_answer_streamed(backend: str) -> None:
    """DoD-5 body: streamed, json_schema answer; tool ran, answer is a JSON object with both keys, contained."""
    tool = _RecordTool()
    recorder = PieceRecorder()
    client = build_client(backend)
    async with client:
        reply = await client.chat_with_tools(
            [{"role": "user", "content": _SCHEMA_PROMPT}],
            tools_definitions=copy.deepcopy(_TOOL_DEFINITIONS),
            tools={_TOOL_NAME: tool.fetch_record},
            stream=True,
            on_delta=recorder,
            response_format=copy.deepcopy(_SCHEMA_FORMAT),
        )

    assert isinstance(reply, str)
    assert tool.calls, "expected the tool to run"
    # Parsed through answer_text so a reasoning block that does appear never breaks the check (DEC-005-4).
    parsed = json.loads(answer_text(reply))
    assert isinstance(parsed, dict), f"expected a JSON object, got {parsed!r}"
    assert {"record_id", "value"} <= parsed.keys(), f"expected keys record_id and value, got {sorted(parsed)}"
    assert reply.strip() in recorder.text, "the returned answer is not contained in the receiver's output"


# ---------------------------------------------------------------------------
# DoD-1 — two records, not streamed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_tools_two_records_not_streamed__DoD1():
    """OpenAI: the tool is called for A and B; the answer is non-empty with no reasoning marker (DoD-1)."""
    await _two_records_not_streamed("openai")


@pytest.mark.asyncio
async def test_llamaswap_tools_two_records_not_streamed__DoD1():
    """llama-swap: the tool is called for A and B; the answer is non-empty with no reasoning marker (DoD-1)."""
    await _two_records_not_streamed("llamaswap")


# ---------------------------------------------------------------------------
# DoD-2 — two records, streamed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_tools_two_records_streamed__DoD2():
    """OpenAI streamed: both ids called; answer non-empty, in the recorder; blocks well-formed (DoD-2)."""
    await _two_records_streamed("openai")


@pytest.mark.asyncio
async def test_llamaswap_tools_two_records_streamed__DoD2():
    """llama-swap streamed: both ids called; answer non-empty, in the recorder; blocks well-formed (DoD-2)."""
    await _two_records_streamed("llamaswap")


# ---------------------------------------------------------------------------
# DoD-3 — caller history already holding tool turns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_tools_history_with_tool_turns__DoD3():
    """OpenAI: history with a prior tool round gives a non-empty answer; caller list unchanged (DoD-3)."""
    await _history_with_tool_turns_not_streamed("openai")


@pytest.mark.asyncio
async def test_llamaswap_tools_history_with_tool_turns__DoD3():
    """llama-swap: history with a prior tool round gives a non-empty answer; caller list unchanged (DoD-3)."""
    await _history_with_tool_turns_not_streamed("llamaswap")


# ---------------------------------------------------------------------------
# DoD-4 — pydantic-built definition, coroutine-function tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_tools_pydantic_async_tool__DoD4():
    """OpenAI: the pydantic-defined async tool runs at least once; the answer is non-empty (DoD-4)."""
    await _pydantic_async_tool_not_streamed("openai")


@pytest.mark.asyncio
async def test_llamaswap_tools_pydantic_async_tool__DoD4():
    """llama-swap: the pydantic-defined async tool runs at least once; the answer is non-empty (DoD-4)."""
    await _pydantic_async_tool_not_streamed("llamaswap")


# ---------------------------------------------------------------------------
# DoD-5 — streamed, json_schema final answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_tools_schema_answer_streamed__DoD5():
    """OpenAI streamed json_schema answer: tool ran; JSON object with both keys; in the recorder (DoD-5)."""
    await _schema_answer_streamed("openai")


# The server's schema grammar is the known environment limit; non-strict so a fixed server shows as XPASS.
@pytest.mark.xfail(strict=False, reason="llama.cpp json_schema grammar")
@pytest.mark.asyncio
async def test_llamaswap_tools_schema_answer_streamed__DoD5():
    """llama-swap streamed json_schema answer, same checks, expected failure per DEC-005-9 (DoD-5)."""
    await _schema_answer_streamed("llamaswap")
