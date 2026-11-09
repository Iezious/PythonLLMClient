"""Live tolerant tool-failure check on both Responses classes (feature 006 step 010, DEC-006-12).

Runs non-streamed ``chat_with_tools`` with the default failure handling (``tool_errors`` not passed)
against real llama-swap and OpenAI servers. The only offered tool, ``fetch_record``, always raises an
ordinary exception. The checks are that the conversation still ends with a non-empty answer, and that a
plain-function observer saw a ``start`` event for the tool immediately followed by an ``error`` event for
the same call with ``error_kind == "tool_raised"`` (US-064.AC-4, US-064.AC-17, US-064.AC-20,
US-065.AC-1, US-065.AC-3). Nothing about the answer text is asserted. One test function per backend
(DEC-005-5); llama-swap is never skipped, OpenAI is skipped without its key.
"""

import copy
from typing import Any

import pytest

from llm import ToolEvent
from tests.integrations.live_support import build_client, openai_api_key

_TOOL_NAME = "fetch_record"

# The nested tool-definition shape of the tool loop contract; a neutral name whose result cannot be
# guessed (tests/integrations/AGENTS.md § Conventions).
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

# DEC-006-12: look up one record with the tool and, if the tool fails, do not retry and say the record
# is unavailable.
_PROMPT = (
    "Use the fetch_record tool to fetch record A. If the tool fails, do not call it again; instead answer "
    "in one sentence that the record is unavailable."
)


class _RecordStoreDown(RuntimeError):
    """The ordinary exception the always-failing tool raises (DEC-006-12)."""


def _fetch_record(**arguments: Any) -> str:
    """The ``fetch_record`` tool: always raises, whatever arguments the model sends.

    It accepts any keyword arguments so the failure seen is always the tool raising (``tool_raised``)
    and never an argument-name mismatch, whatever argument names the live model chooses (tools taking
    ``**kwargs`` run with whatever arguments are sent, step 006/007 DoD-11).
    """
    raise _RecordStoreDown("the record store is offline")


def _assert_start_then_tool_raised(events: list[ToolEvent]) -> None:
    """Assert some ``start`` event for ``fetch_record`` is immediately followed by its ``tool_raised`` error.

    The model may call the tool more than once despite the prompt; the contract checked is that such an
    adjacent pair exists (same tool, same call id), not how many calls there were.
    """
    pairs = list(zip(events, events[1:]))
    found = any(
        first["type"] == "start"
        and first["name"] == _TOOL_NAME
        and second["type"] == "error"
        and second["name"] == _TOOL_NAME
        and second["call_id"] == first["call_id"]
        and second["error_kind"] == "tool_raised"
        for first, second in pairs
    )
    assert found, (
        f"expected a start event for {_TOOL_NAME} immediately followed by its tool_raised error, got {events!r}"
    )


async def _tolerant_failure_not_streamed(backend: str, *, bearer_token: str | None = None) -> None:
    """Shared body for DoD-2 / DoD-3: default handling, failing tool, non-empty answer, start-then-error events."""
    events: list[ToolEvent] = []

    # The observer is a plain function appending each event (step 010 context): the order of the list is
    # the order the library delivered the events in.
    def observer(event: ToolEvent) -> None:
        events.append(event)

    client = build_client(backend, bearer_token=bearer_token)
    async with client:
        # tool_errors is deliberately not passed (the default) and max_loops is left at its default
        # (DEC-006-12).
        reply = await client.chat_with_tools(
            [{"role": "user", "content": _PROMPT}],
            tools_definitions=copy.deepcopy(_TOOL_DEFINITIONS),
            tools={_TOOL_NAME: _fetch_record},
            on_tool_event=observer,
        )

    assert isinstance(reply, str)
    assert reply.strip(), "expected a non-empty answer after the tool failed"
    _assert_start_then_tool_raised(events)


@pytest.mark.asyncio
async def test_llamaswap_responses_tolerant_tool_failure_continues__DoD2():
    """llama-swap Responses (never skipped): a raising tool under the default handling still yields a
    non-empty answer, and the observer saw start then a ``tool_raised`` error for the same call (DoD-2)."""
    await _tolerant_failure_not_streamed("llamaswap")


@pytest.mark.asyncio
async def test_openai_responses_tolerant_tool_failure_continues__DoD3():
    """OpenAI Responses (skipped without the key): same scenario and assertions as DoD-2 (DoD-3)."""
    # The key is read in the test (skips when absent) and passed as bearer_token; the library never reads
    # it (DEC-001-6).
    key = openai_api_key()
    await _tolerant_failure_not_streamed("openai", bearer_token=key)
