"""Live checks of a reply cut at the app's own token cap in tool chat and checked JSON on the Responses classes.

Per DEC-005-13 and DEC-005-14, on
``OpenAIAPIClient`` and ``LlamaSwapAPIClient`` against real servers: a tool conversation whose final answer is
cut at the cap the app set returns the cut answer, streamed and not (US-062.AC-4), and a checked JSON answer cut
at that cap ends with ``LLMError`` like any invalid JSON answer (US-062.AC-5). Test-only (DEC-005-1): a failure
caused by the library is a finding, never a reason to weaken a check here. Nothing is imported from another
live test file (DEC-005-2); no check is marked as an expected failure (DoD-3, DEC-005-14).
"""

import copy
import warnings
from typing import Any

import pytest

from llm import LLMError
from tests.integrations.live_support import (
    PieceRecorder,
    answer_text,
    assert_no_markers,
    assert_well_formed_reasoning,
    build_client,
)

# ---------------------------------------------------------------------------
# Module-local inputs and helpers (DEC-005-2)
# ---------------------------------------------------------------------------

#: The app's token cap of DEC-005-8, set as a call option so "the app set the limit" holds (DEC-arch-13).
#: 16 is OpenAI's lowest accepted output cap, so the call is valid and a long answer is cut at once.
_TOKEN_CAP_OPTIONS: dict[str, Any] = {"max_tokens": 16}

#: DEC-005-13: tells the model no tool is needed, so the cut lands in the answer round (the case US-062.AC-4
#: names) rather than inside a tool call's arguments.
_NO_TOOL_SYSTEM = (
    "No tool is needed for this request. Do not call any tool. "
    "Answer the user directly in plain prose."
)

#: Neutral tool name and a value that cannot be guessed (tests/integrations/AGENTS.md § Conventions).
_TOOL_NAME = "query_index"
_TOOL_VALUE = "ZR4-8813-quill"

#: The one neutral tool in today's nested ``function`` definition shape, one text argument (DEC-arch-10, DEC-005-13).
_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": "Look up one entry of the archive index by its key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The index key to look up."},
                },
                "required": ["key"],
            },
        },
    }
]


class _RecordingTool:
    """The neutral tool callable; it records every key it is called with.

    The calls are kept only so a failing run can be diagnosed: whether the tool ran is the model's choice and
    is never asserted (DEC-005-13).
    """

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.calls: list[str] = []

    def query_index(self, key: str) -> str:
        """Record ``key`` and return the unguessable index value."""
        self.calls.append(key)
        return _TOOL_VALUE


def _long_answer_messages() -> list[dict[str, str]]:
    """The long-answer prompt (DEC-005-8): asks for far more than the 16-token cap allows."""
    return [
        {
            "role": "user",
            "content": "Write a 500-word essay about the history of lighthouses, in several paragraphs.",
        }
    ]


def _long_json_messages() -> list[dict[str, str]]:
    """The long-JSON prompt (DEC-005-14): many records, far beyond 16 tokens; it names "JSON" for OpenAI's mode."""
    return [
        {
            "role": "user",
            "content": (
                "Reply with a JSON object with one key \"records\" holding a list of 40 fictional library "
                "books. Each record is a JSON object with the keys \"title\" (text), \"author\" (text) and "
                "\"pages\" (an integer). Reply with the JSON only."
            ),
        }
    ]


def _expected_json_shape() -> dict[str, Any]:
    """The expected shape passed to ``chat_json``; it only turns the parse check on (DEC-000-37)."""
    return {
        "type": "object",
        "properties": {
            "records": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "author": {"type": "string"},
                        "pages": {"type": "integer"},
                    },
                    "required": ["title", "author", "pages"],
                },
            }
        },
        "required": ["records"],
    }


def _assert_no_library_notice(caught: list[warnings.WarningMessage]) -> None:
    """Assert no recorded warning is of the ``UserWarning`` family (DEC-005-12 "No notice").

    The library's own notices live in that family (DEC-arch-6), and a cut at the app's limit gives none
    (DEC-arch-13). Other categories (third-party deprecation / resource warnings) are deliberately not counted.
    """
    notices = [w for w in caught if issubclass(w.category, UserWarning)]
    assert not notices, f"unexpected library notice(s): {[f'{w.category.__name__}: {w.message}' for w in notices]}"


# ---------------------------------------------------------------------------
# Shared bodies
# ---------------------------------------------------------------------------


async def _run_cut_tool_conversation(backend: str, *, stream: bool) -> tuple[str, PieceRecorder | None]:
    """Run the capped tool conversation and return ``(reply, recorder)``; checks no error and no notice.

    One neutral tool offered, the long-answer prompt, the no-tool system instruction and the cap as a call
    option; ``max_loops`` stays at its default (DEC-005-13). OpenAI uses ``OPENAI_MODEL`` and llama-swap
    runs thinking off, both through the builder (DEC-005-12, DEC-005-4). The warning block surrounds the call
    with ``simplefilter("always")`` so de-duplication can neither hide nor fake a notice (DEC-005-5).
    """
    tool = _RecordingTool()
    recorder = PieceRecorder() if stream else None
    client = build_client(backend)
    async with client:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                reply = await client.chat_with_tools(
                    _long_answer_messages(),
                    tools_definitions=copy.deepcopy(_TOOL_DEFINITIONS),
                    tools={_TOOL_NAME: tool.query_index},
                    system=_NO_TOOL_SYSTEM,
                    options=dict(_TOKEN_CAP_OPTIONS),
                    stream=stream,
                    on_delta=recorder,
                )
            except LLMError as err:
                pytest.fail(
                    f"a tool answer cut at the app's limit must be returned, not raise LLMError: {err} "
                    f"(tool calls: {tool.calls})"
                )
    _assert_no_library_notice(caught)
    assert isinstance(reply, str)
    assert answer_text(reply), "expected non-empty answer text from the cut tool conversation"
    return reply, recorder


async def _assert_cut_tool_answer_not_streamed(backend: str) -> None:
    """DoD-1 body: the non-streamed cut answer is returned, non-empty, with no reasoning marker, no notice."""
    reply, _ = await _run_cut_tool_conversation(backend, stream=False)
    assert_no_markers(reply)


async def _assert_cut_tool_answer_streamed(backend: str) -> None:
    """DoD-2 body: the streamed cut answer is returned, contained in the recorder output, well-formed, no notice.

    Containment, not equality: on the tool path the receiver gets every round's pieces while only the final
    answer is returned (interfaces.md § Streaming callback; DEC-005-13).
    """
    reply, recorder = await _run_cut_tool_conversation(backend, stream=True)
    assert recorder is not None
    assert reply.strip() in recorder.text, "the returned answer must be contained in what the receiver got"
    assert_well_formed_reasoning(recorder.text)


async def _assert_cut_checked_json_raises(backend: str) -> None:
    """DoD-3 body: a checked ``chat_json`` answer cut at the cap ends with ``LLMError``.

    The expected shape is given so the parse check runs, and no ``response_format`` is in the options, so the
    plain JSON text format is used (DEC-005-14). Only the error type is asserted, never its text (DEC-005-6).
    """
    client = build_client(backend)
    async with client:
        with pytest.raises(LLMError):
            await client.chat_json(
                _long_json_messages(),
                response_schema=_expected_json_shape(),
                options=dict(_TOKEN_CAP_OPTIONS),
            )


# ---------------------------------------------------------------------------
# DoD-1: non-streamed tool conversation, final answer cut at the app's cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_tool_answer_cut_at_app_limit_returned__DoD1() -> None:
    """OpenAI: non-streamed tool chat cut at the app's cap returns non-empty text, no marker, no notice (DoD-1)."""
    await _assert_cut_tool_answer_not_streamed("openai")


@pytest.mark.asyncio
async def test_llamaswap_tool_answer_cut_at_app_limit_returned__DoD1() -> None:
    """llama-swap: non-streamed tool chat cut at the app's cap returns non-empty text, no marker, no notice (DoD-1)."""
    await _assert_cut_tool_answer_not_streamed("llamaswap")


# ---------------------------------------------------------------------------
# DoD-2: streamed tool conversation, final answer cut at the app's cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_streamed_tool_answer_cut_at_app_limit_returned__DoD2() -> None:
    """OpenAI: streamed cut tool answer is returned, contained in the recorder, well-formed, no notice (DoD-2)."""
    await _assert_cut_tool_answer_streamed("openai")


@pytest.mark.asyncio
async def test_llamaswap_streamed_tool_answer_cut_at_app_limit_returned__DoD2() -> None:
    """llama-swap: streamed cut tool answer is returned, contained in the recorder, well-formed, no notice (DoD-2)."""
    await _assert_cut_tool_answer_streamed("llamaswap")


# ---------------------------------------------------------------------------
# DoD-3: checked JSON answer cut at the app's cap fails as invalid JSON
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_chat_json_cut_at_app_limit_raises__DoD3() -> None:
    """OpenAI: ``chat_json`` with an expected shape, cut at the app's cap, ends with ``LLMError`` (DoD-3)."""
    await _assert_cut_checked_json_raises("openai")


@pytest.mark.asyncio
async def test_llamaswap_chat_json_cut_at_app_limit_raises__DoD3() -> None:
    """llama-swap: ``chat_json`` with an expected shape, cut at the app's cap, ends with ``LLMError`` (DoD-3)."""
    await _assert_cut_checked_json_raises("llamaswap")
