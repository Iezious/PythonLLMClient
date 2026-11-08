"""Live settings and answer-format tests on the Responses classes (feature 005, step 002).

Covers, on ``OpenAIAPIClient`` and ``LlamaSwapAPIClient`` against real servers: a reply cut at the token cap the
app set is returned as text with no error and no library notice, in ``chat`` and ``generate``, streamed and not,
with the cap per call or as a client default (US-062, DEC-arch-13, DEC-005-12); a no-counterpart setting is
dropped with one warning per client; ``json_schema`` answers in ``chat`` and ``chat_json``; and accepted
sampling / reasoning-effort settings.
Plan: ``docs/plans/005.responses-live-tests/002.settings-formats.md``. Test-only (DEC-005-1): a failure caused
by the library is a finding, never a reason to weaken a check here.
"""

import json
import warnings
from typing import Any

import pytest

from llm import LLMError, LLMSettingIgnoredWarning
from tests.integrations.live_support import (
    PieceRecorder,
    answer_text,
    assert_no_markers,
    assert_well_formed_reasoning,
    build_client,
    openai_reasoning_model,
)

# ---------------------------------------------------------------------------
# Module-local inputs and helpers (step 002 Interface intent)
# ---------------------------------------------------------------------------

#: The token cap of DEC-005-8: 16 is OpenAI's lowest accepted output cap, so the call is valid and is cut
#: off at once by a long-answer prompt.
_TOKEN_CAP_OPTIONS: dict[str, Any] = {"max_tokens": 16}

#: The ignored setting of DEC-005-8: ``seed`` has no Responses counterpart on either class (DEC-003-9).
_IGNORED_SETTING = "seed"
_IGNORED_SETTING_OPTIONS: dict[str, Any] = {_IGNORED_SETTING: 7}

#: The two keys the ``json_schema`` answer format requires (DoD-4, DoD-5).
_REQUIRED_KEYS = ("title", "pages")

#: Sampling settings with a Responses counterpart (DEC-003-9); used on the plain models only, since
#: reasoning models may refuse them (002.context.md).
_SAMPLING_OPTIONS: dict[str, Any] = {"temperature": 0.7, "top_p": 0.9}

#: DEC-005-9: llama.cpp's schema grammar is known to fail on the server; non-strict so a fixed server
#: shows as XPASS without failing the run. The only expected-failure mark in this file.
_LLAMA_SWAP_JSON_SCHEMA_XFAIL = pytest.mark.xfail(strict=False, reason="llama.cpp json_schema grammar")


#: The long-answer prompt (DEC-005-8): asks for far more than the 16-token cap allows. Shared by ``chat``
#: (as a user turn) and ``generate`` (as the prompt) so both operations get the same request.
_LONG_ANSWER_PROMPT = "Write a 500-word essay about the history of lighthouses, in several paragraphs."


def _long_answer_messages() -> list[dict[str, str]]:
    """The long-answer prompt of DEC-005-8 as a single user turn."""
    return [{"role": "user", "content": _LONG_ANSWER_PROMPT}]


def _short_messages() -> list[dict[str, str]]:
    """A short prompt any model answers with a few words; used where only a non-empty reply is checked."""
    return [{"role": "user", "content": "Name one colour of the rainbow. Answer in one short sentence."}]


def _json_messages() -> list[dict[str, str]]:
    """A prompt that asks for an object with the two required keys of :func:`_json_schema_format`."""
    return [
        {
            "role": "user",
            "content": (
                "Describe one fictional library book as a JSON object with the keys "
                '"title" (text) and "pages" (an integer). Reply with the JSON object only.'
            ),
        }
    ]


def _json_schema() -> dict[str, Any]:
    """The object schema requiring both :data:`_REQUIRED_KEYS`; also the ``chat_json`` expected shape."""
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "pages": {"type": "integer"},
        },
        "required": list(_REQUIRED_KEYS),
        "additionalProperties": False,
    }


def _json_schema_format() -> dict[str, Any]:
    """The ``json_schema`` ``response_format`` in today's nested shape (002.context.md; flattened by DEC-003-10)."""
    return {
        "type": "json_schema",
        "json_schema": {"name": "library_book", "schema": _json_schema()},
    }


def _ignored_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    """Keep only ``LLMSettingIgnoredWarning`` records.

    The filter matters: unrelated warnings (aiohttp, pydantic) recorded in the same block must not change
    the once-per-client count (002.context.md, DEC-005-5).
    """
    return [w for w in caught if issubclass(w.category, LLMSettingIgnoredWarning)]


def _assert_no_library_notice(caught: list[warnings.WarningMessage]) -> None:
    """The "no library notice" check of DEC-005-12: no recorded warning is of the ``UserWarning`` family.

    The library's own notices live in that family (DEC-arch-6), ``LLMSettingIgnoredWarning`` included, so a
    cut notice of any library category is found. Other categories (third-party deprecation, resource or
    runtime warnings) are deliberately not counted, so unrelated noise cannot break the check.
    """
    notices = [w for w in caught if issubclass(w.category, UserWarning)]
    assert not notices, (
        "a reply cut at the app's own cap must raise no notice: "
        f"{[f'{w.category.__name__}: {w.message}' for w in notices]}"
    )


def _assert_json_object_with_keys(reply: str) -> None:
    """Assert ``reply`` parses as a JSON object holding both required keys (DEC-005-6: JSON validity, keys).

    The answer text is parsed, so a reasoning block that does appear never breaks the check (DEC-005-4).
    """
    assert isinstance(reply, str)
    parsed = json.loads(answer_text(reply))
    assert isinstance(parsed, dict), f"expected a JSON object, got {type(parsed).__name__}"
    for key in _REQUIRED_KEYS:
        assert key in parsed, f"required key {key!r} missing from the answer"


# ---------------------------------------------------------------------------
# Shared bodies
# ---------------------------------------------------------------------------


async def _run_capped(
    backend: str,
    operation: str,
    *,
    stream: bool,
    cap_as_default: bool = False,
) -> tuple[str, PieceRecorder | None]:
    """Run one capped long-answer call and check the no-error and no-notice rules (DoD-1/2/8/9, DEC-005-12).

    ``operation`` is ``"chat"`` or ``"generate"``. The cap of DEC-005-8 goes in the call options, or, with
    ``cap_as_default``, in the client's ``default_options`` with no cap in the call options (DoD-9). Models:
    the builder's plain model (``OPENAI_MODEL`` on OpenAI); on llama-swap the builder merges the defaults over
    thinking off, so the 16 tokens go to answer text (DEC-005-4).

    The call runs inside ``catch_warnings(record=True)`` + ``simplefilter("always")`` so no notice is hidden by
    Python's de-duplication. An ``LLMError`` is turned into a test failure naming the rule it breaks (a cut at
    the app's own cap is not an error, DEC-arch-13). Only the structure of DEC-005-6 is checked afterwards by
    the caller: never the exact text, never whether the reply was in fact cut (002.context.md).

    Returns:
        The call's return value and the recorder (``None`` when not streamed).
    """
    client = build_client(backend, default_options=dict(_TOKEN_CAP_OPTIONS) if cap_as_default else None)
    call_options = None if cap_as_default else dict(_TOKEN_CAP_OPTIONS)
    recorder = PieceRecorder() if stream else None
    stream_kwargs: dict[str, Any] = {"stream": True, "on_delta": recorder} if stream else {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        async with client:
            try:
                if operation == "chat":
                    reply = await client.chat(_long_answer_messages(), options=call_options, **stream_kwargs)
                else:
                    reply = await client.generate(_LONG_ANSWER_PROMPT, options=call_options, **stream_kwargs)
            except LLMError as err:
                pytest.fail(f"a reply cut at the app's own cap must be returned as text, not LLMError: {err}")
    _assert_no_library_notice(caught)
    assert isinstance(reply, str), f"the capped {operation} must return text, got {type(reply).__name__}"
    assert answer_text(reply), f"the capped {operation} must return non-empty answer text"
    return reply, recorder


async def _assert_capped_not_streamed(backend: str, operation: str, *, cap_as_default: bool = False) -> None:
    """Capped non-streamed call: text, no error, no notice, no reasoning marker (DoD-1, DoD-8, DoD-9)."""
    reply, _ = await _run_capped(backend, operation, stream=False, cap_as_default=cap_as_default)
    assert_no_markers(reply)


async def _assert_capped_streamed(backend: str, operation: str) -> None:
    """Capped streamed call: text, no error, no notice, return equal to the joined pieces (DoD-2, DoD-8).

    The recorder output must also hold only well-formed reasoning blocks (DEC-005-6); on a cut stream an
    open block must have been closed (DEC-003-19).
    """
    reply, recorder = await _run_capped(backend, operation, stream=True)
    assert recorder is not None
    assert reply == recorder.text, "the streamed return must equal the joined recorder pieces"
    assert_well_formed_reasoning(recorder.text)


async def _assert_ignored_setting_warns_once_per_client(backend: str) -> None:
    """Check the once-per-setting-per-client warning rule (DoD-3, DEC-003-6, DEC-003-9).

    Two calls on one client record exactly one warning naming the setting; a second client instance called
    once records exactly one more. Counted under ``record=True`` + ``simplefilter("always")`` so Python's
    per-location de-duplication neither hides nor fakes the once-per-client rule.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        first = build_client(backend)
        async with first:
            for _ in range(2):
                reply = await first.chat(_short_messages(), options=dict(_IGNORED_SETTING_OPTIONS))
                assert isinstance(reply, str)
                assert reply.strip(), "the call with an ignored setting must still return text"
        after_first = _ignored_warnings(caught)
        assert len(after_first) == 1, f"expected one ignored-setting warning per client, got {len(after_first)}"
        assert _IGNORED_SETTING in str(after_first[0].message), "the warning must name the ignored setting"

        second = build_client(backend)
        async with second:
            reply = await second.chat(_short_messages(), options=dict(_IGNORED_SETTING_OPTIONS))
            assert isinstance(reply, str)
            assert reply.strip(), "the call with an ignored setting must still return text"
        after_second = _ignored_warnings(caught)
        assert len(after_second) == 2, f"a second client must add exactly one warning, got {len(after_second)}"
        assert _IGNORED_SETTING in str(after_second[1].message), "the warning must name the ignored setting"


async def _assert_chat_json_schema(backend: str) -> None:
    """Non-streamed ``chat`` with the nested ``json_schema`` format; the reply holds both keys (DoD-4)."""
    client = build_client(backend)
    async with client:
        reply = await client.chat(_json_messages(), options={"response_format": _json_schema_format()})
    _assert_json_object_with_keys(reply)


async def _assert_chat_json_with_schema_format(backend: str) -> None:
    """``chat_json`` with the ``json_schema`` format in options and an expected shape (DoD-5).

    ``response_schema`` only turns on the parse check (002.context.md); the format in options drives the
    server.
    """
    client = build_client(backend)
    async with client:
        reply = await client.chat_json(
            _json_messages(),
            response_schema=_json_schema(),
            options={"response_format": _json_schema_format()},
        )
    _assert_json_object_with_keys(reply)


async def _assert_sampling_accepted(backend: str) -> None:
    """``temperature`` + ``top_p`` give non-empty text and no ignored-setting warning (DoD-6, DEC-003-9)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # build_client with model=None uses the plain model (OPENAI_MODEL on OpenAI), as DoD-6 requires.
        client = build_client(backend)
        async with client:
            reply = await client.chat(_short_messages(), options=dict(_SAMPLING_OPTIONS))
    assert isinstance(reply, str)
    assert reply.strip(), "the call with sampling settings must return text"
    ignored = _ignored_warnings(caught)
    assert not ignored, f"sampling settings must not be reported as ignored: {[str(w.message) for w in ignored]}"


# ---------------------------------------------------------------------------
# DoD-1 — capped reply, not streamed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_token_cap_chat_not_streamed_returns_text__DoD1() -> None:
    """Capped non-streamed OpenAI chat returns non-empty text, no LLMError, no marker, no notice (DoD-1)."""
    await _assert_capped_not_streamed("openai", "chat")


@pytest.mark.asyncio
async def test_llamaswap_token_cap_chat_not_streamed_returns_text__DoD1() -> None:
    """Capped non-streamed llama-swap chat returns non-empty text, no LLMError, no marker, no notice (DoD-1)."""
    await _assert_capped_not_streamed("llamaswap", "chat")


# ---------------------------------------------------------------------------
# DoD-2 — capped reply, streamed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_token_cap_chat_streamed_returns_text__DoD2() -> None:
    """Capped streamed OpenAI chat returns the joined pieces as non-empty text, well-formed, no notice (DoD-2)."""
    await _assert_capped_streamed("openai", "chat")


@pytest.mark.asyncio
async def test_llamaswap_token_cap_chat_streamed_returns_text__DoD2() -> None:
    """Capped streamed llama-swap chat returns the joined pieces as non-empty text, well-formed (DoD-2)."""
    await _assert_capped_streamed("llamaswap", "chat")


# ---------------------------------------------------------------------------
# DoD-8 — capped generate, not streamed and streamed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_token_cap_generate_not_streamed_returns_text__DoD8() -> None:
    """Capped non-streamed OpenAI generate returns non-empty text, no LLMError, no marker, no notice (DoD-8)."""
    await _assert_capped_not_streamed("openai", "generate")


@pytest.mark.asyncio
async def test_llamaswap_token_cap_generate_not_streamed_returns_text__DoD8() -> None:
    """Capped non-streamed llama-swap generate returns non-empty text, no LLMError, no marker (DoD-8)."""
    await _assert_capped_not_streamed("llamaswap", "generate")


@pytest.mark.asyncio
async def test_openai_token_cap_generate_streamed_returns_text__DoD8() -> None:
    """Capped streamed OpenAI generate returns the joined pieces as non-empty text, well-formed (DoD-8)."""
    await _assert_capped_streamed("openai", "generate")


@pytest.mark.asyncio
async def test_llamaswap_token_cap_generate_streamed_returns_text__DoD8() -> None:
    """Capped streamed llama-swap generate returns the joined pieces as non-empty text, well-formed (DoD-8)."""
    await _assert_capped_streamed("llamaswap", "generate")


# ---------------------------------------------------------------------------
# DoD-9 — cap as a client default
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_token_cap_client_default_returns_text__DoD9() -> None:
    """OpenAI chat with the cap only in default_options returns non-empty text, no error/marker/notice (DoD-9)."""
    await _assert_capped_not_streamed("openai", "chat", cap_as_default=True)


@pytest.mark.asyncio
async def test_llamaswap_token_cap_client_default_returns_text__DoD9() -> None:
    """llama-swap chat with the cap only in default_options returns non-empty text, no error/marker (DoD-9)."""
    await _assert_capped_not_streamed("llamaswap", "chat", cap_as_default=True)


# ---------------------------------------------------------------------------
# DoD-3 — ignored setting, one warning per client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_ignored_setting_warns_once_per_client__DoD3() -> None:
    """OpenAI: seed is dropped with one warning per client, naming the setting; calls still answer (DoD-3)."""
    await _assert_ignored_setting_warns_once_per_client("openai")


@pytest.mark.asyncio
async def test_llamaswap_ignored_setting_warns_once_per_client__DoD3() -> None:
    """llama-swap: seed is dropped with one warning per client, naming the setting; calls still answer (DoD-3)."""
    await _assert_ignored_setting_warns_once_per_client("llamaswap")


# ---------------------------------------------------------------------------
# DoD-4 — json_schema answer in chat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_chat_json_schema_answer__DoD4() -> None:
    """OpenAI chat with a nested json_schema format returns a JSON object holding both keys (DoD-4)."""
    await _assert_chat_json_schema("openai")


@_LLAMA_SWAP_JSON_SCHEMA_XFAIL
@pytest.mark.asyncio
async def test_llamaswap_chat_json_schema_answer__DoD4() -> None:
    """llama-swap chat with a nested json_schema format returns a JSON object holding both keys (DoD-4)."""
    await _assert_chat_json_schema("llamaswap")


# ---------------------------------------------------------------------------
# DoD-5 — json_schema answer in chat_json
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_chat_json_schema_format__DoD5() -> None:
    """OpenAI chat_json with the json_schema format and an expected shape returns both keys (DoD-5)."""
    await _assert_chat_json_with_schema_format("openai")


@_LLAMA_SWAP_JSON_SCHEMA_XFAIL
@pytest.mark.asyncio
async def test_llamaswap_chat_json_schema_format__DoD5() -> None:
    """llama-swap chat_json with the json_schema format and an expected shape returns both keys (DoD-5)."""
    await _assert_chat_json_with_schema_format("llamaswap")


# ---------------------------------------------------------------------------
# DoD-6 — sampling settings accepted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_sampling_settings_accepted__DoD6() -> None:
    """OpenAI (OPENAI_MODEL) chat with temperature and top_p answers with no ignored-setting warning (DoD-6)."""
    await _assert_sampling_accepted("openai")


@pytest.mark.asyncio
async def test_llamaswap_sampling_settings_accepted__DoD6() -> None:
    """llama-swap chat with temperature and top_p answers with no ignored-setting warning (DoD-6)."""
    await _assert_sampling_accepted("llamaswap")


# ---------------------------------------------------------------------------
# DoD-7 — explicit reasoning effort on the OpenAI reasoning model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_reasoning_effort_medium_accepted__DoD7() -> None:
    """OpenAI reasoning model with reasoning_effort "medium" returns non-empty text with no marker (DoD-7).

    Skips without a key (the builder's key reader). The effort is not combined with sampling settings,
    which reasoning models may refuse (002.context.md).
    """
    client = build_client("openai", openai_reasoning_model())
    async with client:
        reply = await client.chat(_short_messages(), options={"reasoning_effort": "medium"})
    assert isinstance(reply, str)
    assert reply.strip(), "the reasoning-effort call must return text"
    assert_no_markers(reply)
