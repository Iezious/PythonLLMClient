"""Public error and tool-event types (feature 006 tolerant-tool-loop, step 001).

Covers the ``LLMError`` keyword-only ``status`` / ``kind`` attributes with an unchanged text form
(US-067), the closed ``LLMErrorKind`` name set (data-model.md § Error kinds), and the closed
``ToolErrorKind`` name set plus the total six-key ``ToolEvent`` shape (data-model.md § Tool events,
DEC-006-4, US-065). Expected values come from the step's Definition of done.
"""

import typing

import pytest

from llm.exceptions import LLMError, LLMErrorKind
from llm.tool_events import ToolErrorKind, ToolEvent

# The eight error kind names fixed by DoD-3 (US-067.AC-2).
_EXPECTED_ERROR_KINDS = {
    "status",
    "timeout",
    "connection",
    "interrupted",
    "invalid_reply",
    "server_failed",
    "incomplete",
    "tool",
}

# The five tool failure kind names fixed by DoD-4 (US-065.AC-3).
_EXPECTED_TOOL_ERROR_KINDS = {
    "not_offered",
    "bad_arguments",
    "unknown_argument",
    "tool_raised",
    "unserialisable_result",
}

# The six tool-event keys fixed by DoD-4 (US-065.AC-1, AC-3).
_EXPECTED_TOOL_EVENT_KEYS = {"type", "name", "call_id", "arguments", "output", "error_kind"}


# --- DoD-1: defaults and unchanged text form -------------------------------------------------


def test_error_with_message_only_has_no_status_or_kind__DoD1():
    """DoD-1: an error built from a message alone has ``status`` and ``kind`` equal to none."""
    err = LLMError("Request failed")
    assert err.status is None
    assert err.kind is None


def test_error_with_message_and_description_has_no_status_or_kind__DoD1():
    """DoD-1: an error built from a message and a description has ``status`` and ``kind`` equal to none."""
    err = LLMError("Request failed", "server said no")
    assert err.status is None
    assert err.kind is None

    err_kw = LLMError("Request failed", description="server said no")
    assert err_kw.status is None
    assert err_kw.kind is None


def test_error_text_is_the_message_alone__DoD1():
    """DoD-1 (US-067.AC-3): without a description the text form is exactly the message."""
    assert str(LLMError("Request failed")) == "Request failed"
    assert str(LLMError("Request failed", None)) == "Request failed"


def test_error_text_is_message_colon_description__DoD1():
    """DoD-1 (US-067.AC-3): with a description the text form is exactly ``message: description``."""
    assert str(LLMError("Request failed", "server said no")) == "Request failed: server said no"
    assert str(LLMError("Request failed", description="server said no")) == "Request failed: server said no"


# --- DoD-2: keyword-only status / kind -------------------------------------------------------


def test_error_exposes_status_and_kind_given_as_keywords__DoD2():
    """DoD-2 (US-067.AC-1): ``status`` and ``kind`` passed by keyword are exposed unchanged."""
    err = LLMError("Request refused", "unauthorised", status=401, kind="status")
    assert err.status == 401
    assert err.kind == "status"


def test_error_accepts_kind_without_status__DoD2():
    """DoD-2: ``kind`` alone may be given; ``status`` then stays none."""
    err = LLMError("Timed out", kind="timeout")
    assert err.kind == "timeout"
    assert err.status is None


def test_error_accepts_status_without_kind__DoD2():
    """DoD-2: ``status`` alone may be given; ``kind`` then stays none."""
    err = LLMError("Request refused", status=503)
    assert err.status == 503
    assert err.kind is None


@pytest.mark.parametrize("kind", sorted(_EXPECTED_ERROR_KINDS))
def test_error_keeps_every_kind_name_unchanged__DoD2(kind):
    """DoD-2: each error kind name passed by keyword is exposed unchanged as ``kind``."""
    assert LLMError("failure", kind=kind).kind == kind


def test_error_refuses_status_as_positional_argument__DoD2():
    """DoD-2: ``status`` is keyword-only, so a third positional argument is refused."""
    with pytest.raises(TypeError):
        LLMError("Request refused", "unauthorised", 401)


def test_error_refuses_status_and_kind_as_positional_arguments__DoD2():
    """DoD-2: ``status`` and ``kind`` are keyword-only, so they cannot be passed positionally."""
    with pytest.raises(TypeError):
        LLMError("Request refused", "unauthorised", 401, "status")


def test_status_and_kind_do_not_change_the_text_form__DoD2():
    """DoD-2 (US-067.AC-3): passing ``status`` / ``kind`` leaves the text form exactly as without them."""
    assert str(LLMError("Request refused", status=401, kind="status")) == "Request refused"
    assert (
        str(LLMError("Request refused", "unauthorised", status=401, kind="status"))
        == "Request refused: unauthorised"
    )
    assert str(LLMError("Timed out", description="after 5s", kind="timeout")) == "Timed out: after 5s"


def test_error_with_kind_is_still_raised_and_caught_as_llm_error__DoD2():
    """DoD-2: an error carrying ``status`` / ``kind`` keeps them when raised and caught as ``LLMError``."""
    with pytest.raises(LLMError) as exc_info:
        raise LLMError("Request refused", "unauthorised", status=403, kind="status")
    assert exc_info.value.status == 403
    assert exc_info.value.kind == "status"
    assert str(exc_info.value) == "Request refused: unauthorised"


# --- DoD-3: LLMErrorKind ---------------------------------------------------------------------


def test_error_kind_lists_exactly_the_eight_names__DoD3():
    """DoD-3 (US-067.AC-2): ``LLMErrorKind`` is a closed set of exactly the eight error kind names."""
    names = typing.get_args(LLMErrorKind)
    assert set(names) == _EXPECTED_ERROR_KINDS
    assert len(names) == len(_EXPECTED_ERROR_KINDS)


# --- DoD-4: ToolErrorKind and ToolEvent ------------------------------------------------------


def test_tool_error_kind_lists_exactly_the_five_names__DoD4():
    """DoD-4 (US-065.AC-3): ``ToolErrorKind`` is a closed set of exactly the five tool failure kind names."""
    names = typing.get_args(ToolErrorKind)
    assert set(names) == _EXPECTED_TOOL_ERROR_KINDS
    assert len(names) == len(_EXPECTED_TOOL_ERROR_KINDS)


def test_tool_event_declares_exactly_the_six_keys__DoD4():
    """DoD-4 (US-065.AC-1, AC-3): ``ToolEvent`` declares exactly ``type``, ``name``, ``call_id``,
    ``arguments``, ``output`` and ``error_kind``."""
    assert set(ToolEvent.__annotations__) == _EXPECTED_TOOL_EVENT_KEYS


def test_tool_event_keys_are_all_required__DoD4():
    """DoD-4 (DEC-006-4): every ``ToolEvent`` key is required; none is optional."""
    assert set(ToolEvent.__required_keys__) == _EXPECTED_TOOL_EVENT_KEYS
    assert set(ToolEvent.__optional_keys__) == set()
