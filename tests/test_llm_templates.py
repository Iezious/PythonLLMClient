"""Repeated template filling (`recursive_format_placeholders`).

Covers plan 002.edge-fixes, step 005 (DEC-002-15): filling stops after the first pass that changes
nothing, keeps its 10-pass cap, fills slots found inside values, evaluates computed values only when
their slot appears, and passes their errors through (US-054, US-055).
"""

from collections import Counter
from collections.abc import Callable

import pytest

from llm.llm import recursive_format_placeholders


class CountingData(dict):
    """A mapping that counts membership tests per key.

    The step context says the helper looks up the data with a membership test per slot per pass,
    so the number of membership tests for a slot that occurs once equals the number of passes that
    saw it. This is how the tests observe when filling stops (DoD-1).
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.contains_calls: Counter[str] = Counter()

    def __contains__(self, key: object) -> bool:
        self.contains_calls[str(key)] += 1
        return super().__contains__(key)


def _counting_callable(result: str | Callable[[int], str]) -> tuple[Callable[[], str], list[int]]:
    """Return a computed value plus a call log; `result` may depend on the 1-based call number."""
    calls: list[int] = []

    def compute() -> str:
        calls.append(len(calls) + 1)
        return result(len(calls)) if callable(result) else result

    return compute, calls


def test_unknown_slots_returned_as_written_and_known_filled__DoD1() -> None:
    """Known slots are filled and unknown slots are returned exactly as written (DoD-1)."""
    data = CountingData(name="Ada")

    out = recursive_format_placeholders("Hi {{name}}, meet {{stranger}}!", data)

    assert out == "Hi Ada, meet {{stranger}}!"


def test_only_unknown_slots_stops_after_first_pass__DoD1() -> None:
    """A template with only unknown slots is looked up in one pass only, then filling stops (DoD-1)."""
    data = CountingData(name="Ada")

    out = recursive_format_placeholders("{{missing}} and {{other}}", data)

    assert out == "{{missing}} and {{other}}"
    # Pass 1 changes nothing, so there is no second pass: one lookup per unknown slot.
    assert data.contains_calls["missing"] == 1
    assert data.contains_calls["other"] == 1


def test_stops_after_pass_that_changes_nothing_once_known_slots_filled__DoD1() -> None:
    """After the known slot is filled, the next pass changes nothing and is the last (DoD-1)."""
    data = CountingData(name="Ada")

    out = recursive_format_placeholders("{{name}} / {{missing}}", data)

    assert out == "Ada / {{missing}}"
    # Pass 1 fills `name` (text changed); pass 2 sees only `missing` and changes nothing -> stop.
    assert data.contains_calls["missing"] == 2


def test_nested_fill_then_unknown_slot_stops_without_extra_passes__DoD1_DoD3() -> None:
    """Nested known slots fill across passes; the first pass that changes nothing ends filling (DoD-1, DoD-3)."""
    data = CountingData(outer="[{{inner}}]", inner="in {{missing}}")

    out = recursive_format_placeholders("{{outer}}", data)

    assert out == "[in {{missing}}]"
    # `missing` appears from pass 3 on; pass 3 changes nothing, so it is looked up exactly once.
    assert data.contains_calls["missing"] == 1


def test_self_referring_computed_value_computed_once_and_returned__DoD2() -> None:
    """A computed value yielding its own slot is computed once, its slot stays, no error (DoD-2)."""
    compute, calls = _counting_callable("{{loop}}")

    out = recursive_format_placeholders("start {{loop}} end", {"loop": compute})

    assert out == "start {{loop}} end"
    assert len(calls) == 1


def test_self_referring_plain_value_returned_without_error__DoD2() -> None:
    """A plain value equal to its own slot is returned still in the text without an error (DoD-2)."""
    data = CountingData(me="{{me}}")

    out = recursive_format_placeholders("<{{me}}>", data)

    assert out == "<{{me}}>"
    # The first pass leaves the text unchanged, so it is the only pass.
    assert data.contains_calls["me"] == 1


def test_value_containing_known_slot_is_filled_across_passes__DoD3() -> None:
    """A slot inside a value is filled too, at any nesting depth (DoD-3)."""
    data = {"greeting": "Hello {{name}}{{punct}}", "name": "{{first}} {{last}}", "first": "Ada",
            "last": "Lovelace", "punct": "!"}

    out = recursive_format_placeholders("{{greeting}}", data)

    assert out == "Hello Ada Lovelace!"


def test_nested_values_inserted_as_text_form__DoD3() -> None:
    """Non-text values found through a nested slot are inserted as their text form (DoD-3)."""
    data = {"line": "count={{n}} ratio={{r}}", "n": 3, "r": 0.5}

    out = recursive_format_placeholders("{{line}}", data)

    assert out == "count=3 ratio=0.5"


def test_ever_changing_value_ends_after_cap_without_error__DoD4() -> None:
    """A value that changes the text on every pass ends after the 10-pass cap, silently (DoD-4)."""
    # Each call yields a new number plus its own slot, so every pass changes the text.
    compute, calls = _counting_callable(lambda n: f"{n}-{{{{grow}}}}")

    out = recursive_format_placeholders("{{grow}}", {"grow": compute})

    # One occurrence per pass, computed per occurrence per pass, 10 passes at most.
    assert len(calls) == 10
    assert out == "1-2-3-4-5-6-7-8-9-10-{{grow}}"


def test_long_chain_of_values_ends_after_cap_without_error__DoD4() -> None:
    """A chain of slots deeper than 10 levels ends after the cap without error, unfinished (DoD-4)."""
    # k0 -> {{k1}} -> ... -> {{k15}} -> "end": each pass resolves one level, so 10 passes are not enough.
    data: dict[str, object] = {f"k{i}": f"{{{{k{i + 1}}}}}" for i in range(15)}
    data["k15"] = "end"

    out = recursive_format_placeholders("{{k0}}", data)

    assert out == "{{k10}}"


def test_computed_value_without_slot_is_not_computed__DoD5() -> None:
    """A computed value whose slot does not appear is never computed (DoD-5)."""
    unused, unused_calls = _counting_callable("never")
    used, used_calls = _counting_callable("yes")

    out = recursive_format_placeholders("used={{used}}", {"used": used, "unused": unused})

    assert out == "used=yes"
    assert unused_calls == []
    assert len(used_calls) == 1


def test_failing_computed_value_error_passes_through_unchanged__DoD5() -> None:
    """The error raised by a computed value reaches the caller as the same exception object (DoD-5)."""
    error = ValueError("boom")

    def fail() -> str:
        raise error

    with pytest.raises(ValueError) as excinfo:
        recursive_format_placeholders("a {{bad}} b", {"bad": fail})

    assert excinfo.value is error
