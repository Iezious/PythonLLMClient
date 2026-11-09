"""Public data shapes for the tool-event observer (DEC-006-3).

Declares ``ToolEvent`` and ``ToolErrorKind`` once, so the four tool-loop classes share one definition
that none of them owns. Data definitions only; no behaviour lives here.
"""

from typing import Literal, TypedDict

ToolErrorKind = Literal[
    "not_offered",
    "bad_arguments",
    "unknown_argument",
    "tool_raised",
    "unserialisable_result",
]
"""Closed set of tool failure kind names carried by ``ToolEvent["error_kind"]``.

Meanings are in ``docs/architecture/interfaces.md`` § Tool failures; the set itself is in
``docs/architecture/data-model.md`` § Tool events.
"""


class ToolEvent(TypedDict):
    """One event delivered to the tool-event observer (``on_tool_event``), same on both OpenAI-family protocols.

    A total typed dictionary (DEC-006-4): all six keys are present in every event, with ``None`` where
    the data model says none. A tool call yields ``start`` before the tool runs, then ``result`` or
    ``error`` (unless the observer itself failed under tolerant handling).

    Keys:
        type: ``"start"``, ``"result"`` or ``"error"``.
        name: the tool name the model requested (an empty name stays ``""``).
        call_id: the server's id for the call, or ``None`` when the server gave none.
        arguments: the raw arguments as the model sent them; a mapping is JSON-encoded, missing is ``""``.
        output: the result text the model receives, on a ``result`` event; ``None`` otherwise.
        error_kind: the failure kind, on an ``error`` event; ``None`` otherwise.
    """

    type: Literal["start", "result", "error"]
    name: str
    call_id: str | None
    arguments: str
    output: str | None
    error_kind: ToolErrorKind | None
