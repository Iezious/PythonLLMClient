"""Public error and warning types of the library.

Declares ``LLMError`` with its ``LLMErrorKind`` set and the library warning categories
(``LLMSettingIgnoredWarning``, ``LLMToolObserverWarning``), so apps can branch and filter by type (DEC-006-3).
"""

from typing import Literal

LLMErrorKind = Literal[
    "status",
    "timeout",
    "connection",
    "interrupted",
    "invalid_reply",
    "server_failed",
    "incomplete",
    "tool",
]
"""Closed set of error kind names carried by ``LLMError.kind`` (``docs/architecture/data-model.md`` § Error kinds).

Which failure carries which kind is in ``docs/architecture/interfaces.md`` § Exposed: error contract.
"""


class LLMError(Exception):
    """Raised for any non-2xx HTTP response from the LLM server or other client-side errors.

    Carries the human ``message`` and optional ``description``, plus two optional machine-readable
    attributes: ``status`` (the HTTP status, when the failure was an HTTP status) and ``kind`` (an
    ``LLMErrorKind``). Both default to ``None``; neither affects the text form.
    """

    def __init__(
        self,
        message: str,
        description: str | None = None,
        *,
        status: int | None = None,
        kind: LLMErrorKind | None = None,
    ) -> None:
        """Build the error.

        Args:
            message: Human-readable summary; also the exception's sole ``args`` entry.
            description: Optional detail appended to the text form as ``message: description``.
            status: Keyword-only HTTP status code of the failed response, or ``None``.
            kind: Keyword-only failure kind (``LLMErrorKind``), or ``None``.
        """
        super().__init__(message)
        self.message = message
        self.description = description
        self.status = status
        self.kind = kind

    def __str__(self) -> str:
        """Return ``message``, or ``message: description`` when a description is set."""
        if self.description:
            return f"{self.message}: {self.description}"
        return self.message


class LLMSettingIgnoredWarning(UserWarning):
    """Warning category for a setting the library accepted but could not apply (DEC-003-5, DEC-arch-6).

    A direct ``UserWarning`` subclass with no behaviour of its own; it exists so an app can filter
    or escalate the library's ignored-setting warnings by category (``warnings.filterwarnings(...,
    category=LLMSettingIgnoredWarning)``). Emitted by the Responses clients, never raised as an error.
    """


class LLMToolObserverWarning(UserWarning):
    """Warning category emitted when the tool-event observer fails in tolerant mode (DEC-006-3).

    A direct ``UserWarning`` subclass with no behaviour of its own, distinct from
    ``LLMSettingIgnoredWarning`` so an app can filter or escalate observer failures on their own
    (``warnings.filterwarnings(..., category=LLMToolObserverWarning)``). Never raised as an error.
    """
