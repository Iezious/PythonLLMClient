"""Custom exceptions for the LLM client module."""

class LLMError(Exception):
    """Raised for any non-2xx HTTP response from the LLM server or other client-side errors."""
    def __init__(self, message: str, description: str | None = None):
        super().__init__(message)
        self.message = message
        self.description = description

    def __str__(self):
        if self.description:
            return f"{self.message}: {self.description}"
        return self.message


class LLMSettingIgnoredWarning(UserWarning):
    """Warning category for a setting the library accepted but could not apply (DEC-003-5, DEC-arch-6).

    A direct ``UserWarning`` subclass with no behaviour of its own; it exists so an app can filter
    or escalate the library's ignored-setting warnings by category (``warnings.filterwarnings(...,
    category=LLMSettingIgnoredWarning)``). Emitted by the Responses clients, never raised as an error.
    """
