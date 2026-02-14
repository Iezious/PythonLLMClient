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
