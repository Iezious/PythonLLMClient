from typing import TypedDict


class LLMMessage(TypedDict):
    """
    Represents a single message in a conversation with the LLM.
    """
    role: str
    content: str
