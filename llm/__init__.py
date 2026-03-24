"""LLM client module for handling interactions with different language model APIs."""

__version__ = "0.1.4"

from .exceptions import LLMError
from .llm_client import LLMClient
from .ollama import OllamaAPIClient
from .openai import OpenAIAPIClient
from .llama_swap import LlamaSwapAPIClient
from .llm import get_llm_client, substitute_placeholders
from .tools import pydantic_to_openai_tool

__all__ = [
    "__version__",
    "LLMError",
    "LLMClient",
    "OllamaAPIClient",
    "OpenAIAPIClient",
    "LlamaSwapAPIClient",
    "get_llm_client",
    "substitute_placeholders",
    "pydantic_to_openai_tool",
]
