"""LLM client module for handling interactions with different language model APIs."""

__version__ = "0.3.0"

from .exceptions import LLMError, LLMSettingIgnoredWarning
from .llm_client import LLMClient
from .ollama import OllamaAPIClient
from .openai import OpenAIAPIClient
from .llama_swap import LlamaSwapAPIClient
from .openai_chat_completions import OpenAIChatCompletionsClient
from .llama_swap_chat_completions import LlamaSwapChatCompletionsClient
from .llm import get_llm_client, substitute_placeholders
from .tools import pydantic_to_openai_tool
from .trace import LLMTracer

__all__ = [
    "__version__",
    "LLMError",
    "LLMSettingIgnoredWarning",
    "LLMClient",
    "OllamaAPIClient",
    "OpenAIAPIClient",
    "LlamaSwapAPIClient",
    "OpenAIChatCompletionsClient",
    "LlamaSwapChatCompletionsClient",
    "get_llm_client",
    "substitute_placeholders",
    "pydantic_to_openai_tool",
    "LLMTracer",
]
