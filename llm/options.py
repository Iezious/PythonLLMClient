from typing import Optional, Dict, List, Union
from typing_extensions import TypedDict

class BaseLLMOptions(TypedDict, total=False):
    """Base options for LLM clients."""
    temperature: Optional[float]
    top_p: Optional[float]
    seed: Optional[int]

class OllamaOptions(BaseLLMOptions, total=False):
    """
    Options specific to Ollama.
    Supported keys mirror the Ollama API options.
    """
    top_k: Optional[int]
    min_p: Optional[float]
    repeat_penalty: Optional[float]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    mirostat: Optional[int]
    mirostat_tau: Optional[float]
    mirostat_eta: Optional[float]
    num_predict: Optional[int]
    num_ctx: Optional[int]
    num_keep: Optional[int]
    stop: Optional[list[str]]
    tfs_z: Optional[float]
    typical_p: Optional[float]
    repeat_last_n: Optional[int]
    penalize_newline: Optional[bool]
    num_thread: Optional[int]

class OpenAIOptions(BaseLLMOptions, total=False):
    """
    Options specific to OpenAI and OpenAI-compatible endpoints (like llama-swap).
    """
    max_tokens: Optional[int]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    logit_bias: Optional[Dict[str, float]]
    user: Optional[str]
    stop: Optional[Union[list[str], str]]
    response_format: Optional[Dict[str, str]]
