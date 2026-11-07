"""Client factory, prompt placeholder helpers and the retired CLI smoke test.

Builds the right backend client from a server kind (``get_llm_client``) and fills prompt templates.
"""
# ---------------------------------------------------------------------------
# Client Factory
# ---------------------------------------------------------------------------
import argparse
from typing import Optional, Dict, Any

from llm.llm_client import LLMClient
from llm.ollama import OllamaAPIClient
from llm.openai import OpenAIAPIClient
from llm.llama_swap import LlamaSwapAPIClient

__all__ = ["OllamaAPIClient", "OpenAIAPIClient", "LlamaSwapAPIClient", "get_llm_client", "substitute_placeholders", "recursive_format_placeholders"]


def get_llm_client(
    server_type: str,
    base_url: str,
    model: str,
    timeout: float = 600.0,
    bearer_token: Optional[str] = None,
    default_options: Optional[Dict[str, Any]] = None,
) -> LLMClient:
    """Build the client for ``server_type``, forwarding every argument to its constructor.

    Args:
        server_type: Exact backend kind: ``"ollama"``, ``"openai"`` or ``"llamaswap"``.
        base_url: API root of the server.
        model: Model name sent with every request.
        timeout: Total per-request timeout in seconds.
        bearer_token: The app's credential, forwarded unchanged to every backend (``None`` builds
            a client without one).
        default_options: Client-level settings, merged under per-call settings.

    Returns:
        The constructed client; no network call is made.

    Raises:
        ValueError: ``server_type`` is not a supported kind; the message names it.
    """
    if server_type == "ollama":
        return OllamaAPIClient(model, base_url, timeout, bearer_token, default_options)
    elif server_type == "openai":
        # The app owns the credential; the factory forwards it like every other branch and the
        # library never reads it from the environment (DEC-001-5).
        return OpenAIAPIClient(model, base_url, timeout, bearer_token, default_options)
    elif server_type == "llamaswap":
        return LlamaSwapAPIClient(model, base_url, timeout, bearer_token, default_options)
    else:
        raise ValueError(f"Unsupported server type: {server_type}")


def substitute_placeholders(template: str, placeholders: Dict[str, str]) -> str:
    """
    Substitute {{PLACEHOLDER}} patterns in template with values from placeholders dict.

    Args:
        template: String containing {{PLACEHOLDER}} patterns
        placeholders: Dict mapping placeholder names to replacement values

    Returns:
        Template with placeholders replaced by values

    Example:
        template = "Hello {{NAME}}, you have {{COUNT}} messages"
        placeholders = {"NAME": "Alice", "COUNT": "5"}
        result = substitute_placeholders(template, placeholders)
        # Returns: "Hello Alice, you have 5 messages"
    """
    result = template
    for key, value in placeholders.items():
        placeholder_pattern = f"{{{{{key}}}}}"
        result = result.replace(placeholder_pattern, str(value))
    return result


def recursive_format_placeholders(text: str, data: Dict[str, str | bytes | int | float | None | object]) -> str:
    """Recursively formats a string with placeholders until no placeholders are left.

    - Supports up to 10 passes to avoid runaway loops.
    - Lazily evaluates callables in the mapping.
    - Coerces non-string values via str().
    """
    import re
    # Safer, iterative replacement to avoid runaway recursion
    out = text
    for _ in range(10):
        found = re.findall(r"{{(\w+)}}", out)
        if not found:
            break
        for key in found:
            if key in data:
                val = data[key]() if callable(data[key]) else data[key]
                out = out.replace(f"{{{{{key}}}}}", str(val))
    return out


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def _cli() -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="Quick Ollama smoke test")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--temperature", type=float)
    p.add_argument("--top-p", type=float, dest="top_p")
    p.add_argument("--top-k", type=int, dest="top_k")
    p.add_argument("--max-tokens", type=int, dest="max_tokens")
    args = p.parse_args()

    kw = {k: v for k, v in vars(args).items() if v is not None}
    client = OllamaAPIClient(args.model) # Default to Ollama for this CLI
    print(client.generate(args.prompt, **kw))

if __name__ == "__main__":
    _cli()
