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
    """Fill ``{{NAME}}`` slots in ``text`` from ``data``, pass after pass, until nothing more can be filled.

    Each pass finds the ``{{NAME}}`` slots in the current text and replaces every slot whose name is in
    ``data`` (membership test, then subscription). A value that itself contains a known slot is filled on a
    later pass. Filling stops as soon as one of these holds (DEC-002-15):

    - no slot is found in the text;
    - a whole pass leaves the text unchanged (only unknown slots remain, or a slot refers to itself);
    - 10 passes have run (the cap ends silently, without an error).

    Args:
        text: Template containing ``{{NAME}}`` slots.
        data: Mapping of slot names to values. A callable value is called with no arguments, once per
            occurrence per pass, and only when its slot appears; any other value is inserted as ``str(value)``.

    Returns:
        The filled text. Unknown slots and self-referring slots are returned as written, without an error.

    Raises:
        Any exception raised by a callable value passes through unchanged.
    """
    import re

    def _fill(match: "re.Match[str]") -> str:
        """Return the text for one slot occurrence: its value as text, or the slot as written if unknown."""
        key = match.group(1)
        # Membership test, then subscription (Step 005 context): a mapping that counts lookups observes
        # exactly one test per slot occurrence per pass.
        if key in data:
            value = data[key]
            return str(value() if callable(value) else value)
        return match.group(0)

    out = text
    # The 10-pass cap stays as the guard against values that keep growing (DEC-002-15).
    for _ in range(10):
        if not re.search(r"{{(\w+)}}", out):
            break
        # One substitution over the pass-start text: only slots present when the pass begins are filled,
        # so a slot introduced by a value waits for the next pass (one nesting level per pass).
        filled = re.sub(r"{{(\w+)}}", _fill, out)
        # A pass that changes nothing means only unknown or self-referring slots remain; another pass
        # would repeat the same lookups and computations for nothing (DEC-002-15).
        if filled == out:
            break
        out = filled
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
