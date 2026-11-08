"""Factory tests for ``get_llm_client``.

Feature 001 (DEC-001-5): the factory forwards the app's credential to the OpenAI client
(DoD-4), still forwards it to llama-swap and builds llama-swap without one, and
refuses an unknown server type (DoD-5, DEC-000-3); those tests now build the chat-completions classes
(DEC-arch-12).
Feature 003 (DEC-003-8, DEC-arch-4): the ``protocol`` argument picks the Responses or the
chat-completions class (DoD-3, DoD-4), never depending on the address (DoD-5); unknown protocol
values (DoD-6) and any protocol on Ollama (DoD-7) are refused; the earlier refusals still hold
(DoD-8).
Every client is built with explicit arguments and no process environment variable is set
anywhere in this file (001.core-fixes DoD-6); the last test checks that property on this file's
own source, so every call here passes kind, base URL and model positionally and every ``openai``
call passes ``bearer_token=`` by keyword.
"""

import ast
from pathlib import Path

import pytest

from llm.llama_swap import LlamaSwapAPIClient
from llm.llama_swap_chat_completions import LlamaSwapChatCompletionsClient
from llm.llm import get_llm_client
from llm.ollama import OllamaAPIClient
from llm.openai import OpenAIAPIClient
from llm.openai_chat_completions import OpenAIChatCompletionsClient

# A distinctive credential so the assertion cannot be satisfied by any value the
# environment of the test run might happen to hold.
_OPENAI_TOKEN = "sk-factory-credential-001"
_OPENAI_URL = "https://api.openai.com/v1"
_LLAMASWAP_URL = "http://localhost:8080/v1"
_OLLAMA_URL = "http://localhost:11434"

# Values the spec (DEC-003-8) says must be refused: exact matching, no case folding or trimming.
_UNKNOWN_PROTOCOLS = ["Responses", "CHAT_COMPLETIONS", "chat-completions", "completions", " responses"]

# Every protocol choice the spec accepts, including the explicit ``None`` that means "not passed".
_ACCEPTED_PROTOCOLS = [None, "responses", "chat_completions"]


def test_openai_factory_forwards_bearer_token__DoD4():
    """001.core-fixes DoD-4: the ``openai`` kind returns an OpenAI client whose credential is the caller's token."""
    client = get_llm_client(
        "openai", _OPENAI_URL, "gpt-4o-mini", bearer_token=_OPENAI_TOKEN, protocol="chat_completions"
    )
    assert isinstance(client, OpenAIChatCompletionsClient)
    assert client.model == "gpt-4o-mini"
    assert client.bearer_token == _OPENAI_TOKEN


def test_llamaswap_factory_forwards_bearer_token__DoD5():
    """001.core-fixes DoD-5: the ``llamaswap`` kind still forwards the caller's credential."""
    client = get_llm_client(
        "llamaswap", _LLAMASWAP_URL, "mistral", bearer_token="some-token", protocol="chat_completions"
    )
    assert isinstance(client, LlamaSwapChatCompletionsClient)
    assert client.model == "mistral"
    assert client.bearer_token == "some-token"


def test_llamaswap_factory_builds_without_credential__DoD5():
    """001.core-fixes DoD-5: the ``llamaswap`` kind still builds a client when no credential is given."""
    client = get_llm_client("llamaswap", _LLAMASWAP_URL, "mistral", protocol="chat_completions")
    assert isinstance(client, LlamaSwapChatCompletionsClient)
    assert client.model == "mistral"
    assert client.base_url == _LLAMASWAP_URL
    assert client.bearer_token is None


def test_factory_refuses_unknown_server_type__DoD5():
    """001.core-fixes DoD-5: an unknown server type raises ``ValueError`` whose message names that type."""
    with pytest.raises(ValueError) as exc_info:
        get_llm_client("no-such-backend", _LLAMASWAP_URL, "mistral", bearer_token="some-token")
    assert "no-such-backend" in str(exc_info.value)


# --- Feature 003: the protocol argument (DEC-003-8, DEC-arch-4) --------------------------------


def test_openai_factory_without_protocol_returns_responses_class__DoD3():
    """DoD-3: ``openai`` with no protocol returns ``OpenAIAPIClient`` (US-056.AC-1)."""
    client = get_llm_client("openai", _OPENAI_URL, "gpt-4o-mini", bearer_token=_OPENAI_TOKEN)
    assert type(client) is OpenAIAPIClient


def test_openai_factory_with_none_protocol_returns_responses_class__DoD3():
    """DoD-3: ``protocol=None`` passed explicitly is the same as not passing it (DEC-003-8)."""
    client = get_llm_client("openai", _OPENAI_URL, "gpt-4o-mini", bearer_token=_OPENAI_TOKEN, protocol=None)
    assert type(client) is OpenAIAPIClient


def test_openai_factory_with_responses_protocol_returns_responses_class__DoD3():
    """DoD-3: ``openai`` with ``protocol="responses"`` returns ``OpenAIAPIClient``."""
    client = get_llm_client(
        "openai", _OPENAI_URL, "gpt-4o-mini", bearer_token=_OPENAI_TOKEN, protocol="responses"
    )
    assert type(client) is OpenAIAPIClient


def test_llamaswap_factory_without_protocol_returns_responses_class__DoD3():
    """DoD-3: ``llamaswap`` with no protocol returns ``LlamaSwapAPIClient`` (US-056.AC-1)."""
    client = get_llm_client("llamaswap", _LLAMASWAP_URL, "mistral")
    assert type(client) is LlamaSwapAPIClient


def test_llamaswap_factory_with_none_protocol_returns_responses_class__DoD3():
    """DoD-3: ``protocol=None`` passed explicitly is the same as not passing it (DEC-003-8)."""
    client = get_llm_client("llamaswap", _LLAMASWAP_URL, "mistral", protocol=None)
    assert type(client) is LlamaSwapAPIClient


def test_llamaswap_factory_with_responses_protocol_returns_responses_class__DoD3():
    """DoD-3: ``llamaswap`` with ``protocol="responses"`` returns ``LlamaSwapAPIClient``."""
    client = get_llm_client("llamaswap", _LLAMASWAP_URL, "mistral", protocol="responses")
    assert type(client) is LlamaSwapAPIClient


def test_openai_factory_chat_completions_carries_all_arguments__DoD4():
    """DoD-4: ``protocol="chat_completions"`` gives ``OpenAIChatCompletionsClient`` with every given value."""
    defaults = {"temperature": 0.25, "max_tokens": 64}
    client = get_llm_client(
        "openai",
        _OPENAI_URL,
        "gpt-4o-mini",
        42.5,
        bearer_token=_OPENAI_TOKEN,
        default_options=defaults,
        protocol="chat_completions",
    )
    assert type(client) is OpenAIChatCompletionsClient
    assert client.model == "gpt-4o-mini"
    assert client.base_url == _OPENAI_URL
    assert client.timeout == 42.5
    assert client.bearer_token == _OPENAI_TOKEN
    assert client.default_options == defaults


def test_llamaswap_factory_chat_completions_carries_all_arguments__DoD4():
    """DoD-4: ``protocol="chat_completions"`` gives ``LlamaSwapChatCompletionsClient`` with every given value."""
    defaults = {"temperature": 0.75, "top_p": 0.5}
    client = get_llm_client(
        "llamaswap",
        _LLAMASWAP_URL,
        "mistral",
        17.0,
        bearer_token="swap-token",
        default_options=defaults,
        protocol="chat_completions",
    )
    assert type(client) is LlamaSwapChatCompletionsClient
    assert client.model == "mistral"
    assert client.base_url == _LLAMASWAP_URL
    assert client.timeout == 17.0
    assert client.bearer_token == "swap-token"
    assert client.default_options == defaults


@pytest.mark.parametrize("protocol", _ACCEPTED_PROTOCOLS)
def test_openai_factory_class_does_not_depend_on_address__DoD5(protocol):
    """DoD-5: the same protocol choice on an OpenAI-looking and a local address gives the same class."""
    remote = get_llm_client("openai", _OPENAI_URL, "gpt-4o-mini", bearer_token=_OPENAI_TOKEN, protocol=protocol)
    local = get_llm_client("openai", _LLAMASWAP_URL, "gpt-4o-mini", bearer_token=_OPENAI_TOKEN, protocol=protocol)
    assert type(remote) is type(local)
    expected = OpenAIChatCompletionsClient if protocol == "chat_completions" else OpenAIAPIClient
    assert type(local) is expected


@pytest.mark.parametrize("protocol", _ACCEPTED_PROTOCOLS)
def test_llamaswap_factory_class_does_not_depend_on_address__DoD5(protocol):
    """DoD-5: the same protocol choice on a local and an OpenAI-looking address gives the same class."""
    local = get_llm_client("llamaswap", _LLAMASWAP_URL, "mistral", protocol=protocol)
    remote = get_llm_client("llamaswap", _OPENAI_URL, "mistral", protocol=protocol)
    assert type(remote) is type(local)
    expected = LlamaSwapChatCompletionsClient if protocol == "chat_completions" else LlamaSwapAPIClient
    assert type(remote) is expected


@pytest.mark.parametrize("protocol", _UNKNOWN_PROTOCOLS)
def test_openai_factory_refuses_unknown_protocol__DoD6(protocol):
    """DoD-6: an unknown protocol value (different case included) raises ``ValueError`` naming it."""
    with pytest.raises(ValueError) as exc_info:
        get_llm_client("openai", _OPENAI_URL, "gpt-4o-mini", bearer_token=_OPENAI_TOKEN, protocol=protocol)
    assert protocol in str(exc_info.value)


@pytest.mark.parametrize("protocol", _UNKNOWN_PROTOCOLS)
def test_llamaswap_factory_refuses_unknown_protocol__DoD6(protocol):
    """DoD-6: an unknown protocol value (different case included) raises ``ValueError`` naming it."""
    with pytest.raises(ValueError) as exc_info:
        get_llm_client("llamaswap", _LLAMASWAP_URL, "mistral", protocol=protocol)
    assert protocol in str(exc_info.value)


@pytest.mark.parametrize("protocol", ["responses", "chat_completions", "Responses", "bogus"])
def test_ollama_factory_refuses_any_protocol__DoD7(protocol):
    """DoD-7: Ollama with any protocol raises ``ValueError`` saying the choice is not offered on Ollama."""
    with pytest.raises(ValueError) as exc_info:
        get_llm_client("ollama", _OLLAMA_URL, "llama3", protocol=protocol)
    message = str(exc_info.value).lower()
    assert "protocol" in message
    assert "ollama" in message


def test_ollama_factory_without_protocol_returns_ollama_client__DoD7():
    """DoD-7: Ollama without a protocol still returns the Ollama client."""
    client = get_llm_client("ollama", _OLLAMA_URL, "llama3")
    assert isinstance(client, OllamaAPIClient)
    assert client.model == "llama3"


def test_ollama_factory_with_none_protocol_returns_ollama_client__DoD7():
    """DoD-7: an explicit ``protocol=None`` counts as no protocol, so Ollama is still built (DEC-003-8)."""
    client = get_llm_client("ollama", _OLLAMA_URL, "llama3", protocol=None)
    assert isinstance(client, OllamaAPIClient)


@pytest.mark.parametrize("protocol", [*_ACCEPTED_PROTOCOLS, "Bogus"])
def test_factory_refuses_unknown_server_type_whatever_the_protocol__DoD8(protocol):
    """DoD-8: an unknown server type is refused with ``ValueError`` naming it, before the protocol is looked at."""
    with pytest.raises(ValueError) as exc_info:
        get_llm_client("no-such-backend", _LLAMASWAP_URL, "mistral", bearer_token="some-token", protocol=protocol)
    assert "no-such-backend" in str(exc_info.value)


@pytest.mark.parametrize("protocol", _ACCEPTED_PROTOCOLS)
@pytest.mark.parametrize("token", [None, ""])
def test_openai_factory_refuses_missing_credential__DoD8(protocol, token):
    """DoD-8: an OpenAI client without a credential is refused with ``ValueError`` on either protocol."""
    with pytest.raises(ValueError):
        get_llm_client("openai", _OPENAI_URL, "gpt-4o-mini", bearer_token=token, protocol=protocol)


@pytest.mark.parametrize("protocol", _ACCEPTED_PROTOCOLS)
def test_llamaswap_factory_builds_with_and_without_credential__DoD8(protocol):
    """DoD-8: llama-swap is built with or without a credential on either protocol."""
    with_token = get_llm_client("llamaswap", _LLAMASWAP_URL, "mistral", bearer_token="swap-token", protocol=protocol)
    without_token = get_llm_client("llamaswap", _LLAMASWAP_URL, "mistral", protocol=protocol)
    assert with_token.bearer_token == "swap-token"
    assert without_token.bearer_token is None
    expected = LlamaSwapChatCompletionsClient if protocol == "chat_completions" else LlamaSwapAPIClient
    assert type(with_token) is expected
    assert type(without_token) is expected


def _factory_calls(tree: ast.AST) -> list[ast.Call]:
    """Return every ``get_llm_client(...)`` call expression in ``tree``."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "get_llm_client"
    ]


def test_factory_tests_set_no_environment_and_pass_explicit_arguments__DoD6():
    """001.core-fixes DoD-6: this file sets no process environment variable and builds every client explicitly.

    Checked on the parsed source (names and attributes, not string literals): no ``os`` import,
    no ``environ`` / ``putenv`` / ``setenv`` use; every factory call passes kind, base URL and
    model, and every ``openai`` build passes ``bearer_token`` explicitly.
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name != "os" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "os"
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"environ", "putenv", "setenv", "unsetenv"}
        if isinstance(node, ast.Name):
            assert node.id not in {"environ", "putenv", "setenv"}

    calls = _factory_calls(tree)
    assert calls, "expected factory calls in this file"
    for call in calls:
        keywords = {kw.arg for kw in call.keywords}
        positional = len(call.args)
        # kind, base_url, model: positionally or by keyword.
        explicit = positional + len(keywords & {"server_type", "base_url", "model"})
        assert explicit >= 3, f"factory call at line {call.lineno} does not pass kind, base_url and model"
        kind = call.args[0] if call.args else None
        if isinstance(kind, ast.Constant) and kind.value == "openai":
            assert "bearer_token" in keywords, f"openai build at line {call.lineno} lacks bearer_token"
