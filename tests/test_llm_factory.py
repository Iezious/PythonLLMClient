"""Factory tests for feature 001.core-fixes, step 001 (package surface).

``get_llm_client`` forwards the app's credential to the OpenAI client (DoD-4), still forwards it
to llama-swap and builds llama-swap without one, and refuses an unknown server type (DoD-5).
Every client is built with explicit arguments and no process environment variable is set
anywhere in this file (DoD-6); the last test checks that property on this file's own source.
"""

import ast
from pathlib import Path

import pytest

from llm.llama_swap import LlamaSwapAPIClient
from llm.llm import get_llm_client
from llm.openai import OpenAIAPIClient

# A distinctive credential so the assertion cannot be satisfied by any value the
# environment of the test run might happen to hold.
_OPENAI_TOKEN = "sk-factory-credential-001"
_OPENAI_URL = "https://api.openai.com/v1"
_LLAMASWAP_URL = "http://localhost:8080/v1"


def test_openai_factory_forwards_bearer_token__DoD4():
    """DoD-4: the ``openai`` kind returns an OpenAI client whose credential is the caller's token."""
    client = get_llm_client("openai", _OPENAI_URL, "gpt-4o-mini", bearer_token=_OPENAI_TOKEN)
    assert isinstance(client, OpenAIAPIClient)
    assert client.model == "gpt-4o-mini"
    assert client.bearer_token == _OPENAI_TOKEN


def test_llamaswap_factory_forwards_bearer_token__DoD5():
    """DoD-5: the ``llamaswap`` kind still forwards the caller's credential."""
    client = get_llm_client("llamaswap", _LLAMASWAP_URL, "mistral", bearer_token="some-token")
    assert isinstance(client, LlamaSwapAPIClient)
    assert client.model == "mistral"
    assert client.bearer_token == "some-token"


def test_llamaswap_factory_builds_without_credential__DoD5():
    """DoD-5: the ``llamaswap`` kind still builds a client when no credential is given."""
    client = get_llm_client("llamaswap", _LLAMASWAP_URL, "mistral")
    assert isinstance(client, LlamaSwapAPIClient)
    assert client.model == "mistral"
    assert client.base_url == _LLAMASWAP_URL
    assert client.bearer_token is None


def test_factory_refuses_unknown_server_type__DoD5():
    """DoD-5: an unknown server type raises ``ValueError`` whose message names that type."""
    with pytest.raises(ValueError) as exc_info:
        get_llm_client("no-such-backend", _LLAMASWAP_URL, "mistral", bearer_token="some-token")
    assert "no-such-backend" in str(exc_info.value)


def _factory_calls(tree: ast.AST) -> list[ast.Call]:
    """Return every ``get_llm_client(...)`` call expression in ``tree``."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "get_llm_client"
    ]


def test_factory_tests_set_no_environment_and_pass_explicit_arguments__DoD6():
    """DoD-6: this file sets no process environment variable and builds every client explicitly.

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
