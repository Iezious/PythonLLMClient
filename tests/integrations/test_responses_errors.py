"""Live failure tests on the Responses classes (feature 005.responses-live-tests, step 003, DoD-1..4).

A bad OpenAI credential (non-streamed and streamed) must end with ``LLMError`` carrying the refusal status
and the server's reply but never the credential (DEC-000-11, DEC-arch-5). A closed local port must end
with ``LLMError`` naming the address (DEC-001-7). Inputs come from DEC-005-8. None of these tests skips:
the dummy key is passed as the builder's credential override, so the environment key reader (and its
skip) is never used (DEC-005-5). The bad-credential tests need outbound HTTPS to OpenAI's address only.
"""

import pytest

from llm import LLMError
from llm.message import LLMMessage
from tests.integrations.live_support import PieceRecorder, assert_secret_free_error, build_client

# ---------------------------------------------------------------------------
# Module-local inputs (DEC-005-8)
# ---------------------------------------------------------------------------

# A fixed, obviously fake key: no real key material can leak into test output (DEC-005-8 rejects a
# corrupted real key for that reason).
_DUMMY_KEY = "sk-invalid-005-live-test-0000000000000000"

# A closed local port: refuses at once, needs no remote server (DEC-005-8).
_CLOSED_PORT_URL = "http://127.0.0.1:1/v1"
# What the error text must name (DEC-005-8).
_CLOSED_PORT_ADDRESS = "127.0.0.1:1"
# Bounds a filtered port so the run cannot stall (DEC-005-8).
_CLOSED_PORT_TIMEOUT = 10.0

_REFUSAL_STATUS = "401"


def _messages() -> list[LLMMessage]:
    """A minimal one-turn conversation; its content is irrelevant because the call must fail."""
    return [{"role": "user", "content": "Say hello."}]


# ---------------------------------------------------------------------------
# Shared bodies
# ---------------------------------------------------------------------------


async def _bad_credential(stream: bool) -> None:
    """DoD-1 / DoD-2 body: OpenAI ``chat`` with the dummy key ends with a secret-free ``LLMError``.

    The error text holds the refusal status ``401``, the description (the server's reply) is non-empty,
    and the key appears in none of the error's surfaces.
    """
    client = build_client("openai", bearer_token=_DUMMY_KEY)
    recorder = PieceRecorder() if stream else None
    async with client:
        with pytest.raises(LLMError) as excinfo:
            await client.chat(_messages(), stream=stream, on_delta=recorder)

    err = excinfo.value
    # Checked before the status assertion so a failure message never echoes the key from str(err).
    assert_secret_free_error(err, _DUMMY_KEY)
    assert _REFUSAL_STATUS in str(err), f"expected the refusal status {_REFUSAL_STATUS} in the error: {err}"
    assert err.description, "expected the server's reply as the error's description"
    assert str(err.description).strip(), "expected a non-blank server reply as the error's description"


async def _unreachable(backend: str, stream: bool) -> None:
    """DoD-3 / DoD-4 body: ``chat`` against the closed local port ends with ``LLMError`` naming the address.

    The OpenAI client gets the dummy key so no environment key is read and the test never skips.
    """
    bearer_token = _DUMMY_KEY if backend == "openai" else None
    client = build_client(backend, base_url=_CLOSED_PORT_URL, timeout=_CLOSED_PORT_TIMEOUT, bearer_token=bearer_token)
    recorder = PieceRecorder() if stream else None
    async with client:
        with pytest.raises(LLMError) as excinfo:
            await client.chat(_messages(), stream=stream, on_delta=recorder)

    assert _CLOSED_PORT_ADDRESS in str(excinfo.value), (
        f"expected the address {_CLOSED_PORT_ADDRESS} in the error: {excinfo.value}"
    )


# ---------------------------------------------------------------------------
# DoD-1 / DoD-2 — bad OpenAI credential
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_bad_credential_not_streamed__DoD1():
    """DoD-1: non-streamed OpenAI ``chat`` with the dummy key -> ``LLMError`` with 401, reply, no key."""
    await _bad_credential(stream=False)


@pytest.mark.asyncio
async def test_openai_bad_credential_streamed__DoD2():
    """DoD-2: streamed OpenAI ``chat`` with the dummy key and a recorder -> ``LLMError`` with 401, reply, no key."""
    await _bad_credential(stream=True)


# ---------------------------------------------------------------------------
# DoD-3 — unreachable server, non-streamed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_unreachable_server_not_streamed__DoD3():
    """DoD-3: non-streamed OpenAI ``chat`` at the closed port -> ``LLMError`` naming ``127.0.0.1:1``."""
    await _unreachable("openai", stream=False)


@pytest.mark.asyncio
async def test_llamaswap_unreachable_server_not_streamed__DoD3():
    """DoD-3: non-streamed llama-swap ``chat`` at the closed port -> ``LLMError`` naming ``127.0.0.1:1``."""
    await _unreachable("llamaswap", stream=False)


# ---------------------------------------------------------------------------
# DoD-4 — unreachable server, streamed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_unreachable_server_streamed__DoD4():
    """DoD-4: streamed OpenAI ``chat`` at the closed port -> ``LLMError`` naming ``127.0.0.1:1``."""
    await _unreachable("openai", stream=True)


@pytest.mark.asyncio
async def test_llamaswap_unreachable_server_streamed__DoD4():
    """DoD-4: streamed llama-swap ``chat`` at the closed port -> ``LLMError`` naming ``127.0.0.1:1``."""
    await _unreachable("llamaswap", stream=True)
