"""Live tests: streamed chat delivers non-English text complete on llama-swap and OpenAI.

Covers plan 001.core-fixes, step 003 (stream parsing), DoD-15 and DoD-16: every character of a
reply in a non-ASCII script reaches the receiver, and the returned text equals the pieces
received. Only robust properties are asserted (non-empty, returned == joined pieces, the
expected script present, no replacement character), never exact model text.
"""

import os
import re

import pytest

from llm.llama_swap_chat_completions import LlamaSwapChatCompletionsClient
from llm.openai_chat_completions import OpenAIChatCompletionsClient

# A prompt whose answer must be written in Cyrillic and Japanese, so multi-byte characters
# (2- and 3-byte UTF-8) are certain to cross network read boundaries somewhere in the stream.
_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Write two short sentences about the sea: the first one in Russian (Cyrillic letters), "
            "the second one in Japanese. Do not add any English."
        ),
    }
]

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_JAPANESE = re.compile(r"[぀-ヿ一-鿿]")


def _assert_complete(response: str, pieces: list[str]) -> None:
    """Shared robust checks: returned text equals the received pieces and both scripts survive intact."""
    streamed = "".join(pieces)
    assert isinstance(response, str)
    assert response, "streamed reply must not be empty"
    assert response == streamed, "returned text must equal the pieces the receiver got"
    # A character torn between reads would surface as U+FFFD (DEC-001-10, errors="replace").
    assert "�" not in streamed
    assert _CYRILLIC.search(streamed), "expected Cyrillic characters in the reply"
    assert _JAPANESE.search(streamed), "expected Japanese characters in the reply"


@pytest.mark.asyncio
async def test_llamaswap_stream_non_english_complete__DoD15():
    """llama-swap: a streamed non-English reply reaches the receiver whole and equals the returned text (DoD-15)."""
    base_url = (
        os.environ.get("LLAMA_SWITCH_URL") or os.environ.get("LLAMA_SWAP_BASE_URL") or "http://india.loc:9292/v1"
    )
    model = os.environ.get("LLAMA_SWAP_MODEL") or "Qwen3.5-27B-heretic-Q5_K_M"
    client = LlamaSwapChatCompletionsClient(model=model, base_url=base_url)
    pieces: list[str] = []

    async def on_delta(delta: str) -> None:
        pieces.append(delta)

    async with client:
        response = await client.chat(list(_MESSAGES), stream=True, on_delta=on_delta)

    _assert_complete(response, pieces)


@pytest.mark.asyncio
async def test_openai_stream_non_english_complete__DoD16():
    """OpenAI: a streamed non-English reply reaches the receiver whole and equals the returned text (DoD-16)."""
    # The test (the app side) reads its own key and passes it as the credential; skip when absent.
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_AI_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY / OPEN_AI_KEY not set")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAIChatCompletionsClient(model=model, bearer_token=api_key)
    pieces: list[str] = []

    async def on_delta(delta: str) -> None:
        pieces.append(delta)

    async with client:
        response = await client.chat(list(_MESSAGES), stream=True, on_delta=on_delta)

    _assert_complete(response, pieces)
