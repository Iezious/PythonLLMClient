"""Typing contract of the package (feature 006, step 002; extended by step 009).

Covers docs/plans/006.tolerant-tool-loop/002.typing-foundation.md:

- DoD-1: ``mypy --strict llm`` from the repository root, with the project configuration, reports no errors.
- DoD-2: a sample application module, checked in strict mode, type-checks without errors while using the
  public surface named by interfaces.md § Exposed: typing contract (DEC-006-2).
- DoD-3: ``chat_with_tools`` on each of the four OpenAI-family classes accepts a tuple of mappings as
  ``messages`` and leaves the caller's sequence unchanged (non-streamed, ``_request`` faked).

Extended by docs/plans/006.tolerant-tool-loop/009.llamaswap-chat-completions-tool-loop.md:

- DoD-25: the sample also calls ``chat_with_tools`` with ``tool_errors="raise"`` and a typed ``on_tool_event``
  observer (plain function and coroutine function taking ``ToolEvent``) on a factory-built client and on each
  directly built OpenAI-family class, and imports ``ToolEvent``, ``ToolErrorKind``, ``LLMErrorKind``,
  ``LLMToolObserverWarning`` from ``llm``; ``mypy --strict llm`` still reports no errors (the DoD-1 test).

mypy runs in a subprocess (``sys.executable -m mypy``) with the repository root as working directory, so
``llm`` resolves from the source tree and ``pyproject.toml``'s ``[tool.mypy]`` table applies (DEC-006-1,
DEC-006-2). The sample module is written to ``tmp_path`` and passed by path.
"""

from __future__ import annotations

import copy
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from llm import (
    LlamaSwapAPIClient,
    LlamaSwapChatCompletionsClient,
    OpenAIAPIClient,
    OpenAIChatCompletionsClient,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Generous: a cold mypy cache on the whole package can take a while on slow machines.
_MYPY_TIMEOUT_SECONDS = 600


def _run_mypy(*args: str) -> subprocess.CompletedProcess[str]:
    """Run mypy with the repository root as working directory (DEC-006-2) and return the finished process."""
    return subprocess.run(
        [sys.executable, "-m", "mypy", *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=_MYPY_TIMEOUT_SECONDS,
        check=False,
    )


def _report(result: subprocess.CompletedProcess[str]) -> str:
    """Full mypy output, shown when an assertion fails."""
    return f"exit code {result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"


# ---------------------------------------------------------------------------------------------------------
# Sample application module (DoD-2, DEC-006-2).
#
# Built from an import block plus named sections so a later step (009: ``tool_errors`` and a typed
# ``on_tool_event`` observer on a factory-built client) can add imports to ``_SAMPLE_IMPORTS`` and append
# one more ``(name, source)`` pair to ``_SAMPLE_SECTIONS`` without touching the existing ones. Every
# section is a set of annotated functions; the sample is only type-checked, never run.
# ---------------------------------------------------------------------------------------------------------

_SAMPLE_IMPORTS: tuple[str, ...] = (
    "from typing import Any, assert_type",
    "",
    "from llm import (",
    "    LlamaSwapAPIClient,",
    "    LlamaSwapChatCompletionsClient,",
    "    LLMClient,",
    "    LLMError,",
    "    LLMErrorKind,",
    "    LLMToolObserverWarning,",
    "    OpenAIAPIClient,",
    "    OpenAIChatCompletionsClient,",
    "    ToolErrorKind,",
    "    ToolEvent,",
    "    get_llm_client,",
    ")",
    "from llm.message import LLMMessage",
    "from llm.options import OpenAIOptions",
)

_SAMPLE_SHARED = '''
MESSAGES: list[LLMMessage] = [{"role": "user", "content": "Fetch record r-1."}]

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fetch_record",
            "description": "Fetch one record by id.",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
            },
        },
    }
]


def fetch_record(record_id: str) -> str:
    """A tool callable as an application writes it."""
    return record_id
'''

_SAMPLE_SECTIONS: tuple[tuple[str, str], ...] = (
    (
        "usage scope returns the class's own type",
        '''
async def usage_scope_returns_own_class() -> None:
    """``async with`` on a directly built client yields that client class (Self)."""
    async with OpenAIAPIClient(model="gpt-4o-mini", bearer_token="sk-sample") as responses:
        assert_type(responses, OpenAIAPIClient)
        same_responses: OpenAIAPIClient = responses
        reply: str = await same_responses.chat(MESSAGES)
    async with LlamaSwapAPIClient(model="some-model") as swap:
        assert_type(swap, LlamaSwapAPIClient)
        same_swap: LlamaSwapAPIClient = swap
        reply = await same_swap.chat(MESSAGES)
    async with OpenAIChatCompletionsClient(model="gpt-4o-mini", bearer_token="sk-sample") as completions:
        assert_type(completions, OpenAIChatCompletionsClient)
        same_completions: OpenAIChatCompletionsClient = completions
        reply = await same_completions.chat(MESSAGES)
    async with LlamaSwapChatCompletionsClient(model="some-model") as swap_completions:
        assert_type(swap_completions, LlamaSwapChatCompletionsClient)
        same_swap_completions: LlamaSwapChatCompletionsClient = swap_completions
        reply = await same_swap_completions.chat(MESSAGES)
    print(reply)
''',
    ),
    (
        "OpenAI-shaped options and default_options on every OpenAI-family class",
        '''
SAMPLE_DEFAULTS: OpenAIOptions = {"temperature": 0.2, "max_tokens": 128}
SAMPLE_CALL_OPTIONS: OpenAIOptions = {"temperature": 0.7, "top_p": 0.9}


async def openai_shaped_options() -> None:
    """``default_options`` and call ``options`` take the OpenAI settings shape."""
    responses = OpenAIAPIClient(model="gpt-4o-mini", bearer_token="sk-sample", default_options=SAMPLE_DEFAULTS)
    swap = LlamaSwapAPIClient(model="some-model", default_options=SAMPLE_DEFAULTS)
    completions = OpenAIChatCompletionsClient(
        model="gpt-4o-mini", bearer_token="sk-sample", default_options={"temperature": 0.1}
    )
    swap_completions = LlamaSwapChatCompletionsClient(model="some-model", default_options={"max_tokens": 64})

    await responses.chat(MESSAGES, options=SAMPLE_CALL_OPTIONS)
    await responses.chat(MESSAGES, options={"temperature": 0.3, "max_tokens": 32})
    await responses.chat_json(MESSAGES, options=SAMPLE_CALL_OPTIONS)
    await responses.embed("text", options=SAMPLE_CALL_OPTIONS)
    await swap.chat(MESSAGES, options=SAMPLE_CALL_OPTIONS)
    await swap.embed_batch(["a", "b"], options={"temperature": 0.0})
    await completions.chat(MESSAGES, options=SAMPLE_CALL_OPTIONS)
    await completions.chat_json(MESSAGES, options={"top_p": 0.5})
    await swap_completions.chat(MESSAGES, options={"temperature": 0.4})
    await swap_completions.chat_with_tools(
        [{"role": "user", "content": "Fetch r-1."}],
        tools_definitions=TOOL_DEFINITIONS,
        tools={"fetch_record": fetch_record},
        options=SAMPLE_CALL_OPTIONS,
    )
''',
    ),
    (
        "factory result typed LLMClient[Any] and its operations",
        '''
async def factory_client_operations() -> None:
    """``get_llm_client`` returns ``LLMClient[Any]``; its operations are callable without type errors."""
    client = get_llm_client("llamaswap", "http://localhost:8080/v1", "some-model")
    assert_type(client, LLMClient[Any])
    typed: LLMClient[Any] = client
    async with typed as entered:
        assert_type(entered, LLMClient[Any])
        text: str = await entered.chat(MESSAGES, options={"temperature": 0.1})
        parsed = await entered.chat_json(MESSAGES)
        vector: list[float] = await entered.embed("text")
        vectors = await entered.embed_batch(["a", "b"])
        models = await entered.list_models()
        answer: str = await entered.chat_with_tools(
            MESSAGES,
            tools_definitions=TOOL_DEFINITIONS,
            tools={"fetch_record": fetch_record},
            options={"max_tokens": 256},
        )
    print(text, parsed, vector, vectors, models, answer)
''',
    ),
    (
        "chat_with_tools messages carrying tool-call and tool-result turns",
        '''
async def tool_chat_with_tool_turns() -> None:
    """``chat_with_tools`` accepts dict turns with ``tool_calls`` and ``tool_call_id`` on every class."""
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "Fetch record r-1."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "fetch_record", "arguments": '{"record_id": "r-1"}'},
                }
            ],
        },
        {"role": "tool", "content": "r-1", "tool_call_id": "call_1"},
    ]
    responses = OpenAIAPIClient(model="gpt-4o-mini", bearer_token="sk-sample")
    swap = LlamaSwapAPIClient(model="some-model")
    completions = OpenAIChatCompletionsClient(model="gpt-4o-mini", bearer_token="sk-sample")
    swap_completions = LlamaSwapChatCompletionsClient(model="some-model")
    factory_client = get_llm_client("openai", "https://api.openai.com/v1", "gpt-4o-mini", bearer_token="sk-sample")

    # An inline list literal, as an application typically writes it.
    await responses.chat_with_tools(
        [
            {"role": "user", "content": "Fetch record r-1."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "fetch_record", "arguments": '{"record_id": "r-1"}'},
                    }
                ],
            },
            {"role": "tool", "content": "r-1", "tool_call_id": "call_1"},
        ],
        tools_definitions=TOOL_DEFINITIONS,
        tools={"fetch_record": fetch_record},
    )
    await swap.chat_with_tools(history, tools_definitions=TOOL_DEFINITIONS, tools={"fetch_record": fetch_record})
    await completions.chat_with_tools(
        history, tools_definitions=TOOL_DEFINITIONS, tools={"fetch_record": fetch_record}
    )
    await swap_completions.chat_with_tools(
        history, tools_definitions=TOOL_DEFINITIONS, tools={"fetch_record": fetch_record}
    )
    await factory_client.chat_with_tools(
        history, tools_definitions=TOOL_DEFINITIONS, tools={"fetch_record": fetch_record}
    )
    # A read-only sequence (tuple) is accepted too.
    await responses.chat_with_tools(
        tuple(history), tools_definitions=TOOL_DEFINITIONS, tools={"fetch_record": fetch_record}
    )
''',
    ),
    (
        # Step 009, DoD-25 (DEC-006-2, DEC-006-10): strict handling and a typed observer on the factory client
        # and on each directly built OpenAI-family class; the 0.2.1 names imported from ``llm``.
        "strict tool handling and typed tool-event observers (step 009)",
        '''
OBSERVED: list[ToolEvent] = []


def observe_tool_event(event: ToolEvent) -> None:
    """A plain-function observer taking ``ToolEvent``."""
    kind: ToolErrorKind | None = event["error_kind"]
    call_id: str | None = event["call_id"]
    output: str | None = event["output"]
    arguments: str = event["arguments"]
    print(event["type"], event["name"], call_id, arguments, output, kind)
    OBSERVED.append(event)


async def observe_tool_event_async(event: ToolEvent) -> None:
    """A coroutine-function observer taking ``ToolEvent``."""
    OBSERVED.append(event)


def error_kind_of(error: LLMError) -> LLMErrorKind | None:
    """Read the error kind as an application does."""
    kind: LLMErrorKind | None = error.kind
    return kind


def observer_warning_category() -> type[UserWarning]:
    """The observer warning is a warning category an application can filter."""
    return LLMToolObserverWarning


async def strict_tools_with_typed_observers() -> None:
    """``tool_errors="raise"`` and a typed ``on_tool_event`` on the factory client and every OpenAI-family class."""
    factory_client: LLMClient[Any] = get_llm_client("llamaswap", "http://localhost:8080/v1", "some-model")
    responses = OpenAIAPIClient(model="gpt-4o-mini", bearer_token="sk-sample")
    swap = LlamaSwapAPIClient(model="some-model")
    completions = OpenAIChatCompletionsClient(model="gpt-4o-mini", bearer_token="sk-sample")
    swap_completions = LlamaSwapChatCompletionsClient(model="some-model")

    async with factory_client as entered:
        answer: str = await entered.chat_with_tools(
            MESSAGES,
            tools_definitions=TOOL_DEFINITIONS,
            tools={"fetch_record": fetch_record},
            tool_errors="raise",
            on_tool_event=observe_tool_event,
        )
        answer = await entered.chat_with_tools(
            MESSAGES,
            tools_definitions=TOOL_DEFINITIONS,
            tools={"fetch_record": fetch_record},
            tool_errors="raise",
            on_tool_event=observe_tool_event_async,
        )
        answer = await entered.chat_with_tools(
            MESSAGES,
            tools_definitions=TOOL_DEFINITIONS,
            tools={"fetch_record": fetch_record},
            tool_errors="result",
            on_tool_event=observe_tool_event,
        )

    answer = await responses.chat_with_tools(
        MESSAGES,
        tools_definitions=TOOL_DEFINITIONS,
        tools={"fetch_record": fetch_record},
        tool_errors="raise",
        on_tool_event=observe_tool_event,
    )
    answer = await responses.chat_with_tools(
        MESSAGES,
        tools_definitions=TOOL_DEFINITIONS,
        tools={"fetch_record": fetch_record},
        tool_errors="raise",
        on_tool_event=observe_tool_event_async,
    )
    answer = await swap.chat_with_tools(
        MESSAGES,
        tools_definitions=TOOL_DEFINITIONS,
        tools={"fetch_record": fetch_record},
        tool_errors="raise",
        on_tool_event=observe_tool_event,
    )
    answer = await swap.chat_with_tools(
        MESSAGES,
        tools_definitions=TOOL_DEFINITIONS,
        tools={"fetch_record": fetch_record},
        tool_errors="raise",
        on_tool_event=observe_tool_event_async,
    )
    answer = await completions.chat_with_tools(
        MESSAGES,
        tools_definitions=TOOL_DEFINITIONS,
        tools={"fetch_record": fetch_record},
        tool_errors="raise",
        on_tool_event=observe_tool_event,
    )
    answer = await completions.chat_with_tools(
        MESSAGES,
        tools_definitions=TOOL_DEFINITIONS,
        tools={"fetch_record": fetch_record},
        tool_errors="raise",
        on_tool_event=observe_tool_event_async,
    )
    answer = await swap_completions.chat_with_tools(
        MESSAGES,
        tools_definitions=TOOL_DEFINITIONS,
        tools={"fetch_record": fetch_record},
        tool_errors="raise",
        on_tool_event=observe_tool_event,
    )
    answer = await swap_completions.chat_with_tools(
        MESSAGES,
        tools_definitions=TOOL_DEFINITIONS,
        tools={"fetch_record": fetch_record},
        tool_errors="raise",
        on_tool_event=observe_tool_event_async,
    )
    print(answer)
''',
    ),
)


def _sample_source() -> str:
    """Assemble the sample module: docstring, imports, shared values, then every section in order."""
    parts = ['"""Sample application module for the typing contract (generated by tests/test_typing_contract.py)."""']
    parts.append("\n".join(_SAMPLE_IMPORTS))
    parts.append(textwrap.dedent(_SAMPLE_SHARED).strip())
    for name, source in _SAMPLE_SECTIONS:
        parts.append(f"# --- {name} ---\n" + textwrap.dedent(source).strip())
    return "\n\n\n".join(parts) + "\n"


# A module mypy --strict must reject (an unannotated function). Proves the harness really checks the file
# in strict mode, so a clean result for the sample is not vacuous.
_STRICT_CONTROL_SOURCE = '"""Strict-mode control module."""\n\n\ndef untyped(value):\n    return value\n'


def _check_module_strict(path: Path) -> subprocess.CompletedProcess[str]:
    """Check one application module in strict mode.

    ``--follow-imports=silent`` keeps the package analysed (so the module's uses of it are checked) but
    reports only the module's own errors; package errors are DoD-1's concern.
    """
    return _run_mypy("--strict", "--follow-imports=silent", str(path))


def test_mypy_strict_reports_no_errors_on_the_package__DoD1() -> None:
    """``mypy --strict llm`` from the repository root with the project config reports no errors (DoD-1)."""
    result = _run_mypy("--strict", "llm")

    assert result.returncode == 0, _report(result)
    assert "Success: no issues found" in result.stdout, _report(result)


def test_sample_application_type_checks_in_strict_mode__DoD2(tmp_path: Path) -> None:
    """A strict-mode application using the typing contract's surface type-checks without errors (DoD-2)."""
    control = tmp_path / "typing_contract_control.py"
    control.write_text(_STRICT_CONTROL_SOURCE, encoding="utf-8")
    control_result = _check_module_strict(control)
    assert control_result.returncode != 0, "strict-mode control was not rejected:\n" + _report(control_result)
    assert "no-untyped-def" in control_result.stdout, _report(control_result)

    sample = tmp_path / "typing_contract_sample.py"
    sample.write_text(_sample_source(), encoding="utf-8")
    result = _check_module_strict(sample)

    assert result.returncode == 0, _report(result) + "\n--- sample ---\n" + _sample_source()
    assert "Success: no issues found in 1 source file" in result.stdout, _report(result)


# Title of the step-009 section in ``_SAMPLE_SECTIONS`` (DoD-25); looked up by name so the test checks it alone.
_STEP_009_SECTION = "strict tool handling and typed tool-event observers (step 009)"


def _sample_source_with(section_names: tuple[str, ...]) -> str:
    """Assemble a sample module holding the imports, the shared values and only the named sections."""
    chosen = [(name, source) for name, source in _SAMPLE_SECTIONS if name in section_names]
    assert [name for name, _ in chosen] == list(section_names), "a named sample section is missing"
    parts = ['"""Partial sample application module for the typing contract."""']
    parts.append("\n".join(_SAMPLE_IMPORTS))
    parts.append(textwrap.dedent(_SAMPLE_SHARED).strip())
    for name, source in chosen:
        parts.append(f"# --- {name} ---\n" + textwrap.dedent(source).strip())
    return "\n\n\n".join(parts) + "\n"


def test_sample_with_strict_tool_errors_and_typed_observers_type_checks__DoD25(tmp_path: Path) -> None:
    """Strict mode: ``tool_errors="raise"`` and typed observers on factory and direct clients type-check (DoD-25).

    The section is checked on its own so a failure points at the step-009 surface; the full sample (all
    sections together) is the DoD-2 test, and the package itself stays clean per the DoD-1 test.
    """
    source = _sample_source_with((_STEP_009_SECTION,))
    sample = tmp_path / "typing_contract_sample_009.py"
    sample.write_text(source, encoding="utf-8")
    result = _check_module_strict(sample)

    assert result.returncode == 0, _report(result) + "\n--- sample ---\n" + source
    assert "Success: no issues found in 1 source file" in result.stdout, _report(result)


def test_strict_mode_rejects_a_misspelt_tool_errors_value__DoD25(tmp_path: Path) -> None:
    """The ``tool_errors`` parameter is typed as the two literals: a strict check rejects any other text (DoD-25).

    Guards against the DoD-25 sample passing only because the parameter is untyped or missing from the base
    declaration a factory-built client exposes.
    """
    source = "\n".join(_SAMPLE_IMPORTS) + "\n\n\n" + textwrap.dedent(_SAMPLE_SHARED).strip() + textwrap.dedent(
        '''


        async def misspelt_mode() -> None:
            """Pass a value that is neither "raise" nor "result"."""
            client: LLMClient[Any] = get_llm_client("llamaswap", "http://localhost:8080/v1", "some-model")
            await client.chat_with_tools(
                MESSAGES,
                tools_definitions=TOOL_DEFINITIONS,
                tools={"fetch_record": fetch_record},
                tool_errors="strict",
            )
        '''
    )
    sample = tmp_path / "typing_contract_misspelt_009.py"
    sample.write_text('"""Negative control for tool_errors typing."""\n' + source, encoding="utf-8")
    result = _check_module_strict(sample)

    assert result.returncode != 0, "a tool_errors value outside the literals was accepted:\n" + _report(result)
    assert "tool_errors" in result.stdout or "arg-type" in result.stdout, _report(result)


# ---------------------------------------------------------------------------------------------------------
# DoD-3: tuple of mappings accepted, caller's sequence unchanged (runtime, non-streamed, ``_request`` faked).
# ---------------------------------------------------------------------------------------------------------

ANSWER_TEXT = "Record r-1 is fine."

# The caller's conversation: plain dict turns inside a tuple (a read-only sequence of mappings).
CALLER_TURNS: tuple[dict[str, Any], ...] = (
    {"role": "user", "content": "Is record r-1 fine?"},
    {"role": "assistant", "content": "Let me check."},
    {"role": "user", "content": "Please check it."},
)

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fetch_record",
            "description": "Fetch one record by id.",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
            },
        },
    }
]


@pytest.fixture(autouse=True)
def _no_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tracing off whatever the developer's environment holds, so no trace files are written."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _responses_answer(text: str) -> dict[str, Any]:
    """A completed Responses reply holding one assistant message and no tool call."""
    return {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    }


def _chat_completions_answer(text: str) -> dict[str, Any]:
    """A chat-completions reply whose assistant message carries text and no tool calls."""
    return {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"},
        ]
    }


def _install_single_reply(
    monkeypatch: pytest.MonkeyPatch, client: Any, path: str, reply: dict[str, Any]
) -> list[dict[str, Any]]:
    """Fake ``_request`` with one canned reply; record each request (deep-copied to avoid aliasing)."""
    records: list[dict[str, Any]] = []

    async def fake_request(method: str, request_path: str, **kw: Any) -> Any:
        records.append({"method": method, "path": request_path, "json": copy.deepcopy(kw.get("json"))})
        if len(records) > 1:
            raise AssertionError("a reply without tool calls must end the conversation after one request")
        return copy.deepcopy(reply)

    monkeypatch.setattr(client, "_request", fake_request)
    return records


def _record_tool_calls(calls: list[dict[str, Any]]) -> Any:
    """A ``fetch_record`` callable that records its calls; the reply asks for no tool, so it must not run."""

    def fetch_record(record_id: str) -> str:
        calls.append({"record_id": record_id})
        return record_id

    return fetch_record


_CASES = [
    pytest.param(
        lambda: OpenAIAPIClient(model="gpt-4o-mini", bearer_token="sk-test"),
        "/responses",
        _responses_answer,
        id="openai_responses",
    ),
    pytest.param(
        lambda: LlamaSwapAPIClient(model="some-model", base_url="http://localhost:8080/v1", bearer_token="sk-test"),
        "/responses",
        _responses_answer,
        id="llamaswap_responses",
    ),
    pytest.param(
        lambda: OpenAIChatCompletionsClient(model="gpt-4o-mini", bearer_token="sk-test"),
        "/chat/completions",
        _chat_completions_answer,
        id="openai_chat_completions",
    ),
    pytest.param(
        lambda: LlamaSwapChatCompletionsClient(
            model="some-model", base_url="http://localhost:8080/v1", bearer_token="sk-test"
        ),
        "/chat/completions",
        _chat_completions_answer,
        id="llamaswap_chat_completions",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("make_client", "path", "make_reply"), _CASES)
async def test_openai_family_chat_with_tools_accepts_tuple_of_mappings__DoD3(
    monkeypatch: pytest.MonkeyPatch, make_client: Any, path: str, make_reply: Any
) -> None:
    """Each OpenAI-family class takes a tuple of mappings as ``messages`` and returns the answer (DoD-3)."""
    client = make_client()
    records = _install_single_reply(monkeypatch, client, path, make_reply(ANSWER_TEXT))
    tool_calls: list[dict[str, Any]] = []
    messages = tuple(dict(turn) for turn in CALLER_TURNS)

    answer = await client.chat_with_tools(
        messages,
        tools_definitions=copy.deepcopy(TOOL_DEFINITIONS),
        tools={"fetch_record": _record_tool_calls(tool_calls)},
    )

    assert answer == ANSWER_TEXT
    assert len(records) == 1
    assert records[0]["method"] == "POST"
    assert records[0]["path"] == path
    assert tool_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("make_client", "path", "make_reply"), _CASES)
async def test_openai_family_chat_with_tools_leaves_caller_tuple_unchanged__DoD3(
    monkeypatch: pytest.MonkeyPatch, make_client: Any, path: str, make_reply: Any
) -> None:
    """The caller's tuple and the mappings in it are unchanged after the call (DoD-3)."""
    client = make_client()
    _install_single_reply(monkeypatch, client, path, make_reply(ANSWER_TEXT))
    turns = [dict(turn) for turn in CALLER_TURNS]
    messages = tuple(turns)
    snapshot = copy.deepcopy(messages)

    await client.chat_with_tools(
        messages,
        tools_definitions=copy.deepcopy(TOOL_DEFINITIONS),
        tools={"fetch_record": _record_tool_calls([])},
    )

    assert messages == snapshot
    assert len(messages) == len(CALLER_TURNS)
    # Same turn objects, each with its original content: nothing was swapped in or mutated in place.
    assert all(sent is original for sent, original in zip(messages, turns, strict=True))
    assert list(messages) == list(CALLER_TURNS)
