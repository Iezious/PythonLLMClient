"""Per-client trace switch tests (feature 006, step 004; DEC-arch-27, DEC-006-5, DEC-006-6, DEC-000-4).

Covers the three-state ``trace`` constructor argument on all five client classes and on ``get_llm_client``:
``trace=False`` never traces whatever the environment says (DoD-1); ``trace=True`` traces into
``LLM_TRACE_FOLDER`` without consulting ``LLM_TRACE`` (DoD-2) and refuses creation with ``ValueError`` naming
``LLM_TRACE_FOLDER`` when that folder is unset or empty, before any server is contacted (DoD-3); leaving
``trace`` out (or passing ``None``) keeps the environment rule of DEC-000-4 (DoD-4); all of it per class and
through the factory for every ``server_type`` / ``protocol`` combination (DoD-5); the public ``LLMTracer()``
keeps deciding from the environment (DoD-6). The private construction path frozen by the skeleton
(``LLMTracer._from_switch``) is checked directly as well.

The environment is controlled only with ``monkeypatch.setenv`` / ``delenv`` (tests/AGENTS.md), and a trace
record is observed as a ``*.log`` file under ``tmp_path`` (docs/architecture/data-model.md § Trace record).
Every expected value comes from the step's Definition of done, US-069 and the cited decisions.
"""

from pathlib import Path
from typing import Any

import aiohttp
import pytest

from llm.exceptions import LLMError
from llm.llama_swap import LlamaSwapAPIClient
from llm.llama_swap_chat_completions import LlamaSwapChatCompletionsClient
from llm.llm import get_llm_client
from llm.ollama import OllamaAPIClient
from llm.openai import OpenAIAPIClient
from llm.openai_chat_completions import OpenAIChatCompletionsClient
from llm.trace import LLMTracer

MODEL = "trace-switch-model"
USER_MESSAGES = [{"role": "user", "content": "Say hello."}]

# Addresses on reserved test domains: nothing here may ever reach a real server.
_OPENAI_FAMILY_URL = "http://llm.test.invalid/v1"
_OLLAMA_URL = "http://ollama.test.invalid:11434"
_FACTORY_OPENAI_TOKEN = "sk-test"

# The five client classes, each with the constructor arguments it needs (Ollama has no ``/v1``).
_CLASSES = [
    pytest.param(OpenAIAPIClient, _OPENAI_FAMILY_URL, id="openai_responses"),
    pytest.param(OpenAIChatCompletionsClient, _OPENAI_FAMILY_URL, id="openai_chat_completions"),
    pytest.param(LlamaSwapAPIClient, _OPENAI_FAMILY_URL, id="llamaswap_responses"),
    pytest.param(LlamaSwapChatCompletionsClient, _OPENAI_FAMILY_URL, id="llamaswap_chat_completions"),
    pytest.param(OllamaAPIClient, _OLLAMA_URL, id="ollama"),
]

# Every factory choice the spec accepts, with the class it must pick (interfaces.md § Exposed: factory).
_FACTORY_COMBINATIONS = [
    pytest.param("openai", None, OpenAIAPIClient, id="openai-omitted"),
    pytest.param("openai", "responses", OpenAIAPIClient, id="openai-responses"),
    pytest.param("openai", "chat_completions", OpenAIChatCompletionsClient, id="openai-chat_completions"),
    pytest.param("llamaswap", None, LlamaSwapAPIClient, id="llamaswap-omitted"),
    pytest.param("llamaswap", "responses", LlamaSwapAPIClient, id="llamaswap-responses"),
    pytest.param("llamaswap", "chat_completions", LlamaSwapChatCompletionsClient, id="llamaswap-chat_completions"),
    pytest.param("ollama", None, OllamaAPIClient, id="ollama"),
]

# ``LLM_TRACE`` states that do not enable tracing on their own (DEC-000-4: true-like is 1 / true / on).
_SWITCH_NOT_ON = [None, "0", "false", "off"]
# True-like ``LLM_TRACE`` values (DEC-000-4).
_SWITCH_ON = ["1", "true", "on"]
# The two ways the trace folder is missing (DoD-3: unset or empty).
_FOLDER_MISSING = ["unset", "empty"]


class _CallFailed(Exception):
    """Raised by the fake server seams: the call is traced whatever its outcome, so it need not succeed."""


class _RefusingSession:
    """Stands in for ``client.session`` so a call that bypasses ``_request`` still never reaches the network.

    Every request records itself and fails with ``_CallFailed``; ``_ensure_session`` reuses an open session
    (tests/AGENTS.md, fake-session seam).
    """

    closed = False

    def __init__(self, log: list[str]) -> None:
        self._log = log

    def request(self, method: str, url: Any, **kwargs: Any) -> Any:
        self._log.append(f"session {method} {url}")
        raise _CallFailed("fake session refuses every request")

    def post(self, url: Any, **kwargs: Any) -> Any:
        return self.request("POST", url, **kwargs)

    def get(self, url: Any, **kwargs: Any) -> Any:
        return self.request("GET", url, **kwargs)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _clean_trace_environment(monkeypatch, tmp_path):
    """Start every test with no trace variables and the working directory in ``tmp_path``.

    Running from ``tmp_path`` means a record written relative to the working directory would also be found by
    ``_records``; clearing the variables keeps the developer's own environment out of the expectations.
    """
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)
    monkeypatch.chdir(tmp_path)


def _set_switch(monkeypatch, value: str | None) -> None:
    """Set ``LLM_TRACE`` to ``value``, or remove it when ``value`` is ``None``."""
    if value is None:
        monkeypatch.delenv("LLM_TRACE", raising=False)
    else:
        monkeypatch.setenv("LLM_TRACE", value)


def _set_folder(monkeypatch, mode: str, folder: Path) -> None:
    """Make ``LLM_TRACE_FOLDER`` name ``folder`` (``"set"``), be empty (``"empty"``) or be absent (``"unset"``)."""
    if mode == "set":
        monkeypatch.setenv("LLM_TRACE_FOLDER", str(folder))
    elif mode == "empty":
        monkeypatch.setenv("LLM_TRACE_FOLDER", "")
    else:
        monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)


def _build(cls: type, base_url: str, **kwargs: Any) -> Any:
    """Build one of the five classes directly with an explicit credential (tests/AGENTS.md)."""
    return cls(model=MODEL, base_url=base_url, bearer_token="sk-test", **kwargs)


def _build_through_factory(server_type: str, protocol: str | None, **kwargs: Any) -> Any:
    """Build through ``get_llm_client`` with kind, address and model given and an explicit credential."""
    base_url = _OLLAMA_URL if server_type == "ollama" else _OPENAI_FAMILY_URL
    if server_type == "ollama":
        # Ollama refuses any protocol, so none is passed (DEC-003-8).
        return get_llm_client(server_type, base_url, MODEL, bearer_token=_FACTORY_OPENAI_TOKEN, **kwargs)
    return get_llm_client(
        server_type, base_url, MODEL, bearer_token=_FACTORY_OPENAI_TOKEN, protocol=protocol, **kwargs
    )


async def _make_call(monkeypatch, client: Any) -> list[str]:
    """Make one traced ``chat`` call against faked server seams; returns what reached the fakes.

    ``chat`` is a traced operation on every class (subsystems.md § Tracer) and errors raised inside the traced
    part are recorded, so a failing fake server still yields a record when tracing is on. Both the ``_request``
    seam and the session are faked so no class can reach the network.
    """
    reached: list[str] = []

    async def fake_request(method: str, path: str, **kw: Any) -> Any:
        reached.append(f"_request {method} {path}")
        raise _CallFailed("fake server refuses every request")

    monkeypatch.setattr(client, "_request", fake_request)
    client.session = _RefusingSession(reached)
    try:
        await client.chat(list(USER_MESSAGES))
    except (_CallFailed, LLMError):
        pass
    assert reached, "the call never reached the fake server, so it proves nothing about tracing"
    return reached


def _records(root: Path) -> list[Path]:
    return sorted(root.rglob("*.log"))


def _assert_one_record_in(folder: Path, root: Path) -> None:
    """Exactly one record exists below ``root``, and it sits in a date folder of ``folder`` (data-model.md)."""
    records = _records(root)
    assert len(records) == 1, records
    assert records[0].parent.parent == folder, records[0]


def _install_server_contact_log(monkeypatch) -> list[str]:
    """Record any HTTP request an ``aiohttp`` session would send, so a test can prove none was made."""
    contacted: list[str] = []

    async def recording_request(self, method, url, *args, **kwargs):
        contacted.append(f"{method} {url}")
        raise AssertionError("a server was contacted")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", recording_request, raising=False)
    return contacted


# ---------------------------------------------------------------------------
# DoD-1 / DoD-5 — trace=False writes nothing even when the environment turns tracing on
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("switch", _SWITCH_ON)
@pytest.mark.parametrize(("cls", "base_url"), _CLASSES)
async def test_client_built_with_trace_false_writes_no_record__DoD1_DoD5(monkeypatch, tmp_path, cls, base_url, switch):
    """DoD-1 / DoD-5 (US-069.AC-1): each class built with ``trace=False`` writes no record for a call although
    ``LLM_TRACE`` is true-like and ``LLM_TRACE_FOLDER`` names a folder."""
    folder = tmp_path / "traces"
    _set_switch(monkeypatch, switch)
    _set_folder(monkeypatch, "set", folder)
    client = _build(cls, base_url, trace=False)

    await _make_call(monkeypatch, client)

    assert _records(tmp_path) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("server_type", "protocol", "expected"), _FACTORY_COMBINATIONS)
async def test_factory_forwards_trace_false__DoD1_DoD5(monkeypatch, tmp_path, server_type, protocol, expected):
    """DoD-1 / DoD-5: ``get_llm_client(..., trace=False)`` gives the picked class a tracer that never writes,
    with tracing switched on in the environment, for every ``server_type`` / ``protocol`` combination."""
    folder = tmp_path / "traces"
    _set_switch(monkeypatch, "1")
    _set_folder(monkeypatch, "set", folder)
    client = _build_through_factory(server_type, protocol, trace=False)
    assert type(client) is expected

    await _make_call(monkeypatch, client)

    assert _records(tmp_path) == []


@pytest.mark.parametrize("switch", _SWITCH_ON)
def test_tracer_switch_false_is_disabled_whatever_the_environment__DoD1(monkeypatch, tmp_path, switch):
    """DoD-1 (DEC-006-6): the private path given ``False`` returns a disabled tracer with tracing on in the
    environment; a scope on it writes nothing."""
    _set_switch(monkeypatch, switch)
    _set_folder(monkeypatch, "set", tmp_path / "traces")

    tracer = LLMTracer._from_switch(False)

    assert isinstance(tracer, LLMTracer)
    assert tracer.enabled is False
    with tracer.trace("chat", MODEL) as t:
        t.log_response("must not be written")
    assert _records(tmp_path) == []


# ---------------------------------------------------------------------------
# DoD-2 / DoD-5 — trace=True with a folder writes a record while LLM_TRACE is off
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("switch", _SWITCH_NOT_ON)
@pytest.mark.parametrize(("cls", "base_url"), _CLASSES)
async def test_client_built_with_trace_true_writes_record__DoD2_DoD5(monkeypatch, tmp_path, cls, base_url, switch):
    """DoD-2 / DoD-5 (US-069.AC-2): each class built with ``trace=True`` and ``LLM_TRACE_FOLDER`` set writes one
    record for a call into that folder while ``LLM_TRACE`` is unset or false-like."""
    folder = tmp_path / "traces"
    _set_switch(monkeypatch, switch)
    _set_folder(monkeypatch, "set", folder)
    client = _build(cls, base_url, trace=True)

    await _make_call(monkeypatch, client)

    _assert_one_record_in(folder, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize(("server_type", "protocol", "expected"), _FACTORY_COMBINATIONS)
async def test_factory_forwards_trace_true__DoD2_DoD5(monkeypatch, tmp_path, server_type, protocol, expected):
    """DoD-2 / DoD-5: ``get_llm_client(..., trace=True)`` with a folder and ``LLM_TRACE`` unset gives the picked
    class a tracer that writes one record per call, for every ``server_type`` / ``protocol`` combination."""
    folder = tmp_path / "traces"
    _set_switch(monkeypatch, None)
    _set_folder(monkeypatch, "set", folder)
    client = _build_through_factory(server_type, protocol, trace=True)
    assert type(client) is expected

    await _make_call(monkeypatch, client)

    _assert_one_record_in(folder, tmp_path)


@pytest.mark.parametrize("switch", _SWITCH_NOT_ON)
def test_tracer_switch_true_is_enabled_without_llm_trace__DoD2(monkeypatch, tmp_path, switch):
    """DoD-2 (DEC-006-6): the private path given ``True`` returns an enabled tracer writing into
    ``LLM_TRACE_FOLDER`` while ``LLM_TRACE`` is unset or false-like."""
    folder = tmp_path / "traces"
    _set_switch(monkeypatch, switch)
    _set_folder(monkeypatch, "set", folder)

    tracer = LLMTracer._from_switch(True)

    assert isinstance(tracer, LLMTracer)
    assert tracer.enabled is True
    with tracer.trace("chat", MODEL) as t:
        t.log_response("written under the forced switch")
    _assert_one_record_in(folder, tmp_path)
    assert "written under the forced switch" in _records(tmp_path)[0].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# DoD-3 / DoD-5 — trace=True without a folder is refused at creation, naming the setting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("switch", [None, "1"])
@pytest.mark.parametrize("folder_mode", _FOLDER_MISSING)
@pytest.mark.parametrize(("cls", "base_url"), _CLASSES)
def test_client_built_with_trace_true_without_folder_is_refused__DoD3_DoD5(
    monkeypatch, tmp_path, cls, base_url, folder_mode, switch
):
    """DoD-3 / DoD-5 (US-069.AC-3): building each class with ``trace=True`` while ``LLM_TRACE_FOLDER`` is unset or
    empty raises ``ValueError`` naming ``LLM_TRACE_FOLDER`` (whatever ``LLM_TRACE`` says); no server is contacted
    and nothing is written."""
    _set_switch(monkeypatch, switch)
    _set_folder(monkeypatch, folder_mode, tmp_path / "traces")
    contacted = _install_server_contact_log(monkeypatch)

    with pytest.raises(ValueError) as exc_info:
        _build(cls, base_url, trace=True)

    assert "LLM_TRACE_FOLDER" in str(exc_info.value)
    assert contacted == []
    assert _records(tmp_path) == []


@pytest.mark.parametrize("folder_mode", _FOLDER_MISSING)
@pytest.mark.parametrize(("server_type", "protocol", "expected"), _FACTORY_COMBINATIONS)
def test_factory_forwards_trace_true_without_folder_refusal__DoD3_DoD5(
    monkeypatch, tmp_path, server_type, protocol, expected, folder_mode
):
    """DoD-3 / DoD-5: ``get_llm_client(..., trace=True)`` without a trace folder raises ``ValueError`` naming
    ``LLM_TRACE_FOLDER`` for every ``server_type`` / ``protocol`` combination; no server is contacted."""
    _set_switch(monkeypatch, "1")
    _set_folder(monkeypatch, folder_mode, tmp_path / "traces")
    contacted = _install_server_contact_log(monkeypatch)

    with pytest.raises(ValueError) as exc_info:
        _build_through_factory(server_type, protocol, trace=True)

    assert "LLM_TRACE_FOLDER" in str(exc_info.value)
    assert contacted == []


@pytest.mark.parametrize("switch", [None, "1"])
@pytest.mark.parametrize("folder_mode", _FOLDER_MISSING)
def test_tracer_switch_true_without_folder_raises_value_error__DoD3(monkeypatch, tmp_path, folder_mode, switch):
    """DoD-3 (DEC-006-6): the private path given ``True`` raises ``ValueError`` naming ``LLM_TRACE_FOLDER`` when that
    variable is unset or empty."""
    _set_switch(monkeypatch, switch)
    _set_folder(monkeypatch, folder_mode, tmp_path / "traces")

    with pytest.raises(ValueError) as exc_info:
        LLMTracer._from_switch(True)

    assert "LLM_TRACE_FOLDER" in str(exc_info.value)


# ---------------------------------------------------------------------------
# DoD-4 — trace not passed (or None): the environment decides, as before
# ---------------------------------------------------------------------------

# (LLM_TRACE, folder mode, traced?) per DEC-000-4: on only with a true-like switch and a non-empty folder.
_ENVIRONMENT_CASES = [
    pytest.param("1", "set", True, id="on-1"),
    pytest.param("true", "set", True, id="on-true"),
    pytest.param("on", "set", True, id="on-on"),
    pytest.param(None, "set", False, id="switch-unset"),
    pytest.param("0", "set", False, id="switch-0"),
    pytest.param("off", "set", False, id="switch-off"),
    pytest.param("1", "unset", False, id="folder-unset"),
    pytest.param("1", "empty", False, id="folder-empty"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("pass_none", [False, True], ids=["omitted", "explicit_none"])
@pytest.mark.parametrize(("switch", "folder_mode", "traced"), _ENVIRONMENT_CASES)
@pytest.mark.parametrize(("cls", "base_url"), _CLASSES)
async def test_client_without_trace_follows_the_environment__DoD4(
    monkeypatch, tmp_path, cls, base_url, switch, folder_mode, traced, pass_none
):
    """DoD-4 (US-069.AC-4): a client built without ``trace`` (or with ``trace=None``) writes a record exactly when
    ``LLM_TRACE`` is true-like and ``LLM_TRACE_FOLDER`` is non-empty; otherwise none."""
    folder = tmp_path / "traces"
    _set_switch(monkeypatch, switch)
    _set_folder(monkeypatch, folder_mode, folder)
    client = _build(cls, base_url, trace=None) if pass_none else _build(cls, base_url)

    await _make_call(monkeypatch, client)

    if traced:
        _assert_one_record_in(folder, tmp_path)
    else:
        assert _records(tmp_path) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("switch", "folder_mode", "traced"), _ENVIRONMENT_CASES)
@pytest.mark.parametrize(("server_type", "protocol", "expected"), _FACTORY_COMBINATIONS)
async def test_factory_without_trace_follows_the_environment__DoD4(
    monkeypatch, tmp_path, server_type, protocol, expected, switch, folder_mode, traced
):
    """DoD-4: a factory-built client without ``trace`` traces exactly when the environment enables it."""
    folder = tmp_path / "traces"
    _set_switch(monkeypatch, switch)
    _set_folder(monkeypatch, folder_mode, folder)
    client = _build_through_factory(server_type, protocol)
    assert type(client) is expected

    await _make_call(monkeypatch, client)

    if traced:
        _assert_one_record_in(folder, tmp_path)
    else:
        assert _records(tmp_path) == []


@pytest.mark.parametrize(("switch", "folder_mode", "traced"), _ENVIRONMENT_CASES)
def test_tracer_switch_none_reads_the_environment__DoD4(monkeypatch, tmp_path, switch, folder_mode, traced):
    """DoD-4 (DEC-006-6): the private path given ``None`` is exactly the environment read of DEC-000-4."""
    folder = tmp_path / "traces"
    _set_switch(monkeypatch, switch)
    _set_folder(monkeypatch, folder_mode, folder)

    tracer = LLMTracer._from_switch(None)

    assert isinstance(tracer, LLMTracer)
    assert tracer.enabled is traced
    with tracer.trace("chat", MODEL) as t:
        t.log_response("environment decides")
    if traced:
        _assert_one_record_in(folder, tmp_path)
    else:
        assert _records(tmp_path) == []


# ---------------------------------------------------------------------------
# DoD-6 — LLMTracer() with no arguments behaves as before
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("switch", "folder_mode", "traced"), _ENVIRONMENT_CASES)
def test_public_tracer_without_arguments_follows_the_environment__DoD6(
    monkeypatch, tmp_path, switch, folder_mode, traced
):
    """DoD-6 (US-069.AC-4, DEC-006-6): ``LLMTracer()`` is enabled, and a scope on it writes one record into
    ``LLM_TRACE_FOLDER``, exactly when ``LLM_TRACE`` is true-like and the folder is non-empty."""
    folder = tmp_path / "traces"
    _set_switch(monkeypatch, switch)
    _set_folder(monkeypatch, folder_mode, folder)

    tracer = LLMTracer()

    assert tracer.enabled is traced
    with tracer.trace("chat", MODEL) as t:
        t.log_response("public constructor")
    if traced:
        _assert_one_record_in(folder, tmp_path)
    else:
        assert _records(tmp_path) == []
