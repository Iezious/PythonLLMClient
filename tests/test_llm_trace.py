"""Unit tests for trace robustness (plan 002.edge-fixes, step 004).

Covers: a record that cannot be set up leaves the call untraced (DEC-002-11); a record-write failure never
replaces the call's result or error and never leaks the file (DEC-002-12); exclusive record creation with a
counter suffix on name collisions (DEC-002-13); the system instruction shown once among the conversation
sections (DEC-002-14); the record layout and the on/off switch (US-048). Uses the DEC-002-16 seams: the
module-level ``llm.trace.datetime`` clock and the ``open`` looked up through ``llm.trace``. Every expected
value comes from the step's Definition of done, the cited decisions and the trace-record data model.
"""

import asyncio
import builtins
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import llm.trace
from llm.exceptions import LLMError
from llm.llama_swap_chat_completions import LlamaSwapChatCompletionsClient
from llm.openai_chat_completions import OpenAIChatCompletionsClient
from llm.trace import LLMTracer

BASE_URL = "https://api.example.test/v1"
MODEL = "trace-test-model"

# One fixed UTC instant for the frozen clock; non-zero microseconds so the file name shows them.
FIXED_NOW = datetime(2026, 3, 4, 5, 6, 7, 890123, tzinfo=UTC)
# The documented layout for FIXED_NOW: UTC date folder, timestamp file name (DEC-000-44, DEC-002-13).
DATE_FOLDER = "2026-03-04"
BASE_NAME = "llm-trace-20260304-050607890123"

INSTRUCTION = "Reply only in French, citing rule 7781."
CALLER_SYSTEM = "Caller house rule: keep answers brief."
USER_TEXT = "What time is it?"
USER_MESSAGES = [{"role": "user", "content": USER_TEXT}]
REPLY_TEXT = "Il est midi."

ERROR_MESSAGE = f"Request to {BASE_URL}/chat/completions failed with status 500"
ERROR_DESCRIPTION = "upstream exploded"

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_record",
            "description": "Retrieve a stored record by its key.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    }
]
TOOL_CALLS = [
    {"id": "call_1", "type": "function", "function": {"name": "fetch_record", "arguments": '{"key": "k1"}'}},
]

# Captured before any test patches ``llm.trace.open`` so the fakes can still create real files.
_REAL_OPEN = builtins.open

# Section titles of the trace record (docs/architecture/data-model.md § Trace record).
_HEADINGS = frozenset({"SYSTEM PROMPT", "MESSAGE HISTORY", "REQUEST", "RESPONSE", "CHUNK", "ERROR", "END TRACE"})


def _fetch_record(key: str) -> str:
    return "alpha-77"


TOOLS = {"fetch_record": _fetch_record}


# ---------------------------------------------------------------------------
# Seams and fakes
# ---------------------------------------------------------------------------


class _FrozenClock:
    """Stands in for ``llm.trace.datetime`` so every session starts at FIXED_NOW (DEC-002-16)."""

    @staticmethod
    def now(tz: Any = None) -> datetime:
        return FIXED_NOW


class _FlakyFile:
    """A real record file whose writes start failing on demand (DEC-002-16 ``open`` seam).

    The first failing write is triggered either by ``fail_now`` or by the first write whose text contains
    ``fail_on`` (``""`` matches the very first write, i.e. the header). Once failed, every later write
    attempt is counted in ``writes_after_failure`` and fails too, so a test can see whether the session
    kept writing after its first failure.
    """

    def __init__(self, inner: Any, fail_on: str | None) -> None:
        self._inner = inner
        self._fail_on = fail_on
        self.fail_now = False
        self.broken = False
        self.writes_after_failure = 0
        self.close_calls = 0

    def write(self, text: str) -> int:
        if self.broken:
            self.writes_after_failure += 1
            raise OSError("simulated disk failure")
        if self.fail_now or (self._fail_on is not None and self._fail_on in text):
            self.broken = True
            raise OSError("simulated disk failure")
        return self._inner.write(text)

    def flush(self) -> None:
        if self.broken:
            raise OSError("simulated disk failure")
        self._inner.flush()

    def close(self) -> None:
        self.close_calls += 1
        self._inner.close()

    @property
    def closed(self) -> bool:
        return self._inner.closed


def _install_flaky_open(monkeypatch, fail_on: str | None = None) -> list[_FlakyFile]:
    """Route the tracer's ``open`` to real files wrapped in ``_FlakyFile``; returns every file it opened."""
    opened: list[_FlakyFile] = []

    def fake_open(file, mode="r", *args, **kwargs):
        wrapped = _FlakyFile(_REAL_OPEN(file, mode, *args, **kwargs), fail_on)
        opened.append(wrapped)
        return wrapped

    monkeypatch.setattr(llm.trace, "open", fake_open, raising=False)
    return opened


def _install_refusing_open(monkeypatch) -> None:
    """Make every record open fail with an ``OSError`` that is not a name collision (DoD-1)."""

    def refusing_open(file, mode="r", *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(file))

    monkeypatch.setattr(llm.trace, "open", refusing_open, raising=False)


def _enable_tracing(monkeypatch, folder: Path | str) -> None:
    """Turn tracing on; must run before the client or tracer is built (env read at construction, DEC-000-4)."""
    monkeypatch.setenv("LLM_TRACE", "1")
    monkeypatch.setenv("LLM_TRACE_FOLDER", str(folder))


def _freeze_clock(monkeypatch) -> None:
    monkeypatch.setattr(llm.trace, "datetime", _FrozenClock)


def _openai() -> OpenAIChatCompletionsClient:
    return OpenAIChatCompletionsClient(model=MODEL, base_url=BASE_URL, bearer_token="sk-test")


def _llamaswap() -> LlamaSwapChatCompletionsClient:
    return LlamaSwapChatCompletionsClient(model=MODEL, base_url=BASE_URL, bearer_token="sk-test")


def _reply(content: str | None, tool_calls: list[dict] | None = None) -> dict:
    """A non-streamed chat-completions reply body."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}


def _fake_replies(monkeypatch, client: Any, replies: list[dict]) -> list[dict]:
    """Replace ``_request`` with a fake returning canned replies; records each request body (messages copied)."""
    sent: list[dict] = []
    pending = list(replies)

    async def fake_request(method, path, **kw):
        assert method == "POST"
        assert path.endswith("/chat/completions")
        body = dict(kw["json"])
        body["messages"] = [dict(m) for m in body["messages"]]
        sent.append(body)
        return pending.pop(0)

    monkeypatch.setattr(client, "_request", fake_request)
    return sent


def _call_error() -> LLMError:
    return LLMError(ERROR_MESSAGE, description=ERROR_DESCRIPTION)


def _fake_failure(monkeypatch, client: Any, before: Any = None) -> None:
    """Replace ``_request`` with a fake that fails the call with the library's own error.

    ``before`` runs just before the error is raised, so a test can make record writes fail exactly while
    the error is being recorded.
    """

    async def fake_request(method, path, **kw):
        if before is not None:
            before()
        raise _call_error()

    monkeypatch.setattr(client, "_request", fake_request)


# ---------------------------------------------------------------------------
# Record reading
# ---------------------------------------------------------------------------


def _records(root: Path) -> list[Path]:
    return sorted(root.rglob("*.log"))


def _only_record(root: Path) -> str:
    records = _records(root)
    assert len(records) == 1, records
    return records[0].read_text(encoding="utf-8")


def _heading(line: str) -> str | None:
    stripped = line.strip()
    if stripped in _HEADINGS or stripped.startswith(("TOOL CALL:", "TOOL RESULT:")):
        return stripped
    return None


def _sections(text: str) -> list[tuple[str, list[str]]]:
    """Split a record into ``(title, body lines)``; lines before the first section form ``"HEADER"``.

    Blank lines and pure ``=``/``-`` rule lines are layout, not content, and are skipped.
    """
    sections: list[tuple[str, list[str]]] = [("HEADER", [])]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) <= {"=", "-"}:
            continue
        title = _heading(line)
        if title is not None:
            sections.append((title, []))
        else:
            sections[-1][1].append(line.rstrip())
    return sections


def _titles(text: str) -> list[str]:
    return [title for title, _ in _sections(text)]


def _bodies(text: str, title: str) -> list[list[str]]:
    return [[line.strip() for line in body] for name, body in _sections(text) if name == title]


def _header(text: str) -> str:
    return " ".join(_sections(text)[0][1])


def _history(text: str) -> list[str]:
    bodies = _bodies(text, "MESSAGE HISTORY")
    assert len(bodies) == 1, bodies
    return bodies[0]


# ---------------------------------------------------------------------------
# DoD-1 — unusable trace folder: the call is untraced
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("layout", ["folder_is_file", "date_folder_is_file"])
async def test_openai_unusable_trace_folder_returns_result_without_record__DoD1(monkeypatch, tmp_path, layout):
    """DoD-1: when the date folder cannot be created (a file is in the way), the call returns normally, untraced."""
    _freeze_clock(monkeypatch)
    if layout == "folder_is_file":
        folder = tmp_path / "traces.txt"
        blocker = folder
    else:
        folder = tmp_path / "traces"
        folder.mkdir()
        blocker = folder / DATE_FOLDER
    blocker.write_text("not a folder", encoding="utf-8")
    _enable_tracing(monkeypatch, folder)
    client = _openai()
    _fake_replies(monkeypatch, client, [_reply(REPLY_TEXT)])

    assert await client.chat(USER_MESSAGES, system=INSTRUCTION) == REPLY_TEXT
    assert _records(tmp_path) == []
    assert blocker.read_text(encoding="utf-8") == "not a folder"


@pytest.mark.asyncio
async def test_openai_record_open_failure_returns_result_without_record__DoD1(monkeypatch, tmp_path):
    """DoD-1: an ``OSError`` opening the record leaves the call untraced and its result unchanged."""
    _enable_tracing(monkeypatch, tmp_path)
    _install_refusing_open(monkeypatch)
    client = _openai()
    _fake_replies(monkeypatch, client, [_reply(REPLY_TEXT)])

    assert await client.chat(USER_MESSAGES, system=INSTRUCTION) == REPLY_TEXT
    assert _records(tmp_path) == []


@pytest.mark.asyncio
async def test_openai_record_open_failure_keeps_the_call_error__DoD1(monkeypatch, tmp_path):
    """DoD-1: after a setup failure the call's own error still reaches the app unchanged."""
    _enable_tracing(monkeypatch, tmp_path)
    _install_refusing_open(monkeypatch)
    client = _openai()
    _fake_failure(monkeypatch, client)

    with pytest.raises(LLMError) as excinfo:
        await client.chat(USER_MESSAGES)
    assert type(excinfo.value) is LLMError
    assert str(excinfo.value) == str(_call_error())
    assert _records(tmp_path) == []


def test_tracer_exhausted_collision_suffixes_leave_call_untraced__DoD1(monkeypatch, tmp_path):
    """DoD-1 / DEC-002-13: every suffix up to the bound taken -> untraced scope, taken records untouched."""
    _freeze_clock(monkeypatch)
    _enable_tracing(monkeypatch, tmp_path)
    monkeypatch.setattr(llm.trace, "_MAX_COLLISION_SUFFIX", 2)
    date_dir = tmp_path / DATE_FOLDER
    date_dir.mkdir()
    taken = [date_dir / f"{BASE_NAME}.log", date_dir / f"{BASE_NAME}-1.log", date_dir / f"{BASE_NAME}-2.log"]
    for path in taken:
        path.write_text("someone else's record", encoding="utf-8")
    tracer = LLMTracer()
    body_ran = False

    with tracer.trace("chat", MODEL) as t:
        t.log_response("must not be written anywhere")
        body_ran = True

    assert body_ran
    assert _records(tmp_path) == sorted(taken)
    for path in taken:
        assert path.read_text(encoding="utf-8") == "someone else's record"


# ---------------------------------------------------------------------------
# DoD-2 — write failures during a successful call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_on", ["", "SYSTEM PROMPT", "MESSAGE HISTORY", "REQUEST", "RESPONSE"])
async def test_openai_record_write_failure_returns_result_and_closes_file__DoD2(monkeypatch, tmp_path, fail_on):
    """DoD-2: writes failing from the header or any later section never break a successful call; file closed."""
    _enable_tracing(monkeypatch, tmp_path)
    opened = _install_flaky_open(monkeypatch, fail_on=fail_on)
    client = _openai()
    _fake_replies(monkeypatch, client, [_reply(REPLY_TEXT)])

    assert await client.chat(USER_MESSAGES, system=INSTRUCTION) == REPLY_TEXT
    assert len(opened) == 1
    assert opened[0].broken, "the write failure was not exercised"
    assert opened[0].closed


def test_tracer_session_skips_every_write_after_first_failure__DoD2(monkeypatch, tmp_path):
    """DoD-2 / DEC-002-12: after the first failed write the session attempts no further write; the file is closed."""
    _enable_tracing(monkeypatch, tmp_path)
    opened = _install_flaky_open(monkeypatch, fail_on="REQUEST")
    tracer = LLMTracer()

    with tracer.trace("chat", MODEL) as t:
        t.log_system(INSTRUCTION)
        t.log_messages(USER_MESSAGES)
        t.log_request("POST", "/chat/completions", {"model": MODEL})
        t.log_response(REPLY_TEXT)
        t.log_tool_call("fetch_record", {"key": "k1"})
        t.log_tool_result("fetch_record", "alpha-77")

    assert len(opened) == 1
    assert opened[0].broken
    assert opened[0].writes_after_failure == 0
    assert opened[0].closed
    assert REPLY_TEXT not in _only_record(tmp_path)


# ---------------------------------------------------------------------------
# DoD-3 — write failure while the call's error is recorded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_from", ["error_recording", "first_write"])
async def test_openai_record_write_failure_keeps_the_call_error__DoD3(monkeypatch, tmp_path, fail_from):
    """DoD-3: a failing call whose error cannot be recorded raises its own error (type and text); file closed."""
    _enable_tracing(monkeypatch, tmp_path)
    opened = _install_flaky_open(monkeypatch, fail_on="" if fail_from == "first_write" else None)
    client = _openai()

    def break_record_writes() -> None:
        for record in opened:
            record.fail_now = True

    _fake_failure(monkeypatch, client, before=break_record_writes)

    with pytest.raises(LLMError) as excinfo:
        await client.chat(USER_MESSAGES, system=INSTRUCTION)
    assert type(excinfo.value) is LLMError
    assert str(excinfo.value) == str(_call_error())
    assert len(opened) == 1
    assert opened[0].broken, "the write failure was not exercised"
    assert opened[0].closed


# ---------------------------------------------------------------------------
# DoD-4 — only the footer write fails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_footer_write_failure_returns_result__DoD4(monkeypatch, tmp_path):
    """DoD-4: a failing footer write leaves a successful call's own result; the file is closed."""
    _enable_tracing(monkeypatch, tmp_path)
    opened = _install_flaky_open(monkeypatch, fail_on="END TRACE")
    client = _openai()
    _fake_replies(monkeypatch, client, [_reply(REPLY_TEXT)])

    assert await client.chat(USER_MESSAGES, system=INSTRUCTION) == REPLY_TEXT
    assert len(opened) == 1
    assert opened[0].broken, "the footer write was not attempted"
    assert opened[0].closed


@pytest.mark.asyncio
async def test_openai_footer_write_failure_keeps_the_call_error__DoD4(monkeypatch, tmp_path):
    """DoD-4: a failing footer write leaves a failing call's own error (type and text); the file is closed."""
    _enable_tracing(monkeypatch, tmp_path)
    opened = _install_flaky_open(monkeypatch, fail_on="END TRACE")
    client = _openai()
    _fake_failure(monkeypatch, client)

    with pytest.raises(LLMError) as excinfo:
        await client.chat(USER_MESSAGES, system=INSTRUCTION)
    assert type(excinfo.value) is LLMError
    assert str(excinfo.value) == str(_call_error())
    assert len(opened) == 1
    assert opened[0].broken, "the footer write was not attempted"
    assert opened[0].closed


# ---------------------------------------------------------------------------
# DoD-5 — failing call with working writes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_failing_call_record_ends_with_error_then_end_marker__DoD5(monkeypatch, tmp_path):
    """DoD-5: the record ends with the ERROR section then END TRACE, and the error matches the untraced call's."""
    _enable_tracing(monkeypatch, tmp_path)
    traced = _openai()
    _fake_failure(monkeypatch, traced)
    with pytest.raises(LLMError) as traced_exc:
        await traced.chat(USER_MESSAGES, system=INSTRUCTION)

    monkeypatch.delenv("LLM_TRACE")
    monkeypatch.delenv("LLM_TRACE_FOLDER")
    untraced = _openai()
    _fake_failure(monkeypatch, untraced)
    with pytest.raises(LLMError) as untraced_exc:
        await untraced.chat(USER_MESSAGES, system=INSTRUCTION)

    assert type(traced_exc.value) is type(untraced_exc.value)
    assert str(traced_exc.value) == str(untraced_exc.value)

    sections = _sections(_only_record(tmp_path))
    assert [title for title, _ in sections[-2:]] == ["ERROR", "END TRACE"]
    assert [line.strip() for line in sections[-2][1]] == [f"LLMError: {_call_error()}"]
    assert sections[-1][1] == []


def test_tracer_body_error_is_recorded_and_reraised_unchanged__DoD5(monkeypatch, tmp_path):
    """DoD-5 / DEC-002-11: an ``OSError`` from the body is no setup failure; it is recorded and re-raised."""
    _enable_tracing(monkeypatch, tmp_path)
    tracer = LLMTracer()
    body_error = OSError("body failure")

    with pytest.raises(OSError) as excinfo, tracer.trace("chat", MODEL) as t:
        t.log_response("partial")
        raise body_error

    assert excinfo.value is body_error
    sections = _sections(_only_record(tmp_path))
    assert [title for title, _ in sections[-2:]] == ["ERROR", "END TRACE"]
    assert [line.strip() for line in sections[-2][1]] == ["OSError: body failure"]


# ---------------------------------------------------------------------------
# DoD-6 — cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_cancelled_call_propagates_and_record_has_no_error__DoD6(monkeypatch, tmp_path):
    """DoD-6: a cancelled traced call re-raises the cancellation; the record has no ERROR and ends with END TRACE."""
    _enable_tracing(monkeypatch, tmp_path)
    client = _openai()
    cancellation = asyncio.CancelledError()

    async def cancelled_request(method, path, **kw):
        raise cancellation

    monkeypatch.setattr(client, "_request", cancelled_request)

    with pytest.raises(asyncio.CancelledError) as excinfo:
        await client.chat(USER_MESSAGES, system=INSTRUCTION)

    assert excinfo.value is cancellation
    titles = _titles(_only_record(tmp_path))
    assert "ERROR" not in titles
    assert titles[-1] == "END TRACE"


# ---------------------------------------------------------------------------
# DoD-7 — overlapping sessions at the same instant
# ---------------------------------------------------------------------------


def _assert_own_records(root: Path, labels: list[str], expected_names: set[str]) -> None:
    """Each label has exactly one complete record (header, its own content, END TRACE) and nothing of the others."""
    records = _records(root)
    assert {path.name for path in records} == expected_names
    assert all(path.parent == root / DATE_FOLDER for path in records)
    owners = []
    for path in records:
        text = path.read_text(encoding="utf-8")
        header = _header(text)
        owner = [label for label in labels if label in header]
        assert len(owner) == 1, header
        label = owner[0]
        owners.append(label)
        assert MODEL in header
        assert FIXED_NOW.isoformat() in header
        responses = [line for body in _bodies(text, "RESPONSE") for line in body]
        assert responses == [f"first-{label}", f"second-{label}"]
        for other in labels:
            if other != label:
                assert f"-{other}" not in text
        assert _titles(text)[-1] == "END TRACE"
    assert sorted(owners) == sorted(labels)


def test_tracer_nested_sessions_same_instant_get_own_records__DoD7(monkeypatch, tmp_path):
    """DoD-7: nested sessions at one frozen instant write two distinct complete records (base name, then ``-1``)."""
    _freeze_clock(monkeypatch)
    _enable_tracing(monkeypatch, tmp_path)
    tracer = LLMTracer()

    with tracer.trace("op-outer", MODEL) as outer:
        outer.log_response("first-op-outer")
        with tracer.trace("op-inner", MODEL) as inner:
            inner.log_response("first-op-inner")
            inner.log_response("second-op-inner")
        outer.log_response("second-op-outer")

    _assert_own_records(tmp_path, ["op-outer", "op-inner"], {f"{BASE_NAME}.log", f"{BASE_NAME}-1.log"})


def test_tracer_three_nested_sessions_use_increasing_suffixes__DoD7(monkeypatch, tmp_path):
    """DoD-7 / DEC-002-13: a third session at the same instant takes the ``-2`` suffix."""
    _freeze_clock(monkeypatch)
    _enable_tracing(monkeypatch, tmp_path)
    tracer = LLMTracer()

    with tracer.trace("op-one", MODEL) as one:
        one.log_response("first-op-one")
        with tracer.trace("op-two", MODEL) as two:
            two.log_response("first-op-two")
            with tracer.trace("op-three", MODEL) as three:
                three.log_response("first-op-three")
                three.log_response("second-op-three")
            two.log_response("second-op-two")
        one.log_response("second-op-one")

    _assert_own_records(
        tmp_path,
        ["op-one", "op-two", "op-three"],
        {f"{BASE_NAME}.log", f"{BASE_NAME}-1.log", f"{BASE_NAME}-2.log"},
    )


@pytest.mark.asyncio
async def test_tracer_concurrent_sessions_same_instant_get_own_records__DoD7(monkeypatch, tmp_path):
    """DoD-7: two concurrent sessions at one frozen instant write two distinct complete records."""
    _freeze_clock(monkeypatch)
    _enable_tracing(monkeypatch, tmp_path)
    tracer = LLMTracer()
    # Both sessions wait on this event while open, so their lifetimes are guaranteed to overlap.
    both_open = asyncio.Event()
    entered: list[str] = []

    async def traced(label: str) -> None:
        with tracer.trace(label, MODEL) as t:
            t.log_response(f"first-{label}")
            entered.append(label)
            if len(entered) == 2:
                both_open.set()
            await both_open.wait()
            t.log_response(f"second-{label}")

    await asyncio.wait_for(asyncio.gather(traced("op-north"), traced("op-south")), timeout=5)

    _assert_own_records(tmp_path, ["op-north", "op-south"], {f"{BASE_NAME}.log", f"{BASE_NAME}-1.log"})


def test_tracer_existing_record_name_is_never_overwritten__DoD7(monkeypatch, tmp_path):
    """DoD-7 / DEC-002-13: a record already holding the instant's name is kept; the new record takes ``-1``."""
    _freeze_clock(monkeypatch)
    _enable_tracing(monkeypatch, tmp_path)
    date_dir = tmp_path / DATE_FOLDER
    date_dir.mkdir()
    existing = date_dir / f"{BASE_NAME}.log"
    existing.write_text("someone else's record", encoding="utf-8")
    tracer = LLMTracer()

    with tracer.trace("op-late", MODEL) as t:
        t.log_response("first-op-late")
        t.log_response("second-op-late")

    assert existing.read_text(encoding="utf-8") == "someone else's record"
    existing.unlink()
    _assert_own_records(tmp_path, ["op-late"], {f"{BASE_NAME}-1.log"})


def test_tracer_last_suffix_within_bound_is_used__DoD7(monkeypatch, tmp_path):
    """DoD-7 / DEC-002-13: with the suffix bound lowered to 2, ``-2`` is still a usable name."""
    _freeze_clock(monkeypatch)
    _enable_tracing(monkeypatch, tmp_path)
    monkeypatch.setattr(llm.trace, "_MAX_COLLISION_SUFFIX", 2)
    date_dir = tmp_path / DATE_FOLDER
    date_dir.mkdir()
    taken = [date_dir / f"{BASE_NAME}.log", date_dir / f"{BASE_NAME}-1.log"]
    for path in taken:
        path.write_text("someone else's record", encoding="utf-8")
    tracer = LLMTracer()

    with tracer.trace("op-last", MODEL) as t:
        t.log_response("first-op-last")
        t.log_response("second-op-last")

    for path in taken:
        assert path.read_text(encoding="utf-8") == "someone else's record"
        path.unlink()
    _assert_own_records(tmp_path, ["op-last"], {f"{BASE_NAME}-2.log"})


# ---------------------------------------------------------------------------
# DoD-8 — layout without collision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_record_layout_without_collision__DoD8(monkeypatch, tmp_path):
    """DoD-8: one record under the UTC date folder, named with the timestamp and no suffix."""
    _freeze_clock(monkeypatch)
    _enable_tracing(monkeypatch, tmp_path)
    client = _openai()
    _fake_replies(monkeypatch, client, [_reply(REPLY_TEXT)])

    assert await client.chat(USER_MESSAGES) == REPLY_TEXT
    assert _records(tmp_path) == [tmp_path / DATE_FOLDER / f"{BASE_NAME}.log"]


def test_tracer_record_layout_with_real_clock__DoD8(monkeypatch, tmp_path):
    """DoD-8: with the real clock the record sits in today's UTC date folder with a matching timestamp name."""
    _enable_tracing(monkeypatch, tmp_path)
    tracer = LLMTracer()
    before = datetime.now(UTC).strftime("%Y-%m-%d")

    with tracer.trace("chat", MODEL) as t:
        t.log_response(REPLY_TEXT)

    after = datetime.now(UTC).strftime("%Y-%m-%d")
    records = _records(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record.parent.parent == tmp_path
    folder = record.parent.name
    assert folder in {before, after}
    assert re.fullmatch(r"llm-trace-\d{8}-\d{12}\.log", record.name)
    assert record.name[len("llm-trace-") : len("llm-trace-") + 8] == folder.replace("-", "")


# ---------------------------------------------------------------------------
# DoD-9 — tracing off or no folder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("flag", "folder_mode"),
    [(None, "set"), ("0", "set"), ("off", "set"), ("1", "unset"), ("1", "empty")],
)
async def test_openai_no_record_when_tracing_off_or_no_folder__DoD9(monkeypatch, tmp_path, flag, folder_mode):
    """DoD-9: with tracing off, or on without a folder, neither a client call nor a direct scope writes a record."""
    # Run from tmp_path so a record written relative to the working directory would also be found.
    monkeypatch.chdir(tmp_path)
    if flag is None:
        monkeypatch.delenv("LLM_TRACE", raising=False)
    else:
        monkeypatch.setenv("LLM_TRACE", flag)
    if folder_mode == "set":
        monkeypatch.setenv("LLM_TRACE_FOLDER", str(tmp_path / "traces"))
    elif folder_mode == "empty":
        monkeypatch.setenv("LLM_TRACE_FOLDER", "")
    else:
        monkeypatch.delenv("LLM_TRACE_FOLDER", raising=False)

    tracer = LLMTracer()
    assert tracer.enabled is False
    with tracer.trace("chat", MODEL) as t:
        t.log_response(REPLY_TEXT)

    client = _openai()
    _fake_replies(monkeypatch, client, [_reply(REPLY_TEXT)])
    assert await client.chat(USER_MESSAGES, system=INSTRUCTION) == REPLY_TEXT

    assert _records(tmp_path) == []


# ---------------------------------------------------------------------------
# DoD-10 / DoD-11 — the instruction appears once among the conversation sections
# ---------------------------------------------------------------------------


async def _traced_chat(monkeypatch, tmp_path, client: Any, messages: list[dict]) -> tuple[str, list[dict]]:
    sent = _fake_replies(monkeypatch, client, [_reply(REPLY_TEXT)])
    assert await client.chat(messages, system=INSTRUCTION) == REPLY_TEXT
    return _only_record(tmp_path), sent


def _assert_instruction_once(text: str, caller_lines: list[str]) -> None:
    """The instruction is in SYSTEM PROMPT, not repeated in MESSAGE HISTORY, and the caller's own messages are."""
    system_bodies = _bodies(text, "SYSTEM PROMPT")
    assert len(system_bodies) == 1
    assert INSTRUCTION in system_bodies[0]
    history = _history(text)
    assert not any(INSTRUCTION in line for line in history), history
    assert history == caller_lines


@pytest.mark.asyncio
async def test_openai_chat_instruction_once_with_callers_system_message__DoD10(monkeypatch, tmp_path):
    """DoD-10: OpenAI chat shows the instruction in SYSTEM PROMPT only; all caller messages in history."""
    _enable_tracing(monkeypatch, tmp_path)
    messages = [{"role": "system", "content": CALLER_SYSTEM}, {"role": "user", "content": USER_TEXT}]
    text, _ = await _traced_chat(monkeypatch, tmp_path, _openai(), messages)
    _assert_instruction_once(text, [f"[system] {CALLER_SYSTEM}", f"[user] {USER_TEXT}"])


@pytest.mark.asyncio
async def test_llamaswap_chat_instruction_once_with_callers_system_message__DoD10(monkeypatch, tmp_path):
    """DoD-10: llama-swap chat shows the instruction in SYSTEM PROMPT only; all caller messages in history."""
    _enable_tracing(monkeypatch, tmp_path)
    messages = [{"role": "system", "content": CALLER_SYSTEM}, {"role": "user", "content": USER_TEXT}]
    text, _ = await _traced_chat(monkeypatch, tmp_path, _llamaswap(), messages)
    _assert_instruction_once(text, [f"[system] {CALLER_SYSTEM}", f"[user] {USER_TEXT}"])


@pytest.mark.asyncio
async def test_openai_chat_callers_system_message_equal_to_instruction_kept__DoD10(monkeypatch, tmp_path):
    """DoD-10 / DEC-002-14: only the inserted copy is dropped; the caller's own identical system message stays."""
    _enable_tracing(monkeypatch, tmp_path)
    messages = [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": USER_TEXT}]
    text, _ = await _traced_chat(monkeypatch, tmp_path, _openai(), messages)
    assert _history(text) == [f"[system] {INSTRUCTION}", f"[user] {USER_TEXT}"]


@pytest.mark.asyncio
async def test_llamaswap_chat_callers_system_message_equal_to_instruction_kept__DoD10(monkeypatch, tmp_path):
    """DoD-10 / DEC-002-14: on llama-swap too, the caller's own identical system message stays in history."""
    _enable_tracing(monkeypatch, tmp_path)
    messages = [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": USER_TEXT}]
    text, _ = await _traced_chat(monkeypatch, tmp_path, _llamaswap(), messages)
    assert _history(text) == [f"[system] {INSTRUCTION}", f"[user] {USER_TEXT}"]


async def _assert_chat_json_history(monkeypatch, tmp_path, client: Any) -> None:
    _fake_replies(monkeypatch, client, [_reply('{"answer": "midi"}')])
    assert await client.chat_json(USER_MESSAGES, system=INSTRUCTION) == '{"answer": "midi"}'
    history = _history(_only_record(tmp_path))
    assert not any(INSTRUCTION in line for line in history), history
    assert f"[user] {USER_TEXT}" in history


async def _assert_chat_with_tools_history(monkeypatch, tmp_path, client: Any) -> None:
    _fake_replies(monkeypatch, client, [_reply("", TOOL_CALLS), _reply(REPLY_TEXT)])
    result = await client.chat_with_tools(USER_MESSAGES, tools_definitions=TOOL_DEFS, tools=TOOLS, system=INSTRUCTION)
    assert result == REPLY_TEXT
    history = _history(_only_record(tmp_path))
    assert not any(INSTRUCTION in line for line in history), history
    assert f"[user] {USER_TEXT}" in history


@pytest.mark.asyncio
async def test_openai_chat_json_history_does_not_repeat_instruction__DoD11(monkeypatch, tmp_path):
    """DoD-11: OpenAI ``chat_json`` history does not repeat the instruction."""
    _enable_tracing(monkeypatch, tmp_path)
    await _assert_chat_json_history(monkeypatch, tmp_path, _openai())


@pytest.mark.asyncio
async def test_llamaswap_chat_json_history_does_not_repeat_instruction__DoD11(monkeypatch, tmp_path):
    """DoD-11: llama-swap ``chat_json`` history does not repeat the instruction."""
    _enable_tracing(monkeypatch, tmp_path)
    await _assert_chat_json_history(monkeypatch, tmp_path, _llamaswap())


@pytest.mark.asyncio
async def test_openai_chat_with_tools_history_does_not_repeat_instruction__DoD11(monkeypatch, tmp_path):
    """DoD-11: OpenAI ``chat_with_tools`` history does not repeat the instruction."""
    _enable_tracing(monkeypatch, tmp_path)
    await _assert_chat_with_tools_history(monkeypatch, tmp_path, _openai())


@pytest.mark.asyncio
async def test_llamaswap_chat_with_tools_history_does_not_repeat_instruction__DoD11(monkeypatch, tmp_path):
    """DoD-11: llama-swap ``chat_with_tools`` history does not repeat the instruction."""
    _enable_tracing(monkeypatch, tmp_path)
    await _assert_chat_with_tools_history(monkeypatch, tmp_path, _llamaswap())


# ---------------------------------------------------------------------------
# DoD-12 — no instruction: a caller's leading system message is written
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("system", [None, ""])
async def test_openai_chat_without_instruction_keeps_leading_system_message__DoD12(monkeypatch, tmp_path, system):
    """DoD-12: without an instruction (absent or empty) the caller's leading system message is in history."""
    _enable_tracing(monkeypatch, tmp_path)
    client = _openai()
    _fake_replies(monkeypatch, client, [_reply(REPLY_TEXT)])
    messages = [{"role": "system", "content": CALLER_SYSTEM}, {"role": "user", "content": USER_TEXT}]

    assert await client.chat(messages, system=system) == REPLY_TEXT
    assert _history(_only_record(tmp_path)) == [f"[system] {CALLER_SYSTEM}", f"[user] {USER_TEXT}"]


@pytest.mark.parametrize("system", [None, ""])
def test_tracer_session_without_instruction_writes_leading_system_message__DoD12(monkeypatch, tmp_path, system):
    """DoD-12 / DEC-002-14: an empty or absent logged instruction never drops a leading system message."""
    _enable_tracing(monkeypatch, tmp_path)
    tracer = LLMTracer()
    leading = "" if system == "" else CALLER_SYSTEM

    with tracer.trace("chat", MODEL) as t:
        t.log_system(system)
        t.log_messages([{"role": "system", "content": leading}, {"role": "user", "content": USER_TEXT}])

    history = _history(_only_record(tmp_path))
    assert history == [f"[system] {leading}".strip(), f"[user] {USER_TEXT}"]


def test_tracer_session_writes_leading_system_message_with_other_text__DoD12(monkeypatch, tmp_path):
    """DoD-12 / DEC-002-14: a leading system message whose text differs from the instruction is written."""
    _enable_tracing(monkeypatch, tmp_path)
    tracer = LLMTracer()

    with tracer.trace("chat", MODEL) as t:
        t.log_system(INSTRUCTION)
        t.log_messages([{"role": "system", "content": CALLER_SYSTEM}, {"role": "user", "content": USER_TEXT}])

    assert _history(_only_record(tmp_path)) == [f"[system] {CALLER_SYSTEM}", f"[user] {USER_TEXT}"]


# ---------------------------------------------------------------------------
# DoD-13 — REQUEST shows the payload as sent
# ---------------------------------------------------------------------------


def _assert_request_as_sent(text: str, sent: dict) -> None:
    """The REQUEST section is the method/path line followed by the sent payload as JSON, instruction included."""
    bodies = _bodies(text, "REQUEST")
    assert len(bodies) == 1
    body = bodies[0]
    assert body[0].startswith("POST")
    payload = json.loads("\n".join(body[1:]))
    assert payload == sent
    assert payload["messages"][0] == {"role": "system", "content": INSTRUCTION}


@pytest.mark.asyncio
async def test_openai_request_section_shows_instruction_as_sent__DoD13(monkeypatch, tmp_path):
    """DoD-13: on OpenAI the REQUEST section still holds the payload as sent, with the instruction message."""
    _enable_tracing(monkeypatch, tmp_path)
    text, sent = await _traced_chat(monkeypatch, tmp_path, _openai(), USER_MESSAGES)
    assert len(sent) == 1
    _assert_request_as_sent(text, sent[0])


@pytest.mark.asyncio
async def test_llamaswap_request_section_shows_instruction_as_sent__DoD13(monkeypatch, tmp_path):
    """DoD-13: on llama-swap the REQUEST section still holds the payload as sent, with the instruction message."""
    _enable_tracing(monkeypatch, tmp_path)
    text, sent = await _traced_chat(monkeypatch, tmp_path, _llamaswap(), USER_MESSAGES)
    assert len(sent) == 1
    _assert_request_as_sent(text, sent[0])


# ---------------------------------------------------------------------------
# DoD-14 — direct use with an app's own operation label
# ---------------------------------------------------------------------------


def test_tracer_direct_use_with_own_operation_label__DoD14(monkeypatch, tmp_path):
    """DoD-14: an app's own label yields a record of the same kind: layout, header, content, END TRACE."""
    _freeze_clock(monkeypatch)
    _enable_tracing(monkeypatch, tmp_path)
    tracer = LLMTracer()
    assert tracer.enabled is True

    with tracer.trace("nightly-digest", MODEL) as t:
        t.log_system(INSTRUCTION)
        t.log_messages(USER_MESSAGES)
        t.log_response(REPLY_TEXT)

    assert _records(tmp_path) == [tmp_path / DATE_FOLDER / f"{BASE_NAME}.log"]
    text = _only_record(tmp_path)
    header = _header(text)
    assert "nightly-digest" in header
    assert MODEL in header
    assert FIXED_NOW.isoformat() in header
    assert _bodies(text, "SYSTEM PROMPT") == [[INSTRUCTION]]
    assert _history(text) == [f"[user] {USER_TEXT}"]
    assert _bodies(text, "RESPONSE") == [[REPLY_TEXT]]
    assert _titles(text)[-1] == "END TRACE"
