"""LLM interaction trace logging.

When both LLM_TRACE (1/true/on) and LLM_TRACE_FOLDER are set,
every LLM call writes a human-readable log file capturing the full
request/response cycle.

File layout:  <LLM_TRACE_FOLDER>/<YYYY-MM-DD>/llm-trace-<YYYYMMDD-HHMMSSffffff>.log
A record whose name is already taken gains a counter suffix before the extension:
``llm-trace-<timestamp>-1.log``, ``-2``, ... (DEC-002-13).

Tracing never changes a call's outcome: a record that cannot be set up leaves the call
untraced (DEC-002-11), and a record-write failure stops the record, never the call (DEC-002-12).

Test seams (DEC-002-16): the session time is taken through the module-level ``datetime``
name (``datetime.now(timezone.utc)``), and the record file is opened with the builtin
``open`` looked up through this module's globals (not ``Path.open`` / ``os.open``), so tests
may replace ``llm.trace.datetime`` and ``llm.trace.open``.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

_SEPARATOR = "=" * 60
_SECTION = "-" * 40

_MAX_COLLISION_SUFFIX = 1000
"""Highest counter suffix tried when a record name is taken; exhausting it is a setup failure (DEC-002-13)."""


class TraceSession:
    """Writes human-readable sections into an open trace file.

    Invariants:
    - The first write failure (any ``Exception``) marks the session broken; every later
      write is skipped quietly. The header written by the constructor is covered too, so
      constructing a session never raises because of a write failure (DEC-002-12).
    - The session remembers the instruction passed to ``log_system`` so that
      ``log_messages`` can drop the copy the clients insert at index 0 (DEC-002-14).
    - The session never closes the file; ``LLMTracer.trace`` owns the file's lifetime.
    """

    def __init__(self, f, model: str, operation: str, timestamp: datetime) -> None:
        """Bind the session to an open text file and write the record header.

        Args:
            f: Open, writable text file object the record is written to.
            model: Model name shown in the header.
            operation: Operation label shown in the header (any label, e.g. ``"chat"``).
            timestamp: Session start time (aware UTC) shown in ISO form in the header.

        A header write failure marks the session broken instead of raising.
        """
        self._f = f
        # Both must exist before the first header write: ``_write`` reads ``_broken``, and
        # ``log_messages`` reads ``_system`` even when ``log_system`` was never called.
        self._broken = False
        self._system: str | None = None
        self._write(_SEPARATOR)
        self._write(f"LLM TRACE: {operation} | model: {model} | {timestamp.isoformat()}")
        self._write(_SEPARATOR)
        self._write("")

    def _write(self, text: str) -> None:
        """Write one line and flush, unless the session is broken.

        On the first write or flush failure (any ``Exception``) the session is marked
        broken and the failure is swallowed; every later call writes nothing (DEC-002-12).
        """
        if self._broken:
            return
        # Mark broken first and clear it only once write and flush both succeed, so any
        # Exception escaping into ``suppress`` leaves the session broken. A record is a
        # debugging aid: after the first failure it is abandoned and the call goes on.
        self._broken = True
        with suppress(Exception):
            self._f.write(text + "\n")
            self._f.flush()
            self._broken = False

    def _section(self, title: str) -> None:
        self._write(f"\n{_SECTION}")
        self._write(f"  {title}")
        self._write(_SECTION)

    def log_system(self, system: Optional[str]) -> None:
        """Write the SYSTEM PROMPT section and remember the instruction for ``log_messages``.

        Args:
            system: The call's system instruction; ``None`` writes nothing. An empty string
                still writes an empty section, but is not used to drop a history message.
        """
        if system is None:
            return
        self._system = system
        self._section("SYSTEM PROMPT")
        self._write(system)

    def log_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Write the MESSAGE HISTORY section: ``[role] content`` lines plus tool-call lines.

        When this session logged a non-empty instruction through ``log_system``, the first
        message is skipped only if its role is ``"system"`` and its content equals that
        instruction exactly (the copy the clients insert). Every other message, including a
        caller's own system message, is written (DEC-002-14). An empty list writes nothing.

        Args:
            messages: The messages as sent, in order.
        """
        if not messages:
            return
        first = messages[0]
        # Only an exact, leading system copy of a non-empty instruction is the clients' insert
        # (they prepend it only when the instruction is truthy); anything else is the caller's.
        if self._system and first.get("role") == "system" and first.get("content") == self._system:
            messages = messages[1:]
            # Nothing of the caller's remains, so there is no history to show.
            if not messages:
                return
        self._section("MESSAGE HISTORY")
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            self._write(f"[{role}] {content}")
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    self._write(f"  -> tool_call: {func.get('name', '?')}({func.get('arguments', '')})")

    def log_request(self, method: str, path: str, payload: Dict[str, Any]) -> None:
        self._section("REQUEST")
        self._write(f"{method} {path}")
        self._write(_format_json(payload))

    def log_response(self, content: str) -> None:
        self._section("RESPONSE")
        self._write(content)

    def log_chunk(self, chunk_text: str) -> None:
        self._section("CHUNK")
        self._write(chunk_text)

    def log_tool_call(self, name: str, arguments: Any) -> None:
        self._section(f"TOOL CALL: {name}")
        if isinstance(arguments, str):
            self._write(arguments)
        else:
            self._write(_format_json(arguments))

    def log_tool_result(self, name: str, result: str) -> None:
        self._section(f"TOOL RESULT: {name}")
        self._write(result)

    def log_error(self, error: Exception) -> None:
        """Write the ERROR section as ``<TypeName>: <text>``; never raises on write failure."""
        self._section("ERROR")
        self._write(f"{type(error).__name__}: {error}")

    def close(self) -> None:
        """Write the END TRACE footer (does not close the file); never raises on write failure."""
        self._write(f"\n{_SEPARATOR}")
        self._write("END TRACE")
        self._write(_SEPARATOR)


class LLMTracer:
    """Checks env vars and creates trace sessions for LLM interactions."""

    def __init__(self) -> None:
        trace_flag = os.environ.get("LLM_TRACE", "").strip().lower()
        self._folder = os.environ.get("LLM_TRACE_FOLDER", "").strip()
        self._enabled = trace_flag in ("1", "true", "on") and bool(self._folder)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def trace(self, operation: str, model: str) -> Generator[TraceSession, None, None]:
        """Open a trace record for one call and yield its session.

        Args:
            operation: Operation label written in the header (an app may use its own).
            model: Model name written in the header.

        Yields:
            A ``TraceSession`` writing to a new record, or the shared no-op session when
            tracing is disabled or the record cannot be set up.

        Contract:
        - Setup: the date folder ``<folder>/<YYYY-MM-DD>`` (UTC) is created and the record is
          opened with exclusive create (``open(filepath, "x", encoding="utf-8")``). On
          ``FileExistsError`` from that open, the name gains a counter suffix ``-1``, ``-2``,
          ... up to ``_MAX_COLLISION_SUFFIX`` attempts (DEC-002-13). Any ``OSError`` while
          creating the folder (including the folder path naming a file) or opening the
          record, or exhausting the suffixes, yields the no-op session exactly once and the
          call runs untraced, with nothing reported (DEC-002-11).
        - Only setup is guarded: an exception from the caller's body is never taken as a
          setup failure and propagates unchanged on both branches.
        - Teardown: an ``Exception`` from the body is recorded as the ERROR section (guarded)
          and re-raised unchanged; the footer write is guarded; the file is closed in every
          case, its close guarded too. Nothing raised by recording replaces the caller's
          result or error (DEC-002-12). A ``BaseException`` such as ``CancelledError`` writes
          no ERROR section, the footer is still attempted, the file closed, and it propagates.
        """
        if not self._enabled:
            yield _NOOP_SESSION  # type: ignore[misc]
            return

        now = datetime.now(timezone.utc)
        f = None
        # Setup only is guarded (DEC-002-11); the yields stay outside this try so an OSError
        # raised by the caller's body is never mistaken for a setup failure.
        try:
            date_dir = Path(self._folder) / now.strftime("%Y-%m-%d")
            # A date-folder path naming a file raises FileExistsError here: that is a setup
            # failure (caught as OSError below), not a name collision.
            date_dir.mkdir(parents=True, exist_ok=True)
            stem = f"llm-trace-{now.strftime('%Y%m%d-%H%M%S%f')}"
            # Attempt 0 keeps the documented unsuffixed name; then -1 .. -_MAX_COLLISION_SUFFIX.
            # Exclusive create makes the name check atomic, so overlapping sessions of one
            # instant never share or overwrite a record (DEC-002-13).
            for suffix in range(_MAX_COLLISION_SUFFIX + 1):
                filepath = date_dir / (f"{stem}.log" if suffix == 0 else f"{stem}-{suffix}.log")
                try:
                    f = open(filepath, "x", encoding="utf-8")
                except FileExistsError:
                    continue
                break
        except OSError:
            f = None

        if f is None:
            # Folder/open failure or every suffix taken: run untraced, report nothing.
            yield _NOOP_SESSION  # type: ignore[misc]
            return

        session = TraceSession(f, model, operation, now)
        try:
            yield session
        except Exception as exc:
            # Recording the error must never replace it (DEC-002-12); BaseException such as
            # CancelledError bypasses this branch on purpose, so it gets no ERROR section.
            with suppress(Exception):
                session.log_error(exc)
            raise
        finally:
            # Each teardown action is guarded separately so a failing footer cannot skip the
            # close, and neither can replace the caller's result or error.
            with suppress(Exception):
                session.close()
            with suppress(Exception):
                f.close()


class _NoopSession:
    """Placeholder that silently discards all trace calls (zero overhead)."""

    def log_system(self, *a: Any, **kw: Any) -> None: pass
    def log_messages(self, *a: Any, **kw: Any) -> None: pass
    def log_request(self, *a: Any, **kw: Any) -> None: pass
    def log_response(self, *a: Any, **kw: Any) -> None: pass
    def log_chunk(self, *a: Any, **kw: Any) -> None: pass
    def log_tool_call(self, *a: Any, **kw: Any) -> None: pass
    def log_tool_result(self, *a: Any, **kw: Any) -> None: pass
    def log_error(self, *a: Any, **kw: Any) -> None: pass
    def close(self) -> None: pass


_NOOP_SESSION = _NoopSession()


def _format_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(obj)
