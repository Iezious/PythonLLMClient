"""LLM interaction trace logging.

When both LLM_TRACE (1/true/on) and LLM_TRACE_FOLDER are set,
every LLM call writes a human-readable log file capturing the full
request/response cycle.

File layout:  <LLM_TRACE_FOLDER>/<YYYY-MM-DD>/llm-trace-<YYYYMMDD-HHMMSSffffff>.log
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

_SEPARATOR = "=" * 60
_SECTION = "-" * 40


class TraceSession:
    """Writes human-readable sections into an open trace file."""

    def __init__(self, f, model: str, operation: str, timestamp: datetime) -> None:
        self._f = f
        self._write(_SEPARATOR)
        self._write(f"LLM TRACE: {operation} | model: {model} | {timestamp.isoformat()}")
        self._write(_SEPARATOR)
        self._write("")

    def _write(self, text: str) -> None:
        self._f.write(text + "\n")
        self._f.flush()

    def _section(self, title: str) -> None:
        self._write(f"\n{_SECTION}")
        self._write(f"  {title}")
        self._write(_SECTION)

    def log_system(self, system: Optional[str]) -> None:
        if system is None:
            return
        self._section("SYSTEM PROMPT")
        self._write(system)

    def log_messages(self, messages: List[Dict[str, Any]]) -> None:
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
        self._section("ERROR")
        self._write(f"{type(error).__name__}: {error}")

    def close(self) -> None:
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
        if not self._enabled:
            yield _NOOP_SESSION  # type: ignore[misc]
            return

        now = datetime.now(timezone.utc)
        date_dir = Path(self._folder) / now.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        filename = f"llm-trace-{now.strftime('%Y%m%d-%H%M%S%f')}.log"
        filepath = date_dir / filename

        f = open(filepath, "w", encoding="utf-8")
        session = TraceSession(f, model, operation, now)
        try:
            yield session
        except Exception as exc:
            session.log_error(exc)
            raise
        finally:
            session.close()
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
