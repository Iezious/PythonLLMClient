"""OpenAI Chat Completions client (``OpenAIChatCompletionsClient``): the app's credential, shared helpers, settings.

A renamed copy of the pre-Responses ``llm/openai.py`` (feature 003, DEC-003-7): same behaviour,
tables and helpers; it changes only for chat-completions fixes, never for Responses work (DEC-arch-2).
History carried over from the source module:

Fixes from feature 001: credential only from ``bearer_token``, streamed requests via ``_stream_post``,
null reply content as empty text, and only app-set settings sent.
Feature 002: the tool loop maps an unoffered tool, unreadable arguments and an
unserialisable result to ``LLMError`` and names the round limit in its error (DEC-002-2..5).
Feature 002: embeddings and model-list replies are checked, embeddings ordered by
position, and unreadable bodies mapped to ``LLMError`` (DEC-002-8..10);
``chat_json`` checks only the JSON parse (DEC-002-7).
Feature 006: ``chat_with_tools`` tells the model about tool failures by default (strict opt-out
``tool_errors="raise"``) and reports each call to an optional tool-event observer (``_run_tool_call``).
"""

from collections.abc import Mapping, Sequence
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional
import json
import inspect
import warnings

import aiohttp

from llm.llm_client import LLMClient
from llm.exceptions import LLMError, LLMToolObserverWarning
from llm.message import LLMMessage
from llm.options import OpenAIOptions
from llm.tool_events import ToolErrorKind, ToolEvent
from llm.trace import TraceSession

# Settings the OpenAI client accepts. Every value is None: the table names the keys, it supplies no
# built-in value, so only settings the app set reach the server (DEC-001-13, DEC-001-14).
_OPENAI_SUPPORTED_PARAMS: Dict[str, Any] = {
    "temperature": None,
    "top_p": None,
    "max_tokens": None,
    "presence_penalty": None,
    "frequency_penalty": None,
    "seed": None,
    "enable_thinking": None,
    "reasoning_effort": None,
    "stop": None,
    "logit_bias": None,
    "user": None,
    "response_format": None,
}

_OPENAI_ALIAS_MAP = {
    "context_length": "max_tokens", # OpenAI uses max_tokens for context length
}

_ToolEventObserver = Callable[[ToolEvent], Optional[Awaitable[None]]]
"""This class's own type of the ``on_tool_event`` observer (DEC-arch-15: not imported from another client module).

A plain function returning ``None`` or a coroutine function; the loop awaits the returned value when it is
awaitable, inline, before going on (DEC-arch-18).
"""

def _offered_tool_name(call: Any, offered: set[str]) -> str:
    """Return the tool name of one requested call after checking it was offered.

    Args:
        call: One entry of the reply's ``tool_calls``; expected shape ``{"function": {"name": ...}}``.
        offered: Names advertised in ``tools_definitions``.

    Returns:
        The requested name, guaranteed to be in ``offered``.

    Raises:
        LLMError: The name is missing or empty (also when the call or its ``function`` is not a
            mapping), or it is not advertised, even when ``tools`` has a callable for it (DEC-002-3).
            ``kind="tool"`` (DEC-arch-19); ``_run_tool_call`` maps it to ``not_offered``.
    """
    function = call.get("function") if isinstance(call, Mapping) else None
    name = function.get("name") if isinstance(function, Mapping) else None
    # A malformed call is reported as an empty name rather than a raw KeyError/TypeError.
    if not name:
        raise LLMError("The model requested a tool call with an empty tool name", kind="tool")
    if not isinstance(name, str) or name not in offered:
        raise LLMError(f"The model requested tool '{name}', which was not offered", kind="tool")
    return name


def _read_tool_arguments(name: str, raw: Any) -> dict[str, Any]:
    """Turn a requested call's arguments into the keyword mapping the tool is called with.

    Args:
        name: The (offered) tool name, used in the error message.
        raw: The call's ``arguments`` value as received.

    Returns:
        ``{}`` for missing or empty arguments; a mapping as given; the decoded JSON object for text.

    Raises:
        LLMError: Text that is not JSON (chained from the decode error), JSON that is not an object,
            or a value that is neither text nor a mapping; the description carries the reason
            (DEC-002-4). ``kind="tool"`` (DEC-arch-19); ``_run_tool_call`` maps it to ``bad_arguments``.
    """
    # Only None and "" count as "no arguments"; other falsy values (0, [], False) are unusable
    # forms and fall through to the error below (DEC-002-4).
    if raw is None or raw == "":
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"Arguments for tool '{name}' are not valid JSON", description=str(exc), kind="tool"
            ) from exc
        if not isinstance(decoded, dict):
            raise LLMError(
                f"Arguments for tool '{name}' are not a JSON object",
                description=f"decoded to {type(decoded).__name__}",
                kind="tool",
            )
        return decoded
    raise LLMError(
        f"Arguments for tool '{name}' are not readable",
        description=f"unsupported type {type(raw).__name__}",
        kind="tool",
    )


def _tool_result_text(name: str, result: Any) -> str:
    """Turn a tool's return value into the tool message content.

    Args:
        name: The tool name, used in the error message.
        result: The value the tool returned (already awaited).

    Returns:
        JSON text (non-ASCII kept unescaped) for a ``dict`` or ``list``; ``str(result)`` otherwise.

    Raises:
        LLMError: The ``dict`` or ``list`` cannot be serialised (chained from the error, DEC-002-5).
            ``kind="tool"`` (DEC-arch-19); ``_run_tool_call`` maps it to ``unserialisable_result``.
    """
    # Only dict/list are JSON-encoded, as before; other values (including non-dict mappings) keep
    # their text form so an object with a custom __str__ is still sent as that text.
    if not isinstance(result, (dict, list)):
        return str(result)
    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        # TypeError: a non-serialisable element; ValueError: a circular reference.
        raise LLMError(
            f"Result of tool '{name}' cannot be serialised to JSON", description=str(exc), kind="tool"
        ) from exc


async def _fetch_reply(client: LLMClient[Any], operation: str, method: str, path: str, **kwargs: Any) -> Any:
    """Send one non-streamed request for listing or embeddings and map an undecodable 2xx body.

    Args:
        client: The client whose ``_request`` seam sends the request.
        operation: The operation phrase used in the error message (e.g. ``"embedding"``).
        method: HTTP method.
        path: Path below the client's base URL.
        **kwargs: Passed to ``_request`` unchanged (e.g. ``json=payload``).

    Returns:
        Whatever ``_request`` returned: decoded JSON, or ``bytes`` for a non-JSON body.

    Raises:
        LLMError: The successful body could not be decoded as JSON (chained from the decode error,
            DEC-002-10). An ``LLMError`` raised by ``_request`` (non-2xx, transport) passes through.
            The decode error carries kind ``invalid_reply`` (DEC-arch-19); a pass-through keeps its own.
    """
    try:
        return await client._request(method, path, **kwargs)
    # DEC-002-10: only the two decode errors ``_request`` lets escape on a 2xx JSON body are mapped;
    # a broad ValueError catch could swallow unrelated errors. ``_request`` is not edited.
    except (json.JSONDecodeError, aiohttp.ContentTypeError) as exc:
        raise LLMError(
            f"Unreadable reply from OpenAI API when {operation}",
            description=str(exc),
            kind="invalid_reply",
        ) from exc


def _reply_text(response: Any) -> str:
    """Return a reply as text for an error description; a bytes body is decoded with replacement."""
    if isinstance(response, (bytes, bytearray)):
        return bytes(response).decode("utf-8", errors="replace")
    return str(response)


def _reply_entries(response: Any, operation: str) -> list[Any]:
    """Check that a listing or embeddings reply is a mapping with a list under ``data``.

    Args:
        response: The value ``_request`` returned.
        operation: The operation phrase used in the error message.

    Returns:
        The ``data`` list, unchanged.

    Raises:
        LLMError: The reply is not a mapping (including a bytes body), has no ``data`` (the
            pre-existing message, reply inline), or ``data`` is not a list.
            Every such error carries kind ``invalid_reply`` (DEC-arch-19).
    """
    if not isinstance(response, Mapping):
        raise LLMError(
            f"Invalid response from OpenAI API when {operation}: the reply is not a JSON object",
            description=_reply_text(response),
            kind="invalid_reply",
        )
    if "data" not in response:
        # Wording kept from before DEC-002-8 / DEC-002-9; older tests may match it.
        raise LLMError(f"Invalid response from OpenAI API when {operation}: {response}", kind="invalid_reply")
    entries = response["data"]
    if not isinstance(entries, list):
        raise LLMError(
            f"Invalid response from OpenAI API when {operation}: 'data' is not a list",
            description=_reply_text(response),
            kind="invalid_reply",
        )
    return entries


def _model_entries(response: Any, operation: str) -> list[Any]:
    """Check a model-list reply (DEC-002-9) and return its entries unchanged, in server order.

    Args:
        response: The value ``_request`` returned.
        operation: ``"listing models"`` or ``"fetching models"``, used in the error message.

    Returns:
        The ``data`` list, every entry a mapping with a non-empty string ``id``.

    Raises:
        LLMError: The reply shape is wrong (see ``_reply_entries``), an entry is not a mapping, or
            an entry's ``id`` is missing, not a string or empty.
            Every such error carries kind ``invalid_reply`` (DEC-arch-19).
    """
    entries = _reply_entries(response, operation)
    for position, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise LLMError(
                f"Invalid response from OpenAI API when {operation}: model entry {position} is not an object",
                description=_reply_text(response),
                kind="invalid_reply",
            )
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id:
            raise LLMError(
                f"Invalid response from OpenAI API when {operation}: model entry {position} has no non-empty text id",
                description=_reply_text(response),
                kind="invalid_reply",
            )
    return entries


def _embedding_vectors(response: Any, count: int) -> list[Any]:
    """Check an embeddings reply (DEC-002-8) and return one vector per text, in the texts' order.

    Args:
        response: The value ``_request`` returned.
        count: The number of texts sent.

    Returns:
        ``count`` vectors, ordered by the entries' ``index`` when every entry carries an integer
        one, otherwise in server order. Vector elements are not type-checked.

    Raises:
        LLMError: The reply shape is wrong (see ``_reply_entries``); an entry is not a mapping or
            its ``embedding`` is missing, not a list or empty; the entry count differs from
            ``count``; or all entries carry integer positions that are not exactly ``0..count-1``.
            Every such error carries kind ``invalid_reply`` (DEC-arch-19).
    """
    operation = "embedding"
    entries = _reply_entries(response, operation)
    for position, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            problem = "is not an object"
        elif "embedding" not in entry or entry["embedding"] is None:
            problem = "has no vector"
        elif not isinstance(entry["embedding"], list):
            problem = "has a vector that is not a list"
        elif not entry["embedding"]:
            problem = "has an empty vector"
        else:
            continue
        raise LLMError(
            f"Invalid response from OpenAI API when {operation}: entry {position} {problem}",
            description=_reply_text(response),
            kind="invalid_reply",
        )
    if len(entries) != count:
        raise LLMError(
            f"Invalid response from OpenAI API when {operation}: {len(entries)} vectors returned for {count} texts",
            description=_reply_text(response),
            kind="invalid_reply",
        )
    indices = [entry.get("index") for entry in entries]
    # DEC-002-8: positions are used only when every entry has one; a bool is an int subclass but is
    # not a position (DEC-002-8). Partial positions are ambiguous, so server order.
    if not all(isinstance(index, int) and not isinstance(index, bool) for index in indices):
        return [entry["embedding"] for entry in entries]
    if sorted(indices) != list(range(count)):
        raise LLMError(
            f"Invalid response from OpenAI API when {operation}: positions are not exactly 0..{count - 1}",
            description=_reply_text(response),
            kind="invalid_reply",
        )
    ordered: list[Any] = [None] * count
    for index, entry in zip(indices, entries):
        ordered[index] = entry["embedding"]
    return ordered


class OpenAIChatCompletionsClient(LLMClient[OpenAIOptions]):
    """Convenience layer over OpenAI's Chat Completions API (the chat-completions protocol class).

    Built by ``get_llm_client(..., protocol="chat_completions")`` or directly; independent of the
    Responses class ``OpenAIAPIClient`` (neither subclasses the other, DEC-arch-2).

    This client is designed for the official OpenAI API.
    It supports:
    - The credential the app passes as ``bearer_token`` (required; never read from the environment)
    - Standard OpenAI Chat Completions endpoint
    - OpenAI-specific parameter handling: only settings the app set are sent
    """

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1",
                 timeout: float = 600.0, bearer_token: Optional[str] = None,
                 default_options: Optional[OpenAIOptions] = None, *,
                 read_timeout: float | None = None, trace: bool | None = None) -> None:
        """Build a client that authenticates with the app's credential only (DEC-001-6).

        Args:
            model: Model name sent with every request.
            base_url: API root; a trailing slash is stripped by the base class.
            timeout: Total per-request timeout in seconds.
            bearer_token: The app's OpenAI credential, sent as ``Authorization: Bearer <token>``.
                The environment is never consulted.
            default_options: Client-level settings, merged under per-call settings.
            read_timeout: Silence limit in seconds for streamed requests (the per-read limit), or ``None``
                for none; forwarded to the base (DEC-arch-20, DEC-006-5).
            trace: Per-client trace switch (``None`` = environment decides, ``False`` = off, ``True`` = on into
                ``LLM_TRACE_FOLDER``); forwarded to the base (DEC-arch-27, DEC-006-5).

        Raises:
            ValueError: ``trace=True`` while ``LLM_TRACE_FOLDER`` is unset or empty (raised by the base).
            ValueError: ``bearer_token`` is missing or empty. The message tells the developer to
                provide a credential, names no environment variable and contains no credential value.
        """
        # DEC-001-6: refuse before any session exists; the message names the argument, never an
        # environment variable, and never echoes a credential value.
        if not bearer_token:
            raise ValueError("OpenAI credential not provided. Pass the API key as the bearer_token argument.")
        super().__init__(
            model, base_url, timeout, bearer_token, default_options, read_timeout=read_timeout, trace=trace
        )

    def _merge_options(
        self,
        *,
        options: Optional[OpenAIOptions],
    ) -> Dict[str, Any]:
        """Merge settings via the base three-layer merge and map the thinking switch (DEC-000-19).

        Returns only keys the app set (the table carries no built-in values, DEC-001-13).
        ``enable_thinking`` is removed; when truthy and no ``reasoning_effort`` is set,
        ``reasoning_effort`` becomes ``"medium"``. Nothing is added when the switch is unset.
        """
        opts = self._merge_options_base(
            options=options,
            supported_params=_OPENAI_SUPPORTED_PARAMS,
            alias_map=_OPENAI_ALIAS_MAP,
            default_opts=self.default_options,
        )
        # OpenAI uses reasoning_effort, not enable_thinking.
        # Translate enable_thinking into reasoning_effort for a unified interface.
        enable_thinking = opts.pop("enable_thinking", None)
        if enable_thinking and "reasoning_effort" not in opts:
            opts["reasoning_effort"] = "medium"
        return opts

    async def list_models(self) -> List[str]:
        """Return the ids of the models the server offers, in the server's order.

        The reply is checked (DEC-002-9): it must be a mapping with a list under ``data``, and every
        entry a mapping whose ``id`` is a non-empty string. The same rule as ``get_models``, so a
        list accepted by one is accepted by the other. Untraced.

        Returns:
            The entries' ``id`` values, in the order the server sent them.

        Raises:
            LLMError: The request failed (non-2xx or transport; passed through unchanged from
                ``_request``); the successful body could not be decoded as JSON (raised from the
                decode error, DEC-002-10); the body is not a mapping (including a non-JSON body
                returned as bytes); ``data`` is missing (existing "Invalid response from OpenAI API
                when listing models" message) or not a list; or an entry is not a mapping or lacks
                a non-empty string ``id``. New messages name the backend, the operation and what
                was wrong; the reply goes in the description.
        """
        response = await _fetch_reply(self, "listing models", "GET", "/models")
        entries = _model_entries(response, "listing models")
        return [model["id"] for model in entries]

    async def get_models(self) -> List[Dict[str, Any]]:
        """Return the model records the server offers, exactly as sent.

        The reply is checked with the same rule as ``list_models`` (DEC-002-9): a mapping with a
        list under ``data``, every entry a mapping whose ``id`` is a non-empty string. Entries are
        returned unchanged, in server order. Untraced.

        Returns:
            The ``data`` entries as the server sent them.

        Raises:
            LLMError: The request failed (non-2xx or transport; passed through unchanged from
                ``_request``); the successful body could not be decoded as JSON (raised from the
                decode error, DEC-002-10); the body is not a mapping (including a non-JSON body
                returned as bytes); ``data`` is missing (existing "Invalid response from OpenAI API
                when fetching models" message) or not a list; or an entry is not a mapping or lacks
                a non-empty string ``id``.
        """
        response = await _fetch_reply(self, "fetching models", "GET", "/models")
        entries = _model_entries(response, "fetching models")
        return entries

    async def chat(
        self,
        messages: List[LLMMessage],
        *,
        system: Optional[str] = None,
        options: Optional[OpenAIOptions] = None,
        stream: bool = False,
        on_delta: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> str:
        """Send one chat-completions request and return the reply text.

        Merged settings are spread top-level into the request body. Non-streamed: returns the
        reply content, or empty text when the server's content is null (DEC-001-12). Streamed:
        the request is opened through the base ``_stream_post`` helper (DEC-001-8); each text
        piece goes to ``on_delta`` in order (through the trace wrapper only when tracing is
        enabled, DEC-000-48) and their concatenation is returned.

        Raises:
            LLMError: Non-2xx reply (address, status and server reply; never the credential) or
                transport failure. A receiver failure propagates unchanged (DEC-000-23).
        """
        opt = self._merge_options(options=options)

        msgs = list(messages)
        if system:
            msgs.insert(0, {"role": "system", "content": system})

        payload = {
            "model": self.model,
            "messages": msgs,
            "stream": stream,
            **opt, # OpenAI options are top-level in payload
        }

        with self._tracer.trace("chat", self.model) as t:
            t.log_system(system)
            t.log_messages(msgs)
            t.log_request("POST", "/chat/completions", payload)

            if not stream:
                content = (await self._request("POST", "/chat/completions", json=payload))["choices"][0]["message"]["content"]
                # DEC-001-12: coerce null before tracing and returning, so neither sees None.
                result = content if content is not None else ""
                t.log_response(result)
                return result

            _orig_delta = on_delta

            async def _tracing_delta(text: str) -> None:
                t.log_chunk(text)
                if _orig_delta:
                    result = _orig_delta(text)
                    if inspect.isawaitable(result):
                        await result

            # DEC-001-8: the helper maps non-2xx and transport failures to LLMError. The trace wrapper
            # is installed only when tracing is enabled (DEC-000-48).
            async with self._stream_post("/chat/completions", payload) as response:
                result = await self._stream_openai_chat_response(
                    response, _tracing_delta if self._tracer.enabled else on_delta
                )
                t.log_response(result)
                return result

    async def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        options: Optional[OpenAIOptions] = None,
        stream: bool = False,
        on_delta: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> str:
        # OpenAI's primary text generation is via chat completions.
        # We can simulate 'generate' by wrapping the prompt in a user message.
        messages : List[LLMMessage] = [{"role": "user", "content": prompt}]
        return await self.chat(
            messages,
            system=system,
            options=options,
            stream=stream,
            on_delta=on_delta,
        )

    async def chat_json(
        self,
        messages: List[LLMMessage],
        *,
        response_schema: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
        options: Optional[OpenAIOptions] = None,
    ) -> str:
        """
        Chat with structured JSON output using OpenAI's JSON mode.

        Sends the app's ``response_format`` setting when one is set, otherwise
        ``{"type": "json_object"}`` (DEC-001-2). Returns the reply content, or empty text when
        the server's content is null (DEC-001-12).
        When ``response_schema`` is given, the only check is that the reply text parses as JSON
        (DEC-002-7, DEC-000-37): any valid JSON value (object, list, string, number, boolean or
        null) is returned unchanged; the schema's fields are not checked. Without
        ``response_schema`` the text is returned without any check. The schema is not sent
        (US-040, out of scope).

        Raises:
            LLMError: Request failure, or ``response_schema`` given and the reply is not valid JSON
                ("LLM response was not valid JSON, despite requesting JSON mode.", parse error in
                the description; kind ``invalid_reply``, DEC-arch-19).
        """
        opt = self._merge_options(options=options)

        msgs = list(messages)
        if system:
            msgs.insert(0, {"role": "system", "content": system})

        # DEC-001-2: the JSON-object format is the fallback only; ``opt`` is spread after it, so an
        # app-set ``response_format`` setting replaces it. Keep this order.
        payload = {
            "model": self.model,
            "messages": msgs,
            "stream": False,
            "response_format": {"type": "json_object"},
            **opt,
        }

        with self._tracer.trace("chat_json", self.model) as t:
            t.log_system(system)
            t.log_messages(msgs)
            t.log_request("POST", "/chat/completions", payload)

            content = (await self._request("POST", "/chat/completions", json=payload))["choices"][0]["message"]["content"]
            # DEC-001-12: null content becomes empty text before tracing and the parse check.
            response_content = content if content is not None else ""
            t.log_response(response_content)

        # DEC-002-7: the only check is the JSON parse; any valid JSON value passes unchanged.
        # TypeError stays caught because a non-string content cannot be parsed.
        if response_schema:
            try:
                json.loads(response_content)
            except (json.JSONDecodeError, TypeError) as e:
                raise LLMError(
                    "LLM response was not valid JSON, despite requesting JSON mode.",
                    description=str(e),
                    kind="invalid_reply",
                )

        return response_content

    async def _tools_stream_request(
        self,
        payload: Dict[str, Any],
        on_delta: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Run one streamed tool round (test seam; inputs and outputs unchanged).

        Opens the request to ``/chat/completions`` through the base ``_stream_post`` helper
        (DEC-001-8) and parses it with ``_stream_openai_tools_response``.

        Args:
            payload: Full request body for the round.
            on_delta: Optional receiver for streamed text pieces.

        Returns:
            The round's text content and its assembled tool calls.

        Raises:
            LLMError: Non-2xx reply or transport failure.
        """
        async with self._stream_post("/chat/completions", payload) as response:
            return await self._stream_openai_tools_response(response, on_delta)

    async def chat_with_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools_definitions: List[Dict[str, Any]],
        tools: Dict[str, Callable[..., Any]],
        system: Optional[str] = None,
        max_loops: int = 5,
        options: Optional[OpenAIOptions] = None,
        stream: bool = False,
        on_delta: Optional[Callable[[str], Awaitable[None]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tool_errors: Literal["raise", "result"] = "result",
        on_tool_event: Optional[_ToolEventObserver] = None,
    ) -> str:
        """Run the tool loop and return the model's final answer.

        ``messages`` is a read-only sequence of string-keyed mappings (any value type, so tool-call and
        tool turns type-check); the loop works on its own copy and the caller's sequence is never modified.

        Each round's body carries the merged settings; a truthy ``response_format`` parameter is
        sent and replaces the ``response_format`` setting, otherwise the setting (if set) is sent
        (DEC-001-15). Null content of the final answer, and of a recorded assistant tool-call
        turn, becomes empty text (DEC-001-12). Streamed rounds go through ``_tools_stream_request``.

        Each requested tool call is handled in order: offered-tool check, argument reading,
        ``log_tool_call``, argument-name check, run, result conversion, ``log_tool_result``. The
        exceptions named below are those of strict mode; tolerant mode tells the model instead (see
        Tool failures).

        - Offered-tool check: a missing, empty or not-advertised name (not in
          ``tools_definitions``) raises ``LLMError`` naming the requested name, even when ``tools``
          has a callable under that name; the tool is not run (DEC-002-3). A call without a
          ``function`` key or name counts as an empty name.
        - Argument reading: missing or empty arguments mean no arguments; a mapping is used as
          given; text is decoded as JSON and must give a JSON object. Undecodable text, a
          non-object JSON value, or any other type raises ``LLMError`` naming the tool, with the
          reason in its description; the tool is not run (DEC-002-4).
        - Result conversion: a mapping or list becomes JSON text with non-ASCII kept unescaped; a
          failed serialisation raises ``LLMError`` naming the tool (DEC-002-5). Anything else
          becomes ``str(result)``. This conversion is outside the wrap that turns a tool's own
          exception into ``RuntimeError``.
        - Round limit: when ``max_loops`` rounds pass without a final answer, ``RuntimeError`` is
          raised whose message states the round limit was reached, contains ``max_loops`` and
          the configured value (DEC-002-2). No partial answer is returned.

        Tool failures (``tool_errors``, DEC-arch-28, DEC-006-9): the five failure kinds are
        ``not_offered``, ``bad_arguments``, ``unknown_argument`` (an argument name the callable does not
        declare, the first one in the model's order; skipped for a callable taking ``**kwargs`` or one
        whose signature cannot be read, DEC-arch-16), ``tool_raised`` (an ordinary exception from the
        tool) and ``unserialisable_result``.

        - ``"result"`` (the default, tolerant): the failing call's tool is not run, or its failure is
          caught; the model receives a tool message with the fixed text for the kind (DEC-arch-14) —
          never exception text — carrying ``tool_call_id`` when the server gave an id; the trace gets
          ``TOOL RESULT`` then ``TOOL ERROR`` (DEC-006-7); the remaining calls and rounds go on.
        - ``"raise"`` (strict): the call ends with the exception listed under Raises, texts as before.
          Only ``"raise"`` selects strict handling; any other value behaves as ``"result"``.

        Cancellation and other non-ordinary exceptions propagate unchanged in both modes, with no
        event and no tool message for that call. The round limit applies in both modes.

        Tool event observer (``on_tool_event``): when given, it receives one ``ToolEvent`` per event,
        in call order: ``start`` before the call is checked or run, then one ``result`` (``output`` =
        the tool message text sent) or ``error`` (with its ``error_kind``). It may be a plain or a
        coroutine function; an awaitable result is awaited before the loop goes on (DEC-arch-18).
        ``call_id`` is the server's tool-call id or ``None``; ``arguments`` is the raw text as sent,
        JSON text for a mapping, empty text when absent (DEC-006-4). In strict mode the ``error``
        event arrives before the exception (DEC-arch-17), and an observer failure propagates
        unchanged (after a ``start`` failure the tool does not run). In tolerant mode an observer
        failure gets no further event for that call, a tool not yet run is not run, the model is
        told ``Tool '<n>' failed.`` instead of any result, one ``LLMToolObserverWarning`` is
        emitted and the trace records it (DEC-arch-26, DEC-006-7, DEC-006-8).

        Args:
            messages: The conversation so far; copied, never modified.
            tools_definitions: Tools advertised to the model (chat-completions ``function`` shape).
            tools: Callables by tool name; every advertised tool needs one.
            system: Optional system prompt placed first.
            max_loops: Maximum number of model rounds.
            options: Call-time settings (merged over the client's defaults).
            stream: Stream each round through ``_tools_stream_request``.
            on_delta: Receiver of streamed text pieces (streamed rounds only).
            response_format: Optional ``response_format`` sent on every round (DEC-001-15).
            tool_errors: ``"result"`` (default, tolerant) or ``"raise"`` (strict), see above.
            on_tool_event: Optional tool-event observer, see above.

        Returns:
            The model's final answer text.

        Raises:
            ValueError: An advertised tool has no callable (before any request); in strict mode, a
                call's argument names do not fit the callable (the tool is not run).
            RuntimeError: In strict mode, a tool raises (message names the tool, chained from the
                tool's exception); in both modes, the round limit is reached.
            LLMError: Request failure; in strict mode (``kind="tool"``), a requested tool not offered or
                with an empty name, unreadable arguments, or an unserialisable mapping or list result.
            Exception: In strict mode, whatever the observer raised, unchanged.
        """
        advertised = {d["function"]["name"] for d in tools_definitions}
        missing = advertised - tools.keys()
        if missing:
            raise ValueError(f"Missing callables for tools {sorted(missing)}")

        opt = self._merge_options(options=options)

        # DEC-006-2 / Step 002 DoD-3: the loop's own list of dict turns; the caller's read-only sequence
        # (a tuple is fine) and its mappings are never modified.
        msgs: list[dict[str, Any]] = [dict(m) for m in messages]
        if system:
            msgs.insert(0, {"role": "system", "content": system})

        with self._tracer.trace("chat_with_tools", self.model) as t:
            t.log_system(system)
            t.log_messages(msgs)

            for loop_idx in range(max_loops):
                payload = {
                    "model": self.model,
                    "messages": msgs,
                    "tools": tools_definitions,
                    "stream": stream,
                    **opt,
                }
                # DEC-001-15: applied after the settings spread, so a truthy parameter replaces the
                # ``response_format`` setting on every round; without it the setting (if any) stays.
                if response_format:
                    payload["response_format"] = response_format

                t.log_request("POST", f"/chat/completions (loop {loop_idx})", payload)

                if stream:
                    content, tool_calls = await self._tools_stream_request(payload, on_delta)
                    assistant_msg: Dict[str, Any] = {"content": content, "tool_calls": tool_calls}
                else:
                    assistant_msg = (await self._request("POST", "/chat/completions", json=payload))["choices"][0]["message"]
                    tool_calls = assistant_msg.get("tool_calls") or []

                # DEC-001-12: a missing or null content is empty text, both for the final answer and
                # for the assistant tool-call turn recorded in the conversation.
                content_text = assistant_msg.get("content") or ""

                if not tool_calls:
                    final = content_text
                    msgs.append({"role": "assistant", "content": final})
                    t.log_response(final)
                    return final

                msgs.append(
                    {
                        "role": "assistant",
                        "content": content_text,
                        "tool_calls": tool_calls,
                    }
                )

                for call in tool_calls:
                    # Sequential, in the order requested; shared by streamed and non-streamed rounds. A
                    # tolerated failure appends the same tool message as a success, carrying the
                    # model-facing text; calls earlier in this reply have already run and keep their effects.
                    content = await self._run_tool_call(
                        call,
                        advertised=advertised,
                        tools=tools,
                        tool_errors=tool_errors,
                        on_tool_event=on_tool_event,
                        t=t,
                    )
                    tool_message: Dict[str, Any] = {"role": "tool", "content": content}
                    # DEC-000-29: ``tool_call_id`` only when the server gave a truthy id, also for
                    # tolerant failure messages.
                    tool_call_id = call.get("id") if isinstance(call, Mapping) else None
                    if tool_call_id:
                        tool_message["tool_call_id"] = tool_call_id
                    msgs.append(tool_message)

        # DEC-002-2: stays RuntimeError and outside the trace session; the text keeps `max_loops`.
        raise RuntimeError(f"Tool loop reached its round limit (max_loops={max_loops}) without a final answer")

    async def _run_tool_call(
        self,
        call: Any,
        *,
        advertised: set[str],
        tools: Dict[str, Callable[..., Any]],
        tool_errors: Literal["raise", "result"],
        on_tool_event: Optional[_ToolEventObserver],
        t: TraceSession,
    ) -> str:
        """Handle one requested tool call and return the content of its tool message.

        This class's own copy of the per-call procedure (feature 006 ``context.md`` § Per-call
        procedure; DEC-arch-15): ``start`` event, offered check, argument reading, argument-name check
        (DEC-arch-16), ``TOOL CALL`` trace, run (awaited if awaitable), result to text, then the
        ``result`` event and ``TOOL RESULT`` trace. The caller builds the tool message (adding
        ``tool_call_id`` only when the server's id is truthy, DEC-000-29) and appends it.

        Args:
            call: One entry of the round's ``tool_calls`` as received (``id``, ``function.name``,
                ``function.arguments`` as text, mapping or none).
            advertised: Tool names offered in ``tools_definitions``.
            tools: Callables by tool name.
            tool_errors: ``"raise"`` selects strict handling; any other value is tolerant (DEC-006-9).
            on_tool_event: Optional observer; each event is awaited inline when awaitable (DEC-arch-18).
            t: The trace session of the running conversation.

        Returns:
            The tool result text on success; in tolerant mode, the fixed model-facing text for the
            failure kind (DEC-arch-14) or ``Tool '<n>' failed.`` after an observer failure (DEC-006-8).

        Raises:
            LLMError: Strict mode, ``kind="tool"``: not offered / empty name, unreadable arguments,
                unserialisable result (texts as before, DEC-002-3..5).
            ValueError: Strict mode, an argument name the callable does not declare.
            RuntimeError: Strict mode, the tool raised an ordinary exception (chained from it).
            Exception: Strict mode, the observer's own exception, unchanged.
            BaseException: Cancellation or another non-ordinary exception from the tool, in both
                modes, unchanged and without an ``error`` event.

        Side effects:
            Sends events to the observer; writes trace entries (``TOOL CALL`` / ``TOOL RESULT`` /
            ``TOOL ERROR``, DEC-006-7); in tolerant mode emits ``LLMToolObserverWarning`` once per
            observer failure.
        """
        strict = tool_errors == "raise"  # DEC-006-9: only "raise" selects strict handling
        function = call.get("function") if isinstance(call, Mapping) else None
        raw_name = function.get("name") if isinstance(function, Mapping) else None
        raw_arguments = function.get("arguments") if isinstance(function, Mapping) else None
        raw_id = call.get("id") if isinstance(call, Mapping) else None
        # Event identity (DEC-006-4): the requested name as sent (missing -> "", a non-text value as its
        # text form), the server's call id only when it is a non-empty string, raw arguments as text
        # (string verbatim, none -> "", a mapping as JSON text with non-ASCII kept). The same name is
        # ``<n>`` in every model-facing text (DEC-arch-14).
        event_name = raw_name if isinstance(raw_name, str) else ("" if raw_name is None else str(raw_name))
        call_id = raw_id if isinstance(raw_id, str) and raw_id else None
        if isinstance(raw_arguments, str):
            arguments_text = raw_arguments
        elif raw_arguments is None:
            arguments_text = ""
        else:
            try:
                arguments_text = json.dumps(raw_arguments, ensure_ascii=False)
            except (TypeError, ValueError):
                # A value JSON cannot encode (or a circular one) is shown as its text form.
                arguments_text = str(raw_arguments)
        failed_text = f"Tool '{event_name}' failed."
        call_logged = False

        def log_call(arguments: Any) -> None:
            """Write this call's single ``TOOL CALL`` entry once (DEC-006-7): the parsed arguments when
            read, otherwise the raw argument text."""
            nonlocal call_logged
            if not call_logged:
                t.log_tool_call(event_name, arguments)
                call_logged = True

        async def notify(
            event_type: Literal["start", "result", "error"],
            *,
            output: str | None = None,
            error_kind: ToolErrorKind | None = None,
        ) -> bool:
            """Send one event to the observer, inline and awaited when awaitable (DEC-arch-18, DEC-000-23).

            Returns ``True`` when the observer failed in tolerant mode; that failure is already handled
            here (one ``LLMToolObserverWarning``, trace entries, DEC-006-7 / DEC-006-8) and the caller must
            return ``failed_text`` without any further event for this call (DEC-arch-26). Returns ``False``
            when there is no observer or it finished normally. Strict mode: the observer's ordinary
            exception propagates unchanged. Cancellation always propagates.
            """
            if on_tool_event is None:
                return False
            event: ToolEvent = {
                "type": event_type,
                "name": event_name,
                "call_id": call_id,
                "arguments": arguments_text,
                "output": output,
                "error_kind": error_kind,
            }
            try:
                outcome = on_tool_event(event)
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception as exc:
                if strict:
                    raise
                observer_error = exc
            else:
                return False
            # Handled outside the except block so the warning and trace writes carry no implicit
            # exception context.
            description = f"{type(observer_error).__name__}: {observer_error}"
            warnings.warn(
                f"Tool-event observer failed on the '{event_type}' event of tool '{event_name}': {description}",
                LLMToolObserverWarning,
                # notify -> _run_tool_call -> chat_with_tools -> the app's call site (outcome.md, step 006).
                stacklevel=4,
            )
            log_call(arguments_text)
            t.log_tool_result(event_name, failed_text)
            t.log_tool_error(event_name, f"observer failed: {description}")
            return True

        async def fail(
            kind: ToolErrorKind, model_text: str, strict_error: Exception, cause: Exception | None = None
        ) -> str:
            """Handle one tool failure of ``kind``: ``error`` event first, then raise or tolerate.

            Strict (DEC-arch-17): raise ``strict_error`` (today's exception and text), chained from
            ``cause`` when given (a helper's ``LLMError`` keeps its own chaining). Tolerant: trace
            ``TOOL RESULT`` with ``model_text`` then ``TOOL ERROR`` with the kind name (DEC-006-7) and
            return ``model_text``. An observer failure on the ``error`` event replaces both outcomes
            (strict: propagates; tolerant: ``failed_text`` with only the observer-failure trace pair).
            """
            log_call(arguments_text)
            if await notify("error", error_kind=kind):
                return failed_text
            if strict:
                if cause is not None:
                    raise strict_error from cause
                raise strict_error
            t.log_tool_result(event_name, model_text)
            t.log_tool_error(event_name, kind)
            return model_text

        # Each check records its failure and handles it after leaving the ``except`` block, so a strict
        # re-raise or an observer exception is not implicitly chained to it.
        if await notify("start"):
            # Observer failed on ``start`` (tolerant): the tool is not run (DEC-arch-26).
            return failed_text

        offered_error: LLMError | None = None
        name = event_name
        try:
            name = _offered_tool_name(call, advertised)
        except LLMError as exc:
            offered_error = exc
        if offered_error is not None:
            return await fail("not_offered", f"Tool '{event_name}' is not available.", offered_error)

        func = tools[name]  # pre-flight guarantees every advertised name has a callable
        arguments_error: LLMError | None = None
        kwargs: dict[str, Any] = {}
        try:
            kwargs = _read_tool_arguments(name, raw_arguments)
        except LLMError as exc:
            arguments_error = exc
        if arguments_error is not None:
            return await fail("bad_arguments", f"Invalid arguments for tool '{name}'.", arguments_error)

        # Today's trace position: after the arguments are read, before the name check (DEC-006-7).
        log_call(kwargs)
        # DEC-arch-16: signature read once; the check is skipped for an unreadable signature or a
        # ``**kwargs`` parameter. Names are tested in ``kwargs`` order, the order the model sent them
        # (JSON decoding and ``dict`` copies keep it). Membership is today's rule (DEC-000-27).
        undeclared: str | None = None
        parameters: Any = None
        try:
            parameters = inspect.signature(func).parameters
        except (TypeError, ValueError):
            parameters = None
        if parameters is not None and not any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        ):
            undeclared = next((key for key in kwargs if key not in parameters), None)
        if undeclared is not None:
            return await fail(
                "unknown_argument",
                f"Tool '{name}' has no argument '{undeclared}'.",
                ValueError(f"Bad args for tool '{name}': {kwargs}"),
            )

        tool_error: Exception | None = None
        result: Any = None
        try:
            result = func(**kwargs)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            # Only ordinary exceptions are tool failures; cancellation and other BaseExceptions
            # propagate with no event and no tool message (interfaces.md § Tool failures).
            tool_error = exc
        if tool_error is not None:
            return await fail(
                "tool_raised", failed_text, RuntimeError(f"Tool '{name}' failed: {tool_error}"), cause=tool_error
            )

        # Outside the tool's try above, so a conversion failure stays its own kind (DEC-002-5). Any
        # ordinary exception counts (e.g. a failing ``__str__``); strict re-raises it unchanged.
        conversion_error: Exception | None = None
        content = ""
        try:
            content = _tool_result_text(name, result)
        except Exception as exc:
            conversion_error = exc
        if conversion_error is not None:
            return await fail("unserialisable_result", failed_text, conversion_error)

        if await notify("result", output=content):
            # DEC-006-8: the result already produced is discarded; the model is told the tool failed.
            return failed_text
        t.log_tool_result(name, content)
        return content

    async def embed_batch(
        self,
        texts: List[str],
        *,
        options: Optional[OpenAIOptions] = None,
    ) -> List[List[float]]:
        """Embed a batch of texts and return one vector per text, in the texts' order.

        Sends ``{"model", "input": texts}`` plus ``user`` when set in call-time or default options;
        other settings are not sent. The empty batch is still sent. Inside the trace session, the
        reply is checked before the response count is recorded (DEC-002-8): it must be a mapping
        with a list under ``data``; every entry a mapping whose ``embedding`` is a non-empty list
        (elements are not type-checked); the number of entries equal to ``len(texts)``. When every
        entry carries an integer ``index`` (a boolean does not count), the indices must be exactly
        ``0..n-1`` and the vectors are returned in index order; otherwise the server's order is
        kept. The count recorded in the trace is the validated count.

        Args:
            texts: The texts to embed.
            options: Per-call settings; only ``user`` is used.

        Returns:
            Exactly ``len(texts)`` vectors; vector ``i`` belongs to ``texts[i]`` when the server
            sent positions, otherwise the server's order.

        Raises:
            LLMError: The request failed (non-2xx or transport; passed through unchanged from
                ``_request``); the successful body could not be decoded as JSON (raised from the
                decode error, DEC-002-10); the body is not a mapping (including a non-JSON body
                returned as bytes); ``data`` is missing (existing "Invalid response from OpenAI API
                when embedding" message) or not a list; an entry is not a mapping or its
                ``embedding`` is missing, not a list or empty; the entry count differs from the
                number of texts; or all entries carry integer positions that are not exactly
                ``0..n-1`` (duplicate, gap or out of range). New messages name the backend, the
                operation and what was wrong; the reply goes in the description. Raised inside the
                trace session, so it is recorded as an ERROR section.
        """
        # For embeddings, we only want model and input.
        # Sampling parameters like temperature are for completions.
        payload = {
            "model": self.model,
            "input": texts,
        }

        # Merge 'user' if provided in call-time options or default options
        user = (options or {}).get("user") or (self.default_options or {}).get("user")
        if user:
            payload["user"] = user

        with self._tracer.trace("embed_batch", self.model) as t:
            t.log_request("POST", "/embeddings", payload)
            response = await _fetch_reply(self, "embedding", "POST", "/embeddings", json=payload)
            # DEC-002-8: checked inside the session so a violation is recorded as ERROR, and
            # before log_response so the recorded count is the validated one.
            vectors = _embedding_vectors(response, len(texts))
            t.log_response(f"{len(vectors)} embeddings returned")

        return vectors
