"""llama-swap Responses client: optional credential, Responses request/reply for chat, generate and JSON answers.

Feature 003 step 004 rewrites ``LlamaSwapAPIClient`` for the Responses protocol (``POST /responses``,
DEC-003-9..12, DEC-003-2); its chat-completions behaviour lives on in ``llm/llama_swap_chat_completions.py``.
Step 005 adds the streamed path: its own Responses event parser (DEC-003-13, DEC-arch-2).
Feature 004 step 003 adds the non-streamed tool loop on Responses: tool flattening, typed call/result
items, per-call reasoning carry (DEC-004-5..12), all its own copy (DEC-arch-2); step 004 adds the
streamed tool rounds with their own tool-round event parser (DEC-004-9).
Model listing and embeddings keep the reply checks of feature 002 step 003 (DEC-002-8..10).
"""

from collections.abc import Mapping
from typing import Any, Awaitable, Callable, Dict, List, Optional
import inspect
import json
import warnings

import aiohttp

from llm.llm_client import LLMClient
from llm.exceptions import LLMError, LLMSettingIgnoredWarning
from llm.message import LLMMessage
from llm.options import OpenAIOptions

# Settings the llama-swap Responses client accepts (DEC-003-9). Every value is None: the table
# contributes no built-in value, so only keys the app set are considered (DEC-001-13). Each key's
# wire form is decided by `_merge_options`; see `_LLAMA_SWAP_IGNORED_SETTINGS` for the dropped ones.
_LLAMA_SWAP_SUPPORTED_PARAMS: Dict[str, Any] = {
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

# Setting aliases (unchanged): `context_length` acts as `max_tokens`.
_LLAMA_SWAP_ALIAS_MAP = {
    "context_length": "max_tokens",
}

# Supported settings with no Responses counterpart (DEC-003-9): never sent; the first time a client
# sees one set it emits `LLMSettingIgnoredWarning` (DEC-003-6).
_LLAMA_SWAP_IGNORED_SETTINGS: tuple[str, ...] = (
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "stop",
    "logit_bias",
)


def _responses_text_format(response_format: Any) -> Any:
    """Translate an app-facing ``response_format`` value into the Responses text format (DEC-003-10).

    The result is the value placed under ``text.format`` in the request body.

    Args:
        response_format: The merged ``response_format`` setting as the app gave it (chat-completions
            shape or any other value).

    Returns:
        ``{"type": "json_object"}`` and ``{"type": "text"}`` unchanged; for
        ``{"type": "json_schema", "json_schema": {...}}`` a new mapping ``{"type": "json_schema", ...}``
        with every key of the nested mapping (``name``, ``schema``, ``strict``, ``description``, ...)
        lifted beside ``type``; any other value (including an already flat Responses format) unchanged.
        The input value is never modified. No validation; never raises.
    """
    if not isinstance(response_format, Mapping):
        return response_format
    nested = response_format.get("json_schema")
    # Only the chat-completions json_schema shape (nested mapping) is flattened; a flat Responses
    # json_schema format has no "json_schema" mapping and passes through unchanged (DEC-003-10).
    if response_format.get("type") == "json_schema" and isinstance(nested, Mapping):
        return {"type": "json_schema", **nested}
    return response_format


def _responses_reply_text(reply: Any) -> str:
    """Return the answer text of a non-streamed Responses reply, or raise for a failed one (DEC-003-12).

    Args:
        reply: The decoded body ``_request`` returned for ``POST /responses``.

    Returns:
        The ``text`` of every ``output_text`` content part of every ``message`` item in ``output``,
        in order, concatenated with no separator. Reasoning items (including ``reasoning_text``
        content with empty ``summary`` / ``encrypted_content``, as llama.cpp sends) and other part
        types are ignored. A missing or non-list ``output``, no message item, or a null/missing
        ``text`` contributes ``""``. Any status other than ``failed`` / ``incomplete`` (or none)
        extracts the text.

    Raises:
        LLMError: ``status == "failed"`` — description ``error.message`` when ``error`` is a mapping
            with a string message, else ``error`` as JSON text; ``status == "incomplete"`` —
            description ``incomplete_details.reason`` when present, else ``incomplete_details`` as
            JSON text. A reply that is not a mapping surfaces as a generic error (accepted).
    """
    status = reply.get("status")
    if status == "failed":
        error = reply.get("error")
        if isinstance(error, Mapping) and isinstance(error.get("message"), str):
            reason = error["message"]
        else:
            reason = json.dumps(error, ensure_ascii=False, default=str)
        raise LLMError("llama-swap Responses reply failed", description=reason)
    if status == "incomplete":
        details = reply.get("incomplete_details")
        if isinstance(details, Mapping) and details.get("reason") is not None:
            reason = str(details["reason"])
        else:
            reason = json.dumps(details, ensure_ascii=False, default=str)
        raise LLMError("llama-swap Responses reply is incomplete", description=reason)

    output = reply.get("output")
    if not isinstance(output, list):
        return ""
    parts: list[str] = []
    for item in output:
        # Reasoning items (llama.cpp: reasoning_text content, empty summary / encrypted_content)
        # are skipped by type, so no reasoning text appears off-stream (US-028.AC-1).
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "output_text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


def _tool_definition_name(definition: Any) -> Any:
    """Return the tool name an app tool definition advertises (DEC-004-5).

    Used by the pre-flight check (advertised without a callable) and to build the offered-name set
    of the offered check (DEC-002-3). Never modifies ``definition``.

    Args:
        definition: One entry of ``tools_definitions``: nested
            ``{"type": "function", "function": {"name", ...}}`` (as ``pydantic_to_openai_tool``
            builds it) or flat Responses ``{"type": "function", "name", ...}``.

    Returns:
        ``function.name`` for an entry with a ``function`` mapping, ``name`` for a flat entry; the
        value as given (``None`` when absent).
    """
    if not isinstance(definition, Mapping):
        return None
    function = definition.get("function")
    # Same nested/flat split as `_responses_tools`: only a `function` mapping marks a nested entry.
    if isinstance(function, Mapping):
        return function.get("name")
    return definition.get("name")


def _responses_tools(tools_definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten the app's tool definitions into Responses function tools (DEC-004-5, DEC-004-1).

    Per entry, in order: an entry with a ``function`` mapping becomes
    ``{"type": "function", "name": function.name, "description": function.description (only when
    present), "parameters": function.parameters (only when present), "strict": function.strict when
    present, else False}``; an entry without a ``function`` key is passed through unchanged.

    Args:
        tools_definitions: The app's definitions; neither the list nor any entry is modified.

    Returns:
        A new list, same length and order, sent as ``tools`` on every round.
    """
    flat: list[dict[str, Any]] = []
    for definition in tools_definitions:
        function = definition.get("function") if isinstance(definition, Mapping) else None
        if not isinstance(function, Mapping):
            # Already flat (or unrecognised): sent as given, keeping its own `strict` (DEC-004-5).
            flat.append(definition)
            continue
        tool: dict[str, Any] = {"type": "function", "name": function.get("name")}
        if "description" in function:
            tool["description"] = function["description"]
        if "parameters" in function:
            tool["parameters"] = function["parameters"]
        # DEC-004-1: copy the app's flag; otherwise false, so typical non-strict schemas
        # (pydantic_to_openai_tool output) are not constrained by strict mode.
        tool["strict"] = function.get("strict", False)
        flat.append(tool)
    return flat


def _responses_tool_round(reply: Any) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Read one non-streamed tool-round reply into round text, tool calls and carry items (DEC-004-8).

    The status is checked first exactly as ``_responses_reply_text`` does (DEC-003-12). Then
    ``output`` is visited in order: ``function_call`` items become calls, ``message`` items give the
    round text, ``reasoning`` items are kept verbatim; every other item type is ignored.

    Args:
        reply: The decoded body ``_request`` returned for one ``POST /responses`` round.

    Returns:
        ``(text, calls, reasoning)``:

        - ``text`` — the round text (output-text parts of message items concatenated, no separator;
          ``""`` when none), never reasoning text or a marker;
        - ``calls`` — one ``{"call_id", "name", "arguments"}`` mapping per ``function_call`` item in
          order of appearance (list position is the dense 0-based call number, DEC-arch-10), values
          as the server sent them (missing → ``None``), without the server's item ``id``;
        - ``reasoning`` — the reasoning items exactly as received, in output order (DEC-004-7).

    Raises:
        LLMError: ``status`` ``failed`` / ``incomplete``, carrying the server's reason (DEC-003-12).
    """
    # Reusing the chat path's reader gives exactly DEC-003-12: status first (failed / incomplete →
    # LLMError with the server's reason), then the message items' output-text parts joined.
    text = _responses_reply_text(reply)
    calls: list[dict[str, Any]] = []
    reasoning: list[dict[str, Any]] = []
    output = reply.get("output")
    if not isinstance(output, list):
        return text, calls, reasoning
    for item in output:
        if not isinstance(item, Mapping):
            continue
        kind = item.get("type")
        if kind == "function_call":
            # The server's item `id` is dropped: the round input sends the reduced call (DEC-004-6).
            calls.append(
                {
                    "call_id": item.get("call_id"),
                    "name": item.get("name"),
                    "arguments": item.get("arguments"),
                }
            )
        elif kind == "reasoning":
            # Kept as received (DEC-004-7); llama.cpp's empty summary / encrypted content included.
            reasoning.append(item)
    return text, calls, reasoning


def _offered_tool_name(call: Any, offered: set[str]) -> str:
    """Return the tool name of one Responses function call after checking it was offered (DEC-002-3).

    llama-swap Responses copy (DEC-arch-2): the name is read from the call's top-level ``name``
    (the shape ``_responses_tool_round`` returns), not from a nested ``function``.

    Args:
        call: One call as returned by ``_responses_tool_round`` (``{"call_id", "name", "arguments"}``).
        offered: Names advertised in ``tools_definitions`` (via ``_tool_definition_name``).

    Returns:
        The requested name, guaranteed to be in ``offered``.

    Raises:
        LLMError: The name is missing or empty (also when the call is not a mapping) —
            "The model requested a tool call with an empty tool name"; or it is not text / not
            advertised, even when ``tools`` holds a callable for it —
            "The model requested tool '<name>', which was not offered".
    """
    name = call.get("name") if isinstance(call, Mapping) else None
    # A malformed call is reported as an empty name rather than a raw KeyError/TypeError.
    if not name:
        raise LLMError("The model requested a tool call with an empty tool name")
    if not isinstance(name, str) or name not in offered:
        raise LLMError(f"The model requested tool '{name}', which was not offered")
    return name


def _read_tool_arguments(name: str, raw: Any) -> dict[str, Any]:
    """Turn a requested call's arguments into the keyword mapping the tool is called with (DEC-002-4).

    Copy of the chat-completions helper (not imported, DEC-arch-2).

    Args:
        name: The (offered) tool name, used in the error message.
        raw: The call's ``arguments`` value as received.

    Returns:
        ``{}`` for ``None`` or ``""``; a mapping as a new ``dict``; the decoded JSON object for text.

    Raises:
        LLMError: Text that is not JSON (chained from the decode error), JSON that is not an object,
            or a value that is neither text nor a mapping; the message names the tool and the
            description carries the reason.
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
            raise LLMError(f"Arguments for tool '{name}' are not valid JSON", description=str(exc)) from exc
        if not isinstance(decoded, dict):
            raise LLMError(
                f"Arguments for tool '{name}' are not a JSON object",
                description=f"decoded to {type(decoded).__name__}",
            )
        return decoded
    raise LLMError(
        f"Arguments for tool '{name}' are not readable",
        description=f"unsupported type {type(raw).__name__}",
    )


def _tool_result_text(name: str, result: Any) -> str:
    """Turn a tool's return value into the ``function_call_output`` text (DEC-002-5).

    Copy of the chat-completions helper (not imported, DEC-arch-2).

    Args:
        name: The tool name, used in the error message.
        result: The value the tool returned (already awaited).

    Returns:
        JSON text with non-ASCII kept for a ``dict`` or ``list``; ``str(result)`` otherwise.

    Raises:
        LLMError: The ``dict`` or ``list`` cannot be serialised; names the tool, chained from the error.
    """
    # Only dict/list are JSON-encoded; other values keep their text form so an object with a custom
    # __str__ is still sent as that text (DEC-002-5).
    if not isinstance(result, (dict, list)):
        return str(result)
    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        # TypeError: a non-serialisable element; ValueError: a circular reference.
        raise LLMError(f"Result of tool '{name}' cannot be serialised to JSON", description=str(exc)) from exc


async def _fetch_reply(client: LLMClient, operation: str, method: str, path: str, **kwargs: Any) -> Any:
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
    """
    try:
        return await client._request(method, path, **kwargs)
    # DEC-002-10: only the two decode errors ``_request`` lets escape on a 2xx JSON body are mapped;
    # a broad ValueError catch could swallow unrelated errors (003.context.md). ``_request`` is not edited.
    except (json.JSONDecodeError, aiohttp.ContentTypeError) as exc:
        raise LLMError(f"Unreadable reply from llama-swap when {operation}", description=str(exc)) from exc


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
    """
    if not isinstance(response, Mapping):
        raise LLMError(
            f"Invalid response from llama-swap when {operation}: the reply is not a JSON object",
            description=_reply_text(response),
        )
    if "data" not in response:
        # Wording kept as before this step; older tests may match it (003.context.md § Message wording).
        raise LLMError(f"Invalid response from llama-swap when {operation}: {response}")
    entries = response["data"]
    if not isinstance(entries, list):
        raise LLMError(
            f"Invalid response from llama-swap when {operation}: 'data' is not a list",
            description=_reply_text(response),
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
    """
    entries = _reply_entries(response, operation)
    for position, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise LLMError(
                f"Invalid response from llama-swap when {operation}: model entry {position} is not an object",
                description=_reply_text(response),
            )
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id:
            raise LLMError(
                f"Invalid response from llama-swap when {operation}: model entry {position} has no non-empty text id",
                description=_reply_text(response),
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
            f"Invalid response from llama-swap when {operation}: entry {position} {problem}",
            description=_reply_text(response),
        )
    if len(entries) != count:
        raise LLMError(
            f"Invalid response from llama-swap when {operation}: {len(entries)} vectors returned for {count} texts",
            description=_reply_text(response),
        )
    indices = [entry.get("index") for entry in entries]
    # DEC-002-8: positions are used only when every entry has one; a bool is an int subclass but is
    # not a position (003.context.md § Positions). Partial positions are ambiguous, so server order.
    if not all(isinstance(index, int) and not isinstance(index, bool) for index in indices):
        return [entry["embedding"] for entry in entries]
    if sorted(indices) != list(range(count)):
        raise LLMError(
            f"Invalid response from llama-swap when {operation}: positions are not exactly 0..{count - 1}",
            description=_reply_text(response),
        )
    ordered: list[Any] = [None] * count
    for index, entry in zip(indices, entries):
        ordered[index] = entry["embedding"]
    return ordered


class LlamaSwapAPIClient(LLMClient):
    """llama-swap client speaking the Responses protocol (``POST /responses``) for chat-style calls.

    ``chat``, ``generate`` and ``chat_json`` send Responses requests (DEC-arch-1, DEC-003-11): only
    the app's settings are sent, translated to Responses names; supported settings without a
    Responses counterpart are dropped with one ``LLMSettingIgnoredWarning`` per setting per client
    instance (DEC-003-6); thinking and an explicit effort reach the engine as ``reasoning.effort``,
    never with a summary (DEC-003-2). ``chat_with_tools`` runs the tool loop on Responses with its
    own copy of flattening, item conversion and round reading (feature 004, DEC-004-5..11).
    ``list_models``, ``get_models`` and ``embed_batch`` are unchanged (``/models``, ``/embeddings``).
    The credential is optional (US-005). The chat-completions behaviour is
    ``LlamaSwapChatCompletionsClient``. Written independently of ``OpenAIAPIClient`` (DEC-arch-2).

    Invariant: ``_warned_settings`` only grows; a key in it is never warned about again by this
    instance.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8080/v1",
        timeout: float = 600.0,
        bearer_token: Optional[str] = None,
        default_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create the client; no request is sent and no warning is emitted.

        Args:
            model: Model id sent with every request.
            base_url: llama-swap address including ``/v1``.
            timeout: Request timeout in seconds.
            bearer_token: Optional credential, sent as the bearer header when given (US-005).
            default_options: Client-default settings, merged under per-call settings.
        """
        super().__init__(model, base_url, timeout, bearer_token, default_options)
        # DEC-003-6: no-counterpart settings this instance has already warned about.
        self._warned_settings: set[str] = set()

    def _merge_options(
        self,
        *,
        options: Optional[OpenAIOptions],
    ) -> Dict[str, Any]:
        """Merge table, client-default and per-call settings and return the Responses wire fields.

        Uses ``_merge_options_base`` unchanged with the llama-swap tables (DEC-000-5): aliases
        applied (``context_length`` → ``max_tokens``), unknown keys dropped silently (no warning),
        a per-call value wins, a higher-layer ``None`` removes the key. The merged settings then
        become wire fields (DEC-003-9):

        - ``temperature``, ``top_p``, ``user`` → same name; ``max_tokens`` → ``max_output_tokens``.
        - ``response_format`` → ``text: {"format": _responses_text_format(value)}`` (DEC-003-10).
        - ``_LLAMA_SWAP_IGNORED_SETTINGS`` keys → not sent; for each one present whose name is not
          in ``self._warned_settings``, ``warnings.warn(..., LLMSettingIgnoredWarning)`` naming the
          setting and saying Responses has no counterpart so it was not sent, then the name is
          added to ``self._warned_settings`` (DEC-003-6).
        - ``enable_thinking`` / ``reasoning_effort`` → never top-level; truthy thinking →
          ``reasoning: {"effort": <explicit effort or "medium">}``; thinking set and falsy →
          ``reasoning: {"effort": <explicit effort or "none">}``; thinking unset with an effort →
          ``reasoning: {"effort": <effort>}``; neither → no ``reasoning``. Never a ``summary``
          (DEC-003-1, DEC-003-2).

        Values are passed through unvalidated.

        Args:
            options: Per-call settings, or ``None``.

        Returns:
            A new mapping of wire fields only (no ``model``, ``input``, ``stream``, ``store``,
            ``include``).

        Side effects:
            May emit ``LLMSettingIgnoredWarning`` and grow ``self._warned_settings``.
        """
        merged = self._merge_options_base(
            options=options,
            supported_params=_LLAMA_SWAP_SUPPORTED_PARAMS,
            alias_map=_LLAMA_SWAP_ALIAS_MAP,
            default_opts=self.default_options,
        )
        wire: Dict[str, Any] = {}
        for name in ("temperature", "top_p", "user"):
            if name in merged:
                wire[name] = merged[name]
        if "max_tokens" in merged:
            wire["max_output_tokens"] = merged["max_tokens"]
        if "response_format" in merged:
            wire["text"] = {"format": _responses_text_format(merged["response_format"])}

        for name in _LLAMA_SWAP_IGNORED_SETTINGS:
            # The base merge already stripped None, so presence means the app set a value.
            if name in merged and name not in self._warned_settings:
                # stacklevel 3 points at the caller of chat / chat_json (the app, or generate),
                # not at this module; the per-instance set, not the location, gives "once" (DEC-003-6).
                warnings.warn(
                    f"Setting '{name}' has no counterpart in the Responses protocol and was not sent",
                    LLMSettingIgnoredWarning,
                    stacklevel=3,
                )
                self._warned_settings.add(name)

        # DEC-003-2 / DEC-003-1: "thinking set" means the key survived the merge (None stripped).
        effort = merged.get("reasoning_effort")
        if "enable_thinking" in merged:
            default_effort = "medium" if merged["enable_thinking"] else "none"
            wire["reasoning"] = {"effort": effort if effort is not None else default_effort}
        elif effort is not None:
            wire["reasoning"] = {"effort": effort}
        return wire

    def _responses_payload(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None,
        settings: dict[str, Any],
        stream: bool,
    ) -> dict[str, Any]:
        """Build a Responses request body (DEC-003-11); shared by the non-streamed and streamed paths.

        Args:
            messages: Caller messages; never modified.
            system: System instruction; a truthy one becomes ``{"role": "system", "content": system}``
                at the head of the input; an empty or ``None`` one adds nothing.
            settings: Wire fields as returned by ``_merge_options`` (possibly adjusted by the caller,
                e.g. ``chat_json``'s ``text`` fallback), spread into the body.
            stream: Value of the always-present ``stream`` key.

        Returns:
            A new body holding exactly ``model``, ``input`` (one item per message, values as given,
            in order: ``{"type": "message", "role": "assistant", "content"}`` for an assistant
            message, ``{"role", "content"}`` for any other; no other message keys), ``stream``,
            ``store: False``,
            ``include: ["reasoning.encrypted_content"]``, plus ``settings``. No ``instructions``.
        """
        # DEC-003-11 (amended 2026-11-08): llama.cpp refuses an untyped assistant item with 400
        # "Cannot determine type of 'item'", so assistant turns are typed message items; system and
        # user items stay bare role/content. No other message keys are copied.
        items: list[dict[str, Any]] = [
            {"type": "message", "role": "assistant", "content": m["content"]}
            if m["role"] == "assistant"
            else {"role": m["role"], "content": m["content"]}
            for m in messages
        ]
        if system:
            items.insert(0, {"role": "system", "content": system})
        # Fixed keys first; settings come from _merge_options and never contain these names.
        return {
            "model": self.model,
            "input": items,
            "stream": stream,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            **settings,
        }

    def _responses_tool_input(
        self,
        history: list[dict[str, Any]],
        carried: dict[int, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Convert the loop's internal copy into one round's ``input`` items (DEC-004-6, -7, -4).

        By turn shape, in order:

        - assistant turn with ``tool_calls`` → the carry items ``carried[<turn position>]`` (verbatim,
          in order; none when the position is absent), then
          ``{"type": "message", "role": "assistant", "content": <content>}`` only when its content is
          non-empty text, then one ``{"type": "function_call", "call_id": <id>, "name": <name>,
          "arguments": <text>}`` per call in order (mapping arguments as JSON text, missing as ``""``;
          no server item id);
        - ``role: tool`` turn → ``{"type": "function_call_output", "call_id": <tool_call_id>,
          "output": <content as text>}``;
        - any other turn → the item ``_responses_payload`` makes for it (assistant typed ``message``;
          system and user bare ``{"role", "content"}``).

        Args:
            history: Internal copy in chat-completions turn shapes (system turn first when given);
                never modified.
            carried: Per-call record of reasoning items keyed by the position in ``history`` of the
                assistant tool-call turn they belong to; never modified.

        Returns:
            A new list of input items; every item built here carries its ``type``.
        """
        items: list[dict[str, Any]] = []
        for position, turn in enumerate(history):
            tool_calls = turn.get("tool_calls")
            if turn.get("role") == "assistant" and tool_calls:
                # DEC-004-7: the round's reasoning goes ahead of its calls, as received; a caller
                # history turn has no record and is sent without reasoning (DEC-004-6).
                items.extend(carried.get(position, []))
                content = turn.get("content")
                # DEC-004-4: the round text travels as a typed message, and only when non-empty.
                if isinstance(content, str) and content:
                    items.append({"type": "message", "role": "assistant", "content": content})
                for call in tool_calls:
                    function = call.get("function") if isinstance(call, Mapping) else None
                    if not isinstance(function, Mapping):
                        function = {}
                    arguments = function.get("arguments")
                    # DEC-004-6: arguments travel as text — a mapping (caller history) as JSON
                    # text, missing as "", text as given.
                    if arguments is None:
                        arguments = ""
                    elif isinstance(arguments, Mapping):
                        arguments = json.dumps(dict(arguments), ensure_ascii=False)
                    elif not isinstance(arguments, str):
                        arguments = str(arguments)
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call.get("id") if isinstance(call, Mapping) else None,
                            "name": function.get("name"),
                            "arguments": arguments,
                        }
                    )
            elif turn.get("role") == "tool":
                output = turn.get("content")
                # The output must be text: a raw JSON object gets 400 (context wire vocabulary).
                # The loop's own turns already hold text; a caller's structured content is JSON text.
                if output is None:
                    output = ""
                elif isinstance(output, (dict, list)):
                    output = json.dumps(output, ensure_ascii=False, default=str)
                elif not isinstance(output, str):
                    output = str(output)
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": turn.get("tool_call_id"),
                        "output": output,
                    }
                )
            else:
                # Delegated so caller turns get exactly the chat path's item rule (DEC-003-11 as
                # amended, DEC-004-4) without a second copy of it in this module.
                single: list[Any] = [turn]
                items.extend(self._responses_payload(single, system=None, settings={}, stream=False)["input"])
        return items

    def _responses_tools_payload(
        self,
        items: list[dict[str, Any]],
        *,
        settings: dict[str, Any],
        tools: list[dict[str, Any]],
        stream: bool,
    ) -> dict[str, Any]:
        """Build one tool-round request body (DEC-004-10, DEC-003-11).

        Args:
            items: The round's input items (from ``_responses_tool_input``), sent as ``input``.
            settings: Wire fields from ``_merge_options``, with ``text`` already set to the winning
                translated response format (parameter over setting) or absent.
            tools: Flattened tools (from ``_responses_tools``).
            stream: Value of the always-present ``stream`` key.

        Returns:
            A new body holding exactly ``model``, ``input``, ``stream``, ``store: False``,
            ``include: ["reasoning.encrypted_content"]``, ``tools`` and the ``settings`` fields.
        """
        # DEC-003-11 keys first, then `tools`; settings come from _merge_options and never hold these names.
        return {
            "model": self.model,
            "input": items,
            "stream": stream,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "tools": tools,
            **settings,
        }

    async def _stream_responses_response(
        self,
        response: aiohttp.ClientResponse,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Read an open Responses event stream, deliver its pieces and return the full delivered text.

        llama-swap's own copy of the Responses stream parser (DEC-arch-2); handling per DEC-003-13.
        Payload strings are read only through ``self._iter_sse_data(response)`` and decoded as JSON;
        non-JSON payloads (including a ``[DONE]`` sentinel), non-object values and objects without a
        string ``type`` are skipped. Dispatch on ``type``:

        - ``response.output_text.delta`` → text piece (``delta``).
        - ``response.reasoning_text.delta`` / ``response.reasoning_summary_text.delta`` → reasoning
          piece (``delta``); no separator between reasoning or summary parts.
        - ``response.completed`` → complete; returns at once, reads nothing further.
        - ``response.failed`` → ``LLMError`` described by ``response.error.message`` when a string,
          else ``response.error`` as JSON text; ``response.incomplete`` → ``LLMError`` described by
          ``response.incomplete_details.reason`` when present, else that value as JSON text;
          ``error`` → ``LLMError`` described by the event's ``message`` when a string, else the event
          as JSON text. No piece is delivered after a failure event.
        - Any other type is ignored. Empty or missing deltas deliver nothing.

        Markers per run of reasoning (DEC-000-20): the first reasoning piece after text (or at the
        start) is prefixed with ``<think>``; the first text piece after reasoning is prefixed with
        ``</think>``; a block still open at ``response.completed`` is closed by delivering
        ``</think>`` as its own piece. Each piece goes to ``on_delta`` as it is parsed (result awaited
        only if awaitable, DEC-000-23) and is appended to the returned text.

        Args:
            response: The open streamed response (from ``_stream_post``); only passed to
                ``_iter_sse_data``.
            on_delta: Optional receiver of pieces; its failure propagates unchanged and ends the read.

        Returns:
            The concatenation of every delivered piece, markers included (equals what the receiver
            got; the full text when there is no receiver).

        Raises:
            LLMError: A failed, incomplete or error event (carrying the server's reason); the stream
                ended without ``response.completed`` (also after ``[DONE]``), saying it ended before
                the reply was complete (DEC-001-1); a reader failure (DEC-001-7).
        """
        text = ""
        # True while a <think> block has been opened and not yet closed (DEC-000-20, per run).
        in_reasoning = False

        async def _deliver(piece: str) -> None:
            # Append before the receiver runs so the return always equals what was handed out;
            # a receiver failure propagates unchanged and ends the read (DEC-000-23).
            nonlocal text
            text += piece
            if on_delta is not None:
                outcome = on_delta(piece)
                if inspect.isawaitable(outcome):
                    await outcome

        async for data in self._iter_sse_data(response):
            # Non-JSON payloads (including a `[DONE]` sentinel), non-objects and objects without a
            # string `type` carry no event: skipped (DEC-003-13).
            try:
                event = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("type")
            if not isinstance(kind, str):
                continue

            if kind == "response.output_text.delta":
                delta = event.get("delta")
                if not isinstance(delta, str) or not delta:
                    continue
                if in_reasoning:
                    delta = "</think>" + delta
                    in_reasoning = False
                await _deliver(delta)
            elif kind in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
                delta = event.get("delta")
                if not isinstance(delta, str) or not delta:
                    continue
                if not in_reasoning:
                    delta = "<think>" + delta
                    in_reasoning = True
                await _deliver(delta)
            elif kind == "response.completed":
                # The protocol's only completion signal: close an open block as its own piece and
                # return without reading further, so nothing after it is delivered.
                if in_reasoning:
                    await _deliver("</think>")
                return text
            elif kind == "response.failed":
                body = event.get("response")
                error = body.get("error") if isinstance(body, dict) else None
                message = error.get("message") if isinstance(error, dict) else None
                raise LLMError(
                    "llama-swap reported a failed response",
                    description=message if isinstance(message, str) else json.dumps(error),
                )
            elif kind == "response.incomplete":
                body = event.get("response")
                details = body.get("incomplete_details") if isinstance(body, dict) else None
                reason = details.get("reason") if isinstance(details, dict) else None
                raise LLMError(
                    "llama-swap reported an incomplete response",
                    description=reason if isinstance(reason, str) and reason else json.dumps(details),
                )
            elif kind == "error":
                message = event.get("message")
                raise LLMError(
                    "llama-swap reported an error in the stream",
                    description=message if isinstance(message, str) else json.dumps(event),
                )
            # Every other type (created, in_progress, output_item.*, content_part.*, *.done, ...) is
            # ignored; the *.done events repeat the deltas (DEC-003-13 rejected reading them).

        # The Responses form of DEC-001-1: without `response.completed` the reply is truncated,
        # even when a `[DONE]` sentinel arrived (llama.cpp never sends one).
        raise LLMError("llama-swap stream ended before the reply was complete")

    async def _stream_responses_tool_round(
        self,
        response: aiohttp.ClientResponse,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        """Read one streamed tool round's Responses events into round text, tool calls and carry items.

        llama-swap's own tool-round parser (DEC-004-9, DEC-arch-2), separate from
        ``_stream_responses_response`` (which is not changed). Payload strings are read only through
        ``self._iter_sse_data(response)`` and decoded as in DEC-003-13: non-JSON payloads (including a
        ``[DONE]`` sentinel), non-object values and objects without a string ``type`` are skipped.
        Dispatch on ``type``:

        - ``response.output_text.delta`` → text piece; ``response.reasoning_text.delta`` /
          ``response.reasoning_summary_text.delta`` → reasoning piece. Markers per DEC-003-13 with
          fresh marker state for this round; a block still open at ``response.completed`` is closed by
          delivering ``</think>`` as its own piece, so the round's reasoning reaches the receiver
          closed. Empty or missing deltas deliver nothing.
        - ``response.output_item.added`` with a ``function_call`` item → starts a call keyed by the
          item's ``id`` (fallback ``output_index``), taking ``call_id``, ``name`` and any initial
          ``arguments``.
        - ``response.function_call_arguments.delta`` → appends ``delta`` to the call keyed by
          ``item_id`` (fallback ``output_index``); a delta for an unknown key starts a call with no
          name.
        - ``response.function_call_arguments.done`` and ``response.output_item.done`` with a
          ``function_call`` item → string ``arguments`` replace the assembled text; a non-empty
          ``call_id`` / ``name`` on the done item is set; a done item for an unknown key starts the
          call.
        - ``response.output_item.done`` with a ``reasoning`` item → carry item, kept verbatim and
          recorded with its ``output_index`` (DEC-004-7).
        - ``response.completed`` → the round is complete; returns at once, reads nothing further.
        - ``response.failed``, ``response.incomplete``, ``error`` → ``LLMError`` carrying the
          server's reason, as in DEC-003-13. Every other type is ignored.

        Handles the llama.cpp sequence: created, in progress; reasoning item added at position 0 and
        reasoning deltas; function-call item added at position 1 with name, call id and empty
        arguments, then argument deltas; items finishing out of order (the reasoning item's done event
        may arrive after the call's deltas or after text deltas); no
        ``response.function_call_arguments.done`` event; ``response.completed`` with no ``[DONE]``.

        Pieces are delivered to ``on_delta`` as parsed (result awaited only if awaitable,
        DEC-000-23). Tool-call names and argument text are never delivered to the receiver.

        Args:
            response: The open streamed response (from ``_stream_post``); only passed to
                ``_iter_sse_data``.
            on_delta: Optional receiver of text and reasoning pieces; its failure propagates
                unchanged and ends the read (no later piece is delivered).

        Returns:
            A triple, same shape as ``_responses_tool_round``:
            the round text — text deltas only, joined, without reasoning or markers;
            the tool calls as ``{"call_id", "name", "arguments"}`` dicts in order of first
            appearance, numbered densely from 0 (missing values → ``None``; DEC-arch-10);
            the reasoning carry items, verbatim, ordered by ``output_index`` regardless of the order
            in which they finished.

        Raises:
            LLMError: A failed, incomplete or error event (carrying the server's reason); the stream
                ended without ``response.completed`` (also after a ``[DONE]`` sentinel); a reader
                failure from ``_iter_sse_data``.
        """
        text = ""
        # True while a <think> block is open; fresh for every round so each round's reasoning is its
        # own closed block (DEC-004-9, markers per DEC-003-13 / DEC-000-20).
        in_reasoning = False
        # Calls in order of first appearance (dense 0-based numbering, DEC-arch-10), each found by
        # the server's item id and by its output position: llama.cpp deltas carry both, but either
        # alone must still reach the call (DEC-004-9: key `id` / `item_id`, fallback `output_index`).
        calls: list[dict[str, Any]] = []
        by_id: dict[str, dict[str, Any]] = {}
        by_index: dict[int, dict[str, Any]] = {}
        # Reasoning carry items with their output position and arrival order; sorted at return
        # because reasoning's done event can arrive after the call's deltas (DEC-004-9, DEC-004-7).
        carry: list[tuple[float, int, dict[str, Any]]] = []

        async def _deliver(piece: str) -> None:
            # Receiver awaited only if awaitable; a failure propagates unchanged (DEC-000-23).
            if on_delta is not None:
                outcome = on_delta(piece)
                if inspect.isawaitable(outcome):
                    await outcome

        def _find_call(item_id: Any, index: Any) -> dict[str, Any] | None:
            # Item id first, output position as the fallback (DEC-004-9).
            if isinstance(item_id, str) and item_id in by_id:
                return by_id[item_id]
            if isinstance(index, int) and index in by_index:
                return by_index[index]
            return None

        def _start_call(item_id: Any, index: Any) -> dict[str, Any]:
            # A call first seen through a delta or done event has no name yet; the offered check
            # in `chat_with_tools` then ends the loop with LLMError (DEC-004-9, DEC-002-3).
            call: dict[str, Any] = {"call_id": None, "name": None, "arguments": None}
            calls.append(call)
            if isinstance(item_id, str) and item_id:
                by_id[item_id] = call
            if isinstance(index, int):
                by_index[index] = call
            return call

        def _register(call: dict[str, Any], item_id: Any, index: Any) -> None:
            # Later events may carry a key the first one lacked; remember it so both lookups work.
            if isinstance(item_id, str) and item_id and item_id not in by_id:
                by_id[item_id] = call
            if isinstance(index, int) and index not in by_index:
                by_index[index] = call

        def _set_identity(call: dict[str, Any], source: Mapping[str, Any]) -> None:
            # Only non-empty text replaces what is known: an empty field on a done event must not
            # erase the name / call id taken from the added event (DEC-004-9).
            for key in ("call_id", "name"):
                value = source.get(key)
                if isinstance(value, str) and value:
                    call[key] = value

        async for data in self._iter_sse_data(response):
            # Non-JSON payloads (including a `[DONE]` sentinel), non-objects and objects without a
            # string `type` carry no event: skipped (DEC-003-13).
            try:
                event = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("type")
            if not isinstance(kind, str):
                continue

            if kind == "response.output_text.delta":
                delta = event.get("delta")
                if not isinstance(delta, str) or not delta:
                    continue
                # The round text keeps the bare delta; only the receiver's piece carries the marker.
                text += delta
                piece = delta
                if in_reasoning:
                    piece = "</think>" + delta
                    in_reasoning = False
                await _deliver(piece)
            elif kind in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
                delta = event.get("delta")
                if not isinstance(delta, str) or not delta:
                    continue
                if not in_reasoning:
                    delta = "<think>" + delta
                    in_reasoning = True
                await _deliver(delta)
            elif kind == "response.output_item.added":
                item = event.get("item")
                if not isinstance(item, dict) or item.get("type") != "function_call":
                    continue
                item_id, index = item.get("id"), event.get("output_index")
                call = _find_call(item_id, index) or _start_call(item_id, index)
                _register(call, item_id, index)
                _set_identity(call, item)
                arguments = item.get("arguments")
                if isinstance(arguments, str) and call["arguments"] is None:
                    call["arguments"] = arguments
            elif kind == "response.function_call_arguments.delta":
                delta = event.get("delta")
                item_id, index = event.get("item_id"), event.get("output_index")
                call = _find_call(item_id, index) or _start_call(item_id, index)
                if isinstance(delta, str):
                    call["arguments"] = (call["arguments"] or "") + delta
            elif kind in ("response.function_call_arguments.done", "response.output_item.done"):
                index = event.get("output_index")
                if kind == "response.output_item.done":
                    item = event.get("item")
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "reasoning":
                        # Kept verbatim for the later rounds (DEC-004-7); the done event is the one
                        # source of carry items (DEC-004-9 rejected reading `response.completed`).
                        # An item without a position sorts after the positioned ones, in arrival order.
                        position = index if isinstance(index, int) else float("inf")
                        carry.append((position, len(carry), item))
                        continue
                    if item.get("type") != "function_call":
                        continue
                    source: Mapping[str, Any] = item
                    item_id = item.get("id")
                else:
                    source = event
                    item_id = event.get("item_id")
                call = _find_call(item_id, index) or _start_call(item_id, index)
                _register(call, item_id, index)
                _set_identity(call, source)
                # Full arguments replace the assembled deltas (DEC-004-9); llama.cpp sends them only
                # on the item's done event, never an arguments-done event.
                arguments = source.get("arguments")
                if isinstance(arguments, str):
                    call["arguments"] = arguments
            elif kind == "response.completed":
                # The round's only completion signal: close an open block as its own piece so the
                # next round's pieces never land inside it, and return without reading further.
                if in_reasoning:
                    await _deliver("</think>")
                reasoning = [item for _, _, item in sorted(carry, key=lambda entry: (entry[0], entry[1]))]
                return text, calls, reasoning
            elif kind == "response.failed":
                body = event.get("response")
                error = body.get("error") if isinstance(body, dict) else None
                message = error.get("message") if isinstance(error, dict) else None
                raise LLMError(
                    "llama-swap reported a failed response",
                    description=message if isinstance(message, str) else json.dumps(error),
                )
            elif kind == "response.incomplete":
                body = event.get("response")
                details = body.get("incomplete_details") if isinstance(body, dict) else None
                reason = details.get("reason") if isinstance(details, dict) else None
                raise LLMError(
                    "llama-swap reported an incomplete response",
                    description=reason if isinstance(reason, str) and reason else json.dumps(details),
                )
            elif kind == "error":
                message = event.get("message")
                raise LLMError(
                    "llama-swap reported an error in the stream",
                    description=message if isinstance(message, str) else json.dumps(event),
                )
            # Every other type (created, in_progress, content_part.*, output_text.done, ...) is ignored.

        # Without `response.completed` the round is truncated, even after a `[DONE]` sentinel
        # (DEC-004-9); no tool of this round runs.
        raise LLMError("llama-swap stream ended before the tool round was complete")

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
                returned as bytes); ``data`` is missing (existing "Invalid response from llama-swap
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
                returned as bytes); ``data`` is missing (existing "Invalid response from llama-swap
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
        """Send one Responses request and return the answer text.

        Merges settings (``_merge_options``, may warn), builds the body with
        ``_responses_payload(messages, system=system, settings=opt, stream=stream)``, and inside
        ``self._tracer.trace("chat", self.model)`` logs ``log_system``, ``log_messages`` (the input
        items) and ``log_request("POST", "/responses", payload)``. Non-streamed: sends
        ``self._request("POST", "/responses", json=payload)`` (DEC-003-16), extracts the text with
        ``_responses_reply_text``, logs it with ``log_response`` and returns it. Streamed: the same
        body with ``stream`` true, opened with ``async with self._stream_post("/responses", payload)``
        (DEC-001-8, DEC-003-16) and read by ``_stream_responses_response`` inside the trace session;
        the receiver is wrapped in a local ``_tracing_delta`` (``t.log_chunk`` then the receiver) only
        when ``self._tracer.enabled`` (DEC-000-48); the parsed text is logged with ``log_response``
        and returned. Never calls ``/chat/completions``.

        Args:
            messages: Conversation turns; never modified.
            system: Optional system instruction, sent as the first input item when truthy.
            options: Per-call settings.
            stream: Stream the reply as Responses events (DEC-003-13).
            on_delta: Optional receiver of streamed pieces (reasoning marked); its failure propagates
                unchanged.

        Returns:
            The answer text (never ``None``; ``""`` when the reply has no text).

        Raises:
            LLMError: Non-2xx reply (address, status, server reply; never the credential), an
                unreachable server, or a ``failed`` / ``incomplete`` reply; streamed, also a failed /
                incomplete / error event or a stream ending without ``response.completed``
                (DEC-003-13); raised inside the trace session.
        """
        opt = self._merge_options(options=options)
        payload = self._responses_payload(messages, system=system, settings=opt, stream=stream)

        with self._tracer.trace("chat", self.model) as t:
            t.log_system(system)
            t.log_messages(payload["input"])
            t.log_request("POST", "/responses", payload)

            if not stream:
                reply = await self._request("POST", "/responses", json=payload)
                result = _responses_reply_text(reply)
                t.log_response(result)
                return result

            async def _tracing_delta(text: str) -> None:
                t.log_chunk(text)
                if on_delta is not None:
                    outcome = on_delta(text)
                    if inspect.isawaitable(outcome):
                        await outcome

            # DEC-000-48: the receiver is wrapped only when tracing is on.
            receiver = _tracing_delta if self._tracer.enabled else on_delta
            async with self._stream_post("/responses", payload) as response:
                result = await self._stream_responses_response(response, receiver)
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
        """Answer one prompt: a one-message ``chat`` with ``[{"role": "user", "content": prompt}]`` (DEC-000-8).

        Args:
            prompt: The user prompt.
            system: Optional system instruction, sent as the first input item when truthy.
            options: Per-call settings.
            stream: Stream the reply (through ``chat``).
            on_delta: Receiver of streamed pieces (through ``chat``).

        Returns:
            What ``chat`` returns.

        Raises:
            LLMError: As ``chat``.
        """
        messages: List[LLMMessage] = [{"role": "user", "content": prompt}]
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
        """Request a JSON answer over Responses and return its text.

        Non-streamed (``stream`` false) ``POST /responses`` through ``self._request`` inside
        ``self._tracer.trace("chat_json", self.model)``, body from ``_responses_payload``. The text
        format is the translated ``response_format`` setting when the app set one, otherwise
        ``text: {"format": {"type": "json_object"}}`` (DEC-001-2, DEC-003-10). Text extracted by
        ``_responses_reply_text``. When ``response_schema`` is truthy the only check, after the
        trace session, is that the text parses as JSON (DEC-000-37, DEC-002-7): any valid JSON value
        is returned unchanged; the schema is not sent and its fields are not checked.

        Args:
            messages: Conversation turns; never modified.
            response_schema: When truthy, enables the JSON parse check.
            system: Optional system instruction, sent as the first input item when truthy.
            options: Per-call settings.

        Returns:
            The answer text (never ``None``).

        Raises:
            LLMError: Request failure, a ``failed`` / ``incomplete`` reply, or the text is not valid
                JSON while ``response_schema`` is given.
        """
        opt = self._merge_options(options=options)
        # DEC-001-2 / DEC-003-10: the app's translated response_format wins; JSON object otherwise.
        if "text" not in opt:
            opt["text"] = {"format": {"type": "json_object"}}
        payload = self._responses_payload(messages, system=system, settings=opt, stream=False)

        with self._tracer.trace("chat_json", self.model) as t:
            t.log_system(system)
            t.log_messages(payload["input"])
            t.log_request("POST", "/responses", payload)
            reply = await self._request("POST", "/responses", json=payload)
            content = _responses_reply_text(reply)
            t.log_response(content)

        # DEC-000-37 / DEC-002-7: parse check only, after the trace session, schema not enforced.
        if response_schema:
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                raise LLMError(
                    "LLM response was not valid JSON, despite requesting JSON mode.",
                    description=str(e),
                ) from e
        return content

    async def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        *,
        tools_definitions: List[Dict[str, Any]],
        tools: Dict[str, Callable[..., Any]],
        system: Optional[str] = None,
        max_loops: int = 5,
        options: Optional[OpenAIOptions] = None,
        stream: bool = False,
        on_delta: Optional[Callable[[str], Awaitable[None]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Run a tool conversation over Responses and return the final answer text (DEC-004-10, DEC-004-11).

        llama-swap's own copy of the tool loop (DEC-arch-2). Order per call (002's rules):
        pre-flight — every advertised name (``_tool_definition_name``) needs a callable in ``tools``,
        else ``ValueError`` before any request; settings merged (``_merge_options``, may warn once per
        client); internal copy of ``messages`` (system turn first when truthy); then inside
        ``self._tracer.trace("chat_with_tools", self.model)`` (``log_system``, ``log_messages``), up
        to ``max_loops`` rounds, each one ``POST /responses`` through ``self._request`` with body
        ``_responses_tools_payload(_responses_tool_input(copy, carried), ...)`` — flattened tools,
        ``stream`` false, ``text.format`` from the ``response_format`` parameter (else the setting)
        translated per DEC-003-10, no ``text`` with neither; logged as
        ``log_request("POST", "/responses (loop <n>)", payload)`` with n from 1. The reply is read by
        ``_responses_tool_round``. No calls → ``log_response(text)`` and return. Calls → the
        assistant tool-call turn is appended to the copy and the round's reasoning items recorded
        under its position (call-local, never on messages); per call in order: offered check
        (``_offered_tool_name``), ``_read_tool_arguments``, ``log_tool_call``, argument-name check,
        run (awaited only if awaitable), ``_tool_result_text``, ``log_tool_result``, tool turn.
        The final round's reasoning is never kept. The bearer header is sent only when the client
        holds a credential. ``stream=True`` opens each round with ``_stream_post`` and reads it with
        ``_stream_responses_tool_round`` (step 004), the receiver passed unwrapped.

        Args:
            messages: Conversation turns; never modified (tool-call / tool-result turns in it are
                sent as function-call / function-call-output items).
            tools_definitions: Advertised tools, nested or flat; never modified.
            tools: Callables by tool name.
            system: Optional system instruction.
            max_loops: Round limit.
            options: Per-call settings.
            stream: Streamed rounds (step 004).
            on_delta: Receiver of streamed pieces (step 004); not trace-wrapped.
            response_format: Final-answer format; wins over the ``response_format`` setting.

        Returns:
            The final round's text (no reasoning, no markers).

        Raises:
            ValueError: An advertised tool has no callable (before any request); an argument name the
                callable lacks (tool not run).
            LLMError: Non-2xx reply (address and server reply) or transport failure; a ``failed`` /
                ``incomplete`` round; a call naming an unoffered tool or with an empty / missing
                name; arguments not JSON or not an object; an unserialisable result — each naming
                the tool where there is one.
            RuntimeError: A tool raised (names the tool; no further request); the round limit was
                reached — text contains ``max_loops`` and its value, raised after the trace session.
        """
        # DEC-004-11 / DEC-004-5: pre-flight before merge and request, names nested or flat.
        advertised = {_tool_definition_name(d) for d in tools_definitions}
        missing = advertised - tools.keys()
        if missing:
            # key=str: a definition without a name contributes None, which cannot sort among text.
            raise ValueError(f"Missing callables for tools {sorted(missing, key=str)}")
        offered = {name for name in advertised if isinstance(name, str)}

        # DEC-004-12: merged once per call, so a no-counterpart setting warns at most once and is
        # absent from every round.
        settings = self._merge_options(options=options)
        # DEC-004-10 / DEC-001-15: a truthy parameter wins over the setting on every round; with
        # neither, `settings` holds no `text`.
        if response_format:
            settings["text"] = {"format": _responses_text_format(response_format)}
        flat_tools = _responses_tools(tools_definitions)

        # Internal copy in chat-completions turn shapes (DEC-004-6); the caller's list is untouched.
        msgs: list[dict[str, Any]] = list(messages)
        if system:
            msgs.insert(0, {"role": "system", "content": system})
        # DEC-004-7: reasoning items of each tool round, keyed by the position of that round's
        # assistant tool-call turn in `msgs`; local to this call, discarded when it ends.
        carried: dict[int, list[dict[str, Any]]] = {}

        with self._tracer.trace("chat_with_tools", self.model) as t:
            t.log_system(system)
            t.log_messages(msgs)

            for loop_idx in range(max_loops):
                items = self._responses_tool_input(msgs, carried)
                payload = self._responses_tools_payload(items, settings=settings, tools=flat_tools, stream=stream)
                # DEC-004-10: round number in the path, counted from 1.
                t.log_request("POST", f"/responses (loop {loop_idx + 1})", payload)

                if stream:
                    # Step 004 parser; the receiver is passed raw (no trace wrapper on the tool path).
                    async with self._stream_post("/responses", payload) as response:
                        text, calls, reasoning = await self._stream_responses_tool_round(response, on_delta)
                else:
                    reply = await self._request("POST", "/responses", json=payload)
                    text, calls, reasoning = _responses_tool_round(reply)

                if not calls:
                    # The final round's reasoning is dropped here, never sent (DEC-004-7).
                    t.log_response(text)
                    return text

                carried[len(msgs)] = reasoning
                msgs.append(
                    {
                        "role": "assistant",
                        "content": text,
                        "tool_calls": [
                            {
                                "id": call["call_id"],
                                "type": "function",
                                "function": {"name": call["name"], "arguments": call["arguments"]},
                            }
                            for call in calls
                        ],
                    }
                )

                for call in calls:
                    # DEC-004-11 order: offered check, then argument reading, both before the tool can
                    # run; calls earlier in this reply have already run and keep their effects.
                    name = _offered_tool_name(call, offered)
                    func = tools[name]
                    kwargs = _read_tool_arguments(name, call.get("arguments"))
                    t.log_tool_call(name, kwargs)
                    if not all(k in inspect.signature(func).parameters for k in kwargs):
                        raise ValueError(f"Bad args for tool '{name}': {kwargs}")
                    try:
                        result = func(**kwargs)
                        if inspect.isawaitable(result):
                            result = await result
                    except Exception as exc:
                        raise RuntimeError(f"Tool '{name}' failed: {exc}") from exc

                    # Outside the try above, so a conversion failure stays LLMError and is not
                    # reported as the tool's own failure (DEC-002-5).
                    content = _tool_result_text(name, result)
                    t.log_tool_result(name, content)
                    msgs.append({"role": "tool", "content": content, "tool_call_id": call.get("call_id")})

        # DEC-002-2: stays RuntimeError and outside the trace session; the text keeps `max_loops`.
        raise RuntimeError(f"Tool loop reached its round limit (max_loops={max_loops}) without a final answer")

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
                returned as bytes); ``data`` is missing (existing "Invalid response from llama-swap
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
