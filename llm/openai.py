"""OpenAI Responses client: chat, generate and JSON-only answers over ``POST /responses``.

Feature 003: settings translated to Responses fields and no-counterpart settings dropped
with ``LLMSettingIgnoredWarning`` (DEC-003-6, DEC-003-9, DEC-003-10); request body and non-streamed
reply handling per DEC-003-11 / DEC-003-12; streamed ``chat`` / ``generate`` parse the
Responses event stream (DEC-003-13). Feature 004: ``chat_with_tools`` runs the tool loop
on Responses (flattened tools, call / result items, reasoning carried between rounds; DEC-004-5..11),
replacing the DEC-003-4 refusal; streamed tool rounds are read by a separate tool-round
parser (DEC-004-9). Model listing and
embeddings are unchanged from the chat-completions client (US-057.AC-2), which now lives in
``llm/openai_chat_completions.py``.
"""

from collections.abc import Mapping
from typing import Any, Awaitable, Callable, Dict, List, NamedTuple, Optional
import json
import inspect
import warnings

import aiohttp

from llm.llm_client import LLMClient
from llm.exceptions import LLMError, LLMSettingIgnoredWarning
from llm.message import LLMMessage
from llm.options import OpenAIOptions

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

# Supported settings that have no Responses field (DEC-003-9): never sent; the first time a client
# sees one in its merged settings it emits ``LLMSettingIgnoredWarning`` (DEC-003-6).
_OPENAI_NO_COUNTERPART_PARAMS: frozenset[str] = frozenset(
    {"presence_penalty", "frequency_penalty", "seed", "stop", "logit_bias"}
)


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
    # a broad ValueError catch could swallow unrelated errors. ``_request`` is not edited.
    except (json.JSONDecodeError, aiohttp.ContentTypeError) as exc:
        raise LLMError(f"Unreadable reply from OpenAI API when {operation}", description=str(exc)) from exc


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
            f"Invalid response from OpenAI API when {operation}: the reply is not a JSON object",
            description=_reply_text(response),
        )
    if "data" not in response:
        # Wording kept from before DEC-002-8 / DEC-002-9; older tests may match it.
        raise LLMError(f"Invalid response from OpenAI API when {operation}: {response}")
    entries = response["data"]
    if not isinstance(entries, list):
        raise LLMError(
            f"Invalid response from OpenAI API when {operation}: 'data' is not a list",
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
                f"Invalid response from OpenAI API when {operation}: model entry {position} is not an object",
                description=_reply_text(response),
            )
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id:
            raise LLMError(
                f"Invalid response from OpenAI API when {operation}: model entry {position} has no non-empty text id",
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
            f"Invalid response from OpenAI API when {operation}: entry {position} {problem}",
            description=_reply_text(response),
        )
    if len(entries) != count:
        raise LLMError(
            f"Invalid response from OpenAI API when {operation}: {len(entries)} vectors returned for {count} texts",
            description=_reply_text(response),
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
        )
    ordered: list[Any] = [None] * count
    for index, entry in zip(indices, entries):
        ordered[index] = entry["embedding"]
    return ordered


def _responses_text_format(response_format: Any) -> Any:
    """Translate an app-facing ``response_format`` value into the Responses text format (DEC-003-10).

    The result is what the request carries under ``text: {"format": <result>}``.

    - ``{"type": "json_object"}`` -> ``{"type": "json_object"}``.
    - ``{"type": "json_schema", "json_schema": {...}}`` -> ``{"type": "json_schema", ...}``: every key
      of the nested mapping (``name``, ``schema``, ``strict``, ``description``, whatever is present)
      is lifted beside ``type``.
    - ``{"type": "text"}`` -> ``{"type": "text"}``.
    - Any other value (including a format already in the flat Responses shape) is returned unchanged.

    The input is never modified.

    Args:
        response_format: The merged ``response_format`` setting, as the app set it.

    Returns:
        The Responses text-format value.
    """
    if not isinstance(response_format, Mapping):
        return response_format
    if response_format.get("type") == "json_schema":
        nested = response_format.get("json_schema")
        # Only the chat-completions nested form is flattened; a json_schema format without a nested
        # mapping is taken to be already in the flat Responses shape and passes through (DEC-003-10).
        if isinstance(nested, Mapping):
            # ``type`` stays "json_schema" even if the nested mapping happens to carry its own.
            return {"type": "json_schema", **{k: v for k, v in nested.items() if k != "type"}}
    # json_object, text and every other shape are already valid Responses formats: unchanged.
    return response_format


def _responses_reply_text(response: Any, *, output_limit_set: bool = False) -> str:
    """Return the answer text of a non-streamed Responses reply (DEC-003-12, DEC-arch-13).

    Visits every item of type ``message`` in the reply's ``output`` list in order and concatenates,
    with no separator, the ``text`` of each content part of type ``output_text``. Reasoning items and
    other part types (e.g. refusal) are ignored. A missing or non-list ``output``, no message item, or
    a null/missing ``text`` gives ``""``.

    When ``output_limit_set`` is true and the reply is ``incomplete`` with ``incomplete_details`` a
    mapping whose ``reason`` is exactly ``"max_output_tokens"`` (DEC-003-18), the reply was cut at the
    app's own output limit: the text is extracted and returned exactly as for a completed reply
    (DEC-arch-13, DEC-003-19). With the flag off (the default), or any other incomplete reason, the
    reply raises as listed under Raises (DEC-003-17).

    Args:
        response: The decoded reply body (expected to be a mapping).
        output_limit_set: Keyword-only; true when the calling operation's wire settings hold
            ``max_output_tokens``, i.e. the app set the output limit (DEC-003-17, DEC-003-20).

    Returns:
        The concatenated output text; ``""`` when there is none.

    Raises:
        LLMError: ``status`` is ``"failed"`` -- description is ``error.message`` when ``error`` is a
            mapping with a string ``message``, else the ``error`` value as JSON text; or ``status`` is
            ``"incomplete"`` -- description is ``incomplete_details.reason`` when it is a non-empty
            string, else the ``incomplete_details`` value as JSON text (DEC-003-21) -- unless
            ``output_limit_set`` is true and the reason is exactly ``"max_output_tokens"`` (then the
            text is returned, DEC-003-19). Any other
            or missing status extracts the text.
            A reply that is not a mapping surfaces as a generic error.
    """
    if not isinstance(response, Mapping):
        raise LLMError(
            "Invalid response from OpenAI API when chatting: the reply is not a JSON object",
            description=_reply_text(response),
        )
    status = response.get("status")
    if status == "failed":
        error = response.get("error")
        message = error.get("message") if isinstance(error, Mapping) else None
        reason = message if isinstance(message, str) else json.dumps(error, default=str)
        raise LLMError("OpenAI Responses request failed", description=reason)
    if status == "incomplete":
        details = response.get("incomplete_details")
        detail_reason = details.get("reason") if isinstance(details, Mapping) else None
        # DEC-arch-13 / DEC-003-18: only the exact string reason, and only when the app set the limit,
        # is a cut the app asked for; it falls through to the completed-reply extraction (DEC-003-19).
        if not (output_limit_set and isinstance(detail_reason, str) and detail_reason == "max_output_tokens"):
            # DEC-003-21: the reason is the description only as a non-empty string; otherwise the
            # details go as JSON text, so an empty or non-string reason never hides what the server sent.
            reason = (
                detail_reason
                if isinstance(detail_reason, str) and detail_reason
                else json.dumps(details, default=str)
            )
            # DEC-003-12: any other cut-off answer is never returned as if complete.
            raise LLMError("OpenAI Responses reply is incomplete", description=reason)

    output = response.get("output")
    if not isinstance(output, list):
        return ""
    pieces: list[str] = []
    for item in output:
        # Reasoning and any other item types are skipped, so no reasoning text appears off-stream
        # (US-028.AC-1).
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping) or part.get("type") != "output_text":
                continue
            text = part.get("text")
            # Null or missing text contributes nothing (the Responses form of DEC-001-12).
            if isinstance(text, str):
                pieces.append(text)
    return "".join(pieces)


class _ResponsesToolRound(NamedTuple):
    """What one non-streamed tool round's reply yields (DEC-004-8).

    Attributes:
        text: The round text: output-text parts of every ``message`` item concatenated per DEC-003-12
            (``""`` when none); never holds reasoning text or ``<think>`` markers.
        tool_calls: The round's function calls, dense and in order of appearance in ``output``, each in
            the internal assistant-turn shape of DEC-004-6:
            ``{"id": <call_id>, "type": "function", "function": {"name": <name>, "arguments": <arguments>}}``
            with ``call_id``, ``name`` and ``arguments`` taken as the server sent them (a missing value is
            ``None``). Empty when the reply holds no ``function_call`` item (the text is the answer).
        reasoning: The round's ``reasoning`` output items, verbatim and in order -- the carry items of
            DEC-004-7.
    """

    text: str
    tool_calls: list[dict[str, Any]]
    reasoning: list[dict[str, Any]]


def _advertised_tool_names(tools_definitions: list[dict[str, Any]]) -> set[str]:
    """Return the tool names the app advertised, reading either definition shape (DEC-004-5).

    The name is ``function.name`` for a nested entry (one with a ``function`` mapping) and ``name`` for
    a flat entry. Used by the pre-flight check (advertised without a callable -> ``ValueError``) and by
    the offered check (DEC-002-3). The definitions are not modified.

    Args:
        tools_definitions: The app's tool definitions, nested and/or flat.

    Returns:
        The set of advertised names.
    """
    names: set[str] = set()
    for definition in tools_definitions:
        function = definition.get("function") if isinstance(definition, Mapping) else None
        if isinstance(function, Mapping):
            name = function.get("name")
        else:
            name = definition.get("name") if isinstance(definition, Mapping) else None
        # Only text names can be requested by the model; anything else is left to the server.
        if isinstance(name, str):
            names.add(name)
    return names


def _flatten_responses_tools(tools_definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the app's tool definitions into the Responses ``tools`` list (DEC-004-5, DEC-004-1).

    Per entry, in order: an entry with a ``function`` mapping becomes
    ``{"type": "function", "name": ..., "description": ... (only when present), "parameters": ... (only
    when present), "strict": <function.strict when the app gave it, else False>}``; an entry without a
    ``function`` key (already flat) is sent unchanged. Accepts ``pydantic_to_openai_tool`` output. The
    app's list and its entries are never modified; the result is a new list.

    Args:
        tools_definitions: The app's tool definitions.

    Returns:
        The flattened tool list, sent as ``tools`` on every round.
    """
    flat_tools: list[dict[str, Any]] = []
    for definition in tools_definitions:
        function = definition.get("function") if isinstance(definition, Mapping) else None
        if not isinstance(function, Mapping):
            # Already flat (or a shape the server must judge): sent as given, DEC-004-5. A shallow
            # copy keeps the app's entry out of reach of anything that later touches the body.
            flat_tools.append(dict(definition) if isinstance(definition, Mapping) else definition)
            continue
        flat: dict[str, Any] = {"type": "function", "name": function.get("name")}
        # "Only when present" means the key exists; the value is passed through unvalidated.
        if "description" in function:
            flat["description"] = function["description"]
        if "parameters" in function:
            flat["parameters"] = function["parameters"]
        # DEC-004-1: Responses defaults to strict on OpenAI; false keeps chat-completions behaviour
        # unless the app asked for strict itself.
        flat["strict"] = function.get("strict", False)
        flat_tools.append(flat)
    return flat_tools


def _read_responses_tool_round(response: Any, *, output_limit_set: bool = False) -> _ResponsesToolRound:
    """Read one non-streamed tool-round reply into text, tool calls and carry items (DEC-004-8).

    The status is checked first exactly as in DEC-003-12 (with the output-limit exception of
    DEC-003-17 / DEC-003-18 handed on); then ``output`` is visited in order: ``function_call`` items
    become the round's calls (dense from 0), ``message`` items give the round text, ``reasoning`` items
    are kept verbatim; every other item type is ignored.

    A round cut at the app's own output limit (DEC-arch-13): with ``output_limit_set`` on and the
    reply ``incomplete`` for reason exactly ``"max_output_tokens"``, a round holding no
    ``function_call`` item is returned with its text and no tool calls, so the loop treats it as the
    answer; a round holding any ``function_call`` item raises ``LLMError`` with the server's reason,
    because its calls' arguments may be truncated (DEC-004-16). With no app limit (flag off), and for
    every other early end, the round raises as before (DEC-004-8).

    Args:
        response: The decoded reply body of ``POST /responses``.
        output_limit_set: Keyword-only; ``True`` when the app set the output limit for this call
            (DEC-004-15, computed once per call by ``chat_with_tools``). With it on, a round cut at the
            app's limit (DEC-003-18) is the answer when it holds no ``function_call`` item and raises
            ``LLMError`` with the server's reason when it holds any (DEC-arch-13, DEC-004-16). Default
            off: a cut round raises as before.

    Returns:
        The round's text, tool calls and reasoning items.

    Raises:
        LLMError: The reply is not a mapping, or its ``status`` is ``"failed"`` / ``"incomplete"``
            (description carries the server's reason, as in ``_responses_reply_text``) -- except an
            ``incomplete`` reply cut at the app's limit with ``output_limit_set`` on and no
            ``function_call`` item, which is returned (DEC-004-16).
    """
    # Status (and the mapping check) first, and the round text, by the very rule of the non-tool
    # path: DEC-004-8 reuses DEC-003-12, and the flag hands on the DEC-003-18 match (DEC-004-15).
    text = _responses_reply_text(response, output_limit_set=output_limit_set)
    tool_calls: list[dict[str, Any]] = []
    reasoning: list[dict[str, Any]] = []
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            kind = item.get("type")
            if kind == "function_call":
                # Positions in ``output`` count every item; appending renumbers the calls densely
                # (DEC-arch-10). The server's item ``id`` is not kept (DEC-004-6).
                tool_calls.append(
                    {
                        "id": item.get("call_id"),
                        "type": "function",
                        "function": {"name": item.get("name"), "arguments": item.get("arguments")},
                    }
                )
            elif kind == "reasoning":
                # Verbatim, including empty encrypted content from local servers (DEC-004-7).
                reasoning.append(item)
    # An ``incomplete`` reply only gets here when ``_responses_reply_text`` matched a cut at the app's
    # limit. DEC-004-16: its calls' arguments may be truncated, so the round is not run; the error
    # keeps the wording and reason of any other incomplete round.
    if tool_calls and response.get("status") == "incomplete":
        raise LLMError("OpenAI Responses reply is incomplete", description="max_output_tokens")
    return _ResponsesToolRound(text=text, tool_calls=tool_calls, reasoning=reasoning)


def _offered_tool_name(call: Any, offered: set[str]) -> str:
    """Return the tool name of one requested call after checking it was offered (DEC-002-3).

    This module's own copy of the chat-completions rule, reading the internal tool-call shape that
    ``_read_responses_tool_round`` produces (``{"function": {"name": ...}}``).

    Args:
        call: One entry of ``_ResponsesToolRound.tool_calls``.
        offered: Names advertised in ``tools_definitions`` (``_advertised_tool_names``).

    Returns:
        The requested name, guaranteed to be in ``offered``.

    Raises:
        LLMError: The name is missing or empty (also for a malformed call), or it is not advertised,
            even when ``tools`` has a callable for it; the message names the tool or says it was empty.
    """
    function = call.get("function") if isinstance(call, Mapping) else None
    name = function.get("name") if isinstance(function, Mapping) else None
    # A malformed call is reported as an empty name rather than a raw KeyError/TypeError.
    if not name:
        raise LLMError("The model requested a tool call with an empty tool name")
    if not isinstance(name, str) or name not in offered:
        raise LLMError(f"The model requested tool '{name}', which was not offered")
    return name


def _read_tool_arguments(name: str, raw: Any) -> dict[str, Any]:
    """Turn a requested call's arguments into the keyword mapping the tool is called with (DEC-002-4).

    This module's own copy of the chat-completions rule.

    Args:
        name: The (offered) tool name, used in the error message.
        raw: The call's ``arguments`` value as received.

    Returns:
        ``{}`` for missing (``None``) or empty (``""``) arguments; a mapping as given (copied); the
        decoded JSON object for text.

    Raises:
        LLMError: Text that is not JSON (chained from the decode error), JSON that is not an object, or
            a value that is neither text nor a mapping; the message names the tool.
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
    """Turn a tool's (already awaited) return value into the ``function_call_output`` text (DEC-002-5).

    This module's own copy of the chat-completions rule.

    Args:
        name: The tool name, used in the error message.
        result: The value the tool returned.

    Returns:
        JSON text with non-ASCII kept for a ``dict`` or ``list``; ``str(result)`` otherwise.

    Raises:
        LLMError: The ``dict`` or ``list`` cannot be serialised (chained from the error); the message
            names the tool.
    """
    # Only dict/list are JSON-encoded; other values (including non-dict mappings) keep their text
    # form so an object with a custom __str__ is still sent as that text.
    if not isinstance(result, (dict, list)):
        return str(result)
    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        # TypeError: a non-serialisable element; ValueError: a circular reference.
        raise LLMError(f"Result of tool '{name}' cannot be serialised to JSON", description=str(exc)) from exc


class OpenAIAPIClient(LLMClient):
    """Client for OpenAI and OpenAI-compatible servers speaking the Responses protocol.

    ``chat``, ``generate`` and ``chat_json`` send ``POST /responses`` (DEC-arch-1, DEC-003-11):
    the body carries ``model``, ``input`` (role/content items), ``stream``, ``store: false``,
    ``include: ["reasoning.encrypted_content"]`` and the app's settings translated to Responses
    fields (DEC-003-9, DEC-003-10). Settings without a Responses counterpart are not sent and are
    reported once per setting per client instance with ``LLMSettingIgnoredWarning`` (DEC-003-6).
    ``chat_with_tools`` runs the tool loop on ``/responses`` (DEC-004-10, DEC-004-11); the
    chat-completions protocol is available as ``OpenAIChatCompletionsClient``. ``list_models``,
    ``get_models`` and ``embed_batch``
    behave exactly as on the chat-completions client (US-057.AC-2).

    Invariants:
    - The credential comes only from ``bearer_token`` (required; never read from the environment,
      DEC-001-6) and never appears in error or trace text.
    - ``_warned_settings`` holds the no-counterpart settings this instance already warned about;
      it starts empty and only grows.
    """

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1",
                 timeout: float = 600.0, bearer_token: Optional[str] = None,
                 default_options: Optional[OpenAIOptions] = None) -> None:
        """Build a client that authenticates with the app's credential only (DEC-001-6).

        Args:
            model: Model name sent with every request.
            base_url: API root; a trailing slash is stripped by the base class.
            timeout: Total per-request timeout in seconds.
            bearer_token: The app's OpenAI credential, sent as ``Authorization: Bearer <token>``.
                The environment is never consulted.
            default_options: Client-level settings, merged under per-call settings.

        Raises:
            ValueError: ``bearer_token`` is missing or empty. The message tells the developer to
                provide a credential, names no environment variable and contains no credential value.
        """
        # DEC-001-6: refuse before any session exists; the message names the argument, never an
        # environment variable, and never echoes a credential value.
        if not bearer_token:
            raise ValueError("OpenAI credential not provided. Pass the API key as the bearer_token argument.")
        super().__init__(model, base_url, timeout, bearer_token, default_options)
        # Per-instance record of no-counterpart settings already warned about (DEC-003-6).
        self._warned_settings: set[str] = set()

    def _merge_options(
        self,
        *,
        options: Optional[OpenAIOptions],
    ) -> Dict[str, Any]:
        """Merge settings and turn them into Responses wire fields (DEC-003-9, DEC-003-10, DEC-003-1).

        The merge is ``_merge_options_base`` unchanged (table -> ``default_options`` -> ``options``;
        aliases mapped, unknown keys dropped silently without a warning, ``None`` values stripped).
        The merged settings then become wire fields:

        - ``temperature``, ``top_p``, ``user``: same name.
        - ``max_tokens`` (also set as ``context_length``): ``max_output_tokens``.
        - ``response_format``: ``text: {"format": _responses_text_format(value)}``.
        - ``presence_penalty``, ``frequency_penalty``, ``seed``, ``stop``, ``logit_bias``: not sent.
          For each one present that is not yet in ``self._warned_settings``, ``warnings.warn`` is
          called with ``LLMSettingIgnoredWarning`` and a message naming the setting and saying the
          Responses protocol has no counterpart so it was not sent; the key is then recorded.
          A client default and a per-call value are treated alike.
        - ``enable_thinking`` / ``reasoning_effort``: never top-level; a ``reasoning`` object.
          Thinking on (truthy): ``{"effort": <explicit effort or "medium">, "summary": "auto"}``.
          Thinking off (set, falsy): ``{"effort": <explicit effort or "none">}``.
          Thinking unset: ``{"effort": <explicit>}`` when an effort is set, else no ``reasoning``.

        Values are passed through unvalidated.

        Args:
            options: Per-call settings, or ``None``.

        Returns:
            The wire fields to place in the request body beside the fixed keys and ``input``.

        Side effects:
            May emit ``LLMSettingIgnoredWarning`` and add keys to ``self._warned_settings``.
        """
        opts = self._merge_options_base(
            options=options,
            supported_params=_OPENAI_SUPPORTED_PARAMS,
            alias_map=_OPENAI_ALIAS_MAP,
            default_opts=self.default_options,
        )
        # DEC-003-6: warn at call time, after the merge, so client defaults and per-call values are
        # treated alike; the per-instance record gives "once per setting per client" whatever the
        # app's warning filters do (Python's own filters still apply on top).
        for key in opts:
            if key in _OPENAI_NO_COUNTERPART_PARAMS and key not in self._warned_settings:
                warnings.warn(
                    f"Setting '{key}' has no counterpart in the OpenAI Responses protocol and was not sent",
                    LLMSettingIgnoredWarning,
                    # Points at the caller of chat / chat_json (one level deeper for generate).
                    stacklevel=3,
                )
                self._warned_settings.add(key)

        wire: dict[str, Any] = {}
        for key in ("temperature", "top_p", "user"):
            if key in opts:
                wire[key] = opts[key]
        if "max_tokens" in opts:
            wire["max_output_tokens"] = opts["max_tokens"]
        if "response_format" in opts:
            wire["text"] = {"format": _responses_text_format(opts["response_format"])}

        # Thinking set means present after the merge (None was stripped), so False / 0 count as off.
        effort = opts.get("reasoning_effort")
        if "enable_thinking" in opts:
            if opts["enable_thinking"]:
                # DEC-arch-7: only the thinking switch asks for a summary, never an effort alone.
                wire["reasoning"] = {"effort": effort if effort is not None else "medium", "summary": "auto"}
            else:
                # DEC-003-1: "none" is the off value; an explicit effort set by the app wins.
                wire["reasoning"] = {"effort": effort if effort is not None else "none"}
        elif effort is not None:
            wire["reasoning"] = {"effort": effort}
        return wire

    def _build_responses_payload(
        self,
        messages: list[LLMMessage],
        *,
        system: str | None,
        settings: dict[str, Any],
        stream: bool,
    ) -> dict[str, Any]:
        """Build a Responses request body (DEC-003-11); used by the non-streamed and streamed paths.

        The body holds exactly: ``model`` (``self.model``); ``input``; ``stream``; ``store: False``;
        ``include: ["reasoning.encrypted_content"]`` (DEC-arch-7); plus every key of ``settings``.
        No ``instructions`` field. ``input`` is a new list: a truthy ``system`` becomes
        ``{"role": "system", "content": system}`` at its head (an empty one adds nothing), then each
        caller message becomes one item, values as given, in order: an ``assistant`` message becomes
        ``{"type": "message", "role": "assistant", "content": ...}`` -- exactly those three keys; any
        other message (system, user) becomes ``{"role": ..., "content": ...}`` -- only those two keys.

        Args:
            messages: The caller's messages; neither the list nor its items are modified.
            system: Optional system instruction.
            settings: Wire fields from ``_merge_options``.
            stream: The value sent as ``stream``.

        Returns:
            The request body.
        """
        items: list[dict[str, Any]] = []
        if system:
            items.append({"role": "system", "content": system})
        # Only role and content are copied: Responses rejects unknown item fields (DEC-003-11).
        # Assistant turns carry "type": "message" because llama.cpp's Responses route refuses an
        # untyped assistant item (400 "Cannot determine type of 'item'"); system and user items stay
        # bare (DEC-003-11 as amended 2026-11-08).
        for m in messages:
            if m["role"] == "assistant":
                items.append({"type": "message", "role": "assistant", "content": m["content"]})
            else:
                items.append({"role": m["role"], "content": m["content"]})
        return {
            "model": self.model,
            "input": items,
            "stream": stream,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            **settings,
        }

    def _responses_round_input(
        self,
        turns: list[dict[str, Any]],
        reasoning: dict[int, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Convert the loop's internal copy into one round's ``input`` item list (DEC-004-6, DEC-004-7).

        Each turn, by shape, in order:

        - assistant turn with ``tool_calls`` at position ``i`` -> the carry items ``reasoning[i]``
          (verbatim, in order; none when absent), then ``{"type": "message", "role": "assistant",
          "content": <content>}`` only when its content is non-empty text, then one
          ``{"type": "function_call", "call_id": <id>, "name": <name>, "arguments": <text>}`` per call
          in order (mapping arguments as JSON text; missing as ``""``);
        - ``role: tool`` turn -> ``{"type": "function_call_output", "call_id": <tool_call_id>,
          "output": <content as text>}``;
        - every other turn -> exactly the item ``_build_responses_payload`` makes for it (DEC-003-11 as
          corrected, DEC-004-4); the system instruction is already a turn at the head of ``turns``.

        Neither ``turns`` nor ``reasoning`` (nor any of their items) is modified.

        Args:
            turns: The internal copy of the conversation, in chat-completions turn shapes.
            reasoning: The per-call reasoning record: position of an assistant tool-call turn in
                ``turns`` -> that round's reasoning items.

        Returns:
            A new list of input items for the round's request body.
        """
        items: list[dict[str, Any]] = []
        for index, turn in enumerate(turns):
            role = turn.get("role")
            tool_calls = turn.get("tool_calls")
            if role == "assistant" and tool_calls:
                # DEC-arch-9 / DEC-004-7: the round's reasoning goes ahead of its function calls
                # because ``store`` is false and OpenAI needs it re-sent with its encrypted content.
                items.extend(reasoning.get(index, []))
                content = turn.get("content")
                # DEC-004-6: an empty or null round text sends no assistant item at all.
                if isinstance(content, str) and content:
                    items.append({"type": "message", "role": "assistant", "content": content})
                for call in tool_calls:
                    function = call.get("function") if isinstance(call, Mapping) else None
                    if not isinstance(function, Mapping):
                        function = {}
                    arguments = function.get("arguments")
                    if arguments is None:
                        arguments = ""
                    elif not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False, default=str)
                    # Reduced form without the server's item id (DEC-004-6; accepted by the probe).
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call.get("id") if isinstance(call, Mapping) else None,
                            "name": function.get("name"),
                            "arguments": arguments,
                        }
                    )
            elif role == "tool":
                content = turn.get("content")
                # The output must be text: a raw JSON object is refused with 400 (probe).
                output = content if isinstance(content, str) else ("" if content is None else str(content))
                items.append({"type": "function_call_output", "call_id": turn.get("tool_call_id"), "output": output})
            else:
                # Delegated to the builder itself so caller-style turns get exactly 003's item
                # (DEC-003-11 as corrected, DEC-004-4); the system turn is already in ``turns``.
                items.extend(self._build_responses_payload([turn], system=None, settings={}, stream=False)["input"])
        return items

    async def _read_responses_stream(
        self,
        response: aiohttp.ClientResponse,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        *,
        output_limit_set: bool = False,
    ) -> str:
        """Parse an open Responses event stream into the full delivered text (DEC-003-13, DEC-arch-11).

        This class's own parser (DEC-arch-2): nothing is shared with the llama-swap class or the base;
        the chat-completions parsers are not used. Payload strings are read only through
        ``self._iter_sse_data(response)`` and each is decoded as JSON. Non-JSON payloads (including a
        ``[DONE]`` sentinel), non-object values and objects without a string ``type`` are skipped
        without error and without a receiver call. Dispatch on ``type``:

        - ``response.output_text.delta`` -> text piece (the event's ``delta``).
        - ``response.reasoning_text.delta`` and ``response.reasoning_summary_text.delta`` -> reasoning
          piece (``delta``); no separator is inserted between reasoning or summary parts.
        - ``response.completed`` -> the stream is complete; returns at once and reads nothing further.
        - ``response.failed`` -> ``LLMError``, description ``response.error.message`` when a string,
          else ``response.error`` as JSON text.
        - ``response.incomplete`` -> ``LLMError``, description ``response.incomplete_details.reason``
          when it is a non-empty string, else that value as JSON text (DEC-003-21) -- except when
          ``output_limit_set`` is true and that ``reason`` is exactly ``"max_output_tokens"``
          (DEC-003-18): the stream was cut at the
          app's own output limit and ends exactly as at ``response.completed`` (an open reasoning
          block is closed by its own ``</think>`` piece, nothing further is read or delivered, the
          return equals what the receiver got; DEC-arch-13, DEC-003-19).
        - ``error`` -> ``LLMError``, description the event's ``message`` when a string, else the
          event as JSON text.
        - Every other type (created, in_progress, output_item.*, content_part.*, ``*.done``, tool
          events, ...) is ignored.

        Empty or missing deltas deliver nothing. Reasoning markers follow DEC-000-20 per run of
        reasoning: the first reasoning piece at the start or after text is prefixed with
        ``<think>``; the first text piece after reasoning is prefixed with ``</think>``; a block still
        open when ``response.completed`` (or a cut at the app's limit, see above) arrives is closed by
        delivering ``</think>`` as its own piece.
        A stream without reasoning events carries no markers. Each piece is delivered to
        ``on_delta`` as soon as it is parsed (its result awaited when awaitable, DEC-000-23) and
        appended to the returned text, so the return equals the concatenation of what the receiver
        got. No piece is delivered after a failure event.

        Args:
            response: The open streamed response yielded by ``self._stream_post``; only passed to
                ``self._iter_sse_data`` (tests pass ``None`` with that reader replaced, DEC-003-16).
            on_delta: Optional receiver for each piece; when ``None`` pieces are only accumulated.
            output_limit_set: Keyword-only; true when the calling operation's wire settings hold
                ``max_output_tokens``, i.e. the app set the output limit (DEC-003-17, DEC-003-20).
                Default false keeps every ``response.incomplete`` event an error.

        Returns:
            The full delivered text, reasoning included with its markers.

        Raises:
            LLMError: On a ``response.failed``, ``response.incomplete`` (other than a cut at the app's
                limit with ``output_limit_set`` true, DEC-003-18) or ``error`` event (carrying the
                server's reason); when the stream ends without ``response.completed`` or such a cut
                (also after a ``[DONE]`` sentinel; the Responses form of DEC-001-1); on reader
                failures (DEC-001-7, raised by ``_iter_sse_data``).
            Exception: Whatever ``on_delta`` raises propagates unchanged and ends the parse; no later
                piece is delivered (DEC-000-23).
        """
        delivered: list[str] = []
        in_reasoning = False

        async def _deliver(piece: str) -> None:
            # Append before calling the receiver so the return always equals what was handed out;
            # a receiver failure propagates unchanged and ends the parse (DEC-000-23).
            delivered.append(piece)
            if on_delta is not None:
                result = on_delta(piece)
                if inspect.isawaitable(result):
                    await result

        async for data in self._iter_sse_data(response):
            try:
                event = json.loads(data)
            except (ValueError, TypeError):
                # Non-JSON payloads, including a ``[DONE]`` sentinel, are skipped (DEC-003-13).
                continue
            if not isinstance(event, Mapping):
                continue
            event_type = event.get("type")
            if not isinstance(event_type, str):
                continue

            if event_type == "response.output_text.delta":
                delta = event.get("delta")
                if not isinstance(delta, str) or not delta:
                    continue
                if in_reasoning:
                    # First text after a reasoning run closes the block on the same piece (DEC-000-20).
                    in_reasoning = False
                    delta = "</think>" + delta
                await _deliver(delta)
            elif event_type in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
                delta = event.get("delta")
                if not isinstance(delta, str) or not delta:
                    continue
                if not in_reasoning:
                    # Each reasoning run opens its own block, so reasoning/text/reasoning gives two
                    # blocks (DEC-003-13); no separator between summary parts.
                    in_reasoning = True
                    delta = "<think>" + delta
                await _deliver(delta)
            elif event_type == "response.completed":
                if in_reasoning:
                    await _deliver("</think>")
                # Completion is final: anything the server sends after it is never read.
                return "".join(delivered)
            elif event_type == "response.failed":
                inner = event.get("response")
                error = inner.get("error") if isinstance(inner, Mapping) else None
                message = error.get("message") if isinstance(error, Mapping) else None
                reason = message if isinstance(message, str) else json.dumps(error, default=str)
                raise LLMError("OpenAI Responses request failed", description=reason)
            elif event_type == "response.incomplete":
                inner = event.get("response")
                details = inner.get("incomplete_details") if isinstance(inner, Mapping) else None
                detail_reason = details.get("reason") if isinstance(details, Mapping) else None
                if output_limit_set and isinstance(detail_reason, str) and detail_reason == "max_output_tokens":
                    # DEC-arch-13 / DEC-003-19: a cut at the app's own limit ends the stream exactly as
                    # ``response.completed`` does, so markers, receiver and return follow one rule.
                    if in_reasoning:
                        await _deliver("</think>")
                    return "".join(delivered)
                # DEC-003-21: the reason is the description only as a non-empty string; otherwise the
                # details go as JSON text, so an empty or non-string reason never hides what the server sent.
                reason = (
                    detail_reason
                    if isinstance(detail_reason, str) and detail_reason
                    else json.dumps(details, default=str)
                )
                # Same wording as the non-streamed check (DEC-003-12): any other cut-off answer is an error.
                raise LLMError("OpenAI Responses reply is incomplete", description=reason)
            elif event_type == "error":
                message = event.get("message")
                reason = message if isinstance(message, str) else json.dumps(event, default=str)
                raise LLMError("The server reported an error in the stream", description=reason)
            # Every other event type (created, in_progress, output_item.*, content_part.*, *.done,
            # tool events) carries nothing to deliver and is ignored (DEC-003-13).

        # ``response.completed`` is the only completion signal; ``[DONE]`` does not count (DEC-003-13,
        # the Responses form of DEC-001-1).
        raise LLMError("The stream ended before the reply was complete")

    async def _read_responses_tool_stream(
        self,
        response: aiohttp.ClientResponse,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        *,
        output_limit_set: bool = False,
    ) -> _ResponsesToolRound:
        """Parse one streamed tool round's Responses event stream (DEC-004-9, DEC-arch-11).

        This class's own tool-round parser (DEC-arch-2), separate from ``_read_responses_stream``,
        which is not changed. Payload strings are read only through ``self._iter_sse_data(response)``
        and decoded as in DEC-003-13: non-JSON payloads (including a ``[DONE]`` sentinel), non-object
        values and objects without a string ``type`` are skipped. Dispatch on ``type``:

        - ``response.output_text.delta`` -> text piece; ``response.reasoning_text.delta`` and
          ``response.reasoning_summary_text.delta`` -> reasoning piece. Markers per DEC-003-13 with
          fresh marker state for this round; a block still open when ``response.completed`` arrives is
          closed by a ``</think>`` piece, so the round's reasoning reaches the receiver closed.
        - ``response.output_item.added`` with a ``function_call`` item -> starts a call keyed by the
          item's ``id`` (fallback ``output_index``), taking ``call_id``, ``name`` and any initial
          ``arguments``.
        - ``response.function_call_arguments.delta`` -> appends ``delta`` to the call keyed by
          ``item_id`` (fallback ``output_index``); a delta for an unknown key starts a call with no name.
        - ``response.function_call_arguments.done`` and ``response.output_item.done`` with a
          ``function_call`` item -> string ``arguments`` replace the assembled text; a non-empty
          ``call_id`` / ``name`` on the done item is set; a done item for an unknown key starts the call.
        - ``response.output_item.done`` with a ``reasoning`` item -> carry item (DEC-004-7), recorded
          with its ``output_index``.
        - ``response.completed`` -> the round is complete; returns at once and reads nothing further.
        - ``response.incomplete`` with ``output_limit_set`` on and reason exactly ``"max_output_tokens"``
          (a cut at the app's own limit, DEC-arch-13, DEC-004-17): when no tool call has started in the
          round (no added ``function_call`` item, arguments fragment or done ``function_call`` item),
          ends the round exactly as ``response.completed`` does (open reasoning closed by its own
          ``</think>`` piece, returns at once); when a call has started, ``LLMError`` with the server's
          reason and no further piece, not even a closing marker.
        - ``response.failed``, ``response.incomplete`` (no app limit, or any other reason / no reason /
          no details), ``error`` -> ``LLMError`` as in DEC-003-13.
        - Every other type is ignored.

        Pieces are delivered to ``on_delta`` as parsed (result awaited when awaitable, DEC-000-23).
        Tool-call names and argument fragments are never delivered to the receiver. No piece is
        delivered after a failure event.

        Args:
            response: The open streamed response yielded by ``self._stream_post``; only passed to
                ``self._iter_sse_data`` (tests pass ``None`` with that reader replaced, DEC-004-14).
            on_delta: Optional receiver for each text / reasoning piece; when ``None`` nothing is
                delivered.
            output_limit_set: Keyword-only; ``True`` when the app set the output limit for this call
                (DEC-004-15). With it on, a ``response.incomplete`` event whose reason matches
                DEC-003-18 ends the round as ``response.completed`` does when no tool call has started
                in the round; when one has started it raises ``LLMError`` with the server's reason and
                delivers no further piece, not even a closing marker (DEC-arch-13, DEC-004-17). Default
                off: the event raises as before.

        Returns:
            A ``_ResponsesToolRound``: ``text`` is the text deltas only (no reasoning, no markers);
            ``tool_calls`` are dense, in order of first appearance, each in the internal shape
            ``{"id": <call_id>, "type": "function", "function": {"name": <name>, "arguments": <arguments>}}``
            (missing values ``None``); ``reasoning`` holds the reasoning carry items verbatim, ordered by
            ``output_index`` regardless of the order in which they finished.

        Raises:
            LLMError: On a ``response.failed``, ``response.incomplete`` or ``error`` event (carrying the
                server's reason) -- except a cut at the app's limit with no started call, which returns
                (DEC-004-17); when the stream ends without ``response.completed`` (also after a
                ``[DONE]`` sentinel); on reader failures raised by ``_iter_sse_data``.
            Exception: Whatever ``on_delta`` raises propagates unchanged and ends the parse (DEC-000-23).
        """
        text_parts: list[str] = []
        in_reasoning = False
        # Calls in order of first appearance; each record holds ``call_id``, ``name`` and the assembled
        # ``arguments`` (``None`` until any argument text is seen). Two lookups point into it because
        # events name their item by ``id`` / ``item_id`` with ``output_index`` as the fallback
        # (DEC-004-9); either key reaches the same record once both are known.
        calls: list[dict[str, Any]] = []
        by_item_id: dict[str, dict[str, Any]] = {}
        by_output_index: dict[Any, dict[str, Any]] = {}
        # (output_index, arrival order, item): sorted at completion because reasoning items can finish
        # after the call's deltas (probe, DEC-004-9); arrival order breaks ties stably.
        carried: list[tuple[Any, int, dict[str, Any]]] = []

        async def _deliver(piece: str) -> None:
            # Inline, awaited only when awaitable; a receiver failure propagates unchanged and ends the
            # round before anything later is read (DEC-000-23).
            if on_delta is not None:
                result = on_delta(piece)
                if inspect.isawaitable(result):
                    await result

        def _call_for(item_id: Any, output_index: Any) -> dict[str, Any]:
            # Find the call an event refers to, or start one with no name (DEC-004-9: an unknown key
            # starts a call; the offered check later ends the loop with LLMError, DEC-002-3).
            record = None
            if isinstance(item_id, str) and item_id:
                record = by_item_id.get(item_id)
            if record is None and output_index is not None:
                record = by_output_index.get(output_index)
            if record is None:
                record = {"call_id": None, "name": None, "arguments": None}
                calls.append(record)
            if isinstance(item_id, str) and item_id:
                by_item_id.setdefault(item_id, record)
            if output_index is not None:
                by_output_index.setdefault(output_index, record)
            return record

        def _finish_call(record: dict[str, Any], source: Mapping[str, Any]) -> None:
            # A done event's full ``arguments`` replace the assembled text (OpenAI sends both the
            # arguments-done and the item-done; llama.cpp only the item-done); non-empty identity
            # fields fill in what the added event lacked (DEC-004-9).
            arguments = source.get("arguments")
            if isinstance(arguments, str):
                record["arguments"] = arguments
            for field in ("call_id", "name"):
                value = source.get(field)
                if isinstance(value, str) and value:
                    record[field] = value

        async def _end_round() -> _ResponsesToolRound:
            """Close the round as ``response.completed`` does and build its result (DEC-004-9).

            Shared by completion and by a cut at the app's limit without a started call (DEC-004-17),
            so markers, the receiver and the returned round follow one rule.

            Returns:
                The round's text deltas, tool calls in order of first appearance and carry items
                ordered by ``output_index``.

            Raises:
                Exception: Whatever ``on_delta`` raises for the closing marker (DEC-000-23).
            """
            if in_reasoning:
                # Closes a block left open (e.g. reasoning followed only by function calls) so
                # the next round's pieces start outside it (DEC-004-9).
                await _deliver("</think>")
            carried.sort(key=lambda entry: (entry[0] if isinstance(entry[0], int) else float("inf"), entry[1]))
            tool_calls = [
                {
                    "id": record["call_id"],
                    "type": "function",
                    "function": {"name": record["name"], "arguments": record["arguments"]},
                }
                for record in calls
            ]
            return _ResponsesToolRound(
                text="".join(text_parts),
                tool_calls=tool_calls,
                reasoning=[entry[2] for entry in carried],
            )

        async for data in self._iter_sse_data(response):
            try:
                event = json.loads(data)
            except (ValueError, TypeError):
                # Non-JSON payloads, including a ``[DONE]`` sentinel, are skipped (DEC-003-13).
                continue
            if not isinstance(event, Mapping):
                continue
            event_type = event.get("type")
            if not isinstance(event_type, str):
                continue

            if event_type == "response.output_text.delta":
                delta = event.get("delta")
                if not isinstance(delta, str) or not delta:
                    continue
                # The round text keeps the text deltas only; the marker goes to the receiver alone.
                text_parts.append(delta)
                if in_reasoning:
                    in_reasoning = False
                    delta = "</think>" + delta
                await _deliver(delta)
            elif event_type in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
                delta = event.get("delta")
                if not isinstance(delta, str) or not delta:
                    continue
                if not in_reasoning:
                    in_reasoning = True
                    delta = "<think>" + delta
                await _deliver(delta)
            elif event_type == "response.output_item.added":
                item = event.get("item")
                if isinstance(item, Mapping) and item.get("type") == "function_call":
                    record = _call_for(item.get("id"), event.get("output_index"))
                    record["call_id"] = item.get("call_id")
                    record["name"] = item.get("name")
                    initial = item.get("arguments")
                    if isinstance(initial, str):
                        record["arguments"] = initial
            elif event_type == "response.function_call_arguments.delta":
                record = _call_for(event.get("item_id"), event.get("output_index"))
                delta = event.get("delta")
                if isinstance(delta, str):
                    record["arguments"] = (record["arguments"] or "") + delta
            elif event_type == "response.function_call_arguments.done":
                _finish_call(_call_for(event.get("item_id"), event.get("output_index")), event)
            elif event_type == "response.output_item.done":
                item = event.get("item")
                if not isinstance(item, Mapping):
                    continue
                kind = item.get("type")
                if kind == "function_call":
                    _finish_call(_call_for(item.get("id"), event.get("output_index")), item)
                elif kind == "reasoning":
                    # Verbatim from its done event, the one source of carry items (DEC-004-7, DEC-004-9).
                    carried.append((event.get("output_index"), len(carried), item))
            elif event_type == "response.completed":
                # Completion is final: anything the server sends after it is never read.
                return await _end_round()
            elif event_type == "response.failed":
                inner = event.get("response")
                error = inner.get("error") if isinstance(inner, Mapping) else None
                message = error.get("message") if isinstance(error, Mapping) else None
                reason = message if isinstance(message, str) else json.dumps(error, default=str)
                raise LLMError("OpenAI Responses request failed", description=reason)
            elif event_type == "response.incomplete":
                inner = event.get("response")
                details = inner.get("incomplete_details") if isinstance(inner, Mapping) else None
                detail_reason = details.get("reason") if isinstance(details, Mapping) else None
                # DEC-004-17: a cut at the app's own limit (exact reason, DEC-003-18) with no started
                # call is the round's answer and ends as completion does; a started call (any record
                # in ``calls``, even a nameless fragment) may be truncated, so it falls through to the
                # error below, which delivers nothing further.
                if (
                    output_limit_set
                    and not calls
                    and isinstance(detail_reason, str)
                    and detail_reason == "max_output_tokens"
                ):
                    return await _end_round()
                # DEC-003-21: the reason is the description only as a non-empty string; otherwise the
                # details go as JSON text, so an empty or non-string reason never hides what the server sent.
                reason = (
                    detail_reason
                    if isinstance(detail_reason, str) and detail_reason
                    else json.dumps(details, default=str)
                )
                raise LLMError("OpenAI Responses reply is incomplete", description=reason)
            elif event_type == "error":
                message = event.get("message")
                reason = message if isinstance(message, str) else json.dumps(event, default=str)
                raise LLMError("The server reported an error in the stream", description=reason)
            # Every other event type (created, in_progress, content_part.*, other *.done) carries
            # nothing for the round and is ignored (DEC-004-9).

        # ``response.completed`` is the only completion signal; ``[DONE]`` does not count (DEC-004-9).
        # Raised before any tool of the round runs, since the caller only runs tools after a return.
        raise LLMError("The stream ended before the reply was complete")

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
        """Send one Responses chat request and return the answer text.

        Settings are merged by ``_merge_options`` (may warn, DEC-003-6) and the body is built by
        ``_build_responses_payload`` with ``stream`` as given. Inside the ``"chat"`` trace session
        the system instruction, messages and ``("POST", "/responses", payload)`` are logged.
        Non-streamed: sends ``self._request("POST", "/responses", json=payload)`` (DEC-003-16),
        extracts the text with ``_responses_reply_text``, logs and returns it. Streamed
        (``stream=True``): the same body with ``stream`` true is opened once through
        ``self._stream_post("/responses", payload)`` (DEC-001-8, DEC-003-16) and parsed by
        ``_read_responses_stream`` inside the trace session (DEC-003-13); pieces go to ``on_delta`` in
        order, through a chunk-logging trace wrapper only when tracing is enabled (DEC-000-48); the
        returned text (equal to everything the receiver got) is logged and returned.

        Output limit (DEC-arch-13, DEC-003-17..19): when the merged wire settings hold
        ``max_output_tokens`` (the app set ``max_tokens`` / ``context_length`` as a call option or a
        client default, not cleared with ``None``), a reply or ``response.incomplete`` event cut for
        reason ``max_output_tokens`` is returned (streamed: delivered) and logged exactly as a
        completed reply. With no app limit, or any other incomplete reason, it still raises.

        Args:
            messages: Conversation messages; the caller's list is never modified.
            system: Optional system instruction, sent as a leading system item when truthy.
            options: Per-call settings.
            stream: Stream the answer.
            on_delta: Receiver for streamed pieces.

        Returns:
            The answer text (``""`` when the reply holds none).

        Raises:
            LLMError: Non-2xx reply (address and server reply; never the credential), transport
                failure, or a failed / incomplete reply other than a cut at the app's own limit
                (DEC-003-12, DEC-arch-13); streamed: a failure event (same exception for the cut) or
                a stream ending before ``response.completed`` (DEC-003-13). A receiver failure
                propagates unchanged (DEC-000-23).
        """
        settings = self._merge_options(options=options)
        # DEC-003-17: the wire field is the merge's verdict on "the app set the output limit"
        # (aliases, defaults and ``None`` clearing already applied).
        output_limit_set = "max_output_tokens" in settings
        payload = self._build_responses_payload(messages, system=system, settings=settings, stream=stream)

        with self._tracer.trace("chat", self.model) as t:
            t.log_system(system)
            # The input items are the copied messages with the system item first (operation shape).
            t.log_messages(payload["input"])
            t.log_request("POST", "/responses", payload)

            if not stream:
                reply = await self._request("POST", "/responses", json=payload)
                # DEC-003-12: a failed / incomplete reply raises here, inside the session, so it is recorded.
                result = _responses_reply_text(reply, output_limit_set=output_limit_set)
                t.log_response(result)
                return result

            _orig_delta = on_delta

            async def _tracing_delta(text: str) -> None:
                t.log_chunk(text)
                if _orig_delta:
                    piece_result = _orig_delta(text)
                    if inspect.isawaitable(piece_result):
                        await piece_result

            # Streamed path: DEC-001-8 maps non-2xx and transport failures to LLMError; the trace
            # wrapper is installed only when tracing is enabled (DEC-000-48).
            async with self._stream_post("/responses", payload) as response:
                result = await self._read_responses_stream(
                    response,
                    _tracing_delta if self._tracer.enabled else on_delta,
                    output_limit_set=output_limit_set,
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
        """Answer one prompt: a one-message chat through ``chat`` (DEC-000-8).

        Sends the system item (when truthy) and one ``{"role": "user", "content": prompt}`` item
        (DEC-003-11); every argument is forwarded to ``chat`` unchanged.

        A reply cut at the app's own output limit is returned as ``chat`` returns it; with no app
        limit, or any other early end, it raises as ``chat`` does (DEC-arch-13, DEC-003-19).

        Returns:
            The answer text, as ``chat`` returns it.

        Raises:
            LLMError: As ``chat``.
        """
        messages: list[LLMMessage] = [{"role": "user", "content": prompt}]
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
        Chat with structured JSON output over Responses (DEC-003-10).

        Sends one non-streamed ``POST /responses`` (``stream: False``) built by
        ``_build_responses_payload``, inside the ``"chat_json"`` trace session, through
        ``self._request("POST", "/responses", json=payload)`` (DEC-003-16). The text format is
        ``text: {"format": ...}`` holding the translated ``response_format`` setting when the app set
        one, otherwise ``{"type": "json_object"}`` (the Responses form of DEC-001-2). Returns the
        text from ``_responses_reply_text`` (``""`` when the reply holds none).
        When ``response_schema`` is given, the only check is that the reply text parses as JSON
        (DEC-002-7, DEC-000-37): any valid JSON value (object, list, string, number, boolean or
        null) is returned unchanged; the schema's fields are not checked. Without
        ``response_schema`` the text is returned without any check. The schema is not sent
        (US-040, out of scope).

        Output limit (DEC-arch-13, DEC-003-17..19): when the merged wire settings hold
        ``max_output_tokens``, a reply cut for reason ``max_output_tokens`` yields its text as a
        completed reply does; the JSON parse check above still applies to it. With no app limit, or
        any other incomplete reason, it raises.

        Raises:
            LLMError: Request failure; a failed / incomplete reply other than a cut at the app's own
                limit (DEC-003-12, DEC-arch-13); or
                ``response_schema`` given and the reply is not valid JSON ("LLM response was not
                valid JSON, despite requesting JSON mode.", parse error in the description).
        """
        settings = self._merge_options(options=options)
        # DEC-003-10 / DEC-001-2: the JSON-object format is the fallback only; an app-set
        # ``response_format`` setting has already become ``text`` in ``settings`` and is kept.
        if "text" not in settings:
            settings["text"] = {"format": {"type": "json_object"}}
        # DEC-003-17: computed from the wire settings, as in ``chat``.
        output_limit_set = "max_output_tokens" in settings
        payload = self._build_responses_payload(messages, system=system, settings=settings, stream=False)

        with self._tracer.trace("chat_json", self.model) as t:
            t.log_system(system)
            t.log_messages(payload["input"])
            t.log_request("POST", "/responses", payload)

            reply = await self._request("POST", "/responses", json=payload)
            response_content = _responses_reply_text(reply, output_limit_set=output_limit_set)
            t.log_response(response_content)

        # DEC-002-7: the only check is the JSON parse; any valid JSON value passes unchanged.
        # TypeError stays caught because a non-string content cannot be parsed.
        if response_schema:
            try:
                json.loads(response_content)
            except (json.JSONDecodeError, TypeError) as e:
                raise LLMError(
                    "LLM response was not valid JSON, despite requesting JSON mode.",
                    description=str(e)
                )

        return response_content

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
        """Run a tool conversation over ``POST /responses`` and return the final answer text.

        Order (DEC-004-11, 002's rules): every advertised tool (nested or flat definition) must have
        a callable, else ``ValueError`` before any request; settings are merged (the ignored-setting
        warning fires once per client, DEC-004-12); an internal copy of ``messages`` is made with a
        truthy ``system`` at its head. Each round (at most ``max_loops``) sends one ``POST /responses``
        through ``_request`` with the body keys of DEC-003-11 plus the flattened ``tools`` (DEC-004-5,
        DEC-004-10), ``stream`` false; ``response_format`` wins over the setting and is sent as
        ``text: {"format": ...}`` on every round. A round without function calls returns its text.
        Otherwise each call runs in order (offered check, argument reading, argument-name check, call
        and await, result as text), the assistant tool-call turn and tool-result turns are appended to
        the internal copy, and the round's reasoning items are carried to later rounds ahead of that
        turn's function calls (DEC-004-7). The caller's ``messages`` and ``tools_definitions`` are never
        modified; reasoning never reaches the answer or the caller's messages. Trace: one
        ``chat_with_tools`` session, each round's request logged on ``"/responses (loop <n>)"`` (n from
        1), each tool call and result, the final answer.

        With ``stream=True`` each round sends the same body with ``stream`` true, opened through
        ``_stream_post("/responses", ...)`` and read by ``_read_responses_tool_stream`` (DEC-004-9): the
        receiver gets every round's text pieces and that round's reasoning as its own closed
        ``<think>`` block, never tool-call text, and is called without the trace wrapper (DEC-004-10).
        The rest of the loop is the non-streamed one. The return is the final round's text only, with
        no reasoning and no markers.

        A round cut at the app's own output limit (DEC-arch-13): when the wire settings hold
        ``max_output_tokens`` (the app set ``max_tokens`` / ``context_length`` as a call option or a
        default and did not clear it with ``None``), computed once per call and passed to every round's
        reader / parser (DEC-004-15), a round cut for reason ``max_output_tokens`` is the answer when it
        holds no tool call -- non-streamed: no ``function_call`` item (DEC-004-16); streamed: no call
        started (DEC-004-17) -- and is returned and logged as a completed answer round, even on the last
        allowed round. A cut round with calls raises ``LLMError`` with the server's reason and runs none
        of them; tools of earlier rounds stay run (DEC-004-18). With no app limit, and for every other
        early end, a cut round still raises.

        Returns:
            The final round's text.

        Raises:
            ValueError: An advertised tool has no callable (before any request), or a call names an
                argument the callable lacks.
            LLMError: A call names a tool not advertised or an empty name, its arguments are not a
                JSON object, a tool result cannot be serialised, a round reply is failed / incomplete
                (except a cut at the app's limit without tool calls, DEC-004-16 / DEC-004-17), a
                streamed round reports a failure / error event or ends without completion, or a
                request fails (non-2xx, transport).
            Exception: Streamed, whatever ``on_delta`` raises propagates unchanged (DEC-000-23).
            RuntimeError: A tool raised (naming the tool), or the round limit was reached without a
                final answer (text contains ``max_loops`` and its value; raised after the trace).
        """
        # Pre-flight before the merge and before any request (DEC-004-11, DEC-004-5).
        advertised = _advertised_tool_names(tools_definitions)
        missing = advertised - tools.keys()
        if missing:
            raise ValueError(f"Missing callables for tools {sorted(missing)}")

        # Merged once per call, not per round, so the ignored-setting warning follows DEC-004-12.
        settings = self._merge_options(options=options)
        # DEC-001-15 / DEC-004-10: a truthy parameter replaces the setting's translated format on
        # every round; without it the setting's ``text`` (if any) stays.
        if response_format:
            settings["text"] = {"format": _responses_text_format(response_format)}
        flat_tools = _flatten_responses_tools(tools_definitions)
        # DEC-004-15: the settings are fixed for the call, so one verdict serves every round; it is
        # read after the merge so defaults, the alias and per-call ``None`` clearing all count.
        output_limit_set = "max_output_tokens" in settings

        msgs: list[dict[str, Any]] = list(messages)
        if system:
            msgs.insert(0, {"role": "system", "content": system})
        # Call-local carry record (DEC-004-7): position of an assistant tool-call turn in ``msgs`` ->
        # that round's reasoning items. Kept apart from the turns and dropped when the call ends.
        reasoning: dict[int, list[dict[str, Any]]] = {}

        with self._tracer.trace("chat_with_tools", self.model) as t:
            t.log_system(system)
            t.log_messages(msgs)

            for loop_idx in range(max_loops):
                # 003's builder supplies the fixed keys and settings (DEC-003-11); the round's own
                # input replaces its empty one and the tools are added (DEC-004-10).
                payload = self._build_responses_payload([], system=None, settings=settings, stream=stream)
                payload["input"] = self._responses_round_input(msgs, reasoning)
                payload["tools"] = flat_tools

                t.log_request("POST", f"/responses (loop {loop_idx + 1})", payload)

                if stream:
                    # Same body with ``stream`` true, one streamed request per round (DEC-004-10,
                    # DEC-003-16). The receiver is passed raw: the tool path applies no trace wrapper
                    # (DEC-004-10). Everything after reading the round is shared with the
                    # non-streamed path (DEC-004-6, DEC-004-7, DEC-004-11).
                    async with self._stream_post("/responses", payload) as response:
                        round_ = await self._read_responses_tool_stream(
                            response, on_delta, output_limit_set=output_limit_set
                        )
                else:
                    round_ = _read_responses_tool_round(
                        await self._request("POST", "/responses", json=payload), output_limit_set=output_limit_set
                    )

                if not round_.tool_calls:
                    # The final round's reasoning is never kept (DEC-004-7). A cut answer round lands
                    # here too, before the round-limit check (DEC-004-18).
                    t.log_response(round_.text)
                    return round_.text

                turn_index = len(msgs)
                msgs.append({"role": "assistant", "content": round_.text, "tool_calls": round_.tool_calls})
                if round_.reasoning:
                    reasoning[turn_index] = round_.reasoning

                for call in round_.tool_calls:
                    # 002's order: offered check, then argument reading, both before the tool can
                    # run; calls earlier in this reply have already run and keep their effects.
                    name = _offered_tool_name(call, advertised)
                    func = tools[name]
                    kwargs = _read_tool_arguments(name, call["function"].get("arguments"))
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
                    msgs.append({"role": "tool", "content": content, "tool_call_id": call.get("id")})

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
