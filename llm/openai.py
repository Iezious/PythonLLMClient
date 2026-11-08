"""OpenAI Responses client: chat, generate and JSON-only answers over ``POST /responses``.

Feature 003 step 002: settings translated to Responses fields and no-counterpart settings dropped
with ``LLMSettingIgnoredWarning`` (DEC-003-6, DEC-003-9, DEC-003-10); request body and non-streamed
reply handling per DEC-003-11 / DEC-003-12; tool chat refused (DEC-003-4). Step 003: streamed
``chat`` / ``generate`` parse the Responses event stream (DEC-003-13). Model listing and
embeddings are unchanged from the chat-completions client (US-057.AC-2), which now lives in
``llm/openai_chat_completions.py``.
"""

from collections.abc import Mapping
from typing import Any, Awaitable, Callable, Dict, List, Optional
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
    # a broad ValueError catch could swallow unrelated errors (003.context.md). ``_request`` is not edited.
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
        # Wording kept as before this step; older tests may match it (003.context.md § Message wording).
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
    # not a position (003.context.md § Positions). Partial positions are ambiguous, so server order.
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


def _responses_reply_text(response: Any) -> str:
    """Return the answer text of a non-streamed Responses reply (DEC-003-12).

    Visits every item of type ``message`` in the reply's ``output`` list in order and concatenates,
    with no separator, the ``text`` of each content part of type ``output_text``. Reasoning items and
    other part types (e.g. refusal) are ignored. A missing or non-list ``output``, no message item, or
    a null/missing ``text`` gives ``""``.

    Args:
        response: The decoded reply body (expected to be a mapping).

    Returns:
        The concatenated output text; ``""`` when there is none.

    Raises:
        LLMError: ``status`` is ``"failed"`` -- description is ``error.message`` when ``error`` is a
            mapping with a string ``message``, else the ``error`` value as JSON text; or ``status`` is
            ``"incomplete"`` -- description is ``incomplete_details.reason`` when present, else the
            ``incomplete_details`` value as JSON text. Any other or missing status extracts the text.
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
        reason = str(detail_reason) if detail_reason is not None else json.dumps(details, default=str)
        # DEC-003-12: a cut-off answer is never returned as if complete.
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


class OpenAIAPIClient(LLMClient):
    """Client for OpenAI and OpenAI-compatible servers speaking the Responses protocol.

    ``chat``, ``generate`` and ``chat_json`` send ``POST /responses`` (DEC-arch-1, DEC-003-11):
    the body carries ``model``, ``input`` (role/content items), ``stream``, ``store: false``,
    ``include: ["reasoning.encrypted_content"]`` and the app's settings translated to Responses
    fields (DEC-003-9, DEC-003-10). Settings without a Responses counterpart are not sent and are
    reported once per setting per client instance with ``LLMSettingIgnoredWarning`` (DEC-003-6).
    Tool conversations are refused until feature 004 (DEC-003-4); the chat-completions protocol is
    available as ``OpenAIChatCompletionsClient``. ``list_models``, ``get_models`` and ``embed_batch``
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

    async def _read_responses_stream(
        self,
        response: aiohttp.ClientResponse,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
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
          when present, else that value as JSON text.
        - ``error`` -> ``LLMError``, description the event's ``message`` when a string, else the
          event as JSON text.
        - Every other type (created, in_progress, output_item.*, content_part.*, ``*.done``, tool
          events, ...) is ignored.

        Empty or missing deltas deliver nothing. Reasoning markers follow DEC-000-20 per run of
        reasoning: the first reasoning piece at the start or after text is prefixed with
        ``<think>``; the first text piece after reasoning is prefixed with ``</think>``; a block still
        open when ``response.completed`` arrives is closed by delivering ``</think>`` as its own piece.
        A stream without reasoning events carries no markers. Each piece is delivered to
        ``on_delta`` as soon as it is parsed (its result awaited when awaitable, DEC-000-23) and
        appended to the returned text, so the return equals the concatenation of what the receiver
        got. No piece is delivered after a failure event.

        Args:
            response: The open streamed response yielded by ``self._stream_post``; only passed to
                ``self._iter_sse_data`` (tests pass ``None`` with that reader replaced, DEC-003-16).
            on_delta: Optional receiver for each piece; when ``None`` pieces are only accumulated.

        Returns:
            The full delivered text, reasoning included with its markers.

        Raises:
            LLMError: On a ``response.failed``, ``response.incomplete`` or ``error`` event (carrying
                the server's reason); when the stream ends without ``response.completed`` (also after
                a ``[DONE]`` sentinel; the Responses form of DEC-001-1); on reader failures
                (DEC-001-7, raised by ``_iter_sse_data``).
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
                reason = str(detail_reason) if detail_reason is not None else json.dumps(details, default=str)
                # Same wording as the non-streamed check (DEC-003-12): a cut-off answer is an error.
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
                failure, or a failed / incomplete reply (DEC-003-12); streamed: a failure event or a
                stream ending before ``response.completed`` (DEC-003-13). A receiver failure
                propagates unchanged (DEC-000-23).
        """
        settings = self._merge_options(options=options)
        payload = self._build_responses_payload(messages, system=system, settings=settings, stream=stream)

        with self._tracer.trace("chat", self.model) as t:
            t.log_system(system)
            # The input items are the copied messages with the system item first (operation shape).
            t.log_messages(payload["input"])
            t.log_request("POST", "/responses", payload)

            if not stream:
                reply = await self._request("POST", "/responses", json=payload)
                # DEC-003-12: a failed / incomplete reply raises here, inside the session, so it is recorded.
                result = _responses_reply_text(reply)
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
        """Answer one prompt: a one-message chat through ``chat`` (DEC-000-8).

        Sends the system item (when truthy) and one ``{"role": "user", "content": prompt}`` item
        (DEC-003-11); every argument is forwarded to ``chat`` unchanged.

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

        Raises:
            LLMError: Request failure; a failed / incomplete reply (DEC-003-12); or
                ``response_schema`` given and the reply is not valid JSON ("LLM response was not
                valid JSON, despite requesting JSON mode.", parse error in the description).
        """
        settings = self._merge_options(options=options)
        # DEC-003-10 / DEC-001-2: the JSON-object format is the fallback only; an app-set
        # ``response_format`` setting has already become ``text`` in ``settings`` and is kept.
        if "text" not in settings:
            settings["text"] = {"format": {"type": "json_object"}}
        payload = self._build_responses_payload(messages, system=system, settings=settings, stream=False)

        with self._tracer.trace("chat_json", self.model) as t:
            t.log_system(system)
            t.log_messages(payload["input"])
            t.log_request("POST", "/responses", payload)

            reply = await self._request("POST", "/responses", json=payload)
            response_content = _responses_reply_text(reply)
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
        """Refuse tool conversations: not available on Responses until feature 004 (DEC-003-4).

        Raises ``NotImplementedError`` before settings are merged or any request is sent (so no
        warning is emitted), with a message saying tool conversations on Responses are not available
        yet and naming ``OpenAIChatCompletionsClient`` as the interim choice. Never falls back to
        ``/chat/completions``. Signature unchanged.

        Raises:
            NotImplementedError: Always.
        """
        # DEC-003-4: refuse first, before _merge_options (no warning) and before any request; never
        # fall back to /chat/completions (US-056.AC-4).
        raise NotImplementedError(
            "Tool conversations are not available on the OpenAI Responses protocol yet; "
            "use OpenAIChatCompletionsClient for chat_with_tools in the meantime."
        )

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
