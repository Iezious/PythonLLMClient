"""llama-swap Chat Completions client (``LlamaSwapChatCompletionsClient``): optional credential, shared helpers.

A renamed copy of the pre-Responses ``llm/llama_swap.py`` (feature 003 step 001, DEC-003-7): same behaviour,
tables and helpers; it changes only for chat-completions fixes, never for Responses work (DEC-arch-2).
History carried over from the source module:

Fixes from feature 001: streamed requests via ``_stream_post``, null reply content as empty text,
and only app-set settings sent (including stop sequences, token bias, end-user id, response format).
Feature 002 step 001: the tool loop maps an unoffered tool, unreadable arguments and an
unserialisable result to ``LLMError`` and names the round limit in its error (DEC-002-2..5).
Feature 002 step 003: embeddings and model-list replies are checked, embeddings ordered by
position, and unreadable bodies mapped to ``LLMError`` (DEC-002-8..10).
"""

from collections.abc import Mapping
from typing import Any, Awaitable, Callable, Dict, List, Optional
import json
import inspect

import aiohttp

from llm.llm_client import LLMClient
from llm.exceptions import LLMError
from llm.message import LLMMessage
from llm.options import OpenAIOptions

# Settings the llama-swap client accepts. Every value is None: the table contributes no
# built-in value, so only keys the app set reach the server (DEC-001-13). `stop`,
# `logit_bias`, `user` and `response_format` are offered settings (DEC-001-14).
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
    """Turn a requested call's arguments into the keyword mapping the tool is called with.

    Args:
        name: The (offered) tool name, used in the error message.
        raw: The call's ``arguments`` value as received.

    Returns:
        ``{}`` for missing or empty arguments; a mapping as given; the decoded JSON object for text.

    Raises:
        LLMError: Text that is not JSON (chained from the decode error), JSON that is not an object,
            or a value that is neither text nor a mapping; the description carries the reason
            (DEC-002-4).
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
    """Turn a tool's return value into the tool message content.

    Args:
        name: The tool name, used in the error message.
        result: The value the tool returned (already awaited).

    Returns:
        JSON text (non-ASCII kept unescaped) for a ``dict`` or ``list``; ``str(result)`` otherwise.

    Raises:
        LLMError: The ``dict`` or ``list`` cannot be serialised (chained from the error, DEC-002-5).
    """
    # Only dict/list are JSON-encoded, as before; other values (including non-dict mappings) keep
    # their text form so an object with a custom __str__ is still sent as that text.
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


class LlamaSwapChatCompletionsClient(LLMClient):
    """OpenAI-compatible client for llama-swap (no auth by default), the chat-completions protocol class.

    Built by ``get_llm_client(..., protocol="chat_completions")`` or directly; independent of the
    Responses class ``LlamaSwapAPIClient`` (neither subclasses the other, DEC-arch-2).

    This client is designed for the llama-swap proxy.
    It supports:
    - OpenAI-compatible endpoints (/v1/chat/completions, /v1/models)
    - No default authentication (bearer token optional)
    - llama-swap specific configuration defaults
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8080/v1",
        timeout: float = 600.0,
        bearer_token: Optional[str] = None,
        default_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model, base_url, timeout, bearer_token, default_options)

    def _merge_options(
        self,
        *,
        options: Optional[OpenAIOptions],
    ) -> Dict[str, Any]:
        """Merge table, client-default and per-call settings into the request settings.

        Uses the base three-layer merge (DEC-000-5) with the llama-swap tables: aliases applied,
        unknown keys dropped, a per-call value wins over a client default, and a higher-layer
        ``None`` removes the key. ``reasoning_effort`` is always dropped; ``enable_thinking`` is
        sent as-is, and only when set (DEC-000-19).

        Args:
            options: Per-call settings, or ``None``.

        Returns:
            Only the settings the app set, keyed by their server names.
        """
        opts = self._merge_options_base(
            options=options,
            supported_params=_LLAMA_SWAP_SUPPORTED_PARAMS,
            alias_map=_LLAMA_SWAP_ALIAS_MAP,
            default_opts=self.default_options,
        )
        # llama-swap uses enable_thinking, not reasoning_effort — drop it.
        opts.pop("reasoning_effort", None)
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
        """Send one chat-completions request and return the reply text.

        The merged settings are spread top-level into the request body. Non-streamed: returns
        the reply's content, with a null content returned (and traced) as ``""`` (DEC-001-12).
        Streamed: opens the request through the base ``_stream_post("/chat/completions", payload)``
        helper (DEC-001-8), passes each text piece to ``on_delta`` in order, and returns their
        concatenation; when tracing is enabled the receiver is wrapped so every piece is traced
        first (DEC-000-48).

        Args:
            messages: Conversation turns.
            system: Optional system prompt, inserted as the first turn.
            options: Per-call settings.
            stream: Stream the reply.
            on_delta: Receiver of streamed text pieces; its failure propagates unchanged.

        Returns:
            The reply text (never ``None``).

        Raises:
            LLMError: Non-2xx reply (address, status, server reply; never the credential) or an
                unreachable server, on either branch.
        """
        opt = self._merge_options(options=options)

        msgs = list(messages)
        if system:
            msgs.insert(0, {"role": "system", "content": system})

        payload = {
            "model": self.model,
            "messages": msgs,
            "stream": stream,
            **opt,
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
        """Request a JSON answer and return its text.

        Sends the merged ``response_format`` setting when the app set one, otherwise
        ``{"type": "json_object"}`` (DEC-001-2). A null reply content is returned as ``""``
        (DEC-001-12). When ``response_schema`` is given the only check is that the text parses as
        JSON (DEC-002-7): any valid JSON value (object, list, string, number, boolean or null) is
        returned unchanged; the schema's fields are not checked and the schema is not sent.

        Args:
            messages: Conversation turns.
            response_schema: When truthy, enables the JSON parse check.
            system: Optional system prompt, inserted as the first turn.
            options: Per-call settings.

        Returns:
            The reply text (never ``None``).

        Raises:
            LLMError: Request failure, or the reply is not valid JSON while ``response_schema``
                is given.
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

        if response_schema:
            try:
                json.loads(response_content)
            except (json.JSONDecodeError, TypeError) as exc:
                raise LLMError(
                    "LLM response was not valid JSON, despite requesting JSON mode.",
                    description=str(exc),
                )

        return response_content

    async def _tools_stream_request(
        self,
        payload: Dict[str, Any],
        on_delta: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> tuple[str, list[dict]]:
        """Run one streamed tool round (test seam; inputs and outputs unchanged).

        Opens the request through the base ``_stream_post("/chat/completions", payload)`` helper
        (DEC-001-8) and parses the stream with the tools parser.

        Args:
            payload: The full request body of this round.
            on_delta: Receiver of streamed text pieces; its failure propagates unchanged.

        Returns:
            ``(content, tool_calls)`` of the assistant turn.

        Raises:
            LLMError: Non-2xx reply (address, status, server reply; never the credential) or an
                unreachable server.
        """
        async with self._stream_post("/chat/completions", payload) as response:
            return await self._stream_openai_tools_response(response, on_delta)

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
        """Run the tool loop and return the final answer text.

        Each round sends the merged settings spread top-level; a truthy ``response_format``
        parameter is sent and replaces the ``response_format`` setting, otherwise the setting
        (if set) is sent, on every round (DEC-001-15). A null content becomes ``""`` both in the
        final answer and in the assistant tool-call turn appended to the next round's
        conversation (DEC-001-12). Streamed rounds go through ``_tools_stream_request``.

        Each requested tool call is handled in order: offered-tool check, argument reading,
        ``log_tool_call``, argument-name check, run, result conversion, ``log_tool_result``.

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

        Args:
            messages: Conversation turns.
            tools_definitions: Advertised tool definitions (OpenAI function format).
            tools: Callables by tool name; every advertised tool must have one.
            system: Optional system prompt, inserted as the first turn.
            max_loops: Maximum number of rounds.
            options: Per-call settings.
            stream: Stream each round.
            on_delta: Receiver of streamed text pieces; its failure propagates unchanged.
            response_format: Response format for this call; wins over the setting.

        Returns:
            The final answer text (never ``None``).

        Raises:
            ValueError: An advertised tool has no callable (before any request), or a call's
                argument names do not fit the callable (the tool is not run).
            RuntimeError: A tool raises (message names the tool), or the round limit is reached.
            LLMError: Request failure; a requested tool not offered or with an empty name; unreadable
                arguments; an unserialisable mapping or list result.
        """
        advertised = {d["function"]["name"] for d in tools_definitions}
        missing = advertised - tools.keys()
        if missing:
            raise ValueError(f"Missing callables for tools {sorted(missing)}")

        opt = self._merge_options(options=options)

        msgs = list(messages)
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
                    # Step 001 order: offered check, then argument reading, both before the tool can
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
                    tool_message: Dict[str, Any] = {"role": "tool", "content": content}
                    tool_call_id = call.get("id")
                    if tool_call_id:
                        tool_message["tool_call_id"] = tool_call_id
                    msgs.append(tool_message)

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
