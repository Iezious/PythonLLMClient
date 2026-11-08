"""Shared base client for every LLM backend: HTTP request and streamed-request plumbing, SSE reading and
stream parsing, and the three-layer option merge that the backend clients inherit.
"""
from typing import Any, Awaitable, Callable, Dict, List, Optional, AsyncIterator
import codecs
import inspect
import json
import re
from contextlib import AsyncExitStack, asynccontextmanager
import aiohttp

from llm.exceptions import LLMError
from llm.message import LLMMessage
from llm.options import BaseLLMOptions
from llm.trace import LLMTracer


class LLMClient:
    """Base class for LLM API clients; subclasses override the methods that raise NotImplementedError."""
    def __init__(self, model: str, base_url: str, timeout: float, bearer_token: Optional[str] = None, default_options: Optional[BaseLLMOptions] = None) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = timeout
        self.bearer_token = bearer_token
        self.default_options = default_options
        self._tracer = LLMTracer()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure session is created."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def _request(self, method: str, path: str, **kw) -> Any:
        """Send one non-streamed request to ``base_url + path`` and return the decoded reply.

        Args:
            method: HTTP method (e.g. ``"GET"``, ``"POST"``).
            path: Path appended to the client's base address.
            **kw: Request keywords passed to ``aiohttp`` (``json``, ``headers``, ``timeout`` ...). The bearer
                header is added when the client holds a token.

        Returns:
            The decoded JSON body for an ``application/json`` 2xx reply, otherwise the raw body bytes.

        Raises:
            LLMError: On a non-2xx reply, with message ``"Request to <url> failed with status <n>"`` and the
                reply text as description (DEC-000-11). On a transport failure while connecting, sending or
                reading the body (connection error, server disconnect, payload error, timeout), chained from
                the original, with a message naming the address and the failure kind and the original
                exception's text as description; that text never contains the credential nor the phrase
                ``"status <n>"`` (DEC-001-7).
        """
        session = await self._ensure_session()
        url = f"{self.base_url}{path}"

        headers = kw.pop("headers", {})
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        timeout = aiohttp.ClientTimeout(total=kw.pop("timeout", self.timeout))

        try:
            async with session.request(method, url, headers=headers, timeout=timeout, **kw) as r:
                try:
                    r.raise_for_status()
                except aiohttp.ClientResponseError as exc:
                    # The body must be read while the response is still open (inside the ``async with``).
                    text = await r.text()
                    raise LLMError(f"Request to {url} failed with status {r.status}", description=text) from exc

                content_type = r.headers.get("Content-Type", "")
                if content_type.startswith("application/json"):
                    return await r.json()
                else:
                    return await r.read()
        except aiohttp.ClientResponseError:
            # ClientResponseError subclasses ClientError; it is a status failure, not a transport one, so it
            # must never be relabelled by the transport branch below (DEC-000-11 keeps its own format).
            raise
        except (aiohttp.ClientError, TimeoutError) as exc:
            # DEC-001-7: connection errors, disconnects, payload errors and timeouts (on 3.11
            # asyncio.TimeoutError is TimeoutError; ServerTimeoutError is both) become LLMError. Timeout is
            # tested first because ServerTimeoutError is also a ClientError.
            if isinstance(exc, TimeoutError):
                kind = "timeout"
            elif isinstance(exc, (aiohttp.ServerDisconnectedError, aiohttp.ClientPayloadError)):
                kind = "interrupted transfer"
            else:
                kind = "connection failure"
            # Timeouts often carry no text; the type name keeps the description informative. The token is
            # redacted defensively: the text comes from aiohttp, never from our headers (DEC-001-7).
            description = str(exc) or type(exc).__name__
            if self.bearer_token:
                description = description.replace(self.bearer_token, "***")
            raise LLMError(f"Request to {url} failed: {kind}", description=description) from exc

    @asynccontextmanager
    async def _stream_post(self, path: str, payload: dict[str, Any]) -> AsyncIterator[aiohttp.ClientResponse]:
        """Open a streamed POST to ``base_url + path`` and yield the open response (DEC-001-8).

        Ensures the session, adds ``Authorization: Bearer <token>`` only when the client holds a token,
        bounds the request by ``ClientTimeout(total=self.timeout)`` and posts ``payload`` as JSON. On a 2xx
        reply the open response is yielded for the caller to read and released when the block exits.

        Args:
            path: Path appended to the client's base address.
            payload: JSON request body.

        Yields:
            The open 2xx ``aiohttp.ClientResponse``.

        Raises:
            LLMError: On a non-2xx reply, after reading the reply body while the response is open, with
                message ``"Request to <url> failed with status <n>"`` and the reply text as description
                (DEC-000-11, DEC-arch-5). On a transport failure while opening the request (connection
                error, timeout), chained from the original, per DEC-001-7. The credential never appears
                in the error text.

        Exceptions raised inside the caller's ``async with`` block are not caught or converted; they reach
        the caller unchanged (DEC-000-23).
        """
        session = await self._ensure_session()
        url = f"{self.base_url}{path}"

        # US-005.AC-2: the bearer header is sent only when the client holds a token.
        headers: dict[str, str] = {}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with AsyncExitStack() as stack:
            # Only opening the request and the non-2xx check are mapped; the ``yield`` sits outside this
            # ``try`` so exceptions from the caller's block pass through unchanged (DEC-001-8, DEC-000-23).
            try:
                response = await stack.enter_async_context(
                    session.post(url, json=payload, headers=headers, timeout=timeout)
                )
                if not 200 <= response.status < 300:
                    # Read the reply while the response is still open; same format as _request (DEC-000-11).
                    text = await response.text()
                    raise LLMError(f"Request to {url} failed with status {response.status}", description=text)
            except aiohttp.ClientResponseError:
                # A status error is not a transport failure; never relabel it (see _request).
                raise
            except (aiohttp.ClientError, TimeoutError) as exc:
                # DEC-001-7 mapping, identical to _request (timeout first: ServerTimeoutError is both).
                if isinstance(exc, TimeoutError):
                    kind = "timeout"
                elif isinstance(exc, (aiohttp.ServerDisconnectedError, aiohttp.ClientPayloadError)):
                    kind = "interrupted transfer"
                else:
                    kind = "connection failure"
                description = str(exc) or type(exc).__name__
                if self.bearer_token:
                    description = description.replace(self.bearer_token, "***")
                raise LLMError(f"Request to {url} failed: {kind}", description=description) from exc

            # The exit stack releases the response when the caller's block exits, normally or by exception.
            yield response

    async def _iter_sse_data(self, response: aiohttp.ClientResponse) -> AsyncIterator[str]:
        """Read an open streamed response and yield the data of each server-sent event (DEC-001-10).

        Protocol-neutral reader shared by the chat-completions parsers. Bytes are decoded as UTF-8
        incrementally across reads, so a multi-byte character split between two reads arrives whole.
        Invalid bytes are replaced by U+FFFD rather than dropped.
        LF, CRLF and bare CR all terminate a line; a CR ending one read followed by an LF starting the
        next counts as one CRLF terminator. A blank line dispatches the current event. Comment lines
        (starting with ``:``) and non-data fields (``event:``, ``id:``, ``retry:`` ...) are ignored. One
        space after ``data:`` is removed; the data lines of a multi-line event are joined by ``"\\n"``.
        A final event not followed by a blank line is still yielded. The ``[DONE]`` sentinel is yielded
        as a payload; interpreting it is the parsers' job.

        Args:
            response: The open streamed response; only ``response.content.iter_chunked(n)`` is read.

        Yields:
            The data string of each event, once per event.

        Raises:
            LLMError: When reading the stream fails in transport (payload error, server disconnect,
                timeout), chained from the original, with the DEC-001-7 message rule (DEC-001-1). Only the
                reading is mapped; exceptions raised by the consumer between yields are not caught here.
        """
        # One incremental decoder for the whole stream keeps a character split between reads whole;
        # "replace" makes invalid bytes visible instead of dropping them (DEC-001-10).
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        remainder = ""
        # True when the last text seen ended with a CR that was taken as a terminator: if the next read
        # starts with LF, the pair is one CRLF and the LF must not end an extra (empty) line (DEC-001-10).
        skip_lf = False
        data_lines: list[str] = []
        url = getattr(response, "url", "")
        reader = aiter(response.content.iter_chunked(1024))

        def take_line(line: str) -> str | None:
            """Apply one SSE line to the pending event; return the event's data if the line dispatches it."""
            nonlocal data_lines
            if not line:
                # Blank line: dispatch only an event that has data.
                if not data_lines:
                    return None
                event, data_lines = "\n".join(data_lines), []
                return event
            if line.startswith(":"):
                return None
            name, sep, value = line.partition(":")
            if sep and value.startswith(" "):
                value = value[1:]
            # event / id / retry and unknown fields carry nothing the chat-completions parsers use.
            if name == "data":
                data_lines.append(value)
            return None

        def split(text: str) -> list[str]:
            """Append decoded text and return the completed lines; the unterminated tail stays pending."""
            nonlocal remainder, skip_lf
            if not text:
                # A read holding only part of a multi-byte character decodes to nothing; keep skip_lf.
                return []
            if skip_lf and text.startswith("\n"):
                text = text[1:]
            buffer = remainder + text
            parts = re.split(r"\r\n|\r|\n", buffer)
            remainder = parts.pop()
            skip_lf = buffer.endswith("\r")
            return parts

        while True:
            # Only the read itself is inside the try: the parser runs the receiver between our yields, and
            # its exceptions must never be relabelled as transport failures (DEC-001-7, DEC-000-23).
            try:
                chunk = await anext(reader)
            except StopAsyncIteration:
                break
            except (aiohttp.ClientError, TimeoutError) as exc:
                # Same kinds and shape as _request / _stream_post (DEC-001-7); the address is the
                # response's URL. Timeout first: ServerTimeoutError is both a TimeoutError and a ClientError.
                if isinstance(exc, TimeoutError):
                    kind = "timeout"
                elif isinstance(exc, (aiohttp.ServerDisconnectedError, aiohttp.ClientPayloadError)):
                    kind = "interrupted transfer"
                else:
                    kind = "connection failure"
                description = str(exc) or type(exc).__name__
                if self.bearer_token:
                    description = description.replace(self.bearer_token, "***")
                raise LLMError(f"Request to {url} failed: {kind}", description=description) from exc

            for line in split(decoder.decode(chunk)):
                event = take_line(line)
                if event is not None:
                    yield event

        # End of stream: flush the decoder (a truncated trailing character becomes U+FFFD), treat an
        # unterminated last line as complete, and dispatch a pending event with no trailing blank line.
        lines = split(decoder.decode(b"", final=True))
        if remainder:
            lines.append(remainder)
        for line in lines:
            event = take_line(line)
            if event is not None:
                yield event
        if data_lines:
            yield "\n".join(data_lines)

    @staticmethod
    def _read_stream_frame(data: str) -> tuple[dict[str, Any], bool] | None:
        """Decode one chat-completions data payload for both stream parsers (DEC-001-9, DEC-001-1).

        Private helper, not a seam.

        Args:
            data: One event's data string from ``_iter_sse_data`` (``[DONE]`` is handled by the caller).

        Returns:
            ``None`` for a payload to skip: not JSON, JSON but not an object, or an object without usable
            ``choices`` and without ``error`` (e.g. a usage frame). Otherwise ``(delta, finished)``: the
            first choice's ``delta`` (``{}`` when absent or not an object) and whether any choice carries
            a non-null ``finish_reason``.

        Raises:
            LLMError: When the object has a truthy ``error``, whether or not it also has ``choices``; the
                description is ``error.message`` when that is a string, otherwise the error as JSON text.
        """
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        error = payload.get("error")
        if error:
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                reason = error["message"]
            else:
                reason = json.dumps(error, ensure_ascii=False)
            raise LLMError("The server reported an error in the stream", description=reason)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        finished = any(isinstance(c, dict) and c.get("finish_reason") is not None for c in choices)
        first = choices[0] if isinstance(choices[0], dict) else {}
        delta = first.get("delta")
        return (delta if isinstance(delta, dict) else {}), finished

    async def _stream_openai_chat_response(
        self,
        response: aiohttp.ClientResponse,
        on_delta: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> str:
        """Parse a chat-completions stream into the full text, reasoning tagged, feeding the receiver.

        Reads data payloads from ``self._iter_sse_data(response)``; never touches ``response`` itself, so
        tests may pass ``None`` when the reader is replaced. ``[DONE]`` ends the stream. A payload that is
        not JSON, is JSON but not an object, or is an object with neither ``choices`` nor ``error`` is
        skipped (DEC-001-9). Only ``choices[0].delta`` is read.

        Reasoning rules: ``reasoning_content`` is delivered inside one ``<think>`` ... ``</think>`` block
        before the answer; the block is opened by the first reasoning text, closed by a ``</think>`` prefix
        on the first content text after it, and closed at the end if still open; a model that does not
        reason gets no block. A piece carrying both reasoning and content delivers the reasoning first
        (with the opening marker if it starts the block) and then the content (with the closing marker
        before it) (DEC-001-11). The returned text equals the concatenation of what the receiver got.

        Args:
            response: The open streamed response, passed through to the reader.
            on_delta: Optional receiver, called inline with each text piece; an awaitable result is
                awaited (DEC-000-23).

        Returns:
            The full streamed text, reasoning included and tagged.

        Raises:
            LLMError: On an error frame (an object with ``error``), with the server's reason in the text;
                no later piece reaches the receiver (DEC-001-9). On an early end: the stream ends with
                neither ``[DONE]`` nor a non-null ``finish_reason`` on any choice (DEC-001-1). On a
                transport failure raised by the reader.
            Exception: Any exception raised by ``on_delta`` propagates unchanged and ends the call
                (DEC-000-23).
        """
        chunks: List[str] = []
        in_reasoning = False
        # DEC-001-1: either completion signal is enough; neither means the stream ended early.
        completed = False
        async for data in self._iter_sse_data(response):
            if data.strip() == "[DONE]":
                completed = True
                break
            frame = self._read_stream_frame(data)
            if frame is None:
                continue
            delta_obj, finished = frame
            completed = completed or finished
            reasoning = delta_obj.get("reasoning_content")
            content = delta_obj.get("content")

            # Emit reasoning_content wrapped in <think> tags. No early ``continue``: a delta may also carry
            # content, which must follow as its own piece (DEC-001-11, supersedes DEC-000-21).
            if reasoning:
                text = ""
                if not in_reasoning:
                    in_reasoning = True
                    text = "<think>" + reasoning
                else:
                    text = reasoning
                chunks.append(text)
                if on_delta:
                    result = on_delta(text)
                    if inspect.isawaitable(result):
                        await result

            if content:
                text = content
                if in_reasoning:
                    in_reasoning = False
                    text = "</think>" + content
                chunks.append(text)
                if on_delta:
                    result = on_delta(text)
                    if inspect.isawaitable(result):
                        await result

        if not completed:
            raise LLMError("The stream ended before the reply was complete")

        # Close unclosed thinking tag
        if in_reasoning:
            chunks.append("</think>")
            if on_delta:
                result = on_delta("</think>")
                if inspect.isawaitable(result):
                    await result

        return "".join(chunks)

    @staticmethod
    def _merge_options_base(
            *,
        options: Optional[Dict[str, Any]],
        supported_params: Dict[str, Any],
        alias_map: Dict[str, str],
        default_opts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        opts: Dict[str, Any] = dict(supported_params)          # defaults

        # 1. constructor default options
        if default_opts:
            for raw_key, val in default_opts.items():
                key = alias_map.get(raw_key, raw_key)
                if key in supported_params:
                    opts[key] = val

        # 2. call-time options
        if options:
            for raw_key, val in options.items():
                key = alias_map.get(raw_key, raw_key)
                if key in supported_params:
                    opts[key] = val

        # 3. clean None
        return {k: v for k, v in opts.items() if v is not None}

    def _merge_options(
        self,
        *,
        options: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # This method will be implemented by subclasses
        raise NotImplementedError

    async def list_models(self) -> List[str]:
        raise NotImplementedError

    async def get_models(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        on_delta: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> str:
        raise NotImplementedError

    async def chat(
        self,
        messages: List[LLMMessage],
        *,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        on_delta: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> str:
        raise NotImplementedError

    async def chat_json(
        self,
        messages: List[LLMMessage],
        *,
        response_schema: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Chat with structured JSON output using response schema.

        Args:
            messages: List of message dicts with role and content
            response_schema: JSON schema for the expected response structure
            system: Optional system message
            options: Optional LLM parameters (typed object)

        Returns:
            JSON string response conforming to schema
        """
        raise NotImplementedError

    async def embed(
        self,
        text: str,
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """Embed a single text string and return its vector."""
        vectors = await self.embed_batch([text], options=options)
        if not vectors:
            raise LLMError("Embedding response was empty.")
        return vectors[0]

    async def embed_batch(
        self,
        texts: List[str],
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[List[float]]:
        """Embed multiple text strings and return a list of vectors."""
        raise NotImplementedError

    async def _stream_openai_tools_response(
        self,
        response: aiohttp.ClientResponse,
        on_delta: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> tuple[str, list[dict]]:
        """Parse an OpenAI streaming response that may contain tool_calls.

        Assembles streamed tool_call fragments into calls and calls on_delta for content chunks.
        Returns (content, tool_calls) where tool_calls is empty for final answer rounds.

        Tool-call fragments are grouped per round as follows (DEC-002-6):

        - **Positioned fragments** (an integer ``index``) belong to the call at that position. A truthy
          ``id`` is set on the call. A non-empty ``function.name`` is set, not appended, so a name repeated
          in every fragment appears once. Non-empty ``function.arguments`` text is appended.
        - **Positionless fragments** (``index`` missing or not an integer) are matched against the current
          positionless call, the last one started. A fragment starts a new call when it carries an id
          different from the current call's non-empty id, or when it carries no id but a non-empty name
          while the current call already has a name. A fragment carrying the same id as the current call
          continues it (its name is set, so a repeated name is harmless). Otherwise the fragment continues
          the current call; with no current call it starts one.
        - **Order**: positioned calls sorted by position, then positionless calls in order of appearance.

        Every call has the shape ``{"id": str, "type": "function", "function": {"name": str,
        "arguments": str}}``; ``id`` is ``""`` when the server sent none.

        `reasoning_content` deltas are streamed through on_delta wrapped in <think>/</think>
        tags, exactly as _stream_openai_chat_response does. Unlike that method, the reasoning
        text is deliberately NOT part of the returned content string: chat_with_tools feeds
        the return value back into `messages` as the assistant turn for the next loop
        iteration, and re-sending the model's own thinking wastes context and is rejected or
        mishandled by some servers. This asymmetry is intentional -- do not "fix" it.

        The in_reasoning state is per call (i.e. per tool-calling round), not shared across
        the chat_with_tools loop. Reasoning and content handling is otherwise unchanged (DEC-000-22).

        Reads data payloads from ``self._iter_sse_data(response)`` and never touches ``response`` itself.
        ``[DONE]`` ends the stream. A payload that is not JSON, is JSON but not an object, or is an object
        with neither ``choices`` nor ``error`` is skipped (DEC-001-9). Tool-call fragment grouping follows
        DEC-002-6 above (supersedes the open part of DEC-000-32).

        Args:
            response: The open streamed response, passed through to the reader.
            on_delta: Optional receiver, called inline with each streamed text piece; an awaitable result
                is awaited (DEC-000-23).

        Returns:
            ``(content, tool_calls)``: the content text without reasoning, and the assembled tool calls,
            positioned calls by position then positionless calls in order of appearance (empty for a
            final-answer round).

        Raises:
            LLMError: On an error frame, with the server's reason in the text (DEC-001-9). On an early end:
                the stream ends with neither ``[DONE]`` nor a non-null ``finish_reason`` on any choice
                (DEC-001-1). On a transport failure raised by the reader.
            Exception: Any exception raised by ``on_delta`` propagates unchanged and ends the call
                (DEC-000-23).
        """
        content_chunks: List[str] = []
        # DEC-002-6: positioned calls keyed by position; positionless calls kept in order of appearance,
        # the last one being the "current" positionless call that new positionless fragments match against.
        tool_call_map: Dict[int, dict] = {}
        positionless_calls: list[dict] = []
        in_reasoning = False

        def new_tool_call() -> dict:
            """Return an empty call in the shape the tool loop reads (id "" until the server sends one)."""
            return {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}

        async def emit(text: str) -> None:
            if on_delta:
                result = on_delta(text)
                if inspect.isawaitable(result):
                    await result

        # DEC-001-1: either completion signal is enough; neither means the stream ended early.
        completed = False
        async for data in self._iter_sse_data(response):
            if data.strip() == "[DONE]":
                completed = True
                break
            frame = self._read_stream_frame(data)
            if frame is None:
                continue
            delta, finished = frame
            completed = completed or finished

            # Tool calls first and unconditionally: a single delta may carry both
            # reasoning_content and tool_calls, so reasoning must not short-circuit this.
            for tc in delta.get("tool_calls") or []:
                func = tc.get("function") or {}
                call_id = tc.get("id")
                name = func.get("name")
                idx = tc.get("index")
                # DEC-002-6: only a real integer is a position; bool is excluded because
                # True/False are not positions a server means to send.
                if isinstance(idx, int) and not isinstance(idx, bool):
                    if idx not in tool_call_map:
                        tool_call_map[idx] = new_tool_call()
                    entry = tool_call_map[idx]
                else:
                    current = positionless_calls[-1] if positionless_calls else None
                    # DEC-002-6: a different non-empty id, or (without an id) a name arriving on a call that
                    # already has one, starts a new call. A matching id continues the current call even when
                    # the name is repeated, so servers that resend id and name in every fragment still work.
                    if (
                        current is None
                        or (call_id and current["id"] and call_id != current["id"])
                        or (not call_id and name and current["function"]["name"])
                    ):
                        current = new_tool_call()
                        positionless_calls.append(current)
                    entry = current
                if call_id:
                    entry["id"] = call_id
                # DEC-002-6: the name is set, not appended, so a name repeated in every fragment appears once.
                if name:
                    entry["function"]["name"] = name
                if func.get("arguments"):
                    entry["function"]["arguments"] += func["arguments"]

            # Emit reasoning_content wrapped in <think> tags (streamed only, not returned)
            reasoning = delta.get("reasoning_content")
            if reasoning:
                if not in_reasoning:
                    in_reasoning = True
                    await emit("<think>" + reasoning)
                else:
                    await emit(reasoning)

            content = delta.get("content")
            if content:
                content_chunks.append(content)
                text = content
                if in_reasoning:
                    in_reasoning = False
                    text = "</think>" + content
                await emit(text)

        if not completed:
            raise LLMError("The stream ended before the reply was complete")

        # Close unclosed thinking tag: a round may end with reasoning plus tool_calls only
        if in_reasoning:
            in_reasoning = False
            await emit("</think>")

        # DEC-002-6 order: positioned calls by position, then positionless calls in order of appearance.
        tool_calls = [tool_call_map[i] for i in sorted(tool_call_map)] + positionless_calls
        return "".join(content_chunks), tool_calls

    async def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        *,
        tools_definitions: List[Dict[str, Any]],
        tools: Dict[str, Callable[..., Any]],
        system: Optional[str] = None,
        max_loops: int = 5,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        on_delta: Optional[Callable[[str], Awaitable[None]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        raise NotImplementedError

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session is not None:
            await self.session.close()
            self.session = None
