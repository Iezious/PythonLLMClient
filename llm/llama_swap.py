"""llama-swap Chat Completions client: optional credential, the shared request/stream helpers, typed settings.

Fixes from feature 001: streamed requests via ``_stream_post``, null reply content as empty text,
and only app-set settings sent (including stop sequences, token bias, end-user id, response format).
"""

from typing import Any, Awaitable, Callable, Dict, List, Optional
import json
import inspect

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


class LlamaSwapAPIClient(LLMClient):
    """OpenAI-compatible client for llama-swap (no auth by default).

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
        response = await self._request("GET", "/models")
        if "data" not in response:
            raise LLMError(f"Invalid response from llama-swap when listing models: {response}")
        return [model["id"] for model in response["data"]]

    async def get_models(self) -> List[Dict[str, Any]]:
        response = await self._request("GET", "/models")
        if "data" not in response:
            raise LLMError(f"Invalid response from llama-swap when fetching models: {response}")
        return response["data"]

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
        (DEC-001-12). When ``response_schema`` is given the text must parse as JSON.

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
            ValueError: A tool callable is missing or a call carries bad arguments.
            RuntimeError: A tool failed, or the loop exceeded ``max_loops``.
            LLMError: Request failure.
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
                    name = call["function"]["name"]
                    func = tools[name]
                    kwargs = call["function"].get("arguments") or {}
                    if isinstance(kwargs, str):
                        kwargs = json.loads(kwargs)
                    t.log_tool_call(name, kwargs)
                    if not all(k in inspect.signature(func).parameters for k in kwargs):
                        raise ValueError(f"Bad args for tool '{name}': {kwargs}")
                    try:
                        result = func(**kwargs)
                        if inspect.isawaitable(result):
                            result = await result
                    except Exception as exc:
                        raise RuntimeError(f"Tool '{name}' failed: {exc}") from exc

                    content = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
                    t.log_tool_result(name, content)
                    tool_message: Dict[str, Any] = {"role": "tool", "content": content}
                    tool_call_id = call.get("id")
                    if tool_call_id:
                        tool_message["tool_call_id"] = tool_call_id
                    msgs.append(tool_message)

        raise RuntimeError("Tool loop exceeded max_loops -- possible infinite cycle.")

    async def embed_batch(
        self,
        texts: List[str],
        *,
        options: Optional[OpenAIOptions] = None,
    ) -> List[List[float]]:
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
            response = await self._request("POST", "/embeddings", json=payload)
            if "data" not in response:
                raise LLMError(f"Invalid response from llama-swap when embedding: {response}")
            t.log_response(f"{len(response['data'])} embeddings returned")

        return [item["embedding"] for item in response["data"]]
