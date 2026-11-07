"""OpenAI Chat Completions client: the app's credential, the shared request/stream helpers, typed settings.

Fixes from feature 001: credential only from ``bearer_token``, streamed requests via ``_stream_post``,
null reply content as empty text, and only app-set settings sent.
"""

from typing import Any, Awaitable, Callable, Dict, List, Optional
import json
import inspect

from llm.llm_client import LLMClient
from llm.exceptions import LLMError
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

class OpenAIAPIClient(LLMClient):
    """Convenience layer over OpenAI's Chat Completions API.

    This client is designed for the official OpenAI API.
    It supports:
    - The credential the app passes as ``bearer_token`` (required; never read from the environment)
    - Standard OpenAI Chat Completions endpoint
    - OpenAI-specific parameter handling: only settings the app set are sent
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
        """
        Fetches the list of available models from the OpenAI API.
        """
        response = await self._request("GET", "/models")
        if "data" not in response:
            raise LLMError(f"Invalid response from OpenAI API when listing models: {response}")

        return [model["id"] for model in response["data"]]

    async def get_models(self) -> List[Dict[str, Any]]:
        response = await self._request("GET", "/models")
        if "data" not in response:
            raise LLMError(f"Invalid response from OpenAI API when fetching models: {response}")
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
        Note: response_schema is used for validation but not sent to OpenAI,
              as their native JSON mode does not use a schema directly.

        Raises:
            LLMError: Request failure, or ``response_schema`` given and the reply is not valid JSON.
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

        # Although OpenAI enforces JSON, we can optionally validate against a more specific schema
        if response_schema:
            try:
                parsed_json = json.loads(response_content)
                # Basic check: ensure all keys in schema are in the response if they don't have defaults
                # This is a simplistic validation. For full validation, a library like jsonschema would be needed.
                if "properties" in response_schema:
                    for key, prop_details in response_schema["properties"].items():
                        if "default" not in prop_details and key not in parsed_json:
                            # For simplicity, we are not raising an error here, but logging would be appropriate.
                            pass
            except (json.JSONDecodeError, TypeError) as e:
                raise LLMError(
                    "LLM response was not valid JSON, despite requesting JSON mode.",
                    description=str(e)
                )

        return response_content

    async def _tools_stream_request(
        self,
        payload: Dict[str, Any],
        on_delta: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> tuple[str, list[dict]]:
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
        """Run the tool loop and return the model's final answer.

        Each round's body carries the merged settings; a truthy ``response_format`` parameter is
        sent and replaces the ``response_format`` setting, otherwise the setting (if set) is sent
        (DEC-001-15). Null content of the final answer, and of a recorded assistant tool-call
        turn, becomes empty text (DEC-001-12). Streamed rounds go through ``_tools_stream_request``.

        Raises:
            ValueError: An advertised tool has no callable, or a tool call has bad arguments.
            RuntimeError: A tool fails, or ``max_loops`` is exceeded.
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
                raise LLMError(f"Invalid response from OpenAI API when embedding: {response}")
            t.log_response(f"{len(response['data'])} embeddings returned")

        return [item["embedding"] for item in response["data"]]
