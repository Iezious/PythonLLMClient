from typing import Any, Awaitable, Callable, Dict, List, Optional
import json
import inspect
import aiohttp

from llm.llm_client import LLMClient
from llm.exceptions import LLMError
from llm.message import LLMMessage
from llm.options import OpenAIOptions

_LLAMA_SWAP_SUPPORTED_PARAMS: Dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": None,
    "enable_thinking": None,
    "reasoning_effort": None,
}

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
        if not stream:
            return (await self._request("POST", "/chat/completions", json=payload))["choices"][0]["message"]["content"]

        session = await self._ensure_session()
        url = f"{self.base_url}/chat/completions"
        headers = {}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with session.post(url, headers=headers, timeout=timeout, json=payload) as response:
            response.raise_for_status()
            return await self._stream_openai_chat_response(response, on_delta)

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
        opt = self._merge_options(options=options)

        msgs = list(messages)
        if system:
            msgs.insert(0, {"role": "system", "content": system})

        payload = {
            "model": self.model,
            "messages": msgs,
            "stream": False,
            "response_format": {"type": "json_object"},
            **opt,
        }

        response_content = (await self._request("POST", "/chat/completions", json=payload))["choices"][0]["message"]["content"]

        if response_schema:
            try:
                json.loads(response_content)
            except (json.JSONDecodeError, TypeError) as exc:
                raise LLMError(
                    "LLM response was not valid JSON, despite requesting JSON mode.",
                    description=str(exc),
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
    ) -> str:
        if stream:
            raise ValueError("Streaming not supported by this wrapper.")

        advertised = {d["function"]["name"] for d in tools_definitions}
        missing = advertised - tools.keys()
        if missing:
            raise ValueError(f"Missing callables for tools {sorted(missing)}")

        opt = self._merge_options(options=options)

        msgs = list(messages)
        if system:
            msgs.insert(0, {"role": "system", "content": system})

        for _ in range(max_loops):
            payload = {
                "model": self.model,
                "messages": msgs,
                "tools": tools_definitions,
                "stream": False,
                **opt,
            }
            assistant_msg = (await self._request("POST", "/chat/completions", json=payload))["choices"][0]["message"]
            tool_calls = assistant_msg.get("tool_calls", [])

            if not tool_calls:
                msgs.append({"role": "assistant", "content": assistant_msg.get("content", "")})
                return assistant_msg.get("content", "")

            msgs.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.get("content", ""),
                    "tool_calls": tool_calls,
                }
            )

            for call in tool_calls:
                name = call["function"]["name"]
                func = tools[name]
                kwargs = call["function"].get("arguments") or {}
                if isinstance(kwargs, str):
                    kwargs = json.loads(kwargs)
                if not all(k in inspect.signature(func).parameters for k in kwargs):
                    raise ValueError(f"Bad args for tool '{name}': {kwargs}")
                try:
                    result = func(**kwargs)
                    if inspect.isawaitable(result):
                        result = await result
                except Exception as exc:
                    raise RuntimeError(f"Tool '{name}' failed: {exc}") from exc

                content = json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
                tool_message = {"role": "tool", "content": content}
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

        response = await self._request("POST", "/embeddings", json=payload)
        if "data" not in response:
            raise LLMError(f"Invalid response from llama-swap when embedding: {response}")

        return [item["embedding"] for item in response["data"]]
