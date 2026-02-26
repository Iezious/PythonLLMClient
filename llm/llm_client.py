from typing import Any, Awaitable, Callable, Dict, List, Optional, AsyncIterator
import inspect
import json
import aiohttp

from llm.exceptions import LLMError
from llm.message import LLMMessage
from llm.options import BaseLLMOptions


class LLMClient:
    """Abstract base class for LLM API clients."""
    def __init__(self, model: str, base_url: str, timeout: float, bearer_token: Optional[str] = None, default_options: Optional[BaseLLMOptions] = None) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = timeout
        self.bearer_token = bearer_token
        self.default_options = default_options

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure session is created."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    async def _request(self, method: str, path: str, **kw) -> Any:
        session = await self._ensure_session()
        url = f"{self.base_url}{path}"

        headers = kw.pop("headers", {})
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        timeout = aiohttp.ClientTimeout(total=kw.pop("timeout", self.timeout))

        async with session.request(method, url, headers=headers, timeout=timeout, **kw) as r:
            try:
                r.raise_for_status()
            except aiohttp.ClientResponseError as exc:                    # pragma: no cover
                text = await r.text()
                raise LLMError(f"Request to {url} failed with status {r.status}", description=text) from exc

            content_type = r.headers.get("Content-Type", "")
            if content_type.startswith("application/json"):
                return await r.json()
            else:
                return await r.read()

    async def _iter_sse_data(self, response: aiohttp.ClientResponse) -> AsyncIterator[str]:
        buffer = ""
        async for chunk in response.content.iter_chunked(1024):
            text = chunk.decode("utf-8", errors="ignore")
            buffer += text
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    yield line[len("data:"):].strip()
        if buffer.strip().startswith("data:"):
            yield buffer.strip()[len("data:"):].strip()

    async def _stream_openai_chat_response(
        self,
        response: aiohttp.ClientResponse,
        on_delta: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> str:
        chunks: List[str] = []
        in_reasoning = False
        async for data in self._iter_sse_data(response):
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = payload.get("choices") or []
            if not choices:
                continue
            delta_obj = choices[0].get("delta") or {}
            reasoning = delta_obj.get("reasoning_content")
            content = delta_obj.get("content")

            # Emit reasoning_content wrapped in <think> tags
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
                continue

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
    ) -> str:
        raise NotImplementedError

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session is not None:
            await self.session.close()
            self.session = None
