#!/usr/bin/env python3
"""
ollama_wrapper.py
=================
Lightweight, dependency-free wrapper around Ollama's REST API.

Key features
------------
* **Unified options interface** -- pass generation parameters as keywords or an
  explicit ``options`` mapping.  Validation happens centrally in
  :py:meth:`_merge_options`.
* **Two endpoints** -- :py:meth:`OllamaClient.generate` (single prompt) and
  :py:meth:`OllamaClient.chat` (OpenAI-style messages array).
* **Strict validation** -- unknown parameter names raise immediately.
* **Streaming deliberately disabled** -- wrapper is stateless and returns the
  full reply text.

-----------------------------------------------------------------
Supported option keys (Ollama API 2025-05)
-----------------------------------------------------------------
temperature, top_p, top_k, repeat_penalty,
presence_penalty, frequency_penalty,
max_tokens          (alias: num_predict, length)
num_ctx             (alias: ctx_len, ctx_length, context_length)
num_keep,
mirostat            (alias: mirostat_mode)   0 / 1 / 2
mirostat_tau, mirostat_eta,
seed
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Dict, List, Optional

from llm.llm_client import LLMClient
from llm.exceptions import LLMError
from llm.message import LLMMessage
from llm.options import OllamaOptions

# ---------------------------------------------------------------------------
# Parameter tables
# ---------------------------------------------------------------------------

_OLLAMA_SUPPORTED_PARAMS: Dict[str, Any] = {
    # Sampling
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "min_p": 0.05,
    "repeat_penalty": 1.1,

    # Penalties
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,

    # Length / context
    "num_predict": 512,
    "num_ctx": None,
    "num_keep": None,

    # Mirostat
    "mirostat": 0,          # 0 (off), 1, 2
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.1,

    # Misc
    "seed": None,
}

_OLLAMA_ALIAS_MAP = {
    # length & context
    "max_tokens": "num_predict",
    "length": "num_predict",
    "ctx_len": "num_ctx",
    "ctx_length": "num_ctx",
    "context_length": "num_ctx",
    # mirostat
    "mirostat_mode": "mirostat",
}


# ---------------------------------------------------------------------------
# Ollama Client
# ---------------------------------------------------------------------------

class OllamaAPIClient(LLMClient):
    """Small convenience layer over Ollama's HTTP API.

    This client is designed for interacting with Ollama servers.
    It supports:
    - Custom base URL (default: http://localhost:11434)
    - Optional bearer token authentication
    - Unified option handling for Ollama-specific parameters
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 timeout: float = 600.0, bearer_token: Optional[str] = None,
                 default_options: Optional[OllamaOptions] = None) -> None:
        super().__init__(model, base_url, timeout, bearer_token, default_options)

    # option merge / validate
    def _merge_options(
        self,
        *,
        options: Optional[OllamaOptions],
    ) -> Dict[str, Any]:
        opts = self._merge_options_base(
            options=options,
            supported_params=_OLLAMA_SUPPORTED_PARAMS,
            alias_map=_OLLAMA_ALIAS_MAP,
            default_opts=self.default_options,
        )
        # remove mirostat_tau/eta unless mirostat is active
        if opts.get("mirostat", 0) == 0:
            opts.pop("mirostat", None)
            opts.pop("mirostat_tau", None)
            opts.pop("mirostat_eta", None)
        return opts

    # model utils
    async def list_models(self) -> List[str]:
        data = await self._request("GET", "/api/tags")
        return [m["name"] for m in data.get("models", [])]

    # generate (single prompt)
    async def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        options: Optional[OllamaOptions] = None,
        stream: bool = False,
    ) -> str:
        if stream:
            raise ValueError("Streaming not supported by this wrapper.")

        opt = self._merge_options(options=options)

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": opt,
        }
        if system:
            payload["system"] = system

        with self._tracer.trace("generate", self.model) as t:
            t.log_system(system)
            t.log_messages([{"role": "user", "content": prompt}])
            t.log_request("POST", "/api/generate", payload)
            result = (await self._request("POST", "/api/generate", json=payload))["response"]
            t.log_response(result)

        return result

    # chat (messages array)
    async def chat(
        self,
        messages: List[LLMMessage],
        *,
        system: Optional[str] = None,
        options: Optional[OllamaOptions] = None,
        stream: bool = False,
    ) -> str:
        if stream:
            raise ValueError("Streaming not supported by this wrapper.")

        opt = self._merge_options(options=options)

        msgs = list(messages)
        if system:
            msgs.insert(0, {"role": "system", "content": system})

        payload = {
            "model": self.model,
            "messages": msgs,
            "stream": False,
            "options": opt,
        }

        with self._tracer.trace("chat", self.model) as t:
            t.log_system(system)
            t.log_messages(msgs)
            t.log_request("POST", "/api/chat", payload)
            response = await self._request("POST", "/api/chat", json=payload)
            result = response["message"]["content"]
            t.log_response(result)

        return result

    async def chat_json(
        self,
        messages: List[LLMMessage],
        *,
        response_schema: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
        options: Optional[OllamaOptions] = None,
    ) -> str:
        """
        Chat with structured JSON output using Ollama's JSON mode.
        Note: response_schema is for client-side validation reference and is not passed to Ollama.
        """
        opt = self._merge_options(options=options)

        msgs = list(messages)
        if system:
            msgs.insert(0, {"role": "system", "content": system})

        payload = {
            "model": self.model,
            "messages": msgs,
            "stream": False,
            "format": "json",
            "options": opt,
        }

        with self._tracer.trace("chat_json", self.model) as t:
            t.log_system(system)
            t.log_messages(msgs)
            t.log_request("POST", "/api/chat", payload)
            response = await self._request("POST", "/api/chat", json=payload)
            response_content = response["message"]["content"]
            t.log_response(response_content)

        # Basic validation that the response is valid JSON
        try:
            json.loads(response_content)
        except json.JSONDecodeError:
            raise LLMError("LLM response was not valid JSON, despite requesting JSON mode.")

        return response_content

    # chat + tools (optional)
    async def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        *,
        tools_definitions: List[Dict[str, Any]],
        tools: Dict[str, Callable[..., Any]],
        system: Optional[str] = None,
        max_loops: int = 5,
        options: Optional[OllamaOptions] = None,
        stream: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
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

        with self._tracer.trace("chat_with_tools", self.model) as t:
            t.log_system(system)
            t.log_messages(msgs)

            for loop_idx in range(max_loops):
                payload = {
                    "model": self.model,
                    "messages": msgs,
                    "tools": tools_definitions,
                    "stream": False,
                    "options": opt,
                }
                if response_format:
                    payload["format"] = "json"

                t.log_request("POST", f"/api/chat (loop {loop_idx})", payload)
                assistant_msg = (await self._request("POST", "/api/chat", json=payload))["message"]
                tool_calls = assistant_msg.get("tool_calls", [])

                if not tool_calls:                        # finished
                    final = assistant_msg["content"]
                    msgs.append({"role": "assistant", "content": final})
                    t.log_response(final)
                    return final

                # run tools
                for call in tool_calls:
                    name = call["function"]["name"]
                    func = tools[name]
                    kwargs = call["function"].get("arguments") or {}
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
                    msgs.append({"role": "tool", "name": name, "content": content})

        raise RuntimeError("Tool loop exceeded max_loops -- possible infinite cycle.")

    async def embed_batch(
        self,
        texts: List[str],
        *,
        options: Optional[OllamaOptions] = None,
    ) -> List[List[float]]:
        opt = self._merge_options(options=options)

        payload: Dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "options": opt,
        }

        with self._tracer.trace("embed_batch", self.model) as t:
            t.log_request("POST", "/api/embed", payload)

            try:
                response = await self._request("POST", "/api/embed", json=payload)
                if "embeddings" not in response:
                    raise LLMError(f"Invalid response from Ollama embed endpoint: {response}")
                t.log_response(f"{len(response['embeddings'])} embeddings returned")
                return response["embeddings"]
            except LLMError as exc:
                if "status 404" not in str(exc):
                    raise

            t.log_response("Falling back to /api/embeddings (per-text)")
            vectors: List[List[float]] = []
            for text in texts:
                single_payload = {
                    "model": self.model,
                    "prompt": text,
                    "options": opt,
                }
                response = await self._request("POST", "/api/embeddings", json=single_payload)
                if "embedding" not in response:
                    raise LLMError(f"Invalid response from Ollama embeddings endpoint: {response}")
                vectors.append(response["embedding"])

            t.log_response(f"{len(vectors)} embeddings returned (fallback)")

        return vectors
