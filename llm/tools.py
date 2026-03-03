"""Utilities for converting Pydantic models to OpenAI-compatible tool definitions."""

from typing import Any

from pydantic import BaseModel


def pydantic_to_openai_tool(name: str, description: str, schema: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model into an OpenAI function-calling tool definition.

    Args:
        name: The tool/function name the LLM will use to invoke it.
        description: Human-readable description of what the tool does.
        schema: A Pydantic BaseModel subclass whose fields define the parameters.

    Returns:
        A dict in OpenAI's tool format ready to pass as a ``tools_definitions`` entry.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema.model_json_schema(),
        },
    }
