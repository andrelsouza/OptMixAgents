"""
Tool registry for OptMix agents.

Maps tool names to schemas and callable functions. Converts tool
definitions to the Anthropic Messages API format for LLM tool calling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Definition of a single tool parameter."""

    name: str
    type: str  # "string", "number", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


@dataclass
class ToolSchema:
    """Schema describing a tool that agents can invoke."""

    name: str
    description: str
    parameters: list[ToolParameter]
    returns_description: str
    agent_scope: list[str]  # which agents can use this tool


class ToolRegistry:
    """Registry mapping tool names to schemas and implementations."""

    def __init__(self) -> None:
        self._schemas: dict[str, ToolSchema] = {}
        self._funcs: dict[str, Callable[..., Any]] = {}

    def register(self, schema: ToolSchema, func: Callable[..., Any]) -> None:
        """Register a tool with its schema and implementation."""
        if schema.name in self._schemas:
            raise ValueError(f"Tool '{schema.name}' is already registered.")
        self._schemas[schema.name] = schema
        self._funcs[schema.name] = func

    def get(self, name: str) -> tuple[ToolSchema, Callable[..., Any]]:
        """Get a tool's schema and function by name."""
        if name not in self._schemas:
            raise KeyError(f"Tool '{name}' is not registered.")
        return self._schemas[name], self._funcs[name]

    def list_for_agent(self, agent_name: str) -> list[ToolSchema]:
        """List all tool schemas available to a specific agent."""
        return [
            schema
            for schema in self._schemas.values()
            if agent_name in schema.agent_scope
        ]

    def execute(self, name: str, arguments: dict[str, Any], state: Any) -> dict[str, Any]:
        """Execute a tool by name with given arguments and state."""
        if name not in self._funcs:
            return {"status": "error", "message": f"Tool '{name}' not registered."}

        schema = self._schemas[name]
        func = self._funcs[name]

        # Fill in defaults for missing optional parameters
        filled_args = dict(arguments)
        for param in schema.parameters:
            if param.name not in filled_args and not param.required and param.default is not None:
                filled_args[param.name] = param.default

        try:
            return func(state, **filled_args)
        except Exception as e:
            logger.error("Tool '%s' execution failed: %s", name, e)
            return {
                "status": "error",
                "message": f"Tool '{name}' failed: {str(e)}",
                "summary": f"Error executing {name}: {str(e)}",
            }

    def to_anthropic_tools(self, agent_name: str) -> list[dict[str, Any]]:
        """Convert agent's tools to Anthropic Messages API format."""
        tools = []
        for schema in self.list_for_agent(agent_name):
            properties: dict[str, Any] = {}
            required: list[str] = []

            for param in schema.parameters:
                prop: dict[str, Any] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.default is not None:
                    prop["default"] = param.default
                if param.enum:
                    prop["enum"] = param.enum
                properties[param.name] = prop

                if param.required:
                    required.append(param.name)

            tool_def: dict[str, Any] = {
                "name": schema.name,
                "description": schema.description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
            tools.append(tool_def)

        return tools

    def __contains__(self, name: str) -> bool:
        return name in self._schemas

    def __len__(self) -> int:
        return len(self._schemas)
