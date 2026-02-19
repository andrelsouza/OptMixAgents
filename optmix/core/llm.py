"""
LLM integration layer for OptMix.

Provides an abstract interface for LLM providers with tool/function calling
support. The Anthropic implementation is the primary backend.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """A tool invocation requested by the LLM."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user", "assistant", "tool_result"
    content: Any  # str or list of content blocks

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


class LLMClient(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """
        Send a chat completion request.

        Args:
            system: System prompt.
            messages: Conversation history in provider-native format.
            tools: Tool definitions in provider-native format.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with content and/or tool calls.
        """
        ...

    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'anthropic', 'openai')."""
        ...


class AnthropicClient(LLMClient):
    """
    Anthropic Claude LLM client with tool calling support.

    Uses the official anthropic Python SDK.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPTMIX_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY or OPTMIX_API_KEY "
                "environment variable, or pass api_key parameter."
            )

        self.model = model
        self.max_retries = max_retries

        try:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=self.api_key,
                max_retries=max_retries,
            )
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required. Install with: pip install anthropic"
            )

    def provider_name(self) -> str:
        return "anthropic"

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """
        Send a chat request to Anthropic's Messages API.

        Args:
            system: System prompt text.
            messages: List of message dicts with 'role' and 'content'.
            tools: Tool definitions in Anthropic format.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with parsed content and tool calls.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = tools

        try:
            response = self._client.messages.create(**kwargs)
        except Exception as e:
            logger.error("Anthropic API call failed: %s", e)
            raise

        return self._parse_response(response)

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse an Anthropic API response into our standardized format."""
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        input=block.input,
                    )
                )

        return LLMResponse(
            content="\n".join(content_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            raw_response=response,
        )


class OpenAIClient(LLMClient):
    """
    OpenAI LLM client with tool calling support.

    Translates Anthropic-format tools/messages (used internally by OptMix)
    to OpenAI format on the fly, keeping the executor layer unchanged.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPTMIX_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY or OPTMIX_API_KEY "
                "environment variable, or pass api_key parameter."
            )

        self.model = model
        self.max_retries = max_retries

        try:
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                max_retries=max_retries,
            )
        except ImportError:
            raise ImportError(
                "The 'openai' package is required. Install with: pip install openai"
            )

    def provider_name(self) -> str:
        return "openai"

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """
        Send a chat request to OpenAI's Chat Completions API.

        Translates Anthropic-format tools and messages to OpenAI format,
        then converts the response back to the shared LLMResponse format.
        """
        # Build OpenAI messages: system prompt + translated conversation
        oai_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        oai_messages.extend(self._translate_messages(messages))

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": oai_messages,
        }

        if tools:
            kwargs["tools"] = [self._translate_tool(t) for t in tools]

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error("OpenAI API call failed: %s", e)
            raise

        return self._parse_response(response)

    # ── Format Translation ──────────────────────────────────────────

    @staticmethod
    def _translate_tool(anthropic_tool: dict[str, Any]) -> dict[str, Any]:
        """Anthropic tool schema → OpenAI function tool format."""
        return {
            "type": "function",
            "function": {
                "name": anthropic_tool["name"],
                "description": anthropic_tool.get("description", ""),
                "parameters": anthropic_tool.get("input_schema", {}),
            },
        }

    @staticmethod
    def _translate_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Translate Anthropic-format messages to OpenAI format.

        Handles three key transformations:
        1. Assistant messages with tool_use blocks → assistant with tool_calls
        2. User messages with tool_result blocks → role:tool messages
        3. Plain text messages pass through unchanged
        """
        result: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            # Plain text message — pass through
            if isinstance(content, str):
                result.append({"role": role, "content": content})
                continue

            # List of content blocks
            if isinstance(content, list):
                # Check if this is a tool_result message
                tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
                if tool_results:
                    for tr in tool_results:
                        result.append({
                            "role": "tool",
                            "tool_call_id": tr.get("tool_use_id", ""),
                            "content": tr.get("content", ""),
                        })
                    continue

                # Check if this is an assistant message with tool_use blocks
                tool_uses = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
                text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]

                if tool_uses:
                    oai_msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": "\n".join(text_parts) if text_parts else None,
                        "tool_calls": [
                            {
                                "id": tu.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": tu.get("name", ""),
                                    "arguments": json.dumps(tu.get("input", {})),
                                },
                            }
                            for tu in tool_uses
                        ],
                    }
                    result.append(oai_msg)
                    continue

                # Plain text blocks — join them
                if text_parts:
                    result.append({"role": role, "content": "\n".join(text_parts)})
                    continue

            # Fallback
            result.append({"role": role, "content": str(content) if content else ""})

        return result

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse an OpenAI ChatCompletion response into our standardized format."""
        choice = response.choices[0]
        message = choice.message

        content = message.content or ""
        tool_calls: list[ToolCall] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    input_data = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    input_data = {}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        input=input_data,
                    )
                )

        # Map OpenAI finish_reason to Anthropic-style stop_reason
        finish_reason = choice.finish_reason or "stop"
        stop_reason = "tool_use" if finish_reason == "tool_calls" else "end_turn"

        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            raw_response=response,
        )


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing without API calls.

    Can be configured with scripted responses and tool call sequences.
    """

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self._responses = responses or []
        self._call_index = 0
        self.call_history: list[dict[str, Any]] = []

    def provider_name(self) -> str:
        return "mock"

    def add_response(self, content: str, tool_calls: list[ToolCall] | None = None) -> None:
        """Add a scripted response."""
        self._responses.append(
            LLMResponse(
                content=content,
                tool_calls=tool_calls or [],
                stop_reason="tool_use" if tool_calls else "end_turn",
            )
        )

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Return the next scripted response."""
        self.call_history.append({
            "system": system,
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

        if self._call_index < len(self._responses):
            response = self._responses[self._call_index]
            self._call_index += 1
            return response

        # Default: return a simple text response
        return LLMResponse(
            content="[MockLLM] No more scripted responses available.",
            stop_reason="end_turn",
        )

    def reset(self) -> None:
        """Reset call index and history."""
        self._call_index = 0
        self.call_history.clear()


def create_llm_client(
    provider: str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
) -> LLMClient:
    """
    Factory function to create an LLM client.

    Args:
        provider: LLM provider name ('anthropic' or 'mock').
        model: Model identifier. Provider-specific defaults used if None.
        api_key: API key. Falls back to environment variables if None.

    Returns:
        Configured LLMClient instance.
    """
    if provider == "anthropic":
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if model:
            kwargs["model"] = model
        return AnthropicClient(**kwargs)
    elif provider == "openai":
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if model:
            kwargs["model"] = model
        return OpenAIClient(**kwargs)
    elif provider == "mock":
        return MockLLMClient()
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'. Supported: anthropic, openai, mock")


def build_tool_result_message(tool_call_id: str, result: Any) -> dict[str, Any]:
    """
    Build an Anthropic-format tool_result message block.

    Args:
        tool_call_id: The ID from the original tool_use block.
        result: The tool execution result (will be JSON-serialized).

    Returns:
        Message dict ready to append to conversation.
    """
    if isinstance(result, str):
        content = result
    else:
        try:
            content = json.dumps(result, default=str, indent=2)
        except (TypeError, ValueError):
            content = str(result)

    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": content,
            }
        ],
    }
