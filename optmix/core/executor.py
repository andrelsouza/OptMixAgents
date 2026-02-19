"""
Agent Executor — runs an agent with its tools via LLM.

This is the core execution loop: given an agent definition, its tools,
shared state, and an LLM client, it builds the system prompt, calls the LLM,
handles tool invocations, and returns the final response.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from optmix.agents.schema import AgentDefinition
from optmix.core.llm import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

# Maximum number of tool-calling round trips before forcing a final answer
MAX_TOOL_ROUNDS = 15


@dataclass
class AgentResponse:
    """Result from an agent execution."""

    agent_name: str
    content: str
    tool_calls_made: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0
    usage: dict[str, int] = field(default_factory=dict)


class AgentExecutor:
    """
    Executes a single agent with its tools via LLM tool-calling loop.

    The executor:
    1. Builds a system prompt from the agent's YAML definition
    2. Injects relevant shared state context
    3. Registers the agent's tools for LLM function calling
    4. Runs the tool-calling loop until the LLM produces a final answer
    5. Returns the response and updates shared state
    """

    def __init__(
        self,
        agent: AgentDefinition,
        llm_client: LLMClient,
        tool_registry: Any,  # ToolRegistry from optmix.tools.registry
        state: Any,  # SharedState from optmix.core.state
        max_rounds: int = MAX_TOOL_ROUNDS,
    ) -> None:
        self.agent = agent
        self.llm = llm_client
        self.tool_registry = tool_registry
        self.state = state
        self.max_rounds = max_rounds

    def run(
        self,
        message: str,
        conversation_history: list[dict[str, Any]] | None = None,
        extra_context: str = "",
    ) -> AgentResponse:
        """
        Execute the agent with a user message.

        Args:
            message: The user's message or task description.
            conversation_history: Optional prior messages for context.
            extra_context: Additional context to inject (e.g., workflow step info).

        Returns:
            AgentResponse with the agent's final text and tool call log.
        """
        # 1. Build system prompt
        system_prompt = self._build_system_prompt(extra_context)

        # 2. Get tools for this agent
        tools = self._get_agent_tools()

        # 3. Build initial messages
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": message})

        # 4. Tool-calling loop
        tool_calls_log: list[dict[str, Any]] = []
        total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        rounds = 0

        while rounds < self.max_rounds:
            rounds += 1

            response = self.llm.chat(
                system=system_prompt,
                messages=messages,
                tools=tools if tools else None,
            )

            # Accumulate usage
            for k, v in response.usage.items():
                total_usage[k] = total_usage.get(k, 0) + v

            if not response.has_tool_calls:
                # Agent produced a final answer
                return AgentResponse(
                    agent_name=self.agent.metadata.name,
                    content=response.content,
                    tool_calls_made=tool_calls_log,
                    rounds=rounds,
                    usage=total_usage,
                )

            # Agent wants to call tools — execute them
            # First, add the assistant's response (with tool_use blocks) to messages
            assistant_content = self._build_assistant_content(response)
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute each tool call and collect results
            tool_results: list[dict[str, Any]] = []
            for tool_call in response.tool_calls:
                logger.info(
                    "Agent '%s' calling tool '%s' with args: %s",
                    self.agent.metadata.name,
                    tool_call.name,
                    json.dumps(tool_call.input, default=str)[:200],
                )

                result = self._execute_tool(tool_call.name, tool_call.input)

                tool_calls_log.append(
                    {
                        "tool": tool_call.name,
                        "input": tool_call.input,
                        "result_summary": result.get("summary", str(result)[:200]),
                    }
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": self._serialize_tool_result(result),
                    }
                )

            # Add all tool results as a single user message
            messages.append({"role": "user", "content": tool_results})

        # Hit max rounds — force a response
        logger.warning(
            "Agent '%s' hit max tool rounds (%d). Forcing final answer.",
            self.agent.metadata.name,
            self.max_rounds,
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "You've used the maximum number of tool calls. "
                    "Please provide your final answer now based on the information gathered so far."
                ),
            }
        )

        response = self.llm.chat(system=system_prompt, messages=messages, tools=None)
        for k, v in response.usage.items():
            total_usage[k] = total_usage.get(k, 0) + v

        return AgentResponse(
            agent_name=self.agent.metadata.name,
            content=response.content,
            tool_calls_made=tool_calls_log,
            rounds=rounds + 1,
            usage=total_usage,
        )

    def _build_system_prompt(self, extra_context: str = "") -> str:
        """Build the full system prompt for this agent."""
        parts = [
            self.agent.system_prompt,
            "",
            "## Current State Context",
            "",
        ]

        # Inject relevant state information
        if hasattr(self.state, "get_context_for_agent"):
            state_context = self.state.get_context_for_agent(self.agent.metadata.name)
            if state_context:
                parts.append(state_context)
            else:
                parts.append("No shared state available yet. Use your tools to gather data.")
        else:
            parts.append("State context unavailable.")

        parts.append("")

        # Add tool usage instructions
        parts.extend(
            [
                "## Tool Usage Instructions",
                "",
                "You have tools available to perform analysis. Use them to answer the user's question.",
                "Always call the relevant tools to get real data — never make up numbers or results.",
                "After using tools, synthesize the results into a clear, actionable response.",
                "Speak as your persona would — use your communication style and domain expertise.",
                "",
            ]
        )

        if extra_context:
            parts.extend(
                [
                    "## Additional Context",
                    "",
                    extra_context,
                    "",
                ]
            )

        return "\n".join(parts)

    def _get_agent_tools(self) -> list[dict[str, Any]]:
        """Get tool definitions for this agent in Anthropic format."""
        if not hasattr(self.tool_registry, "to_anthropic_tools"):
            return []

        return self.tool_registry.to_anthropic_tools(self.agent.metadata.name)

    def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return its result."""
        try:
            result = self.tool_registry.execute(tool_name, arguments, self.state)
            return result
        except Exception as e:
            logger.error("Tool '%s' failed: %s", tool_name, e)
            return {
                "status": "error",
                "message": f"Tool '{tool_name}' failed: {str(e)}",
                "summary": f"Error executing {tool_name}: {str(e)}",
            }

    def _build_assistant_content(self, response: LLMResponse) -> list[dict[str, Any]]:
        """Build Anthropic-format assistant content blocks from response."""
        blocks: list[dict[str, Any]] = []

        if response.content:
            blocks.append({"type": "text", "text": response.content})

        for tc in response.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                }
            )

        return blocks

    def _serialize_tool_result(self, result: Any) -> str:
        """Serialize a tool result for the LLM."""
        if isinstance(result, str):
            return result

        try:
            return json.dumps(result, default=str, indent=2)
        except (TypeError, ValueError):
            return str(result)
