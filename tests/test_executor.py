"""Tests for AgentExecutor — tool-calling loop with MockLLM."""

import pytest

from optmix.core.executor import AgentExecutor, AgentResponse


@pytest.fixture()
def agents():
    """Load all agent definitions."""
    from optmix.agents.schema import AgentLoader

    return AgentLoader().load_all()


@pytest.fixture()
def state():
    """Create a fresh SharedState with sample data loaded."""
    from optmix.core.state import SharedState
    from optmix.data.samples import load_sample

    s = SharedState()
    s.set("raw_data", load_sample("ecommerce"), source_agent="test")
    return s


@pytest.fixture()
def registry():
    """Create the default tool registry."""
    from optmix.tools import create_default_registry

    return create_default_registry()


class TestAgentExecutorBasic:
    """Test basic executor behavior with immediate text responses."""

    def test_simple_text_response(self, agents, state, registry):
        from optmix.core.llm import MockLLMClient

        mock = MockLLMClient()
        mock.add_response("The data has been analyzed and looks good.")

        executor = AgentExecutor(
            agent=agents["analyst"],
            llm_client=mock,
            tool_registry=registry,
            state=state,
        )

        result = executor.run("Analyze the data")
        assert isinstance(result, AgentResponse)
        assert result.agent_name == "analyst"
        assert "analyzed" in result.content.lower()
        assert result.rounds == 1
        assert len(result.tool_calls_made) == 0

    def test_multiple_rounds_with_tools(self, agents, state, registry):
        """Test that the executor handles tool call -> tool result -> final response."""
        from optmix.core.llm import MockLLMClient, ToolCall

        mock = MockLLMClient()

        # Round 1: LLM requests a tool call
        mock.add_response(
            content="Let me validate the data.",
            tool_calls=[
                ToolCall(id="tc_1", name="validate_data", input={}),
            ],
        )
        # Round 2: LLM gives final response after seeing tool result
        mock.add_response("Data validation complete: 6/6 checks passed.")

        executor = AgentExecutor(
            agent=agents["analyst"],
            llm_client=mock,
            tool_registry=registry,
            state=state,
        )

        result = executor.run("Validate my data")
        assert result.rounds == 2
        assert len(result.tool_calls_made) == 1
        assert result.tool_calls_made[0]["tool"] == "validate_data"
        assert "validation" in result.content.lower() or "complete" in result.content.lower()

    def test_tool_execution_stores_results(self, agents, state, registry):
        """Tool execution should update shared state."""
        from optmix.core.llm import MockLLMClient, ToolCall

        mock = MockLLMClient()
        mock.add_response(
            content="Running EDA.",
            tool_calls=[ToolCall(id="tc_1", name="run_eda", input={})],
        )
        mock.add_response("EDA complete.")

        executor = AgentExecutor(
            agent=agents["analyst"],
            llm_client=mock,
            tool_registry=registry,
            state=state,
        )

        executor.run("Run exploratory analysis")
        assert state.has("eda_report")


class TestAgentExecutorToolLoop:
    """Test the tool-calling loop edge cases."""

    def test_max_rounds_forces_final_answer(self, agents, state, registry):
        """If LLM keeps calling tools, executor should force a final answer."""
        from optmix.core.llm import MockLLMClient, ToolCall

        mock = MockLLMClient()

        # Keep requesting tool calls for max_rounds iterations
        for i in range(3):
            mock.add_response(
                content=f"Calling tool round {i}",
                tool_calls=[ToolCall(id=f"tc_{i}", name="validate_data", input={})],
            )
        # After max rounds, executor sends a forced-final-answer user message
        # and calls the LLM again with tools=None. This is the 4th response:
        mock.add_response("OK, here is my final answer based on what I gathered.")

        executor = AgentExecutor(
            agent=agents["analyst"],
            llm_client=mock,
            tool_registry=registry,
            state=state,
            max_rounds=3,
        )

        result = executor.run("Analyze everything")
        assert result.rounds == 4  # 3 tool rounds + 1 forced final
        assert len(result.tool_calls_made) == 3

    def test_unknown_tool_returns_error(self, agents, state, registry):
        """Calling a tool that doesn't exist should return an error dict, not crash."""
        from optmix.core.llm import MockLLMClient, ToolCall

        mock = MockLLMClient()
        mock.add_response(
            content="Let me call a tool.",
            tool_calls=[ToolCall(id="tc_1", name="nonexistent_tool", input={})],
        )
        mock.add_response("The tool failed, but I can still respond.")

        executor = AgentExecutor(
            agent=agents["analyst"],
            llm_client=mock,
            tool_registry=registry,
            state=state,
        )

        result = executor.run("Do something")
        assert result.rounds == 2
        # Should have logged the failed tool call
        assert result.tool_calls_made[0]["tool"] == "nonexistent_tool"


class TestAgentExecutorSystemPrompt:
    """Test system prompt construction."""

    def test_system_prompt_includes_agent_persona(self, agents, state, registry):
        from optmix.core.llm import MockLLMClient

        mock = MockLLMClient()
        mock.add_response("Hello!")

        executor = AgentExecutor(
            agent=agents["analyst"],
            llm_client=mock,
            tool_registry=registry,
            state=state,
        )

        executor.run("Hi")

        # Check the system prompt that was sent to the LLM
        assert len(mock.call_history) == 1
        system = mock.call_history[0]["system"]
        assert agents["analyst"].persona.name in system

    def test_system_prompt_includes_extra_context(self, agents, state, registry):
        from optmix.core.llm import MockLLMClient

        mock = MockLLMClient()
        mock.add_response("Got it.")

        executor = AgentExecutor(
            agent=agents["analyst"],
            llm_client=mock,
            tool_registry=registry,
            state=state,
        )

        executor.run("Analyze", extra_context="This is a quick optimization workflow step.")

        system = mock.call_history[0]["system"]
        assert "quick optimization" in system.lower()

    def test_tools_passed_to_llm(self, agents, state, registry):
        from optmix.core.llm import MockLLMClient

        mock = MockLLMClient()
        mock.add_response("Done.")

        executor = AgentExecutor(
            agent=agents["analyst"],
            llm_client=mock,
            tool_registry=registry,
            state=state,
        )

        executor.run("Do analysis")

        tools = mock.call_history[0]["tools"]
        assert tools is not None
        tool_names = [t["name"] for t in tools]
        assert "validate_data" in tool_names
        assert "run_eda" in tool_names


class TestAgentExecutorConversation:
    """Test conversation history handling."""

    def test_conversation_history_passed(self, agents, state, registry):
        from optmix.core.llm import MockLLMClient

        mock = MockLLMClient()
        mock.add_response("Based on our earlier discussion, here is the analysis.")

        executor = AgentExecutor(
            agent=agents["analyst"],
            llm_client=mock,
            tool_registry=registry,
            state=state,
        )

        history = [
            {"role": "user", "content": "I have marketing data"},
            {"role": "assistant", "content": "Great, let's analyze it"},
        ]

        executor.run("Now validate it", conversation_history=history)

        messages = mock.call_history[0]["messages"]
        # History + new message
        assert len(messages) == 3
        assert messages[0]["content"] == "I have marketing data"
        assert messages[2]["content"] == "Now validate it"

    def test_usage_tracking(self, agents, state, registry):
        from optmix.core.llm import MockLLMClient

        mock = MockLLMClient()
        mock.add_response("Done.")

        executor = AgentExecutor(
            agent=agents["analyst"],
            llm_client=mock,
            tool_registry=registry,
            state=state,
        )

        result = executor.run("Validate data")
        assert "input_tokens" in result.usage or "output_tokens" in result.usage
