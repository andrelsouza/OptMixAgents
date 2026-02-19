"""Tests for inter-agent delegation — agents collaborating via delegate_to_agent tool."""

from unittest.mock import MagicMock, patch

import pytest

from optmix.core.llm import LLMResponse, MockLLMClient, ToolCall


@pytest.fixture()
def mock_team():
    """Create an OptMixTeam with mock LLM."""
    from optmix.core.team import OptMixTeam

    team = OptMixTeam(llm="mock")
    return team


class TestDelegationToolRegistered:
    """Verify the delegation tool is properly registered."""

    def test_delegate_tool_exists(self, mock_team):
        assert "delegate_to_agent" in mock_team._tool_registry

    def test_delegate_tool_available_to_strategist(self, mock_team):
        tools = mock_team._tool_registry.list_for_agent("strategist")
        names = [t.name for t in tools]
        assert "delegate_to_agent" in names

    def test_delegate_tool_available_to_analyst(self, mock_team):
        tools = mock_team._tool_registry.list_for_agent("analyst")
        names = [t.name for t in tools]
        assert "delegate_to_agent" in names

    def test_delegate_tool_available_to_modeler(self, mock_team):
        tools = mock_team._tool_registry.list_for_agent("modeler")
        names = [t.name for t in tools]
        assert "delegate_to_agent" in names

    def test_delegate_tool_not_available_to_orchestrator(self, mock_team):
        tools = mock_team._tool_registry.list_for_agent("orchestrator")
        names = [t.name for t in tools]
        assert "delegate_to_agent" not in names

    def test_delegate_tool_has_agent_enum(self, mock_team):
        schema, _ = mock_team._tool_registry.get("delegate_to_agent")
        agent_param = [p for p in schema.parameters if p.name == "agent_name"][0]
        assert agent_param.enum is not None
        assert "modeler" in agent_param.enum
        assert "strategist" in agent_param.enum
        assert "orchestrator" not in agent_param.enum


class TestDelegationExecution:
    """Test actual delegation between agents."""

    def test_delegate_returns_agent_response(self, mock_team):
        """Delegating to an agent returns their response."""
        result = mock_team._tool_registry.execute(
            "delegate_to_agent",
            {"agent_name": "analyst", "task": "Describe the loaded data"},
            mock_team._state,
        )
        assert result["status"] == "success"
        assert result["agent"] == "analyst"
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_delegate_unknown_agent(self, mock_team):
        """Delegating to a nonexistent agent returns error."""
        result = mock_team._tool_registry.execute(
            "delegate_to_agent",
            {"agent_name": "ceo", "task": "Do something"},
            mock_team._state,
        )
        assert result["status"] == "error"
        assert "Unknown agent" in result["message"]

    def test_delegation_depth_limit(self, mock_team):
        """Prevent infinite delegation loops."""
        mock_team._delegation_depth = 3  # Already at max
        result = mock_team._tool_registry.execute(
            "delegate_to_agent",
            {"agent_name": "modeler", "task": "Fit model"},
            mock_team._state,
        )
        assert result["status"] == "error"
        assert "depth" in result["message"].lower()

    def test_delegation_depth_resets_after_call(self, mock_team):
        """Delegation depth counter decrements after each call."""
        assert mock_team._delegation_depth == 0
        mock_team._tool_registry.execute(
            "delegate_to_agent",
            {"agent_name": "analyst", "task": "Quick check"},
            mock_team._state,
        )
        assert mock_team._delegation_depth == 0

    def test_delegate_with_data_in_state(self, mock_team):
        """Delegation works when shared state has data loaded."""
        mock_team.load_data(sample="ecommerce")
        result = mock_team._tool_registry.execute(
            "delegate_to_agent",
            {"agent_name": "analyst", "task": "Analyze the loaded data"},
            mock_team._state,
        )
        assert result["status"] == "success"


class TestDelegationInToolLoop:
    """Test that LLM-driven agents can use delegation in the tool-calling loop."""

    def test_agent_delegates_via_tool_call(self, mock_team):
        """Simulate the full loop: LLM calls delegate_to_agent tool."""
        from optmix.core.executor import AgentExecutor

        # Mock LLM needs 3 responses:
        # 1. Strategist: tool call to delegate to modeler
        # 2. Modeler: text response (consumed during delegation)
        # 3. Strategist: final answer after getting modeler's result
        llm = MockLLMClient()
        llm.add_response(
            content="Let me ask the modeler.",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="delegate_to_agent",
                    input={"agent_name": "modeler", "task": "What MMM approaches do you recommend?"},
                )
            ],
        )
        llm.add_response(
            content="I recommend a Bayesian MMM with adstock and saturation.",
        )
        llm.add_response(
            content="Based on the modeler's input, Bayesian MMM is the way to go.",
        )

        mock_team._llm_client = llm

        executor = AgentExecutor(
            agent=mock_team._agents["strategist"],
            llm_client=llm,
            tool_registry=mock_team._tool_registry,
            state=mock_team._state,
        )

        result = executor.run("Should we use Bayesian or Ridge MMM?")

        # Verify delegation happened
        assert len(result.tool_calls_made) == 1
        assert result.tool_calls_made[0]["tool"] == "delegate_to_agent"
        # Verify the strategist got a final answer (response #3)
        assert "Bayesian" in result.content
