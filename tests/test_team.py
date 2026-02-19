"""Tests for OptMixTeam — end-to-end integration with mock LLM."""

import pytest


@pytest.fixture()
def team():
    """Create an OptMixTeam with mock LLM."""
    from optmix.core.team import OptMixTeam

    # Use mock provider to avoid needing API key
    t = OptMixTeam(llm="mock")
    return t


class TestTeamInit:
    """Test team initialization."""

    def test_agents_loaded(self, team):
        assert len(team.agents) >= 5
        assert "analyst" in team.agents
        assert "modeler" in team.agents
        assert "optimizer" in team.agents
        assert "reporter" in team.agents
        assert "strategist" in team.agents

    def test_state_initialized(self, team):
        assert team.state is not None

    def test_repr(self, team):
        r = repr(team)
        assert "OptMixTeam" in r
        assert "mock" in r


class TestTeamDataLoading:
    """Test data loading via the team."""

    def test_load_sample_data(self, team):
        result = team.load_data(sample="ecommerce")
        assert "ecommerce" in result
        assert team.state.has("raw_data")

    def test_load_sample_retail(self, team):
        result = team.load_data(sample="retail_chain")
        assert "retail_chain" in result

    def test_load_sample_saas(self, team):
        result = team.load_data(sample="saas_b2b")
        assert "saas_b2b" in result

    def test_load_no_args_raises(self, team):
        with pytest.raises(ValueError, match="Provide either"):
            team.load_data()


class TestTeamChat:
    """Test chat routing with mock LLM."""

    def test_chat_returns_string(self, team):
        response = team.chat("What should I do with my marketing data?")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_chat_history_tracked(self, team):
        team.chat("Hello")
        assert len(team._history) == 2  # user + assistant

    def test_chat_routes_to_keyword_match(self, team):
        # "model" keyword should route to modeler agent
        team.chat("Fit a Bayesian MMM model")
        last_entry = team._history[-1]
        assert last_entry["agent"] == "modeler"

    def test_chat_routes_optimizer_keywords(self, team):
        team.chat("Optimize my budget allocation")
        last_entry = team._history[-1]
        assert last_entry["agent"] == "optimizer"


class TestTeamInvoke:
    """Test direct agent invocation."""

    def test_invoke_analyst(self, team):
        response = team.invoke("analyst", "Validate the data")
        assert isinstance(response, str)

    def test_invoke_modeler(self, team):
        response = team.invoke("modeler", "Fit a ridge model")
        assert isinstance(response, str)

    def test_invoke_unknown_agent_raises(self, team):
        with pytest.raises(ValueError, match="Unknown agent"):
            team.invoke("nonexistent_agent", "Do something")


class TestTeamGetAgent:
    """Test agent retrieval."""

    def test_get_valid_agent(self, team):
        agent = team.get_agent("analyst")
        assert agent.persona.name  # has a name
        assert len(agent.tools) > 0

    def test_get_invalid_agent_raises(self, team):
        with pytest.raises(ValueError, match="Unknown agent"):
            team.get_agent("fake_agent")


class TestTeamOfflineMode:
    """Test graceful fallback when LLM is unavailable."""

    def test_offline_invoke_returns_message(self):
        from optmix.core.team import OptMixTeam

        # Create team without valid LLM config
        team = OptMixTeam(llm="anthropic", api_key=None)
        # This should fall back to offline mode since no API key
        response = team.invoke("analyst", "Validate data")
        assert isinstance(response, str)
        # Should mention the agent or tools
        assert (
            "analyst" in response.lower() or "api" in response.lower() or "key" in response.lower()
        )


class TestTeamRun:
    """Test workflow execution through the team."""

    def test_run_with_sample_data(self, team):
        """Running a workflow with mock LLM should work end-to-end."""
        result = team.run(
            sample_dataset="ecommerce",
            objective="Test workflow",
            workflow="quick_optimization",
        )
        # With mock LLM, workflow should complete
        assert result.status == "completed"
        assert result.total_steps == 5

    def test_run_stores_objective_in_state(self, team):
        team.run(
            sample_dataset="ecommerce",
            objective="Maximize revenue from digital channels",
            workflow="quick_optimization",
        )
        assert team.state.get("objective") == "Maximize revenue from digital channels"
