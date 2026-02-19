"""Tests for OrchestratorRouter — keyword and LLM-based routing."""

import json

import pytest

from optmix.core.orchestrator import OrchestratorRouter, RoutingDecision


@pytest.fixture()
def agents():
    """Load all agent definitions."""
    from optmix.agents.schema import AgentLoader

    loader = AgentLoader()
    return loader.load_all()


@pytest.fixture()
def keyword_router(agents):
    """Router with no LLM — keyword-only routing."""
    return OrchestratorRouter(agents=agents, llm_client=None)


class TestKeywordRouting:
    """Test keyword-based routing."""

    def test_model_keywords_route_to_modeler(self, keyword_router):
        decision = keyword_router.route("Fit a Bayesian MMM model on my data")
        assert decision.agent_name == "modeler"
        assert decision.method == "keyword"

    def test_budget_keywords_route_to_optimizer(self, keyword_router):
        decision = keyword_router.route("Optimize my budget allocation across channels")
        assert decision.agent_name == "optimizer"
        assert decision.method == "keyword"

    def test_data_keywords_route_to_analyst(self, keyword_router):
        decision = keyword_router.route("Check data quality and validate my CSV")
        assert decision.agent_name == "analyst"
        assert decision.method == "keyword"

    def test_report_keywords_route_to_reporter(self, keyword_router):
        decision = keyword_router.route("Generate an executive report with charts")
        assert decision.agent_name == "reporter"
        assert decision.method == "keyword"

    def test_strategy_keywords_route_to_strategist(self, keyword_router):
        decision = keyword_router.route("Help me define KPIs and business objectives")
        assert decision.agent_name == "strategist"
        assert decision.method == "keyword"

    def test_ambiguous_message_defaults_to_strategist(self, keyword_router):
        decision = keyword_router.route("Hello, can you help me?")
        assert decision.agent_name == "strategist"
        assert decision.method == "default"
        assert decision.confidence == "low"

    def test_multiple_keywords_increase_confidence(self, keyword_router):
        decision = keyword_router.route("Optimize budget allocation and run a scenario what-if")
        assert decision.agent_name == "optimizer"
        assert decision.confidence == "high"

    def test_returns_routing_decision(self, keyword_router):
        decision = keyword_router.route("Fit a model")
        assert isinstance(decision, RoutingDecision)
        assert decision.agent_name in ["modeler", "strategist", "analyst", "optimizer", "reporter"]
        assert decision.confidence in ["high", "medium", "low"]
        assert decision.reasoning


class TestLLMRouting:
    """Test LLM-based routing with mock client."""

    def test_llm_routing_used_when_available(self, agents):
        from optmix.core.llm import MockLLMClient

        mock = MockLLMClient()
        mock.add_response(
            json.dumps(
                {
                    "agent": "modeler",
                    "confidence": "high",
                    "reasoning": "User wants to fit a model",
                }
            )
        )

        router = OrchestratorRouter(agents=agents, llm_client=mock)
        decision = router.route("Fit a model on my data")

        assert decision.agent_name == "modeler"
        assert decision.method == "llm"
        assert decision.confidence == "high"

    def test_llm_routing_falls_back_on_bad_json(self, agents):
        from optmix.core.llm import MockLLMClient

        mock = MockLLMClient()
        mock.add_response("This is not valid JSON at all")

        router = OrchestratorRouter(agents=agents, llm_client=mock)
        # Should fall back to keyword routing
        decision = router.route("Fit a Bayesian model")
        assert decision.method in ("keyword", "default")

    def test_llm_routing_falls_back_on_unknown_agent(self, agents):
        from optmix.core.llm import MockLLMClient

        mock = MockLLMClient()
        mock.add_response(
            json.dumps(
                {
                    "agent": "unknown_agent_xyz",
                    "confidence": "high",
                    "reasoning": "Test",
                }
            )
        )

        router = OrchestratorRouter(agents=agents, llm_client=mock)
        decision = router.route("Fit a Bayesian model")
        # Should fall back since agent doesn't exist
        assert decision.agent_name != "unknown_agent_xyz"

    def test_llm_routing_handles_markdown_code_block(self, agents):
        from optmix.core.llm import MockLLMClient

        mock = MockLLMClient()
        mock.add_response(
            '```json\n{"agent": "optimizer", "confidence": "high", "reasoning": "Budget question"}\n```'
        )

        router = OrchestratorRouter(agents=agents, llm_client=mock)
        decision = router.route("How should I allocate my budget?")
        assert decision.agent_name == "optimizer"
        assert decision.method == "llm"

    def test_llm_routing_falls_back_on_exception(self, agents):
        from optmix.core.llm import MockLLMClient

        mock = MockLLMClient()
        # No responses added — will return default mock response which isn't valid JSON

        router = OrchestratorRouter(agents=agents, llm_client=mock)
        decision = router.route("Fit a model on the data")
        # Should fall back gracefully
        assert decision.method in ("keyword", "default")


class TestRoutingWithState:
    """Test that state context doesn't break routing."""

    def test_routing_with_state(self, agents):
        from optmix.core.llm import MockLLMClient
        from optmix.core.state import SharedState

        mock = MockLLMClient()
        mock.add_response(
            json.dumps(
                {
                    "agent": "analyst",
                    "confidence": "high",
                    "reasoning": "Data task with state context",
                }
            )
        )

        state = SharedState()
        state.set("raw_data", "some_data", source_agent="test")

        router = OrchestratorRouter(agents=agents, llm_client=mock)
        decision = router.route("Validate the loaded data", state=state)
        assert decision.agent_name == "analyst"
