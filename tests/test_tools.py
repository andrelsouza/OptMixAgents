"""Tests for the OptMix tools layer."""

from __future__ import annotations

import pytest


class TestToolRegistry:
    def test_register_and_get(self):
        from optmix.tools.registry import ToolParameter, ToolRegistry, ToolSchema

        registry = ToolRegistry()
        schema = ToolSchema(
            name="dummy",
            description="A test tool.",
            parameters=[ToolParameter(name="x", type="number", description="A number.")],
            returns_description="Result.",
            agent_scope=["tester"],
        )
        registry.register(schema, lambda state, *, x: {"result": x * 2})
        assert "dummy" in registry
        s, f = registry.get("dummy")
        assert s.name == "dummy"

    def test_list_for_agent(self):
        from optmix.tools.registry import ToolRegistry, ToolSchema

        registry = ToolRegistry()
        s1 = ToolSchema(
            name="a",
            description="a",
            parameters=[],
            returns_description="a",
            agent_scope=["analyst"],
        )
        s2 = ToolSchema(
            name="b",
            description="b",
            parameters=[],
            returns_description="b",
            agent_scope=["modeler"],
        )
        registry.register(s1, lambda state: {})
        registry.register(s2, lambda state: {})
        assert len(registry.list_for_agent("analyst")) == 1
        assert len(registry.list_for_agent("modeler")) == 1

    def test_to_anthropic_tools_format(self):
        from optmix.tools.registry import ToolParameter, ToolRegistry, ToolSchema

        registry = ToolRegistry()
        schema = ToolSchema(
            name="my_tool",
            description="My tool.",
            parameters=[
                ToolParameter(name="req", type="string", description="Required.", required=True),
                ToolParameter(
                    name="opt", type="number", description="Optional.", required=False, default=42
                ),
            ],
            returns_description="Result.",
            agent_scope=["tester"],
        )
        registry.register(schema, lambda state, **kw: {})
        tools = registry.to_anthropic_tools("tester")
        assert len(tools) == 1
        assert tools[0]["name"] == "my_tool"
        assert tools[0]["input_schema"]["type"] == "object"
        assert "req" in tools[0]["input_schema"]["properties"]
        assert tools[0]["input_schema"]["required"] == ["req"]

    def test_execute_catches_errors(self):
        from optmix.tools.registry import ToolRegistry, ToolSchema

        registry = ToolRegistry()
        schema = ToolSchema(
            name="boom",
            description="boom",
            parameters=[],
            returns_description="boom",
            agent_scope=["t"],
        )

        def boom(state):
            raise RuntimeError("kaboom")

        registry.register(schema, boom)
        result = registry.execute("boom", {}, state={})
        assert result["status"] == "error"
        assert "kaboom" in result["message"]


class TestDefaultRegistry:
    def test_all_tools_registered(self):
        from optmix.tools import create_default_registry

        registry = create_default_registry()
        expected = [
            "load_csv_data",
            "load_sample_data",
            "validate_data",
            "run_eda",
            "describe_channels",
            "fit_mmm_model",
            "get_channel_contributions",
            "get_saturation_curves",
            "get_model_diagnostics",
            "optimize_budget",
            "run_scenario",
            "get_marginal_roas",
            "generate_markdown_report",
            "generate_chart",
            "create_action_plan",
            "load_industry_benchmarks",
            "load_channel_taxonomy",
            "assess_data_readiness",
        ]
        for name in expected:
            assert name in registry, f"Tool '{name}' not in registry"
        assert len(registry) == len(expected)


class TestDataTools:
    def test_load_sample_ecommerce(self):
        from optmix.tools.data_tools import load_sample_data

        state: dict = {}
        result = load_sample_data(state, dataset_name="ecommerce")
        assert result["status"] == "success"
        assert result["rows"] == 104
        assert "raw_data" in state

    def test_validate_data(self):
        from optmix.tools.data_tools import load_sample_data, validate_data

        state: dict = {}
        load_sample_data(state, dataset_name="ecommerce")
        result = validate_data(state)
        assert result["status"] == "success"
        assert result["passed_count"] >= 4
        assert "validated_data" in state

    def test_run_eda(self):
        from optmix.tools.data_tools import load_sample_data, run_eda, validate_data

        state: dict = {}
        load_sample_data(state, dataset_name="ecommerce")
        validate_data(state)
        result = run_eda(state)
        assert result["status"] == "success"
        assert result["n_channels"] >= 8
        assert "eda_report" in state

    def test_describe_channels(self):
        from optmix.tools.data_tools import describe_channels, load_sample_data

        state: dict = {}
        load_sample_data(state, dataset_name="ecommerce")
        result = describe_channels(state)
        assert result["status"] == "success"
        assert result["n_channels"] >= 8
        assert result["total_marketing_spend"] > 0


class TestMMMTools:
    @pytest.fixture
    def fitted_state(self):
        from optmix.tools.data_tools import load_sample_data, validate_data
        from optmix.tools.mmm_tools import fit_mmm_model

        state: dict = {}
        load_sample_data(state, dataset_name="ecommerce")
        validate_data(state)
        fit_mmm_model(state, target="revenue", date_col="date", controls=["avg_price", "promo"])
        return state

    def test_fit_mmm_model(self):
        from optmix.tools.data_tools import load_sample_data, validate_data
        from optmix.tools.mmm_tools import fit_mmm_model

        state: dict = {}
        load_sample_data(state, dataset_name="ecommerce")
        validate_data(state)
        result = fit_mmm_model(
            state, target="revenue", date_col="date", controls=["avg_price", "promo"]
        )
        assert result["status"] == "success"
        assert result["r_squared"] > 0
        assert result["n_observations"] == 104
        assert len(result["channels"]) >= 8
        assert "fitted_model" in state

    def test_get_channel_contributions(self, fitted_state):
        from optmix.tools.mmm_tools import get_channel_contributions

        result = get_channel_contributions(fitted_state)
        assert result["status"] == "success"
        assert result["n_periods"] == 104

    def test_get_saturation_curves(self, fitted_state):
        from optmix.tools.mmm_tools import get_saturation_curves

        result = get_saturation_curves(fitted_state)
        assert result["status"] == "success"
        assert len(result["curves"]) >= 8

    def test_get_model_diagnostics(self, fitted_state):
        from optmix.tools.mmm_tools import get_model_diagnostics

        result = get_model_diagnostics(fitted_state)
        assert result["status"] == "success"
        assert result["fit_quality"] in ("excellent", "good", "acceptable", "poor")


class TestOptimizationTools:
    @pytest.fixture
    def fitted_state(self):
        from optmix.tools.data_tools import load_sample_data, validate_data
        from optmix.tools.mmm_tools import fit_mmm_model

        state: dict = {}
        load_sample_data(state, dataset_name="ecommerce")
        validate_data(state)
        fit_mmm_model(state, target="revenue", date_col="date", controls=["avg_price", "promo"])
        return state

    def test_optimize_budget(self, fitted_state):
        from optmix.tools.optimization_tools import optimize_budget

        result = optimize_budget(fitted_state, total_budget=500_000)
        assert result["status"] == "success"
        assert "allocation" in result
        total_allocated = sum(result["allocation"].values())
        assert abs(total_allocated - 500_000) < 100

    def test_run_scenario(self, fitted_state):
        from optmix.tools.optimization_tools import optimize_budget, run_scenario

        optimize_budget(fitted_state, total_budget=500_000)
        result = run_scenario(fitted_state, changes={"google_search": -0.20, "meta_ads": 0.15})
        assert result["status"] == "success"
        assert "scenario_allocation" in result

    def test_get_marginal_roas(self, fitted_state):
        from optmix.tools.optimization_tools import get_marginal_roas

        result = get_marginal_roas(fitted_state)
        assert result["status"] == "success"
        assert len(result["marginal_roas"]) >= 8


class TestStrategyTools:
    def test_load_benchmarks(self):
        from optmix.tools.strategy_tools import load_industry_benchmarks

        result = load_industry_benchmarks({}, industry="ecommerce")
        assert result["status"] == "success"
        assert "roas_ranges" in result

    def test_load_taxonomy(self):
        from optmix.tools.strategy_tools import load_channel_taxonomy

        result = load_channel_taxonomy({})
        assert result["status"] == "success"

    def test_assess_data_readiness(self):
        from optmix.tools.data_tools import load_sample_data
        from optmix.tools.strategy_tools import assess_data_readiness

        state: dict = {}
        load_sample_data(state, dataset_name="ecommerce")
        result = assess_data_readiness(state)
        assert result["status"] == "success"
        assert result["readiness"] == "ready"

    def test_assess_data_readiness_with_invalid_date(self):
        """Test that assess_data_readiness handles invalid date formats gracefully."""
        import pandas as pd
        from optmix.tools.strategy_tools import assess_data_readiness

        # Create a dataframe with an invalid date column
        state: dict = {}
        state["raw_data"] = pd.DataFrame({
            "date": ["not-a-date", "another-bad-date", "2023-13-45"],  # Invalid dates
            "revenue": [100, 200, 300],
            "spend_channel_1": [10, 20, 30],
            "spend_channel_2": [15, 25, 35],
        })

        # This should not raise an UnboundLocalError
        result = assess_data_readiness(state)
        assert result["status"] == "success"
        # Should fail the date range check but not crash
        date_range_check = next(c for c in result["checklist"] if c["criterion"] == "date_range_coverage")
        assert date_range_check["passed"] == False
        assert "Insufficient date range" in date_range_check["detail"]
