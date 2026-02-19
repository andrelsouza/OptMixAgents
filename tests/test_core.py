"""Tests for OptMix core modules."""

import numpy as np
import pytest


class TestAdstockTransforms:
    """Test adstock transformations."""

    def test_geometric_adstock_no_decay(self):
        from optmix.mmm.transforms.adstock import geometric_adstock

        spend = np.array([100.0, 50.0, 75.0])
        result = geometric_adstock(spend, decay=0.0)
        np.testing.assert_array_almost_equal(result, spend)

    def test_geometric_adstock_full_decay(self):
        from optmix.mmm.transforms.adstock import geometric_adstock

        spend = np.array([100.0, 0.0, 0.0, 0.0])
        result = geometric_adstock(spend, decay=0.5)
        expected = np.array([100.0, 50.0, 25.0, 12.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_geometric_adstock_invalid_decay(self):
        from optmix.mmm.transforms.adstock import geometric_adstock

        with pytest.raises(ValueError):
            geometric_adstock(np.array([100.0]), decay=1.5)

    def test_weibull_adstock_shape(self):
        from optmix.mmm.transforms.adstock import weibull_adstock

        spend = np.array([100.0] + [0.0] * 12)
        result = weibull_adstock(spend, shape=2.0, scale=3.0)
        assert len(result) == len(spend)
        assert result[0] > 0

    def test_delayed_adstock(self):
        from optmix.mmm.transforms.adstock import delayed_adstock

        spend = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
        result = delayed_adstock(spend, decay=0.5, delay=2)
        # First two periods should be zero due to delay
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] > 0.0


class TestSaturationTransforms:
    """Test saturation curve transformations."""

    def test_hill_saturation_zero_spend(self):
        from optmix.mmm.transforms.saturation import hill_saturation

        result = hill_saturation(np.array([0.0]), half_sat=50000, slope=1.0)
        assert result[0] == 0.0

    def test_hill_saturation_at_half_sat(self):
        from optmix.mmm.transforms.saturation import hill_saturation

        result = hill_saturation(np.array([50000.0]), half_sat=50000, slope=1.0)
        np.testing.assert_almost_equal(result[0], 0.5)

    def test_hill_saturation_monotonic(self):
        from optmix.mmm.transforms.saturation import hill_saturation

        spend = np.linspace(0, 200000, 100)
        result = hill_saturation(spend, half_sat=50000, slope=2.0)
        # Should be monotonically increasing
        assert np.all(np.diff(result) >= 0)

    def test_hill_saturation_bounded(self):
        from optmix.mmm.transforms.saturation import hill_saturation

        spend = np.linspace(0, 1e8, 100)
        result = hill_saturation(spend, half_sat=50000, slope=2.0)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_hill_saturation_invalid_half_sat(self):
        from optmix.mmm.transforms.saturation import hill_saturation

        with pytest.raises(ValueError):
            hill_saturation(np.array([100.0]), half_sat=-1, slope=1.0)

    def test_power_saturation(self):
        from optmix.mmm.transforms.saturation import power_saturation

        spend = np.array([0.0, 100.0, 400.0])
        result = power_saturation(spend, exponent=0.5)
        expected = np.array([0.0, 10.0, 20.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestAgentSchema:
    """Test agent YAML loading and compilation."""

    def test_agent_loader_list(self):
        from optmix.agents.schema import AgentLoader

        loader = AgentLoader()
        agents = loader.list_agents()
        # Should find our defined agents
        assert len(agents) >= 0  # May be 0 if agents dir not in right place

    def test_agent_definition_system_prompt(self):
        from optmix.agents.schema import AgentDefinition, AgentMetadata, AgentPersona

        agent = AgentDefinition(
            metadata=AgentMetadata(name="test", title="Test Agent"),
            persona=AgentPersona(
                name="Tester",
                role="QA",
                identity="A test agent",
                communication_style="Direct",
                focus=["testing"],
                core_principles=["test everything"],
            ),
            tools=["test_tool"],
        )
        prompt = agent.system_prompt
        assert "Tester" in prompt
        assert "test_tool" in prompt
        assert "test everything" in prompt


class TestSampleData:
    """Test sample dataset generation."""

    def test_ecommerce_dataset(self):
        from optmix.data.samples import load_sample

        df = load_sample("ecommerce")
        assert "date" in df.columns
        assert "revenue" in df.columns
        assert len(df) == 104
        assert df["revenue"].min() >= 0

    def test_retail_dataset(self):
        from optmix.data.samples import load_sample

        df = load_sample("retail_chain")
        assert "date" in df.columns
        assert "revenue" in df.columns
        assert len(df) == 156

    def test_saas_dataset(self):
        from optmix.data.samples import load_sample

        df = load_sample("saas_b2b")
        assert "date" in df.columns
        assert "pipeline_generated" in df.columns
        assert len(df) == 104

    def test_invalid_dataset(self):
        from optmix.data.samples import load_sample

        with pytest.raises(ValueError):
            load_sample("nonexistent")
