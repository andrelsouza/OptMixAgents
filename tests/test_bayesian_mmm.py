"""Tests for BayesianMMM — all marked slow (MCMC takes minutes)."""

import numpy as np
import pandas as pd
import pytest

# All tests in this file require MCMC sampling (~1-3 min each)
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def sample_data() -> pd.DataFrame:
    """Generate synthetic marketing data for testing."""
    rng = np.random.default_rng(42)
    n = 104  # ~2 years of weekly data

    dates = pd.date_range("2022-01-01", periods=n, freq="W")
    google = rng.uniform(10_000, 80_000, n)
    meta = rng.uniform(5_000, 50_000, n)
    tv = rng.uniform(20_000, 100_000, n)

    # Simulated revenue with known channel effects + noise
    revenue = 50_000 + 0.8 * google + 0.5 * meta + 0.3 * tv + rng.normal(0, 5_000, n)

    return pd.DataFrame(
        {
            "date": dates,
            "google_ads": google,
            "meta_ads": meta,
            "tv_spend": tv,
            "revenue": revenue,
        }
    )


@pytest.fixture(scope="module")
def fitted_result(sample_data: pd.DataFrame):
    """Fit a BayesianMMM once, shared across tests in this module."""
    from optmix.mmm.models.bayesian_mmm import BayesianMMM

    model = BayesianMMM(
        chains=2,
        tune=200,
        draws=200,
        adstock_max_lag=4,
        yearly_seasonality=2,
    )
    result = model.fit(
        data=sample_data,
        target="revenue",
        date_col="date",
    )
    return model, result


class TestBayesianMMMFit:
    """Test that BayesianMMM fit produces valid ModelResult."""

    def test_fit_returns_model_result(self, fitted_result):
        _, result = fitted_result
        assert result.model_type == "BayesianMMM"
        assert result.r_squared is not None
        assert result.mape is not None
        assert result.rmse is not None
        assert result.n_observations == 104
        assert len(result.channels) == 3
        assert set(result.channels) == {"google_ads", "meta_ads", "tv_spend"}

    def test_fit_metrics_reasonable(self, fitted_result):
        _, result = fitted_result
        # With known linear data + noise, R² should be decent
        assert result.r_squared > 0.3, f"R² too low: {result.r_squared}"
        assert result.mape < 50, f"MAPE too high: {result.mape}"

    def test_channel_roas_populated(self, fitted_result):
        _, result = fitted_result
        assert len(result.channel_roas) == 3
        for ch in result.channels:
            assert ch in result.channel_roas

    def test_channel_share_sums_to_one(self, fitted_result):
        _, result = fitted_result
        total = sum(result.channel_share.values())
        assert abs(total - 1.0) < 0.05, f"Share sums to {total}, expected ~1.0"


class TestBayesianPosterior:
    """Test Bayesian-specific outputs (posteriors, uncertainty)."""

    def test_posterior_samples_populated(self, fitted_result):
        _, result = fitted_result
        assert result.posterior_samples is not None
        assert "predictions" in result.posterior_samples

    def test_credible_intervals_populated(self, fitted_result):
        _, result = fitted_result
        assert result.credible_intervals is not None
        assert "prediction" in result.credible_intervals
        low, high = result.credible_intervals["prediction"]
        assert low < high

    def test_learned_adstock_params(self, fitted_result):
        _, result = fitted_result
        assert len(result.adstock_params) == 3
        for ch, params in result.adstock_params.items():
            assert "decay" in params
            assert 0.0 <= params["decay"] <= 1.0, f"{ch} decay out of range: {params['decay']}"

    def test_learned_saturation_params(self, fitted_result):
        _, result = fitted_result
        assert len(result.saturation_params) == 3
        for ch, params in result.saturation_params.items():
            assert "lam" in params
            assert params["lam"] > 0, f"{ch} lam should be positive: {params['lam']}"

    def test_learned_params_have_hdi(self, fitted_result):
        _, result = fitted_result
        for _ch, params in result.adstock_params.items():
            if "decay_hdi_low" in params:
                assert params["decay_hdi_low"] <= params["decay"]
                assert params["decay_hdi_high"] >= params["decay"]


class TestBayesianCurves:
    """Test saturation curves from learned parameters."""

    def test_saturation_curves_exist(self, fitted_result):
        model, _ = fitted_result
        curves = model.get_saturation_curves()
        assert len(curves) == 3
        for ch in ["google_ads", "meta_ads", "tv_spend"]:
            assert ch in curves

    def test_saturation_curves_shape(self, fitted_result):
        model, _ = fitted_result
        curves = model.get_saturation_curves()
        for _ch, df in curves.items():
            assert "spend" in df.columns
            assert "response" in df.columns
            assert len(df) == 200

    def test_saturation_curves_monotonic(self, fitted_result):
        model, _ = fitted_result
        curves = model.get_saturation_curves()
        for ch, df in curves.items():
            responses = df["response"].values
            diffs = np.diff(responses)
            assert np.all(diffs >= -1e-10), f"{ch} curve is not monotonically increasing"

    def test_saturation_curves_single_channel(self, fitted_result):
        model, _ = fitted_result
        curves = model.get_saturation_curves(channel="google_ads")
        assert len(curves) == 1
        assert "google_ads" in curves


class TestBayesianPredict:
    """Test prediction functionality."""

    def test_predict_returns_array(self, fitted_result, sample_data):
        model, _ = fitted_result
        preds = model.predict(sample_data)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(sample_data)

    def test_contributions_dataframe(self, fitted_result):
        model, _ = fitted_result
        contrib = model.get_channel_contributions()
        assert isinstance(contrib, pd.DataFrame)
        assert "base" in contrib.columns
        assert "total" in contrib.columns


class TestBayesianOptimizerCompat:
    """Test that BayesianMMM works with BudgetOptimizer."""

    def test_optimizer_runs(self, fitted_result):
        model, _ = fitted_result
        from optmix.mmm.optimizer.budget_optimizer import BudgetOptimizer

        optimizer = BudgetOptimizer(model)
        result = optimizer.optimize(total_budget=200_000)
        assert result.allocation is not None
        assert len(result.allocation) == 3
        assert abs(sum(result.allocation.values()) - 200_000) < 1.0

    def test_optimizer_scenario(self, fitted_result):
        model, _ = fitted_result
        from optmix.mmm.optimizer.budget_optimizer import BudgetOptimizer

        optimizer = BudgetOptimizer(model)
        base = {"google_ads": 80_000, "meta_ads": 50_000, "tv_spend": 70_000}
        result = optimizer.run_scenario(base, {"google_ads": 0.10, "meta_ads": -0.10})
        assert result.expected_lift is not None


class TestBayesianPosteriorSummary:
    """Test BayesianMMM-specific methods."""

    def test_posterior_summary(self, fitted_result):
        model, _ = fitted_result
        summary = model.get_posterior_summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) > 0
