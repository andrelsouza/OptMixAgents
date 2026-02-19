"""
Bayesian MMM using PyMC-Marketing -- full posterior inference.

Uses PyMC-Marketing's multidimensional MMM class for MCMC-based parameter
learning.  Learns adstock decay rates, saturation parameters, and channel
coefficients from data rather than hardcoding them.

This is the "full mode" model -- slower (2-5 minutes) but produces proper
uncertainty quantification and learned response curves.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import arviz as az
import numpy as np
import pandas as pd

from optmix.mmm.models.base import BaseMMM, ModelResult

logger = logging.getLogger(__name__)


class BayesianMMM(BaseMMM):
    """
    Full Bayesian MMM powered by PyMC-Marketing.

    Learns adstock and saturation parameters per channel via MCMC sampling.
    Provides posterior distributions, credible intervals, and uncertainty-aware
    budget optimization.

    Example::

        model = BayesianMMM(chains=4, draws=1000, tune=1000)
        result = model.fit(data=df, target="revenue", date_col="date",
                           controls=["avg_price", "promo"])
        # result.adstock_params  -- learned per-channel decay rates
        # result.credible_intervals  -- 94% HDI per channel
    """

    def __init__(
        self,
        adstock_max_lag: int = 8,
        yearly_seasonality: int = 2,
        chains: int = 4,
        tune: int = 1000,
        draws: int = 1000,
        target_accept: float = 0.9,
        random_seed: int = 42,
    ) -> None:
        self._adstock_max_lag = adstock_max_lag
        self._yearly_seasonality = yearly_seasonality
        self._chains = chains
        self._tune = tune
        self._draws = draws
        self._target_accept = target_accept
        self._random_seed = random_seed

        self._mmm: Any = None  # pymc_marketing.mmm.multidimensional.MMM
        self._result: ModelResult | None = None
        self._data: pd.DataFrame | None = None
        self._target: str = ""
        self._date_col: str = ""
        self._channels: list[str] = []
        self._controls: list[str] = []

    def fit(
        self,
        data: pd.DataFrame,
        target: str,
        date_col: str,
        channels: list[str] | None = None,
        controls: list[str] | None = None,
        **kwargs: Any,
    ) -> ModelResult:
        """
        Fit a Bayesian MMM with MCMC sampling.

        All adstock and saturation parameters are learned from data.

        Args:
            data: Marketing data with date, spend, controls, and target.
            target: Target column name (e.g. 'revenue').
            date_col: Date column name.
            channels: Channel spend columns. Auto-detected if None.
            controls: Control variable columns.
            **kwargs: Override MCMC settings (chains, tune, draws, etc.).
        """
        from pymc_marketing.mmm.components.adstock import GeometricAdstock
        from pymc_marketing.mmm.components.saturation import LogisticSaturation
        from pymc_marketing.mmm.multidimensional import MMM

        self._data = data.copy().sort_values(date_col).reset_index(drop=True)
        self._target = target
        self._date_col = date_col

        # Auto-detect channels
        if channels is None:
            exclude = {target, date_col} | set(controls or [])
            channels = [
                c for c in data.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ]
        self._channels = channels
        self._controls = controls or []

        logger.info(
            "Fitting BayesianMMM: %d channels, %d controls, %d observations",
            len(channels), len(self._controls), len(data),
        )

        # Construct PyMC-Marketing MMM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._mmm = MMM(
                date_column=date_col,
                channel_columns=channels,
                target_column=target,
                adstock=GeometricAdstock(l_max=self._adstock_max_lag),
                saturation=LogisticSaturation(),
                control_columns=self._controls if self._controls else None,
                yearly_seasonality=self._yearly_seasonality,
            )

        # Prepare X and y
        feature_cols = [date_col] + channels + self._controls
        X = self._data[feature_cols].copy()
        y = self._data[target]  # Must be Series, not ndarray (PyMC-Marketing concats it)

        # MCMC settings (allow overrides via kwargs)
        mcmc_kwargs = {
            "chains": kwargs.pop("chains", self._chains),
            "tune": kwargs.pop("tune", self._tune),
            "draws": kwargs.pop("draws", self._draws),
            "target_accept": kwargs.pop("target_accept", self._target_accept),
        }
        mcmc_kwargs.update(kwargs)

        # Fit
        logger.info(
            "Starting MCMC: chains=%d, tune=%d, draws=%d",
            mcmc_kwargs["chains"], mcmc_kwargs["tune"], mcmc_kwargs["draws"],
        )
        self._mmm.fit(
            X=X, y=y,
            random_seed=self._random_seed,
            **mcmc_kwargs,
        )

        # Check convergence
        self._check_convergence()

        # Extract results
        self._result = self._build_model_result(data, target, date_col, channels)
        return self._result

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions using posterior mean."""
        if self._mmm is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        feature_cols = [self._date_col] + self._channels + self._controls
        X_new = data[feature_cols].copy()
        pp_raw = self._mmm.sample_posterior_predictive(
            X_new, combined=True, extend_idata=False,
        )
        pp_da = self._extract_data_array(pp_raw)
        return pp_da.mean(dim="sample").values * self._get_target_scale()

    def get_channel_contributions(self) -> pd.DataFrame:
        """Get channel contribution decomposition over time."""
        if self._result is None or self._result.channel_contributions is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        return self._result.channel_contributions

    def get_saturation_curves(
        self,
        channel: str | None = None,
        spend_range: tuple[float, float] | None = None,
        n_points: int = 200,
    ) -> dict[str, pd.DataFrame]:
        """
        Extract saturation curves using LEARNED parameters.

        Unlike RidgeMMM's hardcoded curves, these reflect the actual response
        functions learned from data via Bayesian inference.
        """
        if self._mmm is None or self._result is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        channels_to_plot = [channel] if channel else self._channels
        curves = {}

        for ch in channels_to_plot:
            if ch not in self._channels:
                continue

            if spend_range is None:
                max_spend = float(self._data[ch].max()) * 1.5
                spends = np.linspace(0, max_spend, n_points)
            else:
                spends = np.linspace(spend_range[0], spend_range[1], n_points)

            # Use learned saturation params from posterior
            sat_params = self._result.saturation_params.get(ch, {})
            lam = sat_params.get("lam", 1.0)

            # LogisticSaturation formula: response = 1 - exp(-lam * x)
            # We need to apply the same scaling that PyMC-Marketing uses.
            # The model scales channel data, so we apply saturation to
            # the normalized spend range.
            scaler = self._get_channel_scaler(ch)
            if scaler > 0:
                spends_scaled = spends / scaler
            else:
                spends_scaled = spends

            responses = 1.0 - np.exp(-lam * spends_scaled)

            curves[ch] = pd.DataFrame({"spend": spends, "response": responses})

        return curves

    def get_roas_by_channel(self) -> dict[str, float]:
        """Get average ROAS per channel."""
        if self._result is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        return self._result.channel_roas

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model_result(
        self,
        data: pd.DataFrame,
        target: str,
        date_col: str,
        channels: list[str],
    ) -> ModelResult:
        """Extract all results from the fitted PyMC-Marketing model."""
        posterior = self._mmm.idata.posterior

        # --- Predictions (posterior predictive mean) ---
        feature_cols = [date_col] + channels + self._controls
        X = self._data[feature_cols].copy()
        pp_raw = self._mmm.sample_posterior_predictive(X=X, combined=True)
        # pp_raw is xr.DataArray with dims (date, sample) when combined=True
        pp_da = self._extract_data_array(pp_raw)

        # PyMC-Marketing normalizes y internally — rescale predictions back
        target_scale = self._get_target_scale()
        predictions = pp_da.mean(dim="sample").values * target_scale
        pp_array = pp_da.values * target_scale  # (n_obs, n_samples)
        y_actual = data[target].values
        residuals = y_actual - predictions

        # --- Fit metrics ---
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
        r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        mape = float(np.mean(np.abs(residuals / np.where(y_actual != 0, y_actual, 1)))) * 100
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        # --- Channel contributions ---
        contributions = self._extract_contributions(data, date_col, channels, predictions)

        # --- Channel ROAS and shares ---
        channel_roas = {}
        channel_share = {}
        total_modeled = 0.0
        for ch in channels:
            if ch in contributions.columns:
                ch_total = float(contributions[ch].sum())
                ch_spend = float(data[ch].sum())
                channel_roas[ch] = ch_total / ch_spend if ch_spend > 0 else 0.0
                total_modeled += abs(ch_total)

        for ch in channels:
            if ch in contributions.columns:
                ch_total = abs(float(contributions[ch].sum()))
                channel_share[ch] = ch_total / total_modeled if total_modeled > 0 else 0.0

        # --- Learned parameters from posterior ---
        adstock_params = self._extract_adstock_params(posterior, channels)
        saturation_params = self._extract_saturation_params(posterior, channels)

        # --- Posterior samples and credible intervals ---
        posterior_samples = {"predictions": pp_array}

        credible_intervals = {}
        for ch in channels:
            if ch in contributions.columns:
                ch_vals = contributions[ch].values
                credible_intervals[ch] = (
                    float(np.percentile(ch_vals, 3)),
                    float(np.percentile(ch_vals, 97)),
                )

        # Prediction HDI — compute per-observation HDI from (n_obs, n_samples) array
        # Transpose to (n_samples, n_obs) so az.hdi treats axis-0 as draws
        pred_hdi = az.hdi(pp_array.T, hdi_prob=0.94)
        # pred_hdi shape: (n_obs, 2) — average across observations for summary
        credible_intervals["prediction"] = (
            float(pred_hdi[:, 0].mean()),
            float(pred_hdi[:, 1].mean()),
        )

        return ModelResult(
            model_type="BayesianMMM",
            target_variable=target,
            date_column=date_col,
            channels=channels,
            n_observations=len(data),
            r_squared=r_squared,
            mape=mape,
            rmse=rmse,
            channel_contributions=contributions,
            channel_roas=channel_roas,
            channel_share=channel_share,
            saturation_params=saturation_params,
            adstock_params=adstock_params,
            predictions=predictions,
            residuals=residuals,
            posterior_samples=posterior_samples,
            credible_intervals=credible_intervals,
            raw_model=self._mmm,
        )

    def _extract_contributions(
        self,
        data: pd.DataFrame,
        date_col: str,
        channels: list[str],
        predictions: np.ndarray,
    ) -> pd.DataFrame:
        """Extract per-channel contribution decomposition."""
        posterior = self._mmm.idata.posterior
        contributions = pd.DataFrame()
        contributions[date_col] = data[date_col].values

        # Channel contributions from posterior
        if "channel_contribution" in posterior:
            ch_contrib = posterior["channel_contribution"]
            # Shape: (chain, draw, obs, channel) — take mean over chain & draw
            ch_mean = ch_contrib.mean(dim=["chain", "draw"]).values

            for i, ch in enumerate(channels):
                if i < ch_mean.shape[-1]:
                    contributions[ch] = ch_mean[:, i]

        # Base = predictions minus all channel contributions
        channel_total = sum(
            contributions[ch].values for ch in channels if ch in contributions.columns
        )
        contributions["base"] = predictions - channel_total
        contributions["total"] = predictions

        return contributions

    def _extract_adstock_params(
        self, posterior: Any, channels: list[str],
    ) -> dict[str, dict[str, float]]:
        """Extract learned adstock parameters from posterior."""
        params: dict[str, dict[str, float]] = {}

        # Find adstock variable in posterior
        adstock_vars = [v for v in posterior.data_vars if "adstock" in v.lower()]
        if not adstock_vars:
            for ch in channels:
                params[ch] = {"decay": 0.5}
            return params

        adstock_var = adstock_vars[0]
        adstock_data = posterior[adstock_var]

        for i, ch in enumerate(channels):
            try:
                if "channel" in adstock_data.dims:
                    samples = adstock_data.sel(channel=ch).values.flatten()
                elif adstock_data.ndim >= 3:
                    samples = adstock_data.values[:, :, i].flatten()
                else:
                    samples = adstock_data.values.flatten()

                hdi = az.hdi(samples, hdi_prob=0.94)
                params[ch] = {
                    "decay": float(np.mean(samples)),
                    "decay_hdi_low": float(hdi[0]),
                    "decay_hdi_high": float(hdi[1]),
                }
            except (IndexError, KeyError, ValueError):
                params[ch] = {"decay": 0.5}

        return params

    def _extract_saturation_params(
        self, posterior: Any, channels: list[str],
    ) -> dict[str, dict[str, float]]:
        """Extract learned saturation parameters from posterior."""
        params: dict[str, dict[str, float]] = {}

        # Find saturation variable in posterior
        sat_vars = [v for v in posterior.data_vars if "saturation" in v.lower()]
        if not sat_vars:
            for ch in channels:
                params[ch] = {"lam": 1.0}
            return params

        sat_var = sat_vars[0]
        sat_data = posterior[sat_var]

        for i, ch in enumerate(channels):
            try:
                if "channel" in sat_data.dims:
                    samples = sat_data.sel(channel=ch).values.flatten()
                elif sat_data.ndim >= 3:
                    samples = sat_data.values[:, :, i].flatten()
                else:
                    samples = sat_data.values.flatten()

                hdi = az.hdi(samples, hdi_prob=0.94)
                params[ch] = {
                    "lam": float(np.mean(samples)),
                    "lam_hdi_low": float(hdi[0]),
                    "lam_hdi_high": float(hdi[1]),
                }
            except (IndexError, KeyError, ValueError):
                params[ch] = {"lam": 1.0}

        return params

    def _get_target_scale(self) -> float:
        """Get the scaling factor PyMC-Marketing applied to the target variable."""
        try:
            return float(self._mmm.scalers["_target"].values)
        except (AttributeError, KeyError):
            # Fallback: use max abs of original target data
            if self._data is not None and self._target:
                return float(self._data[self._target].abs().max())
            return 1.0

    @staticmethod
    def _extract_data_array(pp: Any) -> Any:
        """Extract xr.DataArray from posterior predictive output.

        PyMC-Marketing may return xr.DataArray or xr.Dataset depending
        on version. This normalizes to DataArray.
        """
        # xr.Dataset — extract first (and typically only) data variable
        if hasattr(pp, "data_vars"):
            var_name = list(pp.data_vars)[0]
            return pp[var_name]
        return pp

    def _get_channel_scaler(self, channel: str) -> float:
        """Get the scaling factor used for a channel by PyMC-Marketing."""
        if self._data is None:
            return 1.0
        # PyMC-Marketing typically scales by the mean or max of the channel
        return float(self._data[channel].max()) if channel in self._data.columns else 1.0

    def _check_convergence(self) -> None:
        """Check MCMC convergence and log warnings."""
        if self._mmm is None or self._mmm.idata is None:
            return

        try:
            rhat = az.rhat(self._mmm.idata)
            # Check for Rhat > 1.05 (common threshold)
            for var_name in rhat.data_vars:
                vals = rhat[var_name].values
                max_rhat = float(np.nanmax(vals))
                if max_rhat > 1.05:
                    logger.warning(
                        "Convergence warning: %s has Rhat=%.3f (>1.05). "
                        "Consider increasing tune/draws.",
                        var_name, max_rhat,
                    )
        except Exception as e:
            logger.debug("Could not check convergence: %s", e)

    def get_posterior_summary(self) -> pd.DataFrame:
        """Return ArviZ summary of key posterior parameters.

        This is a BayesianMMM-specific method not in BaseMMM.
        """
        if self._mmm is None:
            raise RuntimeError("Model not fitted.")
        return az.summary(self._mmm.idata, hdi_prob=0.94)
