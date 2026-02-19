"""
Ridge regression MMM — fast baseline model.

Uses scikit-learn Ridge regression with manual adstock and saturation transforms.
Ideal for quick iterations and as a sanity check before running Bayesian models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from optmix.mmm.models.base import BaseMMM, ModelResult
from optmix.mmm.transforms.adstock import geometric_adstock
from optmix.mmm.transforms.saturation import hill_saturation


class RidgeMMM(BaseMMM):
    """
    Fast deterministic MMM using Ridge regression.

    This is the "instant" model — no MCMC, no sampling, just a regularized
    linear regression with adstock and saturation transforms applied to inputs.
    Great for rapid prototyping and as a first pass before Bayesian modeling.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._model: Ridge | None = None
        self._scaler = StandardScaler()
        self._result: ModelResult | None = None
        self._data: pd.DataFrame | None = None
        self._target: str = ""
        self._date_col: str = ""
        self._channels: list[str] = []
        self._controls: list[str] = []
        self._adstock_params: dict[str, float] = {}
        self._saturation_params: dict[str, dict[str, float]] = {}
        self._transformed_data: pd.DataFrame | None = None

    def fit(
        self,
        data: pd.DataFrame,
        target: str,
        date_col: str,
        channels: list[str] | None = None,
        controls: list[str] | None = None,
        adstock_rates: dict[str, float] | None = None,
        saturation_params: dict[str, dict[str, float]] | None = None,
        **kwargs: Any,
    ) -> ModelResult:
        """
        Fit Ridge MMM with adstock and saturation transforms.

        Args:
            data: Marketing data with spend, controls, and target.
            target: Target column name.
            date_col: Date column name.
            channels: Channel spend columns. Auto-detected if None.
            controls: Control variable columns.
            adstock_rates: Dict of channel → decay rate (0-1). Default 0.5.
            saturation_params: Dict of channel → {half_sat, slope}. Default auto.
        """
        self._data = data.copy().sort_values(date_col).reset_index(drop=True)
        self._target = target
        self._date_col = date_col

        # Auto-detect channels if not specified
        if channels is None:
            exclude = {target, date_col} | set(controls or [])
            channels = [c for c in data.select_dtypes(include=[np.number]).columns if c not in exclude]
        self._channels = channels
        self._controls = controls or []

        # Set default adstock rates
        self._adstock_params = adstock_rates or {ch: 0.5 for ch in channels}

        # Set default saturation params
        if saturation_params is None:
            saturation_params = {}
            for ch in channels:
                median_spend = float(data[ch].median())
                saturation_params[ch] = {
                    "half_sat": median_spend * 1.5,
                    "slope": 1.0,
                }
        self._saturation_params = saturation_params

        # Apply transforms
        transformed = self._data[[date_col]].copy()
        for ch in channels:
            spend = self._data[ch].values
            # Adstock
            adstocked = geometric_adstock(spend, self._adstock_params.get(ch, 0.5))
            # Saturation
            params = self._saturation_params[ch]
            saturated = hill_saturation(adstocked, params["half_sat"], params["slope"])
            transformed[ch] = saturated

        for ctrl in self._controls:
            transformed[ctrl] = self._data[ctrl].values

        self._transformed_data = transformed

        # Fit Ridge
        feature_cols = channels + self._controls
        X = self._scaler.fit_transform(transformed[feature_cols].values)
        y = self._data[target].values

        self._model = Ridge(alpha=self.alpha, fit_intercept=True)
        self._model.fit(X, y)

        # Predictions and metrics
        predictions = self._model.predict(X)
        residuals = y - predictions

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        mape = float(np.mean(np.abs(residuals / np.where(y != 0, y, 1)))) * 100
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        # Channel ROAS
        channel_roas = {}
        contributions = self._compute_contributions()
        for ch in channels:
            total_contribution = float(contributions[ch].sum())
            total_spend = float(self._data[ch].sum())
            channel_roas[ch] = total_contribution / total_spend if total_spend > 0 else 0.0

        # Channel share of contribution
        total_modeled = sum(float(contributions[ch].sum()) for ch in channels)
        channel_share = {}
        for ch in channels:
            ch_total = float(contributions[ch].sum())
            channel_share[ch] = ch_total / total_modeled if total_modeled > 0 else 0.0

        self._result = ModelResult(
            model_type="RidgeMMM",
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
            saturation_params=self._saturation_params,
            adstock_params={ch: {"decay": v} for ch, v in self._adstock_params.items()},
            predictions=predictions,
            residuals=residuals,
            raw_model=self._model,
        )
        return self._result

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions on new data."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        transformed = pd.DataFrame()
        for ch in self._channels:
            spend = data[ch].values
            adstocked = geometric_adstock(spend, self._adstock_params.get(ch, 0.5))
            params = self._saturation_params[ch]
            transformed[ch] = hill_saturation(adstocked, params["half_sat"], params["slope"])

        for ctrl in self._controls:
            transformed[ctrl] = data[ctrl].values

        feature_cols = self._channels + self._controls
        X = self._scaler.transform(transformed[feature_cols].values)
        return self._model.predict(X)

    def get_channel_contributions(self) -> pd.DataFrame:
        """Get channel contribution decomposition over time."""
        if self._result is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        return self._result.channel_contributions

    def get_saturation_curves(
        self,
        channel: str | None = None,
        spend_range: tuple[float, float] | None = None,
        n_points: int = 200,
    ) -> dict[str, pd.DataFrame]:
        """Extract saturation curves for visualization."""
        if self._result is None:
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

            params = self._saturation_params[ch]
            responses = hill_saturation(spends, params["half_sat"], params["slope"])

            curves[ch] = pd.DataFrame({"spend": spends, "response": responses})

        return curves

    def get_roas_by_channel(self) -> dict[str, float]:
        """Get average ROAS per channel."""
        if self._result is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        return self._result.channel_roas

    def _compute_contributions(self) -> pd.DataFrame:
        """Decompose predictions into per-channel contributions."""
        if self._model is None or self._transformed_data is None:
            raise RuntimeError("Model not fitted.")

        feature_cols = self._channels + self._controls
        X_scaled = self._scaler.transform(self._transformed_data[feature_cols].values)
        coefs = self._model.coef_

        contributions = pd.DataFrame()
        contributions[self._date_col] = self._data[self._date_col]

        for i, col in enumerate(feature_cols):
            contributions[col] = X_scaled[:, i] * coefs[i]

        contributions["base"] = self._model.intercept_
        contributions["total"] = self._model.predict(X_scaled)

        return contributions
