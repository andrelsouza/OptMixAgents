"""
Base MMM model interface.

All MMM backends (Bayesian, Lightweight, Ridge) implement this interface,
ensuring agents can work with any model interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ModelResult:
    """Standardized output from any MMM model."""

    # Model metadata
    model_type: str
    target_variable: str
    date_column: str
    channels: list[str]
    n_observations: int

    # Fit metrics
    r_squared: float | None = None
    mape: float | None = None
    rmse: float | None = None

    # Channel-level results
    channel_contributions: pd.DataFrame | None = None
    channel_roas: dict[str, float] = field(default_factory=dict)
    channel_share: dict[str, float] = field(default_factory=dict)

    # Saturation parameters
    saturation_params: dict[str, dict[str, float]] = field(default_factory=dict)

    # Adstock parameters
    adstock_params: dict[str, dict[str, float]] = field(default_factory=dict)

    # Predictions
    predictions: np.ndarray | None = None
    residuals: np.ndarray | None = None

    # Bayesian-specific
    posterior_samples: dict[str, np.ndarray] | None = None
    credible_intervals: dict[str, tuple[float, float]] | None = None

    # Raw model object for advanced usage
    raw_model: Any = None


@dataclass
class OptimizationResult:
    """Output from budget optimization."""

    # Allocation
    allocation: dict[str, float]
    previous_allocation: dict[str, float] | None = None
    total_budget: float = 0.0

    # Expected outcomes
    expected_outcome: float = 0.0
    previous_outcome: float | None = None
    expected_lift: float | None = None
    expected_lift_pct: float | None = None

    # Per-channel detail
    channel_marginal_roas: dict[str, float] = field(default_factory=dict)
    channel_saturation_pct: dict[str, float] = field(default_factory=dict)

    # Confidence
    confidence_interval: tuple[float, float] | None = None
    confidence_level: float = 0.90

    # Constraints applied
    constraints_applied: dict[str, Any] = field(default_factory=dict)
    binding_constraints: list[str] = field(default_factory=list)


class BaseMMM(ABC):
    """
    Abstract base class for Marketing Mix Models.

    Implement this interface to add a new MMM backend to OptMix.
    All agents interact with models through this interface.
    """

    @abstractmethod
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
        Fit the MMM on historical data.

        Args:
            data: DataFrame with date, channel spends, controls, and target.
            target: Name of the target column (e.g., 'revenue', 'conversions').
            date_col: Name of the date column.
            channels: List of channel spend columns. Auto-detected if None.
            controls: List of control variable columns (e.g., 'price', 'promo').
            **kwargs: Backend-specific parameters.

        Returns:
            ModelResult with fitted parameters and diagnostics.
        """
        ...

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the given data."""
        ...

    @abstractmethod
    def get_channel_contributions(self) -> pd.DataFrame:
        """
        Get channel contribution decomposition over time.

        Returns:
            DataFrame with columns for date, each channel's contribution,
            base contribution, and total.
        """
        ...

    @abstractmethod
    def get_saturation_curves(
        self, channel: str | None = None, spend_range: tuple[float, float] | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        Extract saturation (response) curves for channels.

        Returns:
            Dict mapping channel name to DataFrame with 'spend' and 'response' columns.
        """
        ...

    @abstractmethod
    def get_roas_by_channel(self) -> dict[str, float]:
        """Get average ROAS for each channel."""
        ...

    def get_marginal_roas(
        self, channel: str, at_spend: float | None = None
    ) -> float:
        """
        Calculate marginal ROAS for a channel at a given spend level.

        Default implementation uses numerical differentiation on the saturation curve.
        """
        curves = self.get_saturation_curves(channel=channel)
        if channel not in curves:
            raise ValueError(f"Channel '{channel}' not found in model.")

        curve = curves[channel]
        spend = curve["spend"].values
        response = curve["response"].values

        if at_spend is None:
            # Use the median observed spend
            at_spend = float(np.median(spend))

        # Numerical derivative
        idx = np.argmin(np.abs(spend - at_spend))
        if idx == 0:
            idx = 1
        if idx >= len(spend) - 1:
            idx = len(spend) - 2

        marginal = (response[idx + 1] - response[idx - 1]) / (spend[idx + 1] - spend[idx - 1])
        return float(marginal)
