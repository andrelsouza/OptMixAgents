"""
Budget optimizer for Marketing Mix Modeling.

Takes a fitted MMM and finds the optimal budget allocation across channels,
respecting real-world constraints like minimum spends and channel caps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from optmix.mmm.models.base import BaseMMM, OptimizationResult
from optmix.mmm.transforms.saturation import hill_saturation


@dataclass
class ChannelConstraint:
    """Budget constraint for a single channel."""

    min_spend: float = 0.0
    max_spend: float = float("inf")


class BudgetOptimizer:
    """
    Constrained budget optimization using fitted MMM response curves.

    Takes a fitted model's saturation curves and finds the allocation that
    maximizes (or minimizes) a given objective under real-world constraints.
    """

    def __init__(self, model: BaseMMM) -> None:
        self.model = model
        self._roas = model.get_roas_by_channel()
        self._curves = model.get_saturation_curves()

    def optimize(
        self,
        total_budget: float,
        constraints: dict[str, dict[str, float]] | None = None,
        objective: str = "maximize_revenue",
        current_allocation: dict[str, float] | None = None,
        method: str = "SLSQP",
    ) -> OptimizationResult:
        """
        Find optimal budget allocation.

        Args:
            total_budget: Total budget to allocate across channels.
            constraints: Per-channel constraints.
                Example: {"google_ads": {"min": 50000, "max": 200000}}
            objective: Optimization objective. Currently supports:
                - "maximize_revenue": Maximize total predicted response.
            current_allocation: Current allocation for comparison.
            method: scipy optimization method.

        Returns:
            OptimizationResult with optimal allocation and expected outcomes.
        """
        channels = list(self._curves.keys())
        n_channels = len(channels)

        # Parse constraints
        channel_constraints = {}
        for ch in channels:
            ch_constraint = ChannelConstraint()
            if constraints and ch in constraints:
                c = constraints[ch]
                ch_constraint.min_spend = c.get("min", c.get("min_spend", 0.0))
                ch_constraint.max_spend = c.get("max", c.get("max_spend", float("inf")))
            channel_constraints[ch] = ch_constraint

        # Response function for a channel at a given spend
        def channel_response(channel: str, spend: float) -> float:
            curve = self._curves[channel]
            spend_vals = curve["spend"].values
            response_vals = curve["response"].values
            # Interpolate
            return float(np.interp(spend, spend_vals, response_vals))

        # Objective function (negative because we minimize)
        def neg_total_response(allocation: np.ndarray) -> float:
            total = 0.0
            for i, ch in enumerate(channels):
                total += channel_response(ch, allocation[i])
            return -total

        # Initial guess: proportional to current ROAS or equal split
        if current_allocation:
            x0 = np.array([current_allocation.get(ch, total_budget / n_channels) for ch in channels])
        else:
            x0 = np.full(n_channels, total_budget / n_channels)

        # Normalize to budget
        x0 = x0 * (total_budget / x0.sum()) if x0.sum() > 0 else x0

        # Bounds
        bounds = [
            (channel_constraints[ch].min_spend, min(channel_constraints[ch].max_spend, total_budget))
            for ch in channels
        ]

        # Budget constraint
        budget_constraint = {"type": "eq", "fun": lambda x: np.sum(x) - total_budget}

        # Optimize
        result = minimize(
            neg_total_response,
            x0,
            method=method,
            bounds=bounds,
            constraints=[budget_constraint],
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        # Build output
        optimal = {ch: float(result.x[i]) for i, ch in enumerate(channels)}
        optimal_response = -result.fun

        # Current response for comparison
        prev_response = None
        if current_allocation:
            prev_response = sum(
                channel_response(ch, current_allocation.get(ch, 0)) for ch in channels
            )

        # Marginal ROAS at optimal point
        marginal_roas = {}
        for ch in channels:
            try:
                marginal_roas[ch] = self.model.get_marginal_roas(ch, at_spend=optimal[ch])
            except (ValueError, IndexError):
                marginal_roas[ch] = 0.0

        # Saturation percentage at optimal
        saturation_pct = {}
        for ch in channels:
            curve = self._curves[ch]
            max_response = curve["response"].max()
            current_response = channel_response(ch, optimal[ch])
            saturation_pct[ch] = (current_response / max_response * 100) if max_response > 0 else 0

        # Binding constraints
        binding = []
        for ch in channels:
            c = channel_constraints[ch]
            if abs(optimal[ch] - c.min_spend) < 1.0:
                binding.append(f"{ch}_min")
            if abs(optimal[ch] - c.max_spend) < 1.0:
                binding.append(f"{ch}_max")

        lift = None
        lift_pct = None
        if prev_response is not None and prev_response > 0:
            lift = optimal_response - prev_response
            lift_pct = (lift / prev_response) * 100

        return OptimizationResult(
            allocation=optimal,
            previous_allocation=current_allocation,
            total_budget=total_budget,
            expected_outcome=optimal_response,
            previous_outcome=prev_response,
            expected_lift=lift,
            expected_lift_pct=lift_pct,
            channel_marginal_roas=marginal_roas,
            channel_saturation_pct=saturation_pct,
            confidence_interval=None,  # Available with Bayesian models
            constraints_applied={ch: vars(c) for ch, c in channel_constraints.items()},
            binding_constraints=binding,
        )

    def run_scenario(
        self,
        base_allocation: dict[str, float],
        changes: dict[str, float],
    ) -> OptimizationResult:
        """
        Run a what-if scenario by applying percentage changes to a base allocation.

        Args:
            base_allocation: Current channel → spend mapping.
            changes: Channel → percentage change (e.g., {"tv": -0.30, "meta": +0.15}).

        Returns:
            OptimizationResult comparing scenario to base.
        """
        channels = list(self._curves.keys())

        scenario_allocation = {}
        for ch in channels:
            base = base_allocation.get(ch, 0)
            change = changes.get(ch, 0)
            scenario_allocation[ch] = base * (1 + change)

        new_total = sum(scenario_allocation.values())

        # Compute responses
        def channel_response(channel: str, spend: float) -> float:
            curve = self._curves[channel]
            return float(np.interp(spend, curve["spend"].values, curve["response"].values))

        base_response = sum(channel_response(ch, base_allocation.get(ch, 0)) for ch in channels)
        scenario_response = sum(channel_response(ch, scenario_allocation[ch]) for ch in channels)

        lift = scenario_response - base_response
        lift_pct = (lift / base_response * 100) if base_response > 0 else 0

        return OptimizationResult(
            allocation=scenario_allocation,
            previous_allocation=base_allocation,
            total_budget=new_total,
            expected_outcome=scenario_response,
            previous_outcome=base_response,
            expected_lift=lift,
            expected_lift_pct=lift_pct,
        )
