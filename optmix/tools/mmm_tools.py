"""
MMM tools for the Modeler agent (Priya).

Wraps the core MMM engine into callable tool functions for LLM agents.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from optmix.tools.registry import ToolParameter, ToolSchema

# --- Schemas ---

FIT_MMM_MODEL_SCHEMA = ToolSchema(
    name="fit_mmm_model",
    description="Fit a Marketing Mix Model on the loaded data. 'ridge' mode is fast (~2s) with hardcoded transforms; 'bayesian' mode uses MCMC to learn adstock/saturation from data (~2-5 min) with full uncertainty.",
    parameters=[
        ToolParameter(
            name="model_type",
            type="string",
            description="Model backend: 'ridge' (fast, hardcoded transforms) or 'bayesian' (MCMC, learned parameters).",
            required=False,
            default="ridge",
            enum=["ridge", "bayesian"],
        ),
        ToolParameter(
            name="target",
            type="string",
            description="Target variable column name.",
            required=False,
            default="revenue",
        ),
        ToolParameter(
            name="date_col",
            type="string",
            description="Date column name.",
            required=False,
            default="date",
        ),
        ToolParameter(
            name="channels",
            type="array",
            description="List of channel spend column names. Auto-detected if not provided.",
            required=False,
        ),
        ToolParameter(
            name="controls",
            type="array",
            description="List of control variable column names.",
            required=False,
        ),
        ToolParameter(
            name="chains",
            type="integer",
            description="MCMC chains (bayesian only).",
            required=False,
            default=4,
        ),
        ToolParameter(
            name="draws",
            type="integer",
            description="MCMC draws per chain (bayesian only).",
            required=False,
            default=1000,
        ),
        ToolParameter(
            name="tune",
            type="integer",
            description="MCMC tuning steps (bayesian only).",
            required=False,
            default=1000,
        ),
    ],
    returns_description="Model summary with R², MAPE, ROAS per channel, channel contributions, and (bayesian) learned parameters with uncertainty.",
    agent_scope=["modeler"],
)

GET_CHANNEL_CONTRIBUTIONS_SCHEMA = ToolSchema(
    name="get_channel_contributions",
    description="Get the channel contribution decomposition from the fitted model, showing how much each channel contributed to the target over time.",
    parameters=[],
    returns_description="Per-channel contribution totals and shares.",
    agent_scope=["modeler", "reporter"],
)

GET_SATURATION_CURVES_SCHEMA = ToolSchema(
    name="get_saturation_curves",
    description="Extract saturation (diminishing returns) curves for marketing channels. Shows how response changes as spend increases.",
    parameters=[
        ToolParameter(
            name="channel",
            type="string",
            description="Specific channel name, or omit for all channels.",
            required=False,
        ),
    ],
    returns_description="Saturation curve data with inflection points per channel.",
    agent_scope=["modeler", "optimizer"],
)

GET_MODEL_DIAGNOSTICS_SCHEMA = ToolSchema(
    name="get_model_diagnostics",
    description="Get model fit diagnostics including R², MAPE, RMSE, per-channel ROAS, and fit quality assessment.",
    parameters=[],
    returns_description="Diagnostics dict with fit metrics and quality assessment.",
    agent_scope=["modeler"],
)


# --- Implementations ---


def fit_mmm_model(
    state: Any,
    *,
    model_type: str = "ridge",
    target: str = "revenue",
    date_col: str = "date",
    channels: list[str] | None = None,
    controls: list[str] | None = None,
    chains: int = 4,
    draws: int = 1000,
    tune: int = 1000,
) -> dict[str, Any]:
    """Fit an MMM model on the data in state."""
    df = _get_state(state, "validated_data")
    if df is None:
        df = _get_state(state, "raw_data")
    if df is None:
        return {
            "status": "error",
            "message": "No data available. Load data first.",
            "summary": "Cannot fit model: no data loaded.",
        }

    if target not in df.columns:
        return {
            "status": "error",
            "message": f"Target column '{target}' not found. Available: {list(df.columns)}",
        }

    try:
        if model_type == "bayesian":
            from optmix.mmm.models.bayesian_mmm import BayesianMMM

            model = BayesianMMM(chains=chains, draws=draws, tune=tune)
        else:
            from optmix.mmm.models.ridge_mmm import RidgeMMM

            model = RidgeMMM(alpha=1.0)

        result = model.fit(
            data=df,
            target=target,
            date_col=date_col,
            channels=channels,
            controls=controls,
        )

        _set_state(state, "model_result", result, "modeler")
        _set_state(state, "fitted_model", model, "modeler")

        response: dict[str, Any] = {
            "status": "success",
            "summary": (
                f"{result.model_type} fitted: R²={result.r_squared:.3f}, MAPE={result.mape:.1f}%, "
                f"{len(result.channels)} channels, {result.n_observations} observations"
            ),
            "model_type": result.model_type,
            "r_squared": round(result.r_squared, 4) if result.r_squared else None,
            "mape": round(result.mape, 2) if result.mape else None,
            "rmse": round(result.rmse, 2) if result.rmse else None,
            "n_observations": result.n_observations,
            "channels": result.channels,
            "channel_roas": {ch: round(v, 3) for ch, v in result.channel_roas.items()},
            "channel_share": {ch: round(v * 100, 1) for ch, v in result.channel_share.items()},
        }

        # Add Bayesian-specific outputs
        if result.credible_intervals:
            response["has_uncertainty"] = True
        if result.adstock_params:
            response["adstock_params"] = {
                ch: {k: round(v, 4) for k, v in params.items()}
                for ch, params in result.adstock_params.items()
            }
        if result.saturation_params:
            response["saturation_params"] = {
                ch: {k: round(v, 4) for k, v in params.items()}
                for ch, params in result.saturation_params.items()
            }

        return response

    except Exception as e:
        return {
            "status": "error",
            "message": f"Model fitting failed: {e}",
            "summary": f"Model fitting error: {e}",
        }


def get_channel_contributions(state: Any) -> dict[str, Any]:
    """Get channel contribution decomposition from fitted model."""
    model = _get_state(state, "fitted_model")
    if model is None:
        return {"status": "error", "message": "No fitted model found. Run fit_mmm_model first."}

    try:
        contributions = model.get_channel_contributions()
        _set_state(state, "channel_contributions", contributions, "modeler")

        result_data = _get_state(state, "model_result")
        channels = result_data.channels if result_data else []

        contribution_totals = {}
        contribution_shares = {}
        total = 0

        for ch in channels:
            if ch in contributions.columns:
                ch_total = float(contributions[ch].sum())
                contribution_totals[ch] = round(ch_total, 2)
                total += abs(ch_total)

        for ch in channels:
            if ch in contribution_totals:
                contribution_shares[ch] = (
                    round(abs(contribution_totals[ch]) / total * 100, 1) if total > 0 else 0
                )

        base_total = float(contributions["base"].sum()) if "base" in contributions.columns else 0

        return {
            "status": "success",
            "summary": f"Channel contributions decomposed for {len(channels)} channels over {len(contributions)} periods",
            "contribution_totals": contribution_totals,
            "contribution_shares": contribution_shares,
            "base_total": round(base_total, 2),
            "n_periods": len(contributions),
        }

    except Exception as e:
        return {"status": "error", "message": f"Contribution decomposition failed: {e}"}


def get_saturation_curves(state: Any, *, channel: str | None = None) -> dict[str, Any]:
    """Extract saturation curves for channels."""
    model = _get_state(state, "fitted_model")
    if model is None:
        return {"status": "error", "message": "No fitted model found. Run fit_mmm_model first."}

    try:
        curves = model.get_saturation_curves(channel=channel)

        curve_summaries = {}
        for ch_name, curve_df in curves.items():
            spend = curve_df["spend"].values
            response = curve_df["response"].values

            # Find approximate inflection point (where second derivative changes most)
            if len(response) > 2:
                first_deriv = np.gradient(response, spend)
                second_deriv = np.gradient(first_deriv, spend)
                inflection_idx = np.argmin(second_deriv)
                inflection_spend = float(spend[inflection_idx])
            else:
                inflection_spend = float(spend[len(spend) // 2])

            # Half-saturation point (where response = 0.5)
            half_idx = np.argmin(np.abs(response - 0.5))
            half_sat_spend = float(spend[half_idx])

            curve_summaries[ch_name] = {
                "half_saturation_spend": round(half_sat_spend, 0),
                "inflection_point_spend": round(inflection_spend, 0),
                "max_response": round(float(response.max()), 4),
                "n_points": len(spend),
            }

        return {
            "status": "success",
            "summary": f"Saturation curves extracted for {len(curve_summaries)} channels",
            "curves": curve_summaries,
        }

    except Exception as e:
        return {"status": "error", "message": f"Saturation curve extraction failed: {e}"}


def get_model_diagnostics(state: Any) -> dict[str, Any]:
    """Get model fit diagnostics."""
    result = _get_state(state, "model_result")
    if result is None:
        return {"status": "error", "message": "No model results found. Run fit_mmm_model first."}

    try:
        r2 = result.r_squared or 0
        if r2 >= 0.90:
            fit_quality = "excellent"
        elif r2 >= 0.80:
            fit_quality = "good"
        elif r2 >= 0.65:
            fit_quality = "acceptable"
        else:
            fit_quality = "poor"

        return {
            "status": "success",
            "summary": (
                f"Model diagnostics: R²={r2:.3f} ({fit_quality}), "
                f"MAPE={result.mape:.1f}%, RMSE={result.rmse:.1f}"
            ),
            "r_squared": round(r2, 4),
            "mape": round(result.mape, 2) if result.mape else None,
            "rmse": round(result.rmse, 2) if result.rmse else None,
            "fit_quality": fit_quality,
            "n_observations": result.n_observations,
            "channels": result.channels,
            "channel_roas": {ch: round(v, 3) for ch, v in result.channel_roas.items()},
            "channel_share": {ch: round(v * 100, 1) for ch, v in result.channel_share.items()},
        }

    except Exception as e:
        return {"status": "error", "message": f"Diagnostics failed: {e}"}


# --- Helpers ---


def _get_state(state: Any, key: str) -> Any:
    if hasattr(state, "get"):
        return state.get(key)
    if isinstance(state, dict):
        return state.get(key)
    return None


def _set_state(state: Any, key: str, value: Any, agent: str) -> None:
    if hasattr(state, "set") and callable(state.set):
        try:
            state.set(key, value, source_agent=agent)
        except TypeError:
            state[key] = value
    elif isinstance(state, dict):
        state[key] = value


MMM_TOOL_SCHEMAS: list[tuple[ToolSchema, Any]] = [
    (FIT_MMM_MODEL_SCHEMA, fit_mmm_model),
    (GET_CHANNEL_CONTRIBUTIONS_SCHEMA, get_channel_contributions),
    (GET_SATURATION_CURVES_SCHEMA, get_saturation_curves),
    (GET_MODEL_DIAGNOSTICS_SCHEMA, get_model_diagnostics),
]
