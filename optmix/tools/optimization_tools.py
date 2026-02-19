"""
Optimization tools for the Optimizer agent (Ravi).

Budget allocation, scenario simulation, and marginal ROAS calculation.
"""

from __future__ import annotations

from typing import Any

from optmix.tools.registry import ToolParameter, ToolSchema

# --- Schemas ---

OPTIMIZE_BUDGET_SCHEMA = ToolSchema(
    name="optimize_budget",
    description="Find the optimal budget allocation across marketing channels that maximizes revenue, respecting per-channel constraints.",
    parameters=[
        ToolParameter(
            name="total_budget",
            type="number",
            description="Total budget to allocate across all channels.",
            required=True,
        ),
        ToolParameter(
            name="constraints",
            type="object",
            description="Per-channel constraints as {channel: {min: X, max: Y}}.",
            required=False,
        ),
        ToolParameter(
            name="objective",
            type="string",
            description="Optimization objective.",
            required=False,
            default="maximize_revenue",
            enum=["maximize_revenue"],
        ),
    ],
    returns_description="Optimal allocation with expected outcome and channel-level details.",
    agent_scope=["optimizer"],
)

RUN_SCENARIO_SCHEMA = ToolSchema(
    name="run_scenario",
    description="Run a what-if scenario by applying percentage changes to the current budget allocation.",
    parameters=[
        ToolParameter(
            name="changes",
            type="object",
            description="Channel percentage changes, e.g. {'tv': -0.30, 'meta_ads': 0.15}. Values are fractions (0.15 = +15%).",
            required=True,
        ),
    ],
    returns_description="Scenario comparison with expected lift/decline.",
    agent_scope=["optimizer"],
)

GET_MARGINAL_ROAS_SCHEMA = ToolSchema(
    name="get_marginal_roas",
    description="Calculate marginal ROAS for marketing channels at their current spend level. Shows the return of the next dollar spent.",
    parameters=[
        ToolParameter(
            name="channel",
            type="string",
            description="Specific channel name, or omit for all channels.",
            required=False,
        ),
    ],
    returns_description="Marginal ROAS per channel.",
    agent_scope=["optimizer", "modeler"],
)


# --- Implementations ---


def optimize_budget(
    state: Any,
    *,
    total_budget: float,
    constraints: dict[str, dict[str, float]] | None = None,
    objective: str = "maximize_revenue",
) -> dict[str, Any]:
    """Find optimal budget allocation using fitted model."""
    model = _get_state(state, "fitted_model")
    if model is None:
        return {
            "status": "error",
            "message": "No fitted model found. Run fit_mmm_model first.",
            "summary": "Cannot optimize: no model.",
        }

    try:
        from optmix.mmm.optimizer.budget_optimizer import BudgetOptimizer

        optimizer = BudgetOptimizer(model)

        # Try to derive current allocation from data
        current_allocation = None
        df = _get_state(state, "validated_data")
        if df is None:
            df = _get_state(state, "raw_data")
        if df is not None:
            model_result = _get_state(state, "model_result")
            if model_result:
                current_allocation = {
                    ch: float(df[ch].mean()) for ch in model_result.channels if ch in df.columns
                }

        result = optimizer.optimize(
            total_budget=total_budget,
            constraints=constraints,
            objective=objective,
            current_allocation=current_allocation,
        )

        _set_state(state, "optimal_allocation", result, "optimizer")

        # Format for LLM
        allocation_summary = {ch: round(v, 0) for ch, v in result.allocation.items()}

        return {
            "status": "success",
            "summary": (
                f"Budget optimized: ${total_budget:,.0f} across {len(result.allocation)} channels. "
                f"Expected lift: {result.expected_lift_pct:.1f}%"
                if result.expected_lift_pct
                else f"Budget optimized: ${total_budget:,.0f} across {len(result.allocation)} channels."
            ),
            "allocation": allocation_summary,
            "total_budget": total_budget,
            "expected_outcome": round(result.expected_outcome, 4),
            "expected_lift_pct": round(result.expected_lift_pct, 1)
            if result.expected_lift_pct
            else None,
            "channel_marginal_roas": {
                ch: round(v, 3) for ch, v in result.channel_marginal_roas.items()
            },
            "channel_saturation_pct": {
                ch: round(v, 1) for ch, v in result.channel_saturation_pct.items()
            },
            "binding_constraints": result.binding_constraints,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Budget optimization failed: {e}",
            "summary": f"Optimization error: {e}",
        }


def run_scenario(state: Any, *, changes: dict[str, float]) -> dict[str, Any]:
    """Run a what-if scenario on the current allocation."""
    model = _get_state(state, "fitted_model")
    if model is None:
        return {"status": "error", "message": "No fitted model found."}

    optimal = _get_state(state, "optimal_allocation")
    if optimal is None or not hasattr(optimal, "allocation"):
        return {
            "status": "error",
            "message": "No base allocation found. Run optimize_budget first.",
        }

    try:
        from optmix.mmm.optimizer.budget_optimizer import BudgetOptimizer

        optimizer = BudgetOptimizer(model)
        result = optimizer.run_scenario(
            base_allocation=optimal.allocation,
            changes=changes,
        )

        _set_state(state, "scenario_results", result, "optimizer")

        return {
            "status": "success",
            "summary": (
                f"Scenario: {', '.join(f'{ch} {v:+.0%}' for ch, v in changes.items())}. "
                f"Expected lift: {result.expected_lift_pct:.1f}%"
                if result.expected_lift_pct
                else f"Scenario simulated with {len(changes)} channel changes."
            ),
            "base_allocation": {
                ch: round(v, 0) for ch, v in (result.previous_allocation or {}).items()
            },
            "scenario_allocation": {ch: round(v, 0) for ch, v in result.allocation.items()},
            "expected_outcome": round(result.expected_outcome, 4),
            "previous_outcome": round(result.previous_outcome, 4)
            if result.previous_outcome
            else None,
            "expected_lift_pct": round(result.expected_lift_pct, 1)
            if result.expected_lift_pct
            else None,
        }

    except Exception as e:
        return {"status": "error", "message": f"Scenario simulation failed: {e}"}


def get_marginal_roas(state: Any, *, channel: str | None = None) -> dict[str, Any]:
    """Calculate marginal ROAS for channels."""
    model = _get_state(state, "fitted_model")
    if model is None:
        return {"status": "error", "message": "No fitted model found."}

    try:
        model_result = _get_state(state, "model_result")
        channels_to_check = (
            [channel] if channel else (model_result.channels if model_result else [])
        )

        marginal_roas = {}
        for ch in channels_to_check:
            try:
                mroas = model.get_marginal_roas(ch)
                marginal_roas[ch] = round(mroas, 4)
            except (ValueError, IndexError):
                marginal_roas[ch] = 0.0

        _set_state(state, "marginal_roas", marginal_roas, "optimizer")

        # Sort by marginal ROAS for summary
        sorted_mroas = sorted(marginal_roas.items(), key=lambda x: x[1], reverse=True)
        top_summary = ", ".join(f"{ch}: {v:.3f}" for ch, v in sorted_mroas[:5])

        return {
            "status": "success",
            "summary": f"Marginal ROAS for {len(marginal_roas)} channels. Top: {top_summary}",
            "marginal_roas": marginal_roas,
        }

    except Exception as e:
        return {"status": "error", "message": f"Marginal ROAS calculation failed: {e}"}


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


OPTIMIZATION_TOOL_SCHEMAS: list[tuple[ToolSchema, Any]] = [
    (OPTIMIZE_BUDGET_SCHEMA, optimize_budget),
    (RUN_SCENARIO_SCHEMA, run_scenario),
    (GET_MARGINAL_ROAS_SCHEMA, get_marginal_roas),
]
